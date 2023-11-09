/**
 * @file main.cc
 * @author Adam Page (adam.page@ambiq.com)
 * @brief Main application
 * @version 1.0
 * @date 2023-03-27
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "arm_math.h"
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
// neuralSPOT
#include "ns_ambiqsuite_harness.h"
#include "ns_malloc.h"
#include "ns_peripherals_button.h"
#include "ns_peripherals_power.h"
#include "ns_rpc_generic_data.h"
#include "ns_usb.h"
// TFLM
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
// Locals
#include "constants.h"
#include "main.h"

static bool modelInitialized = false;
static uint32_t inputIdx = 0;
static uint32_t modelIdx = 0;
static AppState state = IDLE_STATE;

// Model variables
static TfLiteTensor *inputs;
static TfLiteTensor *outputs;
static tflite::ErrorReporter *errorReporter = nullptr;
constexpr int tensorArenaSize = 1024 * MAX_ARENA_SIZE;
alignas(16) static uint8_t tensorArena[tensorArenaSize];
const tflite::Model *model;
unsigned char modelBuffer[MAX_MODEL_SIZE];

static tflite::MicroInterpreter *interpreter;
static tflite::AllOpsResolver opResolver;
static tflite::MicroErrorReporter microErrorReporter;

const ns_power_config_t ns_pwr_config = {
    .api = &ns_power_V1_0_0,
    .eAIPowerMode = NS_MINIMUM_PERF,
    .bNeedAudAdc = false,
    .bNeedSharedSRAM = false,
    .bNeedCrypto = true,
    .bNeedBluetooth = false,
    .bNeedUSB = true,
    .bNeedIOM = false,
    .bNeedAlternativeUART = false,
    .b128kTCM = false
};


void
gpio_init(uint32_t pin, uint32_t mode) {
    am_hal_gpio_pincfg_t config = mode == 0 ?
        am_hal_gpio_pincfg_disabled : mode == 1 ?
        am_hal_gpio_pincfg_output : am_hal_gpio_pincfg_input;
    am_hal_gpio_pinconfig(pin, config);
}

uint32_t
gpio_write(uint32_t pin, uint8_t value) {
    return am_hal_gpio_state_write(pin, (am_hal_gpio_write_type_e)value);
}

uint32_t
gpio_read(uint32_t pin, uint32_t mode, uint32_t value) {
    am_hal_gpio_read_type_e readMode = mode == 0 ?
        AM_HAL_GPIO_INPUT_READ : mode == 1 ?
        AM_HAL_GPIO_OUTPUT_READ : AM_HAL_GPIO_INPUT_READ;
    return am_hal_gpio_state_read(pin, readMode, &value);
}

uint32_t
setup_model() {
    size_t bytesUsed;
    TfLiteStatus allocateStatus;
    errorReporter = &microErrorReporter;

    tflite::InitializeTarget();

    model = tflite::GetModel(modelBuffer);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        TF_LITE_REPORT_ERROR(errorReporter, "Schema mismatch: given=%d != expected=%d.", model->version(), TFLITE_SCHEMA_VERSION);
        return 1;
    }
    static tflite::MicroInterpreter tflm_interpreter(model, opResolver, tensorArena, tensorArenaSize, errorReporter);
    interpreter = &tflm_interpreter;

    allocateStatus = interpreter->AllocateTensors();
    if (allocateStatus != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(errorReporter, "AllocateTensors() failed");
        return 1;
    }
    bytesUsed = interpreter->arena_used_bytes();
    if (bytesUsed > tensorArenaSize) {
        TF_LITE_REPORT_ERROR(errorReporter, "Arena mismatch: given=%d < expected=%d bytes.", tensorArenaSize, bytesUsed);
        return 1;
    }
    inputs = interpreter->input(0);
    outputs = interpreter->output(0);
    modelInitialized = true;
    return 0;
}

void
background_task() {
    /**
     * @brief Run background tasks
     *
     */
}

void
sleep_us(uint32_t time) {
    /**
     * @brief Enable longer sleeps while also running background tasks on interval
     * @param time Sleep duration in microseconds
     */
    uint32_t chunk;
    while (time > 0) {
        chunk = MIN(10000, time);
        ns_delay_us(chunk);
        time -= chunk;
        background_task();
    }
}

status ns_rpc_data_to_evb_cb(const dataBlock *block) {
    /**
     * @brief Callback for sending data block to EVB
     * @param block Data block to send
     * @return status
     */

    // Receive model
    if (block->cmd == 0) {
        memcpy((void *)&modelBuffer[modelIdx], block->buffer.data, block->buffer.dataLength);
        modelIdx += block->buffer.dataLength;
        if (modelIdx >= MAX_MODEL_SIZE) {
            ns_printf("Received model\n");
            setup_model();
            modelIdx = 0;
            state = IDLE_STATE;
        }
    }

    // Receive inputs
    if (block->cmd == 1) {
        memcpy((void *)&(inputs->data.uint8[inputIdx]), block->buffer.data, block->buffer.dataLength);
        inputIdx += block->buffer.dataLength;
        if (inputIdx >= inputs->bytes) {
            ns_printf("Received inputs\n");
            inputIdx = 0;
            state = IDLE_STATE;
        }
    }

    // Perform inference
    if (block->cmd == 4 && modelInitialized) {
        state = INFERENCE_STATE;
    }

    return ns_rpc_data_success;
}


status ns_rpc_data_from_evb_cb(dataBlock *block) {
    /**
     * @brief Callback for fetching data block from EVB
     * @param block Data block to fetch
     * @return status
    */
    static char rpcOutputsDesc[] = "OUTPUTS";
    static char rpcStateDesc[] = "STATE";
    dataBlock commandBlock = {
        .length = 0,
        .dType = uint8_e,
        .description = NULL,
        .cmd = generic_cmd,
        .buffer = {
            .data = NULL,
            .dataLength = 0,
        }
    };

    // Send outputs
    if (block->cmd == 2) {
        commandBlock.description = rpcOutputsDesc;
        for (size_t i = 0; i < outputs->bytes; i += RPC_BUF_LEN) {
            uint32_t numSamples = MIN(outputs->bytes - i, RPC_BUF_LEN);
            commandBlock.length = i;
            commandBlock.buffer.data = (uint8_t *)(&outputs->data.uint8[i]);
            commandBlock.buffer.dataLength = numSamples * sizeof(uint8_t);
            ns_rpc_data_sendBlockToPC(&commandBlock);
            ns_delay_us(200);
        }
    }

    // Send state
    if (block->cmd == 3) {
        commandBlock.description = rpcStateDesc;
        commandBlock.length = 0;
        commandBlock.buffer.data = (uint8_t *)(&state);
        commandBlock.buffer.dataLength = sizeof(AppState);
        ns_rpc_data_sendBlockToPC(&commandBlock);
    }

    return ns_rpc_data_success;
}

status ns_rpc_data_compute_on_evb_cb(const dataBlock *in_block, dataBlock *result_block) {

    return ns_rpc_data_success;
}

void
init_rpc(void) {
    /**
     * @brief Initialize RPC and USB
     *
     */
    ns_rpc_config_t rpcConfig = {.api = &ns_rpc_gdo_V1_0_0,
                                 .mode = NS_RPC_GENERICDATA_SERVER,
                                 .sendBlockToEVB_cb = ns_rpc_data_to_evb_cb,
                                 .fetchBlockFromEVB_cb = ns_rpc_data_from_evb_cb,
                                 .computeOnEVB_cb = ns_rpc_data_compute_on_evb_cb};
    ns_rpc_genericDataOperations_init(&rpcConfig);
}

void
print_to_pc(const char *msg) {
    /**
     * @brief Print to PC over RPC
     *
     */
    // ns_rpc_data_remotePrintOnPC(msg);
    ns_printf(msg);
}

void
wakeup() {
    am_bsp_itm_printf_enable();
    am_bsp_debug_printf_enable();
}

void
deepsleep() {
    am_bsp_itm_printf_disable();
    am_bsp_debug_printf_disable();
    am_hal_sysctrl_sleep(AM_HAL_SYSCTRL_SLEEP_DEEP);
}

void
setup() {
    /**
     * @brief Application setup
     *
     */
    // Power configuration (mem, cache, peripherals, clock)
    ns_core_config_t ns_core_cfg = {.api = &ns_core_V1_0_0};
    ns_core_init(&ns_core_cfg);
    ns_power_config(&ns_pwr_config);
    gpio_init(GPIO_TRIGGER, 1);
    // Enable Interrupts
    am_hal_interrupt_master_enable();
    // Enable SWO/USB
    wakeup();
    // Initialize blocks
    init_rpc();
    // err |= ns_peripheral_button_init(&button_config);
    ns_printf("😴 SleepKit demo running...\n\n");
}

void
loop() {
    /**
     * @brief Application loop
     *
     */
    static uint32_t app_err = 0;
    switch (state) {
    case IDLE_STATE:
        if (true) {
        } else {
            ns_printf("IDLE_STATE\n");
            deepsleep();
        }
        break;

    case INFERENCE_STATE:
        ns_printf("INFERENCE_STATE\n");
        gpio_write(GPIO_TRIGGER, 0);
        interpreter->Invoke();
        gpio_write(GPIO_TRIGGER, 1);
        state = IDLE_STATE;
        break;

    case FAIL_STATE:
        ns_printf("FAIL_STATE err=%d\n", app_err);
        state = IDLE_STATE;
        app_err = 0;
        break;

    default:
        state = IDLE_STATE;
        break;
    }
    background_task();
}

int
main(void) {
    /**
     * @brief Main function
     * @return int
     */
    setup();
    while (1) { loop(); }
}

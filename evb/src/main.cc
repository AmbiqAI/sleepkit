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
#include "ns_energy_monitor.h"
#include "ns_perf_profile.h"
#include "ns_peripherals_power.h"
#include "ns_peripherals_button.h"
#include "ns_peripherals_power.h"
#include "ns_rpc_generic_data.h"
#include "ns_usb.h"
// TFLM
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
// Locals
#include "constants.h"
// #include "model_buffer.h"
#include "main.h"

// RPC Stuff

static uint8_t rpcRxBuffer[USB_RX_BUFSIZE];
static uint8_t rpcTxBuffer[USB_TX_BUFSIZE];
// End RPC Stuff

static bool modelInitialized = false;
static uint32_t inputIdx = 0;
static uint32_t modelIdx = 0;
static uint32_t outputIdx = 0;
static AppState state = IDLE_STATE;

// Model variables
static tflite::ErrorReporter *errorReporter = nullptr;
static tflite::MicroMutableOpResolver<45> *opResolver = nullptr;
constexpr int tensorArenaSize = 1024 * MAX_ARENA_SIZE;
alignas(16) static uint8_t tensorArena[tensorArenaSize];
alignas(16) unsigned char modelBuffer[1024 * MAX_MODEL_SIZE];

static const tflite::Model *model;
static tflite::MicroInterpreter *interpreter = nullptr;
static TfLiteTensor *inputs;
static TfLiteTensor *outputs;
static tflite::MicroProfiler *profiler = nullptr;


uint32_t setup_model();

ns_rpc_config_t rpcConfig = {
    .api = &ns_rpc_gdo_V1_0_0,
    .mode = NS_RPC_GENERICDATA_SERVER,
    .rx_buf = rpcRxBuffer,
    .rx_bufLength = USB_RX_BUFSIZE,
    .tx_buf = rpcTxBuffer,
    .tx_bufLength = USB_TX_BUFSIZE,
    .sendBlockToEVB_cb = nullptr,
    .fetchBlockFromEVB_cb = nullptr,
    .computeOnEVB_cb = nullptr
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
init_model() {
    tflite::MicroErrorReporter micro_error_reporter;
    errorReporter = &micro_error_reporter;

    tflite::InitializeTarget();

    static tflite::MicroMutableOpResolver<45> resolver;
    opResolver = &resolver;

    resolver.AddAbs();
    resolver.AddAdd();
    resolver.AddArgMax();
    resolver.AddArgMin();
    resolver.AddAveragePool2D();
    resolver.AddBatchToSpaceNd();
    resolver.AddBroadcastArgs();
    resolver.AddBroadcastTo();
    resolver.AddCeil();
    resolver.AddConcatenation();
    resolver.AddConv2D();
    resolver.AddDepthToSpace();
    resolver.AddDepthwiseConv2D();
    resolver.AddDequantize();
    resolver.AddDiv();
    resolver.AddExpandDims();
    resolver.AddFullyConnected();
    resolver.AddGather();
    resolver.AddGreater();
    resolver.AddGreaterEqual();
    resolver.AddHardSwish();
    resolver.AddMaxPool2D();
    resolver.AddMean();
    resolver.AddMinimum();
    resolver.AddMul();
    resolver.AddPad();
    resolver.AddPadV2();
    resolver.AddPack();
    resolver.AddPrelu();
    resolver.AddQuantize();
    resolver.AddRelu();
    resolver.AddRelu6();
    resolver.AddReshape();
    resolver.AddRsqrt();
    resolver.AddShape();
    resolver.AddSoftmax();
    resolver.AddSpaceToBatchNd();
    resolver.AddSpaceToDepth();
    resolver.AddSplit();
    resolver.AddSqueeze();
    resolver.AddStridedSlice();
    resolver.AddSqrt();
    resolver.AddSub();
    resolver.AddTransposeConv();
    resolver.AddTranspose();

    return 0;
}

uint32_t
setup_model() {
    size_t bytesUsed;
    TfLiteStatus allocateStatus;

    model = tflite::GetModel(modelBuffer);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        TF_LITE_REPORT_ERROR(errorReporter, "Schema mismatch: given=%d != expected=%d.", model->version(), TFLITE_SCHEMA_VERSION);
        return 1;
    }
    if (interpreter != nullptr) {
       interpreter->Reset();
    }
    static tflite::MicroInterpreter static_interpreter(model, *opResolver, tensorArena, tensorArenaSize, nullptr, profiler);

    interpreter = &static_interpreter;

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
    inputIdx = 0;
    outputIdx = 0;
    modelIdx = 0;
    modelInitialized = true;
    ns_printf("Model initialized\n");
    for (int i = 0; i < inputs->dims->size; i++) {
        ns_printf("Input dim %d has length %d\n", i, inputs->dims->data[i]);
    }
    return 0;
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
        if (modelIdx >= block->length) {
            ns_printf("Received model (%d)\n", block->length);
            setup_model();
            modelIdx = 0;
            state = IDLE_STATE;
        }
    }

    // Receive inputs
    if (block->cmd == 1 && modelInitialized) {
        memcpy((void *)&(inputs->data.int8[inputIdx]), block->buffer.data, block->buffer.dataLength);
        inputIdx += block->buffer.dataLength;
        if (inputIdx >= block->length) {
            ns_printf("Received inputs (%d)\n", block->length);
            inputIdx = 0;
            state = IDLE_STATE;
        }
    }

    // Signal inference (run in loop so not to block RPC)
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
    ns_lp_printf("ns_rpc_data_from_evb_cb...\n");
    return ns_rpc_data_success;
}

status ns_rpc_data_compute_on_evb_cb(const dataBlock *in_block, dataBlock *result_block) {
    static char rpcOutputsDesc[] = "OUTPUTS";
    static char rpcStateDesc[] = "STATE";

    uint32_t len = RPC_BUF_LEN;
    uint8_t *buffer = (uint8_t *)ns_malloc(len * sizeof(uint8_t));
    char *description = (char *)ns_malloc(sizeof(char) * 30);

    result_block->dType = uint8_e;
    result_block->description = description;
    result_block->cmd = in_block->cmd;
    result_block->buffer = {
        .data = buffer,
        .dataLength = 0
    };

    // Send outputs
    if (in_block->cmd == 2 && modelInitialized) {
        if (outputIdx >= outputs->bytes) { outputIdx = 0; }
        uint32_t numSamples = MIN(outputs->bytes - outputIdx, RPC_BUF_LEN);
        result_block->length = outputs->bytes; // TOTAL LENGTH
        result_block->buffer.dataLength = numSamples * sizeof(uint8_t);
        memcpy(result_block->description, rpcOutputsDesc, sizeof(rpcOutputsDesc));
        memcpy(result_block->buffer.data, (void *)&outputs->data.int8[outputIdx], numSamples * sizeof(uint8_t));
        outputIdx += numSamples;
    }

    // Send state
    if (in_block->cmd == 3) {
        result_block->length = sizeof(AppState);
        result_block->buffer.dataLength = sizeof(AppState);
        memcpy(result_block->description, rpcStateDesc, sizeof(rpcStateDesc));
        memcpy(result_block->buffer.data, (void *)&state, sizeof(AppState));
    }
    return ns_rpc_data_success;
}


void
setup() {
    /**
     * @brief Application setup
     */
    ns_core_config_t coreConfig = {.api = &ns_core_V1_0_0};

    NS_TRY(ns_core_init(&coreConfig), "Core init failed\n");
    NS_TRY(ns_power_config(&ns_development_default), "Power Init Failed\n");
    ns_itm_printf_enable();
    ns_interrupt_master_enable();

    volatile int g_intButtonPressed = 0;
    ns_button_config_t buttonConfig = {
        .api = &ns_button_V1_0_0,
        .button_0_enable = true,
        .button_1_enable = false,
        .button_0_flag = &g_intButtonPressed,
        .button_1_flag = NULL};
    NS_TRY(ns_peripheral_button_init(&buttonConfig), "Button init failed\n");

    // gpio_init(GPIO_TRIGGER, 1);
    rpcConfig.sendBlockToEVB_cb = ns_rpc_data_to_evb_cb;
    rpcConfig.fetchBlockFromEVB_cb = ns_rpc_data_from_evb_cb;
    rpcConfig.computeOnEVB_cb = ns_rpc_data_compute_on_evb_cb;

    NS_TRY(ns_rpc_genericDataOperations_init(&rpcConfig), "RPC Init Failed\n");
    NS_TRY(init_model(), "Model init failed\n");
    ns_delay_us(5000);
    ns_lp_printf("SleepKit inference engine running...\n");
    ns_delay_us(5000);
    state = IDLE_STATE;
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
            ns_deep_sleep();
        }
        break;

    case INFERENCE_STATE:
        ns_printf("INFERENCE_STATE\n");
        ns_printf("Start inference\n");
        gpio_write(GPIO_TRIGGER, 0);
        app_err = interpreter->Invoke();
        gpio_write(GPIO_TRIGGER, 1);
        ns_printf("Inference complete err=%d\n", app_err);
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
    ns_rpc_genericDataOperations_pollServer(&rpcConfig);
    ns_deep_sleep();
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

/**
 * @file sensor.cc
 * @author Adam Page (adam.page@ambiq.com)
 * @brief Initializes and collects sensor data from MAX86150
 * @version 1.0
 * @date 2023-03-27
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "sensor.h"
#include "arm_math.h"
#include "constants.h"
#include "ns_ambiqsuite_harness.h"
#include "ns_i2c.h"
#include "ns_max86150_driver.h"

#define NUM_SLOTS (3)
max86150_slot_type maxSlotsConfig[] = {Max86150SlotPpgLed1, Max86150SlotPpgLed2, Max86150SlotEcg, Max86150SlotOff};

uint32_t maxFifoBuffer[MAX86150_FIFO_DEPTH * NUM_SLOTS];

ns_i2c_config_t i2cConfig = {.api = &ns_i2c_V1_0_0, .iom = 1};

static int
max86150_write_read(uint16_t addr, const void *write_buf, size_t num_write, void *read_buf, size_t num_read) {
    return ns_i2c_write_read(&i2cConfig, addr, write_buf, num_write, read_buf, num_read);
}
static int
max86150_read(const void *buf, uint32_t num_bytes, uint16_t addr) {
    return ns_i2c_read(&i2cConfig, buf, num_bytes, addr);
}
static int
max86150_write(const void *buf, uint32_t num_bytes, uint16_t addr) {
    return ns_i2c_write(&i2cConfig, buf, num_bytes, addr);
}

max86150_context_t maxCtx = {
    .addr = MAX86150_ADDR,
    .i2c_write_read = max86150_write_read,
    .i2c_read = max86150_read,
    .i2c_write = max86150_write,
};


uint32_t
init_sensor(void) {
    /**
     * @brief Initialize and configure sensor block (MAX86150)
     *
     */

    ns_i2c_interface_init(&i2cConfig, AM_HAL_IOM_400KHZ);
    max86150_powerup(&maxCtx);
    ns_delay_us(10000);
    max86150_reset(&maxCtx);
    ns_delay_us(10000);
    max86150_set_fifo_slots(&maxCtx, maxSlotsConfig);
    max86150_set_almost_full_rollover(&maxCtx, 1); // FIFOs rollover

    max86150_set_ppg_adc_range(&maxCtx, 2);        // 16,384 nA Scale
    max86150_set_ppg_sample_rate(&maxCtx, 6);      // 400 Samples/sec
    max86150_set_ppg_pulse_width(&maxCtx, 2);      // 200 us

    max86150_set_ppg_sample_average(&maxCtx, 0);   // Avg 1 samples
    max86150_set_prox_int_flag(&maxCtx, 0);        // Disable proximity based PPG
    // max86150_set_proximity_threshold(&i2c_dev, MAX86150_ADDR, 0x1F); // Disabled

    max86150_set_led_current_range(&maxCtx, 0, 1);      // IR LED 50 mA
    max86150_set_led_current_range(&maxCtx, 1, 1);      // RED LED 50 mA
    max86150_set_led_current_range(&maxCtx, 2, 0);      // AMB LED 50 mA

    max86150_set_led_pulse_amplitude(&maxCtx, 0, 0x64); // IR LED 20 mA 0x64
    max86150_set_led_pulse_amplitude(&maxCtx, 1, 0x64); // RED LED 20 mA 0x64
    max86150_set_led_pulse_amplitude(&maxCtx, 2, 0x64); // AMB LED 20 mA 0x64

    max86150_set_ecg_sample_rate(&maxCtx, 2); // Fs = 400 Hz
    max86150_set_ecg_ia_gain(&maxCtx, 2);     // 9.5 V/V
    max86150_set_ecg_pga_gain(&maxCtx, 3);    // 8 V/V
    max86150_powerup(&maxCtx);
    stop_sensor();
    // max86150_set_fifo_enable(&maxCtx, 0);
    return 0;
}

void
start_sensor(void) {
    /**
     * @brief Takes sensor out of low-power mode and enables FIFO
     *
     */
    // max86150_powerup(&maxCtx);
    max86150_set_fifo_enable(&maxCtx, 1);
}

void
stop_sensor(void) {
    /**
     * @brief Puts sensor in low-power mode
     *
     */
    max86150_set_fifo_enable(&maxCtx, 0);
    // max86150_shutdown(&maxCtx);
}

uint32_t
capture_sensor_data(float32_t *slot0, float32_t *slot1, float32_t *slot2, float32_t *slot3, uint32_t maxSamples) {
    uint32_t numSamples;
    int32_t val;
    float32_t *slots[4] = {slot0, slot1, slot2, slot3};
    numSamples = max86150_read_fifo_samples(&maxCtx, maxFifoBuffer, maxSlotsConfig, NUM_SLOTS);
    numSamples = MIN(maxSamples, numSamples);
    for (size_t i = 0; i < numSamples; i++) {
        for (size_t j = 0; j < NUM_SLOTS; j++) {
            val = maxFifoBuffer[NUM_SLOTS * i + j];
            // ECG data is 18-bit 2's complement. If MSB=1 then make negative
            if ((maxSlotsConfig[j] == Max86150SlotEcg) && (val & (1 << 17))) {
                val -= (1 << 18);
            }
            slots[j][i] = (float32_t)(val);
        }
    }
    return numSamples;
}

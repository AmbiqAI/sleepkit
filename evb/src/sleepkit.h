/**
 * @file heartkit.h
 * @author Adam Page (adam.page@ambiq.com)
 * @brief Perform preprocessing of sensor data (standardize and bandpass filter)
 * @version 1.0
 * @date 2023-03-27
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef __SLEEPKIT_H
#define __SLEEPKIT_H


uint32_t
init_sleepkit();
uint32_t
sk_preprocess(float32_t *ppg1Data, float32_t *ppg2Data, float32_t *ecgData, size_t dataLen);

#endif // __SLEEPKIT_H

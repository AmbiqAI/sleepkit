# Feature Set: FS-W-PA-14
## <span class="sk-h2-span">Overview</span>

This feature set is targeted for __sleep stage classification__ based on sensor data available from __wrist__ location. The feature set computes heart rate, heart rate variability (HRV), SpO2, movement, and respiratory rate features over temporal windows (e.g. 30 seconds) captured from PPG and accelerometer sensors.

## <span class="sk-h2-span">Target Location/Sensors</span>

The target location for this feature set is the __wrist__. From this location, the features are compute from the following raw sensors:

- **PPG**: Dual photoplethysmography (PPG) sensor data is used to compute HR, HRV, SpO2, and respiratory rate features.
- **Accelerometer**: Accelerometer data is used to compute movement features.

## <span class="sk-h2-span">Dataset Support</span>

- **[MESA](../datasets/mesa.md)**: This dataset does not directly provide dual PPG nor Accelerometer data from the wrist. However, the dataset does provide SpO2 and single channel PPG which is sufficient for PPG derived features. In place of Accelerometer data, we use leg movement as a proxy for arm movement features.

## <span class="sk-h2-span">Features</span>

This feature set includes the following 14 features:

| Feature Name | Description | Sensor |
| --- | --- | --- |
| hr_bpm | Mean heart rate in beats per minute | PPG |
| hrv_td_mean_nn | Mean of the NN intervals | PPG |
| hrv_td_sd_nn | Standard deviation of the NN intervals | PPG |
| hrv_td_median_nn | Median of the NN intervals | PPG |
| hrv_fd_lfhf_ratio | Ratio of low frequency to high frequency power in the frequency domain | PPG |
| spo2_mu | Mean SpO2 | PPG |
| spo2_std | Standard deviation of SpO2 | PPG |
| spo2_med | Median SpO2 | PPG |
| mov_mu | Mean movement | ACC |
| mov_std | Standard deviation of movement | ACC |
| mov_med | Median movement | ACC |
| rsp_bpm | Mean respiration rate derived from the PPG signal | PPG |
| spo2_qos | Quality of signal derived from SpO2 | PPG |
| hrv_qos | Quality of signal derived from HRV | PPG |


## <span class="sk-h2-span">Output</span>

The feature set is stored as HDF5 files (`.h5`) with one file per subject with path: `{save_path}/{dataset}/{subject_id}.h5`. Each HDF5 file includes the following entries:

* `/features`: Time x Feature tensor (fp32). Features are computed over windows of sensor data.
* `/mask`: Time x Mask tensor (bool). Mask indicates valid feature values.
* `/labels`: Time x Label (int). Labels are sleep stages.

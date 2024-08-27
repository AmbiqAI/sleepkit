# Feature Set: FS-C-EAR-9

## <span class="sk-h2-span">Overview</span>

This feature set is targeted for sleep stage classification based on sensor data available from __chest__ location. The feature set computes heart rate (HR), heart rate variability (HRV), movement, and respiratory rate features over temporal windows (e.g. 30 seconds) captured from ECG, respiratory and accelerometer sensors.

## <span class="sk-h2-span">Target Location/Sensors</span>

The target location for this feature set is the __chest__. From this location, the features are compute from the following raw sensors:

- **ECG**: Electrocardiography (ECG) sensor data is used to compute HR, HRV, and respiratory rate features.
- **RSP**: Respiratory sensor data is used to compute respiratory rate features.
- **IMU**: Accelerometer data is used to compute movement features.

## <span class="sk-h2-span">Dataset Support</span>

- **[MESA](../datasets/mesa.md)**: This dataset does not directly provide accelerometer data from the chest. However, the dataset does provide respiratory signals (RIP) captured from both chest and abdomen. In place of accelerometer data, we use filtered chest respiratory signals as a proxy for body movement features.

## <span class="sk-h2-span">Features</span>

This feature set includes the following 14 features:

| Feature Name | Description | Sensor |
| --- | --- | --- |
| hr_bpm | Mean heart rate in beats per minute | PPG |
| hrv_td_mean_nn | Mean of the NN intervals | PPG |
| hrv_td_sd_nn | Standard deviation of the NN intervals | PPG |
| hrv_td_median_nn | Median of the NN intervals | PPG |
| hrv_fd_lfhf_ratio | Ratio of low frequency to high frequency power in the frequency domain | PPG |
| mov_mu | Mean movement | IMU |
| mov_std | Standard deviation of movement | IMU |
| mov_med | Median movement | IMU |
| rsp_bpm | Mean respiration rate derived from the PPG signal | RSP |
| hrv_qos | Quality of signal derived from HRV | ECG |


## <span class="sk-h2-span">Output</span>

The feature set is stored as HDF5 files (`.h5`) with one file per subject with path: `{save_path}/{dataset}/{subject_id}.h5`. Each HDF5 file includes the following entries:

* `/features`: Time x Feature tensor (fp32). Features are computed over windows of sensor data.
* `/mask`: Time x Mask tensor (bool). Mask indicates valid feature values.
* `/labels`: Time x Label (int). Labels are sleep stages.

# Feature Set: FS-W-A-5

## <span class="sk-h2-span">Overview</span>

This feature set is targeted towards actigraphy style sleep detection using only IMU based sensor data available from wrist location. The feature set computes movement and z-angle features over temporal windows (e.g. 5 seconds) captured from IMU sensor.

## <span class="sk-h2-span">Target Location/Sensors</span>

The target location for this feature set is the __wrist__. From this location, the features are compute from the following raw sensors:

- **IMU**: Inertial measurement unit (IMU) data is used to compute movement and z-angle.

## <span class="sk-h2-span">Dataset Support</span>

- **[CMIDSS](../datasets/cmidss.md)**: This dataset provides pre-computed movement and z-angle data from the wrist location every 5 seconds.

## <span class="sk-h2-span">Features</span>

This feature set includes the following 5 features:

| Feature Name | Description | Sensor |
| --- | --- | --- |
| tod | Time of day encoded using cosine | Time |
| mov_mu | Mean movement | IMU |
| mov_std | Standard deviation of movement | IMU |
| angle_mu | Mean z-angle | IMU |
| angle_std | Standard deviation of z-angle | IMU |


## <span class="sk-h2-span">Output</span>

The feature set is stored as HDF5 files (`.h5`) with one file per subject with path: `{save_path}/{dataset}/{subject_id}.h5`. Each HDF5 file includes the following entries:

* `/features`: Time x Feature tensor (fp32). Features are computed over windows of sensor data.
* `/mask`: Time x Mask tensor (bool). Mask indicates valid feature values.
* `/labels`: Time x Label (int). Labels are sleep stages.

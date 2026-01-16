# Feature Set: FS-W-A-5

## Overview

This feature set is targeted towards actigraphy style sleep detection using only IMU based sensor data available from wrist location. The feature set computes movement and z-angle features over temporal windows (e.g. 5 seconds) captured from IMU sensor.

## Target Location/Sensors

The target location for this feature set is the __wrist__. From this location, the features are compute from the following raw sensors:

- **IMU**: Inertial measurement unit (IMU) data is used to compute movement and z-angle.

## Dataset Support

- **[CMIDSS](../datasets/cmidss.md)**: This dataset provides pre-computed movement and z-angle data from the wrist location every 5 seconds.

## Features

This feature set includes the following 5 features:

| Feature Name | Description | Sensor |
| --- | --- | --- |
| tod | Time of day encoded using cosine | Time |
| mov_mu | Mean movement | IMU |
| mov_std | Standard deviation of movement | IMU |
| angle_mu | Mean z-angle | IMU |
| angle_std | Standard deviation of z-angle | IMU |


## Output

The feature set is stored as HDF5 files (`.h5`) with one file per subject with path: `{save_path}/{dataset}/{subject_id}.h5`. Each HDF5 file includes the following entries:

* `/features`: Time x Feature tensor (fp32). Features are computed over windows of sensor data.
* `/mask`: Time x Mask tensor (bool). Mask indicates valid feature values.
* `/labels`: Time x Label (int). Labels are sleep stages.

# Feature Set: FS-W-P-5

## <span class="sk-h2-span">Overview</span>

This feature set is targeted for sleep apnea classification based on sensor data available from __wrist__ location. The generator computes various features over temporal windows (e.g. 30 seconds) captured from PPG sensors.

## <span class="sk-h2-span">Target Location/Sensors</span>

The target location for this feature set is the __wrist__. From this location, the features are compute from the following raw sensors:

- **PPG**: Dual photoplethysmography (PPG) sensor data is used to compute SpO2 and heart rate variability (HRV) features.

## <span class="sk-h2-span">Dataset Support</span>

- **[MESA](../datasets/mesa.md)**: This dataset does not directly provide dual PPG, however, the dataset does provide SpO2 and single channel PPG which is sufficient.

## <span class="sk-h2-span">Features</span>

This feature set includes the following 4 features:

| Feature Name | Description | Sensor |
| --- | --- | --- |
| spo2 | Blood oxygen saturation | PPG |
| piav | Peak-to-trough amplitude delta of PPG | PPG |
| piiv | Peak-to-peak amplitude delta of PPG | PPG |
| pifv | Peak-to-peak interval delta | PPG |

## <span class="sk-h2-span">Output</span>

The feature set is stored as HDF5 files (`.h5`) with one file per subject with path: `{save_path}/{dataset}/{subject_id}.h5`. Each HDF5 file includes the following entries:

* `/features`: Time x Feature tensor (fp32). Features are computed over windows of sensor data.
* `/mask`: Time x Mask tensor (bool). Mask indicates valid feature values.
* `/labels`: Time x Label (int). Labels are sleep stages.

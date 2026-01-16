# Feature Set: FS-H-E-10

## Overview

This feature set is targeted for sleep stage classification based on single pair of ECG and EOG sensor data collected on __head__ location. The feature set computes frequency band metrics over temporal windows (e.g. 30 seconds) captured from ECG and EOG sensors.

## Target Location/Sensors

The target location for this feature set is the __head__. From this location, the features are compute from the following raw sensors:

- **EEG**: Electroencephalography (EEG) sensor data is used to compute frequency bands, power spectral density, and entropy features.
- **EOG**: Electrooculography (EOG) sensor data is used to compute ocular movement features.

## Dataset Support

- **[MESA](../datasets/mesa.md)**: This dataset provides EEG, EOG, and EMG data from the head location. The dataset also provides sleep stage labels.

## Features

This feature set includes the following features:

| Feature Name | Description | Sensor |
| --- | --- | --- |
| eeg_delta_power | Power in delta frequency band | EEG |
| eeg_theta_power | Power in theta frequency band | EEG |
| eeg_alpha_power | Power in alpha frequency band | EEG |
| eeg_beta_power | Power in beta frequency band | EEG |
| eeg_gamma_power | Power in gamma frequency band | EEG |
| eog_delta_power | Power in delta frequency band | EOG |
| eog_theta_power | Power in theta frequency band | EOG |
| eog_alpha_power | Power in alpha frequency band | EOG |
| eog_beta_power | Power in beta frequency band | EOG |
| eog_gamma_power | Power in gamma frequency band | EOG |

## Output

The feature set is stored as HDF5 files (`.h5`) with one file per subject with path: `{save_path}/{dataset}/{subject_id}.h5`. Each HDF5 file includes the following entries:

* `/features`: Time x Feature tensor (fp32). Features are computed over windows of sensor data.
* `/mask`: Time x Mask tensor (bool). Mask indicates valid feature values.
* `/labels`: Time x Label (int). Labels are sleep stages.

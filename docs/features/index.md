# Feature Store

__SleepKit__ includes a mini _feature store_ to enable crafting rich feature sets to train and evaluate models. The store includes several built-in feature set generators that can be invoked to create feature sets for a variety of use cases. Each feature set generator is a callable that takes a set of parameters, generates set of features, and saves the results in HDF5 format. The saved feature sets can then be used to train and evaluate models. Custom feature set generators can also be added to the feature store by subclassing `SKFeatureSet` and registering the new generator with the feature store.

Included feature set generators:

- **[FS-W-PA-14](./featset01.md)**:__14__ features derived from __PPG__ and __IMU__ on __wrist__ for __sleep stage__ classification.
- **[FS-C-EAR-9](./featset02.md)**: __9__ features derived from __ECG__, __RSP__, and __IMU__ on __chest__ for __sleep stage__ classification.
- **[FS-W-A-5](./featset03.md)**: __5__ features derived from __IMU__ on __wrist__ for actigraphy style __sleep detection__.
- **[FS-H-E-10](./featset04.md)**: __10__ features derived from __ECG__ and __EOG__ on __head__ for __sleep stage__ classification.
- **[FS-W-P-5](./featset05.md)**: __5__ features derived from __PPG__ on __wrist__ for __sleep apnea__ classification.

!!! Note "Note"
    The feature store is designed to be used in a local environment and is not intended to be used in a distributed or production environment.

## <span class="sk-h2-span">Usage</span>


## <span class="sk-h2-span">Arguments </span>

The following tables lists the arguments that can be used with the `feature` command.

--8<-- "assets/modes/feature-params.md"

---

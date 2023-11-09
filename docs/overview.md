# Overview

__SleepKit__ can be used as either a CLI-based app or as a python package to perform advanced experimentation. In both forms, SleepKit exposes a number of modes and tasks discussed below:

## Modes

* `download`: Download datasets
* `feature`: Extract features from dataset(s)
* `train`: Train a model for specified task and features/dataset(s)
* `evaluate`: Evaluate a model for specified task and features/dataset(s)
* `export`: Export a trained model to TensorFlow Lite and TFLM
* `demo`: Run full demo on PC or EVB

## Tasks

* `stage`: Perform 2, 3, 4, or 5 stage sleep detection
<!-- * `apnea`: Detect hypopnea/apnea events
* `arousal`: Detect sleep arousal events -->

## Using CLI

The SleepKit command line interface (CLI) makes it easy to run a variefy of single-line commands without the need for writing any code. You can rull all tasks and modes from the terminal with the `sleepkit` command.

<div class="termy">

```console
$ sleepkit --help

SleepKit CLI Options:
    --task [stage]
    --mode [download, feature, train, evaluate, export, demo]
    --config ["./path/to/config.json", or '{"raw: "json"}']
```

</div>

!!! note
    Before running commands, be sure to activate python environment: `poetry shell`. On Windows using Powershell, use `.venv\Scripts\activate.ps1`.

## __1. Download Datasets__

!!! note
    In order to download MESA and STAGES datasets, permission must be granted by NSSR. Both non-commercial and commercial variants are available for these datasets. Once granted permission, please follow [NSSR documentation](https://github.com/nsrr/nsrr-gem) to install their command line `nssr` tool.
    Ensure `nssr` command is available on terminal and authorization token has been supplied.

The `download` command is used to download all datasets specified in the configuration file. Please refer to [Datasets](./datasets.md) for details on the available datasets.

The following example will download and prepare all currently used datasets.

!!! example

    === "CLI"

        ```bash
        sleepkit --mode download --config ./configs/download-datasets.json
        ```

    === "Python"

        ```python
        import sleepkit as sk

        sk.datasets.download_datasets(sk.defines.SKDownloadParams(
            ds_path="./datasets",
            datasets=["mesa"],
            progress=True,
            force=False
        ))
        ```

## __2. Extract Features__

The `feature` command is used to extract features from the downloaded datasets. The following command will extract feature set `001` from the MESA dataset using the reference configuration. Please refer to `sleepkit/defines.py` to see supported options.

!!! example

    === "CLI"

        ```bash
        sleepkit --mode feature --config ./configs/feature-stage-001.json
        ```

    === "Python"

        ```python
        import sleepkit as sk

        sk.features.generate_feature_set(sk.defines.SKFeatureParams(
            ...
        ))
        ```

## __3. Train Model__

The `train` command is used to train a SleepKit model. The following command will train a 2-stage sleep model using the reference configuration. Please refer to `sleepkit/defines.py` to see supported options.

!!! example

    === "CLI"

        ```bash
        sleepkit --task stage --mode train --config ./configs/train-stage-2.json
        ```

    === "Python"

        ```python
        import sleepkit as sk

        sk.sleepstages.train(sk.defines.SKTrainParams(
            ...
        ))
        ```

## __4. Evaluate Model__

The `evaluate` command will evaluate the performance of the model on the reserved test set.

!!! example

    === "CLI"

        ```bash
        sleepkit --task stage --mode evaluate --config ./configs/test-stage-2.json
        ```

    === "Python"

        ```python
        import sleepkit as sk

        sk.sleepstages.evaluate(sk.defines.SKTestParams(
            ...
        ))
        ```

## __5. Export Model__

The `export` command will convert the trained TensorFlow model into both TensorFlow Lite (TFL) and TensorFlow Lite for microcontroller (TFLM) variants. The command will also verify the models' outputs match. Post-training quantization can also be enabled by setting the `quantization` flag in the configuration.

!!! example

    === "CLI"

        ```bash
        sleepkit --task stage --mode export --config ./configs/export-stage-2.json
        ```

    === "Python"

        ```python
        import sleepkit as sk

        sk.sleepstages.export(sk.defines.SKExportParams(
            ...
        ))
        ```

Once converted, the TFLM header file will be copied to location specified by `tflm_file`. If parameters were changed (e.g. window size, quantization), `./evb/src/constants.h` will need to be updated accordingly.

## __6. Demo__

The `demo` command is used to run a full-fledged SleepKit demonstration.

# Overview

__SleepKit__ can be used as either a CLI-based app or as a python package to perform advanced experimentation. In both forms, SleepKit exposes a number of modes and tasks discussed below:

---

## <span class="sk-h2-span">Modes</span>

* `download`: Download datasets
* `feature`: Extract features from dataset(s)
* `train`: Train a model for specified task and features/dataset(s)
* `evaluate`: Evaluate a model for specified task and features/dataset(s)
* `export`: Export a trained model to TensorFlow Lite and TFLM
* `demo`: Run full demo on PC or EVB

---

!!! Tasks

    === "stage"

        ### Sleep Stage Classification

        Perform 2, 3, 4, or 5 stage sleep detection.
        Refer to [Sleep Stages](./stages/overview.md) for more details.

    === "apnea"

        ### Sleep Apnea Detection
        Detect hypopnea/apnea events. __Not yet implemented.__

    === "arousal"

        ### Sleep Arousal Detection
        Detect sleep arousal events. __Not yet implemented.__

---

## <span class="sk-h2-span">Using CLI</span>

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

## <span class="sk-h2-span">1. Download Datasets</span>

!!! note
    In order to download MESA and STAGES datasets, permission must be granted by NSSR. Both non-commercial and commercial variants are available for these datasets. Once granted permission, please follow [NSSR documentation](https://github.com/nsrr/nsrr-gem) to install their command line `nssr` tool. Ensure `nssr` command is available on terminal and authorization token has been supplied.

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
            datasets=["mesa", "ysyw"],
            progress=True,
            force=False
        ))
        ```

## <span class="sk-h2-span">2. Extract Features</span>

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

## <span class="sk-h2-span">3. Train Model</span>

The `train` command is used to train a SleepKit model. The following command will train a 2-stage sleep model using the reference configuration. Please refer to `sleepkit/defines.py` to see supported options.

!!! example

    === "CLI"

        ```bash
        sleepkit --task stage --mode train --config ./configs/train-stage-2.json
        ```

    === "Python"

        ```python
        import sleepkit as sk

        sk.stage.train(sk.defines.SKTrainParams(
            ...
        ))
        ```

## <span class="sk-h2-span">4. Evaluate Model</span>

The `evaluate` command will evaluate the performance of the model on the reserved test set.

!!! example

    === "CLI"

        ```bash
        sleepkit --task stage --mode evaluate --config ./configs/test-stage-2.json
        ```

    === "Python"

        ```python
        import sleepkit as sk

        sk.stage.evaluate(sk.defines.SKTestParams(
            ...
        ))
        ```

## <span class="sk-h2-span">5. Export Model</span>

The `export` command will convert the trained TensorFlow model into both TensorFlow Lite (TFL) and TensorFlow Lite for microcontroller (TFLM) variants. The command will also verify the models' outputs match. Post-training quantization can also be enabled by setting the `quantization` flag in the configuration.

!!! example

    === "CLI"

        ```bash
        sleepkit --task stage --mode export --config ./configs/export-stage-2.json
        ```

    === "Python"

        ```python
        import sleepkit as sk

        sk.stage.export(sk.defines.SKExportParams(
            ...
        ))
        ```

Once converted, the TFLM header file will be copied to the location specified by `tflm_file`.

## <span class="sk-h2-span">6. Demo</span>

The `demo` command is used to run a full-fledged SleepKit demonstration for the specific task.


!!! example

    === "CLI"

        ```bash
        sleepkit --task stage --mode demo --config ./configs/demo-stage-4.json
        ```

    === "Python"

        ```python
        import sleepkit as sk

        sk.stage.demo(sk.defines.SKDemoParams(
            ...
        ))
        ```

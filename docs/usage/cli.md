# :octicons-terminal-24: SleepKit CLI

<div class="termy">

```console
$ sleepkit --help

SleepKit CLI Options:
    --task [detect, stage, apnea, arousal]
    --mode [download, feature, train, evaluate, export, demo]
    --config ["./path/to/config.json", or '{"raw: "json"}']
```

</div>

The SleepKit command line interface (CLI) makes it easy to run a variety of single-line commands without the need for writing any code. You can run all tasks and modes from the terminal with the `sleepkit` command.

!!! example

    === "Syntax"
        `sleepkit` commands use the following syntax:

        ```bash
        sleepkit --mode [MODE] --task [TASK] --config [CONFIG]
        ```

        Or using short flags:

        ```bash
        sleepkit -m [MODE] -t [TASK] -c [CONFIG]
        ```

        Where:

        * `MODE` is one of `download`, `feature`, `train`, `evaluate`, `export`, or `demo`
        * `TASK` is one of `detect`, `stage`, `apnea`, or `arousal`
        * `CONFIG` is configuration as JSON content or file path

    === "Download"
        Download datasets specified in the configuration file.

        ```bash
        sleepkit -t stage -m download -c ./configuration.json
        ```
    === "Feature"
        Extract features from the datasets using the supplied configuration file.

        ```bash
        sleepkit -t stage -m feature -c ./configuration.json
        ```

    === "Train"
        Train a task model using the supplied configuration file.

        ```bash
        sleepkit -t stage -m train -c ./configuration.json
        ```

    === "Evaluate"
        Evaluate the trained task model using the supplied configuration file.

        ```bash
        sleepkit -t stage -m evaluate  -c ./configuration.json
        ```

    === "Demo"
        Run demo on trained task model using the supplied configuration file.

        ```bash
        sleepkit -t stage -m demo -c ./configuration.json
        ```


!!! Note "Configuration File"

    The configuration file is a JSON file that contains all the necessary parameters for the task. The configuration file can be passed as a file path or as a JSON string. In addition, a single configuration file can be used for all `modes`- only needed parameters will be extracted for the given `mode` running.  Please refer to the [Configuration](../modes/configuration.md) section for more details.

---

## [Download](../modes/download.md)

The `download` command is used to download all datasets specified in the configuration file. Please refer to [Datasets](../datasets/index.md) for details on the available datasets.


!!! Example "CLI"

    The following command will download and prepare all datasets specified in configuration JSON file.

    ```bash
    sleepkit --task stage --mode download --config ./configuration.json
    ```

---

## [Feature](../features/index.md)

The `feature` command is used to extract features from the datasets specified in the configuration file using the specified feature set generator. Additional parameters can be set in the configuration file. Please refer to [FeatureParams](../modes/configuration.md#featureparams) and [TaskParams](../modes/configuration.md#taskparams) to see supported options.

!!! Example "CLI"

    The following command will extract features from the datasets using the reference configuration:

    ```bash
    sleepkit --task stage --mode feature --config ./configuration.json
    ```

---

## [Train](../modes/train.md)

The `train` command is used to train a SleepKit model for the specified `task` and `feature set`. Each task provides a reference routine for training the model. The routine can be customized via the configuration file. Please refer to [TaskParams](../modes/configuration.md#taskparams) to see supported options.

!!! Example "CLI"

    The following command will train a task model using the reference configuration:

    ```bash
    sleepkit --task stage --mode train --config ./configuration.json
    ```

---

## [Evaluate](../modes/evaluate.md)

The `evaluate` command will test the performance of the model on the reserved test sets for the specified `task`. The routine can be customized via the configuration file. Please refer to [TaskParams](../modes/configuration.md#taskparams) to see supported options.

!!! example "CLI"

    The following command will test the task model using the reference configuration:

    ```bash
    sleepkit --task stage --mode evaluate --config ./configuration.json
    ```

---

## [Export](../modes/export.md)

The `export` command will convert the trained TensorFlow model into both TensorFlow Lite (TFL) and TensorFlow Lite for micro-controller (TFLM) variants. The command will also verify the models' outputs match. The activations and weights can be quantized by configuring the `quantization` section in the configuration file. Once converted, the TFLM header file will be copied to location specified by `tflm_file`. Please refer to [TaskParams](../modes/configuration.md#taskparams) to see supported options.

!!! example "CLI"

    The following command will export the task model to TF Lite and TFLM:

    ```bash
    sleepkit --task stage --mode export --config ./configuration.json
    ```

---

## [Demo](../modes/demo.md)


The `demo` command is used to run a task-level demonstration using either the PC or EVB as backend inference engine. Please refer to [TaskParams](../modes/configuration.md#taskparams) to see supported options.

!!! Example "CLI"

    The following command will run a demo on the trained task model using the same supplied configuration file.

    ```bash
    sleepkit --task stage --mode demo --config ./configuration.json
    ```

---

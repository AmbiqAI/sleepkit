# SleepKit

🚧 SleepKit is currently under active development 🚧

---

**Documentation**: <a href="https://ambiqai.github.io/sleepkit" target="_blank">https://ambiqai.github.io/sleepkit</a>

**Source Code**: <a href="https://github.com/AmbiqAI/sleepkit" target="_blank">https://github.com/AmbiqAI/sleepkit</a>

---

Overview info

## Requirements

* [Python 3.11+](https://www.python.org)
* [Poetry 1.2.1+](https://python-poetry.org/docs/#installation)

The following are also required to compile/flash the binary for the EVB demo:

* [Arm GNU Toolchain 11.3](https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads)
* [Segger J-Link v7.56+](https://www.segger.com/downloads/jlink/)

> NOTE: A [VSCode Dev Container](https://code.visualstudio.com/docs/devcontainers/containers) is also available and defined in `./.devcontainer`.


## Installation

To get started, first install the local python package `sleepkit` along with its dependencies via `Poetry`:

```bash
poetry install
```

## Usage

__SleepKit__ is intended to be used as either a CLI-based app or as a python package to perform additional tasks and experiments.

### Modes

* `download`: Download datasets
* `feature`: Extract features from dataset(s)
* `train`: Train a model for specified task and features/dataset(s)
* `evaluate`: Evaluate a model for specified task and features/dataset(s)
* `export`: Export a trained model to TensorFlow Lite and TFLM
* `demo`: Run full demo on PC or EVB

### Tasks

* `stage`: Perform 2, 3, 4, or 5 stage sleep detection
* `apnea`: Detect hypopnea/apnea events
* `arousal`: Detect sleep arousal events


### Using CLI

The CLI provides a number of commands discussed below. In general, reference configurations are provided to download datasets, train/evaluate/export models, and lastly demo model(s) on PC or Apollo 4 EVB. Pre-trained reference models are also included to enable running inference and the demo immediately.

```bash
sleepkit
--task [stage, apnea, arousal]
--mode [download, feature, train, evaluate, export, demo]
--config ["./path/to/config.json", or '{"raw: "json"}']

```

> NOTE: Before running commands, be sure to activate python environment: `poetry shell`. On Windows using Powershell, use `.venv\Scripts\activate.ps1`.

#### __1. Download Datasets__

The `download` command is used to download all datasets specified in the configuration file. Please refer to [Datasets section](#datasets) for details on the available datasets.

The following command will download and prepare all currently used datasets.

```bash
sleepkit --mode download --config ./configs/download-datasets.json
```

> NOTE: Both MESA and STAGES require NSSR approval before downloading.

#### __2. Train Model__

The `train` command is used to train a SleepKit model. The following command will train a 2-stage sleep model using the reference configuration. Please refer to `sleepkit/defines.py` to see supported options.

```bash
sleepkit --task stage --mode train --config ./configs/train-stage-2.json
```

#### __3. Evaluate Model__

The `evaluate` command will evaluate the performance of the model on the reserved test set.

```bash
sleepkit --task stage --mode evaluate --config ./configs/evaluate-stage-2.json
```

#### __4. Export Model__

The `export` command will convert the trained TensorFlow model into both TFLite (TFL) and TFLite for microcontroller (TFLM) variants. The command will also verify the models' outputs match. Post-training quantization can also be enabled by setting the `quantization` flag in the configuration.

```bash
sleepkit --task arrhythmia --mode export --config ./configs/export-stage-2.json
```

Once converted, the TFLM header file will be copied to location specified by `tflm_file`. If parameters were changed (e.g. window size, quantization), `./evb/src/constants.h` will need to be updated.

#### __5. Demo__

...

## Model Architecture

...

## Datasets

SleepKit leverages several large datasets for training each of the SleepKit models. Check out the [Datasets Guide](./datasets.md) to learn more about the datasets used along with their corresponding licenses and limitations.

## Results

The following table provides the latest performance and accuracy results of all models when running on Apollo4 Plus EVB.

| Task           | Params   | FLOPS   | Metric     | Cycles/Inf | Time/Inf   |
| -------------- | -------- | ------- | ---------- | ---------- | ---------- |
| 2-Stage Sleep  | --K      | --M     | ---% F1   | ---ms       | ---M       |
| 3-Stage Sleep  | --K      | --M     | ---% F1   | ---ms       | ---M       |
| 4-Stage Sleep  | --K      | --M     | ---% F1   | ---ms       | ---M       |
| Sleep Apnea    | --K      | --M     | ---% F1   | ---ms       | ---M       |
| Sleep Arousal  | --K      | --M     | ---% F1   | ---ms       | ---M       |



## Experiments

Group sensors based on location.

Location: Head
Device: Earable, glasses, headband
Sensors: EEG, EOG, EMG, ACC/GYRO, PPG (ear)
Features: Raw, freq band power, movement
Networks: 1D/2D CNN+UNET, 1D/2D CNN+LSTM/GRU
Tasks: Sleep detect, Apnea, Sleep stage
Notes:
    * Using EEG and EOG data compute features that will be fed into AI model.
    * EEG/EOG is gold standard for determining sleep stage.
    * Minimal temporal context is needed (i.e. 5-10 seconds).
    * Most invasive

Location: Chest
Device: Chest strap/patch
Sensors: ECG, RIP, ACC/GYRO, TEMP
Features: HR, HRV, BPM, BRV, movement, body temp changes
Networks: 1D CNN+LSTM/GRU, 1D CNN+UNET
Notes:
    * Much longer context as more subject/night specific and looking for subtle changes.
    * Best location for tracking breathing

Location: Wrist/Ankle
Device: Watch, ring, strap
Sensors: PPG/SPO2, ACC/GYRO, TEMP
Features: HR, HRV, BPM?, movement, body temp changes
Networks: 1D CNN+LSTM/GRU, 1D CNN+UNET
Notes:
    * Much longer context as more subject/night specific and looking for subtle changes.
    * Least invasive

## References

* [U-Sleep: Resilient High-Frequency Sleep Staging](https://www.nature.com/articles/s41746-021-00440-5)
* [U-Time: A Fully Convolutional Network for Time Series Segmentation Applied to Sleep Staging](https://arxiv.org/pdf/1910.11162.pdf)
* [DeepSleepNet: a Model for Automatic Sleep Stage Scoring based on Raw Single-Channel EEG](https://arxiv.org/pdf/1703.04046.pdf)

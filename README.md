<p align="center">
  <a href="https://github.com/AmbiqAI/sleepkit"><img src="./docs/assets/sleepkit-banner.png" alt="SleepKit"></a>
</p>

<p style="color:rgb(201,48,198); font-size: 1.2em;">
🚧 SleepKit is under active development
</p>

---

**Documentation**: <a href="https://ambiqai.github.io/sleepkit" target="_blank">https://ambiqai.github.io/sleepkit</a>

**Source Code**: <a href="https://github.com/AmbiqAI/sleepkit" target="_blank">https://github.com/AmbiqAI/sleepkit</a>

---

SleepKit is a collection of optimized open-source TinyML models purpose-built to enable running a variety of real-time sleep-monitoring applications on battery-powered, wearable devices. The objective is to outperform conventional, hand-crafted algorithms with efficient AI models that still fit within the stringent resource constraints of embedded devices. SleepKit explores a number of sleep related tasks including sleep staging, sleep apnea detection, and sleep arousal detection. The models are trainined using a large variety datasets- using a subset of biological signals that can be captured from a single body location such as head, chest, or wrist/hand. The goal is to enable models that can be deployed in real-world commercial and consumer applications that are viable for long-term use.


**Key Features:**

* Efficient: Leverage modern AI techniques coupled with Ambiq's ultra-low power SoCs
* Generalizable: Multi-modal, multi-task, multi-dataset
* Accurate: Achieve SoTA with stringent resource constraints

## Requirements

* [Python ^3.11](https://www.python.org)
* [Poetry ^1.6.1](https://python-poetry.org/docs/#installation)

The following are also required to compile/flash the binary for the EVB demo:

* [Arm GNU Toolchain ^12.2](https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads)
* [Segger J-Link ^7.92](https://www.segger.com/downloads/jlink/)

!!! note
    A [VSCode Dev Container](https://code.visualstudio.com/docs/devcontainers/containers) is also available and defined in [./.devcontainer](https://github.com/AmbiqAI/sleepkit/tree/main/.devcontainer).

## Installation

To get started, first install the local python package `sleepkit` along with its dependencies via `Poetry`:


```bash
poetry install
```

---

## Usage

__SleepKit__ can be used as either a CLI-based app or as a python package to perform advanced experimentation. In both forms, SleepKit exposes a number of modes and tasks discussed below. Refer to the [Overview Guide](./docs/overview.md) to learn more about available options and configurations.

---

## Modes

* `download`: Download datasets
* `feature`: Extract features from dataset(s)
* `train`: Train a model for specified task and dataset(s)
* `evaluate`: Evaluate a model for specified task and dataset(s)
* `export`: Export a trained model to TF Lite and TFLM
* `demo`: Run task-level demo on PC or EVB

---

## Tasks

* `stage`: Perform 2, 3, 4, or 5 stage sleep detection
* `apnea`: Detect hypopnea/apnea events
* `arousal`: Detect sleep arousal events

---

## Architecture

SleepKit leverages modern architectural design strategies to achieve high accuracy while maintaining a small memory footprint and low power consumption. Refer to specific task guides for additional details on the full model design.

* Seperable (depthwise + pointwise) Convolutions
* Inverted Residual Bottlenecks
* Squeeze & Excitation Blocks
* Over-Parameterized Convolutional Branches
* Dilated Convolutions

---

## Datasets

SleepKit uses several open-source datasets for training each of the task's models. In general, we use commercial-use friendly datasets that are publicly available. Check out the [Datasets Guide](./docs/datasets.md) to learn more about the datasets used along with their corresponding licenses and limitations.


---

## Results

The following table provides the latest performance and accuracy results of all models when running on Apollo4 Plus EVB. Additional result details can be found in [Results Section](./docs/results.md).

| Task           | Params   | FLOPS   | Metric     | Cycles/Inf | Time/Inf   |
| -------------- | -------- | ------- | ---------- | ---------- | ---------- |
| 2-Stage Sleep  | 10K      | 1.7M/hr | 88.8% F1   |  11M/hr    | 58ms/hr    |
| 3-Stage Sleep  | 14K      | 2.2M/hr | 84.2% F1   |  16M/hr    | 80ms/hr    |
| 4-Stage Sleep  | 14K      | 2.3M/hr | 76.4% F1   |  16M/hr    | 80ms/hr    |
| 5-Stage Sleep  | 17K      | 2.8M/hr | 70.2% F1   |  18M/hr    | 91ms/hr    |
| Sleep Apnea    | --K      | --M     | ----% F1   | ---M       | ---ms      |
| Sleep Arousal  | --K      | --M     | ----% F1   | ---M       | ---ms      |

---

## References

* [U-Sleep: Resilient High-Frequency Sleep Staging](https://doi.org/10.1038/s41746-021-00440-5)
* [U-Time: A Fully Convolutional Network for Time Series Segmentation Applied to Sleep Staging](https://doi.org/10.48550/arXiv.1910.11162)
* [DeepSleepNet: a Model for Automatic Sleep Stage Scoring based on Raw Single-Channel EEG](https://doi.org/10.48550/arXiv.1703.04046)
* [AI-Driven sleep staging from actigraphy and heart rate](https://doi.org/10.1371/journal.pone.0285703)
* [TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis](https://doi.org/10.48550/arXiv.2210.02186)
* [The Promise of Sleep: A Multi-Sensor Approach for Accurate Sleep Stage Detection Using the Oura Ring](https://doi.org/10.3390/s21134302)
* [Interrater reliability of sleep stage scoring: a meta-analysis](https://doi.org/10.5664/jcsm.9538)
* [Development of generalizable automatic sleep staging using heart rate and movement based on large databases](https://doi.org/10.1007/s13534-023-00288-6)

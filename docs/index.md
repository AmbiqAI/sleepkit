---
title:
---
#

<p align="center">
  <a href="https://github.com/AmbiqAI/sleepkit"><img src="./assets/sleepkit-banner.png" alt="SleepKit"></a>
</p>

---

**Documentation**: <a href="https://ambiqai.github.io/sleepkit" target="_blank">https://ambiqai.github.io/sleepkit</a>

**Source Code**: <a href="https://github.com/AmbiqAI/sleepkit" target="_blank">https://github.com/AmbiqAI/sleepkit</a>

---

SleepKit is an optimized open-source TinyML model purpose-built to enable running a variety of real-time sleep-monitoring applications on battery-powered, edge devices. By leveraging a modern multi-head network architecture coupled with Ambiq's ultra low-power SoC, the model is designed to be **feature 1**, **feature 2**, and **feature 3**.


**Key Features:**

* Feature 1
* Feature 2
* Feature 3

## Requirements

* [Python 3.11+](https://www.python.org)
* [Poetry 1.2.1+](https://python-poetry.org/docs/#installation)

The following are also required to compile/flash the binary for the EVB demo:

* [Arm GNU Toolchain 11.3](https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads)
* [Segger J-Link v7.56+](https://www.segger.com/downloads/jlink/)

!!! note
    A [VSCode Dev Container](https://code.visualstudio.com/docs/devcontainers/containers) is also available and defined in [./.devcontainer](https://github.com/AmbiqAI/sleepkit/tree/main/.devcontainer).

## Installation

<div class="termy">

```console
$ poetry install

---> 100%
```
</div>


## Usage

__SleepKit__ can be used as either a CLI-based app or as a python package to perform advanced experimentation. In both forms, SleepKit exposes a number of modes and tasks discussed below. Refer to the [Overview Guide](./overview.md) to learn more about available options and configurations.

## Modes

* `download`: Download datasets
* `feature`: Extract features from dataset(s)
* `train`: Train a model for specified task and dataset(s)
* `evaluate`: Evaluate a model for specified task and dataset(s)
* `export`: Export a trained model to TensorFlow Lite and TFLM
* `demo`: Run full demo on PC or EVB

## Tasks

* `stage`: Perform 2, 3, or 4 stage sleep detection
* `apnea`: Detect hypopnea/apnea events
* `arousal`: Detect sleep arousal events

****
## Architecture

SleepKit leverages a light weight network architecture...


<p align="center">
  <img src="./assets/sleepkit-banner.png" alt="SleepKit Architecture">
</p>

Refer to [Architecture Overview](./architecture.md) for additional details on the model design.


## Datasets

SleepKit leverages several open-source datasets for training each of the SleepKit models. Additionally, SleepKit contains a customizable synthetic 12-lead ECG generator. Check out the [Datasets Guide](./datasets.md) to learn more about the datasets used along with their corresponding licenses and limitations.


## Results

The following table provides the latest performance and accuracy results of all models when running on Apollo4 Plus EVB. Additional result details can be found in [Results Section](./results.md).

| Task           | Params   | FLOPS   | Metric     | Cycles/Inf | Time/Inf   |
| -------------- | -------- | ------- | ---------- | ---------- | ---------- |
| 2-Stage Sleep  | --K      | --M     | ---% F1   | ---ms       | ---M       |
| 3-Stage Sleep  | --K      | --M     | ---% F1   | ---ms       | ---M       |
| 4-Stage Sleep  | --K      | --M     | ---% F1   | ---ms       | ---M       |
| Sleep Apnea    | --K      | --M     | ---% F1   | ---ms       | ---M       |
| Sleep Arousal  | --K      | --M     | ---% F1   | ---ms       | ---M       |

## References

* [U-Sleep: Resilient High-Frequency Sleep Staging](https://www.nature.com/articles/s41746-021-00440-5)
* [U-Time: A Fully Convolutional Network for Time Series Segmentation Applied to Sleep Staging](https://arxiv.org/pdf/1910.11162.pdf)
* [DeepSleepNet: a Model for Automatic Sleep Stage Scoring based on Raw Single-Channel EEG](https://arxiv.org/pdf/1703.04046.pdf)

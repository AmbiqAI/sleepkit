---
title:
---
#

<p align="center">
  <a href="https://github.com/AmbiqAI/sleepkit"><img src="./assets/sleepkit-banner.png" alt="SleepKit"></a>
</p>

<p align="center" style="color:red;font-size:1.5em;">
🚧 SleepKit is currently under active development 🚧
</p>

---

**Documentation**: <a href="https://ambiqai.github.io/sleepkit" target="_blank">https://ambiqai.github.io/sleepkit</a>

**Source Code**: <a href="https://github.com/AmbiqAI/sleepkit" target="_blank">https://github.com/AmbiqAI/sleepkit</a>

---

SleepKit is a collection of optimized open-source TinyML models purpose-built to enable running a variety of real-time sleep-monitoring applications on battery-powered, edge devices. The objective is to outperform conventional, hand-crafted algorithms with efficient AI models that fit within stringent resource constraints of embedded devices. SleepKit leverages Ambiq's [PhysioKit](https://ambiqai.github.io/physiokit) to extract a variety of rich physiological features from raw sensory data.

**Key Features:**

* Efficient: Leverage modern AI techniques coupled with Ambiq's ultra-low power SoCs
* Generalizable: Multi-modal, multi-task, multi-dataset
* Accurate: Achieve SoTA with minimal resources

## Requirements

* [Python 3.11+](https://www.python.org)
* [Poetry 1.2.1+](https://python-poetry.org/docs/#installation)

The following are also required to compile/flash the binary for the EVB demo:

* [Arm GNU Toolchain 11.3](https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads)
* [Segger J-Link v7.56+](https://www.segger.com/downloads/jlink/)

!!! note
    A [VSCode Dev Container](https://code.visualstudio.com/docs/devcontainers/containers) is also available and defined in [./.devcontainer](https://github.com/AmbiqAI/sleepkit/tree/main/.devcontainer).

## Installation

To get started, first install the local python package `sleepkit` along with its dependencies via `Poetry`:

<div class="termy">

```console
$ poetry install

---> 100%
```

</div>

<!-- ```bash
poetry install
``` -->

## Usage

__SleepKit__ can be used as either a CLI-based app or as a python package to perform advanced experimentation. In both forms, SleepKit exposes a number of modes and tasks discussed below. Refer to the [Overview Guide](./overview.md) to learn more about available options and configurations.

## Modes

* `download`: Download datasets
* `feature`: Extract features from dataset(s)
* `train`: Train a model for specified task and datasets
* `evaluate`: Evaluate a model for specified task and datasets
* `export`: Export a trained model to TensorFlow Lite and TFLM
* `demo`: Run demo on PC or EVB

## Tasks

* `stage`: Perform 2, 3, 4, or 5 stage sleep detection
<!-- * `apnea`: Detect hypopnea/apnea events
* `arousal`: Detect sleep arousal events -->

## Architecture

SleepKit leverages modern architectural design strategies to achieve high accuracy while maintaining a small memory footprint and low power consumption. Refer to specific task guides for additional details on the full model design.

* Seperable (depthwise + pointwise) Convolutions
* Inverted Residual Bottleneck
* Squeeze & Excitation Blocks
* Over-Parameterized Convolutional Branches
* Dilated Convolutions

## Datasets

SleepKit uses several open-source datasets for training each of the task's models. In general, we use commercial use friendly datasets that are publicly available. Check out the [Datasets Guide](./datasets.md) to learn more about the datasets used along with their corresponding licenses and limitations.

## Results

The following table provides the latest performance and accuracy results of all models when running on Apollo4 Plus EVB. Additional result details can be found in [Results Section](./results.md).

| Task           | Params   | FLOPS   | Metric     | Cycles/Inf | Time/Inf   |
| -------------- | -------- | ------- | ---------- | ---------- | ---------- |
| 2-Stage Sleep  | --K      | --M     | ---% F1   | ---ms       | ---M       |
| 3-Stage Sleep  | --K      | --M     | ---% F1   | ---ms       | ---M       |
| 4-Stage Sleep  | --K      | --M     | ---% F1   | ---ms       | ---M       |
| 5-Stage Sleep  | --K      | --M     | ---% F1   | ---ms       | ---M       |
<!-- | Sleep Apnea    | --K      | --M     | ---% F1   | ---ms       | ---M       |
| Sleep Arousal  | --K      | --M     | ---% F1   | ---ms       | ---M       | -->

## References

* [U-Sleep: Resilient High-Frequency Sleep Staging](https://doi.org/10.1038/s41746-021-00440-5)
* [U-Time: A Fully Convolutional Network for Time Series Segmentation Applied to Sleep Staging](https://doi.org/10.48550/arXiv.1910.11162)
* [DeepSleepNet: a Model for Automatic Sleep Stage Scoring based on Raw Single-Channel EEG](https://doi.org/10.48550/arXiv.1703.04046)
* [AI-Driven sleep staging from actigraphy and heart rate](https://doi.org/10.1371/journal.pone.0285703)
* [TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis](https://doi.org/10.48550/arXiv.2210.02186)
* [The Promise of Sleep: A Multi-Sensor Approach for Accurate Sleep Stage Detection Using the Oura Ring](https://doi.org/10.3390/s21134302)
* [Interrater reliability of sleep stage scoring: a meta-analysis](https://doi.org/10.5664/jcsm.9538)
* [Development of generalizable automatic sleep staging using heart rate and movement based on large databases](https://doi.org/10.1007/s13534-023-00288-6)

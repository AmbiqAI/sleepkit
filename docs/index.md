---
title:
---
#

<p align="center">
  <a href="https://github.com/AmbiqAI/sleepkit"><img src="./assets/sleepkit-banner.png" alt="SleepKit"></a>
</p>

<p style="color:rgb(201,48,198); font-size: 1.2em;">
🚧 SleepKit is under active development
</p>

---

**Documentation**: <a href="https://ambiqai.github.io/sleepkit" target="_blank">https://ambiqai.github.io/sleepkit</a>

**Source Code**: <a href="https://github.com/AmbiqAI/sleepkit" target="_blank">https://github.com/AmbiqAI/sleepkit</a>

---

SleepKit is an AI Development Kit (ADK) that enables developers to easily build and deploy real-time __sleep__ monitoring models on Ambiq's family of ultra-low power SoCs. SleepKit explores a number of sleep related tasks including sleep staging, sleep apnea detection, and sleep arousal detection. The kit includes a variety of datasets, efficient model architectures, and a number of pre-trained models. The objective of the models is to outperform conventional, hand-crafted algorithms with efficient AI models that still fit within the stringent resource constraints of embedded devices. Furthermore, the included models are trainined using a large variety datasets- using a subset of biological signals that can be captured from a single body location such as head, chest, or wrist/hand. The goal is to enable models that can be deployed in real-world commercial and consumer applications that are viable for long-term use.


**Key Features:**

* **Real-time**: Inference is performed in real-time on battery-powered, edge devices.
* **Efficient**: Leverage modern AI techniques coupled with Ambiq's ultra-low power SoCs
* **Generalizable**: Multi-modal, multi-task, multi-dataset
* **Accurate**: Achieve SoTA results with stringent resource constraints

## <span class="sk-h2-span">Requirements</span>

* [Python ^3.11](https://www.python.org)
* [Poetry ^1.6.1](https://python-poetry.org/docs/#installation)

The following are also required to compile/flash the binary for the EVB demo:

* [Arm GNU Toolchain ^12.2](https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads)
* [Segger J-Link ^7.92](https://www.segger.com/downloads/jlink/)

!!! note
    A [VSCode Dev Container](https://code.visualstudio.com/docs/devcontainers/containers) is also available and defined in [./.devcontainer](https://github.com/AmbiqAI/sleepkit/tree/main/.devcontainer).

## <span class="sk-h2-span">Installation</span>

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

---

## <span class="sk-h2-span">Usage</span>

__SleepKit__ can be used as either a CLI-based app or as a python package to perform advanced experimentation. In both forms, SleepKit exposes a number of modes and tasks discussed below. Refer to the [Overview Guide](./overview.md) to learn more about available options and configurations.

---

## <span class="sk-h2-span">Modes</span>

* `download`: Download datasets
* `feature`: Extract features from dataset(s)
* `train`: Train a model for specified task and dataset(s)
* `evaluate`: Evaluate a model for specified task and dataset(s)
* `export`: Export a trained model to TF Lite and TFLM
* `demo`: Run task-level demo on PC or EVB

---

## <span class="sk-h2-span">Tasks</span>

* `detect`: Detect sustained sleep/inactivity bouts
* `stage`: Perform advanced 2, 3, 4, or 5 stage sleep assessment
* `apnea`: Detect hypopnea/apnea events
* `arousal`: Detect sleep arousal events

---

## <span class="sk-h2-span">Architecture</span>

SleepKit leverages modern architectural design strategies to achieve high accuracy while maintaining a small memory footprint and low power consumption. Refer to specific task guides for additional details on the full model design.

* Seperable (depthwise + pointwise) Convolutions
* Inverted Residual Bottlenecks
* Squeeze & Excitation Blocks
* Over-Parameterized Convolutional Branches
* Dilated Convolutions

---

## <span class="sk-h2-span">Datasets</span>

SleepKit uses several open-source datasets for training each of the tasks. In general, we use commercial-use friendly datasets that are publicly available. Check out the [Datasets Guide](./datasets.md) to learn more about the datasets used along with their corresponding licenses and limitations.

---

## <span class="sk-h2-span">Model Zoo</span>

A number of pre-trained models are available for each task. These models are trained on a variety of datasets and are optimized for deployment on Ambiq's ultra-low power SoCs. Check out the [Model Zoo](./results.md) to learn more about the available models and their corresponding performance metrics.

---

## <span class="sk-h2-span">References</span>

* [U-Sleep: Resilient High-Frequency Sleep Staging](https://doi.org/10.1038/s41746-021-00440-5)
* [U-Time: A Fully Convolutional Network for Time Series Segmentation Applied to Sleep Staging](https://doi.org/10.48550/arXiv.1910.11162)
* [DeepSleepNet: a Model for Automatic Sleep Stage Scoring based on Raw Single-Channel EEG](https://doi.org/10.48550/arXiv.1703.04046)
* [AI-Driven sleep staging from actigraphy and heart rate](https://doi.org/10.1371/journal.pone.0285703)
* [TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis](https://doi.org/10.48550/arXiv.2210.02186)
* [The Promise of Sleep: A Multi-Sensor Approach for Accurate Sleep Stage Detection Using the Oura Ring](https://doi.org/10.3390/s21134302)
* [Interrater reliability of sleep stage scoring: a meta-analysis](https://doi.org/10.5664/jcsm.9538)
* [Development of generalizable automatic sleep staging using heart rate and movement based on large databases](https://doi.org/10.1007/s13534-023-00288-6)

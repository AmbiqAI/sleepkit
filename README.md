# SleepKit

🚧 SleepKit is currently under development 🚧

---

**Documentation**: <a href="https://ambiqai.github.io/sleepkit" target="_blank">https://ambiqai.github.io/sleepkit</a>

**Source Code**: <a href="https://github.com/AmbiqAI/sleepkit" target="_blank">https://github.com/AmbiqAI/sleepkit</a>

---


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

## Reference Papers

* [U-Sleep: Resilient High-Frequency Sleep Staging](https://www.nature.com/articles/s41746-021-00440-5)
* [U-Time: A Fully Convolutional Network for Time Series Segmentation Applied to Sleep Staging](https://arxiv.org/pdf/1910.11162.pdf)
* [DeepSleepNet: a Model for Automatic Sleep Stage Scoring based on Raw Single-Channel EEG](https://arxiv.org/pdf/1703.04046.pdf)

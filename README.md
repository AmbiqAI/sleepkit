<p align="center">
  <a href="https://github.com/AmbiqAI/sleepkit"><img src="./docs/assets/sleepkit-banner.png" alt="SleepKit"></a>
</p>

---

**Documentation**: <a href="https://ambiqai.github.io/sleepkit" target="_blank">https://ambiqai.github.io/sleepkit</a>

**Source Code**: <a href="https://github.com/AmbiqAI/sleepkit" target="_blank">https://github.com/AmbiqAI/sleepkit</a>

---

SleepKit is an AI Development Kit (ADK) that enables developers to easily build and deploy real-time __sleep-monitoring__ models on Ambiq's family of ultra-low power SoCs. SleepKit explores a number of sleep related tasks including sleep staging, and sleep apnea detection. The kit includes a variety of datasets, efficient model architectures, and a number of pre-trained models. The objective of the models is to outperform conventional, hand-crafted algorithms with efficient AI models that still fit within the stringent resource constraints of embedded devices. Furthermore, the included models are trainined using a large variety datasets- using a subset of biological signals that can be captured from a single body location such as head, chest, or wrist/hand. The goal is to enable models that can be deployed in real-world commercial and consumer applications that are viable for long-term use.


**Key Features:**

* **Real-time**: Inference is performed in real-time on battery-powered, edge devices.
* **Efficient**: Leverage modern AI techniques coupled with Ambiq's ultra-low power SoCs
* **Extensible**: Easily add new tasks, models, and datasets to the framework.
* **Accurate**: Achieve SoTA results with stringent resource constraints

## <span class="sk-h2-span">Requirements</span>

* [Python ^3.11](https://www.python.org)
* [uv ^1.6.1+](https://docs.astral.sh/uv/getting-started/installation/)

The following are also required to compile/flash the binary for the EVB demo:

* [Arm GNU Toolchain ^12.2](https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads)
* [Segger J-Link ^7.92](https://www.segger.com/downloads/jlink/)

!!! note
    A [VSCode Dev Container](https://code.visualstudio.com/docs/devcontainers/containers) is also available and defined in [./.devcontainer](https://github.com/AmbiqAI/sleepkit/tree/main/.devcontainer).

## <span class="sk-h2-span">Installation</span>

To get started, first install the local python package `sleepkit` along with its dependencies via `PyPi`:

```bash
$ pip install sleepkit
```

Alternatively, you can install the package from source by cloning the repository and running the following command:

```bash
git clone https://github.com/AmbiqAI/sleepkit.git
cd sleepkit
uv sync
```

---

## <span class="sk-h2-span">Usage</span>

__SleepKit__ can be used as either a CLI-based app or as a python package to perform advanced experimentation. In both forms, SleepKit exposes a number of modes and tasks discussed below. Refer to the [Overview Guide](https://ambiqai.github.io/sleepkit/quickstart) to learn more about available options and configurations.

---

## <span class="sk-h2-span">Tasks</span>

__SleepKit__ includes a number of built-in **tasks**. Each task provides reference routines for training, evaluating, and exporting the model. The routines can be customized by providing a configuration file or by setting the parameters directly in the code. Additional tasks can be easily added to the __SleepKit__ framework by creating a new task class and registering it to the __task factory__.

- **Detect**: Detect sustained sleep/inactivity bouts
- **Stage**: Perform advanced 2, 3, 4, or 5 stage sleep assessment
- **Apnea**: Detect hypopnea/apnea events

---

## <span class="sk-h2-span">Modes</span>

__SleepKit__ provides a number of **modes** that can be invoked for a given task. These modes can be accessed via the CLI or directly from the `task` within the Python package.

- **Download**: Download specified datasets
- **Feature**: Generate features for given dataset(s)
- **Train**: Train a model for specified task and features
- **Evaluate**: Evaluate a model for specified task and features
- **Export**: Export a trained model to TF Lite and TFLM
- **Demo**: Run task-level demo on PC or EVB

---

## <span class="sk-h2-span">Datasets</span>

__SleepKit__ includes several open-source datasets for training each of the SleepKit tasks via a __dataset factory__. For certain tasks, we also provide synthetic data provided by [PhysioKit](https://ambiqai.github.io/physiokit) to help improve model generalization. Each dataset has a corresponding Python class to aid in downloading and generating data for the given task. Additional datasets can be easily added to the SleepKit framework by creating a new dataset class and registering it to the dataset factory.

- **MESA**: A large-scale polysomnography dataset with 6,814 subjects collected from 6 field centers.
- **CMIDSS**: A dataset of 300 subjects with over 500 multi-day recordings of wrist-worn accelerometer data annotated with two event types: onset, the beginning of sleep, and wakeup, the end of sleep.
- **YSYW**: A dataset of 1,983 polysomnography recordings provided by the Massachusetts General Hospital’s (MGH) Sleep Lab.
- **STAGES**: A dataset from the Stanford Technology Analytics and Genomics in Sleep (STAGES) study involving 20 data collection sites from six centers.

---

## <span class="sk-h2-span">Models</span>

__SleepKit__ provides a __model factory__ that allows you to easily create and train customized models. The model factory includes a number of modern networks well suited for efficient, real-time edge applications. Each model architecture exposes a number of high-level parameters that can be used to customize the network for a given application. These parameters can be set as part of the configuration accessible via the CLI and Python package.

---

## <span class="sk-h2-span">Model Zoo</span>

A number of pre-trained models are available for each task. These models are trained on a variety of datasets and are optimized for deployment on Ambiq's ultra-low power SoCs. In addition to providing links to download the models, __SleepKit__ provides the corresponding configuration files and performance metrics. The configuration files allow you to easily retrain the models or use them as a starting point for a custom model. Furthermore, the performance metrics provide insights into the model's accuracy, precision, recall, and F1 score. For a number of the models, we provide experimental and ablation studies to showcase the impact of various design choices. Check out the [Model Zoo](https://ambiqai.github.io/sleepkit/zoo) to learn more about the available models and their corresponding performance metrics.

---

## <span class="sk-h2-span">Guides</span>

Checkout the [Guides](https://ambiqai.github.io/sleepkit/guides) to see detailed examples and tutorials on how to use SleepKit for a variety of tasks. The guides provide step-by-step instructions on how to train, evaluate, and deploy models for a given task. In addition, the guides provide insights into the design choices and performance metrics for the models. The guides are designed to help you get up and running quickly and to provide a deeper understanding of the models and tasks available in SleepKit.

---

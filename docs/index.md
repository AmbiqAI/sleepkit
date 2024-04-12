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

SleepKit is an AI Development Kit (ADK) that enables developers to easily build and deploy real-time __sleep-monitoring__ models on Ambiq's family of ultra-low power SoCs. SleepKit explores a number of sleep related tasks including sleep staging, and sleep apnea detection. The kit includes a variety of datasets, feature sets, efficient model architectures, and a number of pre-trained models. The objective of the models is to outperform conventional, hand-crafted algorithms with efficient AI models that still fit within the stringent resource constraints of embedded devices. Furthermore, the included models are trainined using a large variety datasets- using a subset of biological signals that can be captured from a single body location such as head, chest, or wrist/hand. The goal is to enable models that can be deployed in real-world commercial and consumer applications that are viable for long-term use.

**Key Features:**

* **Real-time**: Inference is performed in real-time on battery-powered, edge devices.
* **Efficient**: Leverage modern AI techniques coupled with Ambiq's ultra-low power SoCs
* **Generalizable**: Multi-modal, multi-task, multi-dataset
* **Open Source**: SleepKit is open source and available on GitHub.

Please explore the SleepKit Docs, a comprehensive resource designed to help you understand and utilize all the built-in features and capabilities.

## <span class="sk-h2-span">Getting Started</span>

- **Install** `SleepKit` with pip/poetry and getting up and running in minutes. &nbsp; [:material-clock-fast: Install SleepKit](./quickstart.md/#install){ .md-button }
- **Train** a model with a custom network &nbsp; [:fontawesome-solid-brain: Train a Model](modes/train.md){ .md-button }
- **Tasks** `SleepKit` provides tasks like staging, and apnea &nbsp; [:material-magnify-expand: Explore Tasks](tasks/index.md){ .md-button }
- **Datasets** Several built-in datasets can be leveraged &nbsp; [:material-database-outline: Explore Datasets](./datasets/index.md){ .md-button }
- **Model Zoo** Pre-trained models are available for each task &nbsp; [:material-download: Explore Models](./zoo/index.md){ .md-button }

## <span class="sk-h2-span">Installation</span>

To get started, first install the local python package `sleepkit` along with its dependencies via `pip` or `Poetry`:

=== "Poetry install"

    <div class="termy">

    ```console
    $ poetry install .

    ---> 100%
    ```

    </div>

=== "Pip install"

    <div class="termy">

    ```console
    $ pip install sleepkit

    ---> 100%
    ```

    </div>

---

## <span class="sk-h2-span">Usage</span>

__SleepKit__ can be used as either a CLI-based tool or as a Python package to perform advanced development. In both forms, SleepKit exposes a number of modes and tasks outlined below. In addition, by leveraging highly-customizable configurations, SleepKit can be used to create custom workflows for a given application with minimal coding. Refer to the [Quickstart](./quickstart.md) to quickly get up and running in minutes.

---

## <span class="sk-h2-span">Modes</span>

__SleepKit__ provides a number of [modes](./modes/index.md) that can be invoked for a given task. These modes can be accessed via the CLI or directly within the Python package.

- **[Download](./modes/download.md)**: Download specified datasets
- **[Feature](./modes/feature.md)**: Generate features from datasets
- **[Train](./modes/train.md)**: Train a model for specified task and feature set
- **[Evaluate](./modes/evaluate.md)**: Evaluate a model for specified task and feature set
- **[Export](./modes/export.md)**: Export a trained model to TensorFlow Lite and TFLM
- **[Demo](./modes/demo.md)**: Run task-level demo on PC or remotely on Ambiq EVB

---

## <span class="sk-h2-span">Task Factory</span>

__SleepKit__ includes a number of built-in [tasks](./tasks/index.md). Each task provides reference routines for training, evaluating, and exporting the model. The routines can be customized by providing a configuration file or by setting the parameters directly in the code. Additional tasks can be easily added to the __SleepKit__ framework by creating a new task class and registering it to the __task factory__.

- **[Detect](./tasks/detect.md)**: Detect sustained sleep/inactivity bouts
- **[Stage](./tasks/stage.md)**: Perform advanced sleep stage assessment
- **[Apnea](./tasks/apnea.md)**: Detect hypopnea/apnea events
<!-- - **[Arousal](./tasks/arousal.md)**: Detect sleep arousal events -->
- **[BYOT](./tasks/byot.md)**: Bring-Your-Own-Task (BYOT) to create custom tasks

---

## <span class="sk-h2-span">Model Factory</span>

__SleepKit__ provides a __model factory__ that allows you to easily create and train customized models. The model factory includes a number of modern networks well suited for efficient, real-time edge applications. Each model architecture exposes a number of high-level parameters that can be used to customize the network for a given application. These parameters can be set as part of the configuration accessible via the CLI and Python package. Check out the [Model Factory Guide](./models/index.md) to learn more about the available network architectures.

---

## <span class="sk-h2-span">Dataset Factory</span>

__SleepKit__ exposes several open-source datasets via the __dataset factory__. Each dataset has a corresponding Python class to aid in downloading and extracting the data. The datasets are used to generate feature sets that are then used to train and evaluate the models. Check out the [Dataset Factory Guide](./datasets/index.md) to learn more about the available datasets along with their corresponding licenses and limitations.

* **[MESA](./datasets/mesa.md)**: A longitudinal investigation of factors associated with the development of subclinical cardiovascular disease and the progression of subclinical to clinical cardiovascular disease in 6,814 black, white, Hispanic, and Chinese

* **[CMIDSS](./datasets/cmidss.md)**: The Child Mind Institute - Detect Sleep States (CMIDSS) dataset comprises 300 subjects with over 500 multi-day recordings of wrist-worn accelerometer data annotated with two event types: onset, the beginning of sleep, and wakeup, the end of sleep.

* **[YSYW](./datasets/ysyw.md)**: A total of 1,983 PSG recordings were provided by the Massachusetts General Hospital’s (MGH) Sleep Lab in the Sleep Division together with the Computational Clinical Neurophysiology Laboratory, and the Clinical Data Ani- mation Center.

* **[STAGES](./datasets/stages.md)**: The Stanford Technology Analytics and Genomics in Sleep (STAGES) study is a prospective cross-sectional, multi-site study involving 20 data collection sites from six centers including Stanford University, Bogan Sleep Consulting, Geisinger Health, Mayo Clinic, MedSleep, and St. Luke's Hospital.

---

## <span class="sk-h2-span">Feature Store</span>

__SleepKit__ provides a __feature store__ that allows you to easily create and extract features from the datasets. The feature store includes a number of feature sets used to train the included model zoo. Each feature set exposes a number of high-level parameters that can be used to customize the feature extraction process for a given application. These parameters can be set as part of the configuration accessible via the CLI and Python package. Check out the [Feature Store Guide](./features/index.md) to learn more about the available feature set generators.

---

## <span class="sk-h2-span">Model Zoo</span>

A number of pre-trained models are available for each task. These models are trained on a variety of datasets and are optimized for deployment on Ambiq's ultra-low power SoCs. In addition to providing links to download the models, __SleepKit__ provides the corresponding configuration files and performance metrics. The configuration files allow you to easily recreate the models or use them as a starting point for custom solutions. Furthermore, the performance metrics provide insights into the model's accuracy, precision, recall, and F1 score. For a number of the models, we provide experimental and ablation studies to showcase the impact of various design choices. Check out the [Model Zoo](./zoo/index.md) to learn more about the available models and their corresponding performance metrics. Also explore the [Experiments](./experiments/index.md) to learn more about the ablation studies and experimental results.

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

---

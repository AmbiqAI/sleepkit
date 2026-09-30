# From sleep signals to Edge AI

<div class="sleepkit-intro">

sleepKIT is a Python development kit for building sleep-monitoring AI for Ambiq devices. Turn sensor recordings into features, train and evaluate models, and prepare them for deployment at the edge.

</div>

## A starting point for your sleep-monitoring application

Building a sleep model takes more than choosing a network. You need recordings and labels, consistent signal processing, a training setup and a way to evaluate the results. sleepKIT brings these pieces together in a configurable development workflow for engineers and researchers working on wearable and edge applications.

Start with the included dataset integrations, feature sets, model configurations and examples. Run experiments from the command line or compose a custom workflow with the Python API. You can replace individual components as your application develops, then use the export tools to prepare models for integration on Ambiq devices.

## Included tasks

Start with a built-in task, or extend the same workflow with your own training, evaluation and export routines.

<div class="sleepkit-task-grid">
<a class="sleepkit-task-card" href="/sleepkit/tasks/detect/">
<strong>Sleep detection</strong>
<span>Identify sleep and wake periods from wrist-worn motion signals.</span>
<span class="sleepkit-card-link">Explore detection →</span>
</a>
<a class="sleepkit-task-card" href="/sleepkit/tasks/stage/">
<strong>Sleep staging</strong>
<span>Explore feature sets and models for classifying sleep stages.</span>
<span class="sleepkit-card-link">Explore staging →</span>
</a>
<a class="sleepkit-task-card" href="/sleepkit/tasks/apnea/">
<strong>Sleep apnea</strong>
<span>Explore the task and available resources for apnea event detection.</span>
<span class="sleepkit-card-link">Explore apnea →</span>
</a>
<a class="sleepkit-task-card" href="/sleepkit/tasks/byot/">
<strong>Bring your own task</strong>
<span>Register a custom task and reuse sleepKIT’s configuration, datasets and development workflow.</span>
<span class="sleepkit-card-link">Create a task →</span>
</a>
</div>

Code and Ambiq-authored documentation use BSD-3-Clause, except separately licensed material. Model weights and datasets have [their own terms](model-licensing-policy.md).

## Installation

Use **uv** for a Python project, **uvx** to run the CLI in an isolated environment, or **pipx** to keep the CLI installed. Choose **Git clone** when developing sleepKIT itself.

<div class="sleepkit-install">

=== "uv project"

    Create a project with Python 3.12, then add sleepKIT. In an existing uv project, start with `uv add sleepkit`.

    ```bash
    uv init --python 3.12 my-sleep-project
    cd my-sleep-project
    uv add sleepkit
    uv run sleepkit --help
    ```

=== "uvx"

    Run the CLI without adding sleepKIT to a project. The first invocation downloads sleepKIT and its dependencies into an isolated environment.

    ```bash
    uvx --python 3.12 sleepkit --help
    ```

    Use a project installation for Python imports and notebooks.

=== "pipx"

    Install the CLI in its own environment. This command uses an installed Python 3.12 interpreter.

    ```bash
    pipx install --python python3.12 sleepkit
    sleepkit --help
    ```

    If the command is not on your PATH, run `pipx ensurepath` and reopen your terminal. Use a project installation for Python imports and notebooks.

=== "pip"

    Install into an activated virtual environment.

    ```bash
    python -m pip install sleepkit
    sleepkit --help
    ```

=== "Git clone"

    Work with the repository source and its development dependencies.

    ```bash
    git clone https://github.com/AmbiqAI/sleepkit.git
    cd sleepkit
    uv sync --python 3.12
    uv run sleepkit --help
    ```

</div>

Need the package manager first? See the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) or [pipx installation guide](https://pipx.pypa.io/stable/installation/). The [Quickstart](./quickstart.md) covers configuration and your first workflow.


---

## Usage

__sleepKIT__ can be used as either a CLI-based tool or as a Python package to perform advanced development. In both forms, sleepKIT exposes a number of modes and tasks outlined below. In addition, by leveraging highly-customizable configurations, sleepKIT can be used to create custom workflows for a given application with minimal coding. Refer to the [Quickstart](./quickstart.md) to quickly get up and running in minutes.

---

## Modes

The __ADK__ provides a number of [modes](./modes/index.md) that can be invoked for a given task. These modes can be accessed via the CLI or directly within the Python package. Each mode is accompanied by a set of [task parameters](./modes/configuration.md) that can be customized to fit the user's needs.

- **[Download](./modes/download.md)**: Download specified datasets
- **[Feature](./features/index.md)**: Generate features from datasets
- **[Train](./modes/train.md)**: Train a model for specified task and feature set
- **[Evaluate](./modes/evaluate.md)**: Evaluate a model for specified task and feature set
- **[Export](./modes/export.md)**: Export a trained model to TensorFlow Lite and TFLM
- **[Demo](./modes/demo.md)**: Run task-level demo on PC or remotely on Ambiq EVB

---

## Datasets

__sleepKIT__ includes several open-source datasets via the __dataset factory__. Each dataset has a corresponding Python class to aid in downloading and extracting the data. The datasets are used to generate feature sets that are then used to train and evaluate the models. Check out the [Datasets Guide](./datasets/index.md) to learn more about the available datasets along with their corresponding licenses and limitations.

* **[MESA](./datasets/mesa.md)**: A longitudinal investigation of factors associated with the development of subclinical cardiovascular disease and the progression of subclinical to clinical cardiovascular disease in 6,814 black, white, Hispanic, and Chinese
* **[CMIDSS](./datasets/cmidss.md)**: The Child Mind Institute - Detect Sleep States (CMIDSS) dataset comprises 300 subjects with over 500 multi-day recordings of wrist-worn accelerometer data annotated with two event types: onset, the beginning of sleep, and wakeup, the end of sleep.
* **[YSYW](./datasets/ysyw.md)**: A total of 1,983 PSG recordings were provided by the Massachusetts General Hospital’s (MGH) Sleep Lab in the Sleep Division together with the Computational Clinical Neurophysiology Laboratory, and the Clinical Data Ani- mation Center.
* **[STAGES](./datasets/stages.md)**: The Stanford Technology Analytics and Genomics in Sleep (STAGES) study is a prospective cross-sectional, multi-site study involving 20 data collection sites from six centers including Stanford University, Bogan Sleep Consulting, Geisinger Health, Mayo Clinic, MedSleep, and St. Luke's Hospital.

---

## Models

The __ADK__ provides a variety of model architectures geared towards efficient, real-time edge applications. These models are provided by Ambiq's [helia-edge](https://ambiqai.github.io/helia-edge/) and expose a set of parameters that can be used to fully customize the network for a given application. In addition, sleepKIT includes a model factory, [ModelFactory](./models/index.md#model-factory), to register current models as well as allow new custom architectures to be added. Check out the [Models Guide](./models/index.md) to learn more about the available network architectures and model factory.

---

## Features

The __ADK__ provides a __feature store__ that allows you to easily create and extract features from the given datasets. The feature store includes a number of feature sets used to train the included model zoo. Each feature set exposes a number of high-level parameters that can be used to customize the feature extraction process for a given application. These parameters can be set as part of the configuration accessible via the CLI and Python package. Check out the [Features Guide](./features/index.md) to learn more about the available feature set generators.

---

## Model Zoo

A number of pre-trained models are available for each task. These models are trained on a variety of datasets and are optimized for deployment on Ambiq's ultra-low power SoCs. In addition to providing links to download the models, __sleepKIT__ provides the corresponding configuration files and performance metrics. The configuration files allow you to easily recreate the models or use them as a starting point for custom solutions. Furthermore, the performance metrics provide insights into the model's accuracy, precision, recall, and F1 score. For a number of the models, we provide experimental and ablation studies to showcase the impact of various design choices. Check out the [Model Zoo](./zoo/index.md) to learn more about the available models and their corresponding performance metrics.

---

## [Guides](./guides/index.md)

Checkout the [Guides](./guides/index.md) to see detailed examples and tutorials on how to use sleepKIT for a variety of tasks. The guides provide step-by-step instructions on how to train, evaluate, and deploy models for a given task. In addition, the guides provide insights into the design choices and performance metrics for the models. The guides are designed to help you get up and running quickly and to provide a deeper understanding of the capabilities provided by sleepKIT.

---

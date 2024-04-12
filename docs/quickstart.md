# Quickstart

## <span class="sk-h2-span">Install SleepKit</span>

We provide several installation methods including pip, poetry, and Docker. Install __SleepKit__ via pip/poetry for the latest stable release or by cloning the GitHub repo for the most up-to-date. Additionally, a [VSCode Dev Container](https://code.visualstudio.com/docs/devcontainers/containers) is available and defined in [./.devcontainer](https://github.com/AmbiqAI/sleepkit/tree/main/.devcontainer) to run in an isolated Docker environment.

!!! install

    === "Pip/Poetry install"

        Install the SleepKit package using pip or Poetry. Make sure to have the Git command-line tool installed on your system. The @main command installs the main branch and may be modified to another branch, i.e. @release.

        ```bash
        # Install with pip
        pip install git+https://github.com/AmbiqAI/sleepkit.git@main
        ```

        Or, if you prefer to use Poetry, you can install the package with the following command:

        ```bash
        # Install with poetry
        poetry add git+https://github.com/AmbiqAI/sleepkit.git@main
        ```

    === "Git clone"

        Clone the repository if you are interested in contributing to the development or wish to experiment with the latest source code. After cloning, navigate into the directory and install the package. In this mode, Poetry is recommended.

        ```bash
        # Clone the repository
        git clone https://github.com/AmbiqAI/sleepkit.git

        # Navigate to the cloned directory
        cd sleepkit

        # Install the package in editable mode for development
        poetry install
        ```

## <span class="sk-h2-span">Requirements</span>

* [Python ^3.11+](https://www.python.org)
* [Poetry ^1.6.1+](https://python-poetry.org/docs/#installation)

Check the project's [pyproject.toml](https://github.com/AmbiqAI/sleepkit/blob/main/pyproject.toml) file for a list of up-to-date Python dependencies. Note that the installation methods above install all required dependencies. The following are also required to compile and flash the binary to evaluate the demos running on Ambiq's evaluation boards (EVBs):

* [Arm GNU Toolchain ^12.2](https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads)
* [Segger J-Link ^7.92](https://www.segger.com/downloads/jlink/)

Once installed, __SleepKit__ can be used as either a CLI-based tool or as a Python package to perform advanced experimentation.

---

## <span class="sk-h2-span">Use SleepKit with CLI</span>

The SleepKit command line interface (CLI) allows for simple single-line commands without the need for a Python environment. The CLI requires no customization or Python code. You can simply run all tasks from the terminal with the `sleepkit` command. Check out the [CLI Guide](./usage/cli.md) to learn more about available options.

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
        sleepkit -m download -c ./configs/download-datasets.json
        ```
    === "Feature"
        Extract features from the datasets using the supplied configuration file.

        ```bash
        sleepkit -m feature -c ./configs/sleep-stage-features.json
        ```

    === "Train"
        Train a sleep stage model using the supplied configuration file.

        ```bash
        sleepkit -m train -t stage -c ./configs/sleep-stage-2.json
        ```

    === "Evaluate"
        Evaluate the trained sleep stage model using the supplied configuration file.

        ```bash
        sleepkit -m evaluate -t stage  -c ./configs/sleep-stage-2.json
        ```

    === "Demo"
        Run demo on trained sleep stage model using the supplied configuration file.

        ```bash
        sleepkit -m demo -t stage -c ./configs/sleep-stage-2.json
        ```

## <span class="sk-h2-span">Use SleepKit with Python</span>

The __SleepKit__ Python package allows for more fine-grained control and customization. You can use the package to train, evaluate, and deploy models for a variety of tasks. The package is designed to be simple and easy to use.

For example, you can create a custom model, train it, evaluate its performance on a validation set, and even export a quantized TensorFlow Lite model for deployment. Check out the [Python Guide](./usage/python.md) to learn more about using SleepKit as a Python package.

!!! Example

    --8<-- "assets/usage/python-full-snippet.md"


!!! note
    If using editable mode via Poetry, be sure to activate the python environment: `poetry shell`. On Windows using Powershell, use `.venv\Scripts\activate.ps1`.

---

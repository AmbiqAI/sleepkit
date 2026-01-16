# :material-clock-fast: Quickstart

## Install sleepKIT

We provide several installation methods including pip, uv, and Docker. Install __sleepKIT__ via pip/uv for the latest stable release or by cloning the GitHub repo for the most up-to-date. Additionally, a [VSCode Dev Container](https://code.visualstudio.com/docs/devcontainers/containers) is available and defined in [./.devcontainer](https://github.com/AmbiqAI/sleepkit/tree/main/.devcontainer) to run in an isolated Docker environment.

!!! install

    === "Git clone"

        Clone the repository if you are interested in contributing to the development or wish to experiment with the latest source code. After cloning, navigate into the directory and install the package. In this mode, uv is recommended.

        ```bash
        # Clone the repository
        git clone https://github.com/AmbiqAI/sleepkit.git

        # Navigate to the cloned directory
        cd sleepkit

        # Install the package in editable mode for development
        uv sync
        ```

        When using editable mode via uv, be sure to activate the python environment: `source .venv/bin/activate`. <br>
        On Windows using Powershell, use `.venv\Scripts\activate`.

    === "PyPI install"

        Install the sleepKIT package using pip or uv.
        Visit the Python Package Index (PyPI) for more details on the package: [https://pypi.org/project/sleepkit/](https://pypi.org/project/sleepkit/)

        ```bash
        # Install with pip
        pip install sleepkit
        ```

        Or, if you prefer to use uv, you can install the package with the following command:

        ```bash
        # Install with uv
        uv add sleepkit
        ```

        Alternatively, you can install the latest development version directly from the GitHub repository. Make sure to have the Git command-line tool installed on your system. The @main command installs the main branch and may be modified to another branch, i.e. @canary.

        ```bash
        pip install git+https://github.com/AmbiqAI/sleepkit.git@main
        ```

        Or, using uv:

        ```bash
        uv add git+https://github.com/AmbiqAI/sleepkit.git@main
        ```

## Requirements

* [Python ^3.12+](https://www.python.org)
* [uv ^1.6.1+](https://docs.astral.sh/uv/getting-started/installation/)

Check the project's [pyproject.toml](https://github.com/AmbiqAI/sleepkit/blob/main/pyproject.toml) file for a list of up-to-date Python dependencies. Note that the installation methods above install all required dependencies. The following are optional dependencies only needed when running `demo` command using Ambiq's evaluation board (`EVB`) backend:

* [Arm GNU Toolchain ^12.2](https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads)
* [Segger J-Link ^7.92](https://www.segger.com/downloads/jlink/)

Once installed, __sleepKIT__ can be used as either a CLI-based tool or as a Python package to perform advanced experimentation.

---

## Use sleepKIT with CLI

The sleepKIT command line interface (CLI) allows for simple single-line commands without the need for a Python environment. The CLI requires no customization or Python code. You can simply run all tasks from the terminal with the `sleepkit` command. Check out the [CLI Guide](./usage/cli.md) to learn more about available options.

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
        sleepkit -m download -c ./configuration.json
        ```
    === "Feature"
        Extract features from the datasets using the supplied configuration file.

        ```bash
        sleepkit -m feature -c ./configuration.json
        ```

    === "Train"
        Train a sleep stage model using the supplied configuration file.

        ```bash
        sleepkit -m train -t stage -c ./configuration.json
        ```

    === "Evaluate"
        Evaluate the trained sleep stage model using the supplied configuration file.

        ```bash
        sleepkit -m evaluate -t stage  -c ./configuration.json
        ```

    === "Demo"
        Run demo on trained sleep stage model using the supplied configuration file.

        ```bash
        sleepkit -m demo -t stage -c ./configuration.json
        ```

## Use sleepKIT with Python

The __sleepKIT__ Python package allows for more fine-grained control and customization. You can use the package to train, evaluate, and deploy models for a variety of tasks. The package is designed to be simple and easy to use.

For example, you can create a custom model, train it, evaluate its performance on a validation set, and even export a quantized TensorFlow Lite model for deployment. Check out the [Python Guide](./usage/python.md) to learn more about using sleepKIT as a Python package.

!!! Example

    ```py linenums="1"

    import sleepkit as sk

    params = sk.HKTaskParams(...)  # Expand to see example (1)

    task = sk.TaskFactory.get("stage")

    task.download(params)  # Download dataset(s)

    task.feature(params)  # Generate features

    task.train(params)  # Train the model

    task.evaluate(params)  # Evaluate the model

    task.export(params)  # Export to TFLite

    ```

    1. Configuration parameters:
    --8<-- "assets/usage/python-configuration.md"


---

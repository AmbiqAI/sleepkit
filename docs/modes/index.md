# Modes overview

Modes are the steps in a sleepKIT task: acquire data, generate features, train, evaluate, export, and run a demo. Select a mode with `--mode` and provide the task configuration with `--config`.

## Choose a mode

| Mode | Purpose | Guide |
| --- | --- | --- |
| `download` | Fetch the datasets selected in your configuration. | [Download datasets](./download.md) |
| `feature` | Generate features from your datasets. | [Feature generation](../features/index.md) |
| `train` | Train the task model with the configured features and settings. | [Train a model](./train.md) |
| `evaluate` | Evaluate the model on the configured test data. | [Evaluate a model](./evaluate.md) |
| `export` | Convert the trained model for deployment. | [Export a model](./export.md) |
| `demo` | Run a task-level demonstration with the configured backend. | [Run a demo](./demo.md) |

## Run a step

```bash title="Train a staging model"
sleepkit --mode train --task stage --config configuration.json
```

Use the same configuration for related steps so that dataset, feature and model settings stay consistent. Complete data download and feature generation before training; evaluate the trained model before exporting it.

Demo support depends on the task and backend. See the demo guide for setup and limitations; running a demo does not establish model accuracy.

## Python workflow

The Python interface exposes the same task operations for scripts and notebooks. See [Quickstart](../quickstart.md#use-sleepkit-with-python) for a configuration-driven example and [Configuration](./configuration.md) for parameter definitions.

# :simple-python: Python Usage

__sleepKIT__ python package allows for more fine-grained control and customization. You can use the package to train, evaluate, and deploy models for both built-in tasks and custom tasks. In addition, custom datasets and model architectures can be created and registered with corresponding factories.

## Overview

The main components of sleepKIT include the following:

### [Tasks](../tasks/index.md)

A [Task](../tasks/index.md) inherits from the [sk.Task](/sleepkit/api/sleepkit/tasks/task) class and provides implementations for each of the main modes: download, feature, train, evaluate, export, and demo. Each mode is provided with a set of parameters defined by [sk.TaskParams](/sleepkit/api/sleepkit/defines). Additional task-specific parameters can be extended to the `TaskParams` class. These tasks are then registered and accessed via the `sk.TaskFactory` using a unique task name as the key and the custom Task class as the value.

```py linenums="1"
import sleepkit as sk

task = sk.TaskFactory.get('stage')
```

### [Datasets](../datasets/index.md)

A dataset inherits from the [sk.Dataset](/sleepkit/api/sleepkit/datasets/dataset) class and provides implementations for downloading, preparing, and loading the dataset. Each dataset is provided with a set of custom parameters for initialization. The datasets are registered and accessed via the [DatasetFactory](/sleepkit/api/sleepkit/datasets/factory) using a unique dataset name as the key and the Dataset class as the value.

```py linenums="1"
import sleepkit as sk

ds = sk.DatasetFactory.get('cmidss')(path='./datasets/cmidss')
```

### [Features](../features/index.md)

Since each task will require specific transformations of the data, a feature store is used to generate features from the dataset. The feature store provides a set of feature sets that can be used by the task. Each feature set is provided with a set of custom parameters for initialization. The feature sets are registered and accessed via the [sk.FeatureFactory](/sleepkit/api/sleepkit/features/factory) using a unique feature set name as the key and the Feature class as the value.



### [Models](../models/index.md)

Lastly, sleepKIT leverages [helia-edge's](https://ambiqai.github.io/helia-edge/) customizable model architectures. To enable creating custom network topologies from configuration files, sleepKIT provides a `sk.ModelFactory` that allows you to create models by specifying the model key and the model parameters. Each item in the factory is a callable that takes a `keras.Input`, model parameters, and number of classes as arguments and returns a `keras.Model`.

```python
import keras
import sleepkit as sk

inputs = keras.Input((256, 1), dtype="float32")
num_classes = 4
model_params = {"blocks": [{"filters": 16, "kernel": [1, 3]}]}

model = sk.ModelFactory.get('tcn')(
    inputs=inputs,
    params=model_params,
    num_classes=num_classes
)

```

## Usage

### Running a built-in task w/ existing datasets

1. Create a task configuration file defining the model, datasets, class labels, mode parameters, and so on. Have a look at the [sk.TaskParams](../modes/configuration.md#taskparams) for more details on the available parameters.

2. Leverage `sk.TaskFactory` to get the desired built-in task.

3. Run the task's main modes: `download`, `feature`, `train`, `evaluate`, `export`, and/or `demo`.


```py linenums="1"

from pathlib import Path
import sleepkit as sk

params = sk.TaskParams.model_validate_json(
    Path("configuration.json").read_text()
)

task = sk.TaskFactory.get("stage")

task.download(params)  # Download dataset(s)

task.feature(params)  # Generate features

task.train(params)  # Train the model

task.evaluate(params)  # Evaluate the model

task.export(params)  # Export to TFLite

```

**Example configuration**
--8<-- "assets/usage/json-configuration.md"

### Running a custom task w/ custom datasets

To create a custom task, check out the [Bring-Your-Own-Task Guide](../tasks/byot.md).

To create a custom dataset, check out the [Bring-Your-Own-Dataset Guide](../datasets/byod.md).

---

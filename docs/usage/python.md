# :simple-python: Python Usage

__SleepKit__ python package allows for more fine-grained control and customization. You can use the package to train, evaluate, and deploy models for both built-in taks and custom tasks. In addition, custom datasets and model architectures can be created and registered with corresponding factories.

## <span class="sk-h2-span">Overview</span>

The main components of SleepKit include the following:

### [Tasks](../tasks/index.md)

A [Task](../tasks/index.md) inherits from the [sk.Task](/sleepkit/api/sleepkit/tasks/task) class and provides implementations for each of the main modes: download, feature, train, evaluate, export, and demo. Each mode is provided with a set of parameters defined by [sk.TaskParams](/sleepkit/api/sleepkit/defines). Additional task-specific parameters can be extended to the `TaskParams` class. These tasks are then registered and accessed via the `TaskFactory` using a unique task name as the key and the custom Task class as the value.

```py linenums="1"
import sleepkit as sk

task = sk.TaskFactory.get('stage')
```

### [Datasets](../datasets/index.md)

A dataset inherits from the [sk.Dataset](/sleepkit/api/sleepkit/datasets/dataset) class and provides implementations for downloading, preparing, and loading the dataset. Each dataset is provided with a set of custom parameters for initialization. The datasets are registered and accessed via the [DatasetFactory](/sleepkit/api/sleepkit/datasets/factory) using a unique dataset name as the key and the Dataset class as the value.

```py linenums="1"
import sleepkit as sk

ds = sk.DatasetFactory.get('ecg-synthetic')(num_pts=100)
```

### [Features](../features/index.md)

Since each task will require specific transformations of the data, a feature store is used to generate features from the dataset. The feature store provides a set of feature sets that can be used by the task. Each feature set is provided with a set of custom parameters for initialization. The feature sets are registered and accessed via the [sk.FeatureFactory](/sleepkit/api/sleepkit/features/factory) using a unique feature set name as the key and the Feature class as the value.



### [Models](../models/index.md)

Lastly, SleepKit leverages [neuralspot-edge's](https://ambiqai.github.io/neuralspot-edge/) customizable model architectures. To enable creating custom network topologies from configuration files, SleepKit provides a `ModelFactory` that allows you to create models by specifying the model key and the model parameters. Each item in the factory is a callable that takes a `keras.Input`, model parameters, and number of classes as arguments and returns a `keras.Model`.

```
import keras
import sleepkit as sk

inputs = keras.Input((256, 1), dtype="float32")
num_classes = 4
model_params = dict(...)

model = sk.ModelFactory.get('tcn')(
    inputs=inputs,
    params=model_params,
    num_classes=num_classes
)

```

## <span class="sk-h2-span">Usage</span>

### Running a built-in task w/ existing datasets

1. Create a task configuration file defining the model, datasets, class labels, mode parameters, and so on. Have a look at the [sk.TaskParams](../modes/configuration.md#taskparams) for more details on the available parameters.

2. Leverage `TaskFactory` to get the desired built-in task.

3. Run the task's main modes: `download`, `feature`, `train`, `evaluate`, `export`, and/or `demo`.


```py linenums="1"

import sleepkit as hk

params = sk.TaskParams(...)  # (1)

task = sk.TaskFactory.get("stage")

task.download(params)  # Download dataset(s)

task.feature(params)  # Generate features

task.train(params)  # Train the model

task.evaluate(params)  # Evaluate the model

task.export(params)  # Export to TFLite

```

1. Example configuration:
--8<-- "assets/usage/python-configuration.md"

### Running a custom task w/ custom datasets

To create a custom task, check out the [Bring-Your-Own-Task Guide](../tasks/byot.md).

To create a custom dataset, check out the [Bring-Your-Own-Dataset Guide](../datasets/byod.md).

---

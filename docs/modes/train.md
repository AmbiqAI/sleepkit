# :material-chart-ppf: Model Training

## Introduction 

Each task provides a mode to train a model on the specified features. The training mode can be invoked either via CLI or within `sleepkit` python package. At a high level, the training mode performs the following actions based on the provided configuration parameters:

<div class="annotate" markdown>

1. Load the configuration data (e.g. `configuration.json` (1))
1. Load features (e.g. `FS-W-A-5`)
1. Initialize custom model architecture (e.g. `tcn`)
1. Define the metrics, loss, and optimizer (e.g. `accuracy`, `categorical_crossentropy`, `adam`)
1. Train the model (e.g. `model.fit`)
1. Save artifacts (e.g. `model.keras`)

</div>

1. Example configuration:
--8<-- "assets/usage/json-configuration.md"

<br/>

```mermaid
graph LR
A("`Load
configuration
__TaskParams__
`")
B("`Load
features
__FeatureFactory__
`")
C("`Initialize
model
__ModelFactory__
`")
D("`Define
_metrics_, _loss_,
_optimizer_
`")
E("`Train
__model__
`")
F("`Save
__artifacts__
`")
A ==> B
B ==> C
subgraph "Model Training"
    C ==> D
    D ==> E
end
E ==> F
```

---

## Usage

### CLI

The following command will train a sleep stage model using the reference configuration.

```bash
sleepkit --task stage --mode train --config ./configuration.json
```

### Python

The model can be trained using the following snippet:

```py linenums="1"

from pathlib import Path
import sleepkit as sk

task = sk.TaskFactory.get("detect")

params = sk.TaskParams(...)  # (1)

task.train(params)
```

1. Example configuration:
--8<-- "assets/usage/python-configuration.md"

---

## Arguments 

Please refer to [TaskParams](../modes/configuration.md#taskparams) for the list of arguments that can be used with the `train` command.

---


## Logging

__sleepKIT__ provides built-in support for logging to several third-party services including [Weights & Biases](https://wandb.ai/site) (WANDB) and [TensorBoard](https://www.tensorflow.org/tensorboard).

### WANDB

The training mode is able to log all metrics and artifacts (aka models) to [Weights & Biases](https://wandb.ai/site) (WANDB). To enable WANDB logging, simply set environment variable `WANDB=1`. Remember to sign in prior to running experiments by running `wandb login`.


### TensorBoard

The training mode is able to log all metrics to [TensorBoard](https://www.tensorflow.org/tensorboard). To enable TensorBoard logging, simply set environment variable `TENSORBOARD=1`.

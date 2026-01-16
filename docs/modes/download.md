# :material-download: Download Datasets

The `download` command is used to download all datasets specified. Please refer to [Datasets](../datasets/index.md) for details on the available datasets. Additional datasets can be added by creating a new dataset class and registering it with __sleepKIT__ dataset factory.

## Usage

### CLI

Using the CLI, the `download` command can be used to download specified datasets in the configuration file or directly in the command line.

```bash
sleepkit -m download -c '{"datasets": [{"name": "cmidss", "parameters": {"path": ".datatasets/cmidss"}}]}'
```

### Python

In code, the `download` method of a dataset can be used to download the dataset.

```py linenums="1"
import sleepkit as sk

ds = sk.DatasetFactory.get("cmidss")(path="./datasets/cmidss")
ds.download()

```

## Arguments 

Please refer to [TaskParams](../modes/configuration.md#taskparams) for the list of arguments that can be used with the `download` command.

---

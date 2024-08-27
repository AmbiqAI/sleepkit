# Configuration Parameters

For each mode, common configuration parameters, [TaskParams](#taskparams), are required to run the task. These parameters are used to define the task, datasets, model, and other settings. Rather than defining separate configuration files for each mode, a single configuration object is used to simplify configuration files and heavy re-use of parameters between modes.

## <span class="sk-h2-span">QuantizationParams</span>

Quantization parameters define the quantization-aware training (QAT) and post-training quantization (PTQ) settings. This is used for modes: train, evaluate, export, and demo.

| Argument | Type | Opt/Req | Default | Description |
| --- | --- | --- | --- | --- |
| enabled | bool | Optional | False | Enable quantization |
| qat | bool | Optional | False | Enable quantization aware training (QAT) |
| format | Literal["int8", "int16", "float16"] | Optional | int8 | Quantization mode |
| io_type | str | Optional | int8 | I/O type |
| conversion | Literal["keras", "tflite"] | Optional | keras | Conversion method |
| debug | bool | Optional | False | Debug quantization |
| fallback | bool | Optional | False | Fallback to float32 |

## <span class="sk-h2-span">NamedParams</span>

Named parameters are used to provide custom parameters for a given object or callable where parameter types are not known ahead of time. For example, a dataset, 'CustomDataset', may require custom parameters such as 'path', 'label', 'sampling_rate', etc. When a task loads the dataset using `name`, the task will then unpack the custom parameters and pass them to the dataset initializer.

| Argument | Type | Opt/Req | Default | Description |
| --- | --- | --- | --- | --- |
| name | str | Required |  | Named parameters name |
| params | dict[str, Any] | Optional | {} | Named parameters |

```py linenums="1"

import sleepkit as sk

class CustomDataset(sk.Dataset):

    def __init__(self, a: int = 1, b: int = 2) -> None:
        self.a = a
        self.b = b

sk.DatasetFactory.register("custom", CustomDataset)

params = sk.TaskParams(
    datasets=[
        sk.NamedParams(
            name="custom",
            params=dict(a=1, b=2)
        )
    ]
)

```

## <span class="sk-h2-span">FeatureParams</span>

| Argument | Type | Opt/Req | Default | Description |
| --- | --- | --- | --- | --- |
| name | str | Required | feature | Feature set name |
| sampling_rate | float | Optional | 250 | Target sampling rate (Hz) |
| frame_size | int | Optional | 1250 | Frame size in samples |
| loader | str | Optional | hdf5 | Data loader |
| feat_key | str | Optional | features | Feature key |
| label_key | str | Optional | labels | Label key |
| mask_key | str | Optional | mask | Mask key |
| feat_cols | list[str]\|None | Optional | None | Feature columns |
| save_path | Path | Optional | `tempfile.gettempdir` | Save path |
| params | dict[str, Any] | Optional | {} | Feature Parameters |



## <span class="sk-h2-span">TaskParams</span>

These parameters are supplied to a [Task](../tasks/index.md) when running a given mode such as `train`, `evaluate`, `export`, or `demo`. A single configuration object is used to simplify configuration files and heavy re-use of parameters between modes.


| Argument | Type | Opt/Req | Default | Description | Mode |
| --- | --- | --- | --- | --- | --- |
| name | str | Required | experiment | Experiment name | All |
| project | str | Required | sleepkit | Project name | All |
| job_dir | Path | Optional | `tempfile.gettempdir` | Job output directory | All |
| datasets | list[NamedParams] | Optional | [] | Datasets | All |
| dataset_weights | list[float]\|None | Optional | None | Dataset weights | All |
| force_download | bool | Optional | False | Force download dataset- overriding existing files | download |
| feature | FeatureParams | Optional | {} | Feature configuration | feature |
| sampling_rate | float | Optional | 250 | Target sampling rate (Hz) | feature |
| frame_size | int | Optional | 1250 | Frame size in samples | feature |
| samples_per_subject | int\|list[int] | Optional | 1000 | Number train samples per subject | All |
| val_samples_per_subject | int\|list[int] | Optional | 1000 | Number validation samples per subject | All |
| test_samples_per_subject | int\|list[int] | Optional | 1000 | Number test samples per subject | All |
| preprocesses | list[NamedParams] | Optional | [] | Preprocesses | All |
| augmentations | list[NamedParams] | Optional | [] | Augmentations | All |
| num_classes | int | Optional | 1 | # of classes | All |
| class_map | dict[int, int] | Optional | {1: 1} | Class/label mapping | All |
| class_names | list[str]\|None | Optional | None | Class names | All |
| train_subjects | float\|None | Optional | None | # or proportion of subjects for training | All |
| val_subjects | float\|None | Optional | None | # or proportion of subjects for validation | All |
| test_subjects | float\|None | Optional | None | # or proportion of subjects for testing | All |
| val_file | Path\|None | Optional | None | Path to load/store TFDS validation data | All |
| test_file | Path\|None | Optional | None | Path to load/store TFDS test data | All |
| val_size | int\|None | Optional | None | Number of samples for validation | All |
| test_size | int | Optional | 10000 | # samples for testing | All |
| resume | bool | Optional | False | Resume training | train |
| architecture | NamedParams\|None | Optional | None | Custom model architecture | train |
| model_file | Path\|None | Optional | None | Path to load/save model file (.keras) | All |
| use_logits | bool | Optional | True | Use logits output or softmax | All |
| weights_file | Path\|None | Optional | None | Path to a checkpoint weights to load/save | All |
| quantization | QuantizationParams | Optional | {} | Quantization parameters | train, evaluate, export, demo |
| lr_rate | float | Optional | 1e-3 | Learning rate | train |
| lr_cycles | int | Optional | 3 | Number of learning rate cycles | train |
| lr_decay | float | Optional | 0.9 | Learning rate decay | train |
| label_smoothing | float | Optional | 0 | Label smoothing | train |
| batch_size | int | Optional | 32 | Batch size | train |
| buffer_size | int | Optional | 100 | Buffer cache size | train |
| epochs | int | Optional | 50 | Number of epochs | train |
| steps_per_epoch | int | Optional | 10 | Number of steps per epoch | train |
| val_steps_per_epoch | int | Optional | 10 | Number of validation steps | train |
| val_metric | Literal["loss", "acc", "f1"] | Optional | loss | Performance metric | train |
| class_weights | Literal["balanced", "fixed"] | Optional | fixed | Class weights | train |
| threshold | float\|None | Optional | None | Model output threshold | evaluate |
| test_metric | Literal["loss", "acc", "f1"] | Optional | acc | Test metric | evaluate |
| test_metric_threshold | float\|None | Optional | 0.98 | Validation metric threshold | evaluate |
| tflm_var_name | str | Optional | g_model | TFLite Micro C variable name | export |
| tflm_file | Path\|None | Optional | None | Path to copy TFLM header file (e.g. ./model_buffer.h) | export |
| backend | str | Optional | pc | Backend | demo |
| demo_size | int\|None | Optional | 1000 | # samples for demo | demo |
| display_report | bool | Optional | True | Display report | demo |
| seed | int\|None | Optional | None | Random state seed | All |
| num_workers | int | Optional | os.cpu_count() or 1 | Number of workers for parallel processing | All |
| verbose | int | Optional | 1 | Verbosity level | All |

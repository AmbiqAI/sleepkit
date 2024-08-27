# Train a Wrist-based Sleep Detection Model

In this example, we will train a small TCN network to detect sleep and wake stages using accelerometer data collected from the wrist.

## 0. Import Libraries

First, we will import the necessary libraries and define shared variables.

```py linenums="1"
import os
from pathlib import Path
import sleepkit as sk

task = 'detect'
experiment_name = 'sleep-detect-example'
results_path = Path('../results')
ds_path = Path('../datasets')
ds_name = 'cmidss'
fset = 'FS-W-A-5'
fset_name = 'fs-w-a-5-30'
fs_path = ds_path / 'store' / fset_name
```


## 1. Download dataset

Next, we will download the datasets. In this case, we will be using the [CMIDSS](../datasets/cmidss.md) dataset.

```py linenums="1"
ds = sk.DatasetFactory.create(
    ds_name,
    ds_path=ds_path,
    frame_size=1
)
ds.download(
    num_workers=os.cpu_count()
)
```

## 2. Create feature set

From the dataset, let's create a feature set using the [FS-W-A-5 features](../features/fs_w_a_5.md). This feature set computes 5 features over 60-second windows captured from the accelerometer sensor collected on the wrist. The [CMIDSS](../datasets/cmidss.md) dataset already provides accelerometer averaged over 5 secods (i.e. Fs=0.2 Hz). Therefore, we will use a frame size of 12 to capture 60 seconds of data (i.e. 6 samples at 0.2 Hz) with a 50% overlap.

```py linenums="1"
sk.generate_feature_set(args=sk.FeatureParams(
    job_dir=results_path / fset_name,
    ds_path=ds_path,
    datasets=[sk.DatasetParams(name=ds_name, params={})],
    feature_set=fset,
    feature_params={},
    save_path=fs_path,
    sampling_rate=0.2,
    frame_size=6
))
```

__Output:__

```bash
[04/10/24 14:40:22] INFO     #STARTED MODE=feature                                                                                               cli.py:61
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 277/277 [03:44<00:00,  1.23it/s]
[04/10/24 14:44:07] INFO     #FINISHED MODE=feature
```

Once the command finishes, the feature set will be saved in the `fs_path` directory. These features will be stored in HDF5 files with one file per subject. Each HDF5 file will include the following entries:

- `/features`: Time x Feature tensor (fp32). Features are computed over windows of sensor data.
- `/mask`: Time x Mask tensor (bool). Mask indicates valid feature values.
- `/detect_labels`: Time x Label (int). Labels are awake/sleep.


## 3. Define model architecture

Next, we will define the model architecture. In this case, we will use a TCN model with the following configuration:

```py linenums="1"
architecture = sk.ModelArchitecture(
    name='tcn',
    params=dict(
        input_kernel=[1, 5],
        input_norm='batch',
        blocks=[
            dict(depth=1, branch=1, filters=16, kernel=[1, 5], dilation=[1, 1], dropout=0.10, ex_ratio=1, se_ratio=4, norm='batch'),
            dict(depth=1, branch=1, filters=32, kernel=[1, 5], dilation=[1, 2], dropout=0.10, ex_ratio=1, se_ratio=4, norm='batch'),
            dict(depth=1, branch=1, filters=48, kernel=[1, 5], dilation=[1, 4], dropout=0.10, ex_ratio=1, se_ratio=4, norm='batch'),
            dict(depth=1, branch=1, filters=64, kernel=[1, 5], dilation=[1, 8], dropout=0.10, ex_ratio=1, se_ratio=4, norm='batch')
        ],
        output_kernel=[1, 5],
        include_top=True,
        use_logits=True,
        model_name='tcn'
    )
)
```

## 4. Train model

At this point, we can train the model and generated feature set for the sleep detect task. Since we are performing sleep detection, we will use the TaskFactory to get the `detect` task handler to train the model. Since the feature set we generated are stored in HDF5 files, we will use the `hdf5` dataset handler. The model will be trained for 200 epochs with a batch size of 128 and a learning rate of 1e-3. The model will be fed a `frame_size` of 480 samples which equates to 240 minutes.

```py linenums="1"
# 3. Train model
task_handler = sk.TaskFactory.get(task)
task_handler.train(args=sk.TaskParams(
    name=experiment_name,
    job_dir=results_path / experiment_name,
    ds_path=fs_path,
    dataset=sk.DatasetParams(
        name='hdf5',
        params=dict(
            feat_key='features',
            label_key='detect_labels',
            mask_key='mask'
        )
    ),
    sampling_rate=1/60,
    frame_size=480,
    num_classes=2,
    class_map={
        '0': 0,
        '1': 1,
        '2': 1,
        '3': 1,
        '4': 1,
        '5': 1
    },
    class_names=['WAKE', 'SLEEP'],
    samples_per_subject=100,
    val_samples_per_subject=100,
    val_subjects=0.20,
    val_size=4000,
    batch_size=128,
    buffer_size=10000,
    epochs=200,
    steps_per_epoch=25,
    val_metric='loss',
    lr_rate=1e-3,
    lr_cycles=1,
    label_smoothing=0,
    architecture=architecture
))
```

__Output:__

```bash
Loading training dataset
Loading validation dataset
Building model
Creating model from scratch
Model: "tcn"
__________________________________________________________________________________________________
 Layer (type)                Output Shape                 Param #   Connected to
==================================================================================================
 input (InputLayer)          [(None, 240, 5)]             0         []

...

==================================================================================================
Total params: 9364 (36.58 KB)
Trainable params: 8832 (34.50 KB)
Non-trainable params: 532 (2.08 KB)
__________________________________________________________________________________________________
Model requires 3.11 MFLOPS

...

Model saved to results/sleep-detect-2/model.keras
Performing full validation
[TEST SET] ACC=94.15%, F1=94.26% IoU=89.31%

```

## 5. Evaluate model

Now that the model has been trained, we can evaluate its performance on the test set. We will use the same feature set and dataset configuration as before.

```py linenums="1"
task_handler.evaluate(args=sk.TaskParams(
    name=experiment_name,
    job_dir=results_path / experiment_name,
    ds_path=fs_path,
    dataset=sk.DatasetParams(
        name='hdf5',
        params=dict(
            feat_key='features',
            label_key='detect_labels',
            mask_key='mask'
        )
    ),
    sampling_rate=1/60,
    frame_size=480,
    num_classes=2,
    class_map={
        '0': 0,
        '1': 1,
        '2': 1,
        '3': 1,
        '4': 1,
        '5': 1
    },
    class_names=['WAKE', 'SLEEP'],
    test_samples_per_subject=20,
    test_size=1000,
    model_file='model.keras'
))
```

__Output:__

```bash
Loading model
Model: "tcn"
__________________________________________________________________________________________________
 Layer (type)                Output Shape                 Param #   Connected to
==================================================================================================
 input (InputLayer)          [(None, 240, 5)]             0         []

...

Model requires 3.11 MFLOPS
Performing full inference
Testing Results
[TEST SET] ACC=95.94%, F1=95.96%, AP=94.49%, IoU=92.35%
```


## 6. Export model

Once we achieve acceptable performance on the test set, we can export the model to a format that can be used for deployment. In this case, we will export the model to a TensorFlow Lite format with quantization enabled.

```py linenums="1"
task_handler.export(args=sk.TaskParams(
    name=experiment_name,
    job_dir=results_path / experiment_name,
    ds_path=fs_path,
    dataset=sk.DatasetParams(
        name='hdf5',
        params=dict(
            feat_key='features',
            label_key='detect_labels',
            mask_key='mask'
        )
    ),
    sampling_rate=1/60,
    frame_size=480,
    num_classes=2,
    class_map={
        '0': 0,
        '1': 1,
        '2': 1,
        '3': 1,
        '4': 1,
        '5': 1
    },
    class_names=['WAKE', 'SLEEP'],
    test_samples_per_subject=20,
    test_size=1000,
    model_file='model.keras',
    val_acc_threshold=0.98,
    quantization=sk.QuantizationParams(
        enabled=True,
        qat=False,
        ptq=True,
        input_type='int8',
        output_type='int8'
    ),
))
```

__Output:__

```bash
Loading trained model
Model: "tcn"
__________________________________________________________________________________________________
 Layer (type)                Output Shape                 Param #   Connected to
==================================================================================================

...

Converting model to TFLite (quantization=True)
Saving TFLite model to results/sleep-detect-2/model.tflite
Saving TFL micro model to results/sleep-detect-2/model_buffer.h
Validating model results
[TF SET] ACC=95.34%, F1=95.34%
[TFL SET] ACC=94.64%, F1=94.62%
Validation passed (0.70%)
```

## 7. Run demo

Finally, we can a full demo of the model using the `demo` command. This will run the model on a sample subject and generate a report with performance metrics. The current configuration will run inference on the PC. By changing backend to 'evb', the model can be run on the Apollo4 Plus EVB.

```py linenums="1"
task_handler.demo(args=sk.TaskParams(
    name=experiment_name,
    job_dir=results_path / experiment_name,
    ds_path=fs_path,
    dataset=sk.DatasetParams(
        name='hdf5',
        params=dict(
            feat_key='features',
            label_key='detect_labels',
            mask_key='mask'
        )
    ),
    sampling_rate=1/60,
    frame_size=480,
    num_classes=2,
    class_map={
        '0': 0,
        '1': 1,
        '2': 1,
        '3': 1,
        '4': 1,
        '5': 1
    },
    class_names=['WAKE', 'SLEEP'],
    model_file='model.keras',
    backend='pc'
))
```

__Output:__

<div class="sk-plotly-graph-div">
--8<-- "assets/zoo/detect/sleep-detect-demo.html"
</div>

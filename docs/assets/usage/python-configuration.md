```py linenums="1"

sk.TaskParams(
    name="sd-2-tcn-sm",
    job_dir="./results/sd-2-tcn-sm",
    verbose=2,

    datasets=[
        hk.NamedParams(
            name="cmidss",
            params={
                "path": "./datasets/cmidss"
            }
        )
    ],

    feature=hk.FeatureParams(
        name="FS-W-A-5",
        sampling_rate=0.2,
        frame_size=12,
        loader="hdf5",
        feat_key="features",
        label_key="detect_labels",
        mask_key="mask",
        feat_cols=None,
        save_path="./datasets/store/fs-w-a-5-60",
        params={}
    ),

    sampling_rate=0.0083333,
    frame_size=240,

    num_classes=2,
    class_map={
        0: 0,
        1: 1,
        2: 1,
        3: 1,
        4: 1,
        5: 1
    },
    class_names=["WAKE", "SLEEP"],

    samples_per_subject=100,
    val_samples_per_subject=100,
    test_samples_per_subject=50,

    val_size=4000,
    test_size=2500,

    val_subjects=0.20,
    batch_size=128,
    buffer_size=10000,
    epochs=200,
    steps_per_epoch=25,
    val_steps_per_epoch=25,
    val_metric="loss",
    lr_rate=1e-3,
    lr_cycles=1,
    label_smoothing=0,

    test_metric="f1",
    test_metric_threshold=0.02,
    tflm_var_name="sk_detect_flatbuffer",
    tflm_file="sk_detect_flatbuffer.h",

    backend="pc",
    display_report=True,

    quantization=hk.QuantizationParams(
        qat=False,
        mode="INT8",
        io_type="int8",
        concrete=True,
        debug=False
    ),

    model_file="model.keras",
    use_logits=False,
    architecture=hk.NamedParams(
        name="tcn",
        params={
            "input_kernel": [1, 5],
            "input_norm": "batch",
            "blocks": [
                {"depth": 1, "branch": 1, "filters": 16, "kernel": [1, 5], "dilation": [1, 1], "dropout": 0.10, "ex_ratio": 1, "se_ratio": 4, "norm": "batch"},
                {"depth": 1, "branch": 1, "filters": 32, "kernel": [1, 5], "dilation": [1, 2], "dropout": 0.10, "ex_ratio": 1, "se_ratio": 4, "norm": "batch"},
                {"depth": 1, "branch": 1, "filters": 48, "kernel": [1, 5], "dilation": [1, 4], "dropout": 0.10, "ex_ratio": 1, "se_ratio": 4, "norm": "batch"},
                {"depth": 1, "branch": 1, "filters": 64, "kernel": [1, 5], "dilation": [1, 8], "dropout": 0.10, "ex_ratio": 1, "se_ratio": 4, "norm": "batch"}
            ],
            "output_kernel": [1, 5],
            "include_top": True,
            "use_logits": True,
            "model_name": "tcn"
        }
    )
```

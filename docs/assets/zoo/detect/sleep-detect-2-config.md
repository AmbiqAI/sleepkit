```json
{
    "name": "sleep-detect-2",
    "job_dir": "./results/sleep-detect-2",
    "ds_path": "./datasets/store/fs-w-a-5-60/cmidss",
    "dataset": {
        "name": "hdf5",
        "params": {
            "feat_key": "features",
            "label_key": "detect_labels",
            "mask_key": "mask"
        }
    },
    "sampling_rate": 0.01666666,
    "frame_size": 240,
    "model_file": "model.keras",
    "num_classes": 2,
    "class_map": {
        "0": 0,
        "1": 1,
        "2": 1,
        "3": 1,
        "4": 1,
        "5": 1
    },
    "class_names": ["WAKE", "SLEEP"],
    "samples_per_subject": 100,
    "val_samples_per_subject": 100,
    "val_subjects": 0.20,
    "val_size": 4000,
    "val_acc_threshold": 0.98,
    "test_size": 1000,
    "test_samples_per_subject": 20,
    "batch_size": 128,
    "buffer_size": 10000,
    "epochs": 200,
    "steps_per_epoch": 25,
    "val_metric": "loss",
    "lr_rate": 1e-3,
    "lr_cycles": 1,
    "label_smoothing": 0,
    "backend": "pc",
    "quantization": {
        "enabled": true,
        "qat": false,
        "ptq": true,
        "input_type": "int8",
        "output_type": "int8",
        "supported_ops": null
    },
    "architecture": {
        "name": "tcn",
        "params": {
            "input_kernel": [1, 5],
            "input_norm": "batch",
            "blocks": [
                {"depth": 1, "branch": 1, "filters": 16, "kernel": [1, 5], "dilation": [1, 1], "dropout": 0.10, "ex_ratio": 1, "se_ratio": 4, "norm": "batch"},
                {"depth": 1, "branch": 1, "filters": 32, "kernel": [1, 5], "dilation": [1, 2], "dropout": 0.10, "ex_ratio": 1, "se_ratio": 4, "norm": "batch"},
                {"depth": 1, "branch": 1, "filters": 48, "kernel": [1, 5], "dilation": [1, 4], "dropout": 0.10, "ex_ratio": 1, "se_ratio": 4, "norm": "batch"},
                {"depth": 1, "branch": 1, "filters": 64, "kernel": [1, 5], "dilation": [1, 8], "dropout": 0.10, "ex_ratio": 1, "se_ratio": 4, "norm": "batch"}
            ],
            "output_kernel": [1, 5],
            "include_top": true,
            "use_logits": true,
            "model_name": "tcn"
        }
    }
}
```

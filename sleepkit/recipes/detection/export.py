"""Recipe-owned conversion and packaging, using the generic artifact contract."""

from importlib.metadata import version
from pathlib import Path

import numpy as np

from sleepkit.artifacts import Artifact, Check, TensorSpec, stage_bundle
from sleepkit.artifacts.package import sha256, write_json
from sleepkit.artifacts.runtime import create_reference
from .output_contract import output_contract


def export_bundle(model, normalizer, destination, metadata, metrics):
    import keras
    import tensorflow as tf
    from ai_edge_litert.interpreter import Interpreter

    recipe_kind, class_names = output_contract(metadata.get("target"))
    destination = Path(destination)
    workspace = destination.parent
    context = model.input_shape[1]
    keras_path = workspace / "model.keras"
    model.save(keras_path)
    restored = keras.models.load_model(keras_path, compile=False, safe_mode=True)
    probe = np.random.default_rng(42).uniform(-0.5, 0.5, (1, context, 5)).astype(np.float32)
    expected = np.asarray(model(probe, training=False))
    np.testing.assert_allclose(restored(probe, training=False), expected, atol=1e-6, rtol=1e-5)
    model.export(
        workspace / "saved_model",
        format="tf_saved_model",
        verbose=False,
        input_signature=[tf.TensorSpec((1, context, 5), tf.float32, name="features")],
    )
    converter = tf.lite.TFLiteConverter.from_saved_model(str(workspace / "saved_model"))
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    lite_path = workspace / "model.tflite"
    lite_path.write_bytes(converter.convert())
    runner = Interpreter(model_path=str(lite_path))
    runner.allocate_tensors()
    inputs, outputs = runner.get_input_details(), runner.get_output_details()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("Detection export requires one input and one output")
    i, o = inputs[0], outputs[0]
    if (
        tuple(i["shape"]) != (1, context, 5)
        or tuple(o["shape"]) != (1, context, 2)
        or i["dtype"] != np.float32
        or o["dtype"] != np.float32
    ):
        raise ValueError("Unexpected detection export signature")
    runner.set_tensor(i["index"], probe)
    runner.invoke()
    actual = runner.get_tensor(o["index"])
    if not np.isfinite(actual).all():
        raise ValueError("Nonfinite exported output")
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
    artifact = Artifact(
        lite_path,
        "model.tflite",
        "model",
        "tflite",
        "This detection recipe run",
        (
            TensorSpec(
                i["name"],
                tuple(int(d) for d in i["shape_signature"]),
                "float32",
                "Training-statistic normalized wrist features in preprocessing.json order",
            ),
        ),
        (
            TensorSpec(
                o["name"],
                tuple(int(d) for d in o["shape_signature"]),
                "float32",
                f"Per-epoch logits ordered {', '.join(class_names)}; softmax once for probabilities",
            ),
        ),
    )
    runtime_check = create_reference(artifact, workspace / "reference.npz")
    write_json(workspace / "preprocessing.json", normalizer.to_dict())
    write_json(
        workspace / "recipe.json",
        {
            **metadata,
            "context": context,
            "recipe": recipe_kind,
            "class_names": class_names,
            "output": "logits",
            "versions": {name: version(name) for name in ("tensorflow", "keras", "numpy", "ai-edge-litert")},
        },
    )
    write_json(workspace / "metrics.json", metrics)
    files = [
        ("model.keras", "training_checkpoint", "keras"),
        ("reference.npz", "reference", "npz"),
        ("preprocessing.json", "preprocessing_state", "json"),
        ("recipe.json", "recipe_metadata", "json"),
        ("metrics.json", "evaluation", "json"),
    ]
    artifacts = [artifact] + [
        Artifact(workspace / name, name, role, fmt, "This detection recipe run") for name, role, fmt in files
    ]
    hashes = {a.name: sha256(a.source) for a in artifacts}
    checks = [
        runtime_check,
        Check(
            "keras_reload",
            "passed",
            "Safe Keras reload agrees on a deterministic synthetic probe.",
            {name: hashes[name] for name in ("model.keras", "preprocessing.json")},
            {"seed": 42, "atol": 1e-6, "rtol": 1e-5},
        ),
        Check(
            "conversion_parity",
            "passed",
            "Float32 TFLite and Keras logits agree on a synthetic probe; not task-quality equivalence.",
            {name: hashes[name] for name in ("model.keras", "model.tflite")},
            {"seed": 42, "atol": 1e-5, "rtol": 1e-4},
        ),
        Check(
            "held_out_evaluation",
            "passed",
            "Keras evaluated on the explicit test subjects; no quality threshold asserted.",
            {name: hashes[name] for name in ("model.keras", "metrics.json", "recipe.json", "preprocessing.json")},
        ),
        Check("task_quality", "not_run", "No acceptance threshold or historical comparison has been established."),
        Check("target_hardware", "not_run", "Float32 CPU export only; no quantization or MCU validation."),
    ]
    return stage_bundle(
        destination,
        title="Wrist detection recipe experiment",
        artifacts=artifacts,
        checks=checks,
        metadata={"sleepkit": {"recipe": recipe_kind}},
        card=_card(metadata.get("target")),
    )


def _card(target=None):
    card = """---
tags:
- sleepkit
- tflite
- time-series
---
# Wrist detection recipe experiment

This is a newly trained experiment, not a re-export or reproduction of SD-2-TCN-SM.
Input sensor channels are CMIDSS TS (seconds of day), ENMO, and ZANGLE at 0.2 Hz.
Feature generation, invalid-sample policy, order, and fitted training-only normalization
are defined in `preprocessing.json`; the matching implementation is
`sleepkit.recipes.detection.preprocessing`. No code is loaded from this bundle.
Feature windows cover 60 seconds and advance 30 seconds. Predictions are timestamped
at the last source sample (55 seconds for the first window). Model contexts are
nonoverlapping; invalid contexts and incomplete tails are omitted. The convolutional
model uses future features within its context, so this is not causal streaming inference.

`model.tflite` accepts normalized float32 [1, context, 5] features and emits
float32 [1, context, 2] logits ordered WAKE, SLEEP. Apply softmax once for probabilities.
`model.keras` is the trainable checkpoint. `recipe.json` records context, versions,
code hashes, and split/source fingerprints. Local run files retain exact subject splits;
subject identifiers and source samples are not included here. `reference.npz` is synthetic.

`metrics.json` reports held-out Keras performance for this run, including confusion
matrix, evaluated feature frames, and macro F1. It is not historical baseline performance.
The recipe evaluates provided HDF5 labels as-is: legacy CMIDSS conversion treats
unannotated periods as wake, so annotation quality must be reviewed before any
scientific comparison. Synthetic-data runs are identified in recipe metadata.
No accuracy threshold, quantized export, or target-hardware support is claimed.
See `validation.json` for separate conformance, parity, and evaluation checks.

Install this recipe's implementation and `numpy`, `h5py`, `ai-edge-litert`, then use:

```python
from sleepkit.recipes.detection.inference import predict
result = predict("bundle-directory", sensor_data)  # shape [3, samples], no labels needed
# For v3 inputs with an independent clock, pass sample_time=utc_seconds.
# result includes target times, context availability times, logits, and probabilities
```

Model artifact licensing is not specified. Choose an artifact license before publication.
"""

    if target is not None:
        _, names = output_contract(target)
        card = card.replace("logits ordered WAKE, SLEEP", "logits ordered " + ", ".join(names))
        card = card.replace(
            "The recipe evaluates provided HDF5 labels as-is: legacy CMIDSS conversion treats\nunannotated periods as wake, so annotation quality must be reviewed before any\nscientific comparison.",
            "The target is annotated nightly-period membership: inside a supported annotation or\noutside between eligible consecutive nights. It does not establish clinical sleep/wake,\nconfirmed device wear, or absence of naps. Unknown targets are excluded. Frozen source,\nevent, split, and policy hashes bind the dataset; historical labels are not used.\nThe exact scoring index and predictions stay local; their hashes are in recipe.json.",
        )
    return card

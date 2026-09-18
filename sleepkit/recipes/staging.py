"""A copyable raw-record -> preparation -> Keras -> deployment experiment.

This small synthetic two-stage experiment demonstrates the v1 API. It is not a
retraining or reproduction of the historical MESA TCN baselines.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from sleepkit.data import Annotation, Record, Signal, split_subjects
from sleepkit.evaluation import classification_metrics
from sleepkit.preprocessing import Channel, Standardizer, WindowFeatures, model_windows, prepare


def synthetic_records(subjects=10, seed=7):
    """Multi-rate synthetic observations with known targets; no patient data."""
    rng = np.random.default_rng(seed)
    records = []
    for subject in range(subjects):
        labels = rng.integers(0, 2, 32)
        annotations = tuple(Annotation(i * 30, (i + 1) * 30, int(label)) for i, label in enumerate(labels))
        # Deliberately simple distributions; these scores are plumbing checks only.
        movement = np.repeat(1 - labels, 30 * 4) + rng.normal(0, 0.1, 32 * 30 * 4)
        spo2 = 96 + np.repeat(labels, 30) + rng.normal(0, 0.1, 32 * 30)
        records.append(
            Record(
                "synthetic",
                str(subject),
                f"night-{subject}",
                {
                    "movement": Signal(movement, 4, "g", "accelerometer", "wrist"),
                    "spo2": Signal(spo2, 1, "%", "oximetry", "finger"),
                },
                annotations,
                "synthetic-v1",
            )
        )
    return records


def build_model(context, features, classes):
    """An ordinary causal Keras model; replace this function freely."""
    import keras

    inputs = keras.Input((context, features))
    x = keras.layers.Conv1D(8, 3, padding="causal", activation="relu")(inputs)
    x = keras.layers.Conv1D(8, 3, padding="causal", dilation_rate=2, activation="relu")(x)
    outputs = keras.layers.Dense(classes)(x)
    return keras.Model(inputs, outputs)


def collect(records, preprocessing, normalizer, context, cache_dir):
    batches = [
        model_windows(normalizer.transform(prepare(record, preprocessing, cache_dir)), context) for record in records
    ]
    x = np.concatenate([batch[0] for batch in batches])
    y = np.concatenate([batch[1] for batch in batches])
    if not len(x):
        raise ValueError("No complete valid labeled model windows")
    return x, y


def run(
    output,
    *,
    epochs=2,
    records=None,
    preprocessing=None,
    model_builder=build_model,
    class_names=("wake", "sleep"),
    seed=7,
    synthetic=False,
):
    """Train a supplied model on records using explicit, reusable preparation.

    Replace records/preprocessing/model_builder for real experiments. Only this
    recipe chooses how to assemble the functions; no experiment registration is needed.
    """
    import keras
    import tensorflow as tf
    from sleepkit.export import export_bundle
    from sleepkit.runtime import Predictor

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "deploy").exists() or (output / "split.json").exists():
        raise FileExistsError("Use a new output directory for each experiment")
    keras.utils.set_random_seed(seed)
    if records is None:
        records = synthetic_records(seed=seed)
        synthetic = True
    else:
        records = list(records)
    preprocessing = preprocessing or WindowFeatures(
        (
            Channel("movement", "g", "accelerometer", "wrist"),
            Channel("spo2", "%", "oximetry", "finger"),
        )
    )
    split = split_subjects(records, seed=seed)
    (output / "split.json").write_text(json.dumps(split.to_dict(), indent=2))
    cache = output / "cache"
    normalizer = Standardizer.fit(records, preprocessing, split, cache)
    context = 4
    train_x, train_y = collect(split.select(records, "train"), preprocessing, normalizer, context, cache)
    val_x, val_y = collect(split.select(records, "validation"), preprocessing, normalizer, context, cache)
    test_x, test_y = collect(split.select(records, "test"), preprocessing, normalizer, context, cache)
    for targets in (train_y, val_y, test_y):
        if targets.min() < 0 or targets.max() >= len(class_names):
            raise ValueError("Target labels do not match class names")
    model = model_builder(context, train_x.shape[-1], len(class_names))
    model.compile(
        optimizer=keras.optimizers.Adam(0.01),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    # Standard tf.data is optional recipe glue, not a SleepKit data abstraction.
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    train = (
        tf.data.Dataset.from_tensor_slices((train_x, train_y))
        .shuffle(len(train_x), seed=seed)
        .batch(8)
        .with_options(options)
    )
    validation = tf.data.Dataset.from_tensor_slices((val_x, val_y)).batch(8).with_options(options)
    history = model.fit(train, validation_data=validation, epochs=epochs, verbose=0, shuffle=False)
    (output / "history.json").write_text(json.dumps(history.history, indent=2))
    metrics = classification_metrics(
        test_y, np.asarray(model(test_x, training=False)).argmax(axis=-1), len(class_names)
    )
    metrics["data_kind"] = "synthetic_smoke_only" if synthetic else "user_records"
    bundle = export_bundle(
        model,
        output / "deploy",
        preprocessing=preprocessing,
        normalizer=normalizer,
        class_names=class_names,
        calibration=train_x,
        validation_inputs=val_x,
        validation_targets=val_y,
        context=context,
        metrics=metrics,
        provenance={
            "recipe": "staging-v1",
            "seed": seed,
            "synthetic": synthetic,
            "split_sha256": hashlib.sha256((output / "split.json").read_bytes()).hexdigest(),
        },
    )
    predictor = Predictor(bundle)
    lite_metrics = classification_metrics(test_y, predictor.predict_features(test_x).argmax(axis=-1), len(class_names))
    (output / "test_metrics.json").write_text(json.dumps({"keras": metrics, "litert": lite_metrics}, indent=2))
    # Exercise inference without annotations: the model path must never need labels.
    sample = split.select(records, "test")[0]
    unlabeled = Record(
        sample.dataset, sample.subject, sample.recording, sample.signals, source_revision=sample.source_revision
    )
    logits, times = predictor.predict_record(unlabeled)
    return {
        "bundle": str(bundle),
        "test_metrics": metrics,
        "inference_windows": len(logits),
        "feature_frames": int(times.size),
    }

"""Unlabeled sensor-to-prediction path using only NumPy and the optional LiteRT adapter."""

from pathlib import Path

import numpy as np

from sleepkit.artifacts.package import validate_bundle
from sleepkit.artifacts.schema import Artifact, TensorSpec
from .evaluation import read_bound_json
from .output_contract import validate_output
from .preprocessing import Normalizer, contexts, prepare
from .runtime import DetectionRuntime


def predict(bundle, data, *, sample_time=None, model_name="model.tflite"):

    bundle = Path(bundle)
    report = validate_bundle(bundle, profile="runnable")
    hashes = {entry["path"]: entry["sha256"] for entry in report["manifest"]["artifacts"]}
    recipe = read_bound_json(bundle, "recipe.json", hashes)
    if report["manifest"].get("metadata", {}).get("sleepkit", {}).get("recipe") != recipe.get("recipe"):
        raise ValueError("Not a supported detection recipe bundle")
    class_names = validate_output(recipe)
    normalizer = Normalizer.from_dict(read_bound_json(bundle, "preprocessing.json", hashes))
    entries = [entry for entry in report["manifest"]["artifacts"] if entry["path"] == model_name]
    if len(entries) != 1 or entries[0]["role"] != "model" or entries[0]["format"] != "tflite":
        raise ValueError("Selected model must be a declared TFLite model artifact")
    entry = entries[0]
    artifact = Artifact(bundle / entry["path"], entry["path"], entry["role"], entry["format"], entry["origin"],
                        tuple(TensorSpec(**spec) for spec in entry["inputs"]),
                        tuple(TensorSpec(**spec) for spec in entry["outputs"]),
                        expected_sha256=entry["sha256"])
    context = recipe["context"]
    runner = DetectionRuntime(bundle / model_name, context, artifact=artifact)
    features = normalizer.transform(prepare(data, sample_time=sample_time, spec=normalizer.spec))
    predictions, times, availability = [], [], []
    for values, _, timestamps in contexts(features, context):
        logits, _ = runner.predict(values)
        predictions.append(logits)
        times.append(timestamps)
        availability.append(np.full(len(timestamps), timestamps[-1] + 5))
    if not predictions:
        raise ValueError("No complete valid model contexts in this recording")
    logits = np.concatenate(predictions)
    if not np.isfinite(logits).all():
        raise ValueError("Nonfinite predictions")
    runner.verify_unchanged()
    probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return {
        "class_names": class_names,
        "target": recipe.get("target"),
        "times": np.concatenate(times),
        "available_at": np.concatenate(availability),
        "logits": logits,
        "probabilities": probabilities,
    }

"""Unlabeled sensor-to-prediction path using only NumPy and the optional LiteRT adapter."""

import json
from pathlib import Path

import numpy as np

from sleepkit.artifacts.package import validate_bundle
from .output_contract import validate_output
from .preprocessing import Normalizer, contexts, prepare


def predict(bundle, data, *, sample_time=None):
    from ai_edge_litert.interpreter import Interpreter

    bundle = Path(bundle)
    report = validate_bundle(bundle, profile="runnable")
    recipe = json.loads((bundle / "recipe.json").read_text())
    if report["manifest"].get("metadata", {}).get("sleepkit", {}).get("recipe") != recipe.get("recipe"):
        raise ValueError("Not a supported detection recipe bundle")
    class_names = validate_output(recipe)
    normalizer = Normalizer.from_dict(json.loads((bundle / "preprocessing.json").read_text()))
    runner = Interpreter(model_path=str(bundle / "model.tflite"))
    runner.allocate_tensors()
    i, o = runner.get_input_details()[0], runner.get_output_details()[0]
    context = recipe["context"]
    if (
        tuple(i["shape"]) != (1, context, 5)
        or tuple(o["shape"]) != (1, context, 2)
        or i["dtype"] != np.float32
        or o["dtype"] != np.float32
    ):
        raise ValueError("Incompatible runtime tensor contract")
    features = normalizer.transform(prepare(data, sample_time=sample_time, spec=normalizer.spec))
    predictions, times, availability = [], [], []
    for values, _, timestamps in contexts(features, context):
        runner.set_tensor(i["index"], values[None])
        runner.invoke()
        predictions.append(runner.get_tensor(o["index"])[0])
        times.append(timestamps)
        availability.append(np.full(len(timestamps), timestamps[-1] + 5))
    if not predictions:
        raise ValueError("No complete valid model contexts in this recording")
    logits = np.concatenate(predictions)
    if not np.isfinite(logits).all():
        raise ValueError("Nonfinite predictions")
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

"""Fixed post-training int8 conversion, matched evaluation, and release staging."""

from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import shutil

import numpy as np

from sleepkit.recipes._components import implementation_files

from sleepkit.artifacts import Artifact, Check, TensorSpec, stage_bundle
from sleepkit.artifacts.package import sha256, write_json
from sleepkit.artifacts.runtime import create_reference
from .calibration import collect_calibration
from .evaluation import _metrics, read_bound_json, summarize
from .preprocessing import Normalizer, fingerprint, prepare
from .runtime import DetectionRuntime


POLICY = {
    "kind": "sleepkit.detection_int8/v1",
    "conversion": "Fixed parent Keras checkpoint; default optimization; TFLITE_BUILTINS_INT8 only; int8 input/output",
    "calibration": "Up to two uniformly selected eligible native contexts per training series; seed 0; no replacement; fixed fitted normalizer",
    "evaluation": "All original eligible test contexts; exact source/context/endpoint/target matching; original saved Keras logits and preserved float TFLite",
    "selection": "One declared calibration/conversion attempt; no validation/test-driven changes or retraining",
    "scope": "Quantization error measurement only; no int8 logit-equivalence, task acceptance threshold or target-hardware claim",
}


def inspect_integer_graph(path):
    """Inspect every FlatBuffer subgraph tensor; integer I/O alone is insufficient."""
    from tensorflow.lite.python import schema_py_generated as schema

    names = {value: name for name, value in vars(schema.TensorType).items() if isinstance(value, int)}
    op_names = {value: name for name, value in vars(schema.BuiltinOperator).items() if isinstance(value, int)}
    model = schema.Model.GetRootAsModel(Path(path).read_bytes(), 0)
    counts, operators = {}, set()
    allowed = {schema.TensorType.INT8, schema.TensorType.INT16, schema.TensorType.INT32,
               schema.TensorType.INT64, schema.TensorType.BOOL}
    for s in range(model.SubgraphsLength()):
        graph = model.Subgraphs(s)
        for t in range(graph.TensorsLength()):
            kind = graph.Tensors(t).Type()
            if kind not in allowed:
                raise ValueError("Integer-only conversion contains a noninteger tensor")
            name = names[kind]
            counts[name] = counts.get(name, 0) + 1
        for o in range(graph.OperatorsLength()):
            code = model.OperatorCodes(graph.Operators(o).OpcodeIndex()).BuiltinCode()
            if code == schema.BuiltinOperator.CUSTOM:
                raise ValueError("Integer-only conversion contains a custom operator")
            operators.add(op_names[code])
    if not counts or not operators:
        raise ValueError("Empty integer graph")
    return {"tensor_types": counts, "operators": sorted(operators), "subgraphs": model.SubgraphsLength()}


def _accumulator():
    return {"confusion": np.zeros((2, 2), dtype=np.int64), "loss": 0.0}


def _add(accumulator, targets, logits):
    logits = np.asarray(logits, np.float64)
    np.add.at(accumulator["confusion"], (targets, logits.argmax(axis=1)), 1)
    shifted = logits - logits.max(axis=1, keepdims=True)
    accumulator["loss"] += float((np.log(np.exp(shifted).sum(axis=1))
                                  - shifted[np.arange(len(targets)), targets]).sum())


def _finish(accumulator):
    return _metrics(accumulator["confusion"], accumulator["loss"])


def evaluate_quantized(run, source, normalizer, verified, float_runner, int8_runner, destination, *, cache=None):
    """Replay and match the frozen test index; conversion is already fixed."""
    run, destination = Path(run), Path(destination)
    with np.load(run / "test-predictions.npz", allow_pickle=False) as arrays:
        keras_logits, targets = arrays["logits"], arrays["targets"]
    names = ("model.keras", "model.float.tflite", "model.tflite")
    pooled = {name: _accumulator() for name in names}
    series = {}
    raw_outputs, quantized_logits, float_logits = [], [], []
    drift = {"argmax_disagreements": 0, "correct_to_wrong": 0, "wrong_to_correct": 0,
             "max_abs_logit_difference": 0.0, "sum_abs_logit_difference": 0.0}
    float_parity = {"passed": True, "atol": 1e-5, "rtol": 1e-4,
                    "max_abs_logit_difference": 0.0, "argmax_disagreements": 0}
    counters = {"input_values": 0, "clipped_input_values": 0, "argmax_ties": 0, "output_saturated_values": 0}
    offset = 0
    with (run / "test-index.jsonl").open() as index:
        for subject in source.split["test"]:
            recording = source.read(subject, labels=True)
            features = normalizer.transform(prepare(recording.data, cache, sample_time=recording.sample_time))
            subject_metrics = {name: _accumulator() for name in names}
            evaluated = 0
            for start in range(0, len(features.ends) - source.context + 1, source.context):
                stop = start + source.context
                ends, values = features.ends[start:stop], features.values[start:stop]
                target = recording.labels[ends]
                rows = [json.loads(index.readline()) for _ in range(source.context)]
                eligible = bool(features.valid[start:stop].all() and (target >= 0).all())
                for frame, row in enumerate(rows):
                    if (row["subject"] != subject or row["source_sha256"] != source.source_hashes[subject]
                            or row["context_start_sample"] != int(ends[0]) - 11
                            or row["context_end_sample"] != int(ends[-1])
                            or row["feature_end_sample"] != int(ends[frame]) or row["target"] != int(target[frame])
                            or row["context_eligible"] != eligible
                            or row["sensor_valid"] != bool(features.valid[start + frame])
                            or row["eligible_output_index"] != (offset + frame if eligible else None)):
                        raise ValueError("Quantized evaluation differs from the frozen scoring index")
                if not eligible:
                    continue
                if not np.array_equal(targets[offset:offset + source.context], target):
                    raise ValueError("Quantized evaluation targets differ from saved targets")
                original = keras_logits[offset:offset + source.context]
                floating, _ = float_runner.predict(values)
                quantized, runtime_counts = int8_runner.predict(values)
                float_difference = np.abs(floating.astype(np.float64) - original)
                float_parity["passed"] &= bool(np.allclose(floating, original, atol=1e-5, rtol=1e-4))
                float_parity["max_abs_logit_difference"] = max(float_parity["max_abs_logit_difference"],
                                                               float(float_difference.max()))
                float_parity["argmax_disagreements"] += int((floating.argmax(1) != original.argmax(1)).sum())
                difference = np.abs(quantized.astype(np.float64) - original)
                drift["max_abs_logit_difference"] = max(drift["max_abs_logit_difference"], float(difference.max()))
                drift["sum_abs_logit_difference"] += float(difference.sum())
                old_correct, new_correct = original.argmax(1) == target, quantized.argmax(1) == target
                drift["argmax_disagreements"] += int((original.argmax(1) != quantized.argmax(1)).sum())
                drift["correct_to_wrong"] += int((old_correct & ~new_correct).sum())
                drift["wrong_to_correct"] += int((~old_correct & new_correct).sum())
                for key in counters:
                    counters[key] += int(runtime_counts[key])
                for name, logits in zip(names, (original, floating, quantized)):
                    _add(pooled[name], target, logits)
                    _add(subject_metrics[name], target, logits)
                raw_outputs.append(runtime_counts["raw_output"].copy())
                quantized_logits.append(quantized)
                float_logits.append(floating)
                offset += source.context
                evaluated += source.context
            if evaluated != verified["subjects"][subject]["eligible_outputs"]:
                raise ValueError("Subject evaluation coverage changed")
            series[subject] = {name: _finish(value) for name, value in subject_metrics.items()}
        if index.readline() or offset != len(targets):
            raise ValueError("Evaluation denominator differs from the frozen run")
    if not offset or not float_parity["passed"]:
        raise ValueError("Original float TFLite does not reproduce saved Keras logits")
    if _finish(pooled["model.keras"])["confusion_matrix"] != verified["pooled"]["confusion_matrix"]:
        raise ValueError("Original Keras confusion matrix changed")
    drift["mean_abs_logit_difference"] = drift.pop("sum_abs_logit_difference") / (offset * 2)
    np.savez_compressed(destination / "test-predictions.npz", targets=targets, int8_outputs=np.concatenate(raw_outputs),
                        int8_logits=np.concatenate(quantized_logits), float_logits=np.concatenate(float_logits))
    means = {}
    for name in names:
        values = [s[name] for s in series.values() if s[name] is not None]
        means[name] = {"contributing_series": len(values), **{
            key: float(np.mean([m[key] for m in values])) for key in ("accuracy", "macro_f1")}}
    return {"target": verified["target"], "class_order": verified["class_order"], "split": "test",
            "coverage": verified["coverage"], "models": {name: _finish(value) for name, value in pooled.items()},
            "unweighted_eligible_series_mean": means, "float_parity": float_parity, "int8_drift": drift,
            "int8_runtime": counters, "subjects": series,
            "predictions_sha256": sha256(destination / "test-predictions.npz")}


def _artifact(path, name, runner, class_names):
    def tensor(detail, meaning):
        quant = detail["quantization_parameters"]
        scale, zero = (float(quant["scales"][0]), int(quant["zero_points"][0])) if len(quant["scales"]) else (None, None)
        return TensorSpec(detail["name"], tuple(None if int(d) == -1 else int(d) for d in detail["shape_signature"]),
                          np.dtype(detail["dtype"]).name, meaning, scale, zero)
    return Artifact(path, name, "model", "tflite", "Fixed post-training detection quantization experiment",
                    (tensor(runner.input_details, "Training-statistic normalized wrist features; quantize for int8 I/O"),),
                    (tensor(runner.output_details, "Logits ordered " + ", ".join(class_names)
                            + "; dequantize int8 then softmax once for probabilities"),), runner.sha256)


def verified_calibration(samples, metadata, directory):
    """Use the exact persisted calibration array whose hash is declared publicly."""
    directory = Path(directory)
    for key, name in (("array_sha256", "calibration.npz"), ("selected_index_sha256", "calibration-index.jsonl"),
                      ("local_report_sha256", "selection.json")):
        if metadata.get(key) != sha256(directory / name):
            raise ValueError("Calibration metadata differs from persisted evidence")
    with np.load(directory / "calibration.npz", allow_pickle=False) as arrays:
        if set(arrays.files) != {"contexts"}:
            raise ValueError("Unexpected persisted calibration arrays")
        stored = arrays["contexts"]
    if (stored.dtype != np.float32 or stored.ndim != 3 or len(stored) < 1
            or stored.shape[-1] != 5 or not np.isfinite(stored).all()
            or samples.dtype != stored.dtype or samples.shape != stored.shape
            or not np.array_equal(samples, stored)):
        raise ValueError("Calibration inputs differ from persisted contexts")
    return stored


def quantize_run(run_path, source, output_path, *, cache=None):
    """One fixed train-only calibration, conversion and evaluation; no tuning loop."""
    import keras
    import tensorflow as tf

    run, output = Path(run_path), Path(output_path)
    if output.resolve().is_relative_to(run.resolve()):
        raise ValueError("Quantization output must be outside the source run")
    source.verify_unchanged()
    output.mkdir(parents=True, exist_ok=False)
    verified = summarize(run, output / "parent-evaluation.json")
    recipe = read_bound_json(run, "bundle/recipe.json", verified["evidence_sha256"])
    if (source.provenance != recipe.get("dataset") or source.context != recipe.get("context")
            or fingerprint(source.split) != recipe.get("split_sha256")
            or fingerprint(source.source_hashes) != recipe.get("source_sha256")):
        raise ValueError("Calibration source differs from the frozen evaluated run")
    normalizer = Normalizer.from_dict(read_bound_json(run, "bundle/preprocessing.json", verified["evidence_sha256"]))
    code = {name: sha256(path) for name, path in implementation_files(Path(__file__).parent).items()}
    declaration = {"declared_utc": datetime.now(timezone.utc).isoformat(), "policy": POLICY,
                   "parent_evidence_sha256": verified["evidence_sha256"], "source": source.provenance,
                   "implementation_sha256": code,
                   "versions": {name: version(name) for name in ("tensorflow", "keras", "numpy", "ai-edge-litert")}}
    write_json(output / "declaration.json", declaration)
    samples, calibration_metadata = collect_calibration(source, normalizer, output / "calibration", cache=cache,
                                                        seed=0, contexts_per_subject=2)
    samples = verified_calibration(samples, calibration_metadata, output / "calibration")
    write_json(output / "calibration.json", calibration_metadata)
    calibration_hashes = {p: sha256(p) for p in (output / "calibration").iterdir()}
    calibration_hashes[output / "calibration.json"] = sha256(output / "calibration.json")
    model = keras.models.load_model(run / "bundle/model.keras", compile=False, safe_mode=True)
    if tuple(model.input_shape[1:]) != (source.context, 5) or tuple(model.output_shape[1:]) != (source.context, 2):
        raise ValueError("Parent Keras signature differs from detector contract")
    model.export(output / "saved_model", format="tf_saved_model", verbose=False,
                 input_signature=[tf.TensorSpec((1, source.context, 5), tf.float32, name="features")])
    converter = tf.lite.TFLiteConverter.from_saved_model(str(output / "saved_model"))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: ([sample[None].copy()] for sample in samples)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    (output / "model.tflite").write_bytes(converter.convert())
    graph = inspect_integer_graph(output / "model.tflite")
    shutil.copyfile(run / "bundle/model.tflite", output / "model.float.tflite")
    int8_runner = DetectionRuntime(output / "model.tflite", source.context)
    float_runner = DetectionRuntime(output / "model.float.tflite", source.context)
    if int8_runner.input_details["dtype"] != np.int8 or float_runner.input_details["dtype"] != np.float32:
        raise ValueError("Unexpected quantized or preserved float runtime")
    model_hashes = {name: sha256(output / name) for name in ("model.tflite", "model.float.tflite")}
    results = evaluate_quantized(run, source, normalizer, verified, float_runner, int8_runner, output, cache=cache)
    source.verify_unchanged()
    int8_runner.verify_unchanged()
    float_runner.verify_unchanged()
    if (any(sha256(run / name) != digest for name, digest in verified["evidence_sha256"].items())
            or {name: sha256(path) for name, path in implementation_files(Path(__file__).parent).items()} != code
            or any(sha256(path) != digest for path, digest in calibration_hashes.items())):
        raise ValueError("Quantization source evidence or implementation changed")
    results.update(policy=POLICY, model_sha256=model_hashes, graph=graph, context=source.context,
                   data_kind=recipe.get("data_kind"),
                   model_bytes={name: (output / name).stat().st_size for name in model_hashes})
    write_json(output / "evaluation.json", results)
    aggregate = {key: value for key, value in results.items() if key != "subjects"}
    write_json(output / "metrics.json", aggregate)
    deployment = {"kind": POLICY["kind"], "default_model": "model.tflite", "float_model": "model.float.tflite",
                  "parent_model_sha256": verified["evidence_sha256"]["bundle/model.keras"],
                  "calibration": calibration_metadata, "graph": graph, "implementation_sha256": code,
                  "versions": declaration["versions"]}
    write_json(output / "recipe.json", {**recipe, "parent_code_sha256": recipe.get("code_sha256"),
                                        "code_sha256": code, "deployment": deployment})
    shutil.copyfile(run / "bundle/preprocessing.json", output / "preprocessing.json")
    shutil.copyfile(run / "bundle/model.keras", output / "model.keras")
    artifacts, checks = [], []
    for name, runner, reference_name in (("model.tflite", int8_runner, "reference.int8.npz"),
                                          ("model.float.tflite", float_runner, "reference.float.npz")):
        artifact = _artifact(output / name, name, runner, verified["class_order"])
        artifacts.append(artifact)
        checks.append(create_reference(artifact, output / reference_name))
        artifacts.append(Artifact(output / reference_name, reference_name, "reference", "npz", "Synthetic runtime input/output"))
    for name, role, fmt in (("model.keras", "training_checkpoint", "keras"),
                            ("preprocessing.json", "preprocessing_state", "json"),
                            ("recipe.json", "recipe_metadata", "json"), ("metrics.json", "evaluation", "json"),
                            ("calibration.json", "calibration_summary", "json")):
        expected = verified["evidence_sha256"].get(f"bundle/{name}") if name in ("model.keras", "preprocessing.json") else None
        artifacts.append(Artifact(output / name, name, role, fmt, "Fixed detection quantization experiment",
                                  expected_sha256=expected))
    bound = {a.name: sha256(a.source) for a in artifacts}
    checks.extend([
        Check("integer_graph", "passed", "All graph tensors are integer/bool; builtin operators only. No hardware support assertion.",
              {"model.tflite": bound["model.tflite"]}, graph),
        Check("train_only_calibration", "passed", "Calibration used only frozen training series and the parent fitted state; exact selections remain local.",
              {name: bound[name] for name in ("model.tflite", "calibration.json", "preprocessing.json", "recipe.json")}),
        Check("matched_evaluation", "passed", "All three runtimes evaluated on exact original eligible test contexts; quantization error measured, no acceptance threshold.",
              {name: bound[name] for name in ("model.keras", "model.tflite", "model.float.tflite", "metrics.json", "recipe.json", "preprocessing.json")}),
        Check("task_quality", "not_run", "No task acceptance threshold was declared; see measured quantization differences."),
        Check("target_hardware", "not_run", "CPU LiteRT only; MCU operators, arena size, timing and power remain unmeasured."),
    ])
    stage_bundle(output / "bundle", title="SleepKit annotated-period membership TCN: float and int8",
                 artifacts=artifacts, checks=checks, metadata={"sleepkit": {"recipe": recipe["recipe"]}},
                 card=_card(aggregate))
    return aggregate


def _card(metrics):
    floating, quantized = metrics["models"]["model.keras"], metrics["models"]["model.tflite"]
    return f"""---
tags:
- sleepkit
- tflite
- time-series
- int8
---
# Annotated-period membership TCN: float and int8

Run data kind: {metrics['data_kind']}. Synthetic runs are fixtures, not dataset benchmarks.

Predicts membership inside/outside supported annotated nightly periods from CMIDSS
wrist signals. This is not clinical per-sample sleep/wake, confirmed wear, event AP,
or a validated MCU deployment. Source channels are TS (local seconds of day), ENMO,
and ZANGLE at 0.2 Hz. Features use 60-second windows with 30-second stride; fitted
training-only normalization and the exact feature policy are in preprocessing.json.
Complete {metrics['context']}-feature contexts span {(metrics['context'] - 1) * 30 + 60:,} seconds and are noncausal. Invalid contexts
and incomplete tails are omitted. Predictions are timestamped at each window's final
sample and become available after the full context.

Default model.tflite has int8 input/output. model.float.tflite preserves the original
float32 export; model.keras preserves the training checkpoint. Manifest tensor specs
declare actual scales and zero points. Outputs are logits in outside/inside order:
dequantize int8 outputs, then apply softmax once. The installed implementation loads
no code from this bundle:

```python
from sleepkit.recipes.detection.inference import predict
result = predict("bundle-directory", sensor_data, sample_time=utc_seconds)
floating = predict("bundle-directory", sensor_data, sample_time=utc_seconds,
                   model_name="model.float.tflite")
```

Install the matching SleepKit implementation with NumPy and ai-edge-litert; training
and conversion additionally require TensorFlow/Keras. Input sensor_data is [3,samples];
utc_seconds is an aligned signed int64 five-second sample clock.

Calibration used up to two seed-zero-selected eligible contexts per frozen training
series, with no validation/test inputs or normalization refit. The exact real examples
and identifiers stay local. calibration.json contains only counts, policy and hashes.
Both reference NPZ files are synthetic runtime probes.

On {floating['feature_frames_evaluated']:,} identical eligible test frames, Keras accuracy
was {floating['accuracy']:.6%} (macro-F1 {floating['macro_f1']:.6f}); int8 accuracy was
{quantized['accuracy']:.6%} (macro-F1 {quantized['macro_f1']:.6f}). Unknown target contexts
were excluded. metrics.json records all denominators, class recall, drift and runtime
range statistics. No task-quality threshold or int8 logit-equivalence is asserted.
The test set has been inspected; it must not be presented as untouched for future
model selection. Historical training exposure remains unresolved.

Model artifact licensing is unspecified. No Hugging Face upload has been performed.
Choose an appropriate artifact license and release destination before publication.
"""

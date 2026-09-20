"""Quantization matches native frames and keeps calibration evidence local."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from sleepkit.recipes.detection.data import Recording
from sleepkit.recipes.detection.evaluation import summarize
from sleepkit.recipes.detection.preprocessing import Normalizer
from sleepkit.recipes.detection.quantization import evaluate_quantized
from tests.test_detection_evaluation import make_run


MISSING = [name for name in ("tensorflow", "keras", "h5py", "ai_edge_litert") if importlib.util.find_spec(name) is None]


def evaluation_case(tmp_path):
    run = make_run(tmp_path)
    verified = summarize(run, tmp_path / "parent.json")
    recordings = {}
    for subject, samples, label in (("a", 54, 1), ("b", 18, 0), ("empty", 18, -1), ("short", 10, 0)):
        data = np.stack([np.arange(samples) * 5, np.ones(samples), np.ones(samples)]).astype(np.float32)
        recordings[subject] = Recording(data, np.full(samples, label, dtype=np.int8))
    source = SimpleNamespace(context=2, split=json.loads((run / "split.json").read_text()),
                             source_hashes=json.loads((run / "sources.json").read_text()),
                             read=lambda subject, labels=True: recordings[subject])
    normalizer = Normalizer.from_dict(json.loads((run / "bundle/preprocessing.json").read_text()))
    class Floating:
        def predict(self, values):
            return np.tile(np.array([[0., 2.]], dtype=np.float32), (len(values), 1)), {}
    class Integer:
        def predict(self, values):
            return np.tile(np.array([[2., 0.]], dtype=np.float32), (len(values), 1)), {
                "input_values": values.size, "clipped_input_values": 1, "argmax_ties": 0,
                "output_saturated_values": 0,
                "raw_output": np.tile(np.array([[4, 0]], dtype=np.int8), (len(values), 1))}
    destination = tmp_path / "quantized"
    destination.mkdir()
    return run, source, normalizer, verified, Floating(), Integer(), destination


def test_matched_evaluation_reconstructs_different_errors_and_exact_denominator(tmp_path):
    args = evaluation_case(tmp_path)
    result = evaluate_quantized(*args)
    old = result["models"]["model.keras"]
    new = result["models"]["model.tflite"]
    assert old["feature_frames_evaluated"] == new["feature_frames_evaluated"] == 10
    assert old["confusion_matrix"] == [[0, 2], [0, 8]]
    assert new["confusion_matrix"] == [[2, 0], [8, 0]]
    assert old["accuracy"] == .8 and new["accuracy"] == .2
    assert result["int8_drift"]["argmax_disagreements"] == 10
    assert result["int8_drift"]["correct_to_wrong"] == 8
    assert result["int8_drift"]["wrong_to_correct"] == 2
    assert result["int8_drift"]["max_abs_logit_difference"] == 2
    assert result["int8_drift"]["mean_abs_logit_difference"] == 2
    assert result["float_parity"]["passed"] is True
    assert result["subjects"]["empty"]["model.tflite"] is None
    assert result["subjects"]["short"]["model.tflite"] is None
    assert result["int8_runtime"]["input_values"] == 50
    assert result["int8_runtime"]["clipped_input_values"] == 5
    with np.load(args[-1] / "test-predictions.npz", allow_pickle=False) as saved:
        assert saved["int8_outputs"].dtype == np.int8
        assert saved["targets"].tolist() == [1] * 8 + [0] * 2
        np.testing.assert_array_equal(saved["int8_outputs"], np.tile([4, 0], (10, 1)))
        np.testing.assert_array_equal(saved["int8_logits"], saved["int8_outputs"] * .5)


@pytest.mark.parametrize("field,value", [("subject", "b"), ("source_sha256", "different"),
                                          ("context_start_sample", 6), ("context_end_sample", 23),
                                          ("feature_end_sample", 12), ("target", 0),
                                          ("sensor_valid", False), ("context_eligible", False),
                                          ("eligible_output_index", 1)])
def test_replay_rejects_native_coordinate_target_and_eligibility_drift(tmp_path, field, value):
    args = evaluation_case(tmp_path)
    path = args[0] / "test-index.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0][field] = value
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="frozen scoring index"):
        evaluate_quantized(*args)
    assert not (args[-1] / "test-predictions.npz").exists()


@pytest.mark.parametrize("kind", ["target", "extra_predictions", "trailing_index", "subject_coverage", "float_mismatch"])
def test_denominator_saved_target_and_float_parity_guards(tmp_path, kind):
    args = list(evaluation_case(tmp_path))
    if kind in ("target", "extra_predictions"):
        path = args[0] / "test-predictions.npz"
        with np.load(path) as arrays:
            targets, logits = arrays["targets"], arrays["logits"]
        if kind == "target":
            targets[0] = 0
        else:
            targets, logits = np.append(targets, targets[:2]), np.concatenate([logits, logits[:2]])
        np.savez_compressed(path, targets=targets, logits=logits)
    elif kind == "trailing_index":
        with (args[0] / "test-index.jsonl").open("a") as stream:
            stream.write("{}\n")
    elif kind == "subject_coverage":
        args[3]["subjects"]["a"]["eligible_outputs"] += 2
    else:
        args[4] = args[5]
    with pytest.raises(ValueError):
        evaluate_quantized(*args)
    assert not (args[-1] / "test-predictions.npz").exists()


@pytest.mark.skipif(bool(MISSING) and os.environ.get("SLEEPKIT_REQUIRE_DETECTION") != "1",
                    reason="Requires real TensorFlow/Keras, HDF5 and LiteRT")
def test_real_train_calibrate_integer_convert_evaluate_and_infer(tmp_path):
    assert not MISSING, f"Required detection dependencies missing: {MISSING}"
    code = r'''
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, str(Path.cwd() / "tests"))
from test_detection_target_dataset import make_dataset_fixture
from sleepkit.recipes.detection.target_dataset import AnnotatedDataset
from sleepkit.recipes.detection.recipe import run_membership, Config
from sleepkit.recipes.detection.quantization import quantize_run, inspect_integer_graph
from sleepkit.recipes.detection.inference import predict
from sleepkit.artifacts.package import sha256, validate_bundle
from sleepkit.artifacts.runtime import replay_bundle
from sleepkit.artifacts import stage_release
root = Path(sys.argv[1])
source = AnnotatedDataset(*make_dataset_fixture(root, context=2))
run = root / "parent"
run_membership(source, run, Config(context=2, epochs=1, batch_size=2), data_kind="synthetic")
parent_hashes = {p.name: sha256(p) for p in (run / "bundle").iterdir()}
output = root / "int8"
metrics = quantize_run(run, source, output)
bundle = output / "bundle"
validated = validate_bundle(bundle, profile="runnable")
assert len(replay_bundle(bundle)) == 2
assert all(sha256(run / "bundle" / name) == digest for name, digest in parent_hashes.items())
assert sha256(bundle / "model.keras") == parent_hashes["model.keras"]
assert sha256(bundle / "model.float.tflite") == parent_hashes["model.tflite"]
entries = {entry["path"]: entry for entry in validated["manifest"]["artifacts"]}
for key in ("inputs", "outputs"):
    assert entries["model.tflite"][key][0]["dtype"] == "int8"
    assert entries["model.float.tflite"][key][0]["dtype"] == "float32"
assert set(inspect_integer_graph(bundle / "model.tflite")["tensor_types"]) <= {"INT8", "INT16", "INT32", "INT64", "BOOL"}
try:
    inspect_integer_graph(bundle / "model.float.tflite")
except ValueError as error:
    assert "noninteger tensor" in str(error)
else:
    raise AssertionError("Float graph accepted as integer-only")
selection = [json.loads(line) for line in (output / "calibration/calibration-index.jsonl").read_text().splitlines()]
selected_ids = {row["source_id"] for row in selection}
assert selected_ids and selected_ids <= set(source.split["train"])
assert not selected_ids & (set(source.split["validation"]) | set(source.split["test"]))
assert all(sum(row["source_id"] == subject for row in selection) <= 2 for subject in selected_ids)
with np.load(output / "calibration/calibration.npz", allow_pickle=False) as arrays:
    calibration = arrays["contexts"]
    assert calibration.dtype == np.float32 and calibration.shape == (len(selection), 2, 5)
# Public artifacts contain aggregate calibration metadata and synthetic probes only.
assert {name for name in entries if name.endswith(".npz")} == {"reference.int8.npz", "reference.float.npz"}
for path in bundle.glob("*.json"):
    for subject in source.source_hashes:
        assert f'"{subject}"' not in path.read_text()
assert not (bundle / "test-predictions.npz").exists()
assert not (bundle / "calibration.npz").exists()
for reference_name in ("reference.int8.npz", "reference.float.npz"):
    with np.load(bundle / reference_name, allow_pickle=False) as reference:
        assert not any(np.array_equal(reference["input_0"][0], values) for values in calibration)
with np.load(output / "test-predictions.npz", allow_pickle=False) as arrays:
    targets, raw, logits, floating = (arrays[key] for key in ("targets", "int8_outputs", "int8_logits", "float_logits"))
    assert raw.dtype == np.int8
    spec = entries["model.tflite"]["outputs"][0]
    decoded = ((raw.astype(np.float64) - spec["zero_point"]) * spec["scale"]).astype(np.float32)
    np.testing.assert_array_equal(decoded, logits)
    for name, values in (("model.tflite", decoded), ("model.float.tflite", floating)):
        recorded = metrics["models"][name]
        confusion = np.zeros((2, 2), np.int64)
        np.add.at(confusion, (targets, values.argmax(axis=1)), 1)
        assert confusion.tolist() == recorded["confusion_matrix"]
        assert recorded["feature_frames_evaluated"] == len(targets)
        assert recorded["accuracy"] == np.trace(confusion) / len(targets)
        shifted = values.astype(np.float64) - values.max(axis=1, keepdims=True)
        loss = np.mean(np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(targets)), targets])
        np.testing.assert_allclose(loss, recorded["cross_entropy"], rtol=1e-10, atol=1e-12)
    assert metrics["models"]["model.keras"]["feature_frames_evaluated"] == len(targets)
# A local test-only release preserves both real runtimes and raw-sensor predictions.
(root / "test-terms.txt").write_text("Test fixture only; no production model grant.")
(root / "test-decision.md").write_text("Synthetic test data only. No publication.")
release = stage_release(
    bundle, root / "release", license_file=root / "test-terms.txt", license_id="other",
    license_name="Synthetic fixture terms", card_body="# Synthetic fixture release",
    decision_file=root / "test-decision.md", profile="runnable")
assert validate_bundle(release, profile="runnable")["checks"] == validated["checks"]
assert len(replay_bundle(release)) == 2
for entry in validated["manifest"]["artifacts"]:
    assert sha256(release / entry["path"]) == entry["sha256"]
for subject in source.split["test"]:
    recording = source.read(subject, labels=False)
    integer = predict(bundle, recording.data, sample_time=recording.sample_time)
    floating = predict(bundle, recording.data, sample_time=recording.sample_time, model_name="model.float.tflite")
    parent = predict(run / "bundle", recording.data, sample_time=recording.sample_time)
    for name, original in (("model.tflite", integer), ("model.float.tflite", floating)):
        promoted = predict(release, recording.data, sample_time=recording.sample_time, model_name=name)
        for key in ("times", "available_at", "logits", "probabilities"):
            np.testing.assert_array_equal(promoted[key], original[key])
        assert promoted["target"] == original["target"]
        assert promoted["class_names"] == original["class_names"]
    assert integer["target"] == floating["target"] == source.target
    np.testing.assert_array_equal(integer["times"], floating["times"])
    np.testing.assert_array_equal(floating["logits"], parent["logits"])
    expected = np.exp(integer["logits"] - integer["logits"].max(axis=1, keepdims=True))
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_array_equal(integer["probabilities"], expected)
try:
    quantize_run(run, source, output)
except FileExistsError:
    pass
else:
    raise AssertionError("Existing output overwritten")
wrong = SimpleNamespace(provenance={"kind": "wrong"}, context=source.context, split=source.split,
                        source_hashes=source.source_hashes, verify_unchanged=source.verify_unchanged)
try:
    quantize_run(run, wrong, root / "wrong-source")
except ValueError as error:
    assert "source differs" in str(error)
else:
    raise AssertionError("Wrong calibration source accepted")
assert not (root / "wrong-source/bundle").exists()
source.verify_unchanged()
'''
    env = {**os.environ, "TF_NUM_INTEROP_THREADS": "1", "TF_NUM_INTRAOP_THREADS": "1", "TF_CPP_MIN_LOG_LEVEL": "3"}
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], cwd=Path(__file__).resolve().parents[1],
                            env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("kind", ["array", "array_hash", "index_hash", "selection_hash", "dtype", "nonfinite"])
def test_calibration_requires_exact_persisted_inputs_and_evidence(tmp_path, kind):
    from sleepkit.artifacts.package import sha256
    from sleepkit.recipes.detection.quantization import verified_calibration
    samples = np.ones((2, 2, 5), dtype=np.float32)
    np.savez_compressed(tmp_path / "calibration.npz", contexts=samples)
    (tmp_path / "calibration-index.jsonl").write_text("{}\n")
    (tmp_path / "selection.json").write_text("{}")
    metadata = {key: sha256(tmp_path / name) for key, name in (
        ("array_sha256", "calibration.npz"), ("selected_index_sha256", "calibration-index.jsonl"),
        ("local_report_sha256", "selection.json"))}
    np.testing.assert_array_equal(verified_calibration(samples, metadata, tmp_path), samples)
    if kind == "array":
        samples[0, 0, 0] = 2
    elif kind.endswith("hash"):
        metadata[{"array_hash": "array_sha256", "index_hash": "selected_index_sha256",
                  "selection_hash": "local_report_sha256"}[kind]] = "0" * 64
    else:
        stored = samples.astype(np.float64) if kind == "dtype" else samples.copy()
        if kind == "nonfinite":
            stored[0, 0, 0] = np.nan
        np.savez_compressed(tmp_path / "calibration.npz", contexts=stored)
        metadata["array_sha256"] = sha256(tmp_path / "calibration.npz")
    with pytest.raises(ValueError):
        verified_calibration(samples, metadata, tmp_path)

"""Exact common samples, distinct predictions, and read-only evidence bindings."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sleepkit.artifacts import Artifact, stage_bundle
from sleepkit.artifacts.package import sha256, write_json
from sleepkit.recipes.detection import comparison
from sleepkit.recipes.detection.data import Recording
from sleepkit.recipes.detection.preprocessing import Features, Normalizer, fingerprint
from sleepkit.recipes.detection.scoring import write_index
from sleepkit.recipes.detection.split import TARGET


def make_case(tmp_path, monkeypatch, *, unknown=False, empty=False, mutate_runtime=None):
    run, features, baseline, raw = (tmp_path / name for name in ("run", "features", "baseline", "raw"))
    for path in (run, features, baseline, raw):
        path.mkdir()
    split = {"train": ["train"], "validation": ["validation"], "test": ["a", "b"]}
    frame_targets = {"a": [0] * 240 + [1] * 480, "b": [0] * 240 + [1] * 240}
    if unknown:
        frame_targets["a"][250] = -1
    recordings, historic = {}, {}
    for subject, values in frame_targets.items():
        samples = (len(values) - 1) * 6 + 12
        data = np.stack([np.arange(samples) * 5, np.ones(samples), np.ones(samples)]).astype(np.float32)
        if subject == "b":
            data[1, 0] = np.nan  # Historical-only eligibility before the common context.
        labels = np.zeros(samples, dtype=np.int8)
        labels[np.arange(len(values)) * 6 + 11] = values
        recordings[subject] = Recording(data, labels, np.arange(samples, dtype=np.int64) * 5)
        valid = np.ones(480, dtype=bool)
        if subject == "a":
            valid[:240] = False
        if empty:
            valid[:] = False
        historic[subject] = Features(np.zeros((480, 5), dtype=np.float32), valid, np.arange(480) * 6 + 11)
        (features / f"{subject}.h5").write_bytes(f"opaque {subject} features".encode())
    for subject in (s for group in split.values() for s in group):
        (raw / subject).write_bytes(f"source {subject}".encode())
    sources = {p.name: sha256(p) for p in raw.iterdir()}
    checks = []
    def verify():
        checks.append(True)
        if any(sha256(raw / subject) != digest for subject, digest in sources.items()):
            raise ValueError("source changed")
    def read(subject, *, labels=True):
        assert labels is True
        return recordings[subject]
    source = SimpleNamespace(context=240, target=TARGET, split=split, source_hashes=sources,
                             provenance={"kind": "test protocol"}, read=read, verify_unchanged=verify)
    summary = write_index(source, split["test"], 240, run / "test-index.jsonl")
    rows = [json.loads(line) for line in (run / "test-index.jsonl").read_text().splitlines()]
    eligible = [row for row in rows if row["context_eligible"]]
    targets = np.array([row["target"] for row in eligible], dtype=np.int32)
    # New model is perfect on a, entirely wrong on the eligible b context.
    predictions = np.array([row["target"] if row["subject"] == "a" else 0 for row in eligible])
    logits = np.eye(2, dtype=np.float64)[predictions] * 2
    confusion = np.zeros((2, 2), dtype=np.int64)
    np.add.at(confusion, (targets, predictions), 1)
    count = len(targets)
    support, predicted_support = confusion.sum(axis=1), confusion.sum(axis=0)
    np.savez_compressed(run / "test-predictions.npz", logits=logits, targets=targets)
    summary["predictions_sha256"] = sha256(run / "test-predictions.npz")
    recipe = {
        "recipe": "sleepkit.detection/v2", "class_names": TARGET["classes"], "output": "logits", "target": TARGET,
        "context": 240, "config": {"context": 240}, "dataset": source.provenance, "scoring": summary,
        "split_sha256": fingerprint(split), "source_sha256": fingerprint(sources),
        "subjects_per_split": {p: len(s) for p, s in split.items()},
        "contexts_per_split": {"test": count // 240},
    }
    metrics = {
        "model": "model.keras", "split": "test", "target": TARGET, "class_order": TARGET["classes"],
        "feature_frames_evaluated": count, "confusion_matrix": confusion.tolist(),
        "accuracy": float(np.trace(confusion) / count),
        "macro_f1": float(np.mean(2 * confusion.diagonal() / (support + predicted_support))),
        "cross_entropy": float(np.log1p(np.exp(-2)) + 2 * np.count_nonzero(targets != predictions) / count),
        "f1_zero_division": 0,
    }
    for name, value in (("split.json", split), ("sources.json", sources), ("test-index-summary.json", summary),
                        ("recipe.json", recipe), ("metrics.json", metrics),
                        ("preprocessing.json", Normalizer(np.zeros(5), np.ones(5), 10).to_dict())):
        write_json(run / name, value)
    (run / "model.keras").write_bytes(b"opaque model; never load")
    artifacts = [Artifact(run / name, name, role, fmt, "test fixture") for name, role, fmt in (
        ("model.keras", "model", "keras"), ("recipe.json", "recipe_metadata", "json"),
        ("metrics.json", "evaluation", "json"), ("preprocessing.json", "preprocessing_state", "json"))]
    stage_bundle(run / "bundle", title="test", artifacts=artifacts, card="test")
    (baseline / "model.tflite").write_bytes(b"opaque baseline")
    monkeypatch.setattr(comparison, "HASHES", {"model.tflite": sha256(baseline / "model.tflite")})
    monkeypatch.setattr(comparison, "version", lambda name: "test-version")
    def read_features(path, data, *, sample_time):
        subject = path.stem
        assert data is recordings[subject].data
        assert sample_time is recordings[subject].sample_time
        return historic[subject], {"kind": "synthetic verified features"}
    monkeypatch.setattr(comparison, "read_verified_features", read_features)
    runtime_checks = []
    class Runtime:
        def __init__(self, path):
            assert path == baseline / "model.tflite"
        def predict(self, values):
            assert values.shape == (240, 5)
            if mutate_runtime:
                mutate_runtime(run, features, baseline, raw)
            return np.tile(np.array([[1., 0.]], dtype=np.float32), (240, 1)), 3
        def verify_unchanged(self):
            runtime_checks.append(True)
    monkeypatch.setattr(comparison, "SD2Runtime", Runtime)
    checks.clear()  # Count the comparison lifecycle independently of fixture creation.
    return SimpleNamespace(run=run, source=source, features=features, baseline=baseline,
                           recordings=recordings, checks=checks, runtime_checks=runtime_checks)


def execute(case, output):
    return comparison.compare(case.run, case.source, case.features, case.baseline, output)


def test_class_metrics_analytic_and_absent_classes():
    metrics = comparison.class_metrics([[3, 1], [2, 4]])
    assert metrics["accuracy"] == .7
    assert metrics["class_recall"] == [.75, 2 / 3]
    assert metrics["macro_f1"] == pytest.approx((6 / 9 + 8 / 11) / 2)
    assert comparison.class_metrics([[4, 0], [0, 0]])["macro_f1"] == .5
    assert comparison.class_metrics([[4, 0], [0, 0]])["class_recall"] == [1., None]
    assert comparison.class_metrics([[0, 0], [0, 0]]) is None


def matching_rows():
    return [{"subject": "a", "source_sha256": "hash", "context_start_sample": 0,
             "context_end_sample": 17, "feature_end_sample": end, "target": target}
            for end, target in ((11, 0), (17, 1))]


@pytest.mark.parametrize("field,value", [("subject", "b"), ("source_sha256", "other"),
                                          ("context_start_sample", 6), ("context_end_sample", 23),
                                          ("feature_end_sample", 12), ("target", 1)])
def test_match_context_rejects_coordinate_or_target_drift(field, value):
    rows = matching_rows()
    comparison.match_context(rows, subject="a", source_hash="hash", ends=[11, 17], targets=[0, 1])
    rows[0][field] = value
    with pytest.raises(ValueError, match="coordinates or targets"):
        comparison.match_context(rows, subject="a", source_hash="hash", ends=[11, 17], targets=[0, 1])


@pytest.mark.parametrize("rows,ends,targets", [([], [], []), (matching_rows(), [11], [0]),
                                             (matching_rows(), [11, 17], [0])])
def test_match_context_rejects_length_drift(rows, ends, targets):
    with pytest.raises(ValueError, match="lengths"):
        comparison.match_context(rows, subject="a", source_hash="hash", ends=ends, targets=targets)


def test_full_comparison_uses_common_denominator_and_preserves_offsets(tmp_path, monkeypatch):
    case = make_case(tmp_path, monkeypatch)
    output = tmp_path / "comparison"
    report = execute(case, output)
    assert report["new_native"]["feature_frames_evaluated"] == 960
    assert report["new_native"]["accuracy"] == .75
    assert report["historical_native"]["frames"] == 720
    assert report["historical_native"]["accuracy"] == pytest.approx(1 / 3)
    assert report["historical_common"]["frames"] == report["new_common"]["frames"] == 480
    assert report["historical_common"]["confusion_matrix"] == [[0, 0], [480, 0]]
    assert report["new_common"]["confusion_matrix"] == [[0, 0], [240, 240]]
    assert report["new_common"]["accuracy"] == .5
    assert report["coverage"]["new_only_tail_contexts"] == 1
    assert report["coverage"]["excluded_nonfinite_features"] == 1
    assert report["runtime"]["clipped_input_values"] == 9
    assert len(case.checks) == 2 and len(case.runtime_checks) == 1
    with np.load(output / "historical-predictions.npz", allow_pickle=False) as arrays:
        assert arrays["common_historical_starts"].tolist() == [0, 480]
        assert arrays["common_new_starts"].tolist() == [240, 720]
        assert arrays["targets"].tolist() == [1] * 240 + [0] * 240 + [1] * 240
        assert arrays["outputs"].shape == (720, 2)
    common = [json.loads(line) for line in (output / "common-index.jsonl").read_text().splitlines()]
    assert [(row["subject"], row["historical_prediction_start"], row["new_prediction_start"]) for row in common] == [
        ("a", 0, 240), ("b", 480, 720)]
    assert all(row["targets_sha256"] == fingerprint([1] * 240) for row in common)
    historical = [json.loads(line) for line in (output / "historical-index.jsonl").read_text().splitlines()]
    assert [row["prediction_start"] for row in historical] == [None, 0, 240, 480]
    assert json.loads((output / "comparison.json").read_text()) == report
    assert all(sha256(output / name) == digest for name, digest in report["evidence_sha256"].items())
    declaration = json.loads((output / "declaration.json").read_text())
    assert {"shared/classification.py", "components.py"} <= declaration["implementation_sha256"].keys()


def test_shared_implementation_mutation_prevents_completed_comparison(tmp_path, monkeypatch):
    from sleepkit.recipes._components import implementation_files

    shared = tmp_path / "classifier.py"
    shared.write_text("original")
    inventory = implementation_files(Path(comparison.__file__).parent)
    inventory["shared/classification.py"] = shared
    monkeypatch.setattr(comparison, "implementation_files", lambda directory: inventory, raising=False)

    def mutate(*args):
        shared.write_text("changed")

    case = make_case(tmp_path, monkeypatch, mutate_runtime=mutate)
    output = tmp_path / "comparison"
    with pytest.raises(ValueError, match="changed"):
        execute(case, output)
    assert not (output / "comparison.json").exists()


def test_unknown_target_excludes_whole_historical_context(tmp_path, monkeypatch):
    case = make_case(tmp_path, monkeypatch, unknown=True)
    report = execute(case, tmp_path / "comparison")
    assert report["coverage"]["excluded_unknown_target"] == 1
    assert report["historical_native"]["frames"] == 480
    assert report["new_common"]["frames"] == 240
    assert report["subjects"]["a"]["historical_common"] is None


@pytest.mark.parametrize("kind", ["run", "features", "baseline", "raw"])
def test_mutation_during_runtime_leaves_no_completed_report(tmp_path, monkeypatch, kind):
    def mutate(run, features, baseline, raw):
        path = {"run": run / "sources.json", "features": features / "a.h5",
                "baseline": baseline / "model.tflite", "raw": raw / "a"}[kind]
        with path.open("ab") as stream:
            stream.write(b"changed")
    case = make_case(tmp_path, monkeypatch, mutate_runtime=mutate)
    output = tmp_path / "comparison"
    with pytest.raises(ValueError, match="changed"):
        execute(case, output)
    assert (output / "declaration.json").is_file()
    assert not (output / "comparison.json").exists()


def test_wrong_source_binding_rejected_before_output(tmp_path, monkeypatch):
    case = make_case(tmp_path, monkeypatch)
    case.source.provenance = {"kind": "different"}
    output = tmp_path / "comparison"
    with pytest.raises(ValueError, match="Source protocol"):
        execute(case, output)
    assert not output.exists()


def test_changed_candidate_target_rejected(tmp_path, monkeypatch):
    case = make_case(tmp_path, monkeypatch)
    case.recordings["a"].labels[11] = 1
    output = tmp_path / "comparison"
    with pytest.raises(ValueError, match="coordinates or targets"):
        execute(case, output)
    assert not (output / "comparison.json").exists()


def test_empty_intersection_has_no_completed_report(tmp_path, monkeypatch):
    case = make_case(tmp_path, monkeypatch, empty=True)
    output = tmp_path / "comparison"
    with pytest.raises(ValueError, match="No common"):
        execute(case, output)
    assert not (output / "comparison.json").exists()


def test_existing_output_nested_run_and_wrong_context_rejected(tmp_path, monkeypatch):
    case = make_case(tmp_path, monkeypatch)
    output = tmp_path / "comparison"
    output.mkdir()
    marker = output / "keep"
    marker.write_text("existing evidence")
    with pytest.raises(FileExistsError):
        execute(case, output)
    assert marker.read_text() == "existing evidence"
    with pytest.raises(ValueError, match="outside"):
        execute(case, case.run / "nested")
    case.source.context = 120
    with pytest.raises(ValueError, match="240"):
        execute(case, tmp_path / "different")


def test_recipe_swapped_before_summarize_cannot_complete_comparison(tmp_path, monkeypatch):
    case = make_case(tmp_path, monkeypatch)
    original = comparison.summarize

    def swap_then_summarize(run, output):
        # Simulate a consistent replacement run with a different protocol after
        # compare's early source check but before summarize captures its inputs.
        bundle = run / "bundle"
        recipe = json.loads((bundle / "recipe.json").read_text())
        recipe["dataset"] = {"kind": "replacement protocol"}
        summary = json.loads((run / "test-index-summary.json").read_text())
        summary["provenance_sha256"] = fingerprint(recipe["dataset"])
        recipe["scoring"] = summary
        write_json(run / "test-index-summary.json", summary)
        write_json(bundle / "recipe.json", recipe)
        manifest = json.loads((bundle / "manifest.json").read_text())
        for artifact in manifest["artifacts"]:
            if artifact["path"] == "recipe.json":
                artifact["sha256"] = sha256(bundle / "recipe.json")
                artifact["bytes"] = (bundle / "recipe.json").stat().st_size
        write_json(bundle / "manifest.json", manifest)
        checksums = json.loads((bundle / "checksums.json").read_text())
        for name in ("recipe.json", "manifest.json"):
            checksums[name] = sha256(bundle / name)
        write_json(bundle / "checksums.json", checksums)
        return original(run, output)

    monkeypatch.setattr(comparison, "summarize", swap_then_summarize)
    output = tmp_path / "comparison"
    with pytest.raises(ValueError, match="Recipe changed"):
        execute(case, output)
    assert not (output / "declaration.json").exists()
    assert not (output / "comparison.json").exists()

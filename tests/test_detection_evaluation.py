"""Saved scoring evidence is checked without importing a model runtime."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from sleepkit.artifacts import Artifact, stage_bundle
from sleepkit.artifacts.package import sha256, write_json
from sleepkit.recipes.detection.data import Recording
from sleepkit.recipes.detection.evaluation import main, summarize
from sleepkit.recipes.detection.preprocessing import Normalizer, fingerprint
from sleepkit.recipes.detection.scoring import write_index
from sleepkit.recipes.detection.split import TARGET


def make_run(tmp_path, mutate=None):
    run = tmp_path / "run"
    run.mkdir()
    split = {"train": ["train"], "validation": ["validation"], "test": ["a", "b", "empty", "short"]}
    sources = {s: fingerprint(s) for group in split.values() for s in group}
    recordings = {}
    for subject, samples, label in (("a", 54, 1), ("b", 18, 0), ("empty", 18, -1), ("short", 10, 0)):
        data = np.stack([np.arange(samples) * 5, np.ones(samples), np.ones(samples)]).astype(np.float32)
        recordings[subject] = Recording(data, np.full(samples, label, dtype=np.int8))
    provenance = {"kind": "test"}
    source = SimpleNamespace(source_hashes=sources, provenance=provenance, target=TARGET,
                             read=lambda subject, labels=True: recordings[subject])
    summary = write_index(source, split["test"], 2, run / "test-index.jsonl")
    targets = np.array([1] * 8 + [0] * 2, dtype=np.int32)
    logits = np.tile(np.array([[0., 2.]], dtype=np.float64), (10, 1))
    np.savez_compressed(run / "test-predictions.npz", logits=logits, targets=targets)
    summary["predictions_sha256"] = sha256(run / "test-predictions.npz")
    recipe = {
        "recipe": "sleepkit.detection/v2", "class_names": TARGET["classes"], "output": "logits", "target": TARGET,
        "context": 2, "config": {"context": 2}, "dataset": provenance,
        "scoring": summary, "split_sha256": fingerprint(split), "source_sha256": fingerprint(sources),
        "subjects_per_split": {p: len(s) for p, s in split.items()}, "contexts_per_split": {"test": 5},
    }
    metrics = {"model": "model.keras", "split": "test", "target": TARGET, "class_order": TARGET["classes"],
               "feature_frames_evaluated": 10, "confusion_matrix": [[0, 2], [0, 8]], "accuracy": .8,
               "macro_f1": 4 / 9, "cross_entropy": float(np.log1p(np.exp(-2)) + .4), "f1_zero_division": 0}
    if mutate:
        mutate(run, recipe, metrics, summary, split, sources)
    for name, value in (("split.json", split), ("sources.json", sources), ("test-index-summary.json", summary),
                        ("recipe.json", recipe), ("metrics.json", metrics),
                        ("preprocessing.json", Normalizer(np.zeros(5), np.ones(5), 10).to_dict())):
        write_json(run / name, value)
    (run / "model.keras").write_bytes(b"opaque checkpoint; no model load in report")
    artifacts = [Artifact(run / name, name, role, fmt, "test fixture") for name, role, fmt in (
        ("model.keras", "model", "keras"), ("recipe.json", "recipe_metadata", "json"),
        ("metrics.json", "evaluation", "json"), ("preprocessing.json", "preprocessing_state", "json"))]
    stage_bundle(run / "bundle", title="test", artifacts=artifacts, card="test")
    return run


def rewrite_index(run, summary, edit):
    rows = [json.loads(line) for line in (run / "test-index.jsonl").read_text().splitlines()]
    edit(rows)
    (run / "test-index.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary["sha256"] = sha256(run / "test-index.jsonl")


def test_pooled_and_subject_metrics_are_distinct_and_zero_coverage_explicit(tmp_path):
    run = make_run(tmp_path)
    report = summarize(run, tmp_path / "report.json")
    assert report["pooled"]["accuracy"] == .8
    assert report["pooled"]["class_recall"] == [0., 1.]
    assert report["unweighted_eligible_subject_mean"] == {"accuracy": .5, "macro_f1": .25}
    assert report["subjects"]["empty"]["metrics"] is None
    assert report["subjects"]["short"]["metrics"] is None
    assert report["subjects"]["empty"]["excluded_contexts"]["unknown_target"] == 1
    assert report["subjects"]["a"]["metrics"]["class_recall"] == [None, 1.]
    assert report["coverage"]["subjects_without_eligible_outputs"] == 2
    assert json.loads((tmp_path / "report.json").read_text()) == report


def test_cli_does_not_print_subjects(tmp_path, capsys):
    run = make_run(tmp_path)
    main(["--run", str(run), "--output", str(tmp_path / "report.json")])
    stdout = json.loads(capsys.readouterr().out)
    assert "subjects" not in stdout
    assert "evidence_sha256" not in stdout


@pytest.mark.parametrize("field,value", [
    ("output_index", 3), ("eligible_output_index", 1), ("feature_end_sample", 12),
    ("context_end_sample", 18), ("context_start_sample", 1), ("target", 0),
    ("source_sha256", "wrong"), ("subject", "train"), ("context_eligible", False),
    ("target_known", False), ("exclusion_reasons", ["unknown_target"]), ("sensor_valid", 0),
    ("target_known", 1), ("context_eligible", 1), ("eligible_output_index", False),
])
def test_rehashed_malformed_scoring_rows_rejected(field, value, tmp_path):
    def mutate(run, recipe, metrics, summary, split, sources):
        rewrite_index(run, summary, lambda rows: rows[0].update({field: value}))
    run = make_run(tmp_path, mutate)
    with pytest.raises(ValueError):
        summarize(run, tmp_path / "report.json")
    assert not (tmp_path / "report.json").exists()


@pytest.mark.parametrize("kind", ["summary", "metrics", "split", "sources", "predictions", "context", "target"])
def test_bound_metadata_mismatches_rejected(kind, tmp_path):
    def mutate(run, recipe, metrics, summary, split, sources):
        if kind == "summary":
            summary["eligible_contexts"] += 1
        elif kind == "metrics":
            metrics["accuracy"] = .9
        elif kind == "split":
            split["test"].reverse()
        elif kind == "sources":
            sources["a"] = "wrong"
        elif kind == "predictions":
            with np.load(run / "test-predictions.npz") as arrays:
                logits, targets = arrays["logits"], arrays["targets"]
            targets[0] = 0
            np.savez_compressed(run / "test-predictions.npz", logits=logits, targets=targets)
            summary["predictions_sha256"] = sha256(run / "test-predictions.npz")
        elif kind == "context":
            recipe["context"] = True
        else:
            metrics["target"] = None
    run = make_run(tmp_path, mutate)
    with pytest.raises(ValueError):
        summarize(run, tmp_path / "report.json")


def test_tampered_bundle_and_existing_output_rejected(tmp_path):
    run = make_run(tmp_path)
    output = tmp_path / "report.json"
    output.write_text("keep")
    with pytest.raises(FileExistsError):
        summarize(run, output)
    assert output.read_text() == "keep"
    (run / "bundle" / "metrics.json").write_text("{}")
    with pytest.raises(ValueError, match="Checksum"):
        summarize(run, tmp_path / "new.json")


def test_evidence_mutation_during_read_rejected(tmp_path, monkeypatch):
    run = make_run(tmp_path)
    from sleepkit.recipes.detection import evaluation
    original = evaluation._metrics
    def mutate(confusion, loss):
        (run / "sources.json").write_text("{}")
        return original(confusion, loss)
    monkeypatch.setattr(evaluation, "_metrics", mutate)
    with pytest.raises(ValueError, match="changed during"):
        summarize(run, tmp_path / "report.json")
    assert not (tmp_path / "report.json").exists()


@pytest.mark.parametrize("kind", ["nonfinite", "float32", "scalar", "float_target", "unknown", "short", "long"])
def test_invalid_prediction_arrays_rejected(kind, tmp_path):
    def mutate(run, recipe, metrics, summary, split, sources):
        with np.load(run / "test-predictions.npz") as arrays:
            logits, targets = arrays["logits"], arrays["targets"]
        if kind == "nonfinite":
            logits[0, 0] = np.nan
        elif kind == "float32":
            logits = logits.astype(np.float32)
        elif kind == "scalar":
            targets = np.array(1)
        elif kind == "float_target":
            targets = targets.astype(float)
        elif kind == "unknown":
            targets[0] = -1
        elif kind == "short":
            logits, targets = logits[:-1], targets[:-1]
        else:
            logits, targets = np.concatenate([logits, logits[:1]]), np.append(targets, targets[0])
        np.savez_compressed(run / "test-predictions.npz", logits=logits, targets=targets)
        summary["predictions_sha256"] = sha256(run / "test-predictions.npz")
    run = make_run(tmp_path, mutate)
    with pytest.raises(ValueError):
        summarize(run, tmp_path / "report.json")


@pytest.mark.parametrize("kind", ["duplicate", "partial", "nonobject", "reordered"])
def test_malformed_or_reordered_index_rejected(kind, tmp_path):
    def mutate(run, recipe, metrics, summary, split, sources):
        path = run / "test-index.jsonl"
        lines = path.read_text().splitlines()
        if kind == "duplicate":
            lines[0] = lines[0][:-1] + ', "target": 1}'
        elif kind == "partial":
            lines.pop()
        elif kind == "nonobject":
            lines[0] = "[]"
        else:
            lines = lines[8:10] + lines[:8] + lines[10:]
        path.write_text("\n".join(lines) + "\n")
        summary["sha256"] = sha256(path)
    run = make_run(tmp_path, mutate)
    with pytest.raises(ValueError):
        summarize(run, tmp_path / "report.json")


def test_cannot_put_report_in_verified_bundle(tmp_path):
    run = make_run(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        summarize(run, run / "bundle" / "evaluation.json")

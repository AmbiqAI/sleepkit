"""Analytically known errors and boundaries, plus evidence/output protections."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from sleepkit.artifacts import Artifact, stage_bundle
from sleepkit.artifacts.package import sha256, write_json
from sleepkit.recipes.detection import error_analysis
from sleepkit.recipes.detection.data import Recording
from sleepkit.recipes.detection.preprocessing import Normalizer, fingerprint
from sleepkit.recipes.detection.scoring import write_index
from sleepkit.recipes.detection.split import TARGET


def make_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    labels = {"a": [0] * 100 + [1] * 100, "b": [0] * 20 + [-1] * 20 + [1] * 20,
              "c": [1] * 20, "empty": [-1] * 20}
    split = {"train": ["train"], "validation": ["validation"], "test": list(labels)}
    sources = {subject: fingerprint(subject) for group in split.values() for subject in group}
    recordings = {}
    for subject, values in labels.items():
        samples = (len(values) - 1) * 6 + 12
        data = np.stack([np.arange(samples) * 5, np.ones(samples), np.ones(samples)]).astype(np.float32)
        sample_labels = np.zeros(samples, dtype=np.int8)
        sample_labels[np.arange(len(values)) * 6 + 11] = values
        recordings[subject] = Recording(data, sample_labels)
    source = SimpleNamespace(source_hashes=sources, provenance={"kind": "test"}, target=TARGET,
                             read=lambda subject, labels=True: recordings[subject])
    summary = write_index(source, split["test"], 20, run / "test-index.jsonl")
    targets = np.array([0] * 100 + [1] * 100 + [0] * 20 + [1] * 40, dtype=np.int32)
    predicted = np.zeros(len(targets), dtype=np.int64)
    predicted[100] = 1
    logits = np.eye(2, dtype=np.float64)[predicted] * 2
    confusion = np.array([[120, 0], [139, 1]])
    np.savez_compressed(run / "test-predictions.npz", logits=logits, targets=targets)
    summary["predictions_sha256"] = sha256(run / "test-predictions.npz")
    recipe = {
        "recipe": "sleepkit.detection/v2", "class_names": TARGET["classes"], "output": "logits", "target": TARGET,
        "context": 20, "config": {"context": 20}, "dataset": source.provenance, "scoring": summary,
        "split_sha256": fingerprint(split), "source_sha256": fingerprint(sources),
        "subjects_per_split": {p: len(s) for p, s in split.items()}, "contexts_per_split": {"test": 13},
    }
    metrics = {
        "model": "model.keras", "split": "test", "target": TARGET, "class_order": TARGET["classes"],
        "feature_frames_evaluated": 260, "confusion_matrix": confusion.tolist(), "accuracy": 121 / 260,
        "macro_f1": float(np.mean([240 / 379, 2 / 141])),
        "cross_entropy": float(np.log1p(np.exp(-2)) + 2 * 139 / 260), "f1_zero_division": 0,
    }
    for name, value in (("split.json", split), ("sources.json", sources), ("test-index-summary.json", summary),
                        ("recipe.json", recipe), ("metrics.json", metrics),
                        ("preprocessing.json", Normalizer(np.zeros(5), np.ones(5), 10).to_dict())):
        write_json(run / name, value)
    (run / "model.keras").write_bytes(b"not a real model; diagnostics never load it")
    artifacts = [Artifact(run / name, name, role, fmt, "test fixture") for name, role, fmt in (
        ("model.keras", "model", "keras"), ("recipe.json", "recipe_metadata", "json"),
        ("metrics.json", "evaluation", "json"), ("preprocessing.json", "preprocessing_state", "json"))]
    stage_bundle(run / "bundle", title="test", artifacts=artifacts, card="test")
    return run


def test_known_boundary_errors_and_disjoint_context_bins(tmp_path):
    run = make_run(tmp_path)
    report = error_analysis.analyze(run, tmp_path / "diagnostics.json")
    assert report["pooled"]["confusion_matrix"] == [[120, 0], [139, 1]]
    assert report["pooled"]["errors_by_true_class"] == [0, 139]
    assert report["observed_transitions"] == 1
    bins = report["observed_transition_distance"]
    assert [bins[name]["feature_frames_evaluated"] for name in error_analysis.TRANSITION_BINS] == [21, 100, 79, 60]
    assert [bins[name]["errors"] for name in error_analysis.TRANSITION_BINS] == [10, 50, 39, 40]
    context = report["context_position"]
    assert [context[name]["feature_frames_evaluated"] for name in error_analysis.CONTEXT_BINS] == [104, 52, 104]
    assert [context[name]["errors"] for name in error_analysis.CONTEXT_BINS] == [55, 28, 56]
    assert report["subjects"]["b"]["errors"] == 20
    assert report["subjects"]["empty"]["metrics"] is None
    assert report["subjects"]["empty"]["class_support"] == [0, 0]
    assert json.loads((tmp_path / "diagnostics.json").read_text()) == report


def rows(subject, targets, start=11):
    return [{"subject": subject, "feature_end_sample": start + i * 6, "target": target}
            for i, target in enumerate(targets)]


def test_unknown_gap_and_series_change_do_not_create_or_connect_transitions():
    values = rows("a", [0, 1, -1, 1, 1]) + rows("b", [0, 0])
    segments = error_analysis._segments(iter(values))
    assert segments == {"a": [[11, 17, [17]], [29, 35, []]], "b": [[11, 17, []]]}
    assert error_analysis._transition_bin(35, segments["a"], [11, 29]) == "no_observed_transition"
    assert error_analysis._transition_bin(11, segments["b"], [11]) == "no_observed_transition"


def test_grid_gap_breaks_observed_segment():
    values = rows("a", [0, 1]) + rows("a", [0, 0], start=29)
    assert error_analysis._segments(iter(values)) == {"a": [[11, 17, [17]], [29, 35, []]]}


def test_transition_in_excluded_context_remains_observable():
    values = rows("a", [-1, 0, 1, 1, 1, 1])
    for i, row in enumerate(values):
        row.update(context_eligible=i >= 4, sensor_valid=i != 2)
    segments = error_analysis._segments(iter(values))
    assert segments == {"a": [[17, 41, [23]]]}
    assert error_analysis._transition_bin(41, segments["a"], [17]) == "within_5_minutes"


@pytest.mark.parametrize("offset,expected", [(0, "within_5_minutes"), (60, "within_5_minutes"),
                                            (66, "over_5_through_30_minutes"),
                                            (360, "over_5_through_30_minutes"), (366, "over_30_minutes")])
def test_distance_boundaries_in_sample_units(offset, expected):
    assert error_analysis._transition_bin(1000 + offset, [[0, 2000, [1000]]], [0]) == expected
    assert error_analysis._transition_bin(1000 - offset, [[0, 2000, [1000]]], [0]) == expected


@pytest.mark.parametrize("context,counts", [(2, [2, 0, 0]), (12, [8, 0, 4]), (16, [8, 0, 8]), (20, [8, 4, 8])])
def test_short_context_overlap_has_first_bin_precedence(context, counts):
    bins = [error_analysis._context_bin(frame, context) for frame in range(context)]
    assert [bins.count(name) for name in error_analysis.CONTEXT_BINS] == counts


def test_cli_prints_aggregates_only(tmp_path, capsys):
    run = make_run(tmp_path)
    error_analysis.main(["--run", str(run), "--output", str(tmp_path / "diagnostics.json")])
    printed = json.loads(capsys.readouterr().out)
    assert "subjects" not in printed
    assert "evidence_sha256" not in printed
    assert printed["observed_transitions"] == 1


@pytest.mark.parametrize("name", ["test-predictions.npz", "test-index.jsonl", "sources.json", "bundle/recipe.json"])
def test_tampering_before_analysis_rejected(tmp_path, name):
    run = make_run(tmp_path)
    with (run / name).open("ab") as stream:
        stream.write(b"tampered")
    output = tmp_path / "diagnostics.json"
    with pytest.raises((ValueError, json.JSONDecodeError)):
        error_analysis.analyze(run, output)
    assert not output.exists()


def test_evidence_changed_between_passes_rejected(tmp_path, monkeypatch):
    run = make_run(tmp_path)
    original = error_analysis._segments
    def mutate(rows):
        result = original(rows)
        with (run / "test-index.jsonl").open("ab") as stream:
            stream.write(b" ")
        return result
    monkeypatch.setattr(error_analysis, "_segments", mutate)
    with pytest.raises(ValueError):
        error_analysis.analyze(run, tmp_path / "diagnostics.json")
    assert not (tmp_path / "diagnostics.json").exists()


def test_evidence_changed_after_aggregation_rejected(tmp_path, monkeypatch):
    run = make_run(tmp_path)
    original = error_analysis._finish
    def mutate(accumulator):
        (run / "sources.json").write_text("{}")
        return original(accumulator)
    monkeypatch.setattr(error_analysis, "_finish", mutate)
    with pytest.raises(ValueError, match="changed during"):
        error_analysis.analyze(run, tmp_path / "diagnostics.json")
    assert not (tmp_path / "diagnostics.json").exists()


def test_no_overwrite_bundle_output_or_dangling_symlink(tmp_path):
    run = make_run(tmp_path)
    output = tmp_path / "diagnostics.json"
    output.write_text("keep")
    with pytest.raises(FileExistsError):
        error_analysis.analyze(run, output)
    assert output.read_text() == "keep"
    with pytest.raises(ValueError, match="outside"):
        error_analysis.analyze(run, run / "bundle" / "diagnostics.json")
    alias = tmp_path / "bundle-alias"
    alias.symlink_to(run / "bundle", target_is_directory=True)
    with pytest.raises(ValueError, match="outside"):
        error_analysis.analyze(run, alias / "diagnostics.json")
    dangling = tmp_path / "dangling.json"
    dangling.symlink_to(tmp_path / "nonexistent.json")
    with pytest.raises(FileExistsError):
        error_analysis.analyze(run, dangling)

"""Portable contracts to run against local blocks and their eventual upstream replacement."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from sleepkit.recipes._components import (
    ClassificationAccumulator,
    FileSnapshot,
    implementation_files,
    summarize_confusion,
)


def test_multiclass_counts_losses_and_partitioned_updates():
    labels = np.array([0, 1, 2, 2, 0, 1])
    logits = np.array([[2, 1, 0], [0, 3, 0], [2, 1, 0], [0, 0, 0], [0, 2, 1], [0, 0, 3]], float)
    whole, chunked = ClassificationAccumulator(3), ClassificationAccumulator(3)
    whole.update(labels, logits)
    for start, end in ((0, 2), (2, 3), (3, 6)):
        chunked.update(labels[start:end], logits[start:end])
    expected = [[1, 1, 0], [0, 1, 1], [2, 0, 0]]
    result = whole.result()
    assert result["confusion_matrix"] == expected
    assert result["count"] == 6
    assert result["class_support"] == [2, 2, 2]
    assert result["class_recall"] == [0.5, 0.5, 0]
    assert result["accuracy"] == 1 / 3
    assert result["macro_f1"] == pytest.approx((2 / 5 + 1 / 2) / 3)
    expected_loss = sum(float(np.log(np.exp(row).sum()) - row[target]) for row, target in zip(logits, labels))
    assert result["loss_sum"] == pytest.approx(expected_loss)
    assert result["cross_entropy"] == pytest.approx(expected_loss / 6)
    np.testing.assert_array_equal(chunked.confusion, whole.confusion)
    assert chunked.loss_sum == pytest.approx(whole.loss_sum)
    # Public results must not expose mutable accumulator state.
    whole.confusion[:] = 0
    result["confusion_matrix"][0][0] = 999
    assert whole.confusion.tolist() == expected


def test_dense_targets_fixed_class_macro_f1_and_stable_large_logits():
    accumulator = ClassificationAccumulator(3)
    accumulator.update(np.zeros((2, 4), np.int32), np.tile([10000.0, 0.0, -10000.0], (2, 4, 1)))
    result = accumulator.result()
    assert result["macro_f1"] == 1 / 3
    assert result["class_recall"] == [1.0, None, None]
    assert result["cross_entropy"] == 0
    with pytest.raises(ValueError, match="No classification"):
        ClassificationAccumulator(2).result()


@pytest.mark.parametrize(
    "labels,logits",
    [
        ([0.0, 1.0], [[1, 0], [0, 1]]),
        ([-1], [[1, 0]]),
        ([2], [[1, 0]]),
        ([True], [[1, 0]]),
        ([0, 1], [[[1, 0], [0, 1]]]),
        ([0], [[np.nan, 1]]),
        ([0], [[np.inf, 1]]),
        ([0], [[1 + 1j, 1]]),
        ([0], [[-1e308, 1e308]]),
    ],
)
def test_invalid_update_is_rejected_without_changing_state(labels, logits):
    accumulator = ClassificationAccumulator(2)
    accumulator.update(np.array([1]), np.array([[0.0, 2.0]]))
    before = accumulator.result()
    with pytest.raises(ValueError):
        accumulator.update(np.asarray(labels), np.asarray(logits))
    assert accumulator.result() == before


@pytest.mark.parametrize("value", [True, 1, 0, 2.0, -1])
def test_invalid_class_count(value):
    with pytest.raises(ValueError):
        ClassificationAccumulator(value)


def test_empty_update_and_large_count_summary():
    accumulator = ClassificationAccumulator(2)
    accumulator.update(np.empty((0,), np.int32), np.empty((0, 2)))
    assert accumulator.loss_sum == 0 and not accumulator.confusion.any()
    limit = np.iinfo(np.int64).max
    result = summarize_confusion(np.array([[limit, 0], [0, 0]], np.int64))
    assert result["count"] == limit and result["accuracy"] == 1 and result["macro_f1"] == 0.5
    with pytest.raises(ValueError, match="int64"):
        summarize_confusion(np.array([[limit, 1], [0, 0]], np.uint64))


@pytest.mark.parametrize("matrix", [[[1.0, 0], [0, 1]], [[1, -1], [0, 2]], [[0, 0], [0, 0]], [[1, 2, 3]], [[1]]])
def test_invalid_confusion_summary(matrix):
    with pytest.raises(ValueError):
        summarize_confusion(matrix)


def test_snapshot_copies_inventory_and_exposes_only_caller_labels(tmp_path):
    path = tmp_path / "private-name.txt"
    path.write_text("original")
    inventory = {"input": path}
    snapshot = FileSnapshot.capture(inventory)
    inventory.clear()
    hashes = snapshot.hashes()
    hashes.clear()
    assert set(snapshot.hashes()) == {"input"}
    assert str(tmp_path) not in json.dumps(snapshot.hashes())
    snapshot.verify()
    (tmp_path / "unselected").write_text("Outside the explicitly selected inventory")
    snapshot.verify()
    path.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        snapshot.verify()


@pytest.mark.parametrize("change", ["remove", "directory", "symlink"])
def test_snapshot_rejects_missing_or_nonregular_paths(tmp_path, change):
    path = tmp_path / "file"
    path.write_text("original")
    snapshot = FileSnapshot.capture({"file": path})
    path.unlink()
    if change == "directory":
        path.mkdir()
    if change == "symlink":
        target = tmp_path / "target"
        target.write_text("original")
        path.symlink_to(target)
    with pytest.raises(ValueError, match="regular"):
        snapshot.verify()
    with pytest.raises(ValueError, match="regular"):
        FileSnapshot.capture({"file": path})


def test_code_inventory_includes_moved_implementations():
    directory = Path(__file__).resolve().parents[1] / "sleepkit/recipes/detection"
    files = implementation_files(directory)
    assert {
        "recipe.py",
        "components.py",
        "shared/classification.py",
        "shared/evidence.py",
        "artifacts/package.py",
    } <= files.keys()
    snapshot = FileSnapshot.capture(files)
    snapshot.verify()


def test_file_evidence_and_artifacts_are_stdlib_only():
    code = """
import sys
from sleepkit._edge_candidates.evidence import FileSnapshot
from sleepkit.artifacts import stage_bundle
assert not {'numpy', 'keras', 'tensorflow', 'torch', 'helia_edge'} & sys.modules.keys()
"""
    subprocess.run([sys.executable, "-S", "-c", code], check=True)


def test_detection_report_compatibility_against_independent_formula():
    from sleepkit.recipes.detection.recipe import evaluate

    rng = np.random.default_rng(7)
    logits = rng.normal(size=(3, 11, 2))
    labels = rng.integers(0, 2, size=(3, 11), dtype=np.int32)
    confusion = np.zeros((2, 2), np.int64)
    flat_scores, flat_labels = logits.reshape(-1, 2), labels.reshape(-1)
    np.add.at(confusion, (flat_labels, flat_scores.argmax(axis=1)), 1)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    f1 = np.divide(2 * confusion.diagonal(), denominator, out=np.zeros(2), where=denominator != 0)
    shifted = flat_scores - flat_scores.max(axis=1, keepdims=True)
    loss = float((np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(flat_labels)), flat_labels]).sum())
    expected = {
        "model": "model.keras",
        "split": "test",
        "feature_frames_evaluated": len(flat_labels),
        "confusion_matrix": confusion.tolist(),
        "accuracy": float(np.trace(confusion) / len(flat_labels)),
        "macro_f1": float(f1.mean()),
        "cross_entropy": loss / len(flat_labels),
        "f1_zero_division": 0,
        "class_order": ["WAKE", "SLEEP"],
    }
    assert evaluate(lambda x, training: x, [(logits, labels)]) == expected

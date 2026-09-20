"""Focused contract tests for frozen train-only calibration collection."""

import hashlib
from copy import deepcopy

import numpy as np
import pytest

from sleepkit.recipes.detection.calibration import collect_calibration
from sleepkit.recipes.detection.data import Recording
from sleepkit.recipes.detection.preprocessing import Normalizer, prepare


def _recording(offset=0, *, labels=None, samples=120):
    data = np.stack(
        [np.arange(samples) * 5, np.arange(samples, dtype=np.float32) + offset, np.arange(samples) * 2]
    ).astype(np.float32)
    if labels is None:
        labels = np.zeros(samples, np.int32)
    return Recording(data, np.asarray(labels, np.int32), None)


class _Source:
    context = 3

    def __init__(self, root, records, order=None, train=None):
        self.root = root
        self.records = records
        subjects = list(order or train or records)
        train = list(train or subjects)
        self._split = {"train": train, "validation": ["validation"], "test": ["test"]}
        self.reads = []
        self.verifications = 0
        self.source_hashes = {
            subject: hashlib.sha256(record.data.tobytes()).hexdigest() for subject, record in records.items()
        }
        self._initial = {subject: record.data.copy() for subject, record in records.items()}

    @property
    def split(self):
        return deepcopy(self._split)

    def read(self, subject, *, labels=True):
        self.reads.append((subject, labels))
        if subject not in self.records:
            raise AssertionError(f"unexpected read: {subject}")
        record = self.records[subject]
        return record if labels else Recording(record.data, None, record.sample_time)

    def verify_unchanged(self):
        self.verifications += 1
        for subject, initial in self._initial.items():
            if not np.array_equal(self.records[subject].data, initial, equal_nan=True):
                raise ValueError("source changed")


def _normalizer(record):
    return Normalizer.fit([prepare(record.data)])


def test_train_only_frozen_normalizer_and_local_provenance(tmp_path):
    train = _recording(offset=0)
    source = _Source(
        tmp_path / "raw",
        {"train": train, "validation": _recording(offset=10_000), "test": _recording(offset=20_000)},
        train=["train"],
    )
    normalizer = _normalizer(train)
    output = tmp_path / "calibration"

    array, metadata = collect_calibration(source, normalizer, output, seed=4)

    assert array.dtype == np.float32
    assert array.shape[1:] == (source.context, 5)
    assert source.reads == [("train", True)]
    assert source.verifications >= 2
    assert metadata["train_subject_count"] == 1
    assert metadata["contributing_subject_count"] == 1
    assert metadata["frame_count"] == array.shape[0] * source.context
    assert set(metadata) == {
        "policy", "selection_algorithm", "train_subject_count", "contributing_subject_count", "zero_eligible_subject_count",
        "context_count", "frame_count", "seed", "fitted_state_fingerprint", "selected_index_sha256",
        "array_sha256", "local_report_sha256", "context", "contexts_per_subject",
    }
    with np.load(output / "calibration.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["contexts"], array)
        assert set(saved.files) == {"contexts"}
    rows = [__import__("json").loads(line) for line in (output / "calibration-index.jsonl").read_text().splitlines()]
    assert rows and all({"source_id", "source_sha256", "raw_start_sample", "raw_end_sample_exclusive", "first_feature_end_sample", "last_feature_end_sample", "valid_targets_sha256"} <= set(row) for row in rows)
    report = __import__("json").loads((output / "selection.json").read_text())
    assert report["subjects"][0]["source_id"] == "train"


def test_selection_is_stable_when_train_split_order_changes(tmp_path):
    records = {name: _recording(offset=i * 100) for i, name in enumerate(("a", "b", "c"))}
    first = _Source(tmp_path / "raw1", records, order=["c", "a", "b"])
    second = _Source(tmp_path / "raw2", records, order=["b", "c", "a"])
    normalizer = _normalizer(records["a"])
    first_array, first_meta = collect_calibration(first, normalizer, tmp_path / "one", seed=91)
    second_array, second_meta = collect_calibration(second, normalizer, tmp_path / "two", seed=91)
    np.testing.assert_array_equal(first_array, second_array)
    assert first_meta["selected_index_sha256"] == second_meta["selected_index_sha256"]


def test_unknown_and_invalid_features_are_dropped_without_stitching(tmp_path):
    record = _recording(samples=180)
    record.data[1, 18] = np.nan  # invalidates a feature frame in the middle
    record.labels[11] = -1  # unknown target in the first native context
    source = _Source(tmp_path / "raw", {"train": record})
    normalizer = _normalizer(_recording(samples=180))
    array, metadata = collect_calibration(source, normalizer, tmp_path / "calibration", seed=0)
    assert metadata["context_count"] > 0
    assert all(row["context_start_frame"] % source.context == 0 for row in [__import__("json").loads(line) for line in (tmp_path / "calibration" / "calibration-index.jsonl").read_text().splitlines()])
    assert array.shape[1:] == (source.context, 5)


def test_zero_eligible_and_output_safety(tmp_path):
    labels = np.full(120, -1, np.int32)
    source = _Source(tmp_path / "raw", {"train": _recording(labels=labels)})
    normalizer = _normalizer(_recording())
    with pytest.raises(ValueError, match="No eligible"):
        collect_calibration(source, normalizer, tmp_path / "calibration")
    with pytest.raises(ValueError, match="outside"):
        collect_calibration(_Source(tmp_path / "raw", {"train": _recording()}), normalizer, tmp_path / "raw" / "bad")
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        collect_calibration(_Source(tmp_path / "raw", {"train": _recording()}), normalizer, output)


def test_requires_fitted_v3_normalizer(tmp_path):
    source = _Source(tmp_path / "raw", {"train": _recording()})
    normalizer = _normalizer(_recording())
    normalizer.spec["implementation_version"] = 2
    with pytest.raises(ValueError, match="v3"):
        collect_calibration(source, normalizer, tmp_path / "calibration")

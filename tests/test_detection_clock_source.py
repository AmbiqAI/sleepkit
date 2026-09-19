"""Clock materialization preserves sources and requires matching verification evidence."""

import json

import numpy as np
import pytest

from sleepkit.artifacts.package import sha256
from sleepkit.recipes.detection.clock_source import materialize


@pytest.fixture
def evidence(tmp_path):
    h5py = pytest.importorskip("h5py")
    root = tmp_path / "source"
    root.mkdir()
    with h5py.File(root / "a.h5", "w") as stream:
        stream["data"] = np.arange(36, dtype=np.float32).reshape(3, 12)
        stream["sleep_stages"] = np.asarray([-1, 0, 1] * 4, dtype=np.int8)
        stream.attrs["original"] = "preserved"
    report = {
        "schema": "sleepkit.source_alignment/v1", "status": "passed", "issues": {},
        "raw_subjects_without_h5": {}, "parquet_sha256": "a" * 64,
        "h5_subjects": 1, "subjects_passed": 1,
        "subjects": {"a": {"status": "passed", "issues": {}, "h5_samples": 12, "raw_samples": 12,
                           "first_utc_seconds": 1583650780, "h5_sha256": sha256(root / "a.h5")}},
    }
    path = tmp_path / "alignment.json"
    path.write_text(json.dumps(report))
    return root, path, report, tmp_path / "clocked"


def test_copies_labels_and_attaches_clock_with_bound_provenance(evidence):
    import h5py

    root, path, report, destination = evidence
    digest = sha256(root / "a.h5")
    result = materialize(root, path, destination, subjects=["a"])
    assert sha256(root / "a.h5") == digest
    with h5py.File(root / "a.h5") as original, h5py.File(destination / "a.h5") as copied:
        assert "sample_time" not in original
        np.testing.assert_array_equal(copied["data"], original["data"])
        np.testing.assert_array_equal(copied["sleep_stages"], original["sleep_stages"])
        assert copied["sleep_stages"].dtype == original["sleep_stages"].dtype
        assert copied.attrs["original"] == "preserved"
        np.testing.assert_array_equal(copied["sample_time"], 1583650780 + np.arange(12) * 5)
        assert copied["sample_time"].dtype == np.dtype("int64")
        assert copied["sample_time"].attrs["units"] == "unix_seconds"
    assert result["alignment_sha256"] == sha256(path)
    assert result["subjects"]["a"]["output_h5_sha256"] == sha256(destination / "a.h5")
    assert (destination / "clock-source.sha256").read_text().split()[0] == sha256(destination / "clock-source.json")


@pytest.mark.parametrize("change", [
    {"status": "failed"}, {"issues": {"utc_cadence_mismatch": 1}},
    {"first_utc_seconds": None}, {"first_utc_seconds": True},
    {"first_utc_seconds": int(np.iinfo(np.int64).max) - 5},
    {"raw_samples": 11}, {"h5_samples": 0, "raw_samples": 0}, {"h5_sha256": "b" * 64},
])
def test_rejects_incomplete_or_mismatching_subject_evidence(evidence, change):
    root, path, report, destination = evidence
    report["subjects"]["a"].update(change)
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        materialize(root, path, destination, subjects=["a"])
    assert not destination.exists()


@pytest.mark.parametrize("change", [
    {"status": "failed"}, {"schema": "different"}, {"issues": {"row_count_mismatch": 1}},
    {"subjects_passed": 0}, {"parquet_sha256": None}, {"raw_subjects_without_h5": {"b": 12}},
])
def test_rejects_incomplete_aggregate_evidence(evidence, change):
    root, path, report, destination = evidence
    report.update(change)
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        materialize(root, path, destination, subjects=["a"])
    assert not destination.exists()


def test_existing_destination_preserved(evidence):
    root, path, _, destination = evidence
    destination.mkdir()
    with pytest.raises(FileExistsError):
        materialize(root, path, destination, subjects=["a"])
    assert list(destination.iterdir()) == []


def test_source_mutation_during_copy_cleans_stage(evidence, monkeypatch):
    from sleepkit.recipes.detection import clock_source

    root, path, _, destination = evidence
    copy = clock_source.shutil.copyfile

    def changed_copy(source, target):
        result = copy(source, target)
        with open(source, "ab") as stream:
            stream.write(b"changed")
        return result

    monkeypatch.setattr(clock_source.shutil, "copyfile", changed_copy)
    with pytest.raises(ValueError, match="changed while copying"):
        materialize(root, path, destination, subjects=["a"])
    assert not destination.exists()
    assert not list(destination.parent.glob(".clocked-*"))


@pytest.mark.parametrize("subjects", [[], ["a", "a"], ["../a"], "a", ["missing"]])
def test_subject_selection_is_explicit_and_valid(evidence, subjects):
    root, path, _, destination = evidence
    with pytest.raises(ValueError):
        materialize(root, path, destination, subjects=subjects)
    assert not destination.exists()


@pytest.mark.parametrize("start", [int(np.iinfo(np.int64).min), int(np.iinfo(np.int64).max) - 55])
def test_safe_int64_clock_boundaries(evidence, start):
    import h5py

    root, path, report, destination = evidence
    report["subjects"]["a"]["first_utc_seconds"] = start
    path.write_text(json.dumps(report))
    materialize(root, path, destination, subjects=["a"])
    with h5py.File(destination / "a.h5") as stream:
        assert stream["sample_time"][:].tolist() == [start + i * 5 for i in range(12)]


def test_rejects_existing_clock_and_removes_stage(evidence):
    import h5py

    root, path, report, destination = evidence
    with h5py.File(root / "a.h5", "r+") as stream:
        stream["sample_time"] = np.arange(12)
    report["subjects"]["a"]["h5_sha256"] = sha256(root / "a.h5")
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="existing sample clock"):
        materialize(root, path, destination, subjects=["a"])
    assert not destination.exists()
    assert not list(destination.parent.glob(".clocked-*"))


@pytest.mark.parametrize("storage", ["external_bytes", "external_link", "soft_link", "virtual"])
def test_indirect_label_storage_rejected_even_when_h5_hash_matches(evidence, storage):
    import h5py

    from sleepkit.recipes.detection.data import read_recording

    root, path, report, destination = evidence
    labels = np.asarray([-1, 0, 1] * 4, dtype=np.int8)
    external = root / "labels.h5"
    with h5py.File(external, "w") as stream:
        stream["labels"] = labels
    with h5py.File(root / "a.h5", "r+") as stream:
        del stream["sleep_stages"]
        if storage == "external_bytes":
            stream.create_dataset("sleep_stages", data=labels, external=[(str(root / "labels.bin"), 0, labels.nbytes)])
        elif storage == "external_link":
            stream["sleep_stages"] = h5py.ExternalLink(str(external), "labels")
        elif storage == "soft_link":
            stream["internal_labels"] = labels
            stream["sleep_stages"] = h5py.SoftLink("/internal_labels")
        else:
            layout = h5py.VirtualLayout(shape=(12,), dtype=np.int8)
            layout[:] = h5py.VirtualSource(str(external), "labels", shape=(12,))
            stream.create_virtual_dataset("sleep_stages", layout)
    original_hash = sha256(root / "a.h5")
    report["subjects"]["a"]["h5_sha256"] = original_hash
    path.write_text(json.dumps(report))
    if storage == "external_bytes":
        (root / "labels.bin").write_bytes(np.zeros(12, dtype=np.int8).tobytes())
        assert sha256(root / "a.h5") == original_hash
    with pytest.raises(ValueError, match="self-contained"):
        materialize(root, path, destination, subjects=["a"])
    with pytest.raises(ValueError, match="self-contained"):
        read_recording(root, "a", labels=False)
    assert not destination.exists()


def test_internal_hard_links_and_group_cycles_are_supported(evidence):
    import h5py

    from sleepkit.recipes.detection.hdf5 import require_self_contained

    root, path, report, destination = evidence
    with h5py.File(root / "a.h5", "r+") as stream:
        group = stream.create_group("metadata")
        group["self"] = group
        group["root"] = stream["/"]
        stream["labels_alias"] = stream["sleep_stages"]
        require_self_contained(stream)
    report["subjects"]["a"]["h5_sha256"] = sha256(root / "a.h5")
    path.write_text(json.dumps(report))
    materialize(root, path, destination, subjects=["a"])
    with h5py.File(destination / "a.h5") as stream:
        require_self_contained(stream)
        np.testing.assert_array_equal(stream["labels_alias"], stream["sleep_stages"])


def test_alignment_rejects_external_labels_before_accepting_evidence(evidence):
    import h5py

    pytest.importorskip("pyarrow")
    from sleepkit.recipes.detection.alignment import verify

    root, _, _, _ = evidence
    with h5py.File(root / "a.h5", "r+") as stream:
        del stream["sleep_stages"]
        stream.create_dataset("sleep_stages", data=np.zeros(12, dtype=np.int8),
                              external=[(str(root / "labels.bin"), 0, 12)])
    # The storage gate runs before raw-series parsing or emitting any evidence.
    raw = root / "raw.parquet"
    raw.write_bytes(b"")
    with pytest.raises(ValueError, match="self-contained"):
        verify(root, raw)

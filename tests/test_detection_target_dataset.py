"""Frozen target adapters preserve source sensors and ignore historical labels."""

import csv
from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pytest

from sleepkit.artifacts.package import sha256
from sleepkit.recipes.detection import candidate_audit, target_dataset
from sleepkit.recipes.detection.split import freeze
from sleepkit.recipes.detection.target_dataset import AnnotatedDataset


def make_dataset_fixture(tmp_path, *, context=2):
    """Create ten tiny series plus authentic coverage and frozen-protocol reports."""
    h5py = pytest.importorskip("h5py")
    root = tmp_path / "source"
    root.mkdir()
    count = 90
    origin = datetime(2020, 1, 1, tzinfo=timezone.utc)
    details = {}
    for i in range(10):
        subject = f"subject_{i}"
        path = root / f"{subject}.h5"
        data = np.vstack([np.arange(count) * 5, np.arange(count) * .01 + i, np.cos(np.arange(count))]).astype(np.float32)
        # Exercise a local clock transition while verified physical cadence remains 5s.
        data[0, 45:] += 3600
        with h5py.File(path, "w") as stream:
            stream["data"] = data
            stream["sleep_stages"] = "malformed historical labels must never be read"
        details[subject] = {
            "status": "passed", "issues": {}, "h5_samples": count, "raw_samples": count,
            "h5_sha256": sha256(path), "first_utc_seconds": int(origin.timestamp()),
        }
    events = tmp_path / "events.csv"
    with events.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["series_id", "night", "event", "step", "timestamp"])
        writer.writeheader()
        for subject in details:
            for night, onset, wakeup in [(1, 6, 30), (2, 54, 78)]:
                for event, step in [("onset", onset), ("wakeup", wakeup)]:
                    writer.writerow({"series_id": subject, "night": night, "event": event, "step": step,
                                     "timestamp": (origin + timedelta(seconds=step * 5)).isoformat()})
    alignment = tmp_path / "alignment.json"
    alignment.write_text(json.dumps({
        "schema": "sleepkit.source_alignment/v1", "status": "passed", "issues": {},
        "raw_subjects_without_h5": {}, "parquet_sha256": "a" * 64,
        "h5_subjects": len(details), "subjects_passed": len(details), "subjects": details,
        "event_clock": {"status": "passed_for_available_events", "issues": {}, "events_sha256": sha256(events),
                        "available_event_rows": 40, "matched_event_rows": 40},
    }))
    coverage = tmp_path / "coverage.json"
    coverage.write_text(json.dumps(candidate_audit.audit(root, events, alignment, context=context)))
    frozen = tmp_path / "frozen"
    freeze(root, coverage, frozen)
    return root, events, alignment, coverage, frozen


def test_read_reconstructs_target_clock_and_preserves_originals(tmp_path):
    paths = make_dataset_fixture(tmp_path)
    dataset = AnnotatedDataset(*paths)
    before = {p: sha256(p) for p in paths[0].glob("*.h5")}
    recording = dataset.read("subject_0")
    expected = np.full(90, -1, dtype=np.int8)
    expected[6:30], expected[54:78], expected[30:54] = 1, 1, 0
    np.testing.assert_array_equal(recording.labels, expected)
    np.testing.assert_array_equal(recording.sample_time, 1577836800 + np.arange(90) * 5)
    assert recording.sample_time.dtype == np.int64
    assert recording.data[0, 45] - recording.data[0, 44] == 3605
    assert dataset.split_path == paths[4] / "split.json"
    assert dataset.context == 2
    assert {p: sha256(p) for p in before} == before
    dataset.verify_unchanged()


def test_unlabelled_read_never_builds_candidates(tmp_path, monkeypatch):
    dataset = AnnotatedDataset(*make_dataset_fixture(tmp_path))

    def forbidden(*args, **kwargs):
        raise AssertionError("Candidate builder must not run for feature-only reads")

    monkeypatch.setattr(target_dataset, "build_candidates", forbidden)
    assert dataset.read("subject_0", labels=False).labels is None


def test_public_properties_are_copies_and_provenance_has_no_subjects(tmp_path):
    dataset = AnnotatedDataset(*make_dataset_fixture(tmp_path))
    serialized = json.dumps(dataset.provenance)
    assert not any(s in serialized for s in dataset.source_hashes)
    dataset.target["classes"].clear()
    dataset.split["train"].clear()
    dataset.source_hashes.clear()
    dataset.provenance["target"].clear()
    assert len(dataset.target["classes"]) == 2
    assert dataset.split["train"]
    assert dataset.source_hashes
    assert dataset.provenance["target"]


@pytest.mark.parametrize("subject", ["other", "../subject_0", None])
def test_unknown_subject_rejected(tmp_path, subject):
    dataset = AnnotatedDataset(*make_dataset_fixture(tmp_path))
    with pytest.raises(ValueError, match="cohort"):
        dataset.read(subject)


@pytest.mark.parametrize("index", [1, 2, 3])
def test_changed_evidence_rejected_before_and_after_construction(tmp_path, index):
    paths = make_dataset_fixture(tmp_path)
    dataset = AnnotatedDataset(*paths)
    paths[index].write_text(paths[index].read_text() + "\n")
    with pytest.raises(ValueError):
        AnnotatedDataset(*paths)
    with pytest.raises(ValueError, match="changed"):
        dataset.read("subject_0")
    with pytest.raises(ValueError, match="changed"):
        dataset.verify_unchanged()


def test_changed_source_rejected(tmp_path):
    paths = make_dataset_fixture(tmp_path)
    dataset = AnnotatedDataset(*paths)
    with (paths[0] / "subject_0.h5").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        dataset.read("subject_0")
    with pytest.raises(ValueError):
        AnnotatedDataset(*paths)


def test_mutation_during_read_fails(tmp_path, monkeypatch):
    paths = make_dataset_fixture(tmp_path)
    dataset = AnnotatedDataset(*paths)
    original = target_dataset.read_recording

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        with (paths[0] / "subject_0.h5").open("ab") as stream:
            stream.write(b"changed")
        return result

    monkeypatch.setattr(target_dataset, "read_recording", mutate)
    with pytest.raises(ValueError, match="during read"):
        dataset.read("subject_0")


@pytest.mark.parametrize("name", ["protocol", "split", "groups", "sources"])
def test_protocol_file_mutation_rejected(tmp_path, name):
    paths = make_dataset_fixture(tmp_path)
    dataset = AnnotatedDataset(*paths)
    path = paths[4] / f"{name}.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        dataset.verify_unchanged()
    if name != "protocol":
        with pytest.raises(ValueError):
            AnnotatedDataset(*paths)


@pytest.mark.parametrize("change", ["split", "groups", "context", "target", "counts", "cohort"])
def test_semantic_tampering_rejected_even_with_updated_file_hashes(tmp_path, change):
    paths = make_dataset_fixture(tmp_path)
    frozen = paths[4]
    protocol = json.loads((frozen / "protocol.json").read_text())
    if change in ("split", "groups"):
        path = frozen / f"{change}.json"
        value = json.loads(path.read_text())
        if change == "split":
            value["train"][0], value["test"][0] = value["test"][0], value["train"][0]
        else:
            split = json.loads((frozen / "split.json").read_text())
            value[split["train"][0]] = value[split["test"][0]]
        path.write_text(json.dumps(value))
        protocol["files"][path.name] = sha256(path)
    elif change == "context":
        protocol["context_policy"]["features_per_context"] = 3
    elif change == "target":
        protocol["target"]["classes"] = ["wake", "sleep"]
    elif change == "counts":
        protocol["partitions"]["train"]["counts"]["retained_sleep_targets"] += 1
    else:
        (paths[0] / "extra.h5").write_bytes((paths[0] / "subject_0.h5").read_bytes())
    (frozen / "protocol.json").write_text(json.dumps(protocol))
    with pytest.raises(ValueError):
        AnnotatedDataset(*paths)


def test_label_coverage_rechecked_when_reading(tmp_path, monkeypatch):
    dataset = AnnotatedDataset(*make_dataset_fixture(tmp_path))
    original = target_dataset.build_candidates

    def incorrect(*args, **kwargs):
        result = original(*args, **kwargs)
        result.labels[20] = 0
        return result

    monkeypatch.setattr(target_dataset, "build_candidates", incorrect)
    with pytest.raises(ValueError, match="Rebuilt targets"):
        dataset.read("subject_0")


def test_duplicate_alignment_json_keys_rejected(tmp_path):
    paths = make_dataset_fixture(tmp_path)
    report = paths[2].read_text()
    paths[2].write_text(report[:-1] + ', "status": "passed"}')
    with pytest.raises(ValueError, match="Duplicate"):
        AnnotatedDataset(*paths)

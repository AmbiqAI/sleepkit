"""Candidate coverage must preserve source positions and fail stale provenance."""

import csv
from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pytest

from sleepkit.artifacts.package import sha256
from sleepkit.recipes.detection import candidate_audit
from sleepkit.recipes.detection.preprocessing import contexts, extract


def fixture(tmp_path, *, missing_subject=False):
    h5py = pytest.importorskip("h5py")
    root = tmp_path / "data"
    root.mkdir()
    size = 90
    data = np.vstack([np.arange(size) * 5, np.ones(size), np.ones(size)]).astype(np.float32)
    origin = datetime(2020, 1, 1, tzinfo=timezone.utc)
    subjects = ["subject", "missing"] if missing_subject else ["subject"]
    details = {}
    for subject in subjects:
        path = root / f"{subject}.h5"
        with h5py.File(path, "w") as stream:
            stream["data"] = data
        details[subject] = {
            "status": "passed", "issues": {}, "h5_samples": size, "raw_samples": size,
            "h5_sha256": sha256(path), "first_utc_seconds": int(origin.timestamp()),
        }
    events = tmp_path / "events.csv"
    with events.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["series_id", "night", "event", "step", "timestamp"])
        writer.writeheader()
        for night, onset, wakeup in [(1, 6, 30), (2, 54, 78)]:
            for event, step in [("onset", onset), ("wakeup", wakeup)]:
                writer.writerow({
                    "series_id": "subject", "night": night, "event": event, "step": step,
                    "timestamp": (origin + timedelta(seconds=step * 5)).isoformat(),
                })
    report = {
        "schema": "sleepkit.source_alignment/v1", "status": "passed", "issues": {},
        "raw_subjects_without_h5": {}, "parquet_sha256": "a" * 64,
        "h5_subjects": len(subjects), "subjects_passed": len(subjects), "subjects": details,
        "event_clock": {
            "status": "passed_for_available_events", "issues": {}, "events_sha256": sha256(events),
            "available_event_rows": 4, "matched_event_rows": 4,
        },
    }
    alignment = tmp_path / "alignment.json"
    alignment.write_text(json.dumps(report))
    return root, events, alignment


def test_report_counts_and_missing_events(tmp_path):
    paths = fixture(tmp_path, missing_subject=True)
    report = candidate_audit.audit(*paths, context=2)
    assert report["counts"]["samples"] == 180
    assert report["counts"]["sleep_candidate_samples"] == 48
    assert report["counts"]["wake_candidate_samples"] == 24
    assert report["counts"]["unknown_samples"] == 108
    assert report["h5_without_events"] == 1
    assert report["subjects"]["missing"]["counts"]["retained_contexts"] == 0
    assert json.loads(json.dumps(report)) == report


@pytest.mark.parametrize("change", ["status", "events_hash", "source_hash", "origin", "count", "event_count"])
def test_reject_invalid_evidence(tmp_path, change):
    paths = fixture(tmp_path)
    report = json.loads(paths[2].read_text())
    if change == "status":
        report["status"] = "failed"
    elif change == "events_hash":
        report["event_clock"]["events_sha256"] = "b" * 64
    elif change == "source_hash":
        report["subjects"]["subject"]["h5_sha256"] = "b" * 64
    elif change == "origin":
        report["subjects"]["subject"]["first_utc_seconds"] = True
    elif change == "count":
        report["subjects"]["subject"]["raw_samples"] = 91
    else:
        report["event_clock"]["matched_event_rows"] = 3
    paths[2].write_text(json.dumps(report))
    with pytest.raises(ValueError):
        candidate_audit.audit(*paths)


def test_changed_evidence_during_read_fails(tmp_path, monkeypatch):
    paths = fixture(tmp_path)
    original = candidate_audit.build_candidates

    def mutate(*args, **kwargs):
        paths[2].write_text(paths[2].read_text() + "\n")
        return original(*args, **kwargs)

    monkeypatch.setattr(candidate_audit, "build_candidates", mutate)
    with pytest.raises(ValueError, match="changed during"):
        candidate_audit.audit(*paths)


def test_external_hdf5_data_rejected_even_with_matching_hash(tmp_path):
    h5py = pytest.importorskip("h5py")
    paths = fixture(tmp_path)
    subject = paths[0] / "subject.h5"
    with h5py.File(tmp_path / "external.h5", "w") as stream:
        stream["data"] = np.zeros((3, 90))
    with h5py.File(subject, "w") as stream:
        stream["data"] = h5py.ExternalLink(str(tmp_path / "external.h5"), "data")
    report = json.loads(paths[2].read_text())
    report["subjects"]["subject"]["h5_sha256"] = sha256(subject)
    paths[2].write_text(json.dumps(report))
    with pytest.raises(ValueError):
        candidate_audit.audit(*paths)


def test_context_counts_do_not_bridge_excluded_frames():
    data = np.vstack([np.arange(54) * 5, np.ones(54), np.ones(54)]).astype(np.float32)
    labels = np.zeros(54, dtype=np.int8)
    labels[23] = -1
    data[1, 25] = np.nan
    actual = candidate_audit.context_counts(data, labels, 2)
    assert actual["complete_contexts"] == 4
    assert actual["nonfinite_sensor_contexts"] == 2
    assert actual["unknown_target_contexts"] == 1
    assert actual["nonfinite_and_unknown_contexts"] == 1
    assert actual["retained_contexts"] == 2
    expected = list(contexts(extract(data), 2, labels))
    assert actual["retained_contexts"] == len(expected)
    assert actual["retained_wake_targets"] == sum(len(y) for _, y, _ in expected)


@pytest.mark.parametrize("samples", [0, 11, 12, 17, 18, 23, 24, 29, 30, 31])
def test_window_and_context_boundaries(samples):
    data = np.vstack([np.arange(samples) * 5, np.ones(samples), np.ones(samples)]).astype(np.float32)
    labels = np.ones(samples, dtype=np.int8)
    actual = candidate_audit.context_counts(data, labels, 3)
    features = extract(data)
    assert actual["feature_windows"] == len(features.ends)
    assert actual["retained_contexts"] == len(list(contexts(features, 3, labels)))
    assert actual["incomplete_context_feature_windows"] == len(features.ends) % 3


@pytest.mark.parametrize("context", [True, 0, -1, 1.5])
def test_invalid_context(context):
    with pytest.raises(ValueError, match="positive integer"):
        candidate_audit.context_counts(np.zeros((3, 20)), np.zeros(20, dtype=np.int8), context)

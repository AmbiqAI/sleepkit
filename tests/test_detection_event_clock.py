"""Annotation UTC checks do not mistake consistent pair shifts for valid clocks."""

import csv
from datetime import datetime, timedelta
import json
import sys

import pytest

from sleepkit.recipes.detection.alignment import main, verify
from tests.test_detection_alignment import source


def events(tmp_path, rows):
    path = tmp_path / "events.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["series_id", "night", "event", "step", "timestamp"])
        writer.writeheader()
        for i, values in enumerate(rows):
            writer.writerow({"series_id": "a", "night": "1", "event": "onset" if i == 0 else "wakeup", **values})
    return path


def pair(path):
    import pyarrow.parquet as pq

    timestamps = pq.read_table(path)["timestamp"].to_pylist()
    return [{"step": str(i), "timestamp": timestamps[i]} for i in (2, 8)]


def test_exact_utc_match_across_timezone_transition_and_batches(tmp_path):
    path = source(tmp_path)
    rows = pair(path)
    event_path = events(tmp_path, rows + [{"step": "", "timestamp": ""}])
    report = verify(tmp_path, path, events_path=event_path, batch_size=3)
    assert report["status"] == "passed"
    clock = report["event_clock"]
    assert clock["status"] == "passed_for_available_events"
    assert clock["event_rows"] == 3
    assert clock["matched_event_rows"] == clock["available_event_rows"] == 2
    assert clock["missing_step_rows"] == clock["missing_timestamp_rows"] == 1
    assert len(clock["events_sha256"]) == 64
    json.dumps(report, allow_nan=False)


def test_equal_shift_of_both_pair_timestamps_fails(tmp_path):
    path = source(tmp_path)
    rows = pair(path)
    for row in rows:
        row["timestamp"] = (datetime.fromisoformat(row["timestamp"]) + timedelta(hours=1)).isoformat()
    report = verify(tmp_path, path, events_path=events(tmp_path, rows), batch_size=3)
    assert report["status"] == "failed"
    assert report["event_clock"]["issues"] == {"timestamp_mismatch_rows": 2}
    assert report["issues"] == {}


@pytest.mark.parametrize("change, issue", [
    ({"step": "NaN"}, "invalid_step_rows"),
    ({"step": "1.2"}, "invalid_step_rows"),
    ({"step": "-1"}, "invalid_step_rows"),
    ({"step": "12"}, "out_of_range_step_rows"),
    ({"series_id": "unknown"}, "unknown_subject_rows"),
    ({"timestamp": "2020-03-08T06:59:50"}, "invalid_timestamp_rows"),
    ({"timestamp": "garbage", "step": ""}, "invalid_timestamp_rows"),
    ({"timestamp": "", "step": "invalid"}, "invalid_step_rows"),
    ({"timestamp": "2020-03-08T06:59:50.000001+00:00"}, "timestamp_mismatch_rows"),
    ({"timestamp": "2020-03-08T06:59:50.0000001+00:00"}, "invalid_timestamp_rows"),
    ({"timestamp": "2020-03-08T06:59:50+00:00:00.0000001"}, "invalid_timestamp_rows"),
])
def test_invalid_or_mismatched_available_values_fail(tmp_path, change, issue):
    path = source(tmp_path)
    rows = pair(path)
    rows[0].update(change)
    report = verify(tmp_path, path, events_path=events(tmp_path, rows), batch_size=3)
    assert report["status"] == "failed"
    assert report["event_clock"]["issues"][issue] == 1


def test_missing_only_annotations_do_not_claim_verification(tmp_path):
    path = source(tmp_path)
    report = verify(tmp_path, path, events_path=events(tmp_path, [{"step": "", "timestamp": ""}]))
    assert report["event_clock"]["status"] == "failed"
    assert report["event_clock"]["matched_event_rows"] == 0
    assert report["event_clock"]["issues"] == {"no_available_events": 1}


def test_event_without_source_sample_is_not_matched(tmp_path):
    path = source(tmp_path)
    event_path = events(tmp_path, pair(path))
    import pyarrow.parquet as pq

    pq.write_table(pq.read_table(path).slice(0, 8), path)
    report = verify(tmp_path, path, events_path=event_path)
    assert report["status"] == "failed"
    assert report["event_clock"]["issues"] == {"unmatched_event_rows": 1}


def test_cli_event_option_and_output(tmp_path, monkeypatch, capsys):
    path = source(tmp_path)
    event_path = events(tmp_path, pair(path))
    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["alignment", "--data", str(tmp_path), "--parquet", str(path),
                                     "--events", str(event_path), "--output", str(output)])
    main()
    report = json.loads(output.read_text())
    assert report["event_clock"]["matched_event_rows"] == 2
    assert json.loads(capsys.readouterr().out)["event_clock"] == report["event_clock"]


def test_duplicate_source_step_cannot_count_as_verified_event(tmp_path):
    path = source(tmp_path)
    event_path = events(tmp_path, pair(path))
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    steps = np.arange(12)
    steps[3] = 2
    pq.write_table(table.set_column(table.schema.get_field_index("step"), "step", pa.array(steps)), path)
    report = verify(tmp_path, path, events_path=event_path, batch_size=3)
    assert report["status"] == "failed"
    assert report["event_clock"]["issues"]["ambiguous_source_step_rows"] == 1
    assert report["event_clock"]["matched_event_rows"] == 1


def test_events_hash_is_rechecked_after_streaming(tmp_path, monkeypatch):
    from sleepkit.recipes.detection import alignment

    path = source(tmp_path)
    event_path = events(tmp_path, pair(path))
    original = alignment.sha256
    calls = 0

    def changed_hash(candidate):
        nonlocal calls
        if candidate == event_path:
            calls += 1
            if calls == 2:
                return "changed"
        return original(candidate)

    monkeypatch.setattr(alignment, "sha256", changed_hash)
    with pytest.raises(ValueError, match="Events CSV changed"):
        verify(tmp_path, path, events_path=event_path)
    assert calls == 2

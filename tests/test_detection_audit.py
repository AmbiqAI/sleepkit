"""Annotation ambiguity must remain visible rather than becoming negative labels."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from sleepkit.recipes.detection.audit import audit, compare_labels, inspect_clock, inspect_nights


def row(event, step):
    ts = datetime(2020, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=step * 5)
    return {"event": event, "step": str(step), "timestamp": ts.isoformat()}


def test_pair_by_night_not_csv_order():
    nights = {"2": [row("wakeup", 50), row("onset", 30)], "1": [row("onset", 5), row("wakeup", 20)]}
    intervals, issues = inspect_nights(nights, 100)
    assert intervals == [(5, 20), (30, 50)]
    assert issues == {}


@pytest.mark.parametrize("kind", ["missing", "duplicate", "reverse", "bounds", "timestamp", "fractional"])
def test_bad_nights_excluded_with_reasons(kind):
    rows = [row("onset", 5), row("wakeup", 20)]
    if kind == "missing":
        rows[0]["step"] = ""
    elif kind == "duplicate":
        rows.append(row("onset", 6))
    elif kind == "reverse":
        rows = [row("onset", 20), row("wakeup", 5)]
    elif kind == "bounds":
        rows[1] = row("wakeup", 101)
    elif kind == "timestamp":
        rows[1]["timestamp"] = row("wakeup", 25)["timestamp"]
    else:
        rows[0]["step"] = "5.5"
    intervals, issues = inspect_nights({"1": rows}, 100)
    assert intervals == []
    assert sum(issues.values()) == 1


def test_nested_overlaps_exclude_all_conflicting_nights():
    nights = {
        str(i): [row("onset", start), row("wakeup", stop)]
        for i, (start, stop) in enumerate([(1, 60), (5, 10), (20, 30), (70, 80)])
    }
    intervals, issues = inspect_nights(nights, 100)
    assert intervals == [(70, 80)]
    assert issues == {"overlapping_nights": 3}


def test_boundaries_and_disagreements_do_not_change_labels():
    labels = np.array([0, 0, 1, -1, 1, 0, 1])
    original = labels.copy()
    counts = compare_labels(labels, [(1, 5)])
    assert counts["paired_sleep_candidate_samples"] == 4
    assert counts["wake_inside_paired_sleep"] == 1
    assert counts["unknown_inside_paired_sleep"] == 1
    assert counts["sleep_outside_paired_sleep"] == 1
    assert counts["wake_outside_paired_sleep"] == 2
    np.testing.assert_array_equal(labels, original)


def test_audit_preserves_sources_and_missing_subjects(tmp_path):
    h5py = pytest.importorskip("h5py")
    import csv
    from sleepkit.artifacts.package import sha256

    path = tmp_path / "a.h5"
    with h5py.File(path, "w") as stream:
        stream["data"] = np.zeros((3, 30))
        stream["sleep_stages"] = np.zeros(30, dtype=np.int32)
    events = tmp_path / "events.csv"
    with events.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["series_id", "night", "event", "step", "timestamp"])
        writer.writeheader()
        for event in [row("onset", 5), row("wakeup", 20)]:
            writer.writerow({"series_id": "a", "night": "1", **event})
        writer.writerow({"series_id": "absent", "night": "1", **row("onset", 1)})
    before = [sha256(path), sha256(events)]
    result = audit(tmp_path, events)
    assert result["events_without_h5"] == 1
    assert result["counts"]["wake_inside_paired_sleep"] == 15
    assert result["counts"]["wake_outside_paired_sleep"] == 15
    assert [sha256(path), sha256(events)] == before


@pytest.mark.parametrize("attribute,value", [("sample_rate_hz", 0.25), ("channel_names", ["TS", "ZANGLE", "ENMO"])])
def test_audit_rejects_conflicting_source_metadata(tmp_path, attribute, value):
    h5py = pytest.importorskip("h5py")
    with h5py.File(tmp_path / "a.h5", "w") as stream:
        stream["data"] = np.zeros((3, 30))
        stream["sleep_stages"] = np.zeros(30, dtype=np.int32)
        stream.attrs[attribute] = value
    events = tmp_path / "events.csv"
    events.write_text("series_id,night,event,step,timestamp\na,1,onset,,\n")
    with pytest.raises(ValueError, match="Audit requires"):
        audit(tmp_path, events)


def test_clock_audit_distinguishes_midnight_from_clock_shift():
    normal = (86390 + np.arange(8) * 5) % 86400
    assert inspect_clock(normal)["non_nominal_transitions"] == 0
    shifted = normal.copy()
    shifted[4:] = (shifted[4:] + 3600) % 86400
    assert inspect_clock(shifted)["transition_deltas_seconds"] == {"3605.0": 1}
    shifted[4:] = (normal[4:] - 3600) % 86400
    assert inspect_clock(shifted)["transition_deltas_seconds"] == {"82805.0": 1}

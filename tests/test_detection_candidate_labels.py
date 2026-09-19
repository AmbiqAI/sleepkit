"""Boundary and adversarial cases for inactive candidate label policy."""

from datetime import datetime, timedelta, timezone
import itertools

import numpy as np
import pytest

from sleepkit.recipes.detection.candidate_labels import build_candidates

ORIGIN = 1_600_000_000


def event(kind, source_step, **changes):
    row = {"event": kind, "step": str(source_step) if source_step is not None else "", "timestamp": ""}
    if source_step is not None:
        row["timestamp"] = datetime.fromtimestamp(ORIGIN + source_step * 5, timezone.utc).isoformat()
    return {**row, **changes}


def pair(start, stop):
    return [event("onset", start), event("wakeup", stop)]


def build(nights, count=60):
    return build_candidates(nights, count, first_utc_seconds=ORIGIN)


def test_half_open_intervals_unknown_edges_and_exact_class_counts():
    result = build({"1": pair(10, 20), "2": pair(30, 40)})
    assert result.sleep_intervals == ((10, 20), (30, 40))
    assert result.wake_intervals == ((20, 30),)
    assert result.valid_nights == (1, 2)
    np.testing.assert_array_equal(result.labels, [-1] * 10 + [1] * 10 + [0] * 10 + [1] * 10 + [-1] * 20)
    assert result.labels.dtype == np.int8


def test_shuffling_groups_and_rows_does_not_change_masks():
    nights = {"1": pair(10, 20), "2": pair(30, 40), "3": pair(None, None)}
    expected = build(nights)
    for keys in itertools.permutations(nights):
        actual = build({key: nights[key][::-1] for key in keys})
        np.testing.assert_array_equal(actual.labels, expected.labels)
        assert actual.issues == expected.issues


@pytest.mark.parametrize("middle", [None, pair(None, None), [event("onset", None)]])
def test_skipped_or_missing_night_retains_sleep_but_no_wake(middle):
    nights = {"1": pair(10, 20), "3": pair(30, 40)}
    if middle is not None:
        nights["2"] = middle
    result = build(nights)
    assert result.valid_nights == (1, 3)
    assert not result.wake_intervals
    assert (result.labels[20:30] == -1).all()


@pytest.mark.parametrize("bad_id", ["01", "0", "-1", "1.0", "x", " 1", 1])
def test_noncanonical_night_id_invalidates_series(bad_id):
    result = build({"1": pair(10, 20), bad_id: pair(None, None)})
    assert (result.labels == -1).all()
    assert result.issues["series_invalidated"] == 1


@pytest.mark.parametrize(
    "rows",
    [
        [event("onset", 15)],
        [event("onset", 15, timestamp="bad"), event("wakeup", None)],
        [event("onset", 15), event("onset", 15), event("wakeup", 25)],
        [event("alien", 15)],
    ],
)
def test_incomplete_or_invalid_event_inside_sleep_excludes_candidate(rows):
    result = build({"1": pair(10, 20), "2": rows})
    assert not result.valid_nights
    assert (result.labels == -1).all()


def test_incomplete_event_inside_wake_cannot_disappear():
    result = build({"1": pair(10, 20), "2": pair(30, 40), "3": [event("onset", 25)]})
    assert result.valid_nights == (1,)
    assert not result.wake_intervals
    assert (result.labels[20:] == -1).all()


@pytest.mark.parametrize(
    "nights",
    [
        {"1": pair(10, 30), "2": pair(20, 40)},
        {"1": pair(30, 40), "2": pair(10, 20)},
        {"1": pair(10, 50), "2": pair(20, 30), "3": pair(35, 40)},
    ],
)
def test_all_members_of_conflicts_are_excluded(nights):
    result = build(nights)
    assert not result.valid_nights
    assert (result.labels == -1).all()


@pytest.mark.parametrize("bad_step", ["60", "-1", "1.5", "NaN", "Infinity", "garbage"])
def test_unlocatable_nonmissing_steps_fail_closed(bad_step):
    result = build({"1": pair(10, 20), "2": [event("onset", 30, step=bad_step)]})
    assert (result.labels == -1).all()
    assert result.issues["series_invalidated"] == 1


@pytest.mark.parametrize("fraction", [".000001", ".0000001"])
def test_fractional_timestamps_are_not_rounded_to_raw_seconds(fraction):
    rows = pair(10, 20)
    rows[0]["timestamp"] = rows[0]["timestamp"].replace("+00:00", fraction + "+00:00")
    result = build({"1": rows})
    assert not result.valid_nights


def test_equal_timestamp_shift_fails_even_when_duration_matches():
    rows = pair(10, 20)
    for row in rows:
        row["timestamp"] = (datetime.fromisoformat(row["timestamp"]) + timedelta(hours=1)).isoformat()
    assert not build({"1": rows}).valid_nights


def test_timezone_offset_change_preserves_exact_utc_event_alignment():
    rows = pair(10, 20)
    for row, hours in zip(rows, (-5, -4)):
        row["timestamp"] = (
            datetime.fromisoformat(row["timestamp"]).astimezone(timezone(timedelta(hours=hours))).isoformat()
        )
    result = build({"1": rows})
    assert result.sleep_intervals == ((10, 20),)


def test_touching_sleep_intervals_have_no_empty_wake_interval():
    result = build({"1": pair(10, 20), "2": pair(20, 30)})
    assert result.valid_nights == (1, 2)
    assert not result.wake_intervals
    assert (result.labels[10:30] == 1).all()


def test_recording_without_events_is_unknown():
    assert (build({}).labels == -1).all()


def test_complete_wakeup_must_be_existing_sample():
    result = build({"1": pair(10, 60)})
    assert not result.valid_nights


@pytest.mark.parametrize("count, origin", [(0, ORIGIN), (1.5, ORIGIN), (60, 1.5), (60, int(np.iinfo(np.int64).max))])
def test_invalid_source_clock(count, origin):
    with pytest.raises(ValueError):
        build_candidates({}, count, first_utc_seconds=origin)


def test_timestamp_without_step_does_not_disappear_from_conflict_evidence():
    rows = [event("onset", 15, step=""), event("wakeup", None)]
    result = build({"1": pair(10, 20), "2": pair(30, 40), "3": rows})
    assert result.issues["series_invalidated"] == 1
    assert (result.labels == -1).all()


def test_contradictory_clock_coordinate_does_not_leave_other_nights_supported():
    row = event("onset", 15, step="50")
    result = build({"1": pair(10, 20), "2": pair(30, 40), "3": [row]})
    assert result.issues["series_invalidated"] == 1
    assert (result.labels == -1).all()

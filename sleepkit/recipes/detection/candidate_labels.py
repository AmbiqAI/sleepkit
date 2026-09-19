"""Pure, versioned event-supported candidates; never clinical ground truth by default."""

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import re

import numpy as np

POLICY = {
    "kind": "sleepkit.event_candidates/v1",
    "labels": {"unknown": -1, "wake_candidate": 0, "sleep_candidate": 1},
    "sleep": "Complete clock-aligned onset/wakeup pair, [onset, wakeup)",
    "wake": "Between eligible consecutive integer nights, [wakeup_n, onset_n+1)",
    "conflicts": "Exclude all participants in overlaps or night-order inversions, including located incomplete evidence",
    "unlocatable": "Noncanonical IDs, malformed nonmissing coordinates, timestamp-only or contradictory clock evidence invalidate the series",
    "status": "Candidate interpretation only; annotation semantics require separate review",
}


@dataclass(frozen=True)
class CandidateLabels:
    labels: np.ndarray
    sleep_intervals: tuple
    wake_intervals: tuple
    valid_nights: tuple
    issues: dict


def _step(value, count):
    if value is None or str(value).strip() == "":
        return None
    number = Decimal(str(value))
    if not number.is_finite() or number != number.to_integral_value() or not 0 <= number < count:
        raise ValueError("Step must identify an existing source sample")
    return int(number)


def _utc(value):
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip()
    # Restrict precision before datetime can truncate fractional seconds.
    if not re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?(?:Z|[+-][0-9]{2}:?[0-9]{2})", text
    ):
        raise ValueError("Expected timezone-aware ISO event seconds")
    moment = datetime.fromisoformat(text).astimezone(timezone.utc)
    delta = moment - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds


def build_candidates(nights, sample_count, *, first_utc_seconds):
    """Build candidates from one series' raw CSV groups and independently verified clock.

    The caller must establish step/index alignment and five-second UTC continuity.
    This function rechecks event timestamps exactly but cannot authenticate source
    provenance. No source files, labels, or fitted model state are accessed.
    """
    limit = np.iinfo(np.int64)
    if type(sample_count) is not int or sample_count < 1:
        raise ValueError("sample_count must be a positive integer")
    if (
        type(first_utc_seconds) is not int
        or not limit.min <= first_utc_seconds <= first_utc_seconds + 5 * (sample_count - 1) <= limit.max
    ):
        raise ValueError("first_utc_seconds must define a nonoverflowing int64 source clock")
    labels = np.full(sample_count, -1, dtype=np.int8)
    issues = Counter()
    if any(not isinstance(key, str) or not re.fullmatch(r"[1-9][0-9]*", key) for key in nights):
        return CandidateLabels(labels, (), (), (), {"noncanonical_night_id": 1, "series_invalidated": 1})
    pairs, bounds = {}, {}
    unsafe_evidence = False
    for key, rows in nights.items():
        night = int(key)
        positions, parsed = [], {}
        missing, malformed = False, False
        kinds = Counter(row.get("event") for row in rows)
        for row in rows:
            try:
                step = _step(row.get("step"), sample_count)
            except (ValueError, InvalidOperation, OverflowError):
                unsafe_evidence = True
                malformed = True
                continue
            if step is not None:
                positions.append(step)
            try:
                micros = _utc(row.get("timestamp"))
            except (ValueError, OverflowError):
                unsafe_evidence = True
                malformed = True
                continue
            if step is None and micros is not None:
                unsafe_evidence = True
                issues["timestamp_without_step"] += 1
            if step is None or micros is None:
                missing = True
                continue
            if micros != (first_utc_seconds + step * 5) * 1_000_000:
                unsafe_evidence = True
                malformed = True
                continue
            parsed[row.get("event")] = step
        if positions:
            bounds[night] = (min(positions), max(positions))
        complete = kinds == {"onset": 1, "wakeup": 1}
        if not complete:
            issues["duplicate_or_unpaired_events"] += 1
        if missing:
            issues["missing_step_or_timestamp"] += 1
        if malformed:
            issues["invalid_event_value_or_clock"] += 1
        if not complete or missing or malformed:
            continue
        if parsed["onset"] >= parsed["wakeup"]:
            issues["reversed_or_empty_pair"] += 1
            continue
        pairs[night] = (parsed["onset"], parsed["wakeup"])
    if unsafe_evidence:
        issues["series_invalidated"] += 1
        return CandidateLabels(labels, (), (), (), dict(issues))
    # Include all located evidence, even incomplete or clock-invalid groups.
    # Detect every conflict before removing any pair, so results are order-invariant.
    conflicted = set()
    ordered = sorted(bounds)
    for i, left in enumerate(ordered):
        for right in ordered[i + 1 :]:
            if bounds[left][1] > bounds[right][0]:
                conflicted.update((left, right))
    invalid_bounds = [bounds[n] for n in bounds if n not in pairs]
    for night, (start, stop) in pairs.items():
        if any(left <= stop and right >= start for left, right in invalid_bounds):
            conflicted.add(night)
    if conflicted:
        issues["conflicting_nights"] = len(conflicted)
    eligible = {n: interval for n, interval in pairs.items() if n not in conflicted}
    rejected_bounds = [bounds[n] for n in bounds if n not in eligible]
    sleep = tuple(eligible[n] for n in sorted(eligible))
    wake = []
    for n in sorted(eligible):
        if n + 1 not in eligible:
            continue
        start, stop = eligible[n][1], eligible[n + 1][0]
        if start >= stop or any(left <= stop and right >= start for left, right in rejected_bounds):
            continue
        wake.append((start, stop))
    for start, stop in sleep:
        labels[start:stop] = 1
    for start, stop in wake:
        labels[start:stop] = 0
    return CandidateLabels(labels, sleep, tuple(wake), tuple(sorted(eligible)), dict(issues))

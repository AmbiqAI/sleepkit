"""Read-only annotation audit; this does not construct evaluation ground truth."""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from sleepkit.artifacts.package import sha256


def load_events(path):
    """Retain every row so missing/duplicate event pairs cannot disappear silently."""
    subjects = defaultdict(lambda: defaultdict(list))
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"series_id", "night", "event", "step", "timestamp"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("Events CSV must contain series_id, night, event, step, timestamp")
        for row in reader:
            if not row["series_id"] or not row["night"]:
                raise ValueError("Events need a subject and night identifier")
            subjects[row["series_id"]][row["night"]].append(row)
    if not subjects:
        raise ValueError("Events CSV is empty")
    return subjects


def _event(row):
    step = float(row["step"])
    timestamp = datetime.fromisoformat(row["timestamp"])
    if not np.isfinite(step) or step < 0 or not step.is_integer() or timestamp.utcoffset() is None:
        raise ValueError("Events need nonnegative integer steps and timezone-aware timestamps")
    return int(step), timestamp


def inspect_nights(nights, sample_count):
    """Return unambiguous paired sleep candidates in source-step coordinates.

    Mapping source steps to HDF5 indices is an explicit, unverified assumption.
    Conflicting intervals are excluded rather than silently merged into truth.
    """
    issues = Counter()
    intervals = []
    for rows in nights.values():
        counts = Counter(row["event"] for row in rows)
        if counts != {"onset": 1, "wakeup": 1}:
            issues["duplicate_or_unpaired_events"] += 1
            continue
        if any(not row["step"] or not row["timestamp"] for row in rows):
            issues["missing_step_or_timestamp"] += 1
            continue
        try:
            events = {row["event"]: _event(row) for row in rows}
        except (ValueError, OverflowError):
            issues["invalid_event_value"] += 1
            continue
        start, start_time = events["onset"]
        stop, stop_time = events["wakeup"]
        if not 0 <= start < stop <= sample_count:
            issues["out_of_bounds_or_reversed"] += 1
            continue
        if (stop_time - start_time).total_seconds() != (stop - start) * 5:
            issues["timestamp_step_duration_mismatch"] += 1
            continue
        intervals.append((start, stop))
    intervals.sort()
    # Mark every member of an overlapping cluster, including nested intervals.
    conflicted = set()
    active = []
    for i, (start, stop) in enumerate(intervals):
        active = [(j, right) for j, right in active if right > start]
        if active:
            conflicted.add(i)
            conflicted.update(j for j, _ in active)
        active.append((i, stop))
    if conflicted:
        issues["overlapping_nights"] = len(conflicted)
    return [interval for i, interval in enumerate(intervals) if i not in conflicted], dict(issues)


def compare_labels(labels, intervals):
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer) or not np.isin(labels, [-1, 0, 1]).all():
        raise ValueError("Labels must be an integer vector containing -1, 0, or 1")
    candidate = np.zeros(len(labels), dtype=bool)
    for start, stop in intervals:
        candidate[start:stop] = True
    return {
        "samples": len(labels),
        "paired_sleep_candidate_samples": int(candidate.sum()),
        "wake_inside_paired_sleep": int(((labels == 0) & candidate).sum()),
        "unknown_inside_paired_sleep": int(((labels == -1) & candidate).sum()),
        "sleep_outside_paired_sleep": int(((labels == 1) & ~candidate).sum()),
        "wake_outside_paired_sleep": int(((labels == 0) & ~candidate).sum()),
        "unknown_outside_paired_sleep": int(((labels == -1) & ~candidate).sum()),
    }


def inspect_clock(times):
    """Report local-clock discontinuities; do not infer physical sample gaps."""
    finite = np.isfinite(times)
    adjacent = finite[1:] & finite[:-1]
    deltas = np.mod(np.diff(times.astype(np.float64))[adjacent], 86400)
    non_nominal = deltas[~np.isclose(deltas, 5, rtol=0, atol=1e-3)]
    values, counts = np.unique(non_nominal, return_counts=True)
    return {
        "nonfinite_samples": int((~finite).sum()),
        "out_of_range_samples": int(((times[finite] < 0) | (times[finite] >= 86400)).sum()),
        "non_nominal_transitions": int(len(non_nominal)),
        "transition_deltas_seconds": {str(float(value)): int(count) for value, count in zip(values, counts)},
    }


def audit(data_root, events_path):
    """Inspect local source labels without opening files for writing or training a model."""
    import h5py

    events_path = Path(events_path)
    events_hash = sha256(events_path)
    subjects = load_events(events_path)
    files = {p.stem: p for p in Path(data_root).glob("*.h5")}
    if not files:
        raise ValueError("No subject HDF5 files found")
    summary, issues = Counter(), Counter()
    details = {}
    for subject, path in sorted(files.items()):
        before = sha256(path)
        with h5py.File(path, "r") as stream:
            if not np.isclose(float(stream.attrs.get("sample_rate_hz", 0.2)), 0.2, rtol=0, atol=1e-8):
                raise ValueError("Audit requires 0.2 Hz CMIDSS HDF5 data")
            if "channel_names" in stream.attrs:
                channels = [v.decode() if isinstance(v, bytes) else str(v) for v in stream.attrs["channel_names"]]
                if channels != ["TS", "ENMO", "ZANGLE"]:
                    raise ValueError("Audit requires CMIDSS channel order")
            shape = stream["data"].shape
            labels = np.asarray(stream["sleep_stages"])
            if len(shape) != 2 or shape[0] != 3 or labels.shape != (shape[1],):
                raise ValueError(f"Invalid CMIDSS HDF5 shape: {path.name}")
            clock = inspect_clock(np.asarray(stream["data"][0]))
        nights = subjects.get(subject, {})
        intervals, subject_issues = inspect_nights(nights, len(labels))
        counts = compare_labels(labels, intervals)
        summary.update(counts)
        issues.update(subject_issues)
        if sha256(path) != before:
            raise ValueError("HDF5 changed during audit")
        details[subject] = {
            "source_sha256": before,
            "clock": clock,
            "nights": len(nights),
            "paired_sleep_candidates": len(intervals),
            "issues": subject_issues,
            "counts": counts,
            "labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
        }
    if sha256(events_path) != events_hash:
        raise ValueError("Events CSV changed during audit")
    clock_deltas = Counter()
    for detail in details.values():
        clock_deltas.update(detail["clock"]["transition_deltas_seconds"])
    return {
        "schema": "sleepkit.annotation_audit/v1",
        "clock": {
            "subjects_with_non_nominal_transitions": sum(
                d["clock"]["non_nominal_transitions"] > 0 for d in details.values()
            ),
            "nonfinite_samples": sum(d["clock"]["nonfinite_samples"] for d in details.values()),
            "out_of_range_samples": sum(d["clock"]["out_of_range_samples"] for d in details.values()),
            "transition_deltas_seconds": dict(clock_deltas),
        },
        "events_sha256": events_hash,
        "source_subjects": len(files),
        "event_subjects": len(subjects),
        "h5_without_events": len(set(files) - set(subjects)),
        "events_without_h5": len(set(subjects) - set(files)),
        "event_rows": sum(len(rows) for nights in subjects.values() for rows in nights.values()),
        "missing_step_rows": sum(
            not row["step"] for nights in subjects.values() for rows in nights.values() for row in rows
        ),
        "paired_sleep_candidates": sum(d["paired_sleep_candidates"] for d in details.values()),
        "issue_nights": dict(issues),
        "counts": dict(summary),
        "subjects": details,
        "limitations": [
            "Candidate intervals assume HDF5 index equals source step at 0.2 Hz; raw series alignment has not been verified.",
            "Outside a paired sleep interval does not establish confirmed wake or unknown status.",
            "Local-clock discontinuities may be timezone changes; this report does not establish missing samples.",
            "Missing, invalid, and conflicting nights are excluded from candidates, not relabeled.",
            "This local audit contains subject identifiers; do not include it in a public model bundle.",
            "No evaluation labels, split, or model-quality claim is created by this audit.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new local audit path")
    report = audit(args.data, args.events)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "subjects"}, indent=2))


if __name__ == "__main__":
    main()

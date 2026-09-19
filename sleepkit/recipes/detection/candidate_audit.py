"""Read-only candidate-label coverage; local evidence, not benchmark ground truth."""

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import re

import numpy as np

from sleepkit.artifacts.package import sha256

from .audit import load_events
from .candidate_labels import POLICY, build_candidates
from .data import read_recording


def context_counts(data, labels, context):
    """Count native complete contexts without compacting invalid or unknown frames."""
    if type(context) is not int or context < 1:
        raise ValueError("Context must be a positive integer")
    data, labels = np.asarray(data), np.asarray(labels)
    if data.ndim != 2 or data.shape[0] != 3 or labels.shape != (data.shape[1],):
        raise ValueError("Expected aligned data [3, samples] and labels [samples]")
    if not np.issubdtype(labels.dtype, np.integer) or not np.isin(labels, [-1, 0, 1]).all():
        raise ValueError("Labels must be integer -1, 0, or 1")
    starts = np.arange(0, max(0, len(labels) - 11), 6)
    invalid_prefix = np.r_[0, np.cumsum(~np.isfinite(data).all(axis=0), dtype=np.int64)]
    invalid = invalid_prefix[starts + 12] != invalid_prefix[starts]
    count = len(starts) // context
    size = count * context
    sensor_bad = invalid[:size].reshape(count, context).any(axis=1)
    targets = labels[starts[:size] + 11].reshape(count, context)
    unknown = (targets < 0).any(axis=1)
    retained = ~sensor_bad & ~unknown
    selected = targets[retained]
    return {
        "feature_windows": len(starts),
        "incomplete_context_feature_windows": len(starts) - size,
        "complete_contexts": count,
        "nonfinite_sensor_contexts": int(sensor_bad.sum()),
        "unknown_target_contexts": int(unknown.sum()),
        "nonfinite_and_unknown_contexts": int((sensor_bad & unknown).sum()),
        "retained_contexts": int(retained.sum()),
        "retained_wake_targets": int((selected == 0).sum()),
        "retained_sleep_targets": int((selected == 1).sum()),
    }


def _evidence(path, events_hash):
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("schema") != "sleepkit.source_alignment/v1"
        or report.get("status") != "passed"
        or report.get("issues") != {}
        or report.get("raw_subjects_without_h5") != {}
        or not re.fullmatch(r"[0-9a-f]{64}", str(report.get("parquet_sha256", "")))
    ):
        raise ValueError("A fully passed source-alignment report is required")
    event = report.get("event_clock", {})
    if (
        event.get("status") != "passed_for_available_events"
        or event.get("issues") != {}
        or event.get("events_sha256") != events_hash
        or type(event.get("available_event_rows")) is not int
        or event["available_event_rows"] < 1
        or event.get("matched_event_rows") != event["available_event_rows"]
    ):
        raise ValueError("Matching passed event-clock verification is required")
    details = report.get("subjects")
    if (
        not isinstance(details, dict) or not details
        or report.get("h5_subjects") != len(details)
        or report.get("subjects_passed") != len(details)
    ):
        raise ValueError("Report subject counts must describe complete verification")
    for subject, detail in details.items():
        count, start = detail.get("h5_samples"), detail.get("first_utc_seconds")
        if (
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", subject)
            or detail.get("status") != "passed" or detail.get("issues") != {}
            or type(count) is not int or count < 1 or detail.get("raw_samples") != count
            or type(start) is not int
            or not np.iinfo(np.int64).min <= start <= start + 5 * (count - 1) <= np.iinfo(np.int64).max
            or not re.fullmatch(r"[0-9a-f]{64}", str(detail.get("h5_sha256", "")))
        ):
            raise ValueError(f"Invalid verified subject evidence: {subject}")
    return report


def audit(data_root, events_path, alignment_path, *, context=240):
    """Count candidate coverage using unchanged originals bound to trusted local evidence."""
    if type(context) is not int or context < 1:
        raise ValueError("Context must be a positive integer")
    data_root, events_path, alignment_path = Path(data_root), Path(events_path), Path(alignment_path)
    events_hash, alignment_hash = sha256(events_path), sha256(alignment_path)
    evidence = _evidence(alignment_path, events_hash)
    files = {p.stem: p for p in data_root.glob("*.h5")}
    if set(files) != set(evidence["subjects"]):
        raise ValueError("Original HDF5 subjects must exactly match alignment evidence")
    events = load_events(events_path)
    if set(events) - set(files):
        raise ValueError("Event subjects are missing from verified source files")
    details, aggregate, issues = {}, Counter(), Counter()
    for subject, path in sorted(files.items()):
        verified = evidence["subjects"][subject]
        digest = verified["h5_sha256"]
        if sha256(path) != digest:
            raise ValueError(f"Source HDF5 differs from alignment report: {subject}")
        recording = read_recording(data_root, subject, labels=False)
        count = recording.data.shape[1]
        if count != verified["h5_samples"]:
            raise ValueError(f"Source sample count differs from alignment report: {subject}")
        candidate = build_candidates(events.get(subject, {}), count, first_utc_seconds=verified["first_utc_seconds"])
        counts = {
            "samples": count,
            "sleep_candidate_samples": int((candidate.labels == 1).sum()),
            "wake_candidate_samples": int((candidate.labels == 0).sum()),
            "unknown_samples": int((candidate.labels == -1).sum()),
            "sleep_intervals": len(candidate.sleep_intervals),
            "wake_intervals": len(candidate.wake_intervals),
            **context_counts(recording.data, candidate.labels, context),
        }
        if sha256(path) != digest:
            raise ValueError(f"HDF5 changed during candidate audit: {subject}")
        aggregate.update(counts)
        issues.update(candidate.issues)
        details[subject] = {
            "source_h5_sha256": digest,
            "first_utc_seconds": verified["first_utc_seconds"],
            "valid_nights": list(candidate.valid_nights),
            "issues": candidate.issues,
            "counts": counts,
        }
    if sha256(events_path) != events_hash or sha256(alignment_path) != alignment_hash:
        raise ValueError("Events or alignment evidence changed during candidate audit")
    if any(sha256(path) != evidence["subjects"][subject]["h5_sha256"] for subject, path in files.items()):
        raise ValueError("Source HDF5 changed during candidate audit")
    return {
        "schema": "sleepkit.candidate_label_audit/v1",
        "policy": deepcopy(POLICY),
        "events_sha256": events_hash,
        "alignment_sha256": alignment_hash,
        "parquet_sha256": evidence["parquet_sha256"],
        "source_subjects": len(files),
        "h5_without_events": len(set(files) - set(events)),
        "context_policy": {
            "features_per_context": context, "window_samples": 12, "stride_samples": 6,
            "target": "last source sample in each window",
            "grouping": "nonoverlapping native contexts; drop incomplete tail; never stitch",
            "exclusion": "any nonfinite sensor sample or unknown target; reasons counted independently and jointly",
        },
        "counts": dict(aggregate), "issues": dict(issues), "subjects": details,
        "limitations": [
            "Candidate wake semantics remain provisional; no ground truth, benchmark, training, or label materialization.",
            "Alignment evidence is trusted local verification, not a signed attestation; raw Parquet is not rescanned.",
            "This local report contains subject identifiers and must not be included in a public model bundle.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--context", type=int, default=240)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("Output already exists; choose a new local report path")
    report = audit(args.data, args.events, args.alignment, context=args.context)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "subjects"}, indent=2))


if __name__ == "__main__":
    main()

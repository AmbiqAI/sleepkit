"""Descriptive errors from verified local predictions; no threshold/model tuning."""

import argparse
from bisect import bisect_left, bisect_right
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from sleepkit.artifacts.package import sha256, validate_bundle
from .evaluation import _metrics, summarize
from .preprocessing import SPEC


CONTEXT_BINS = ("first_8", "interior", "last_8")
TRANSITION_BINS = ("within_5_minutes", "over_5_through_30_minutes", "over_30_minutes",
                   "no_observed_transition")


def _rows(path, expected_hash):
    """Hash the exact bytes parsed in each pass, including excluded contexts."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            yield json.loads(line)
    if digest.hexdigest() != expected_hash:
        raise ValueError("Scoring index changed during diagnostics")


def _segments(rows):
    """Compact known-target segments; transition location is the later frame."""
    result = {}
    previous = None
    segment = None
    stride = SPEC["stride_samples"]
    for row in rows:
        subject, position, target = row["subject"], row["feature_end_sample"], row["target"]
        contiguous = (previous is not None and previous[0] == subject
                      and previous[1] + stride == position and previous[2] >= 0)
        if target >= 0:
            if not contiguous:
                segment = [position, position, []]
                result.setdefault(subject, []).append(segment)
            else:
                segment[1] = position
                if previous[2] != target:
                    segment[2].append(position)
        previous = (subject, position, target)
    return result


def _transition_bin(position, segments, starts):
    offset = bisect_right(starts, position) - 1
    if offset < 0 or position > segments[offset][1]:
        raise ValueError("Eligible output outside a known-target segment")
    transitions = segments[offset][2]
    if not transitions:
        return "no_observed_transition"
    index = bisect_left(transitions, position)
    neighbors = transitions[max(0, index - 1):index + 1]
    # SPEC declares a fixed five-second sample clock, verified by the evaluator.
    distance_seconds = min(abs(position - transition) for transition in neighbors) * 5
    if distance_seconds <= 300:
        return "within_5_minutes"
    if distance_seconds <= 1800:
        return "over_5_through_30_minutes"
    return "over_30_minutes"


def _context_bin(position, context):
    if position < 8:
        return "first_8"
    if position >= context - 8:
        return "last_8"
    return "interior"


def _accumulator():
    return {"confusion": np.zeros((2, 2), dtype=np.int64), "loss": 0.0}


def _add(accumulator, target, predicted, loss):
    accumulator["confusion"][target, predicted] += 1
    accumulator["loss"] += loss


def _finish(accumulator):
    confusion = accumulator["confusion"]
    return {
        "feature_frames_evaluated": int(confusion.sum()),
        "errors": int(confusion.sum() - np.trace(confusion)),
        "errors_by_true_class": (confusion.sum(axis=1) - confusion.diagonal()).tolist(),
        "confusion_matrix": confusion.tolist(),
        "class_support": confusion.sum(axis=1).tolist(),
        "metrics": _metrics(confusion, accumulator["loss"]),
    }


def analyze(run_path, output_path):
    """Exclusively write local diagnostics, retaining series IDs only in the file.

    Calls the existing evaluator before reading predictions; streams the index
    twice and stores known-target segments/transition positions, not all rows.
    No source recording or model runtime is loaded.
    """
    run, output = Path(run_path), Path(output_path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}")
    if output.resolve().is_relative_to((run / "bundle").resolve()):
        raise ValueError("Diagnostics output must stay outside the artifact bundle")
    with tempfile.TemporaryDirectory(prefix="sleepkit-error-analysis-") as temporary:
        verified = summarize(run, Path(temporary) / "verified-evaluation.json")
    bound = verified["evidence_sha256"]

    def check_evidence():
        if any((run / name).is_symlink() or not (run / name).is_file()
               or sha256(run / name) != digest for name, digest in bound.items()):
            raise ValueError("Run evidence changed during diagnostics")
        validate_bundle(run / "bundle")

    check_evidence()
    recipe = json.loads((run / "bundle" / "recipe.json").read_text())
    context = recipe["context"]
    with np.load(run / "test-predictions.npz", allow_pickle=False) as arrays:
        logits, targets = arrays["logits"], arrays["targets"]
    predicted = logits.argmax(axis=1)
    shifted = logits - logits.max(axis=1, keepdims=True)
    losses = np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(targets)), targets]
    index_path, index_hash = run / "test-index.jsonl", bound["test-index.jsonl"]
    segments = _segments(_rows(index_path, index_hash))
    starts = {subject: [segment[0] for segment in values] for subject, values in segments.items()}
    pooled = _accumulator()
    subjects = {subject: _accumulator() for subject in verified["subjects"]}
    context_bins = {name: _accumulator() for name in CONTEXT_BINS}
    transition_bins = {name: _accumulator() for name in TRANSITION_BINS}
    ordinal = 0
    for row in _rows(index_path, index_hash):
        if not row["context_eligible"]:
            continue
        if row["eligible_output_index"] != ordinal or ordinal >= len(targets) or row["target"] != targets[ordinal]:
            raise ValueError("Scoring outputs differ from verified predictions")
        subject = row["subject"]
        frame = (row["feature_end_sample"] - row["context_start_sample"] - SPEC["window_samples"] + 1) // SPEC["stride_samples"]
        context_bin = _context_bin(frame, context)
        transition_bin = _transition_bin(row["feature_end_sample"], segments[subject], starts[subject])
        for accumulator in (pooled, subjects[subject], context_bins[context_bin], transition_bins[transition_bin]):
            _add(accumulator, targets[ordinal], predicted[ordinal], float(losses[ordinal]))
        ordinal += 1
    if ordinal != len(targets) or pooled["confusion"].tolist() != verified["pooled"]["confusion_matrix"]:
        raise ValueError("Diagnostics differ from verified evaluation")
    report = {
        "schema": "sleepkit.detection_error_analysis/v1", "target": verified["target"],
        "class_order": verified["class_order"], "model": verified["model"], "split": "test",
        "evidence_sha256": bound, "coverage": verified["coverage"],
        "pooled": _finish(pooled),
        "subjects": {subject: _finish(value) for subject, value in subjects.items()},
        "context_position": {name: _finish(value) for name, value in context_bins.items()},
        "observed_transition_distance": {name: _finish(value) for name, value in transition_bins.items()},
        "observed_transitions": sum(len(segment[2]) for values in segments.values() for segment in values),
        "policy": {
            "context_position": "Zero-based frame position: first 8 frames, then last 8 frames excluding the first bin, otherwise interior. Bins are disjoint; contexts shorter than 16 give first-bin precedence.",
            "observed_transition": "A 0/1 target change between adjacent native feature-grid frames in the same series. Located at the later frame endpoint. Excluded contexts remain visible to transition detection, including frames with invalid sensors. Unknown targets, grid gaps, and series changes break segments; nearest transition must lie in the same known-target segment.",
            "distance": "Absolute endpoint distance on the fixed five-second sample clock. Bins: <=300 seconds; >300 and <=1800 seconds; >1800 seconds; no transition observed in the same known-target segment.",
            "coverage": "Only eligible complete-context outputs are scored. The index omits incomplete tails; no transition is inferred from tails or beyond observed known segments.",
            "interpretation": "Observed target transitions are not raw event boundaries. Descriptive diagnostics only; no threshold tuning, event AP, clinical sleep/wake claim, or independent-frame uncertainty estimate. Fixed two-class macro-F1 uses zero_division=0; zero-coverage metrics are null.",
        },
    }
    check_evidence()
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze(args.run, args.output)
    keys = ("schema", "coverage", "pooled", "context_position", "observed_transition_distance", "observed_transitions", "policy")
    print(json.dumps({key: report[key] for key in keys}, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

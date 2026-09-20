"""Local scoring coordinates for native complete contexts, including exclusions."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .preprocessing import SPEC, fingerprint, prepare


def write_index(source, subjects, context, path, *, cache=None, expected_targets=None):
    """Write JSONL in evaluation order without retaining a dataset-wide row list.

    ``eligible_output_index`` is the zero-based flattened prediction position for
    this subject order. Rejected native contexts remain in the file with a null
    position. Individual frame validity is distinct from whole-context eligibility.
    Source IDs and coordinates belong in local run artifacts, not public bundles.
    """
    if type(context) is not int or context < 1:
        raise ValueError("Context must be a positive integer")
    if expected_targets is not None:
        expected_targets = np.asarray(expected_targets)
        if (
            expected_targets.ndim != 1
            or not np.issubdtype(expected_targets.dtype, np.integer)
            or not np.isin(expected_targets, [0, 1]).all()
        ):
            raise ValueError("Expected targets must be an integer vector containing 0 or 1")
    if hasattr(source, "context") and context != source.context:
        raise ValueError("Context must match the frozen source protocol")
    subjects = list(subjects)
    if any(not isinstance(subject, str) or not subject for subject in subjects) or len(set(subjects)) != len(subjects):
        raise ValueError("Subjects must be unique nonempty strings")
    hashes = {subject: source.source_hashes[subject] for subject in subjects}
    summary = {
        "schema": "sleepkit.detection_scoring_index/v1",
        "format": "jsonl",
        "context": context,
        "subjects": len(subjects),
        "source_sha256": fingerprint(hashes),
        "provenance_sha256": fingerprint(source.provenance),
        "target_sha256": fingerprint(source.target),
        "preprocessing_sha256": fingerprint(SPEC),
        "complete_contexts": 0,
        "eligible_contexts": 0,
        "outputs": 0,
        "eligible_outputs": 0,
        "excluded_contexts": {"nonfinite_sensor": 0, "unknown_target": 0},
        "tail_feature_frames": 0,
        "samples_after_last_feature": 0,
    }
    path = Path(path)
    digest = hashlib.sha256()
    verify = getattr(source, "verify_unchanged", None)
    if verify is not None:
        verify()
    # Open before try so a failed exclusive creation cannot remove an existing file.
    stream = path.open("xb")
    try:
        with stream:
            for subject in subjects:
                recording = source.read(subject, labels=True)
                features = prepare(recording.data, cache, sample_time=recording.sample_time)
                labels = np.asarray(recording.labels)
                if (
                    labels.shape != (recording.data.shape[1],)
                    or not np.issubdtype(labels.dtype, np.integer)
                    or not np.isin(labels, [-1, 0, 1]).all()
                ):
                    raise ValueError("Labels must align with samples and contain integer -1, 0, or 1")
                targets = labels[features.ends]
                complete_frames = len(features.ends) // context * context
                summary["tail_feature_frames"] += len(features.ends) - complete_frames
                last_end = int(features.ends[-1]) if len(features.ends) else -1
                summary["samples_after_last_feature"] += recording.data.shape[1] - last_end - 1
                for start in range(0, complete_frames, context):
                    stop = start + context
                    reasons = []
                    if not features.valid[start:stop].all():
                        reasons.append("nonfinite_sensor")
                    if (targets[start:stop] < 0).any():
                        reasons.append("unknown_target")
                    eligible = not reasons
                    summary["complete_contexts"] += 1
                    summary["eligible_contexts"] += int(eligible)
                    for reason in reasons:
                        summary["excluded_contexts"][reason] += 1
                    for frame in range(start, stop):
                        position = summary["eligible_outputs"]
                        if eligible and expected_targets is not None and (
                            position >= len(expected_targets) or targets[frame] != expected_targets[position]
                        ):
                            raise ValueError("Scoring index does not match evaluated targets")
                        row = {
                            "subject": subject,
                            "source_sha256": hashes[subject],
                            "output_index": summary["outputs"],
                            "eligible_output_index": summary["eligible_outputs"] if eligible else None,
                            "context_start_sample": int(features.ends[start]) - SPEC["window_samples"] + 1,
                            "context_end_sample": int(features.ends[stop - 1]),
                            "feature_end_sample": int(features.ends[frame]),
                            "target": int(targets[frame]),
                            "sensor_valid": bool(features.valid[frame]),
                            "target_known": bool(targets[frame] >= 0),
                            "context_eligible": eligible,
                            "exclusion_reasons": reasons,
                        }
                        encoded = (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode()
                        stream.write(encoded)
                        digest.update(encoded)
                        summary["outputs"] += 1
                        summary["eligible_outputs"] += int(eligible)
        if expected_targets is not None and summary["eligible_outputs"] != len(expected_targets):
            raise ValueError("Scoring index does not match evaluated targets")
        if verify is not None:
            verify()
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    summary["sha256"] = digest.hexdigest()
    return summary

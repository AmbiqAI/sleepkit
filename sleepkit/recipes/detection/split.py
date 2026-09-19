"""Freeze a reproducible group-disjoint experiment split before model selection."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from sleepkit.artifacts.package import sha256, write_json
from .candidate_labels import POLICY
from .hdf5 import require_self_contained

PARTITIONS = ("train", "validation", "test")
TARGET = {
    "kind": "sleepkit.annotated_period_membership/v1",
    "classes": ["OUTSIDE_SUPPORTED_ANNOTATED_PERIOD", "INSIDE_ANNOTATED_PERIOD"],
    "unknown": -1,
    "interpretation": "Derived annotated nightly-period membership, not clinical per-sample sleep/wake or official event AP",
    "source": "https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/data",
    "source_content_sha256": "8a2f93fc7f5a9264b14773ef98d0e6a227e6ab5b595edd3a70d0758442813dbf",
}


def _read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    return json.loads(path.read_text(), object_pairs_hook=unique)


def assign(groups, *, seed=0):
    """Hash-rank all groups, independent of input ordering, labels, and model scores."""
    if type(seed) is not int or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    if any(not isinstance(s, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", s) for s in groups):
        raise ValueError("Subject IDs must be filename stems supported by the detection reader")
    if not groups or any(not isinstance(g, str) or not g.strip() or "\0" in g for g in groups.values()):
        raise ValueError("Each subject needs a nonempty string group ID")
    ranked = sorted(
        set(groups.values()), key=lambda g: (hashlib.sha256(f"sleepkit.split/v1\0{seed}\0{g}".encode()).hexdigest(), g)
    )
    train, validation = len(ranked) * 70 // 100, len(ranked) * 15 // 100
    if min(train, validation, len(ranked) - train - validation) < 1:
        raise ValueError("At least seven groups are needed for nonempty 70/15/15 partitions")
    buckets = (ranked[:train], ranked[train : train + validation], ranked[train + validation :])
    membership = {group: name for name, bucket in zip(PARTITIONS, buckets) for group in bucket}
    return {
        name: sorted(subject for subject, group in groups.items() if membership[group] == name) for name in PARTITIONS
    }


def freeze(data_root, coverage_path, destination, *, seed=0, groups_path=None):
    """Freeze all audited subjects; coverage summarizes an already assigned split.

    Reports/group maps are trusted local evidence. Default grouping makes only a
    series-disjoint claim; a supplied group map is recorded, not independently verified.
    Single-writer directory publication follows the other local artifact adapters.
    """
    import h5py

    data_root, coverage_path, destination = Path(data_root), Path(coverage_path), Path(destination)
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    coverage_hash = sha256(coverage_path)
    coverage = _read_json(coverage_path)
    details = coverage.get("subjects", {})
    files = {p.stem: p for p in data_root.glob("*.h5")}
    if (
        coverage.get("schema") != "sleepkit.candidate_label_audit/v1"
        or coverage.get("policy") != POLICY
        or not files
        or set(files) != set(details)
        or coverage.get("source_subjects") != len(files)
    ):
        raise ValueError("Require matching candidate policy and complete source coverage")
    context_policy = coverage.get("context_policy", {})
    context = context_policy.get("features_per_context")
    if (
        type(context) is not int
        or context < 1
        or context_policy
        != {
            "features_per_context": context,
            "window_samples": 12,
            "stride_samples": 6,
            "target": "last source sample in each window",
            "grouping": "nonoverlapping native contexts; drop incomplete tail; never stitch",
            "exclusion": "any nonfinite sensor sample or unknown target; reasons counted independently and jointly",
        }
    ):
        raise ValueError("Unsupported candidate context policy")
    sources = {}
    for subject, path in files.items():
        digest = sha256(path)
        if details[subject].get("source_h5_sha256") != digest:
            raise ValueError("Source differs from candidate coverage evidence")
        with h5py.File(path, "r") as stream:
            require_self_contained(stream)
            shape = stream["data"].shape
        counts = details[subject].get("counts", {})
        required = (
            "samples",
            "sleep_candidate_samples",
            "wake_candidate_samples",
            "unknown_samples",
            "retained_contexts",
            "retained_sleep_targets",
            "retained_wake_targets",
        )
        if any(type(counts.get(k)) is not int or counts[k] < 0 for k in required):
            raise ValueError("Invalid candidate coverage counts")
        if (
            any(type(v) is not int or v < 0 for v in counts.values())
            or shape != (3, counts["samples"])
            or sum(counts[k] for k in ("sleep_candidate_samples", "wake_candidate_samples", "unknown_samples"))
            != counts["samples"]
            or counts["retained_sleep_targets"] + counts["retained_wake_targets"]
            != context * counts["retained_contexts"]
        ):
            raise ValueError("Inconsistent candidate coverage counts")
        windows = max(0, (counts["samples"] - 12) // 6 + 1)
        complete = windows // context
        sensor = counts.get("nonfinite_sensor_contexts")
        unknown = counts.get("unknown_target_contexts")
        both = counts.get("nonfinite_and_unknown_contexts")
        if (
            counts.get("feature_windows") != windows
            or counts.get("complete_contexts") != complete
            or counts.get("incomplete_context_feature_windows") != windows % context
            or any(type(v) is not int or not 0 <= v <= complete for v in (sensor, unknown, both))
            or both > min(sensor, unknown)
            or counts["retained_contexts"] != complete - sensor - unknown + both
            or counts["retained_sleep_targets"] > counts["sleep_candidate_samples"]
            or counts["retained_wake_targets"] > counts["wake_candidate_samples"]
        ):
            raise ValueError("Impossible native context coverage")
        sources[subject] = digest
    groups_hash = None
    if groups_path is None:
        groups = {subject: subject for subject in files}
    else:
        groups_path = Path(groups_path)
        groups_hash = sha256(groups_path)
        groups = _read_json(groups_path)
        if not isinstance(groups, dict) or set(groups) != set(files):
            raise ValueError("Group mapping must cover every source subject exactly")
    split = assign(groups, seed=seed)
    summaries = {}
    for name, subjects in split.items():
        counts = Counter()
        for subject in subjects:
            counts.update(details[subject]["counts"])
        summaries[name] = {
            "subjects": len(subjects),
            "groups": len({groups[s] for s in subjects}),
            "counts": dict(counts),
            "both_candidate_classes_retained": counts["retained_sleep_targets"] > 0
            and counts["retained_wake_targets"] > 0,
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".split-", dir=destination.parent))
    try:
        write_json(stage / "split.json", split)
        write_json(stage / "groups.json", groups)
        write_json(stage / "sources.json", sources)
        manifest = {
            "schema": "sleepkit.frozen_split/v1",
            "seed": seed,
            "assignment": "SHA256 rank of sleepkit.split/v1 NUL seed NUL group; train=floor(70% groups), validation=floor(15%), test=remainder",
            "grouping": "series_id_only" if groups_path is None else "supplied_group_mapping_unverified",
            "coverage_sha256": coverage_hash,
            "supplied_groups_sha256": groups_hash,
            "target": TARGET,
            "identity_basis": "Organizer describes a unique experimental subject per series; no independent identity linkage performed",
            "policy": coverage["policy"],
            "context_policy": coverage["context_policy"],
            "files": {name: sha256(stage / name) for name in ("split.json", "groups.json", "sources.json")},
            "partitions": summaries,
            "limitations": [
                "Subject disjointness relies on organizer series identity description or supplied grouping; no independent identity linkage.",
                "Historical model exposure is unknown; this split does not establish historical holdout.",
                "Assignment only: the existing recipe still reads historical labels; target-aware label materialization is required before the derived benchmark.",
            ],
        }
        write_json(stage / "protocol.json", manifest)
        if (
            sha256(coverage_path) != coverage_hash
            or any(sha256(files[s]) != h for s, h in sources.items())
            or (groups_path is not None and sha256(groups_path) != groups_hash)
        ):
            raise ValueError("Source or evidence changed while freezing split")
        if os.path.lexists(destination):
            raise FileExistsError(destination)
        stage.rename(destination)
        return manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report = freeze(args.data, args.coverage, args.output, seed=args.seed, groups_path=args.groups)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Explicit subject membership for saved-feature staging experiments."""

import hashlib
import json
from pathlib import Path
import re

import numpy as np

PARTITIONS = ("train", "validation", "test")
SCHEMA = "sleepkit.staging_subject_split/v1"


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=_unique)


def validate_split(manifest):
    """Return a normalized copy; filename stems are caller-declared subject identities."""
    if not isinstance(manifest, dict) or set(manifest) != {"schema", "dataset", "partitions"}:
        raise ValueError("Expected a staging subject manifest with schema, dataset, partitions")
    if manifest["schema"] != SCHEMA or not isinstance(manifest["dataset"], str) or not manifest["dataset"].strip():
        raise ValueError("Unsupported split schema or empty dataset identity")
    groups = manifest["partitions"]
    if not isinstance(groups, dict) or set(groups) != set(PARTITIONS):
        raise ValueError("Require train, validation, and test partitions")
    seen, copied = set(), {}
    for name in PARTITIONS:
        subjects = groups[name]
        if not isinstance(subjects, list) or not subjects:
            raise ValueError(f"Partition {name} must be a nonempty subject list")
        if any(not isinstance(s, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", s) for s in subjects):
            raise ValueError("Subject IDs must be safe filename stems")
        if len(set(subjects)) != len(subjects) or seen.intersection(subjects):
            raise ValueError("Subject partitions must be unique and disjoint")
        seen.update(subjects)
        copied[name] = sorted(subjects)
    return {"schema": SCHEMA, "dataset": manifest["dataset"], "partitions": copied}


def load_split(path):
    return validate_split(read_json(path))


def create_split(subjects, *, dataset, validation_count, test_count, seed=0):
    """Split an explicit cohort deterministically, before reading features or labels."""
    subjects = list(subjects)
    if any(type(n) is not int or n < 1 for n in (validation_count, test_count)):
        raise ValueError("Validation and test counts must be positive integers")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer in [0, 2**32)")
    if len(subjects) <= validation_count + test_count:
        raise ValueError("Cohort must also leave a nonempty training partition")
    if any(not isinstance(s, str) for s in subjects) or len(set(subjects)) != len(subjects):
        raise ValueError("Cohort must have unique string subject IDs")
    order = np.random.default_rng(seed).permutation(sorted(subjects)).tolist()
    groups = {
        "test": order[:test_count],
        "validation": order[test_count : test_count + validation_count],
        "train": order[test_count + validation_count :],
    }
    return validate_split({"schema": SCHEMA, "dataset": dataset, "partitions": groups})

"""CMIDSS HDF5 adapter and explicit subject splits; no legacy task imports."""

from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np

from .hdf5 import require_self_contained

from .preprocessing import contexts, prepare, validate_input

READER_SPEC = {
    "kind": "sleepkit.cmidss_hdf5/v2",
    "channels": ["TS", "ENMO", "ZANGLE"],
    "sample_rate_hz": 0.2,
    "sample_clock": "Optional sample_time dataset: signed int64 Unix seconds, units attribute unix_seconds",
    "legacy_missing_metadata": "Assume CMIDSS TS/ENMO/ZANGLE at 0.2 Hz when attributes are absent; validate TS cadence during feature extraction",
}


@dataclass(frozen=True)
class Recording:
    data: np.ndarray
    labels: np.ndarray | None
    sample_time: np.ndarray | None = None


def read_subject(root, subject, *, labels=True):
    """Legacy two-array interface; use read_recording to retain the independent clock."""
    recording = read_recording(root, subject, labels=labels)
    return recording.data, recording.labels


def read_recording(root, subject, *, labels=True):
    import h5py

    if not isinstance(subject, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", subject):
        raise ValueError("Subject ID must be a filename stem without directories")
    with h5py.File(Path(root) / f"{subject}.h5", "r") as stream:
        require_self_contained(stream)
        # HDF5 attributes may use float32: 0.2 is then approximately 0.200000003.
        if not np.isclose(float(stream.attrs.get("sample_rate_hz", 0.2)), 0.2, rtol=0, atol=1e-8):
            raise ValueError("This reader requires 0.2 Hz CMIDSS sensor channels")
        if "channel_names" in stream.attrs:
            channels = [v.decode() if isinstance(v, bytes) else str(v) for v in stream.attrs["channel_names"]]
            if channels != READER_SPEC["channels"]:
                raise ValueError("HDF5 channel_names must be TS, ENMO, ZANGLE in that order")
        data = np.asarray(stream["data"], dtype=np.float32)
        if data.ndim != 2 or data.shape[0] != 3:
            raise ValueError("HDF5 data must have shape [3, samples]")
        target = np.asarray(stream["sleep_stages"]) if labels else None
        sample_time = None
        if "sample_time" in stream:
            clock = stream["sample_time"]
            units = clock.attrs.get("units")
            if isinstance(units, bytes):
                units = units.decode()
            if units != "unix_seconds":
                raise ValueError("sample_time units must be unix_seconds")
            sample_time = np.asarray(clock)
            validate_input(data, sample_time)
    if target is not None and (
        target.shape != (data.shape[1],)
        or not np.issubdtype(target.dtype, np.integer)
        or not np.isin(target, [-1, 0, 1]).all()
    ):
        raise ValueError("sleep_stages must align with samples and contain integer -1, 0, or 1")
    return Recording(data, target, sample_time)


def load_split(path, root):
    split = json.loads(Path(path).read_text())
    if set(split) != {"train", "validation", "test"}:
        raise ValueError("Split must specify train, validation, and test subjects")
    subjects = []
    for group in split.values():
        if not isinstance(group, list) or not group:
            raise ValueError("Each subject partition must be a nonempty list")
        for subject in group:
            if not isinstance(subject, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", subject):
                raise ValueError("Invalid subject ID")
            if not (Path(root) / f"{subject}.h5").is_file():
                raise ValueError(f"Missing subject: {subject}")
        subjects.extend(group)
    if len(set(subjects)) != len(subjects):
        raise ValueError("Subject partitions must be disjoint and contain no duplicates")
    return split


def subject_features(root, subjects, cache=None):
    for subject in subjects:
        recording = read_recording(root, subject, labels=False)
        yield prepare(recording.data, cache, sample_time=recording.sample_time)


def examples(root, subjects, normalizer, context, cache=None):
    for subject in subjects:
        recording = read_recording(root, subject)
        features = normalizer.transform(
            prepare(recording.data, cache, sample_time=recording.sample_time, spec=normalizer.spec)
        )
        for values, target, _ in contexts(features, context, recording.labels):
            yield values, target.astype(np.int32)

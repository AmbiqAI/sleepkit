"""Explicit feature and fitted-state contracts shared by this recipe and inference."""

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

SPEC = {
    "kind": "sleepkit.cmidss_wrist/v2",
    "implementation_version": 2,
    "tod_formula": "mean(cos(2*pi*TS/86400))",
    "source_channels": ["TS", "ENMO", "ZANGLE"],
    "source_rate_hz": 0.2,
    "features": ["tod", "mov_mu", "mov_std", "angle_mu", "angle_std"],
    "window_samples": 12,
    "stride_samples": 6,
    "cadence_seconds": 30,
    "timestamp": "record-relative seconds of the last source sample in each feature window",
    "feature_available_at": "timestamp + 5 seconds",
    "prediction_available_at": "last feature timestamp of the complete model context + 5 seconds",
    "target_alignment": "last sample in window",
    "invalid": "drop contexts containing any nonfinite sensor sample",
    "tail": "drop incomplete windows and model contexts",
    "normalization": "training-subject global mean and population variance; epsilon=1e-6",
}


V2_SPEC = deepcopy(SPEC)
SPEC = {
    **SPEC,
    "kind": "sleepkit.cmidss_wrist/v3",
    "implementation_version": 3,
    "sample_clock": "Optional signed int64 Unix seconds; strictly increasing at five seconds. Without it, validate local TS cadence.",
}


def validate_input(data, sample_time=None, *, spec=None):
    """Validate source clocks before any cache access; clocks are never inferred from row indices."""
    spec = SPEC if spec is None else spec
    if spec != SPEC and spec != V2_SPEC:
        raise ValueError("Unsupported preprocessing implementation or feature contract")
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2 or data.shape[0] != 3:
        raise ValueError("Expected sensor data [3, samples] in TS, ENMO, ZANGLE order")
    ts = data[0]
    finite = np.isfinite(ts)
    if ((ts[finite] < 0) | (ts[finite] >= 86400)).any():
        raise ValueError("TS must contain seconds of day in [0, 86400)")
    if sample_time is not None:
        if spec == V2_SPEC:
            raise ValueError("Preprocessing v2 does not accept an independent sample clock")
        sample_time = np.asarray(sample_time)
        if sample_time.shape != (data.shape[1],) or sample_time.dtype.kind != "i" or sample_time.dtype.itemsize != 8:
            raise ValueError("sample_time must be signed int64 Unix seconds aligned with sensor samples")
        if (sample_time[1:] <= sample_time[:-1]).any() or (np.diff(sample_time) != 5).any():
            raise ValueError("sample_time must contain contiguous increasing 5-second samples")
    else:
        adjacent = finite[1:] & finite[:-1]
        if not np.allclose(np.mod(np.diff(ts)[adjacent], 86400), 5, atol=1e-3, rtol=0):
            raise ValueError("Expected contiguous 5-second samples; gaps must be explicitly represented")
    return data, sample_time


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Features:
    values: np.ndarray
    valid: np.ndarray
    ends: np.ndarray

    @property
    def times(self):
        return self.ends * 5.0


def extract(data, *, sample_time=None, spec=None):
    """Read aligned samples, using an optional independent source UTC clock for cadence."""
    data, _ = validate_input(data, sample_time, spec=spec)
    starts = np.arange(0, max(0, data.shape[1] - 11), 6)
    values = np.zeros((len(starts), 5), np.float32)
    valid = np.ones(len(starts), bool)
    for i, start in enumerate(starts):
        window = data[:, start : start + 12]
        if not np.isfinite(window).all():
            valid[i] = False
            continue
        values[i] = [
            np.mean(np.cos(2 * np.pi * window[0].astype(float) / 86400)),
            np.mean(window[1], dtype=float),
            np.std(window[1], dtype=float),
            np.mean(window[2], dtype=float),
            np.std(window[2], dtype=float),
        ]
    return Features(values, valid, starts + 11)


def prepare(data, cache=None, *, sample_time=None, spec=None):
    """Cache stateless features only: annotations and normalization never enter the key."""
    spec = SPEC if spec is None else spec
    data, sample_time = validate_input(data, sample_time, spec=spec)
    if cache is None:
        return extract(data, sample_time=sample_time, spec=spec)
    data = np.ascontiguousarray(data, dtype=np.float32)
    clock_bytes = b"absent" if sample_time is None else b"unix_seconds:" + sample_time.astype("<i8").tobytes()
    digest = hashlib.sha256(
        fingerprint(spec).encode() + str(data.shape).encode() + data.tobytes() + clock_bytes
    ).hexdigest()
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{digest}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as arrays:
            result = Features(*(arrays[name] for name in ("values", "valid", "ends")))
        count = max(0, (data.shape[1] - 12) // 6 + 1)
        if (
            result.values.shape != (count, 5)
            or result.valid.shape != (count,)
            or result.valid.dtype != bool
            or not np.array_equal(result.ends, np.arange(count) * 6 + 11)
            or not np.isfinite(result.values).all()
        ):
            raise ValueError("Invalid feature cache")
        return result
    result = extract(data, sample_time=sample_time, spec=spec)
    with tempfile.NamedTemporaryFile(dir=cache, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            np.savez_compressed(stream, values=result.values, valid=result.valid, ends=result.ends)
            stream.flush()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return result


@dataclass(frozen=True)
class Normalizer:
    mean: np.ndarray
    scale: np.ndarray
    count: int
    spec: dict = field(default_factory=lambda: deepcopy(SPEC))

    def __post_init__(self):
        if (
            (self.spec != SPEC and self.spec != V2_SPEC)
            or np.shape(self.mean) != (5,)
            or np.shape(self.scale) != (5,)
            or self.count < 1
            or not np.isfinite([self.mean, self.scale]).all()
            or (np.asarray(self.scale) <= 0).any()
        ):
            raise ValueError("Invalid fitted normalization state")

    @classmethod
    def fit(cls, training_features):
        """Merge moments incrementally; caller supplies training subjects only."""
        count, mean, m2 = 0, np.zeros(5), np.zeros(5)
        for features in training_features:
            values = features.values[features.valid].astype(np.float64)
            n = len(values)
            if not n:
                continue
            delta = values.mean(axis=0) - mean
            total = count + n
            m2 += ((values - values.mean(axis=0)) ** 2).sum(axis=0) + delta**2 * count * n / total
            mean += delta * n / total
            count = total
        if not count:
            raise ValueError("No valid training features")
        return cls(mean, np.sqrt(m2 / count + 1e-6), count)

    def transform(self, features):
        values = ((features.values - self.mean) / self.scale).astype(np.float32)
        values[~features.valid] = 0
        if not np.isfinite(values).all():
            raise ValueError("Normalization produced nonfinite values")
        return Features(values, features.valid, features.ends)

    def to_dict(self):
        return {
            "spec": deepcopy(self.spec),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "count": self.count,
        }

    @classmethod
    def from_dict(cls, value):
        if value.get("spec") != SPEC and value.get("spec") != V2_SPEC:
            raise ValueError("Unsupported preprocessing implementation or feature contract")
        return cls(
            np.asarray(value["mean"], dtype=float),
            np.asarray(value["scale"], dtype=float),
            value["count"],
            deepcopy(value["spec"]),
        )


def contexts(features, context, labels=None):
    """Yield complete nonoverlapping model contexts, never bridge dropped frames."""
    if type(context) is not int or context < 1:
        raise ValueError("Context must be a positive integer")
    if labels is not None:
        labels = np.asarray(labels)
        if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer) or not np.isin(labels, [-1, 0, 1]).all():
            raise ValueError("Labels must be a vector of -1 (unknown), 0 (WAKE), or 1 (SLEEP)")
        if len(features.ends) and features.ends[-1] >= len(labels):
            raise ValueError("Labels do not align with sensor samples")
        labels = labels[features.ends]
    for start in range(0, len(features.values) - context + 1, context):
        selection = slice(start, start + context)
        if not features.valid[selection].all():
            continue
        target = None if labels is None else labels[selection]
        if target is not None and (target < 0).any():
            continue
        yield features.values[selection], target, features.times[selection]

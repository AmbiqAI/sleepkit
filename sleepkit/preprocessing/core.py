"""Deterministic feature preparation independent of datasets and training."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from sleepkit.data import Record


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Channel:
    name: str
    unit: str
    modality: str
    location: str


@dataclass
class Prepared:
    values: np.ndarray
    times: np.ndarray
    valid: np.ndarray
    targets: np.ndarray
    target_valid: np.ndarray
    spec: dict

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=np.float32)
        self.times = np.asarray(self.times, dtype=np.float64)
        self.valid = np.asarray(self.valid, dtype=bool)
        self.targets = np.asarray(self.targets)
        self.target_valid = np.asarray(self.target_valid, dtype=bool)
        if self.values.ndim != 2 or self.values.shape[1] < 1:
            raise ValueError("Prepared features must have shape [time, features]")
        length = len(self.values)
        if any(array.shape != (length,) for array in (self.times, self.valid, self.targets, self.target_valid)):
            raise ValueError("Prepared timestamps, masks, and targets must align")
        if not np.isfinite(self.times).all() or (np.diff(self.times) <= 0).any():
            raise ValueError("Prepared timestamps must be finite and increasing")
        if not np.isfinite(self.values[self.valid]).all():
            raise ValueError("Valid prepared features must be finite")
        if not np.issubdtype(self.targets.dtype, np.integer) or (self.targets[self.target_valid] < 0).any():
            raise ValueError("Valid targets must be nonnegative integers")
        if len(self.spec.get("feature_names", [])) != self.values.shape[1]:
            raise ValueError("Feature names must match prepared columns")


def align_targets(times, annotations):
    """Assign labels at window centers using half-open annotation intervals."""
    labels = np.zeros(len(times), dtype=np.int64)
    valid = np.zeros(len(times), dtype=bool)
    for annotation in annotations:
        selected = (times >= annotation.start) & (times < annotation.stop)
        labels[selected] = annotation.label
        valid[selected] = True
    return labels, valid


@dataclass(frozen=True)
class WindowFeatures:
    """Mean/std features over complete time windows, independently per channel.

    All channel starts use one record-relative clock. Rates may differ. Missing
    samples must be represented by validity masks. Partial tails are dropped.
    No sensor substitution, interpolation, or label-dependent transforms occur.
    """

    channels: tuple[Channel, ...]
    window_seconds: float = 30.0
    stride_seconds: float = 30.0
    min_valid_fraction: float = 1.0

    def __post_init__(self):
        if not self.channels or len({c.name for c in self.channels}) != len(self.channels):
            raise ValueError("Channels must be nonempty and unique")
        if (
            not np.isfinite([self.window_seconds, self.stride_seconds]).all()
            or min(self.window_seconds, self.stride_seconds) <= 0
        ):
            raise ValueError("Window and stride must be finite and positive")
        if not 0 < self.min_valid_fraction <= 1:
            raise ValueError("Valid fraction must be in (0, 1]")

    def to_dict(self):
        return {
            "kind": "window_mean_std",
            "version": 1,
            **asdict(self),
            "channels": [asdict(channel) for channel in self.channels],
        }

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        if value.pop("kind") != "window_mean_std" or value.pop("version") != 1:
            raise ValueError("Unsupported preprocessing definition")
        return cls(channels=tuple(Channel(**c) for c in value.pop("channels")), **value)

    @property
    def spec(self):
        return {
            "preprocessing": self.to_dict(),
            "feature_names": [f"{c.name}.{stat}" for c in self.channels for stat in ("mean", "std")],
            "dtype": "float32",
            "cadence_seconds": self.stride_seconds,
            "timestamp": "window_center",
            "available_at": "window_end",
            "tail": "drop_incomplete",
            "target_alignment": "center_half_open",
        }

    def __call__(self, record: Record):
        signals = []
        for channel in self.channels:
            if channel.name not in record.signals:
                raise ValueError(f"Missing channel {channel.name!r} in {record.recording}")
            signal = record.signals[channel.name]
            for attribute in ("unit", "modality", "location"):
                if getattr(signal, attribute) != getattr(channel, attribute):
                    raise ValueError(f"Incompatible {attribute} for {channel.name}")
            signals.append(signal)
        start = max(signal.start for signal in signals)
        stop = min(signal.stop for signal in signals)
        count = max(0, int(np.floor((stop - start - self.window_seconds) / self.stride_seconds + 1e-9)) + 1)
        starts = start + np.arange(count) * self.stride_seconds
        values = np.zeros((count, len(signals) * 2), dtype=np.float32)
        valid = np.ones(count, dtype=bool)
        for row, left in enumerate(starts):
            for col, signal in enumerate(signals):
                first = int(np.ceil((left - signal.start) * signal.sample_rate - 1e-9))
                last = int(np.ceil((left + self.window_seconds - signal.start) * signal.sample_rate - 1e-9))
                mask = signal.valid[first:last]
                if not len(mask) or np.mean(mask) < self.min_valid_fraction or not mask.any():
                    valid[row] = False
                    continue
                segment = signal.values[first:last][mask]
                values[row, 2 * col : 2 * col + 2] = (
                    np.mean(segment, dtype=np.float64),
                    np.std(segment, dtype=np.float64),
                )
        valid &= np.isfinite(values).all(axis=1)
        values[~valid] = 0
        times = starts + self.window_seconds / 2
        targets, target_valid = align_targets(times, record.annotations)
        return Prepared(values, times, valid, targets, target_valid, self.spec)


def prepare(record, transform, cache_dir=None):
    """Run a deterministic transform, optionally using a content-addressed cache.

    Custom transforms expose ``to_dict()`` with their implementation version and
    parameters, and return Prepared. Model parameters never enter the cache key.
    """
    if cache_dir is None:
        return transform(record)
    digest = hashlib.sha256()
    metadata = {
        "dataset": record.dataset,
        "subject": record.subject,
        "recording": record.recording,
        "source_revision": record.source_revision,
        "annotations": [asdict(a) for a in record.annotations],
        "transform": transform.to_dict(),
    }
    digest.update(fingerprint(metadata).encode())
    for name, signal in sorted(record.signals.items()):
        digest.update(
            fingerprint(
                {
                    "name": name,
                    "rate": signal.sample_rate,
                    "start": signal.start,
                    "unit": signal.unit,
                    "modality": signal.modality,
                    "location": signal.location,
                    "length": len(signal.values),
                }
            ).encode()
        )
        digest.update(signal.values.tobytes())
        digest.update(signal.valid.tobytes())
    folder = Path(cache_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{digest.hexdigest()}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            return Prepared(
                **{key: saved[key] for key in ("values", "times", "valid", "targets", "target_valid")},
                spec=json.loads(str(saved["spec"])),
            )
    result = transform(record)
    with tempfile.NamedTemporaryFile(dir=folder, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            np.savez_compressed(
                stream,
                values=result.values,
                times=result.times,
                valid=result.valid,
                targets=result.targets,
                target_valid=result.target_valid,
                spec=json.dumps(result.spec, sort_keys=True),
            )
            stream.flush()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return result


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray
    input_fingerprint: str
    training_subjects: tuple[tuple[str, str], ...]

    @classmethod
    def fit(cls, records, transform, split, cache_dir=None):
        """Select training subjects explicitly; never fit on validation/test rows."""
        training = split.select(records, "train")
        prepared = [prepare(record, transform, cache_dir) for record in training]
        if len({fingerprint(item.spec) for item in prepared}) != 1:
            raise ValueError("Training preprocessing schemas differ")
        values = np.concatenate([item.values[item.valid] for item in prepared])
        if not len(values):
            raise ValueError("No valid training features")
        return cls(
            values.mean(axis=0, dtype=np.float64),
            np.maximum(values.std(axis=0, dtype=np.float64), 1e-6),
            fingerprint(prepared[0].spec),
            tuple(sorted(split.train)),
        )

    def transform(self, prepared):
        if fingerprint(prepared.spec) != self.input_fingerprint:
            raise ValueError("Preprocessing schema does not match fitted normalization")
        values = ((prepared.values - self.mean) / self.scale).astype(np.float32)
        values[~prepared.valid] = 0
        return Prepared(values, prepared.times, prepared.valid, prepared.targets, prepared.target_valid, prepared.spec)

    def to_dict(self):
        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "input_fingerprint": self.input_fingerprint,
            "training_subjects": self.training_subjects,
        }

    @classmethod
    def from_dict(cls, value):
        mean, scale = np.asarray(value["mean"]), np.asarray(value["scale"])
        if mean.ndim != 1 or mean.shape != scale.shape or not np.isfinite([mean, scale]).all() or (scale <= 0).any():
            raise ValueError("Invalid normalization state")
        return cls(mean, scale, value["input_fingerprint"], tuple(tuple(key) for key in value["training_subjects"]))


def model_windows(prepared, context, require_targets=True):
    """Nonoverlapping contexts; drop tails and contexts with invalid frames."""
    if not isinstance(context, int) or context < 1:
        raise ValueError("Context must be a positive integer")
    count = len(prepared.values) // context
    length = count * context
    x = prepared.values[:length].reshape(count, context, prepared.values.shape[1])
    y = prepared.targets[:length].reshape(count, context)
    times = prepared.times[:length].reshape(count, context)
    valid = prepared.valid[:length].reshape(count, context).all(axis=1)
    if require_targets:
        valid &= prepared.target_valid[:length].reshape(count, context).all(axis=1)
    return x[valid], y[valid], times[valid]

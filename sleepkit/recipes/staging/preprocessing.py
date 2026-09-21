"""Explicit saved-feature contract for the historical three-class staging baseline.

Whole-record normalization is offline: statistics include later epochs of the
same subject. These functions do not generate features from raw sensor signals.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

FEATURE_ORDER = (
    "hr_bpm",
    "hrv_td_mean_nn",
    "hrv_td_sd_nn",
    "hrv_td_median_nn",
    "hrv_fd_lfhf_ratio",
    "spo2_mu",
    "spo2_std",
    "spo2_med",
    "mov_mu",
    "mov_std",
    "mov_med",
    "rsp_bpm",
    "spo2_qos",
    "hrv_qos",
)
CLASS_ORDER = ("WAKE", "NREM", "REM")
WINDOW_EPOCHS = 240
EPOCH_SECONDS = 30
_CLASS_MAP = np.array([0, 1, 1, 1, 1, 2, -1], dtype=np.int32)


@dataclass(frozen=True)
class SubjectRecord:
    """Ordered features, original stage codes 0..6, and binary quality mask."""

    features: np.ndarray
    stage_labels: np.ndarray
    mask: np.ndarray


@dataclass(frozen=True)
class PreparedRecord:
    """Normalized features and mapped targets; -1 means unscored/unknown stage."""

    features: np.ndarray
    targets: np.ndarray
    mask: np.ndarray
    mean: np.ndarray
    variance: np.ndarray
    median: np.ndarray


@dataclass(frozen=True)
class WindowedRecord:
    """Complete windows, original epoch offsets, and explicit scoring coverage."""

    features: np.ndarray
    targets: np.ndarray
    scoring_mask: np.ndarray
    starts: np.ndarray
    coverage: dict[str, int]


def _validate_arrays(features, labels, mask, *, mapped=False):
    features, labels, mask = np.asarray(features), np.asarray(labels), np.asarray(mask)
    if features.dtype != np.dtype("float32") or features.ndim != 2 or features.shape[1] != len(FEATURE_ORDER):
        raise ValueError("features must be float32 [time, 14]")
    if labels.shape != (len(features),) or labels.dtype.kind not in "iu":
        raise ValueError("stage labels must be an integer vector matching features")
    allowed = (-1, 0, 1, 2) if mapped else (0, 1, 2, 3, 4, 5, 6)
    if not np.isin(labels, allowed).all():
        raise ValueError("Unsupported stage labels")
    if mask.shape != (len(features),) or mask.dtype.kind not in "biu" or not np.isin(mask, (0, 1)).all():
        raise ValueError("mask must be a binary integer or boolean vector matching features")
    return features, labels, mask


def read_subject(path: str | Path) -> SubjectRecord:
    """Read three self-contained HDF5 datasets and validate their schema.

    Additional datasets (for example apnea labels) are ignored. Required fields
    cannot use indirect links, virtual datasets, or external storage. Numerical
    eligibility and normalization are checked by :func:`prepare_subject`.
    """
    import h5py

    arrays = []
    with h5py.File(path, "r") as stream:
        for name in ("features", "stage_labels", "mask"):
            if name not in stream or not isinstance(stream.get(name, getlink=True), h5py.HardLink):
                raise ValueError(f"Missing or indirect HDF5 dataset: {name}")
            dataset = stream[name]
            if not isinstance(dataset, h5py.Dataset) or dataset.is_virtual or dataset.external:
                raise ValueError(f"Expected a self-contained HDF5 dataset: {name}")
            arrays.append(dataset[:])
    features, labels, mask = _validate_arrays(*arrays)
    return SubjectRecord(features, labels, mask)


def prepare_subject(record: SubjectRecord) -> PreparedRecord:
    """Copy, map labels, impute invalid rows, and normalize the complete record.

    Finite quality-valid rows define float32 median/mean/population variance.
    Invalid rows may contain any feature values: they are entirely replaced.
    Unknown stage 6 remains at its original position and participates in feature
    statistics when quality-valid. It never becomes a supervised target.
    """
    features, labels, mask = _validate_arrays(record.features, record.stage_labels, record.mask)
    valid = mask == 1
    if not valid.any():
        raise ValueError("Record has no quality-valid epochs")
    if not np.isfinite(features[valid]).all():
        raise ValueError("Quality-valid features must be finite")
    valid_features = features[valid]
    # Deliberately preserve historical float32 arithmetic, including epsilon.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        median = np.nanmedian(valid_features, axis=0)
        mean = np.nanmean(valid_features, axis=0)
        variance = np.nanvar(valid_features, axis=0)
        prepared = features.copy()
        prepared[~valid] = median
        prepared = (prepared - mean) / np.sqrt(variance + 1e-6)
    if not all(np.isfinite(value).all() for value in (median, mean, variance, prepared)):
        raise ValueError("Nonfinite normalization statistics or normalized features")
    return PreparedRecord(prepared, _CLASS_MAP[labels], mask.astype(bool, copy=True), mean, variance, median)


def window_subject(record: PreparedRecord) -> WindowedRecord:
    """Return nonoverlapping 240-epoch windows without stitching gaps or records.

    The final incomplete window is omitted, with coverage retained explicitly.
    Short records return zero windows. Scoring requires a known target and a
    quality-valid epoch; excluded positions still provide temporal context.
    """
    features, targets, mask = _validate_arrays(record.features, record.targets, record.mask, mapped=True)
    if not np.isfinite(features).all():
        raise ValueError("Normalized features must be finite")
    count = len(features) // WINDOW_EPOCHS
    retained = count * WINDOW_EPOCHS
    eligible = (mask == 1) & (targets >= 0)
    scored = int(eligible[:retained].sum())
    return WindowedRecord(
        features[:retained].reshape(count, WINDOW_EPOCHS, len(FEATURE_ORDER)).copy(),
        targets[:retained].reshape(count, WINDOW_EPOCHS).copy(),
        eligible[:retained].reshape(count, WINDOW_EPOCHS).copy(),
        np.arange(count, dtype=np.int64) * WINDOW_EPOCHS,
        {
            "input_epochs": len(features),
            "window_count": count,
            "windowed_epochs": retained,
            "remainder_epochs": len(features) - retained,
            "eligible_epochs": int(eligible.sum()),
            "scored_epochs": scored,
            "unscored_epochs": len(features) - scored,
        },
    )

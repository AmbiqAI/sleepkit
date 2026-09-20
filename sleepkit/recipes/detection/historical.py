"""The reviewed SD-2-TCN-SM feature and runtime adapter.

This module deliberately keeps the historical preprocessing separate from the
v2/v3 detector recipe.  The old model receives whole-record normalized legacy
features and emits quantized values from its preserved TFLite graph.
"""

from pathlib import Path
import warnings

import numpy as np

from sleepkit.artifacts import baselines
from sleepkit.artifacts.package import sha256

from .hdf5 import require_self_contained
from .preprocessing import Features, validate_input


WINDOW = 12
STRIDE = 6
FEATURES = 5
INPUT_SCALE = 0.11977417767047882
INPUT_ZERO_POINT = -95
OUTPUT_SCALE = 0.00390625
OUTPUT_ZERO_POINT = -128
MODEL_INPUT_SHAPE = (1, 240, FEATURES)
MODEL_OUTPUT_SHAPE = (1, 240, 2)
MODEL_INPUT_SIGNATURE = (-1, 240, FEATURES)
MODEL_OUTPUT_SIGNATURE = (-1, 240, 2)


def reconstruct_legacy_features(data, sample_time=None):
    """Reconstruct FS-W-A-5 features without reading historical labels.

    The duration and tail deliberately follow ``fs_w_a_5.py`` exactly.  A row
    is valid only when all five resulting features are finite; NaN rows are
    preserved for the caller's context mask and are never imputed.
    """
    data, _ = validate_input(data, sample_time)
    n_samples = data.shape[1]
    duration = max(0, int((n_samples / 0.2) * 0.2) - WINDOW)
    count = max(0, (duration - WINDOW) // STRIDE)
    starts = np.arange(count, dtype=np.int64) * STRIDE
    if count:
        windows = np.lib.stride_tricks.sliding_window_view(data[:, :duration], WINDOW, axis=1)
        windows = windows[:, ::STRIDE, :][:, :count, :]
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            values = np.stack(
                (
                    np.cos(2 * np.pi * np.nanmean(windows[0] / 86400, axis=-1)),
                    np.nanmean(windows[1], axis=-1),
                    np.nanstd(windows[1], axis=-1),
                    np.nanmean(windows[2], axis=-1),
                    np.nanstd(windows[2], axis=-1),
                ),
                axis=1,
            ).astype(np.float32)
    else:
        values = np.empty((0, FEATURES), dtype=np.float32)
    if np.isinf(values).any():
        raise ValueError("Legacy feature reconstruction produced infinity")
    if count and not np.isfinite(values).any(axis=0).all():
        raise ValueError("Legacy feature column has no finite values")
    return Features(values, np.isfinite(values).all(axis=1), starts + WINDOW - 1)


def _exact_features(actual, expected):
    if actual.shape != expected.shape or actual.dtype != np.float32:
        return False
    equal = (actual == expected) | (np.isnan(actual) & np.isnan(expected))
    return bool(equal.all())


def _normalize(values):
    if values.dtype != np.float32 or np.isinf(values).any():
        raise ValueError("Stored legacy features must be finite float32 or NaN")
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(values, axis=0, dtype=np.float32)
        variance = np.nanvar(values, axis=0, dtype=np.float32)
    if not np.isfinite(mean).all() or not np.isfinite(variance).all():
        raise ValueError("Legacy feature column has no finite values")
    scale = np.sqrt(variance + np.float32(1e-6)).astype(np.float32)
    normalized = ((values - mean) / scale).astype(np.float32)
    valid = np.isfinite(normalized).all(axis=1)
    return normalized, valid, mean, variance, scale


def read_verified_features(path, data, sample_time=None):
    """Read and verify legacy features, then apply whole-record normalization.

    Only ``features`` and ``mask`` are read from the HDF5 file.  The source
    file is hash-checked before and after the read, and stored features must
    exactly match :func:`reconstruct_legacy_features`, including NaN positions.
    """
    import h5py

    path = Path(path)
    before = sha256(path)
    with h5py.File(path, "r") as stream:
        require_self_contained(stream)
        stored = np.asarray(stream["features"])
        mask = np.asarray(stream["mask"])
    after = sha256(path)
    if before != after:
        raise ValueError("Legacy feature file changed during read")
    if stored.ndim != 2 or stored.shape[1] != FEATURES or stored.dtype != np.float32:
        raise ValueError("Legacy features must be float32 with shape [frames, 5]")
    if mask.shape != (stored.shape[0],) or not np.array_equal(mask, np.ones(mask.shape, dtype=mask.dtype)):
        raise ValueError("Legacy feature mask must be all ones")
    expected = reconstruct_legacy_features(data, sample_time)
    if not _exact_features(stored, expected.values):
        raise ValueError("Stored legacy features differ from reconstructed features")
    normalized, valid, mean, variance, scale = _normalize(stored)
    if sha256(path) != before:
        raise ValueError("Legacy feature file changed during processing")
    result = Features(normalized, valid, expected.ends)
    metadata = {
        "feature_file_sha256": before,
        "frame_count": int(stored.shape[0]),
        "normalization": {
            "dtype": "float32",
            "scope": "whole_record",
            "epsilon": 1e-6,
            "mean": mean.tolist(),
            "variance": variance.tolist(),
            "scale": scale.tolist(),
        },
        "mask": "all_ones",
    }
    return result, metadata


def _quantization(details, shape, signature, scale, zero_point, label):
    if tuple(int(x) for x in details["shape"]) != shape:
        raise ValueError(f"Unexpected historical {label} allocated shape")
    if tuple(int(x) for x in details["shape_signature"]) != signature:
        raise ValueError(f"Unexpected historical {label} shape signature")
    if np.dtype(details["dtype"]) != np.dtype(np.int8):
        raise ValueError(f"Historical {label} tensor must be int8")
    parameters = details.get("quantization_parameters", {})
    scales = np.asarray(parameters.get("scales", []))
    zeros = np.asarray(parameters.get("zero_points", []))
    if scales.shape != (1,) or zeros.shape != (1,) or float(scales[0]) != scale or int(zeros[0]) != zero_point:
        raise ValueError(f"Unexpected historical {label} quantization")


class SD2Runtime:
    """Lazy LiteRT runtime pinned to the reviewed SD-2-TCN-SM model bytes."""

    def __init__(self, model_path):
        from ai_edge_litert.interpreter import Interpreter

        self.model_path = Path(model_path)
        self.model_sha256 = baselines.HASHES["model.tflite"]
        self.verify_unchanged()
        self._runner = Interpreter(model_path=str(self.model_path), num_threads=1)
        self._runner.allocate_tensors()
        inputs = self._runner.get_input_details()
        outputs = self._runner.get_output_details()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Historical runtime requires one input and one output")
        _quantization(inputs[0], MODEL_INPUT_SHAPE, MODEL_INPUT_SIGNATURE, INPUT_SCALE, INPUT_ZERO_POINT, "input")
        _quantization(outputs[0], MODEL_OUTPUT_SHAPE, MODEL_OUTPUT_SIGNATURE, OUTPUT_SCALE, OUTPUT_ZERO_POINT, "output")
        self._input = inputs[0]
        self._output = outputs[0]
        self.verify_unchanged()

    def verify_unchanged(self):
        if sha256(self.model_path) != self.model_sha256:
            raise ValueError("Historical model bytes changed")

    def predict(self, values):
        values = np.asarray(values, dtype=np.float32)
        if values.shape != MODEL_INPUT_SHAPE[1:] or not np.isfinite(values).all():
            raise ValueError("Historical runtime expects finite float32 features [240, 5]")
        self.verify_unchanged()
        scaled = np.rint(values / np.float32(INPUT_SCALE) + np.float32(INPUT_ZERO_POINT))
        clipped = (scaled < -128) | (scaled > 127)
        quantized = np.clip(scaled, -128, 127).astype(np.int8)
        self._runner.set_tensor(self._input["index"], quantized[None])
        self._runner.invoke()
        output = np.asarray(self._runner.get_tensor(self._output["index"]))
        self.verify_unchanged()
        if output.shape != MODEL_OUTPUT_SHAPE or output.dtype != np.int8:
            raise ValueError("Historical runtime returned an unexpected tensor")
        return output[0].copy(), int(clipped.sum())

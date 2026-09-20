"""Focused checks for the preserved SD-2-TCN-SM preprocessing and runtime."""

import hashlib
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from sleepkit.artifacts import baselines
from sleepkit.recipes.detection.historical import (
    INPUT_SCALE,
    INPUT_ZERO_POINT,
    MODEL_INPUT_SHAPE,
    MODEL_INPUT_SIGNATURE,
    MODEL_OUTPUT_SHAPE,
    MODEL_OUTPUT_SIGNATURE,
    OUTPUT_SCALE,
    OUTPUT_ZERO_POINT,
    SD2Runtime,
    read_verified_features,
    reconstruct_legacy_features,
)


def sensor_data(samples):
    return np.stack(
        [np.arange(samples) * 5, np.arange(samples) + 1, np.arange(samples) * 2], axis=0
    ).astype(np.float32)


def literal_legacy(data):
    duration = max(0, int((data.shape[1] / 0.2) * 0.2) - 12)
    count = max(0, (duration - 12) // 6)
    values = np.full((count, 5), np.nan, np.float32)
    for index, start in enumerate(range(0, duration - 12, 6)):
        if index >= count:
            break
        window = data[:, start : start + 12]
        with np.errstate(all="ignore"):
            values[index] = [
                np.cos(2 * np.pi * np.nanmean(window[0] / 86400)),
                np.nanmean(window[1]),
                np.nanstd(window[1]),
                np.nanmean(window[2]),
                np.nanstd(window[2]),
            ]
    return values


def test_legacy_tail_and_vectorized_math_match_literal_loop():
    for samples in (0, 11, 24, 36, 37, 60):
        data = sensor_data(samples)
        actual = reconstruct_legacy_features(data)
        np.testing.assert_array_equal(actual.values, literal_legacy(data), strict=True)
        assert actual.ends.tolist() == [11 + 6 * i for i in range(actual.values.shape[0])]


def test_legacy_midnight_formula_and_nan_validity():
    data = sensor_data(60)
    data[0] = (86395 + np.arange(60) * 5) % 86400
    data[1, :12] = np.nan
    result = reconstruct_legacy_features(data)
    expected = np.cos(2 * np.pi * np.nanmean(data[0, :12] / 86400))
    assert result.values[0, 0] == pytest.approx(expected)
    assert not result.valid[0]
    assert np.isnan(result.values[0, 1:3]).all()
    assert result.valid[1:].all()


def test_reconstruct_fails_for_infinity_and_all_invalid_column():
    data = sensor_data(36)
    data[1, 0] = np.inf
    with pytest.raises(ValueError, match="infinity"):
        reconstruct_legacy_features(data)
    data = sensor_data(36)
    data[1] = np.nan
    with pytest.raises(ValueError, match="no finite"):
        reconstruct_legacy_features(data)


def test_verified_features_normalize_whole_record_and_ignore_labels(tmp_path):
    data = sensor_data(60)
    expected = reconstruct_legacy_features(data)
    path = tmp_path / "features.h5"
    with h5py.File(path, "w") as stream:
        stream["features"] = expected.values
        stream["mask"] = np.ones(len(expected.values), dtype=np.int32)
        stream["detect_labels"] = np.array(["malformed labels are never read"], dtype="S")
    actual, metadata = read_verified_features(path, data)
    with np.errstate(all="ignore"):
        mean = np.nanmean(expected.values, axis=0, dtype=np.float32)
        variance = np.nanvar(expected.values, axis=0, dtype=np.float32)
    np.testing.assert_array_equal(actual.values, ((expected.values - mean) / np.sqrt(variance + np.float32(1e-6))).astype(np.float32))
    assert actual.valid.all()
    assert metadata["frame_count"] == len(expected.values)
    assert metadata["feature_file_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    np.testing.assert_array_equal(metadata["normalization"]["mean"], mean)
    np.testing.assert_array_equal(metadata["normalization"]["variance"], variance)


def test_verified_features_reject_mismatch_and_mask(tmp_path):
    data = sensor_data(36)
    expected = reconstruct_legacy_features(data)
    path = tmp_path / "features.h5"
    with h5py.File(path, "w") as stream:
        stream["features"] = expected.values
        stream["mask"] = np.ones(len(expected.values), dtype=np.int32)
    with h5py.File(path, "a") as stream:
        stream["features"][0, 0] += 1
    with pytest.raises(ValueError, match="differ"):
        read_verified_features(path, data)
    with h5py.File(path, "w") as stream:
        stream["features"] = expected.values
        stream["mask"] = np.array([1, 0], dtype=np.int32)
    with pytest.raises(ValueError, match="mask"):
        read_verified_features(path, data)


def test_verified_features_reject_external_hdf5_payload(tmp_path):
    data = sensor_data(36)
    expected = reconstruct_legacy_features(data)
    payload = tmp_path / "payload.h5"
    path = tmp_path / "features.h5"
    with h5py.File(payload, "w") as stream:
        stream["features"] = expected.values
        stream["mask"] = np.ones(len(expected.values), dtype=np.int32)
    with h5py.File(path, "w") as stream:
        stream["features"] = h5py.ExternalLink(str(payload), "features")
        stream["mask"] = h5py.ExternalLink(str(payload), "mask")
    with pytest.raises(ValueError, match="self-contained"):
        read_verified_features(path, data)


class FakeInterpreter:
    offset = 0

    def __init__(self, model_path, num_threads):
        assert num_threads == 1
        self.model_path = model_path
        self.input = None

    def allocate_tensors(self):
        pass

    def get_input_details(self):
        return [
            {
                "name": "serving_default_input:0",
                "shape": np.array(MODEL_INPUT_SHAPE),
                "shape_signature": np.array(MODEL_INPUT_SIGNATURE),
                "dtype": np.int8,
                "index": 0,
                "quantization_parameters": {"scales": np.array([INPUT_SCALE], np.float32), "zero_points": np.array([INPUT_ZERO_POINT])},
            }
        ]

    def get_output_details(self):
        return [
            {
                "name": "StatefulPartitionedCall_1:0",
                "shape": np.array(MODEL_OUTPUT_SHAPE),
                "shape_signature": np.array(MODEL_OUTPUT_SIGNATURE),
                "dtype": np.int8,
                "index": 1,
                "quantization_parameters": {"scales": np.array([OUTPUT_SCALE], np.float32), "zero_points": np.array([OUTPUT_ZERO_POINT])},
            }
        ]

    def set_tensor(self, index, value):
        self.input = np.asarray(value)

    def invoke(self):
        pass

    def get_tensor(self, index):
        return np.zeros(MODEL_OUTPUT_SHAPE, dtype=np.int8)


def test_runtime_pins_model_and_reports_input_clipping(tmp_path, monkeypatch):
    model = tmp_path / "model.tflite"
    model.write_bytes(b"historical model")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    monkeypatch.setitem(baselines.HASHES, "model.tflite", digest)
    monkeypatch.setitem(sys.modules, "ai_edge_litert.interpreter", SimpleNamespace(Interpreter=FakeInterpreter))
    runtime = SD2Runtime(model)
    values = np.zeros(MODEL_INPUT_SHAPE[1:], np.float32)
    values[0, 0] = 100
    output, clipped = runtime.predict(values)
    assert output.shape == MODEL_OUTPUT_SHAPE[1:] and output.dtype == np.int8
    assert clipped == 1
    runtime.verify_unchanged()
    model.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        runtime.verify_unchanged()


def test_runtime_rejects_wrong_signature(tmp_path, monkeypatch):
    model = tmp_path / "model.tflite"
    model.write_bytes(b"historical model")
    monkeypatch.setitem(baselines.HASHES, "model.tflite", hashlib.sha256(model.read_bytes()).hexdigest())

    class BadInterpreter(FakeInterpreter):
        def get_output_details(self):
            details = super().get_output_details()
            details[0]["shape"] = np.array([1, 240, 3])
            return details

    monkeypatch.setitem(sys.modules, "ai_edge_litert.interpreter", SimpleNamespace(Interpreter=BadInterpreter))
    with pytest.raises(ValueError, match="shape"):
        SD2Runtime(model)

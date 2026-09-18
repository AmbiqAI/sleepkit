"""Runtime evidence replay rejects invalid tensors and changed predictions."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from sleepkit.artifacts import Artifact, TensorSpec, stage_bundle
from sleepkit.artifacts import runtime


@pytest.fixture
def fake_runtime(monkeypatch):
    class Interpreter:
        offset = 0

        def __init__(self, model_path):
            self.value = None

        def allocate_tensors(self):
            pass

        def get_input_details(self):
            return [
                {
                    "name": "input",
                    "shape": np.array([1, 3]),
                    "shape_signature": np.array([-1, 3]),
                    "dtype": np.int8,
                    "index": 0,
                    "quantization_parameters": {"scales": np.array([0.25]), "zero_points": np.array([0])},
                }
            ]

        def get_output_details(self):
            return [{**self.get_input_details()[0], "name": "output", "index": 1}]

        def set_tensor(self, index, value):
            self.value = value

        def invoke(self):
            pass

        def get_tensor(self, index):
            return self.value + self.offset

    monkeypatch.setitem(sys.modules, "ai_edge_litert.interpreter", SimpleNamespace(Interpreter=Interpreter))
    monkeypatch.setattr(runtime, "version", lambda name: "test")
    return Interpreter


def model_fixture(tmp_path, *, scale=0.25):
    path = tmp_path / "model.tflite"
    path.write_bytes(b"runtime fixture")
    return Artifact(
        path,
        path.name,
        "model",
        "tflite",
        "test",
        (TensorSpec("input", (None, 3), "int8", "test", scale, 0),),
        (TensorSpec("output", (None, 3), "int8", "test", 0.25, 0),),
    )


def test_runtime_replay_detects_prediction_drift(tmp_path, fake_runtime):
    model = model_fixture(tmp_path)
    reference = tmp_path / "reference.npz"
    check = runtime.create_reference(model, reference)
    bundle = stage_bundle(
        tmp_path / "bundle",
        title="test",
        card="test",
        checks=[check],
        artifacts=[model, Artifact(reference, reference.name, "reference", "npz", "synthetic")],
    )
    assert runtime.replay_bundle(bundle)[0]["status"] == "passed"
    fake_runtime.offset = 2
    with pytest.raises(ValueError, match="differs from reference"):
        runtime.replay_bundle(bundle)


def test_runtime_rejects_quantization_mismatch(tmp_path, fake_runtime):
    model = model_fixture(tmp_path, scale=0.5)
    with pytest.raises(ValueError, match="quantization differs"):
        runtime.create_reference(model, tmp_path / "reference.npz")

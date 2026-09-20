"""Detection runtime quantization, declared contracts, and sensor inference."""

from dataclasses import replace
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from sleepkit.artifacts import Artifact, Check, TensorSpec, stage_bundle
from sleepkit.artifacts.package import sha256, write_json
from sleepkit.recipes.detection.inference import predict
from sleepkit.recipes.detection.preprocessing import Normalizer, SPEC, V2_SPEC
from sleepkit.recipes.detection.runtime import DetectionRuntime
from sleepkit.recipes.detection.split import TARGET


def fake_runtime(monkeypatch, *, integer=True, context=2, mutate=None, create_mutation=None):
    dtype = np.int8 if integer else np.float32
    def tensor(name, index, width, scale, zero):
        return {"name": name, "index": index, "dtype": dtype,
                "shape": np.array([1, context, width]), "shape_signature": np.array([1, context, width]),
                "quantization_parameters": {
                    "scales": np.array([scale], dtype=np.float32) if integer else np.array([], dtype=np.float32),
                    "zero_points": np.array([zero], dtype=np.int32) if integer else np.array([], dtype=np.int32),
                    "quantized_dimension": 0}}
    inputs, outputs = [tensor("features", 0, 5, .5, 0)], [tensor("logits", 1, 2, .25, -3)]
    state = SimpleNamespace(inputs=inputs, outputs=outputs, encoded=None,
                            raw=np.tile(np.array([[-3, 5]], dtype=dtype), (1, context, 1)), paths=[])
    if mutate:
        mutate(state)
    class Interpreter:
        def __init__(self, *, model_path, num_threads):
            assert num_threads == 1
            state.paths.append(model_path)
            self.path = model_path
        def allocate_tensors(self):
            if create_mutation:
                create_mutation(self.path)
        def get_input_details(self):
            return state.inputs
        def get_output_details(self):
            return state.outputs
        def set_tensor(self, index, values):
            assert index == 0
            state.encoded = values.copy()
        def invoke(self):
            pass
        def get_tensor(self, index):
            assert index == 1
            return state.raw.copy()
    module = ModuleType("ai_edge_litert.interpreter")
    module.Interpreter = Interpreter
    monkeypatch.setitem(sys.modules, "ai_edge_litert.interpreter", module)
    return state


def model(tmp_path):
    path = tmp_path / "model.tflite"
    path.write_bytes(b"synthetic model")
    return path


def artifact(path, *, integer=True, context=2):
    dtype = "int8" if integer else "float32"
    i = TensorSpec("features", (1, context, 5), dtype, "normalized features", .5 if integer else None, 0 if integer else None)
    o = TensorSpec("logits", (1, context, 2), dtype, "logits", .25 if integer else None, -3 if integer else None)
    return Artifact(path, path.name, "model", "tflite", "test fixture", (i,), (o,))


def test_int8_rounding_clipping_and_dequantized_logits(tmp_path, monkeypatch):
    state = fake_runtime(monkeypatch)
    state.raw = np.array([[[-128, 127], [5, 5]]], dtype=np.int8)
    path = model(tmp_path)
    runtime = DetectionRuntime(path, 2, artifact=artifact(path))
    values = np.array([[-1000, -64.25, -.75, -.25, .25], [.75, 63.25, 63.75, 1000, 0]], np.float32)
    logits, counters = runtime.predict(values)
    np.testing.assert_array_equal(state.encoded, [[[-128, -128, -2, 0, 0], [2, 126, 127, 127, 0]]])
    np.testing.assert_array_equal(logits, [[-31.25, 32.5], [2., 2.]])
    assert logits.dtype == np.float32
    assert counters["input_values"] == 10
    assert counters["clipped_input_values"] == 3
    assert counters["argmax_ties"] == 1
    assert counters["output_saturated_values"] == 2
    np.testing.assert_array_equal(counters["raw_output"], state.raw[0])
    assert runtime.model_path == path
    assert runtime.input_details["name"] == "features"
    assert runtime.output_details["name"] == "logits"


def test_float32_passthrough_has_no_probability_transform(tmp_path, monkeypatch):
    state = fake_runtime(monkeypatch, integer=False)
    path = model(tmp_path)
    runtime = DetectionRuntime(path, 2, artifact=artifact(path, integer=False))
    values = np.arange(10, dtype=np.float32).reshape(2, 5)
    logits, counters = runtime.predict(values)
    np.testing.assert_array_equal(state.encoded[0], values)
    np.testing.assert_array_equal(logits, [[-3, 5], [-3, 5]])
    assert counters["clipped_input_values"] == counters["output_saturated_values"] == 0


@pytest.mark.parametrize("scale,zeros", [([], [0]), ([0], [0]), ([-1], [0]), ([np.nan], [0]),
                                       ([np.inf], [0]), ([.5, .5], [0, 0]), ([.5], []),
                                       ([.5], [128]), ([.5], [-129]), ([.5], [0.5]),
                                       ([[.5]], [0]), ([.5], [[0]])])
def test_hostile_int8_quantization_rejected(tmp_path, monkeypatch, scale, zeros):
    def mutate(state):
        state.inputs[0]["quantization_parameters"] = {"scales": np.array(scale), "zero_points": np.array(zeros)}
    fake_runtime(monkeypatch, mutate=mutate)
    with pytest.raises(ValueError):
        DetectionRuntime(model(tmp_path), 2)


@pytest.mark.parametrize("kind", ["input_count", "output_count", "input_shape", "output_shape", "mixed", "uint8", "float64", "float_quant"])
def test_hostile_tensor_contract_rejected(tmp_path, monkeypatch, kind):
    def mutate(state):
        if kind == "input_count":
            state.inputs.append(state.inputs[0].copy())
        elif kind == "output_count":
            state.outputs.clear()
        elif kind == "input_shape":
            state.inputs[0]["shape"] = [1, 2, 4]
        elif kind == "output_shape":
            state.outputs[0]["shape"] = [1, 2, 3]
        elif kind == "mixed":
            state.outputs[0]["dtype"] = np.float32
        elif kind == "float_quant":
            state.inputs[0]["dtype"] = state.outputs[0]["dtype"] = np.float32
        else:
            state.inputs[0]["dtype"] = state.outputs[0]["dtype"] = getattr(np, kind)
    fake_runtime(monkeypatch, mutate=mutate)
    with pytest.raises(ValueError):
        DetectionRuntime(model(tmp_path), 2)


@pytest.mark.parametrize("kind", ["name", "shape", "dtype", "scale", "zero", "missing", "path", "role", "hash"])
def test_declared_artifact_mismatch_rejected(tmp_path, monkeypatch, kind):
    fake_runtime(monkeypatch)
    path = model(tmp_path)
    declared = artifact(path)
    changes = {"name": {"name": "wrong"}, "shape": {"shape": (1, 3, 5)},
               "dtype": {"dtype": "float32", "scale": None, "zero_point": None},
               "scale": {"scale": .125}, "zero": {"zero_point": 1}}
    if kind in changes:
        declared = replace(declared, inputs=(replace(declared.inputs[0], **changes[kind]),))
    elif kind == "missing":
        declared = replace(declared, inputs=())
    elif kind == "path":
        declared = replace(declared, source=tmp_path / "other.tflite")
    elif kind == "hash":
        declared = replace(declared, expected_sha256="0" * 64)
    else:
        declared = replace(declared, role="reference")
    with pytest.raises(ValueError, match="Declared"):
        DetectionRuntime(path, 2, artifact=declared)


def test_dynamic_signature_must_match_declaration(tmp_path, monkeypatch):
    state = fake_runtime(monkeypatch)
    state.inputs[0]["shape_signature"] = [-1, 2, 5]
    path = model(tmp_path)
    declared = artifact(path)
    with pytest.raises(ValueError, match="signature"):
        DetectionRuntime(path, 2, artifact=declared)
    declared = replace(declared, inputs=(replace(declared.inputs[0], shape=(None, 2, 5)),))
    DetectionRuntime(path, 2, artifact=declared)


def test_model_mutation_during_creation_and_afterward_rejected(tmp_path, monkeypatch):
    from pathlib import Path
    fake_runtime(monkeypatch, create_mutation=lambda path: Path(path).write_bytes(b"changed"))
    path = model(tmp_path)
    with pytest.raises(ValueError, match="changed"):
        DetectionRuntime(path, 2)
    fake_runtime(monkeypatch)
    runtime = DetectionRuntime(path, 2)
    path.write_bytes(b"changed again")
    with pytest.raises(ValueError, match="changed"):
        runtime.verify_unchanged()


@pytest.mark.parametrize("kind", ["input_shape", "input_nonfinite", "output_shape", "output_dtype", "output_nonfinite"])
def test_invalid_inference_tensors_rejected(tmp_path, monkeypatch, kind):
    state = fake_runtime(monkeypatch, integer=False)
    runtime = DetectionRuntime(model(tmp_path), 2)
    values = np.ones((2, 5), dtype=np.float32)
    if kind == "input_shape":
        values = values[None]
    elif kind == "input_nonfinite":
        values[0, 0] = np.nan
    elif kind == "output_shape":
        state.raw = np.ones((1, 1, 2), dtype=np.float32)
    elif kind == "output_dtype":
        state.raw = state.raw.astype(np.float64)
    else:
        state.raw[0, 0, 0] = np.inf
    with pytest.raises(ValueError):
        runtime.predict(values)


def make_bundle(tmp_path, *, integer, legacy=False, model_name="model.tflite"):
    path = tmp_path / model_name
    path.write_bytes(b"synthetic model")
    reference = tmp_path / "reference.npz"
    reference.write_bytes(b"opaque synthetic reference")
    recipe = {"recipe": "sleepkit.detection/v1" if legacy else "sleepkit.detection/v2", "output": "logits",
              "class_names": ["WAKE", "SLEEP"] if legacy else TARGET["classes"], "context": 2}
    if not legacy:
        recipe["target"] = TARGET
    write_json(tmp_path / "recipe.json", recipe)
    write_json(tmp_path / "preprocessing.json", Normalizer(np.zeros(5), np.ones(5), 20,
                                                           V2_SPEC if legacy else SPEC).to_dict())
    artifacts = [artifact(path, integer=integer), Artifact(reference, reference.name, "reference", "npz", "test")]
    artifacts += [Artifact(tmp_path / name, name, role, "json", "test") for name, role in (
        ("recipe.json", "recipe_metadata"), ("preprocessing.json", "preprocessing_state"))]
    check = Check(f"runtime_conformance:{model_name}", "passed", "synthetic test fixture",
                  {model_name: sha256(path), reference.name: sha256(reference)})
    bundle = tmp_path / "bundle"
    stage_bundle(bundle, title="test", artifacts=artifacts, card="test", checks=[check],
                 metadata={"sleepkit": {"recipe": recipe["recipe"]}})
    return bundle


@pytest.mark.parametrize("integer,legacy", [(True, False), (False, False), (False, True)])
def test_sensor_inference_preserves_clock_target_and_softmax_once(tmp_path, monkeypatch, integer, legacy):
    fake_runtime(monkeypatch, integer=integer)
    bundle = make_bundle(tmp_path, integer=integer, legacy=legacy)
    data = np.stack([np.arange(24) * 5, np.ones(24), np.ones(24)]).astype(np.float32)
    clock = None if legacy else np.arange(24, dtype=np.int64) * 5
    if clock is not None:
        data[0, 12:] += 3600  # Independent UTC clock supports a local-time jump.
    result = predict(bundle, data, sample_time=clock)
    assert result["target"] == (None if legacy else TARGET)
    assert result["class_names"] == (["WAKE", "SLEEP"] if legacy else TARGET["classes"])
    np.testing.assert_array_equal(result["times"], [55, 85])
    np.testing.assert_array_equal(result["available_at"], [90, 90])
    logits = [0, 2] if integer else [-3, 5]
    np.testing.assert_array_equal(result["logits"], [logits, logits])
    expected = np.exp(np.array(logits) - max(logits))
    expected /= expected.sum()
    np.testing.assert_allclose(result["probabilities"], np.tile(expected, (2, 1)), rtol=1e-6)
    if not legacy:
        clock[12:] += 5
        with pytest.raises(ValueError, match="contiguous"):
            predict(bundle, data, sample_time=clock)


def test_explicit_model_selection_and_undeclared_paths(tmp_path, monkeypatch):
    state = fake_runtime(monkeypatch, integer=False)
    bundle = make_bundle(tmp_path, integer=False, model_name="model.float.tflite")
    data = np.stack([np.arange(24) * 5, np.ones(24), np.ones(24)]).astype(np.float32)
    predict(bundle, data, model_name="model.float.tflite")
    assert state.paths[-1] == str(bundle / "model.float.tflite")
    for name in ("model.tflite", "recipe.json", "../model.float.tflite"):
        with pytest.raises(ValueError, match="declared TFLite"):
            predict(bundle, data, model_name=name)


def test_inference_binds_runtime_to_previously_validated_manifest(tmp_path, monkeypatch):
    from sleepkit.recipes.detection import inference
    fake_runtime(monkeypatch, integer=False)
    bundle = make_bundle(tmp_path, integer=False)
    original = inference.validate_bundle
    def mutate_after_validation(path, **kwargs):
        report = original(path, **kwargs)
        (bundle / "model.tflite").write_bytes(b"changed after manifest validation")
        return report
    monkeypatch.setattr(inference, "validate_bundle", mutate_after_validation)
    data = np.stack([np.arange(24) * 5, np.ones(24), np.ones(24)]).astype(np.float32)
    with pytest.raises(ValueError, match="Declared artifact"):
        predict(bundle, data)

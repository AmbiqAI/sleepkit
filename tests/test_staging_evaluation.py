"""Aggregate staging replay with synthetic HDF5 and real, untrained Keras graphs."""

import hashlib
import importlib.util
import json
import os

import numpy as np
import pytest

from sleepkit.recipes._components import ClassificationAccumulator
from sleepkit.recipes.staging import WindowedRecord
from sleepkit.recipes.staging import evaluation


MISSING_DEPENDENCIES = [name for name in ("tensorflow", "keras", "h5py") if importlib.util.find_spec(name) is None]


@pytest.fixture
def runtime():
    if MISSING_DEPENDENCIES:
        required = any(
            os.environ.get(name) == "1" for name in ("SLEEPKIT_REQUIRE_STAGING", "SLEEPKIT_REQUIRE_DETECTION")
        )
        if required:
            pytest.fail(f"Required staging dependencies missing: {MISSING_DEPENDENCIES}")
        pytest.skip("Staging runtime tests require TensorFlow/Keras and HDF5")
    import keras
    import h5py

    return keras, h5py


def save_constant_model(keras, path):
    inputs = keras.Input((240, 14), dtype="float32")
    outputs = keras.layers.Dense(
        3, kernel_initializer="zeros", bias_initializer=keras.initializers.Constant([2, 0, -1])
    )(inputs)
    keras.Model(inputs, outputs).save(path)


@pytest.fixture
def replay(tmp_path, runtime):
    keras, h5py = runtime
    source = tmp_path / "features"
    source.mkdir()
    for subject, length in (("synthetic-z", 481), ("synthetic-a", 239)):
        features = np.tile(np.arange(length, dtype=np.float32)[:, None], (1, 14))
        labels = np.zeros(length, np.int32)
        mask = np.ones(length, np.int32)
        if length == 481:
            labels[[1, 2, 3, 480]] = [1, 5, 6, 5]
            mask[4] = 0
        with h5py.File(source / f"{subject}.h5", "w") as stream:
            stream["features"], stream["stage_labels"], stream["mask"] = features, labels, mask
    checkpoint = tmp_path / "model.keras"
    save_constant_model(keras, checkpoint)
    cohort = tmp_path / "cohort.json"
    cohort.write_text(json.dumps(["synthetic-z", "synthetic-a"]))
    return source, checkpoint, cohort, tmp_path / "report.json"


def test_saved_model_replay_has_independent_metrics_coverage_and_private_provenance(replay):
    source, checkpoint, cohort, output = replay
    report = evaluation.evaluate(source, checkpoint, cohort, output, batch_size=1)
    assert json.loads(output.read_text()) == report
    assert report["coverage"] == {
        "subjects": 2,
        "input_epochs": 720,
        "window_count": 2,
        "windowed_epochs": 480,
        "remainder_epochs": 240,
        "eligible_epochs": 718,
        "scored_epochs": 478,
        "unscored_epochs": 242,
    }
    metrics = report["metrics"]
    assert metrics["count"] == 478
    assert metrics["confusion_matrix"] == [[476, 0, 0], [1, 0, 0], [1, 0, 0]]
    assert metrics["class_support"] == [476, 1, 1]
    assert metrics["accuracy"] == pytest.approx(476 / 478)
    assert metrics["macro_f1"] == pytest.approx((952 / 954) / 3)
    expected_loss = np.log(np.exp(2) + 1 + np.exp(-1)) - (476 * 2 - 1) / 478
    assert metrics["cross_entropy"] == pytest.approx(expected_loss)
    assert report["provenance"]["checkpoint"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert report["provenance"]["cohort"] == hashlib.sha256(cohort.read_bytes()).hexdigest()
    assert report["provenance"]["subject_set_sha256"] == hashlib.sha256(b"synthetic-a\nsynthetic-z").hexdigest()
    assert report["class_order"] == ["WAKE", "NREM", "REM"]
    assert report["protocol"]["historical_training_membership"] == "unverified"
    for private in (str(source), str(checkpoint), "synthetic-a", "synthetic-z"):
        assert private not in output.read_text()
    second = evaluation.evaluate(source, checkpoint, cohort, output.with_name("second.json"), batch_size=16)
    assert second["metrics"] == metrics
    assert second["coverage"] == report["coverage"]


def test_excluded_epoch_still_changes_neighbor_prediction_after_checkpoint_reload(tmp_path, runtime):
    keras, _ = runtime
    inputs = keras.Input((240, 14))
    layer = keras.layers.Conv1D(3, 3, padding="same", use_bias=False)
    model = keras.Model(inputs, layer(inputs))
    weights = np.zeros((3, 14, 3), np.float32)
    weights[0, 0, :2] = [-1, 1]  # Previous epoch determines the two-class preference.
    layer.set_weights([weights])
    path = tmp_path / "context.keras"
    model.save(path)
    model = keras.models.load_model(path, compile=False, safe_mode=True)
    evaluation.validate_model(model)
    values = np.zeros((2, 240, 14), np.float32)
    values[0, 0, 0] = 2
    targets = np.zeros((2, 240), np.int32)
    targets[0, :2] = [-1, 1]
    mask = np.ones((2, 240), bool)
    mask[0, [0, 2]] = False
    windows = WindowedRecord(values, targets, mask, np.array([0, 240]), {})
    accumulator = ClassificationAccumulator(3)
    evaluation.evaluate_windows(model, windows, accumulator, batch_size=1)
    assert accumulator.result()["confusion_matrix"] == [[477, 0, 0], [0, 1, 0], [0, 0, 0]]
    # Removing the unscored context value changes the known neighbor's result.
    values[0, 0, 0] = 0
    changed = ClassificationAccumulator(3)
    evaluation.evaluate_windows(model, windows, changed)
    assert changed.result()["confusion_matrix"] == [[477, 0, 0], [1, 0, 0], [0, 0, 0]]


@pytest.mark.parametrize(
    "kind", ["empty", "object", "duplicate", "traversal", "suffix", "number", "separator", "blank"]
)
def test_malformed_cohort_rejected(tmp_path, kind):
    value = {
        "empty": [],
        "object": {},
        "duplicate": ["a", "a"],
        "traversal": ["../a"],
        "suffix": ["a.h5"],
        "number": [1],
        "separator": ["a/b"],
        "blank": [""],
    }[kind]
    path = tmp_path / "cohort.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Cohort"):
        evaluation.read_cohort(path)


def test_cohort_is_sorted_and_existing_output_is_preserved(tmp_path):
    path = tmp_path / "cohort.json"
    path.write_text('["z-2", "a_1"]')
    assert evaluation.read_cohort(path) == ["a_1", "z-2"]
    output = tmp_path / "report.json"
    output.write_text("preserve")
    with pytest.raises(FileExistsError):
        evaluation.evaluate(tmp_path, "absent.keras", path, output)
    assert output.read_text() == "preserve"


@pytest.mark.parametrize(
    "kind", ["features", "length", "classes", "input_dtype", "output_dtype", "softmax", "fixed_batch"]
)
def test_saved_malformed_model_contract_rejected_without_report(replay, runtime, kind):
    keras, _ = runtime
    source, checkpoint, cohort, output = replay
    shape = (239 if kind == "length" else 240, 13 if kind == "features" else 14)
    inputs = keras.Input(
        shape, dtype="float64" if kind == "input_dtype" else "float32", batch_size=1 if kind == "fixed_batch" else None
    )
    outputs = keras.layers.Dense(
        2 if kind == "classes" else 3,
        activation="softmax" if kind == "softmax" else "linear",
        dtype="float64" if kind == "output_dtype" else "float32",
    )(inputs)
    keras.Model(inputs, outputs).save(checkpoint)
    with pytest.raises(ValueError, match="float32"):
        evaluation.evaluate(source, checkpoint, cohort, output)
    assert not output.exists()


@pytest.mark.parametrize("activation", ["linear", "softmax"])
def test_historical_conv_then_reshape_checks_producer_after_reload(tmp_path, runtime, activation):
    keras, _ = runtime
    inputs = keras.Input((240, 14))
    expanded = keras.layers.Reshape((1, 240, 14))(inputs)
    logits = keras.layers.Conv2D(3, (1, 1), activation=activation)(expanded)
    model = keras.Model(inputs, keras.layers.Reshape((240, 3))(logits))
    checkpoint = tmp_path / "reshape.keras"
    model.save(checkpoint)
    reloaded = keras.models.load_model(checkpoint, compile=False, safe_mode=True)
    if activation == "linear":
        evaluation.validate_model(reloaded)
    else:
        with pytest.raises(ValueError, match="linear"):
            evaluation.validate_model(reloaded)


@pytest.mark.parametrize("kind", ["checkpoint", "cohort", "data", "implementation"])
def test_input_mutation_during_evaluation_rejected_without_report(replay, runtime, monkeypatch, kind):
    _, h5py = runtime
    source, checkpoint, cohort, output = replay
    implementation = output.with_name("synthetic-implementation.py")
    implementation.write_text("# fixture\n")
    monkeypatch.setattr(evaluation, "implementation_files", lambda _: {"fixture.py": implementation})
    original = evaluation.evaluate_windows

    def mutate(*args, **kwargs):
        original(*args, **kwargs)
        if kind == "data":
            with h5py.File(source / "synthetic-z.h5", "a") as stream:
                stream.attrs["changed"] = 1
        else:
            path = {"checkpoint": checkpoint, "cohort": cohort, "implementation": implementation}[kind]
            with path.open("ab") as stream:
                stream.write(b" ")

    monkeypatch.setattr(evaluation, "evaluate_windows", mutate)
    with pytest.raises(ValueError, match="changed"):
        evaluation.evaluate(source, checkpoint, cohort, output)
    assert not output.exists()


@pytest.mark.parametrize("kind", ["unknown", "invalid", "short", "missing"])
def test_unscorable_or_missing_cohort_does_not_write_report(replay, runtime, kind):
    _, h5py = runtime
    source, checkpoint, cohort, output = replay
    cohort.write_text(json.dumps(["synthetic-a" if kind == "short" else "synthetic-z"]))
    path = source / "synthetic-z.h5"
    if kind == "missing":
        path.unlink()
    elif kind != "short":
        with h5py.File(path, "a") as stream:
            stream["stage_labels" if kind == "unknown" else "mask"][:] = 6 if kind == "unknown" else 0
    with pytest.raises(ValueError):
        evaluation.evaluate(source, checkpoint, cohort, output)
    assert not output.exists()


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_invalid_batch_size_rejected_before_io(tmp_path, batch_size):
    output = tmp_path / "report.json"
    with pytest.raises(ValueError, match="batch_size"):
        evaluation.evaluate(tmp_path, "absent.keras", "absent.json", output, batch_size=batch_size)
    assert not output.exists()


@pytest.mark.parametrize("kind", ["nonfinite", "wrong_shape", "wrong_dtype"])
def test_runtime_logits_checked_even_at_unscored_positions(runtime, kind):
    values = np.zeros((1, 240, 14), np.float32)
    mask = np.ones((1, 240), bool)
    mask[0, 0] = False
    windows = WindowedRecord(values, np.zeros((1, 240), np.int32), mask, np.array([0]), {})

    def model(features, training=False):
        scores = np.zeros((len(features), 240, 2 if kind == "wrong_shape" else 3), np.float32)
        if kind == "nonfinite":
            scores[0, 0, 0] = np.nan
        return scores.astype(np.float64) if kind == "wrong_dtype" else scores

    model.input = runtime[0].Input((240, 14))
    with pytest.raises(ValueError):
        evaluation.evaluate_windows(model, windows, ClassificationAccumulator(3))

"""Keras staging training, data isolation and actual archive consumer checks."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from sleepkit.artifacts import validate_bundle
from sleepkit.artifacts.package import write_json
from sleepkit.recipes.staging import recipe
from sleepkit.recipes.staging.data import PreparedPartition, prepare_partition
from sleepkit.recipes.staging.split import create_split


@pytest.fixture
def runtime():
    backend = os.environ.get("KERAS_BACKEND", "tensorflow")
    missing = [name for name in ("keras", "h5py", backend) if importlib.util.find_spec(name) is None]
    if missing:
        if any(os.environ.get(name) == "1" for name in ("SLEEPKIT_REQUIRE_DETECTION", "SLEEPKIT_REQUIRE_EDGE")):
            pytest.fail(f"Required staging runtime missing: {missing}")
        pytest.skip(f"Staging runtime unavailable: {missing}")
    import keras
    import h5py

    return keras, h5py


@pytest.fixture
def inputs(tmp_path, runtime):
    _, h5py = runtime
    source = tmp_path / "source"
    source.mkdir()
    subjects = [f"private-person-{n}" for n in range(5)]
    for n, subject in enumerate(subjects):
        rng = np.random.default_rng(n)
        features = rng.normal(size=(481, 14)).astype(np.float32)
        labels = np.arange(481, dtype=np.int32) % 6
        labels[3] = 6
        mask = np.ones(481, np.int32)
        mask[[1, 6]] = 0
        with h5py.File(source / f"{subject}.h5", "w") as stream:
            stream["features"], stream["stage_labels"], stream["mask"] = features, labels, mask
    manifest = create_split(subjects, dataset="synthetic-contract", validation_count=1, test_count=1)
    split = tmp_path / "split.json"
    write_json(split, manifest)
    return source, split, manifest


def prepared(inputs):
    root, _, manifest = inputs
    return {name: prepare_partition(root, subjects) for name, subjects in manifest["partitions"].items()}


def test_prepared_coverage_safe_sparse_targets_and_private_coordinates(inputs):
    parts = prepared(inputs)
    item = parts["train"]
    x, y, weights = item.fit_arrays()
    assert x.shape == (6, 240, 14)
    assert weights.sum() == 3 * 477
    assert np.all(y[~item.scoring_mask] == 0)
    assert item.targets[0, 3] == -1 and not item.scoring_mask[0, 3]
    assert item.coverage["remainder_epochs"] == 3
    assert item.coverage["scored_epochs"] == weights.sum()
    assert all(start in (0, 240) for _, start in item.coordinates)


def test_masked_loss_normalizes_over_scored_epochs(runtime):
    keras, _ = runtime
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="mean_with_sample_weight")
    targets = np.array([[0, 0]], np.int32)
    logits = np.array([[[2, 0, -1], [-9, 9, 2]]], np.float32)
    weights = np.array([[1, 0]], np.float32)
    actual = float(keras.ops.convert_to_numpy(loss(targets, logits, sample_weight=weights)))
    assert actual == pytest.approx(np.log(1 + np.exp(-2) + np.exp(-3)), abs=1e-6)


def test_training_retains_partial_batches_and_heldout_values_do_not_change_weights(inputs, runtime):
    keras, _ = runtime
    parts = prepared(inputs)
    cfg = recipe.Config(epochs=2, batch_size=4, seed=17)
    model, history = recipe.train(parts["train"], parts["validation"], cfg)
    assert int(keras.ops.convert_to_numpy(model.optimizer.iterations)) == 4
    assert len(history["loss"]) == len(history["val_loss"]) == 2
    expected = [w.copy() for w in model.get_weights()]
    item = parts["validation"]
    modified = PreparedPartition(
        item.features * -3,
        np.where(item.targets >= 0, (item.targets + 1) % 3, -1),
        item.scoring_mask,
        item.coordinates,
        item.coverage,
    )
    parts["test"].features[:] = 500  # Test data is not passed into train at all.
    other, _ = recipe.train(parts["train"], modified, cfg)
    for before, after in zip(expected, other.get_weights()):
        np.testing.assert_array_equal(before, after)


def test_golden_run_archive_and_fresh_consumer(inputs, tmp_path):
    source, split, _ = inputs
    golden = tmp_path / "golden.json"
    cfg = recipe.Config(epochs=1, batch_size=4)
    definition = recipe.declare_golden(source, split, golden, cfg, name="synthetic-first")
    assert not (tmp_path / "run").exists()
    recipe.run(source, split, tmp_path / "run", cfg, golden=golden)
    bundle = tmp_path / "run/bundle"
    validate_bundle(bundle)
    declaration = json.loads((tmp_path / "run/declaration.json").read_text())
    assert all(declaration[key] == value for key, value in definition["expected"].items())
    metadata = json.loads((bundle / "recipe.json").read_text())
    assert metadata["coverage"]["test"]["scored_epochs"] == 477
    assert json.loads((bundle / "metrics.json").read_text())["test"]["count"] == 477
    for path in bundle.glob("*.json"):
        assert "private-person" not in path.read_text()
        assert str(source) not in path.read_text()
    code = """
import keras, numpy as np
from preprocessing import prepare_subject, read_subject, window_subject
model = keras.models.load_model("model.keras", compile=False, safe_mode=True)
with np.load("reference.npz", allow_pickle=False) as arrays:
    actual = keras.ops.convert_to_numpy(model(arrays["inputs"], training=False))
    np.testing.assert_allclose(actual, arrays["logits"], atol=1e-5, rtol=1e-4)
import sys
windows = window_subject(prepare_subject(read_subject(sys.argv[1])))
assert model(windows.features, training=False).shape == (2,240,3)
"""
    subprocess.run(
        [sys.executable, "-B", "-c", code, str(next(source.glob("*.h5")))],
        cwd=bundle,
        check=True,
        capture_output=True,
        text=True,
    )
    validate_bundle(bundle)  # Consumer must not mutate the immutable archive.
    assert (tmp_path / "run/completed.json").is_file()
    with pytest.raises(FileExistsError):
        recipe.run(source, split, tmp_path / "run", cfg)


@pytest.mark.parametrize("kind", ["config", "data", "split", "code", "environment"])
def test_golden_mismatch_rejected_before_output(inputs, tmp_path, monkeypatch, runtime, kind):
    _, h5py = runtime
    source, split, manifest = inputs
    golden = tmp_path / "golden.json"
    cfg = recipe.Config(epochs=1)
    definition = recipe.declare_golden(source, split, golden, cfg, name="fixed")
    if kind == "config":
        cfg = recipe.Config(epochs=2)
    elif kind == "data":
        with h5py.File(next(source.glob("*.h5")), "a") as stream:
            stream.attrs["revision"] = 2
    elif kind == "split":
        manifest["partitions"]["test"], manifest["partitions"]["validation"] = (
            manifest["partitions"]["validation"],
            manifest["partitions"]["test"],
        )
        write_json(split, manifest)
    else:
        definition["expected"]["implementation_sha256" if kind == "code" else "environment"] = {}
        write_json(golden, definition)
    monkeypatch.setattr(recipe, "train", lambda *a, **kw: pytest.fail("Must not train"))
    with pytest.raises(ValueError, match="Golden"):
        recipe.run(source, split, tmp_path / "run", cfg, golden=golden)
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("kind", ["data", "split", "code", "declaration"])
def test_midrun_mutation_prevents_completion(inputs, tmp_path, monkeypatch, kind):
    source, split, _ = inputs
    implementation = tmp_path / "implementation.py"
    implementation.write_text("# fixed\n")
    inventory = recipe.implementation_files(Path(recipe.__file__).parent)
    inventory["external.py"] = implementation
    monkeypatch.setattr(recipe, "implementation_files", lambda _: inventory)
    original = recipe.train

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        path = {
            "data": next(source.glob("*.h5")),
            "split": split,
            "code": implementation,
            "declaration": tmp_path / "run/declaration.json",
        }[kind]
        with path.open("ab") as stream:
            stream.write(b"changed")
        return result

    monkeypatch.setattr(recipe, "train", mutate)
    with pytest.raises(ValueError, match="changed"):
        recipe.run(source, split, tmp_path / "run", recipe.Config(epochs=1))
    assert not (tmp_path / "run/completed.json").exists()
    assert not (tmp_path / "run/bundle").exists()


def test_overlap_and_unscorable_partition_fail_before_training(inputs, tmp_path, monkeypatch, runtime):
    _, h5py = runtime
    source, split, manifest = inputs
    monkeypatch.setattr(recipe, "train", lambda *a, **kw: pytest.fail("Must not train"))
    manifest["partitions"]["test"] = manifest["partitions"]["train"][:1]
    write_json(split, manifest)
    with pytest.raises(ValueError, match="disjoint"):
        recipe.run(source, split, tmp_path / "overlap")
    assert not (tmp_path / "overlap").exists()
    manifest = create_split(
        [f"private-person-{n}" for n in range(5)], dataset="synthetic", validation_count=1, test_count=1
    )
    write_json(split, manifest)
    with h5py.File(source / (manifest["partitions"]["test"][0] + ".h5"), "a") as stream:
        stream["stage_labels"][:] = 6
    with pytest.raises(ValueError, match="supervised"):
        recipe.run(source, split, tmp_path / "unscorable")
    assert not (tmp_path / "unscorable/completed.json").exists()


@pytest.mark.parametrize(
    "options",
    [
        {"seed": True},
        {"seed": 2**32},
        {"epochs": 0},
        {"batch_size": True},
        {"learning_rate": float("nan")},
        {"learning_rate": True},
    ],
)
def test_invalid_config(options):
    with pytest.raises(ValueError):
        recipe.Config(**options)


def test_every_training_window_appears_once_per_epoch(runtime):
    keras, _ = runtime
    seen = []

    class ObservedModel(keras.Model):
        def compile(self, *args, **kwargs):
            return super().compile(*args, **kwargs, run_eagerly=True)

        def train_step(self, data):
            seen.extend(np.asarray(keras.ops.convert_to_numpy(data[0]))[:, 0, 0].tolist())
            return super().train_step(data)

    def builder():
        inputs = keras.Input((240, 14))
        return ObservedModel(inputs, keras.layers.Dense(3)(inputs))

    features = np.zeros((5, 240, 14), np.float32)
    features[:, :, 0] = np.arange(5)[:, None]
    part = PreparedPartition(features, np.zeros((5, 240), np.int32), np.ones((5, 240), bool), (), {})
    recipe.train(part, part, recipe.Config(epochs=2, batch_size=3), model_builder=builder)
    assert sorted(seen[:5]) == list(range(5))
    assert sorted(seen[5:]) == list(range(5))


def test_actual_gradient_uses_scored_epoch_denominator(runtime, monkeypatch):
    keras, _ = runtime
    monkeypatch.setattr(keras.optimizers, "Adam", lambda rate: keras.optimizers.SGD(rate))

    def builder():
        inputs = keras.Input((240, 14))
        return keras.Model(inputs, keras.layers.Dense(3, kernel_initializer="zeros", bias_initializer="zeros")(inputs))

    features = np.zeros((2, 240, 14), np.float32)
    targets = np.full((2, 240), -1, np.int32)
    targets[0, 0], targets[1, :] = 0, 1
    mask = targets >= 0
    part = PreparedPartition(features, targets, mask, (), {})
    model, _ = recipe.train(part, part, recipe.Config(epochs=1, batch_size=2, learning_rate=1.0), model_builder=builder)
    expected_bias = np.array([1 / 241, 240 / 241, 0]) - 1 / 3
    np.testing.assert_allclose(model.get_weights()[-1], expected_bias, rtol=1e-5, atol=1e-6)


def test_reload_change_prevents_archive_completion(inputs, tmp_path, monkeypatch, runtime):
    keras, _ = runtime
    source, split, _ = inputs
    original = keras.models.load_model

    def changed(*args, **kwargs):
        model = original(*args, **kwargs)
        weights = model.get_weights()
        weights[-1] += 1
        model.set_weights(weights)
        return model

    monkeypatch.setattr(keras.models, "load_model", changed)
    with pytest.raises(ValueError, match="reload changed"):
        recipe.run(source, split, tmp_path / "run", recipe.Config(epochs=1))
    assert not (tmp_path / "run/bundle").exists()
    assert not (tmp_path / "run/completed.json").exists()

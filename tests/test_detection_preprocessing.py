"""Boundary tests for the detection recipe; NumPy only unless exercising HDF5."""

import json
import sys
import subprocess

import numpy as np
import pytest

from sleepkit.recipes.detection.preprocessing import Normalizer, contexts, extract, prepare
from sleepkit.recipes.detection.recipe import Config, evaluate


def sensor_data(n=60, offset=0):
    return np.stack([np.arange(n) * 5, np.arange(n) + offset, np.arange(n) * 2]).astype(np.float32)


def test_feature_order_cadence_alignment_and_tail():
    data = sensor_data(31)
    features = extract(data)
    assert features.values.shape == (4, 5)
    np.testing.assert_array_equal(features.times, [55, 85, 115, 145])
    np.testing.assert_allclose(
        features.values[0],
        [np.cos(2 * np.pi * 27.5 / 86400), 5.5, np.std(np.arange(12)), 11, np.std(np.arange(12) * 2)],
        rtol=1e-6,
    )
    labels = (np.arange(31) % 2).astype(np.int32)
    windows = list(contexts(features, 3, labels))
    assert len(windows) == 1
    np.testing.assert_array_equal(windows[0][1], labels[[11, 17, 23]])
    np.testing.assert_array_equal(windows[0][2], [55, 85, 115])


def test_cached_and_uncached_features_ignore_label_changes(tmp_path):
    data = sensor_data()
    fresh, cached = prepare(data), prepare(data, tmp_path)
    for key in ("values", "valid", "ends"):
        np.testing.assert_array_equal(getattr(fresh, key), getattr(cached, key))
        np.testing.assert_array_equal(getattr(fresh, key), getattr(prepare(data, tmp_path), key))
    zeros = list(contexts(cached, 3, np.zeros(60, np.int32)))
    ones = list(contexts(prepare(data, tmp_path), 3, np.ones(60, np.int32)))
    assert len(list(tmp_path.glob("*.npz"))) == 1
    np.testing.assert_array_equal(zeros[0][0], ones[0][0])
    assert zeros[0][1].sum() == 0 and ones[0][1].sum() == 3
    changed = data.copy()
    changed[1] += 2
    prepare(changed, tmp_path)
    assert len(list(tmp_path.glob("*.npz"))) == 2


def test_normalizer_roundtrip_train_only_state():
    training = [extract(sensor_data(offset=offset)) for offset in (0, 5)]
    held_out = extract(sensor_data(offset=1000))
    normalizer = Normalizer.fit(iter(training))
    all_train = np.concatenate([f.values for f in training])
    np.testing.assert_allclose(normalizer.mean, all_train.mean(axis=0, dtype=float))
    np.testing.assert_allclose(normalizer.scale, np.sqrt(all_train.var(axis=0, dtype=float) + 1e-6))
    state = json.loads(json.dumps(normalizer.to_dict()))
    restored = Normalizer.from_dict(state)
    np.testing.assert_array_equal(normalizer.transform(held_out).values, restored.transform(held_out).values)
    assert restored.transform(held_out).values[:, 1].mean() > 10  # no test-subject refitting
    state["spec"]["features"].reverse()
    with pytest.raises(ValueError, match="Unsupported"):
        Normalizer.from_dict(state)


def test_invalid_contexts_do_not_bridge_missing_frames():
    data = sensor_data(100)
    data[1, 20] = np.nan
    features = extract(data)
    windows = list(contexts(features, 3))
    assert windows
    for _, labels, times in windows:
        assert labels is None
        np.testing.assert_array_equal(np.diff(times), [30, 30])
    assert windows[0][2][0] == 235  # affected contexts are dropped, not stitched together
    unknown = np.full(100, -1, np.int32)
    assert list(contexts(features, 3, unknown)) == []
    assert list(contexts(extract(sensor_data(10)), 3)) == []


def test_clock_midnight_and_gap_validation():
    data = sensor_data()
    data[0] = (data[0] + 86350) % 86400
    extract(data)
    data[0, 20:] += 5
    with pytest.raises(ValueError, match="contiguous"):
        extract(data)


@pytest.mark.parametrize(
    "values", [{"context": 0}, {"epochs": -1}, {"batch_size": 1.5}, {"learning_rate": float("nan")}, {"seed": -1}]
)
def test_invalid_config(values):
    with pytest.raises(ValueError):
        Config(**values)


def test_held_out_metrics_use_declared_class_order():
    logits = np.array([[[4, 0], [0, 4], [0, 4], [4, 0]]], np.float32)
    labels = np.array([[0, 1, 0, 0]])
    metrics = evaluate(lambda x, training: x, [(logits, labels)])
    assert metrics["confusion_matrix"] == [[2, 1], [0, 1]]
    assert metrics["accuracy"] == 0.75
    assert metrics["macro_f1"] == pytest.approx((0.8 + 2 / 3) / 2)
    with pytest.raises(ValueError, match="No valid"):
        evaluate(None, [])


def test_recipe_imports_are_light():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from sleepkit.recipes.detection.recipe import Config
from sleepkit.recipes.detection.inference import predict
assert not {"tensorflow", "keras", "helia_edge", "huggingface_hub"} & sys.modules.keys()
""",
        ],
        check=True,
    )


def test_subject_split_and_unlabeled_reader(tmp_path):
    h5py = pytest.importorskip("h5py")
    from sleepkit.recipes.detection.data import load_split, read_subject
    from sleepkit.artifacts.package import write_json

    for subject in ("a", "b", "c"):
        with h5py.File(tmp_path / f"{subject}.h5", "w") as stream:
            stream["data"] = sensor_data()
    split = {"train": ["a"], "validation": ["b"], "test": ["c"]}
    write_json(tmp_path / "split.json", split)
    assert load_split(tmp_path / "split.json", tmp_path) == split
    data, labels = read_subject(tmp_path, "a", labels=False)
    assert data.shape == (3, 60) and labels is None
    with pytest.raises(KeyError):
        read_subject(tmp_path, "a")
    split["test"] = ["a"]
    write_json(tmp_path / "split.json", split)
    with pytest.raises(ValueError, match="disjoint"):
        load_split(tmp_path / "split.json", tmp_path)


def test_cache_invalidates_on_implementation_version(tmp_path, monkeypatch):
    from sleepkit.recipes.detection import preprocessing

    data = sensor_data()
    prepare(data, tmp_path)
    monkeypatch.setitem(preprocessing.SPEC, "implementation_version", 2)
    prepare(data, tmp_path)
    assert len(list(tmp_path.glob("*.npz"))) == 2


def test_reader_rejects_incompatible_channel_metadata(tmp_path):
    h5py = pytest.importorskip("h5py")
    from sleepkit.recipes.detection.data import read_subject

    with h5py.File(tmp_path / "subject.h5", "w") as stream:
        stream["data"] = sensor_data()
        stream.attrs["channel_names"] = ["TS", "ZANGLE", "ENMO"]
    with pytest.raises(ValueError, match="channel_names"):
        read_subject(tmp_path, "subject", labels=False)

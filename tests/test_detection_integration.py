"""Train/export/reload the same input path; no real data or network required."""

import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pytest


DETECTION_DEPENDENCIES = ("tensorflow", "keras", "h5py", "ai_edge_litert")
MISSING_DEPENDENCIES = [name for name in DETECTION_DEPENDENCIES if importlib.util.find_spec(name) is None]


@pytest.mark.skipif(
    bool(MISSING_DEPENDENCIES) and os.environ.get("SLEEPKIT_REQUIRE_DETECTION") != "1",
    reason="Detection integration requires TensorFlow/Keras, HDF5, and LiteRT",
)
@pytest.mark.parametrize("timed", [False, True])
def test_detection_train_export_and_unlabeled_inference(tmp_path, timed):
    assert not MISSING_DEPENDENCIES, f"Required detection dependencies missing: {MISSING_DEPENDENCIES}"
    # Keep TensorFlow initialization/threads isolated from lightweight unit tests.
    code = """
import json
from pathlib import Path
import h5py
import numpy as np
import keras
from sleepkit.recipes.detection.smoke import make_fixture
from sleepkit.recipes.detection.recipe import run, Config
from sleepkit.recipes.detection.data import read_recording, subject_features
from sleepkit.recipes.detection.preprocessing import Normalizer, prepare, contexts
from sleepkit.recipes.detection.inference import predict
from sleepkit.artifacts.runtime import replay_bundle
from sleepkit.artifacts.package import validate_bundle
import sys
root = Path(sys.argv[1])
source = root / "source"
split = make_fixture(source)
if sys.argv[2] == "timed":
    for path in source.glob("*.h5"):
        with h5py.File(path, "a") as f:
            samples = f["data"].shape[1]
            values = f["data"][:]
            values[0, 20:] = (values[0, 20:] + 3600) % 86400
            f["data"][:] = values
            f["sample_time"] = np.arange(samples, dtype=np.int64) * 5 + 1600000000
            f["sample_time"].attrs["units"] = "unix_seconds"
run(source, split, root / "run", Config(context=8, epochs=2, batch_size=2), cache=root / "cache", data_kind="synthetic")
bundle = root / "run" / "bundle"
validate_bundle(bundle, profile="runnable")
replay_bundle(bundle)
assert len(json.loads((root / "run" / "history.json").read_text())["loss"]) == 2
normalizer = Normalizer.from_dict(json.loads((bundle / "preprocessing.json").read_text()))
expected_state = Normalizer.fit(subject_features(source, ["synthetic-0", "synthetic-1"]))
np.testing.assert_array_equal(normalizer.mean, expected_state.mean)
# Inference must still work after every annotation has been removed.
with h5py.File(source / "synthetic-3.h5", "a") as f:
    del f["sleep_stages"]
recording = read_recording(source, "synthetic-3", labels=False)
data, sample_time = recording.data, recording.sample_time
assert recording.labels is None
result = predict(bundle, data, sample_time=sample_time)
model = keras.models.load_model(bundle / "model.keras", compile=False)
x = np.stack([x for x, _, _ in contexts(normalizer.transform(prepare(data, sample_time=sample_time)), 8)])
expected = np.asarray(model(x, training=False)).reshape(-1, 2)
np.testing.assert_allclose(result["logits"], expected, atol=1e-5, rtol=1e-4)
np.testing.assert_allclose(result["probabilities"].sum(axis=-1), 1, atol=1e-6)
np.testing.assert_array_equal(result["times"], np.arange(48) * 30 + 55)
np.testing.assert_array_equal(result["available_at"], np.repeat(np.arange(6) * 240 + 270, 8))
if sample_time is None:
    from sleepkit.recipes.detection.export import export_bundle
    from sleepkit.recipes.detection.preprocessing import V2_SPEC
    old_state = normalizer.to_dict()
    old_state["spec"] = V2_SPEC
    legacy_normalizer = Normalizer.from_dict(old_state)
    (root / "legacy").mkdir()
    export_bundle(model, legacy_normalizer, root / "legacy" / "bundle", {}, {})
    legacy_result = predict(root / "legacy" / "bundle", data)
    np.testing.assert_allclose(legacy_result["logits"], result["logits"], atol=1e-5, rtol=1e-4)
    try:
        predict(root / "legacy" / "bundle", data, sample_time=np.arange(data.shape[1], dtype=np.int64) * 5)
    except ValueError as error:
        assert "v2" in str(error)
    else:
        raise AssertionError("v2 bundle accepted a widened clock contract")
np.savez(root / "predictions.npz", **result)
"""
    env = {**os.environ, "TF_NUM_INTEROP_THREADS": "1", "TF_NUM_INTRAOP_THREADS": "1", "TF_CPP_MIN_LOG_LEVEL": "3"}
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), "timed" if timed else "legacy"], env=env, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "input ran out" not in result.stderr
    bundle = tmp_path / "run" / "bundle"
    metadata = json.loads((bundle / "recipe.json").read_text())
    assert metadata["data_kind"] == "synthetic"
    for path in bundle.glob("*.json"):
        assert "synthetic-0" not in path.read_text()  # identifiers stay local
    with np.load(tmp_path / "predictions.npz") as predictions:
        assert predictions["logits"].shape == (48, 2)

"""Lightweight contracts plus subprocess-gated backend integration for signals."""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from sleepkit.recipes.signals import Config, evaluate, make_data, prepare


def test_make_data_and_prepare_are_deterministic():
    first, second = make_data(7), make_data(7)
    for split in ("train", "validation", "test"):
        np.testing.assert_array_equal(first[split]["raw"], second[split]["raw"])
        np.testing.assert_array_equal(first[split]["targets"], second[split]["targets"])
        features = prepare(first[split]["raw"])
        assert features.dtype == np.float32 and features.shape[1:] == (128, 1)
        assert np.isfinite(features).all()


def test_prepare_rejects_wrong_shape_and_nonfinite():
    with pytest.raises(ValueError, match="shape"):
        prepare(np.zeros((2, 127), np.float32))
    values = np.zeros((2, 128), np.float32)
    values[0, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        prepare(values)


def test_config_and_evaluate_use_finite_tail_safe_contract():
    with pytest.raises(ValueError):
        Config(batch_size=0)
    with pytest.raises(ValueError):
        Config(seed=-1)

    class FixedModel:
        def __call__(self, values, training=False):
            return np.tile(np.array([[2.0, 0.0, -1.0]], np.float32), (len(values), 1))

    result = evaluate(FixedModel(), make_data(0), "test")
    assert result["count"] == 30
    assert result["confusion_matrix"] == [[10, 0, 0], [10, 0, 0], [10, 0, 0]]
    assert result["f1_zero_division"] == 0
    assert np.isfinite(result["cross_entropy"])


@pytest.mark.skipif(os.environ.get("SLEEPKIT_REQUIRE_EDGE") != "1", reason="Requires explicit helia-edge integration opt-in")
def test_backend_subprocess_packages_and_reloads_custom_layer(tmp_path):
    code = r'''
import json
from pathlib import Path
import numpy as np
from sleepkit.recipes.signals import Config, run
from sleepkit.artifacts import validate_bundle
root = Path(__import__("sys").argv[1])
output = run(root / "run", Config(batch_size=7, epochs=1, seed=4))
report = validate_bundle(output, profile="archive")
assert report["integrity"] == "passed"
assert any(check["name"] == "model_reload" and check["status"] == "passed" for check in report["checks"])
manifest = report["manifest"]
assert {entry["path"] for entry in manifest["artifacts"]} >= {"model.keras", "metrics.json", "config.json", "preprocessing.json", "history.json", "reference.json", "reference.npz"}
metrics = json.loads((output / "metrics.json").read_text())
assert metrics["count"] == 30
config = json.loads((output / "config.json").read_text())
assert config["preprocessing"]["output"].endswith("[examples, 128, 1]")
assert json.loads((output / "preprocessing.json").read_text()) == config["preprocessing"]
assert config["class_order"] == ["one_cycle", "two_cycles", "three_cycles"]
assert config["split_counts"] == {"train": 75, "validation": 30, "test": 30}
assert config["environment"]["backend"] == __import__("os").environ["KERAS_BACKEND"]
assert config["environment"]["helia_edge_direct_url"]["vcs_info"]["commit_id"] == "0661c5f58cbad27681d5e2f6a8cca7477477c853"
with np.load(output / "reference.npz", allow_pickle=False) as arrays:
    assert arrays["inputs"].shape == (6, 128, 1)
    assert arrays["logits"].shape == (6, 3)
'''
    backend = os.environ.get("KERAS_BACKEND", "tensorflow")
    environment = {**os.environ, "KERAS_BACKEND": backend, "TF_NUM_INTRAOP_THREADS": "1", "TF_NUM_INTEROP_THREADS": "1", "TF_CPP_MIN_LOG_LEVEL": "3"}
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    reload_code = r'''
import sys
from pathlib import Path
import numpy as np
from helia_edge.models import load_model
root = Path(sys.argv[1])
with np.load(root / "run/reference.npz", allow_pickle=False) as arrays:
    expected = arrays["logits"]
    actual = load_model(root / "run/model.keras")(arrays["inputs"], training=False)
if hasattr(actual, "detach"):
    actual = actual.detach().cpu().numpy()
else:
    actual = np.asarray(actual)
np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
'''
    fresh = subprocess.run(
        [sys.executable, "-c", reload_code, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert fresh.returncode == 0, fresh.stdout + fresh.stderr

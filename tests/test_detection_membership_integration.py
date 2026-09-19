"""Exercise derived labels through training, scoring artifacts, and unlabeled runtime."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


MISSING = [name for name in ("tensorflow", "keras", "h5py", "ai_edge_litert") if importlib.util.find_spec(name) is None]


@pytest.mark.skipif(
    bool(MISSING) and os.environ.get("SLEEPKIT_REQUIRE_DETECTION") != "1", reason="Requires detection extras"
)
def test_membership_train_score_export_infer(tmp_path):
    assert not MISSING
    code = r"""
import json
from pathlib import Path
import sys
import numpy as np
import keras
sys.path.insert(0, str(Path.cwd() / "tests"))
from test_detection_target_dataset import make_dataset_fixture
from sleepkit.recipes.detection.target_dataset import AnnotatedDataset
from sleepkit.recipes.detection.recipe import run_membership, Config
from sleepkit.recipes.detection.inference import predict
from sleepkit.recipes.detection.preprocessing import prepare, Normalizer, contexts
from sleepkit.recipes.detection.data import subject_features
from sleepkit.artifacts.package import validate_bundle, sha256
root = Path(sys.argv[1])
paths = make_dataset_fixture(root)
source = AnnotatedDataset(*paths)
output = root / "run"
try:
    run_membership(source, output, Config(context=3, epochs=1))
except ValueError as error:
    assert "Context" in str(error)
else:
    raise AssertionError("Frozen context mismatch accepted")
run_membership(source, output, Config(context=source.context, epochs=1, batch_size=2), cache=root / "cache", data_kind="synthetic")
bundle = output / "bundle"
validate_bundle(bundle, profile="runnable")
metadata = json.loads((bundle / "recipe.json").read_text())
metrics = json.loads((bundle / "metrics.json").read_text())
assert metadata["recipe"] == "sleepkit.detection/v2"
assert metadata["class_names"] == source.target["classes"] == metrics["class_order"]
assert metadata["target"] == source.target
assert "clinical sleep/wake" in (bundle / "README.md").read_text()
assert "labels as-is" not in (bundle / "README.md").read_text()
state = Normalizer.from_dict(json.loads((bundle / "preprocessing.json").read_text()))
expected_state = Normalizer.fit(subject_features(source.root, source.split["train"], reader=source.read))
np.testing.assert_array_equal(state.mean, expected_state.mean)
np.testing.assert_array_equal(state.scale, expected_state.scale)
rows = [json.loads(line) for line in (output / "test-index.jsonl").read_text().splitlines()]
eligible = [row for row in rows if row["context_eligible"]]
assert any(not row["context_eligible"] for row in rows)
with np.load(output / "test-predictions.npz", allow_pickle=False) as predictions:
    targets = predictions["targets"]
    logits = predictions["logits"]
np.testing.assert_array_equal([row["target"] for row in eligible], targets)
np.testing.assert_array_equal([row["eligible_output_index"] for row in eligible], np.arange(len(targets)))
assert metrics["feature_frames_evaluated"] == len(eligible)
assert metadata["scoring"]["sha256"] == sha256(output / "test-index.jsonl")
assert metadata["scoring"]["predictions_sha256"] == sha256(output / "test-predictions.npz")
confusion = np.zeros((2,2), np.int64)
np.add.at(confusion,(targets,np.argmax(logits,axis=1)),1)
assert confusion.tolist() == metrics["confusion_matrix"]
for file in bundle.glob("*.json"):
    for subject in source.source_hashes:
        assert f'"{subject}"' not in file.read_text()
assert not (bundle / "test-index.jsonl").exists()
model = keras.models.load_model(bundle / "model.keras", compile=False)
for subject in source.split["test"]:
    recording = source.read(subject, labels=False)
    assert recording.labels is None
    result = predict(bundle, recording.data, sample_time=recording.sample_time)
    assert result["class_names"] == source.target["classes"]
    assert result["target"] == source.target
    features = state.transform(prepare(recording.data, sample_time=recording.sample_time))
    x = np.stack([x for x, _, _ in contexts(features, source.context)])
    expected = np.asarray(model(x, training=False)).reshape(-1,2)
    np.testing.assert_allclose(result["logits"], expected, atol=1e-5, rtol=1e-4)
source.verify_unchanged()
"""
    env = {**os.environ, "TF_NUM_INTEROP_THREADS": "1", "TF_NUM_INTRAOP_THREADS": "1", "TF_CPP_MIN_LOG_LEVEL": "3"}
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr

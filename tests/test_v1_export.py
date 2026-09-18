import importlib.util

import numpy as np
import pytest

from sleepkit.evaluation import classification_metrics


def test_metrics_handle_absent_classes():
    result = classification_metrics([0, 0, 1], [0, 1, 1], 3)
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["macro_f1"] == pytest.approx(4 / 9)
    assert result["support"] == [2, 1, 0]
    with pytest.raises(ValueError):
        classification_metrics([2], [0], 2)


@pytest.mark.skipif(
    not all(importlib.util.find_spec(name) for name in ("tensorflow", "ai_edge_litert")),
    reason="requires train and runtime extras",
)
def test_full_synthetic_recipe_and_bundle(tmp_path):
    from sleepkit.recipes.staging import run
    from sleepkit.runtime import Predictor, validate_bundle
    from sleepkit.export import publish_bundle
    from sleepkit.data import SubjectSplit
    from sleepkit.evaluation import evaluate_records
    from sleepkit.recipes.staging import synthetic_records
    import json
    import shutil

    result = run(tmp_path / "run", epochs=1)
    bundle = tmp_path / "run" / "deploy"
    predictor = Predictor(bundle)
    assert predictor.check_reference()
    split = SubjectSplit.from_dict(json.loads((tmp_path / "run" / "split.json").read_text()))
    evaluation = evaluate_records(predictor, split.select(synthetic_records(), "test"))
    assert len(evaluation["records"]) == 2
    assert sum(evaluation["pooled"]["support"]) == 64
    state = json.loads((bundle / "preprocessing.json").read_text())
    assert state["normalizer"]["training_subjects"] == []
    assert len(state["normalizer"]["training_subjects_sha256"]) == 64
    assert result["inference_windows"] == 8
    stage = publish_bundle(bundle, "unused/dry-run")
    try:
        assert validate_bundle(stage)["quantization"] == "int8"
    finally:
        shutil.rmtree(stage)
    with pytest.raises(ValueError, match="Expected finite"):
        predictor.predict_features(np.zeros((1, 2, 4)))
    with pytest.raises(FileExistsError):
        run(tmp_path / "run", epochs=1)
    (bundle / "model.tflite").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Checksum"):
        validate_bundle(bundle)


@pytest.mark.skipif(
    not all(importlib.util.find_spec(name) for name in ("tensorflow", "ai_edge_litert")),
    reason="requires train and runtime extras",
)
def test_export_rejects_degraded_model_and_leaves_no_bundle(tmp_path, monkeypatch):
    from sleepkit.recipes.staging import run
    from sleepkit.runtime import Predictor

    original = Predictor.predict_features
    monkeypatch.setattr(Predictor, "predict_features", lambda self, x: original(self, x)[..., ::-1])
    with pytest.raises(ValueError, match="accuracy drop"):
        run(tmp_path / "bad-run", epochs=2)
    assert not (tmp_path / "bad-run" / "deploy").exists()
    assert not list((tmp_path / "bad-run").glob(".export-*"))

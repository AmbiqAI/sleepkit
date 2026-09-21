"""The comparison harness must exercise uncached implementations and bind its result."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments import compare_detection_preparation as comparison
from sleepkit.artifacts.package import sha256
from sleepkit.recipes.detection import preprocessing


def _test_module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_comparison_covers_train_only_and_binds_report(tmp_path, monkeypatch):
    fixture = _test_module("test_detection_target_dataset")
    oracle = _test_module("test_detection_vectorized").scalar_extract
    source = comparison.AnnotatedDataset(*fixture.make_dataset_fixture(tmp_path))
    train = source.split["train"]
    reads = []
    original = source.read

    def read(subject, *, labels=True):
        assert subject in train and labels is False
        reads.append(subject)
        return original(subject, labels=labels)

    source.read = read
    monkeypatch.setattr(comparison, "legacy_module", lambda: SimpleNamespace(SPEC=preprocessing.SPEC, extract=oracle))
    output = tmp_path / "comparison"
    report = comparison.compare(source, output, repeats=2)
    assert report["training_subjects"] == len(train)
    assert reads == train + train[:3] + train[:3]
    assert report["feature_frames_compared"] == 14 * len(train)
    assert report["normalization_frames"] == report["feature_frames_compared"]
    assert report["declaration_sha256"] == sha256(output / "declaration.json")
    assert len(report["repeated_subset_extract_seconds"]["scalar"]) == 2
    assert all(name not in json.dumps(report) for name in train)
    with pytest.raises(FileExistsError):
        comparison.compare(source, output, repeats=2)


def test_mismatch_does_not_create_success_report(tmp_path, monkeypatch):
    fixture = _test_module("test_detection_target_dataset")
    oracle = _test_module("test_detection_vectorized").scalar_extract
    source = comparison.AnnotatedDataset(*fixture.make_dataset_fixture(tmp_path))

    def wrong(*args, **kwargs):
        features = oracle(*args, **kwargs)
        features.values[0, 0] += 1
        return features

    monkeypatch.setattr(comparison, "legacy_module", lambda: SimpleNamespace(SPEC=preprocessing.SPEC, extract=wrong))
    output = tmp_path / "comparison"
    with pytest.raises(ValueError, match="mismatch"):
        comparison.compare(source, output, repeats=2)
    assert (output / "declaration.json").is_file()
    assert not (output / "report.json").exists()


def test_legacy_source_must_match_pinned_hash(monkeypatch):
    monkeypatch.setattr(comparison.subprocess, "check_output", lambda *a, **k: b"incorrect implementation")
    with pytest.raises(ValueError, match="source hash mismatch"):
        comparison.legacy_module()

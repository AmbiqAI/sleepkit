"""Regression cases from summary-only and inline stack review findings."""

import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from experiments import run_detection_golden as golden
from sleepkit.artifacts.package import sha256, validate_bundle, write_json
from sleepkit.recipes.detection import evaluation, inference
from tests.test_detection_evaluation import make_run
from tests.test_detection_runtime import fake_runtime, make_bundle


def test_evaluator_requires_the_named_checkpoint(tmp_path):
    run = make_run(tmp_path)
    bundle = run / "bundle"
    (bundle / "model.keras").rename(bundle / "different.keras")
    manifest = json.loads((bundle / "manifest.json").read_text())
    for entry in manifest["artifacts"]:
        if entry["path"] == "model.keras":
            entry["path"] = "different.keras"
    write_json(bundle / "manifest.json", manifest)
    checks = json.loads((bundle / "checksums.json").read_text())
    checks["different.keras"] = checks.pop("model.keras")
    checks["manifest.json"] = sha256(bundle / "manifest.json")
    write_json(bundle / "checksums.json", checks)
    validate_bundle(bundle)  # A valid generic archive, but not the declared detector checkpoint.
    with pytest.raises(ValueError, match="model.keras"):
        evaluation.summarize(run, tmp_path / "report.json")
    assert not (tmp_path / "report.json").exists()


@pytest.mark.parametrize("name", ["recipe.json", "preprocessing.json"])
def test_inference_rejects_metadata_replacement_after_validation(tmp_path, monkeypatch, name):
    fake_runtime(monkeypatch, integer=False)
    bundle = make_bundle(tmp_path, integer=False)
    original = inference.validate_bundle

    def validate_then_replace(path, **kwargs):
        report = original(path, **kwargs)
        value = json.loads((bundle / name).read_text())
        value.update(mean=[100] * 5) if name == "preprocessing.json" else value.update(context=4)
        write_json(bundle / name, value)
        return report

    monkeypatch.setattr(inference, "validate_bundle", validate_then_replace)
    data = np.stack([np.arange(24) * 5, np.ones(24), np.ones(24)]).astype(np.float32)
    with pytest.raises(ValueError, match="metadata read"):
        inference.predict(bundle, data)


def test_bound_metadata_checks_consumed_bytes_even_if_file_is_restored(tmp_path, monkeypatch):
    path = tmp_path / "metadata.json"
    path.write_text('{"mean": 0}')
    hashes = {path.name: sha256(path)}
    original = Path.read_bytes

    def substituted_bytes(self):
        return b'{"mean": 100}' if self == path else original(self)

    monkeypatch.setattr(Path, "read_bytes", substituted_bytes)
    with pytest.raises(ValueError, match="metadata read"):
        evaluation.read_bound_json(tmp_path, path.name, hashes)
    assert sha256(path) == hashes[path.name]


@pytest.mark.parametrize("key,value", [("window_samples", 999), ("features", ["different"]), ("target_alignment", "first")])
def test_golden_public_preprocessing_must_match_execution(key, value):
    definition = golden._read_definition()
    definition["preprocessing"][key] = value
    with pytest.raises(ValueError, match="public"):
        golden.validate_definition(definition)


@pytest.mark.parametrize("kind", ["missing", "source", "split", "spec", "dataset", "candidate", "baseline", "status", "count"])
def test_golden_equivalence_requires_matching_evidence_chain(tmp_path, kind):
    definition = golden._read_definition()
    original = golden.REPOSITORY / definition["implementation"]["equivalence_evidence"]["path"]
    report = json.loads(original.read_text())
    declaration = json.loads(original.with_name("declaration.json").read_text())
    if kind in {"source", "split", "spec"}:
        declaration[f"{kind}_sha256"] = "0" * 64
    elif kind == "dataset":
        declaration["dataset"]["grouping"] = "different"
    elif kind == "candidate":
        declaration["candidate_code_sha256"]["preprocessing.py"] = "0" * 64
    elif kind == "baseline":
        declaration["baseline_preprocessing_sha256"] = "0" * 64
    elif kind == "status":
        report["status"] = "failed"
    elif kind == "count":
        report["training_subjects"] -= 1
    write_json(tmp_path / "declaration.json", declaration)
    report["declaration_sha256"] = sha256(tmp_path / "declaration.json")
    write_json(tmp_path / "report.json", report)
    definition["implementation"]["equivalence_evidence"] = {
        "path": str(tmp_path / "report.json"), "sha256": sha256(tmp_path / "report.json"),
    }
    if kind == "missing":
        (tmp_path / "report.json").unlink()
    with pytest.raises(ValueError, match="equivalence"):
        golden.validate_definition(definition)


def test_profile_import_and_rss_work_without_unix_resource(monkeypatch):
    from sleepkit.recipes.detection import profile

    monkeypatch.setitem(sys.modules, "resource", None)
    importlib.reload(profile)
    assert profile._rss() is None


@pytest.mark.parametrize("metadata", [None, [], [1], "invalid", 3])
def test_generic_bundle_rejects_nonobject_metadata(tmp_path, metadata):
    bundle = make_bundle(tmp_path, integer=False)
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["metadata"] = metadata
    write_json(bundle / "manifest.json", manifest)
    checks = json.loads((bundle / "checksums.json").read_text())
    checks["manifest.json"] = sha256(bundle / "manifest.json")
    write_json(bundle / "checksums.json", checks)
    with pytest.raises(ValueError, match="metadata"):
        validate_bundle(bundle)

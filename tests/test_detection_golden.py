"""Pure validation tests for the public golden experiment definition."""

from copy import deepcopy
import json

import pytest

from experiments.run_detection_golden import _read_definition, validate_definition, validate_run, validate_source
from sleepkit.recipes.detection.preprocessing import fingerprint


def _source_and_definition(tmp_path):
    definition = _read_definition()
    definition = deepcopy(definition)
    split = {"train": ["train-a"], "validation": ["validation-a"], "test": ["test-a"]}
    source_hashes = {subject: f"{index + 1:064x}" for index, subject in enumerate(sum(split.values(), []))}
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    dataset = definition["dataset"]
    dataset.update(
        source_subjects=3,
        partition_counts={name: {"subjects": 1, "eligible_contexts": 1} for name in split},
        split_fingerprint=fingerprint(split),
        source_inventory_fingerprint=fingerprint(source_hashes),
    )
    dataset["provenance_sha256"] = {
        "events": "events", "alignment": "alignment", "coverage": "coverage", "protocol": "protocol",
        "split_file": "split", "groups": "groups", "sources": "sources", "parquet": "parquet",
    }
    provenance = {
        "kind": dataset["kind"],
        "source_subjects": 3,
        "grouping": dataset["grouping"],
        "seed": dataset["seed"],
        "target": dataset["target"],
        "policy": dataset["candidate_policy"],
        "context_policy": dataset["context_policy"],
        "sample_clock": dataset["sample_clock"],
        **{f"{key}_sha256": value for key, value in dataset["provenance_sha256"].items()},
    }
    source = type(
        "Source",
        (),
        {
            "context": 240,
            "provenance": provenance,
            "split": split,
            "split_path": split_path,
            "source_hashes": source_hashes,
        },
    )()
    # The test fixture uses synthetic hash labels; make the expected split file
    # hash match the actual temporary JSON while retaining the same contract.
    from sleepkit.artifacts.package import sha256

    dataset["provenance_sha256"]["split_file"] = sha256(split_path)
    source.provenance["split_sha256"] = dataset["provenance_sha256"]["split_file"]
    return source, definition


def test_matching_definition_and_source_pass(tmp_path):
    source, definition = _source_and_definition(tmp_path)
    validate_definition(definition)
    validate_source(source, definition)


def test_config_mismatch_is_rejected_before_training():
    definition = _read_definition()
    definition["recipe"]["config"]["epochs"] = 6
    with pytest.raises(ValueError, match="training config"):
        validate_definition(definition)


def test_dataset_provenance_mismatch_is_rejected(tmp_path):
    source, definition = _source_and_definition(tmp_path)
    source.provenance["alignment_sha256"] = "changed"
    with pytest.raises(ValueError, match="alignment provenance hash"):
        validate_source(source, definition)


def test_public_definition_has_no_local_paths_or_subject_identifiers():
    text = json.dumps(_read_definition(), sort_keys=True)
    assert "/home/" not in text and "/tmp/" not in text
    assert "subject-a" not in text and "synthetic-0" not in text
    assert ".h5" not in text


def test_reference_metrics_cannot_become_a_gate():
    definition = _read_definition()
    definition["reference"]["prospective_gate"] = True
    with pytest.raises(ValueError, match="prospective gate"):
        validate_definition(definition)


def test_implementation_drift_is_rejected_before_training():
    definition = _read_definition()
    definition["implementation"]["module_sha256"]["sleepkit/recipes/detection/model.py"] = "changed"
    with pytest.raises(ValueError, match="model.py hash"):
        validate_definition(definition)


def test_run_denominator_drift_is_rejected(tmp_path):
    definition = _read_definition()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    recipe = {
        "subjects_per_split": {"train": 193, "validation": 41, "test": 43},
        "contexts_per_split": {"train": 34473, "validation": 8173, "test": 8582},
        "config": definition["recipe"]["config"],
    }
    (bundle / "recipe.json").write_text(json.dumps(recipe), encoding="utf-8")
    with pytest.raises(ValueError, match="eligible-context counts"):
        validate_run(tmp_path, definition)


@pytest.mark.parametrize("tamper", [None, "definition", "code", "environment", "counts"])
def test_runner_attaches_snapshot_or_leaves_no_completed_output(tmp_path, monkeypatch, tamper):
    from experiments import run_detection_golden as runner
    from sleepkit.artifacts.package import sha256

    source, definition = _source_and_definition(tmp_path)
    source.verify_unchanged = lambda: None
    definition_path = tmp_path / "golden.json"
    definition_path.write_text(json.dumps(definition))
    original_hash = sha256(definition_path)
    output = tmp_path / "completed"
    monkeypatch.setattr(runner, "DEFINITION", definition_path)
    monkeypatch.setattr(runner, "AnnotatedDataset", lambda *args: source)
    monkeypatch.setattr(runner, "_runtime_environment", lambda: {"test": "original"})

    def train(source, staging, cfg, **kwargs):
        bundle = staging / "bundle"
        bundle.mkdir(parents=True)
        counts = definition["dataset"]["partition_counts"]
        recipe = {
            "config": definition["recipe"]["config"],
            "subjects_per_split": {name: value["subjects"] for name, value in counts.items()},
            "contexts_per_split": {name: value["eligible_contexts"] for name, value in counts.items()},
        }
        if tamper == "definition":
            definition_path.write_text(definition_path.read_text() + "\n")
        elif tamper == "code":
            monkeypatch.setattr(runner, "_detection_hashes", lambda: {"changed": "code"})
        elif tamper == "environment":
            monkeypatch.setattr(runner, "_runtime_environment", lambda: {"test": "changed"})
        elif tamper == "counts":
            recipe["contexts_per_split"]["train"] += 1
        (bundle / "recipe.json").write_text(json.dumps(recipe))

    monkeypatch.setattr(runner, "run_membership", train)
    argv = []
    for flag in ("data", "events", "alignment", "coverage", "frozen-split"):
        argv.extend([f"--{flag}", str(tmp_path)])
    argv.extend(["--output", str(output)])
    if tamper:
        with pytest.raises(ValueError):
            runner.main(argv)
        assert not output.exists()
        assert not list(tmp_path.glob(".golden-*"))
    else:
        runner.main(argv)
        record = json.loads((output / "golden-runner.json").read_text())
        assert record["definition_sha256"] == sha256(output / "golden-definition.json") == original_hash
        assert record["runtime"] == {"test": "original"}
        assert record["detection_code_sha256"] == runner._detection_hashes()
        with pytest.raises(FileExistsError):
            runner.main(argv)

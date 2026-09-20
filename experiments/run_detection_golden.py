#!/usr/bin/env python3
"""Validate the public golden definition and run it against supplied local evidence."""

import argparse
import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
import os
import platform
from pathlib import Path
import sys
import tempfile


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path[:] = [str(REPOSITORY), *(entry for entry in sys.path if entry != str(REPOSITORY))]

from sleepkit.artifacts.package import sha256  # noqa: E402
from sleepkit.recipes.detection.preprocessing import SPEC, fingerprint  # noqa: E402
from sleepkit.recipes.detection.recipe import Config, run_membership  # noqa: E402
from sleepkit.recipes.detection.target_dataset import AnnotatedDataset  # noqa: E402


DEFINITION = Path(__file__).with_name("detection-golden.json")


def _read_definition():
    definition = json.loads(DEFINITION.read_text(encoding="utf-8"))
    if definition.get("schema") != "sleepkit.detection_golden_experiment/v1":
        raise ValueError("Unsupported golden experiment definition")
    return definition


def _require(actual, expected, label):
    if actual != expected:
        raise ValueError(f"Pinned {label} differs from the supplied local evidence")


def _runtime_environment():
    versions = {}
    for package in ("tensorflow", "keras", "numpy", "h5py", "ai-edge-litert"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    thread_names = ("TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TF_DETERMINISTIC_OPS", "CUDA_VISIBLE_DEVICES")
    return {
        "python": platform.python_version(),
        "versions": versions,
        "thread_environment": {name: os.environ.get(name) for name in thread_names},
    }


def _detection_hashes():
    directory = REPOSITORY / "sleepkit/recipes/detection"
    return {str(path.relative_to(REPOSITORY)): sha256(path) for path in sorted(directory.glob("*.py"))}


def validate_source(source, definition):
    dataset = definition["dataset"]
    _require(source.context, dataset["context_policy"]["features_per_context"], "context")
    _require(source.provenance["kind"], dataset["kind"], "dataset kind")
    for key in ("source_subjects", "grouping", "seed"):
        _require(source.provenance[key], dataset[key], f"dataset {key}")
    _require(source.provenance["target"], dataset["target"], "target contract")
    _require(source.provenance["policy"], dataset["candidate_policy"], "candidate policy")
    _require(source.provenance["context_policy"], dataset["context_policy"], "context policy")
    _require(source.provenance["sample_clock"], dataset["sample_clock"], "sample clock")
    pinned_hashes = dataset["provenance_sha256"]
    provenance_keys = {
        "events": "events_sha256",
        "alignment": "alignment_sha256",
        "coverage": "coverage_sha256",
        "protocol": "protocol_sha256",
        "split_file": "split_sha256",
        "groups": "groups_sha256",
        "sources": "sources_sha256",
        "parquet": "parquet_sha256",
    }
    for public_name, source_name in provenance_keys.items():
        _require(source.provenance[source_name], pinned_hashes[public_name], f"{public_name} provenance hash")
    _require(sha256(source.split_path), pinned_hashes["split_file"], "split file hash")
    _require(fingerprint(source.split), dataset["split_fingerprint"], "split fingerprint")
    _require(fingerprint(source.source_hashes), dataset["source_inventory_fingerprint"], "source inventory fingerprint")
    expected_partitions = dataset["partition_counts"]
    for name, expected in expected_partitions.items():
        subjects = source.split[name]
        _require(len(subjects), expected["subjects"], f"{name} subject count")
    if set(source.split) != {"train", "validation", "test"}:
        raise ValueError("Pinned source must have train, validation, and test partitions")


def validate_definition(definition):
    config = definition["recipe"]["config"]
    _require(config, {"context": 240, "epochs": 5, "batch_size": 32, "learning_rate": 0.001, "seed": 0}, "training config")
    if definition["reference"]["prospective_gate"]:
        raise ValueError("Historical reference metrics must not become a prospective gate")
    implementation = definition["implementation"]
    _require(fingerprint(SPEC), implementation["preprocessing_spec_fingerprint"], "preprocessing SPEC")
    for relative, expected in implementation["module_sha256"].items():
        _require(sha256(REPOSITORY / relative), expected, f"implementation {relative} hash")


def validate_run(staging, definition):
    recipe = json.loads((Path(staging) / "bundle/recipe.json").read_text(encoding="utf-8"))
    expected = definition["dataset"]["partition_counts"]
    _require(
        recipe.get("subjects_per_split"),
        {name: values["subjects"] for name, values in expected.items()},
        "run subject counts",
    )
    _require(
        recipe.get("contexts_per_split"),
        {name: values["eligible_contexts"] for name, values in expected.items()},
        "run eligible-context counts",
    )
    _require(recipe.get("config"), definition["recipe"]["config"], "run config")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Local original CMIDSS HDF5 directory")
    parser.add_argument("--events", required=True, type=Path, help="Local train_events.csv")
    parser.add_argument("--alignment", required=True, type=Path, help="Local verified source-event alignment JSON")
    parser.add_argument("--coverage", required=True, type=Path, help="Local candidate coverage JSON")
    parser.add_argument("--frozen-split", required=True, type=Path, help="Directory containing frozen split evidence")
    parser.add_argument("--output", required=True, type=Path, help="New local run directory")
    parser.add_argument("--cache", type=Path, help="Optional local stateless feature cache")
    args = parser.parse_args(argv)

    definition_bytes = DEFINITION.read_bytes()
    definition = json.loads(definition_bytes.decode("utf-8"))
    if definition.get("schema") != "sleepkit.detection_golden_experiment/v1":
        raise ValueError("Unsupported golden experiment definition")
    validate_definition(definition)
    source = AnnotatedDataset(args.data, args.events, args.alignment, args.coverage, args.frozen_split)
    validate_source(source, definition)
    config = Config(**definition["recipe"]["config"])
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"Output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    definition_sha = hashlib.sha256(definition_bytes).hexdigest()
    runner_sha = sha256(Path(__file__))
    runtime = _runtime_environment()
    detection_hashes = _detection_hashes()
    with tempfile.TemporaryDirectory(dir=args.output.parent, prefix=".golden-") as directory:
        staging = Path(directory) / "run"
        run_membership(source, staging, config, cache=args.cache, data_kind=definition["recipe"]["data_kind"])
        source.verify_unchanged()
        validate_run(staging, definition)
        if DEFINITION.read_bytes() != definition_bytes or sha256(Path(__file__)) != runner_sha:
            raise ValueError("Golden definition or runner changed during the run")
        if _detection_hashes() != detection_hashes:
            raise ValueError("Detection implementation changed during the run")
        if _runtime_environment() != runtime:
            raise ValueError("Runtime environment changed during the run")
        validate_definition(definition)
        (staging / "golden-definition.json").write_bytes(definition_bytes)
        runner_record = {
            "schema": "sleepkit.detection_golden_runner/v1",
            "definition_sha256": definition_sha,
            "runner_sha256": runner_sha,
            "detection_code_sha256": detection_hashes,
            "implementation": definition["implementation"],
            "historical_git_commit": definition["historical_run"]["git_commit"],
            "runtime": runtime,
        }
        (staging / "golden-runner.json").write_text(
            json.dumps(runner_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging.rename(args.output)
    print(json.dumps({"run": str(args.output), "definition": str(DEFINITION), "reference": definition["reference"]}, sort_keys=True))


if __name__ == "__main__":
    main()

"""Release promotion preserves experiment evidence without a training runtime."""

import json
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from sleepkit.artifacts import Artifact, Check, TensorSpec, stage_bundle, stage_release, validate_bundle
from sleepkit.artifacts.hub import download_bundle, publish_bundle
from sleepkit.artifacts.package import sha256, write_json


@pytest.fixture
def experiment(tmp_path):
    model = tmp_path / "model.bin"
    model.write_bytes(b"opaque test weights")
    reference = tmp_path / "reference.bin"
    reference.write_bytes(b"opaque test inputs and outputs")
    preprocessing = tmp_path / "preprocessing.json"
    preprocessing.write_text('{"offset": 2.5}')
    bundle = stage_bundle(
        tmp_path / "experiment",
        title="Generic fixture",
        artifacts=[
            Artifact(
                model,
                "model.bin",
                "model",
                "opaque",
                "test fixture",
                (TensorSpec("input", (1, 3), "float32", "features"),),
                (TensorSpec("output", (1, 1), "float32", "score"),),
            ),
            Artifact(reference, "reference.bin", "reference", "opaque", "test fixture"),
            Artifact(preprocessing, "preprocessing.json", "preprocessing_state", "json", "test fixture"),
        ],
        card="---\ntags: [test]\n---\n# Experiment\nUnspecified license; fixture only.\n",
        checks=[
            Check(
                "runtime_conformance:model.bin",
                "passed",
                "Persisted fixture evidence; not a real runtime",
                {"model.bin": sha256(model), "reference.bin": sha256(reference)},
            ),
            Check("target_hardware", "not_run", "No device test"),
        ],
        metadata={"recipe": "generic-fixture"},
    )
    terms = tmp_path / "terms.txt"
    terms.write_text("Fixture terms only. No actual model license.\n")
    decision = tmp_path / "decision.md"
    decision.write_text("# Decision\nTest fixture only; no real training data or publication.\n")
    options = dict(
        license_file=terms,
        license_id="other",
        license_name="Test fixture terms",
        card_body="# Release\nA test fixture.",
        decision_file=decision,
        profile="runnable",
    )
    return bundle, tmp_path / "release", options


def test_release_preserves_bytes_signatures_metadata_evidence_and_source(experiment, monkeypatch):
    source, destination, options = experiment
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    original = validate_bundle(source, profile="runnable")
    stage_release(source, destination, **options)
    released = validate_bundle(destination, profile="runnable")
    assert released["checks"] == original["checks"]
    assert (destination / "validation.json").read_bytes() == before["validation.json"]
    old_artifacts = original["manifest"]["artifacts"]
    assert released["manifest"]["artifacts"][: len(old_artifacts)] == old_artifacts
    for artifact in old_artifacts:
        assert (destination / artifact["path"]).read_bytes() == before[artifact["path"]]
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    assert (destination / "experiment-card.md").read_bytes() == before["README.md"]
    assert (destination / "release-decision.md").read_bytes() == options["decision_file"].read_bytes()
    assert (destination / "LICENSE").read_bytes() == options["license_file"].read_bytes()
    assert released["manifest"]["metadata"] == {
        "recipe": "generic-fixture",
        "release_source": {
            f"{name}_sha256": sha256(source / f"{name}.json") for name in ("checksums", "manifest", "validation")
        },
    }
    text = (destination / "README.md").read_text()
    assert text.startswith('---\nlicense: "other"\nlicense_name: "Test fixture terms"\nlicense_link: "LICENSE"\n---\n')
    assert str(source) not in text
    assert publish_bundle(destination, "owner/fixture", profile="runnable")["uploaded"] is False


@pytest.mark.parametrize("license_id", ["bsd-3-clause", "cc-by-nc-sa-4.0", "openrail++"])
def test_standard_license_metadata(experiment, license_id):
    source, destination, options = experiment
    options.update(license_id=license_id, license_name=None)
    stage_release(source, destination, **options)
    assert (destination / "README.md").read_text().startswith(f'---\nlicense: "{license_id}"\n---\n')
    assert validate_bundle(destination)["manifest"]["license"] == license_id


@pytest.mark.parametrize(
    "change",
    [
        "empty_terms",
        "empty_decision",
        "custom_name",
        "standard_name",
        "front_matter",
        "empty_card",
        "license_id",
        "symlink",
    ],
)
def test_invalid_release_inputs_fail_without_output(experiment, change):
    source, destination, options = experiment
    if change == "empty_terms":
        options["license_file"].write_text("  ")
    elif change == "empty_decision":
        options["decision_file"].write_text("\n")
    elif change == "custom_name":
        options["license_name"] = None
    elif change == "standard_name":
        options["license_id"] = "bsd-3-clause"
    elif change == "front_matter":
        options["card_body"] = "---\nlicense: mit\n---\nBody"
    elif change == "empty_card":
        options["card_body"] = " "
    elif change == "license_id":
        options["license_id"] = "bsd-3-clause\nlicense: other"
    else:
        link = source.parent / "linked-decision.md"
        link.symlink_to(options["decision_file"])
        options["decision_file"] = link
    with pytest.raises(ValueError):
        stage_release(source, destination, **options)
    assert not destination.exists()


def test_source_guards_and_no_overwrite(experiment):
    source, destination, options = experiment
    with pytest.raises(ValueError, match="outside"):
        stage_release(source, source / "nested/release", **options)
    assert not (source / "nested").exists()
    stage_release(source, destination, **options)
    with pytest.raises(FileExistsError):
        stage_release(source, destination, **options)
    with pytest.raises(ValueError, match="already licensed"):
        stage_release(destination, source.parent / "relicensed", **options)
    assert not (source.parent / "relicensed").exists()
    (source / "model.bin").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Checksum"):
        stage_release(source, source.parent / "tampered", **options)
    assert not (source.parent / "tampered").exists()


def test_snapshot_detects_source_change(experiment, monkeypatch):
    from sleepkit.artifacts import release

    source, destination, options = experiment
    copytree = shutil.copytree

    def mutate_then_copy(src, dst, **kwargs):
        (src / "README.md").write_text("Different valid card")
        write_json(src / "checksums.json", {p.name: sha256(p) for p in src.iterdir() if p.name != "checksums.json"})
        return copytree(src, dst, **kwargs)

    monkeypatch.setattr(release.shutil, "copytree", mutate_then_copy)
    with pytest.raises(ValueError, match="changed"):
        stage_release(source, destination, **options)
    assert not destination.exists()


def test_runnable_does_not_promote_missing_evidence(experiment):
    source, destination, options = experiment
    report = json.loads((source / "validation.json").read_text())
    report["checks"][0]["status"] = "not_run"
    write_json(source / "validation.json", report)
    write_json(source / "checksums.json", {p.name: sha256(p) for p in source.iterdir() if p.name != "checksums.json"})
    with pytest.raises(ValueError, match="Missing runtime"):
        stage_release(source, destination, **options)
    assert not destination.exists()
    options["profile"] = "archive"
    stage_release(source, destination, **options)
    assert validate_bundle(destination)["checks"][0]["status"] == "not_run"


def test_cli_is_stdlib_only_and_hub_roundtrip_preserves_release(experiment, monkeypatch):
    source, destination, options = experiment
    body = source.parent / "card.md"
    body.write_text(options["card_body"])
    metadata = source.parent / "card-metadata.json"
    metadata.write_text('{"tags": ["test-fixture"]}')
    code = """
import sys
from sleepkit.artifacts.cli import main
main(sys.argv[1:])
assert not {"numpy", "tensorflow", "keras", "h5py", "huggingface_hub"} & sys.modules.keys()
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            "stage-release",
            str(source),
            str(destination),
            "--license-file",
            str(options["license_file"]),
            "--license-id",
            "other",
            "--license-name",
            options["license_name"],
            "--decision-file",
            str(options["decision_file"]),
            "--card-body",
            str(body),
            "--card-metadata",
            str(metadata),
            "--profile",
            "runnable",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(result.stdout)["path"] == str(destination)
    assert '"tags": ["test-fixture"]' in (destination / "README.md").read_text()
    remote = source.parent / "fake-hub"
    remote.mkdir()

    class Api:
        def create_repo(self, **kwargs):
            assert kwargs["private"] is True

        def create_commit(self, **kwargs):
            for operation in kwargs["operations"]:
                shutil.copyfile(operation.path_or_fileobj, remote / operation.path_in_repo)
            return SimpleNamespace(oid="a" * 40)

    def fetch(**kwargs):
        assert kwargs["revision"] == "a" * 40
        return str(remote / kwargs["filename"])

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            HfApi=Api, CommitOperationAdd=lambda **kwargs: SimpleNamespace(**kwargs), hf_hub_download=fetch
        ),
    )
    uploaded = publish_bundle(destination, "owner/fixture", upload=True, profile="runnable")
    downloaded = download_bundle("owner/fixture", revision=uploaded["revision"], profile="runnable")
    try:
        assert {p.name: p.read_bytes() for p in downloaded.iterdir()} == {
            p.name: p.read_bytes() for p in destination.iterdir()
        }
    finally:
        shutil.rmtree(downloaded.parent)


def test_card_metadata_preserves_structured_values_without_license_override(experiment):
    source, destination, options = experiment
    metadata = {
        "tags": ["tflite", "time-series"],
        "model-index": [{"name": "fixture", "results": []}],
        "description": "Newline\nlicense: mit",
        "private-example": False,
        "scientific-example": 1e-6,
    }
    stage_release(source, destination, card_metadata=metadata, **options)
    front_matter = (destination / "README.md").read_text().split("---\n")[1]
    for key, value in metadata.items():
        encoded = next(line for line in front_matter.splitlines() if line.startswith(json.dumps(key) + ": "))
        assert json.loads(encoded.split(": ", 1)[1]) == value
    assert front_matter.count("\nlicense:") == 0  # license is the first field; no injected second field


@pytest.mark.parametrize(
    "metadata",
    [
        [],
        {"license": "mit"},
        {"license_name": "other terms"},
        {"license_link": "elsewhere"},
        {1: "invalid key"},
        {"score": float("nan")},
        {"value": object()},
    ],
)
def test_invalid_card_metadata_is_rejected_before_staging(experiment, metadata):
    source, destination, options = experiment
    with pytest.raises(ValueError, match="metadata"):
        stage_release(source, destination, card_metadata=metadata, **options)
    assert not destination.exists()


def test_nested_card_metrics_keep_numeric_types_in_yaml(experiment):
    yaml = pytest.importorskip("yaml")
    source, destination, options = experiment
    metadata = {
        "model-index": [
            {
                "name": "fixture",
                "results": [
                    {
                        "metrics": [
                            {"type": "example", "value": value}
                            for value in (1e-6, 1e20, -1e-10, 1.25e-6, 0.0, -0.0, 5e-324, 12, True, None)
                        ]
                    }
                ],
            }
        ],
        "strings": ["1e-6", 'quote: "1e20" and backslash: \\', "line\n-1e-10"],
    }
    stage_release(source, destination, card_metadata=metadata, **options)
    header = (destination / "README.md").read_text().split("---\n")[1]
    decoded = yaml.safe_load(header)
    assert decoded == {
        "license": "other",
        "license_name": options["license_name"],
        "license_link": "LICENSE",
        **metadata,
    }
    metrics = decoded["model-index"][0]["results"][0]["metrics"]
    for actual, original in zip(metrics, metadata["model-index"][0]["results"][0]["metrics"]):
        assert type(actual["value"]) is type(original["value"])

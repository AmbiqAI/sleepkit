"""Artifact contracts can be exercised without TensorFlow or Hub credentials."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from sleepkit.artifacts import Artifact, Check, TensorSpec, stage_bundle, validate_bundle
from sleepkit.artifacts.hub import download_bundle, publish_bundle
from sleepkit.artifacts.package import sha256, write_json


def make_bundle(tmp_path, *, license=False, checks=(), references=()):
    model = tmp_path / "source.bin"
    model.write_bytes(b"existing model bytes")
    kwargs = {}
    if license:
        license_file = tmp_path / "terms.txt"
        license_file.write_text("Test-only model license")
        kwargs = {"license_file": license_file, "license_id": "other"}
    return stage_bundle(
        tmp_path / "bundle",
        title="Generic regression",
        card="# Model\n",
        artifacts=[
            Artifact(
                model,
                "model.bin",
                "model",
                "opaque",
                "test fixture",
                (
                    TensorSpec("scalar", (), "float32", "scalar input"),
                    TensorSpec("features", (None, 3), "float32", "other input"),
                ),
                (TensorSpec("prediction", (), "float32", "continuous scalar prediction"),),
            ),
            *references,
        ],
        checks=checks,
        **kwargs,
    )


def rehash(bundle):
    write_json(bundle / "checksums.json", {p.name: sha256(p) for p in bundle.iterdir() if p.name != "checksums.json"})


def test_generic_archive_and_no_overwrite(tmp_path):
    bundle = make_bundle(tmp_path)
    report = validate_bundle(bundle)
    assert report["integrity"] == "passed"
    assert (bundle / "model.bin").read_bytes() == b"existing model bytes"
    assert report["manifest"]["artifacts"][0]["inputs"][0]["shape"] == []
    with pytest.raises(FileExistsError):
        make_bundle(tmp_path)
    with pytest.raises(ValueError, match="Missing runtime"):
        validate_bundle(bundle, "runnable")


@pytest.mark.parametrize("mutation", ["bytes", "missing", "extra", "symlink", "manifest", "stale_evidence"])
def test_integrity_rejects_invalid_bundles(tmp_path, mutation):
    bundle = make_bundle(tmp_path)
    if mutation == "bytes":
        (bundle / "model.bin").write_bytes(b"different model")
    elif mutation == "missing":
        (bundle / "model.bin").unlink()
    elif mutation == "extra":
        (bundle / "credentials.txt").write_text("should never be uploaded")
    elif mutation == "symlink":
        (bundle / "model.bin").unlink()
        (bundle / "model.bin").symlink_to(tmp_path / "source.bin")
    elif mutation == "manifest":
        manifest = json.loads((bundle / "manifest.json").read_text())
        manifest["artifacts"][0]["bytes"] = 999
        write_json(bundle / "manifest.json", manifest)
        rehash(bundle)
    else:
        report = json.loads((bundle / "validation.json").read_text())
        report["checks"] = [Check("quality", "passed", "test", {"model.bin": "0" * 64}).to_dict()]
        write_json(bundle / "validation.json", report)
        rehash(bundle)
    with pytest.raises(ValueError):
        validate_bundle(bundle)


def test_staging_failure_is_atomic(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"model")
    with pytest.raises(ValueError, match="hash mismatch"):
        stage_bundle(
            tmp_path / "out",
            title="test",
            card="test",
            artifacts=[Artifact(source, "model", "model", "any", "test", expected_sha256="0" * 64)],
        )
    assert not (tmp_path / "out").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["source"]


@pytest.mark.parametrize("name", ["../escape", "/absolute", "nested/file", ".hidden", ""])
def test_paths_are_flat(name):
    with pytest.raises(ValueError):
        Artifact(Path("file"), name, "model", "any", "test")


def test_persisted_runtime_requires_bound_reference(tmp_path):
    model = tmp_path / "source.bin"
    model.write_bytes(b"existing model bytes")
    reference = tmp_path / "reference.npz"
    reference.write_bytes(b"test reference")
    check = Check(
        "runtime_conformance:model.bin",
        "passed",
        "fixture only",
        {"model.bin": sha256(model), "reference.npz": sha256(reference)},
    )
    bundle = make_bundle(
        tmp_path,
        checks=[check, Check("accuracy", "not_run", "Not evaluated")],
        references=[Artifact(reference, reference.name, "reference", "npz", "test")],
    )
    report = validate_bundle(bundle, "runnable")
    assert report["checks"][1]["status"] == "not_run"
    # Archive validation is persisted evidence only, not execution of the opaque format.
    (bundle / "reference.npz").write_bytes(b"stale")
    with pytest.raises(ValueError):
        validate_bundle(bundle, "runnable")


def test_dry_run_never_imports_hub_and_license_required(tmp_path, monkeypatch):
    bundle = make_bundle(tmp_path)
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    assert publish_bundle(bundle, "owner/model")["uploaded"] is False
    with pytest.raises(ValueError, match="license"):
        publish_bundle(bundle, "owner/model", upload=True)


def test_hub_upload_download_roundtrip(tmp_path, monkeypatch):
    bundle = make_bundle(tmp_path, license=True)
    remote = tmp_path / "remote"
    remote.mkdir()
    (remote / ".gitattributes").write_text("Hub metadata")
    (remote / "unrelated.txt").write_text("Existing repository file")
    calls = []

    class Api:
        def create_repo(self, **kwargs):
            calls.append(kwargs)

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
    result = publish_bundle(bundle, "owner/model", upload=True)
    assert result["revision"] == "a" * 40
    assert calls[0]["private"] is True
    downloaded = download_bundle("owner/model", revision=result["revision"])
    try:
        assert sorted(p.name for p in downloaded.iterdir()) == result["files"]
        assert sha256(downloaded / "model.bin") == sha256(bundle / "model.bin")
    finally:
        shutil.rmtree(downloaded.parent)
    with pytest.raises(ValueError, match="immutable"):
        download_bundle("owner/model", revision="main")


def test_artifact_cli_does_not_import_training_dependencies():
    code = """
import sys
from sleepkit.artifacts.cli import main
try:
    main(["--help"])
except SystemExit as exc:
    assert exc.code == 0
assert not {"tensorflow", "keras", "helia_edge", "numpy", "huggingface_hub"} & sys.modules.keys()
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)


def test_baseline_pins_sources_and_preserves_bytes(tmp_path, monkeypatch):
    from sleepkit.artifacts import baselines

    source = tmp_path / "source"
    source.mkdir()
    for name in baselines.HASHES:
        (source / name).write_bytes(b"test fixture")
    write_json(
        source / "configuration.json",
        {
            "job_dir": "/private/run",
            "feature": {"save_path": "/private/features"},
            "datasets": [{"params": {"path": "/private/data"}}],
            "frame_size": 240,
        },
    )
    monkeypatch.setattr(baselines, "HASHES", {p.name: sha256(p) for p in source.iterdir()})
    bundle = baselines.stage_baseline(source, tmp_path / "bundle")
    assert (bundle / "model.tflite").read_bytes() == b"test fixture"
    assert not (bundle / "configuration.json").exists()
    config_text = (bundle / "configuration.sanitized.json").read_text()
    assert "/private" not in config_text
    assert json.loads(config_text)["frame_size"] == 240
    assert all(c["status"] == "not_run" for c in validate_bundle(bundle)["checks"])
    (source / "model.tflite").write_bytes(b"different")
    with pytest.raises(ValueError, match="source hash"):
        baselines.stage_baseline(source, tmp_path / "other")

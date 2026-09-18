"""Optional Hub transport for existing artifact packages."""

import json
from pathlib import Path
import re
import shutil
import tempfile

from .package import validate_bundle
from .schema import filename


def publish_bundle(path, repo_id, *, upload=False, private=True, profile="archive"):
    """Dry run by default. Upload a validated snapshot without loading a model."""
    path = Path(path)
    report = validate_bundle(path, profile=profile)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", repo_id):
        raise ValueError("Repository ID must be owner/name")
    files = sorted(p.name for p in path.iterdir())
    result = {
        "repo_id": repo_id,
        "files": files,
        "profile": profile,
        "uploaded": False,
        "license": report["manifest"]["license"],
        "checks": report["checks"],
    }
    if not upload:
        return result
    if not report["manifest"]["license"]:
        raise ValueError("Choose and include a model artifact license before uploading")
    try:
        from huggingface_hub import CommitOperationAdd, HfApi
    except ImportError as exc:
        raise ImportError("Install sleepkit[hf] to upload; local staging needs no Hub dependency") from exc
    with tempfile.TemporaryDirectory(prefix="sleepkit-hf-") as directory:
        snapshot = Path(directory) / "bundle"
        shutil.copytree(path, snapshot, symlinks=True)
        validate_bundle(snapshot, profile=profile)
        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
        info = api.create_commit(
            repo_id=repo_id,
            repo_type="model",
            commit_message="Publish validated artifact bundle",
            operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(snapshot / name)) for name in files],
        )
        result.update(uploaded=True, revision=info.oid)
    return result


def download_bundle(repo_id, *, revision, profile="archive"):
    """Download an explicitly selected revision and validate its artifact inventory."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Download requires a full immutable Hub commit SHA")
    from huggingface_hub import hf_hub_download

    def fetch(name):
        return hf_hub_download(repo_id=repo_id, filename=name, revision=revision, repo_type="model")

    # Hub cache files can be symlinks. Copy only the declared inventory, ignoring
    # repository metadata such as .gitattributes or files from earlier releases.
    directory = Path(tempfile.mkdtemp(prefix="sleepkit-download-"))
    bundle = directory / "bundle"
    bundle.mkdir()
    try:
        shutil.copyfile(fetch("checksums.json"), bundle / "checksums.json")
        checksums = json.loads((bundle / "checksums.json").read_text())
        if not isinstance(checksums, dict) or "checksums.json" in checksums:
            raise ValueError("Invalid checksum inventory")
        for name in checksums:
            filename(name)
            shutil.copyfile(fetch(name), bundle / name)
        validate_bundle(bundle, profile=profile)
        return bundle
    except Exception:
        shutil.rmtree(directory)
        raise

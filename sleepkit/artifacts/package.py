"""Stage and verify existing artifact files; no model import or conversion."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from .schema import Artifact, Check, SCHEMA, TensorSpec, filename

RESERVED = {"manifest.json", "checksums.json", "validation.json", "README.md", "LICENSE"}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def stage_bundle(destination, *, title, artifacts, card, checks=(), metadata=None, license_file=None, license_id=None):
    """Copy an explicit artifact list into an immutable directory atomically.

    Artifact signatures and meanings are supplied by the caller/format adapter.
    A model need not be loaded to stage it. Unknown evidence stays not_run.
    """
    destination = Path(destination)
    artifacts = tuple(artifacts)
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    if not title or not artifacts or not any(a.role == "model" for a in artifacts):
        raise ValueError("A title and at least one model artifact are required")
    names = [a.name for a in artifacts]
    if len(set(names)) != len(names) or set(names) & RESERVED:
        raise ValueError("Duplicate or reserved artifact filenames")
    if bool(license_file) != bool(license_id):
        raise ValueError("Provide both a license file and its identifier, or neither")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".stage-") as directory:
        stage = Path(directory)
        inventory = []
        for artifact in artifacts:
            if not Path(artifact.source).is_file() or Path(artifact.source).is_symlink():
                raise ValueError(f"Artifact must be a regular file: {artifact.source}")
            target = stage / artifact.name
            shutil.copyfile(artifact.source, target)
            digest = sha256(target)
            if artifact.expected_sha256 and artifact.expected_sha256 != digest:
                raise ValueError(f"Source hash mismatch: {artifact.name}")
            inventory.append(
                {
                    "path": artifact.name,
                    "role": artifact.role,
                    "format": artifact.format,
                    "origin": artifact.origin,
                    "sha256": digest,
                    "bytes": target.stat().st_size,
                    "inputs": [asdict(t) for t in artifact.inputs],
                    "outputs": [asdict(t) for t in artifact.outputs],
                }
            )
        (stage / "README.md").write_text(card, encoding="utf-8")
        if license_file:
            if not Path(license_file).read_text(encoding="utf-8").strip():
                raise ValueError("License must not be empty")
            shutil.copyfile(license_file, stage / "LICENSE")
        write_json(
            stage / "manifest.json",
            {
                "schema": SCHEMA,
                "title": title,
                "artifacts": inventory,
                "license": license_id,
                "metadata": metadata or {},
            },
        )
        write_json(stage / "validation.json", {"schema": SCHEMA, "checks": [c.to_dict() for c in checks]})
        write_json(stage / "checksums.json", {p.name: sha256(p) for p in sorted(stage.iterdir())})
        validate_bundle(stage)
        stage.rename(destination)
    return destination


def validate_bundle(path, profile="archive"):
    """Verify structure/integrity and optional persisted runtime evidence.

    This does not rerun a runtime or establish task accuracy. Runtime replay is
    a separate adapter operation. Evidence must match the exact model/vector hashes.
    """
    if profile not in {"archive", "runnable"}:
        raise ValueError("Profile must be archive or runnable")
    path = Path(path)
    if not path.is_dir() or path.is_symlink():
        raise ValueError("Bundle must be a regular directory")
    for required in ("manifest.json", "checksums.json", "validation.json", "README.md"):
        if not (path / required).is_file() or (path / required).is_symlink():
            raise ValueError(f"Missing or invalid required file: {required}")
    manifest = json.loads((path / "manifest.json").read_text())
    checksums = json.loads((path / "checksums.json").read_text())
    report = json.loads((path / "validation.json").read_text())
    if manifest.get("schema") != SCHEMA or report.get("schema") != SCHEMA:
        raise ValueError("Unsupported artifact schema")
    if not isinstance(checksums, dict) or not {"manifest.json", "validation.json", "README.md"} <= checksums.keys():
        raise ValueError("Checksums omit required metadata")
    actual = {p.name for p in path.iterdir()}
    if actual != set(checksums) | {"checksums.json"}:
        raise ValueError("Bundle contains unlisted or missing files")
    for name, digest in checksums.items():
        filename(name)
        file = path / name
        if file.is_symlink() or not file.is_file() or sha256(file) != digest:
            raise ValueError(f"Checksum mismatch or unsafe file: {name}")
    inventory = manifest.get("artifacts", [])
    names = [a["path"] for a in inventory]
    if not inventory or len(set(names)) != len(names) or set(names) & RESERVED:
        raise ValueError("Invalid artifact inventory")
    declared = set(names) | {"manifest.json", "validation.json", "README.md"}
    if manifest.get("license"):
        declared.add("LICENSE")
        if not (path / "LICENSE").read_text().strip():
            raise ValueError("Missing license text")
    if declared != set(checksums):
        raise ValueError("Artifact inventory does not match staged files")
    model_names = set()
    for entry in inventory:
        name = filename(entry["path"])
        if entry["sha256"] != checksums[name] or entry["bytes"] != (path / name).stat().st_size:
            raise ValueError(f"Manifest fingerprint mismatch: {name}")
        spec = Artifact(
            path / name,
            name,
            entry["role"],
            entry["format"],
            entry["origin"],
            tuple(TensorSpec(**t) for t in entry["inputs"]),
            tuple(TensorSpec(**t) for t in entry["outputs"]),
        )
        if spec.role == "model":
            model_names.add(name)
    if not model_names:
        raise ValueError("No model artifacts")
    checks = [Check(**c) for c in report["checks"]]
    if len({c.name for c in checks}) != len(checks):
        raise ValueError("Duplicate validation check names")
    for check in checks:
        for name, digest in check.artifacts.items():
            if name not in names or checksums.get(name) != digest:
                raise ValueError(f"Stale evidence for {name}")
    if profile == "runnable":
        for name in model_names:
            if not any(
                c.name == f"runtime_conformance:{name}"
                and c.status == "passed"
                and name in c.artifacts
                and any(a["role"] == "reference" and a["path"] in c.artifacts for a in inventory)
                for c in checks
            ):
                raise ValueError(f"Missing runtime conformance evidence: {name}")
    return {"profile": profile, "integrity": "passed", "manifest": manifest, "checks": [c.to_dict() for c in checks]}

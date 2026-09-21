"""Promote a license-unspecified experiment bundle to an explicitly licensed release."""

import json
from pathlib import Path
import re
import shutil
import tempfile

from .package import sha256, stage_bundle, validate_bundle
from .schema import Artifact, Check, TensorSpec


RELEASE_FILES = {"experiment-card.md", "release-decision.md"}


def _yaml_value(value):
    """JSON flow values with float spellings also recognized by YAML 1.1 loaders."""
    encoded = json.dumps(value, allow_nan=False)
    # Match quoted strings as whole tokens so their contents are never rewritten.
    # PyYAML requires a decimal point in a scientific-notation float's mantissa.
    return re.sub(
        r'"(?:[^"\\]|\\.)*"|(-?\d+(?:\.\d+)?)([eE][+-]?\d+)',
        lambda match: (match[1] + ("" if "." in match[1] else ".0") + match[2] if match[1] is not None else match[0]),
        encoded,
    )


def stage_release(
    source,
    destination,
    *,
    license_file,
    license_id,
    card_body,
    decision_file,
    license_name=None,
    card_metadata=None,
    profile="archive",
):
    """Preserve model bytes/evidence and add caller-selected release documentation.

    This local, stdlib-only operation does not establish licensing rights, execute
    a runtime, or upload anything. Already licensed bundles are not relicensed.
    The caller supplies publication-ready text; no private evidence is discovered.
    """
    source, destination = Path(source), Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Destination already exists: {destination}")
    if destination.resolve().is_relative_to(source.resolve()):
        raise ValueError("Release destination must be outside the source bundle")
    if not isinstance(license_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9.+-]*", license_id):
        raise ValueError("Provide a Hugging Face license identifier")
    if license_id == "other":
        if not isinstance(license_name, str) or not license_name.strip():
            raise ValueError("A custom license requires license_name")
    elif license_name is not None:
        raise ValueError("license_name is only used with license_id='other'")
    if not isinstance(card_body, str) or not card_body.strip() or card_body.lstrip().startswith("---"):
        raise ValueError("Provide a nonempty model card body without YAML front matter")
    card_metadata = {} if card_metadata is None else card_metadata
    if not isinstance(card_metadata, dict) or any(not isinstance(key, str) or not key.strip() for key in card_metadata):
        raise ValueError("Card metadata must be a mapping with nonempty string keys")
    if {"license", "license_name", "license_link"} & card_metadata.keys():
        raise ValueError("Card license metadata is supplied through the explicit license arguments")
    try:
        metadata_lines = "".join(
            json.dumps(key) + ": " + _yaml_value(value) + "\n" for key, value in card_metadata.items()
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Card metadata must contain finite JSON-compatible values") from exc
    for file in (Path(license_file), Path(decision_file)):
        if file.is_symlink() or not file.is_file() or not file.read_text(encoding="utf-8").strip():
            raise ValueError("License and release decision must be nonempty regular text files")

    validate_bundle(source, profile=profile)
    source_digest = sha256(source / "checksums.json")
    with tempfile.TemporaryDirectory(prefix="sleepkit-release-") as directory:
        snapshot = Path(directory) / "bundle"
        shutil.copytree(source, snapshot, symlinks=True)
        report = validate_bundle(snapshot, profile=profile)
        if sha256(snapshot / "checksums.json") != source_digest:
            raise ValueError("Source bundle changed while taking the release snapshot")
        manifest = report["manifest"]
        if manifest["license"]:
            raise ValueError("Source is already licensed; this operation does not replace existing grants")
        if RELEASE_FILES & {entry["path"] for entry in manifest["artifacts"]}:
            raise ValueError("Source already contains reserved release documentation filenames")
        if "release_source" in manifest["metadata"]:
            raise ValueError("Source already contains release provenance")

        # Snapshot the two explicit release inputs too; stage_bundle pins their bytes.
        license_copy = Path(directory) / "LICENSE"
        decision_copy = Path(directory) / "release-decision.md"
        shutil.copyfile(license_file, license_copy)
        shutil.copyfile(decision_file, decision_copy)
        if (
            not license_copy.read_text(encoding="utf-8").strip()
            or not decision_copy.read_text(encoding="utf-8").strip()
        ):
            raise ValueError("License and release decision must not be empty")
        artifacts = [
            Artifact(
                snapshot / entry["path"],
                entry["path"],
                entry["role"],
                entry["format"],
                entry["origin"],
                tuple(TensorSpec(**value) for value in entry["inputs"]),
                tuple(TensorSpec(**value) for value in entry["outputs"]),
                expected_sha256=entry["sha256"],
            )
            for entry in manifest["artifacts"]
        ]
        artifacts.extend(
            [
                Artifact(
                    snapshot / "README.md",
                    "experiment-card.md",
                    "documentation",
                    "markdown",
                    "Original experiment model card; historical context, not the release license",
                    expected_sha256=sha256(snapshot / "README.md"),
                ),
                Artifact(
                    decision_copy,
                    "release-decision.md",
                    "documentation",
                    "markdown",
                    "Maintainer-supplied release decision; not an automated rights determination",
                    expected_sha256=sha256(decision_copy),
                ),
            ]
        )
        header = "---\nlicense: " + json.dumps(license_id) + "\n"
        if license_id == "other":
            header += "license_name: " + json.dumps(license_name.strip()) + '\nlicense_link: "LICENSE"\n'
        card = (
            header
            + metadata_lines
            + "---\n\n"
            + card_body.strip()
            + "\n\n"
            + (
                "Release terms are in [LICENSE](LICENSE); scope, attribution and the maintainer's\n"
                "decision are in [release-decision.md](release-decision.md).\n"
                "The [original experiment card](experiment-card.md) is retained for provenance;\n"
                "its historical license status does not replace this release's terms.\n"
            )
        )
        return stage_bundle(
            destination,
            title=manifest["title"],
            artifacts=artifacts,
            card=card,
            checks=[Check(**check) for check in report["checks"]],
            metadata={
                **manifest["metadata"],
                "release_source": {
                    "checksums_sha256": source_digest,
                    "manifest_sha256": sha256(snapshot / "manifest.json"),
                    "validation_sha256": sha256(snapshot / "validation.json"),
                },
            },
            license_file=license_copy,
            license_id=license_id,
        )

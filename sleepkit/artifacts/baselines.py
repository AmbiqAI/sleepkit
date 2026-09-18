"""Curated legacy baseline adapter; domain knowledge stays outside the bundle core."""

import json
from pathlib import Path
import tempfile

from .package import sha256, stage_bundle, write_json
from .schema import Artifact, Check, TensorSpec

BASELINE = "sd-2-tcn-sm"
SOURCE = "https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/sleepkit/detect/sd-2-tcn-sm/v1.0/"
HASHES = {
    "model.tflite": "eedf6690f182fabf1872ed85656f6a7a7fd1c259f232ed111a95ac9042e65fff",
    "metrics.json": "c7b7ef3b2371b67c17488fac4891cbb45c2297c7f17fc92d839b3cfe4faa8e6d",
    "configuration.json": "e0c61ed81723498c3f90e4d94274c63e83b4daa064ba305a660fb3a1327c7cb4",
}


def stage_baseline(source, destination, *, baseline=BASELINE, runtime_check=False, license_file=None, license_id=None):
    """Package a pinned historical TFLite baseline without conversion or retraining."""
    if baseline != BASELINE:
        raise ValueError(f"Only the reviewed {BASELINE} baseline is currently supported")
    source = Path(source)
    for name, digest in HASHES.items():
        if sha256(source / name) != digest:
            raise ValueError(f"Historical source hash differs: {name}")
    model = Artifact(
        source / "model.tflite",
        "model.tflite",
        "model",
        "tflite",
        SOURCE + "model.tflite",
        (
            TensorSpec(
                "serving_default_input:0",
                (None, 240, 5),
                "int8",
                "Externally normalized feature epochs in the documented feature order",
                0.11977417767047882,
                -95,
            ),
        ),
        (
            TensorSpec(
                "StatefulPartitionedCall_1:0",
                (None, 240, 2),
                "int8",
                "Per-epoch probabilities ordered WAKE, SLEEP",
                0.00390625,
                -128,
            ),
        ),
        HASHES["model.tflite"],
    )
    requirements = {
        "kind": "external_preprocessing_requirements",
        "executable": False,
        "feature_order": ["tod", "mov_mu", "mov_std", "angle_mu", "angle_std"],
        "raw_signals": ["TS", "ENMO", "ZANGLE"],
        "source_channel_meanings": {"TS": "seconds of day", "ENMO": "movement", "ZANGLE": "z-axis angle"},
        "feature_definitions": [
            "cos(2*pi*nanmean(seconds_of_day/86400))",
            "nanmean(ENMO)",
            "nanstd(ENMO)",
            "nanmean(ZANGLE)",
            "nanstd(ZANGLE)",
        ],
        "legacy_feature_generation": {"source_rate_hz": 0.2, "window_samples": 12, "stride_samples": 6},
        "cadence_unresolved": "Configuration sampling_rate is 0.0083333 Hz; feature code implies 60 s windows / 30 s stride. Verify before deployment.",
        "normalization": {
            "scope": "whole subject record, offline",
            "valid": "mask == 1",
            "invalid_fill": "mask == 0 replaced with per-feature nanmedian over valid frames",
            "formula": "(x - nanmean_valid) / sqrt(nanvar_valid + 1e-6)",
            "bundled_state": False,
        },
        "context_epochs": 240,
        "class_order": ["WAKE", "SLEEP"],
        "label_alignment": "last source label in each feature window",
        "code_reference": "SleepKit 0.11.1: sleepkit/features/fs_w_a_5.py and sleepkit/tasks/stage/utils.py",
        "configuration_source": SOURCE + "configuration.json",
        "configuration_sha256": HASHES["configuration.json"],
    }
    checks = [
        Check(
            "task_quality",
            "not_run",
            "metrics.json contains historical results; dataset/split provenance has not been revalidated.",
        ),
        Check(
            "preprocessing_equivalence",
            "not_run",
            "External whole-record preprocessing and the cadence discrepancy require verification.",
        ),
        Check(
            "keras_tflite_parity",
            "not_run",
            "Keras is not included. The preserved TFLite graph contains two final SOFTMAX operators.",
        ),
    ]
    with tempfile.TemporaryDirectory(prefix="sleepkit-baseline-") as directory:
        directory = Path(directory)
        write_json(directory / "preprocessing.json", requirements)
        configuration = json.loads((source / "configuration.json").read_text())
        # Preserve resolved settings offline, with filesystem locations removed.
        for key in ("job_dir", "model_file", "weights_file", "tflm_file", "val_file", "test_file"):
            if configuration.get(key) is not None:
                configuration[key] = "<omitted-local-path>"
        if "save_path" in configuration.get("feature", {}):
            configuration["feature"]["save_path"] = "<omitted-local-path>"
        for dataset in configuration.get("datasets", []):
            if "path" in dataset.get("params", {}):
                dataset["params"]["path"] = "<omitted-local-path>"
        write_json(directory / "configuration.sanitized.json", configuration)
        artifacts = [
            model,
            Artifact(
                directory / "configuration.sanitized.json",
                "configuration.sanitized.json",
                "source_configuration",
                "json",
                SOURCE + "configuration.json",
            ),
            Artifact(
                source / "metrics.json",
                "metrics.json",
                "historical_metrics",
                "json",
                SOURCE + "metrics.json",
                expected_sha256=HASHES["metrics.json"],
            ),
            Artifact(
                directory / "preprocessing.json",
                "preprocessing.json",
                "preprocessing_requirements",
                "json",
                "Reviewed legacy source; requirements only, not executable preprocessing",
            ),
        ]
        if runtime_check:
            from .runtime import create_reference

            reference = directory / "reference.npz"
            checks.append(create_reference(model, reference))
            artifacts.append(
                Artifact(
                    reference,
                    reference.name,
                    "reference",
                    "npz",
                    "Deterministic synthetic runtime input/output; seed 0",
                )
            )
        else:
            checks.append(
                Check(
                    "runtime_conformance:model.tflite",
                    "not_run",
                    "Run staging with --runtime-check to record synthetic I/O.",
                )
            )
        return stage_bundle(
            destination,
            title="SleepKit SD-2-TCN-SM historical baseline",
            artifacts=artifacts,
            card=_card(license_id, runtime_check),
            checks=checks,
            metadata={"sleepkit": {"baseline": BASELINE, "source_release": "v1.0", "historical": True}},
            license_file=license_file,
            license_id=license_id,
        )


def _card(license_id, runtime_checked):
    license_line = f"license: {json.dumps(license_id)}\n" if license_id else ""
    return f"""---
{license_line}tags:
- sleepkit
- tflite
- time-series
- sleep-detection
---
# SleepKit SD-2-TCN-SM historical baseline

This package preserves the original int8 TFLite bytes from the [v1.0 source]({SOURCE}model.tflite).
It is an archival baseline for comparison, with explicit input/output contracts in `manifest.json`.
It is not a raw-signal or streaming pipeline. Model artifact licensing: {license_id or "not yet specified; publication is disabled"}.

## Input and output

Input: int8 `[batch, 240, 5]`, features ordered `tod, mov_mu, mov_std, angle_mu, angle_std`.
Normalize the whole subject record externally before quantization. Input scale is
0.11977417767047882 and zero point is -95. Only the allocated batch size 1 is exercised by the runtime check.
Output: int8 `[batch, 240, 2]`, per-epoch probabilities ordered WAKE, SLEEP.
Dequantize with `(output + 128) / 256`. Do not apply another softmax.
The preserved graph contains two final SOFTMAX operators; parity with Keras has not been established.

## Preprocessing limitations

`configuration.sanitized.json` preserves the original resolved settings with local paths replaced
by `<omitted-local-path>`; the original configuration hash is in `preprocessing.json`.
`preprocessing.json` describes requirements, not an executable transform. Legacy preprocessing
uses ENMO/ZANGLE statistics and time of day, fills invalid frames from valid-frame medians,
and standardizes using whole-record valid-frame statistics. No normalization state is bundled.
The legacy feature code implies 60-second windows with a 30-second stride, while the
source configuration records sampling_rate 0.0083333 Hz. This unresolved discrepancy must
be checked before assigning timestamps or using the model in a deployment.

## Evidence

Synthetic runtime conformance: {"recorded; see validation.json and reference.npz" if runtime_checked else "not run"}.
This checks the declared tensors and finite inference, not predictive quality or raw-signal preprocessing.
`metrics.json` is copied from the historical release: reported accuracy approximately 0.9562 and
F1 approximately 0.9565. These results have not been reproduced and their dataset/split provenance
has not been revalidated. Check statuses and exact artifact hashes are recorded in `validation.json`.
Checksums establish package consistency, not publisher authenticity or clinical suitability.

## Minimal synthetic inference

Install `numpy` and `ai-edge-litert`. Download this bundle at a fixed commit revision.
The following exercises the model only; replace the synthetic features with correctly preprocessed data:

```python
import numpy as np
from ai_edge_litert.interpreter import Interpreter

runner = Interpreter(model_path="model.tflite")
runner.allocate_tensors()
i = runner.get_input_details()[0]
o = runner.get_output_details()[0]
features = np.zeros((1, 240, 5), dtype=np.float32)  # synthetic standardized features
quantized = np.clip(np.rint(features / 0.11977417767047882 - 95), -128, 127).astype(np.int8)
runner.set_tensor(i["index"], quantized)
runner.invoke()
probabilities = (runner.get_tensor(o["index"]).astype(np.float32) + 128) / 256
```
"""

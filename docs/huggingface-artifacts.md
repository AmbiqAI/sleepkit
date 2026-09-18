# Publish an existing baseline to Hugging Face

The additive `sleepkit-artifacts` CLI packages existing files without training or
conversion. The existing `sleepkit` CLI is unchanged. The first reviewed adapter
supports **SD-2-TCN-SM v1.0 TFLite**. Other models can use the Python artifact API
once their signatures, provenance, and preprocessing requirements are reviewed.

This is the first independent slice toward the major-version redesign: ordinary
Python recipes can produce artifacts without inheriting a trainer, using a task
registry, or expressing orchestration in configuration. The bundle layer uses
only the standard library; LiteRT and Hub transport are optional adapters.

## Run from a checkout

Python 3.12 or 3.13 is required. For structural staging, run from the checkout
with `python -m sleepkit.artifacts`; no third-party imports are needed. Installing
SleepKit normally still installs its existing training dependencies. This slice
does not change those dependencies or upgrade TensorFlow.

For optional synthetic runtime checks in a separate environment:

```sh
python -m venv /tmp/sleepkit-artifacts-env
. /tmp/sleepkit-artifacts-env/bin/activate
python -m pip install 'numpy>=2.1,<3' 'ai-edge-litert>=2.2,<3'
```

Obtain the three historical source files (the adapter verifies their pinned
SHA-256 hashes before staging). Source files and generated bundles belong outside
the repository:

```sh
mkdir -p /tmp/sd-2-source
for file in model.tflite metrics.json configuration.json; do
  curl --fail --location \
    "https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/sleepkit/detect/sd-2-tcn-sm/v1.0/$file" \
    --output "/tmp/sd-2-source/$file"
done
python -m sleepkit.artifacts stage-baseline \
  --source /tmp/sd-2-source --output /tmp/sd-2-bundle --runtime-check
python -m sleepkit.artifacts validate /tmp/sd-2-bundle --profile runnable --runtime
python -m sleepkit.artifacts publish /tmp/sd-2-bundle AmbiqAI/sleepkit-sd-2-tcn-sm --profile runnable
```

The last command is a local dry run, requires no token or Hub package, and prints
the exact file inventory and check statuses. Staging never overwrites an existing
directory. Omit `--runtime-check` to produce an archive without running a model.

## What validation means

- `archive`: inventory, signatures, checksums, and evidence references are structurally consistent.
- `runnable`: each model also has passing persisted runtime conformance evidence bound
  to its exact model and reference-file hashes. This profile does not execute code.
- `validate --runtime`: additionally loads TFLite and replays the saved synthetic
  inputs and expected outputs. Integer outputs allow one quantization unit of
  deviation; floating outputs use absolute tolerance 1e-6 and relative tolerance 1e-5.

Checks explicitly report `passed`, `failed`, `not_run`, or `not_applicable`.
An archival bundle can contain failed or missing quality evidence. Even the runnable
profile does not establish task accuracy, preprocessing equivalence, or readiness
for deployment. Checksums detect changes, not a malicious publisher rewriting the
entire bundle. LiteRT checks use the allocated shapes (batch 1 for this baseline),
not every possible dynamic shape; the adapter supports per-tensor I/O quantization.

The bundle contains `model.tflite`, original `metrics.json`, descriptive
`preprocessing.json`, a model card, manifest, checksums, validation report, and
a sanitized source configuration, and optionally `reference.npz` and `LICENSE`.
The configuration preserves resolved settings but replaces local paths with
`<omitted-local-path>`; the original configuration hash remains in preprocessing metadata. Synthetic references contain no subject data.

The baseline retains its original graph, including two final SOFTMAX operators.
Output probabilities must not receive another softmax. Historical metrics have
not been reproduced. Whole-record external preprocessing is required, and the
configuration's sampling rate conflicts with the feature-generation cadence.
These issues are documented in the model card and remain open for the next recipe.

## Actual publication

Choose the model artifact license explicitly; the source-code license is not
assumed to cover the weights. Stage a new bundle with `--license-file PATH` and
`--license-id SPDX_ID` (or an appropriate Hugging Face license identifier). Include
any required attribution in that license. Then install `huggingface-hub>=1,<2`,
authenticate using its standard token mechanism, and explicitly invoke:

```sh
python -m sleepkit.artifacts publish /tmp/licensed-sd-2-bundle \
  AmbiqAI/sleepkit-sd-2-tcn-sm --profile runnable --upload
```

Publication copies and revalidates the bundle, then commits only its explicit
inventory. New repositories default to private; `--public` requests public creation.
Existing repository visibility and unrelated files are unchanged. Updating a
repository replaces matching bundle filenames. Use a dedicated model repository.
The returned revision is the Hub commit SHA. This hosts downloadable artifacts;
it does not provision a hosted inference endpoint.

Use the returned immutable revision for download and verification:

```python
import shutil
from sleepkit.artifacts.hub import download_bundle
from sleepkit.artifacts.runtime import replay_bundle

bundle = download_bundle("AmbiqAI/sleepkit-sd-2-tcn-sm", revision="FULL_40_CHARACTER_COMMIT_SHA", profile="runnable")
try:
    print(replay_bundle(bundle))
    # Copy the validated bundle to durable storage if desired.
finally:
    shutil.rmtree(bundle.parent)  # caller owns the materialized temporary download
```

Downloads materialize only the declared inventory from that revision, ignoring
Hub metadata and unrelated repository files. No model code is downloaded or
executed during structural validation.

See Hugging Face's [upload guide](https://huggingface.co/docs/huggingface_hub/guides/upload)
and [model card guide](https://huggingface.co/docs/hub/model-cards) for transport and
card conventions.

## Reuse from future recipes

```python
from pathlib import Path
from sleepkit.artifacts import Artifact, TensorSpec, stage_bundle

stage_bundle(
    "/tmp/experiment-bundle",
    title="My regression experiment",
    artifacts=[Artifact(
        source=Path("results/model.keras"), name="model.keras", role="model",
        format="keras", origin="experiment source revision and run identifier",
        inputs=(TensorSpec("features", (None, 8), "float32", "Eight normalized sensor features"),),
        outputs=(TensorSpec("estimate", (None, 1), "float32", "Continuous estimate in physical units"),),
    )],
    card="# Experiment\nDescribe data, preprocessing, evaluation, and limitations here.\n",
)
```

The caller owns tensor meanings, model cards, preprocessing, and checks. The core
supports multiple model artifacts, multiple inputs/outputs, scalar tensors, and
non-classification tasks. It does not load Keras or infer semantics from a graph.
A future Keras validator can produce its own hash-bound evidence without changing
this API. The schema is provisional until a second, independent recipe exercises it.

Next: one native Python train/evaluate/export recipe with an explicit preprocessing
step and a small recipe-specific config. Keep the legacy CLI available, then extract
shared blocks only as additional recipes demonstrate the need.

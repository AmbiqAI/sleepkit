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

## Stage a licensed release from an experiment

Once the [per-model license decision](model-licensing-policy.md) is established,
promote the verified experiment into a new release directory:

```sh
python -m sleepkit.artifacts stage-release /path/to/experiment/bundle /path/to/release \
  --license-file /path/to/chosen/LICENSE --license-id bsd-3-clause \
  --card-body /path/to/release-card.md --decision-file /path/to/release-decision.md \
  --profile runnable
python -m sleepkit.artifacts validate /path/to/release --profile runnable --runtime
```

The BSD identifier above is an example for a model whose rights permit it, not a
license choice for the current CMIDSS candidate. For an appropriate custom license,
use `--license-id other --license-name "Your finalized license name"`. Standard IDs
and custom metadata follow [Hugging Face's license conventions](https://huggingface.co/docs/hub/repositories-licenses).
The identifier is checked for syntax, not against a live registry or the license
text. The maintainer is responsible for choosing matching terms and metadata.

`release-card.md` is Markdown **without YAML front matter**. Write the model's task,
input/preprocessing contract, metrics, limitations and usage example there. The
helper generates the license front matter and links to the supplied license and
decision. To add tags, dataset identifiers, library information or model-index
entries, supply `--card-metadata /path/to/card-metadata.json`, for example
`{"tags": ["tflite", "time-series"]}`. Python callers can pass the same dictionary as
`card_metadata`. License fields are reserved for the explicit license arguments.
Values must be finite JSON-compatible data; no YAML parser is required. Original
experiment metadata is retained in the old card but is not automatically copied
into the active release card, so select the fields appropriate to this release.

`release-decision.md` is a publication-ready human record of the covered filenames
and hashes, dataset/parent-weight terms, permission basis, attribution, permitted
uses, maintainer and decision date. Review this text for private information: it is
copied verbatim, as is the original experiment card. Do not supply private agreements,
subject lists or internal evidence files. A nonempty decision document records the
maintainer's decision; the tool cannot establish rights or approve publication.

The same operation is available as ordinary Python:

```python
from pathlib import Path
from sleepkit.artifacts import stage_release

release = stage_release(
    "experiment/bundle", "release",
    license_file="chosen/LICENSE", license_id="bsd-3-clause",
    card_body=Path("release-card.md").read_text(encoding="utf-8"),
    decision_file="release-decision.md", profile="runnable",
)
```

Staging needs only the standard library. It snapshots and validates the source,
preserves all declared artifact bytes/signatures and checks, and retains the source
card as `experiment-card.md`. The source manifest, validation and checksum inventory
hashes are recorded in `metadata.release_source`; recipe metadata stays intact.
New documentation, license and checksums go into a new directory. Staging does not
retrain, reconvert, rerun inference, improve a check status, or contact the Hub.
`runnable` requires existing persisted runtime evidence; the separate `validate
--runtime` command replays it.

The source must have no model license recorded in its manifest. This operation
refuses already licensed bundles, existing destinations, destinations inside the
source, and collisions with its release documentation/provenance fields. An
unspecified manifest license does not establish freedom to license the weights;
review any existing grants and upstream terms first. Already licensed bundles can
use the existing validate/publish commands without this promotion step.

## Actual publication

Choose the model artifact license explicitly; the source-code license is not
assumed to cover the weights. Use `stage-release` above for an existing experiment,
or supply `--license-file PATH` and `--license-id HF_LICENSE_ID` when staging a
historical baseline. Include required attribution and retain upstream notices. Then install `huggingface-hub>=1,<2`,
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

The native Python detection recipe now integrates preprocessing, training, evaluation
and export. Next, exercise a deliberately different recipe before promoting shared
blocks into a broader API; keep the legacy CLI available during this migration.

The [membership int8 release preparation](detection-int8.md) now stages a new-model
bundle with preserved float/checkpoint artifacts, train-only calibration evidence,
matched evaluation, and supported raw-sensor inference for both TFLite models.
It can be validated and inspected through the same Hub dry-run interface. Model
artifact licensing remains explicit; staging does not upload a model.

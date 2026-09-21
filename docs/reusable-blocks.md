# Reusable blocks being evaluated for heliaEDGE

sleepKIT should be able to compose a small experiment in Python, add reproducible
data and evaluation contracts, and package a release without inheriting a trainer
framework. The first local extractions cover offline classification arithmetic and
explicit file snapshots. They live under `sleepkit._edge_candidates`; recipes import
them through `sleepkit.recipes._components`. These are private candidates, not new
heliaEDGE APIs or permanent sleepKIT framework classes.

The existing detection recipe consumes these blocks. A second, synthetic
three-class signal recipe exercises ordinary sample classification, optional Keras
backends, an EDGE custom layer, checkpoint reload and the existing artifact bundle
format. This is a supported integration example, not a sleep-model quality result
or evidence of adoption by a second real domain.

## Boundaries and order of adoption

| Candidate | Existing duplication / consumer | Shared contract | KIT responsibility |
| --- | --- | --- | --- |
| Offline classification | Detection evaluation, scoring summaries and historical comparison; synthetic signals | NumPy-only K-class confusion counts, stable logit cross-entropy and explicit aggregation | Eligibility, unknown labels, class names, probability interpretation, split selection and report schemas |
| File snapshots | Detection source/code checks; signal code/evidence checks; artifact hashing | Stdlib-only explicit inventory, SHA-256, copied digest mapping and later verification | Which files matter, stable labels, public/private disclosure and whether directory additions matter |
| Artifact contracts and transport | Existing detector bundles; signal archive bundles | `TensorSpec`, `Artifact`, `Check`, stage/validate plus separate optional Hub transport | Input meaning, calibration, license/attribution, model card and quality/hardware claims |
| Prepared arrays and finite loaders | Existing feature cache and reader | Candidate for later measurement and a second real workload | Source decoding, time/units, annotation semantics, sampling and fitted-state scope |

Keep the optimized wrist features, context construction, CMIDSS evidence adapter,
subject splitting and dataset rights in sleepKIT. EDGE should not acquire those
policies merely because it provides cache, statistics or artifact utilities.
Do not extract a generic pipeline runner, configuration DSL, dataset registry or
universal tensor wrapper. The recipes own their composition and small configs.

## First concrete interfaces

```python
from sleepkit.recipes._components import ClassificationAccumulator, FileSnapshot

metrics = ClassificationAccumulator(classes=3)
metrics.update(integer_targets, raw_logits)  # logits shape = targets.shape + (3,)
report = metrics.result()

snapshot = FileSnapshot.capture({"config": config_path, "source": source_path})
run_selected_operations()
snapshot.verify()
hashes = snapshot.hashes()
```

Classification is unweighted and finite. It retains integer confusion counts,
loss sum and sample count; ties select the first class. Macro-F1 includes every
declared class with zero-division zero, and recall for absent support is `None`.
Unknown labels, mismatched shapes, nonfinite logits and unrepresentable loss/count
sums are rejected before changing state. An empty update is allowed; an empty
result is rejected. Consumers explicitly convert framework tensors to NumPy and
own masking. Confusion-only summaries never infer a probability loss from historical
class predictions. Detection wrappers preserve their existing field names and
empty-result policies.

Snapshots copy a caller-selected inventory and accept regular, non-symlink leaf
files. Missing or changed content is rejected on verification. This is a content
check between calls, not locking, a directory-completeness check or protection
against concurrent mutation/restoration. File labels may be private; the helper
does not redact them. Detection now uses this regular-file rule for source snapshots;
use actual HDF5 files rather than individual HDF5 symlinks for that entry point.
Callers separately check inventory additions where required.

Atomic bundle staging remains separate from snapshots and experiment lifecycle.
Some failed experiments intentionally retain declarations; a completed release
should appear only after validation. Existing staging uses a temporary sibling and
rename, but does not claim concurrency-safe exclusive publication against competing
writers. These policies do not belong in a universal run-manager object.

Moving an implementation must not weaken traceability. The recipe inventory now
includes local shared blocks, the explicit import adapter and artifact modules,
as well as recipe sources. Existing historical reports remain unchanged. A future
EDGE implementation must be bound to its installed version and, for a Git install,
its immutable commit and relevant implementation identity.

## Temporary Git integration

The optional consumer environments use EDGE commit
`0661c5f58cbad27681d5e2f6a8cca7477477c853`. The requirements profiles pin the code
and principal training packages; they are not complete transitive lockfiles. Runs
record the actual environment and PEP 610 installation provenance. They do not
change the legacy project lock or the existing EDGE 0.4.1 baseline environment.

```sh
uv venv --python 3.12 .venv-edge-tf
uv pip install --python .venv-edge-tf/bin/python -r requirements/edge-tensorflow.txt
KERAS_BACKEND=tensorflow .venv-edge-tf/bin/python -m sleepkit.recipes.signals --output /path/to/new-run

uv venv --python 3.12 .venv-edge-torch
uv pip install --python .venv-edge-torch/bin/python 'torch==2.14.0' --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-edge-torch/bin/python -r requirements/edge-torch.txt
KERAS_BACKEND=torch .venv-edge-torch/bin/python -m sleepkit.recipes.signals --output /path/to/another-new-run
```

Use Python 3.12.3 or newer within the supported profile range; the TF example is
qualified on Python 3.12.5. Set the backend before importing Keras. The Torch profile
must remain usable without TensorFlow installed. A Torch `.keras` checkpoint and
successful reload do not imply LiteRT conversion or MCU support.

## Upstream adoption and removal

1. Keep candidates independently usable and prove compatibility through the two
   local recipes and conformance fixtures. Add another real domain consumer before
   promoting broad cross-KIT APIs.
2. Agree names and semantics with EDGE. Its current Keras-backed metrics are not a
   drop-in replacement for an offline NumPy accumulator; its file helpers do not
   provide this snapshot contract. Keep optional transport separate from storage.
3. Implement upstream without importing Keras/TF/Torch from evidence/artifact APIs.
   Run the same fixtures against the upstream classes, preserve old artifact schemas
   and prove the actual detection reports remain equivalent.
4. Pin an exact Git commit while waiting for a release. Explicitly switch the one
   recipe import adapter and artifact hash import; do not silently select between
   local/upstream implementations based on whatever happens to be installed.
5. Delete local implementations only after consumer checks pass. Switch the Git
   requirement to the tested PyPI version when published, retaining historical run
   provenance. Documentation and artifacts must distinguish installed versions from
   the versions used for the original baseline.

`tests/test_edge_candidates.py` holds numerical, malformed-input, immutability,
import-isolation and detection-report compatibility checks. The signal recipe's
integration tests exercise training, a real EDGE custom object, save/reload and
bundle validation in backend-specific environments. Temporary environments and run
artifacts belong outside tracked source; the recipe, contracts and tests are durable
project capabilities.

# Train a reproducible saved-feature staging experiment

This additive recipe connects explicit subject membership, preprocessing,
training, evaluation and a local model archive. It uses ordinary Python blocks
and Keras array adapters. It does not regenerate features from raw sensors.
See [the saved-feature contract](staging-baseline.md) for feature order, label
mapping, whole-subject normalization and its historical limitations.

## Start with explicit subjects

```python
from pathlib import Path
from sleepkit.artifacts.package import write_json
from sleepkit.recipes.staging.split import create_split

root = Path("/private/fs-w-pa-14-60/mesa")
subjects = sorted(path.stem for path in root.glob("*.h5"))
# For the declared 1,900-record MESA cohort only:
assert len(subjects) == 1900
manifest = create_split(
    subjects, dataset="mesa/fs-w-pa-14-60", seed=0,
    validation_count=304, test_count=380,
)
write_json("/private/split.json", manifest)
```

The manifest contains `schema`, `dataset`, and explicit `train`, `validation`,
and `test` lists. Lists must be nonempty, unique and disjoint, with safe filename
stems. The generator permutes sorted IDs with NumPy `default_rng(seed)`, takes
first test IDs, then validation IDs, then training IDs, and sorts each partition.
Persist the resulting manifest; a seed alone is not membership evidence. The
provider must group every recording/alias of one person under one subject identity.
The recipe cannot establish real-person identity from unannotated HDF5 files.

These are new partitions, not a reconstruction of the historical train/validation
split. Splitting occurs before reading features or labels. No class stratification,
label-driven filtering, replacement sampling, or post-test resplitting is used.

## Experiment in Python

```python
from sleepkit.recipes.staging.data import prepare_partition
from sleepkit.recipes.staging.recipe import Config, train, evaluate
from sleepkit.recipes.staging.split import load_split

manifest = load_split("/private/split.json")
training = prepare_partition(root, manifest["partitions"]["train"])
validation = prepare_partition(root, manifest["partitions"]["validation"])
model, history = train(training, validation, Config(epochs=2, batch_size=32))
# train(..., model_builder=my_builder) accepts a regular Python Keras builder.
# For other recipes, use the prepared arrays directly with Keras model.fit.
metrics = evaluate(model, validation)
```

The default model is a small Conv1D network with float32 `[batch,240,14]` inputs
and linear `[batch,240,3]` outputs ordered WAKE, NREM, REM. The lower-level `train`
function accepts another builder with this interface; `train` and `evaluate` reject
models without it, including softmax outputs. The promoted `run` and
golden flow fix the default builder so they can record its implementation.
Custom experimental builders do not automatically inherit golden provenance.

Preparation materializes complete windows in memory and records their private
subject/start coordinates. Each record is normalized independently using all its
quality-valid feature rows, including its future epochs and incomplete tail.
Invalid/unknown epochs stay in context; only quality-valid known stages are scored.
Zero-supervision windows are omitted from fitting/evaluation arrays and counted;
short-record and incomplete-tail coverage remains recorded. An empty supervised
partition fails before fitting. Records with no quality-valid normalization rows
or nonfinite quality-valid features fail, rather than being silently removed.

Every retained training window appears once per epoch, shuffled using the recorded
seed; final partial batches are retained. Validation is separate and does not
update weights. The sparse loss receives valid class-0 placeholders at excluded
positions plus zero weights. `mean_with_sample_weight` divides batch loss by the
number of scored epochs, so ignored epochs do not dilute gradients. Keras history
aggregates batch-normalized losses; final evaluation instead computes pooled
scored-epoch cross entropy and classification metrics. They are different summaries.

This is an in-memory implementation, not a streaming loader. Per-partition
`prepared_array_bytes` records the NumPy arrays, excluding temporary copies,
Keras tensors and framework overhead. There is no backend speed or bounded-RSS
claim. Both Keras backends have synthetic training/reload coverage; the normal
sleepKIT package still declares TensorFlow as a base dependency. Torch-only tests
use the isolated optional EDGE consumer profile, not a TensorFlow-free package
installation claim.

## Promote a configuration before running it

Use the frozen TensorFlow environment for the first real golden experiment.
Choose the configuration before model fitting or test evaluation:

```sh
KERAS_BACKEND=tensorflow python -m sleepkit.recipes.staging declare \
  --features-dir /private/fs-w-pa-14-60/mesa \
  --split /private/split.json \
  --output /private/golden.json \
  --name mesa-staging-conv1d-seed0 \
  --epochs 5 --batch-size 32 --learning-rate 0.001 --seed 0

KERAS_BACKEND=tensorflow python -m sleepkit.recipes.staging run \
  --features-dir /private/fs-w-pa-14-60/mesa \
  --split /private/split.json \
  --golden /private/golden.json \
  --output /private/new-run
```

Declaration hashes files but does not read labels or train. It fixes the config,
dataset identity, semantic split, source inventory, selected implementation files,
Python/package versions and Keras backend. A run must match those identities;
changing code/data/runtime is a new qualification decision, not a silent golden
update. This is procedural reproducibility, not a promise of bitwise training
results across machines/backends. Golden names are labels, not orchestration.

The repository's [prospective MESA declaration](https://github.com/AmbiqAI/sleepkit/blob/main/experiments/staging-golden.json)
pins the first five-epoch, seed-0 configuration. It contains hashes, not private
subject IDs or learned performance thresholds. Recreate or retain the private
manifest above and match the declared runtime to replay it.

The seed-0 results reported in [#41](https://github.com/AmbiqAI/sleepkit/pull/41)
were produced at commit `46605b4`, before `evaluate` validated the model interface.
That change re-pinned the declaration's implementation hashes, so those results are
evidence for that commit, not a run of the current declaration.

The default run uses the final declared epoch, without early stopping, test-driven
selection or tuned thresholds. The test partition is scored after training and
is never passed to fit. The complete manifest is validated before preparation;
reading test data for schema/preprocessing checks does not update the model.

## Inspect the result

The **private run directory** preserves `declaration.json`, `split.json`,
`sources.json`, `windows.json`, `history.json` and `completed.json`. Keep it with
the source dataset. The completion marker is written only after the model archive
is produced; failures may leave intermediate evidence. Existing output directories
are refused. Source, declaration, golden, and selected code bytes are checked for
mutation; training additionally rejects changes to its enumerated Python-file
inventory. These checks do not lock files or prevent concurrent writes.

The `bundle/` directory is a checksummed archive: final Keras model, aggregate
validation/test metrics, recipe/environment/config/history, class/feature contract,
executable `preprocessing.py`, and **synthetic** normalized reload vectors. Subject
IDs, per-subject metrics, actual feature arrays and source paths stay out of it.
Hashes are provenance, not anonymization. Each prepared held-out model-input batch
is compared before/after save/load at `atol=1e-5`, `rtol=1e-4`; actual held-out
logits remain in memory. A fresh-process test exercises model and bundled
preprocessing imports. Follow the model card's no-bytecode example to preserve
the immutable bundle inventory when importing its Python file.

This Keras archive has its own explicit reload check. The existing generic
`replay_bundle` adapter is TFLite-only and is not the consumer for this archive.
No HF upload, model license assignment, TFLite conversion, production, clinical or
hardware qualification is included. Raw extraction provenance remains unverified,
and the historical SS-3 checkpoint remains a separate comparator with unrecorded
training membership. New held-out results apply to the declared IDs and offline
protocol; they do not establish historical model equivalence.

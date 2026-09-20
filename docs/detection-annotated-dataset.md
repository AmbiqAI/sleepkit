# Training with the annotated-period target

`AnnotatedDataset` composes the verified source, raw events, candidate policy, and
frozen assignment as a read-only source adapter. It reconstructs the verified UTC
clock and builds target arrays in memory. It neither loads historical
`sleep_stages` values nor rewrites original files. This keeps target derivation
versioned without duplicating the entire sensor dataset.

```python
from sleepkit.recipes.detection.target_dataset import AnnotatedDataset
from sleepkit.recipes.detection.recipe import Config, run_membership

source = AnnotatedDataset(
    root="/path/to/original-cmidss",
    events_path="/path/to/train_events.csv",
    alignment_path="/path/to/source-event-alignment.json",
    coverage_path="/path/to/candidate-coverage.json",
    frozen_dir="/path/to/frozen-split",
)
run_membership(
    source,
    "/path/to/new-run",
    Config(context=source.context, epochs=5, batch_size=32),
    cache="/path/to/feature-cache",
)
```

The adapter checks protocol/file hashes, target and context policy, group assignment,
exact cohort membership, source hashes, and event/coverage/alignment bindings. Every
labeled read reproduces the recorded candidate coverage; unexpected differences fail.
Evidence and source changes fail before a run is published. Reports remain trusted
local evidence, not external attestations.

`source.read(subject, labels=False)` supplies sensors and an independent clock
without constructing labels. The same reader feeds train-only fitted normalization,
training, validation, and held-out evaluation. Normalization uses all valid frames
from training subjects, including frames with unknown targets, as specified by v3.
Model contexts with unknown targets or nonfinite sensor windows remain excluded;
context grouping never compacts gaps. The recipe rejects a context length that
differs from the frozen protocol. `model_builder` and ordinary Keras callbacks remain
replaceable Python arguments; no pipeline registry or orchestration configuration
is introduced.

The original `run(...)` entry point still evaluates historical HDF5 labels. Selecting
`run_membership(...)` explicitly chooses the frozen derived target. Synthetic fixtures
must declare `data_kind="synthetic"` in either recipe.

## Output contract and local scoring evidence

New membership bundles use `sleepkit.detection/v2` with the target contract and class
order from `sleepkit.annotated_period_membership/v1`. Historical recipe bundles keep
v1 and their WAKE/SLEEP names. The shared inference adapter accepts both, validates
the exact recipe/target/class combination, and returns `class_names` and `target`
alongside logits, probabilities, recording-relative times, and context availability.
Membership means inside/outside supported annotated periods; it is not clinical
per-sample sleep/wake or confirmed wear.

Every membership run retains these local files outside the public bundle:

- `test-index.jsonl`: all complete native test-context outputs, including rejected
  ones, with source ID/hash, context bounds, feature-end sample, target, validity,
  exclusion reasons, and an eligible-output ordinal.
- `test-predictions.npz`: evaluated logits and targets in eligible-output order.
  Logits preserve evaluation precision so saved argmax decisions reproduce metrics.
- `test-index-summary.json`: counts, omitted tails, policy/source fingerprints, and
  index/prediction hashes.

Index creation checks every eligible target against the evaluated target sequence,
not only the total length. The index also rejects a different frozen context length.
Its coordinates provide a basis for later common-index comparisons; this PR does
not implement intersection with historical model outputs. Test prediction collection
currently retains batches in memory before writing NPZ; index writing streams by
subject. This is an offline recipe, not a causal streaming detector.

Only aggregate provenance, scoring hashes/counts, target semantics, fitted state,
and model/metric artifacts enter the export bundle. The model card describes the
derived target. Exact subject identifiers, source arrays, indices, and predictions
stay local. No Hugging Face upload is performed by training/export.

## Validation checkpoint

The real adapter reproduced all 277 subjects and the frozen retained-context counts:
34,474 training, 8,173 validation, and 8,582 test. A two-subject local index probe
recorded 48,960 native output frames and 8,160 eligible outputs, preserving rejected
contexts. These are source/index checks, not benchmark results.

Synthetic end-to-end coverage includes derived labels despite deliberately malformed
historical labels, train-only fitted state, Keras/TFLite parity, exact scored targets,
metric reproduction from saved predictions, unlabeled inference, and artifact privacy.
The legacy training/inference tests remain supported.

The [first declared experiment](detection-first-experiment.md) is complete, with
pooled and per-series membership metrics and sampled real-input Keras/TFLite parity.
Its read-only report can be reproduced from saved scoring evidence. Next: inspect
errors and implement a common scoring intersection for descriptive historical
comparison. Historical training exposure, task acceptance thresholds, artifact
licensing, quantization, and hardware validation remain separate questions.

# Detection golden baseline

`experiments/detection-golden.json` defines a public reconstruction of the first
annotated-period membership run. It records the exact recipe configuration,
dataset evidence fingerprints, target and preprocessing contracts, sampling
policy, and the metrics observed in the historical run. It does not contain
recordings, subject identifiers, predictions, model files, or local paths.

The historical run used sleepKIT commit
`5c0e5ea8f4693d29972dcbcd1400c025a7b492a0`, Python 3.12.5, TensorFlow 2.21.0,
Keras 3.15.1, NumPy 2.1.3, h5py 3.13.0, and LiteRT 2.2.0. Its model used the
default merged `build_model`, context 240, five epochs, batch size 32, learning
rate 0.001, and seed 0. The final epoch was evaluated once; no early stopping or
test-driven selection was used.

This is a reconstruction definition, not a universal runner or a release
manifest. The ordinary Python entry point validates the supplied local evidence
against the pinned aggregate hashes and then calls `run_membership`:

```sh
python experiments/run_detection_golden.py \
  --data /path/to/original-cmidss \
  --events /path/to/train_events.csv \
  --alignment /path/to/source-event-alignment.json \
  --coverage /path/to/candidate-coverage.json \
  --frozen-split /path/to/frozen-split \
  --output /path/to/new-golden-run \
  --cache /path/to/local-feature-cache
```

The source adapter validates the event, clock, candidate-label, split, and HDF5
evidence before training. The script additionally checks the public definition's
target, context policy, partition sizes, source inventory fingerprint, and all
aggregate evidence hashes. Before training it also checks the exact v3 preprocessing
SPEC fingerprint and the reviewed preprocessing and model module hashes. The
output must be a new directory. Local run output contains the subject-level and
prediction evidence needed for review and should remain outside the repository;
the script adds `golden-definition.json` and `golden-runner.json` so the public
definition and runner hash remain attached to that run.

Performance measurements are a separate concern; use the bounded ordinary-Python
profiler at `python -m sleepkit.recipes.detection.profile` with the same local
dataset evidence; see the [profiling guide](detection-profiling.md). Its results
are not part of this golden quality reference.

The target is `sleepkit.annotated_period_membership/v1`: class 0 is
`OUTSIDE_SUPPORTED_ANNOTATED_PERIOD`, class 1 is
`INSIDE_ANNOTATED_PERIOD`, and `-1` is unknown. It represents derived membership
in supported annotated nightly periods. It does not assert clinical sleep or
wake, confirmed wear, or the competition's event average precision. The source
candidate policy uses complete clock-aligned onset/wakeup pairs and bounded wake
periods between eligible consecutive nights; conflicts, unlocatable evidence, and
unknown periods remain excluded.

Preprocessing uses TS, ENMO, and ZANGLE at 0.2 Hz. A feature window has 12 samples,
advances six samples, and ends at its last source sample. The five features are
`mean(cos(2*pi*TS/86400))`, ENMO mean and population standard deviation, and ZANGLE
mean and population standard deviation. Normalization is fitted once from every
valid feature frame in the 193 training series, including unknown-target periods
and incomplete context tails. Each feature uses the global training mean and
`sqrt(global population variance + 1e-6)`. Complete model contexts contain 240
consecutive native feature frames, are nonoverlapping, and are dropped when any
feature is invalid or any target is unknown. Removed contexts are never stitched;
incomplete tails are dropped.

Training uses all 34,474 eligible native training contexts exactly once per epoch,
with no balancing or replacement, shuffled with the declared seed and a bounded
buffer of 256. Batches retain the final partial batch. Validation has 8,173 eligible
contexts and test has 8,582; both preserve declared subject order and native context
order without shuffling. Evaluation covers every eligible test context and 2,059,680
feature frames. The complete context spans 7,230 seconds and uses future features
relative to each target, so this is offline, noncausal inference.

The historical test result was 95.7657% accuracy, 0.955611 macro-F1, and 0.127084
cross-entropy. Its confusion matrix, using actual classes as rows and predicted
classes as columns, was:

|  | Outside | Inside |
| --- | ---: | ---: |
| Outside | 1,207,331 | 43,875 |
| Inside | 43,338 | 765,136 |

These are observed reference values from one recorded run. They are not
prospective acceptance thresholds or quality gates. Seeded execution is not a
bitwise reproducibility guarantee across environments, and a run from the current
checkout is expected to be a reconstruction rather than a byte-identical replay.
The [first experiment report](detection-first-experiment.md) and
[annotated target adapter](detection-annotated-dataset.md) provide the surrounding
evidence and interpretation. Exposure of a separate historical SD-2 comparator to
this cohort remains unresolved; that question does not qualify this newly trained
frozen-split baseline.

The vectorized preparation revision advances only the current preprocessing
implementation pin. The original module hashes remain under `historical_run`, and
the original definition remains available at commit
`042c1eef5608db3bb842f29c6f12b768663f9026`. The
[preparation equivalence report](detection-preparation-performance.md) binds the
revised implementation to exact scalar-reference comparisons. Historical quality
metrics remain observations from the original five-epoch run.

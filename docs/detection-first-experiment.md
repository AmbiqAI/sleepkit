# First frozen-cohort membership experiment

The first new detection recipe run completed on 2026-09-20. Its final Keras model
achieved **95.7657% accuracy and 0.955611 macro-F1** on the frozen test set.
This establishes a measured starting point for annotated nightly-period membership.
It does not establish clinical sleep/wake accuracy, official event AP, or improvement
over a historical model.

## Declared experiment

Training used the unmodified merged source at
`5c0e5ea8f4693d29972dcbcd1400c025a7b492a0` in a separate clean checkout.
Configuration, environment, evidence hashes, and parity selection were recorded
before training. The default 4,018-parameter convolutional model used context 240,
five epochs, batch size 32, learning rate 0.001, and seed 0. Evaluation used the
fixed final epoch; no early stopping or test-driven model selection was performed.

The frozen series assignment and retained contexts were:

| Partition | Assigned series | Series with eligible contexts | Eligible contexts |
| --- | ---: | ---: | ---: |
| Training | 193 | 186 | 34,474 |
| Validation | 41 | 40 | 8,173 |
| Test | 43 | 43 | 8,582 |

The split file SHA-256 is
`08c2ac221b3212b5589d29f623015994da8c6fd104708879b40bcb4db60306e7`.
The [annotated dataset adapter](detection-annotated-dataset.md) reconstructs targets
from verified events and clocks; it ignores historical `sleep_stages` arrays.
Preprocessing v3 uses five features, 60-second windows, and a 30-second stride.
Normalization fits all valid frames from the 193 training series, including
unknown-target periods and incomplete context tails. Model contexts exclude any
unknown target or invalid sensor window, without joining across exclusions.
Each complete context spans 7,230 seconds; this is offline, noncausal inference.

The run used Python 3.12.5, TensorFlow 2.21.0, Keras 3.15.1, NumPy 2.1.3,
h5py 3.13.0, and LiteRT 2.2.0 on CPU. TensorFlow used eight intra-op threads,
one inter-op thread, and deterministic operations. Seeded execution is not a
guarantee of bitwise reproducibility across environments.

## Held-out results

| Metric | Result |
| --- | ---: |
| Evaluated feature frames | 2,059,680 |
| Pooled accuracy | 95.7657% |
| Pooled macro-F1 | 0.955611 |
| Cross-entropy | 0.127084 |
| Outside-period recall | 96.4934% |
| Inside-period recall | 94.6395% |
| Unweighted eligible-series mean accuracy | 95.8477% |
| Unweighted eligible-series mean macro-F1 | 0.933468 |

Confusion matrix, with actual classes as rows and predicted classes as columns:

| Actual / predicted | Outside supported annotated period | Inside annotated period |
| --- | ---: | ---: |
| Outside supported annotated period | 1,207,331 | 43,875 |
| Inside annotated period | 43,338 | 765,136 |

Per-series accuracy ranged from 86.8293% to 100%, with median 96.2899%.
Eligible frame counts ranged from 480 to 98,640 per series. The declared macro-F1
always averages both classes with zero-division set to zero. Two test series have
only inside-class targets, so even a perfect prediction on such a series has
macro-F1 0.5. The lower unweighted mean must be interpreted with this convention.

Of 14,433 complete native test contexts, 8,582 were eligible and 5,851 were excluded
for unknown targets. None were excluded for nonfinite sensors. The native index
contains 3,463,920 output frames, including excluded ones. Another 6,227 feature
frames were dropped as incomplete context tails. These metrics describe eligible
annotated coverage; they do not characterize the excluded periods. Adjacent frames
are correlated and should not be treated as independent observations for confidence
intervals.

## Export and runtime check

The run exported a 95,616-byte Keras file and a 23,212-byte floating-point TFLite
file. Feature preparation took 75.7 seconds; preparation, training, evaluation,
and export together took 229.5 seconds on this machine. These are observations
from one run, not a throughput or device-latency benchmark.

The predeclared parity check selected the first and last eligible context from
each test series: 86 contexts and 20,640 frames. TFLite logits matched saved Keras
logits within `atol=1e-5, rtol=1e-4`; maximum absolute difference was
`1.430511474609375e-6`, with zero argmax disagreements. The check bound the model,
predictions, split, and source evidence before comparing. It does not establish
parity for every test context or target hardware.

An independent review recomputed pooled and per-series metrics directly from the
saved arrays, checked the declaration and coverage, and independently verified all
parity context selections. No material finding remained.

## Reproduce the local report

For a completed membership run, the read-only evaluation helper needs NumPy and
the local scoring evidence, but does not load TensorFlow, source recordings, or
the model:

```sh
python -m sleepkit.recipes.detection.evaluation \
  --run /path/to/run \
  --output /path/to/new-evaluation.json
```

It validates bundle and evidence hashes, split/source bindings, native coordinates,
eligibility, targets, coverage, and reproduction of recorded pooled metrics. It
writes per-series results outside the bundle, refuses an existing output, and
prints only aggregate results to stdout. Tail counts are checked against recorded
metadata; they cannot be reconstructed from the complete-context index alone.
The index streams one context at a time; saved prediction arrays remain in memory.
Hash checks bind local evidence, rather than attesting that source data or model
execution were independently replayed.

The internal experiment directory is
`sleepkit-evaluation-evidence/experiments/membership-tcn-seed0-20260920/` alongside
the repository. It retains `declaration.json`, the exact `run_experiment.py`,
`environment.txt`, logs, `run/`, `evaluation.json`, and the parity script/report.
An initial environment-capture attempt failed because the environment lacked pip,
before preprocessing or training; its log is retained. The successful launcher
used `uv pip freeze` with the same experiment configuration.
Recordings, subject identifiers, prediction arrays, and model binaries are not
committed with this report.

## Next evidence gates

Inspect errors by series, coverage, and proximity to annotated boundaries before
choosing the next experiment. Any changes informed by this test report must be
declared; the current test set is no longer untouched for further model selection.
Build a common scoring intersection for descriptive historical-model comparison,
while documenting unresolved historical training exposure. Hugging Face packaging
is available, but no upload was performed: artifact licensing and publication
metadata still need resolution. Quantization and target-hardware validation remain
separate deployment steps.

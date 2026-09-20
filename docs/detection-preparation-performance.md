# Bounded vectorized feature preparation

sleepKIT's first stage profile spent most of its time preparing a cold feature
cache. The original extractor made five separate mean/standard-deviation calls,
plus a cosine calculation, for each twelve-sample window. The revised extractor
runs the same float64 arithmetic over at most 4,096 windows at a time, then stores
float32 features. It adds no worker processes, dependency or recipe parameter.

A strided view represents overlapping windows without copying the recording into
a full window tensor. Validity checks and calculation temporaries are bounded by
the chunk size. Source arrays, feature outputs and timestamps still scale with
recording length; this is not a streaming reader. Invalid windows remain zeroed
and are skipped before trigonometric/statistical arithmetic. Feature ends, native
context boundaries, unknown-target exclusions and incomplete tails are unchanged.

`SPEC` and feature-cache keys remain unchanged because the tested outputs match
exactly. The golden definition explicitly advances its implementation hash while
retaining the historical hashes and quality reference. Existing fitted-state and
cache formats stay compatible. A future change that alters feature values requires
a separate contract/version decision.

The parity tests cover short inputs, exact and partial chunk boundaries, midnight
and independently verified UTC clocks, nonfinite windows, all-invalid chunks,
signed zeros, subnormals, cancellation, extreme finite sensors and C/Fortran/
positive- and negative-stride arrays. They compare dtype, shape and bytes, then
check fitted normalization and native context labels/timing. They also load cache
entries written by the scalar implementation.

The independent corpus comparison loads the scalar implementation from immutable
commit `042c1eef5608db3bb842f29c6f12b768663f9026` and verifies its source SHA-256
before execution. It calls both extractors directly on every training series,
without feature-cache reads. Only training sensors are used; labels and held-out
features are not read. The feature digest includes dtype/shape/data for every
series in frozen training order. The declaration also binds dataset, code, runner,
specification, Python and NumPy versions. Source and code integrity are checked
again before a successful report is written.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
python experiments/compare_detection_preparation.py \
  --data "$DATA" --events "$EVENTS" --alignment "$ALIGNMENT" \
  --coverage "$COVERAGE" --frozen "$FROZEN" --output "$NEW_COMPARISON_DIRECTORY"
```

The pinned commit must be present in the local Git clone; a shallow clone may need
its history fetched first. The command runs both implementations over the entire
training partition, so it deliberately includes the scalar implementation's cost.
It additionally repeats extraction on the first three training series three times,
alternating implementation order. Source reads, equality checks and normalization
are excluded from extraction timings. These are development-machine measurements,
not a controlled cross-backend or hardware performance certification.

## Recorded equivalence and extraction timing

The [full training comparison](evidence/membership-vectorized-equivalence-20260920/report.json)
passed for all **14,857,307 feature frames from 193 training series**. Feature
values, validity masks and endpoints matched byte for byte. The fitted normalizer
fingerprint also equals the original CPU profile's fingerprint. Measurements used
Python 3.12.5 / NumPy 2.1.3 on the same Intel Core i9-13900K development machine.

Summed extraction time across those series was 387.41 s for the scalar reference
and 4.03 s for the vectorized implementation. Three alternating-order measurements
of the fixed three-series subset produced these extraction-only totals:

| Repetition | Scalar | Vectorized |
| --- | ---: | ---: |
| 1 | 6.653 s | 0.0704 s |
| 2 | 6.509 s | 0.0687 s |
| 3 | 6.508 s | 0.0676 s |

The ratio of median subset times is about 95× for extraction on this machine.
That ratio excludes source reading, integrity checks, cache compression/writes,
normalization, label construction and training. It is not an end-to-end speedup.
The two public evidence files bind the exact implementations, inputs and protocol;
no recordings, features or subject identifiers are included.

## Full recipe stage profile

A [fresh-cache profile of the revised recipe](evidence/membership-vectorized-profile-20260920/report.json)
used the same frozen inputs and thread settings as the first profile. Its counts,
fitted-normalizer fingerprint and resident input/target fingerprints match the
baseline exactly: 34,474 training contexts in 1,078 batches, with ten contexts in
the final batch.

| Stage | Original observation | Vectorized observation |
| --- | ---: | ---: |
| Serial empty-feature-cache preparation | 417.35 s | 18.57 s |
| Warm normalization fit | 6.72 s | 6.60 s |
| Eligible-context counting | 8.09 s | 7.94 s |
| Warm loader epoch | 14.51 s | 14.12 s |
| Fresh-model training epoch, including compilation | 16.13 s | 15.90 s |

The observed cold-preparation ratio is about 22×, smaller than the extraction-only
ratio because preparation also reads, validates, compresses and writes data.
These are separate single development-machine runs; the repeated extraction
measurements above provide narrower evidence. The warm stages remain broadly
unchanged. This work makes no claim of faster model training or a TensorFlow vs
PyTorch advantage, and does not rerun the five-epoch quality experiment.

The next performance investigation is the warm data path. Integrity validation,
label construction, cache decompression and Python iteration remain included;
those costs need separate measurement before changing the loader or concurrency.

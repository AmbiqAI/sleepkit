# Profiling the detection recipe

The profiler measures the existing sleepKIT TensorFlow membership recipe before
we change its loader. It calls the same feature preparation, normalization,
finite dataset and model compilation functions as training. It creates disposable
models and reads **training features and labels only**; it does not evaluate
validation/test predictions, choose a checkpoint, export a model or publish to HF.
Whole-cohort integrity checks still hash the frozen source files.

Run from a fresh process with the detection dependencies installed. Use the
frozen dataset inputs described in [the golden experiment](detection-golden.md):

```sh
TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=2 \
python -m sleepkit.recipes.detection.profile \
  --data "$DATA" --events "$EVENTS" \
  --alignment "$ALIGNMENT" --coverage "$COVERAGE" --frozen "$FROZEN" \
  --output "$NEW_PROFILE_DIRECTORY"
```

The default batch size is 32, seed 0, with two resident warmup steps and 20 measured
resident steps. Context length comes from the frozen protocol. The profiler runs
one complete training epoch regardless of the training recipe's `Config.epochs`;
`profile_epochs` records this difference explicitly. The Python entry point
`profile_membership(source, output, cfg, ...)` also accepts an `AnnotatedDataset`,
so a synthetic fixture can exercise the same flow without downloading data.

The output directory must not exist. A declaration is written before measurements,
followed by environment evidence. A successful run produces `report.json` bound
to both files by SHA-256; failure leaves no successful report. Source evidence,
recordings, the declaration, environment file and recipe code are checked again
before completion. Run from a clean checkout and preserve its commit alongside
the report when comparing branches; the report also binds every detection Python
source file by content hash.

**Keep the entire output directory private.** `private-feature-cache/` contains
dataset-derived feature arrays. The declaration, environment and report contain
aggregate counts, hashes and environment information without subject IDs or
input paths; review those files before sharing. They are benchmark evidence, not
a distributable model bundle.

| Stage | What its wall time includes |
| --- | --- |
| `verify_before` / `verify_after` | Whole-cohort source/evidence integrity checks |
| `cold_feature_cache_prepare` | Serial training-subject reads, integrity checks, extraction and writes into a new feature cache |
| `warm_fit_normalizer` | Training-subject reads/cache loads and fitting all valid feature frames, including unknown-target regions and incomplete context tails |
| `count_eligible_contexts` | Native context construction, reader integrity checks and target reconstruction; the reader independently recounts native-window validity and coverage |
| `warm_loader_epoch` | Dataset construction, one complete shuffled finite pass, host materialization and cardinality checks; no model execution |
| `resident_model_build` / `end_to_end_model_build` | Seed reset, default model construction and compilation configuration |
| `resident_compile_and_warmup` | Actual training-function tracing/compilation and warmup on one host batch |
| `resident_model_steps` | Repeated `train_on_batch` on that one host batch, including transfers and synchronized returned logs |
| `end_to_end_fit_epoch` | Fresh-model `fit` over one finite training epoch, including tracing/compilation and loading; excludes model construction, validation and export |

“Cold” refers to the feature cache; no attempt is made to flush operating-system
caches. Framework imports are timed separately; they exclude later device/runtime
initialization and may already be warm in an existing Python process. The fresh
model and optimizer used for `fit` run after resident process/device warmup. Physical devices are recorded, but this
is not an operation placement trace or a GPU-memory profiler.

The warm loader reports the first, median and p95 observed wait for `next()` plus
host materialization. Prefetch overlaps work, so those values are not raw disk
latency and do not isolate loading stalls during `fit`. Resident steps are a
compute-oriented reference with host transfer overhead, not device-only timings.
Their repeated-batch schedule must never be substituted for a training epoch.

Each stage records process CPU time and process lifetime peak RSS. Peak RSS is
not current memory or an allocation attributed to that stage, and it excludes
device memory. No claim of worker scaling or TensorFlow/PyTorch speed difference
follows from a single report.

For comparable runs, preserve the declared source, split, preprocessing,
normalizer and resident-batch fingerprints; batch/context sizes; finite context
counts and partial batch; model; runtime versions; hardware; and thread settings.
The training schedule retains every eligible context once, with no replacement,
class balancing or sample weights. Shuffling uses the existing bounded buffer of
256, rather than promising a uniform global permutation. Validation/test ordering
is unchanged by this harness.

Repeat measurements in fresh processes on an otherwise idle machine before
claiming a speedup. Do not run benchmarks concurrently with tests or other
benchmarks. Report serial preparation separately from the original experiment's
eight-process preparation: this harness does not reconstruct that wall time.
Keep prediction-quality reproduction separate from performance experiments.

## First CPU observation

The [2026-09-20 evidence](evidence/membership-tcn-seed0-cpu-20260920/report.json)
contains the declaration, environment and report for the frozen membership training
partition: 193 series, 14,857,307 valid normalization frames, 34,474 eligible
contexts, 1,078 batches, and ten contexts in the final batch. No held-out feature
or label reads were used. This was a single development-machine observation on
an Intel Core i9-13900K, TensorFlow 2.21.0 / Keras 3.15.1, with TF intra-op 2,
inter-op 1, OMP 2, OpenBLAS 1 and GPU devices disabled. Machine load was not
controlled; the repository's training tests ran before and after this measurement.

| Stage | Observed wall time |
| --- | ---: |
| Serial empty-feature-cache preparation | 417.35 s |
| Warm normalization fit | 6.72 s |
| Eligible-context counting | 8.09 s |
| Warm loader, complete training epoch | 14.51 s |
| Resident batch, 20 measured steps after two warmups | 0.056 s |
| Fresh-model training epoch, including compilation | 16.13 s |

Warm-loader median/p95 consumer waits were 4.34/65.08 ms. Resident-step median was
2.67 ms, measured through `train_on_batch`; the separate `fit` epoch processed
about 2,137 contexts/s. Lifetime host peak RSS was 1.37 GiB. The short resident
sample is an initial diagnostic, not a stable microbenchmark or a valid basis for
subtracting compute time from the `fit` duration.

Cold preparation dominates this serial run. Warm loader duration is also close
to the separately measured fit epoch, making preparation and loading useful next
investigations. These timings do not isolate their individual sub-operations and
do not establish a TensorFlow backend bottleneck. Candidate optimizations must
preserve the recorded feature/normalizer and sample contracts and be compared in
repeated, controlled runs before any improvement claim.

The next measured change is [bounded vectorized feature preparation](detection-preparation-performance.md), with exact scalar-reference comparisons before reusing the existing feature contract.

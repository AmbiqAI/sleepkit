# Detection diagnostics and historical comparison

The [first experiment](detection-first-experiment.md) now has read-only error
diagnostics and a separate, explicit SD-2 comparison recipe. These steps consume
the same saved run evidence. They do not retrain a model or tune a threshold.

## Error diagnostics

```sh
python -m sleepkit.recipes.detection.error_analysis \
  --run /path/to/membership-run \
  --output /path/to/new-error-analysis.json
```

The evaluator first verifies the run, then reports confusion, class errors, and
coverage by series, position within a model context, and distance to an observed
target transition. Only aggregate results print to stdout; series identifiers
remain in the local report outside the artifact bundle.

An observed transition is a change between known 0/1 targets on adjacent native
feature frames. Its position is the later frame's endpoint, not the original raw
event timestamp. Unknown-target frames break segments. A nearest transition must
be in the same known segment and series. Excluded complete contexts still provide
target observations; incomplete tails do not. The distance bins are disjoint.

For the first experiment, the verified index contains 1,370 observed transitions:

| Distance to observed transition | Eligible frames | Errors | Accuracy |
| --- | ---: | ---: | ---: |
| At most 5 minutes | 28,770 | 8,713 | 69.7150% |
| Over 5 through 30 minutes | 137,000 | 9,738 | 92.8920% |
| Over 30 minutes | 1,870,630 | 67,706 | 96.3806% |
| No transition in the known segment | 23,280 | 1,056 | 95.4639% |

The near-transition error rate is higher, but most errors occur in the much larger
over-30-minute group. These are descriptive observations, not evidence of an error
cause or a selected boundary threshold. The no-transition group contains only
inside-period targets in this run.

Context-position diagnostics use disjoint first-eight, interior, and last-eight
frame bins. Their respective accuracies were 94.6035%, 95.8320%, and 95.0725%, on
68,656, 1,922,368, and 68,656 frames. Class mixtures and temporal correlation limit
interpretation; a future experiment should investigate hypotheses on validation
data rather than select changes from these test metrics.

## Historical adapter

`historical.py` supports the pinned SD-2-TCN-SM int8 release. It reads existing
historical features and masks without reading their legacy labels, reconstructs
every feature from verified raw sensors, and requires exact float32 equality
(with matching NaN positions). It preserves:

- The original `cos(2*pi*nanmean(TS/86400))` arithmetic, including midnight behavior.
- The legacy duration and additional feature-tail truncation.
- Whole-record float32 normalization, including unknown-target periods and feature
  tails that do not enter model contexts. This is an offline transductive policy.
- The int8 graph, I/O scale and zero point, and fixed batch-one contract.

The stored mask must be all ones, matching this feature generator. Residual
nonfinite features exclude their entire model context. Input quantization rounds
to nearest with ties to even, clips to the int8 range, and records clipping counts.
Outputs are used directly for argmax; ties select index zero and are counted. No
additional softmax, cross-entropy, or calibration comparison is applied to the
historical graph with its two final softmax operators.

Matching stored features to this source implementation establishes reproducible
coordinates for this comparison. It does not recover the historical training
inventory or prove that today's feature store was used to train the released model.
Local logs/configurations contain inconsistent seeds and no explicit subject
inventory. The historical loader partitions filesystem discovery order, so replaying
one seed against today's files cannot resolve exposure. The historical configuration's
top-level sampling-rate discrepancy also remains unsuitable as a deployment clock.
This comparison derives timestamps from the verified source grid instead.

## First descriptive comparison

On 2026-09-20, source revision `8d83b23` verified all 3,470,018 historical feature
rows across the 43 frozen test series against raw sensors with exact float32
equality. Both pipelines had 14,433 complete native contexts, of which 8,582 were
eligible and common. All 5,851 exclusions were due to unknown targets; none were
due to nonfinite features. Historical truncation left 6,098 unused feature-tail
rows, versus 6,227 for the new pipeline, but removed no complete context here.

Both models were scored on exactly 2,059,680 common frames, with class support
1,251,206 outside-period and 808,474 inside-period targets:

| Metric on common frames | Historical SD-2 int8 | New membership Keras |
| --- | ---: | ---: |
| Accuracy | 92.2334% | 95.7657% |
| Macro-F1 | 0.916214 | 0.955611 |
| Outside-period recall | 98.1601% | 96.4934% |
| Inside-period recall | 83.0611% | 94.6395% |
| Unweighted series mean accuracy | 90.0772% | 95.8477% |
| Unweighted series mean macro-F1 | 0.865632 | 0.933468 |

The historical confusion matrix, with actual rows and predicted columns in
outside/inside order, was `[[1228185, 23021], [136947, 671527]]`. The new matrix
remained `[[1207331, 43875], [43338, 765136]]`. Both native eligible denominators
equaled the intersection in this run. All 43 series contributed; the same two
single-class series and fixed two-class F1 convention described in the first
experiment apply to both models.

Historical inference clipped 926 of 10,298,400 input values and produced 4,386
argmax ties; ties chose class zero. The preserved historical model is 39,632 bytes.
Verification, inference, and report generation took 35.5 seconds locally, not a
hardware inference benchmark. Full local evidence is retained in
`sleepkit-evaluation-evidence/experiments/sd2-comparison-20260920/` alongside the
repository. The pinned release files are retained under `historical/sd2-v1/` in
the same evidence root.

The new pipeline had higher agreement with this derived target in the observed
comparison. Differences in annotation target, historical exposure, preprocessing,
normalization, training, and quantization prevent an isolated model-quality claim.
No model was selected or retrained from these results, and no Hub upload occurred.

## Compose the comparison in Python

```python
from sleepkit.recipes.detection.comparison import compare
from sleepkit.recipes.detection.target_dataset import AnnotatedDataset

source = AnnotatedDataset(
    root="/path/to/cmidss",
    events_path="/path/to/train_events.csv",
    alignment_path="/path/to/source-event-alignment.json",
    coverage_path="/path/to/candidate-coverage.json",
    frozen_dir="/path/to/frozen-split",
)
report = compare(
    run_path="/path/to/membership-run",
    source=source,
    feature_root="/path/to/fs-w-a-5-60/cmidss",
    baseline_source="/path/to/pinned-sd2-source",
    output_path="/path/to/new-comparison",
)
```

The comparison verifies the new run and source bindings, writes a declaration
before historical inference, and preserves each pipeline's native complete contexts.
The common index requires matching subject, raw-source hash, context start/end,
every feature endpoint, and target, with both contexts eligible. No context is
shifted or stitched to enlarge the intersection.

The historical `WAKE/SLEEP` argmax is explicitly mapped to outside/inside supported
annotated periods as a descriptive proxy. `historical_native` means historical
contexts scored under this candidate-target eligibility policy, not reproduction
of the release's original metrics. Reports retain both native denominators and
the common denominator, pooled and per-series results, class support, exclusions,
ties, and input clipping. This compares complete pipelines, including training,
preprocessing, normalization, and quantization differences. It cannot establish
historical held-out performance, clinical accuracy, or architecture superiority.

Local evidence includes `declaration.json`, `new-evaluation.json`,
`historical-index.jsonl`, `common-index.jsonl`, `historical-predictions.npz`, and
`comparison.json`. Index rows represent complete contexts and explicitly record
their frame grid. Saved historical outputs are int8; common start offsets address
both flattened prediction arrays. These files contain identifiers and remain
outside public bundles. Existing output directories are refused. Failed runs
retain partial evidence without a completed comparison report.

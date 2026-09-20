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

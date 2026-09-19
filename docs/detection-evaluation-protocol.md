# Detection evaluation checkpoint

The pipeline and artifact contracts are implemented. The next question is whether
a comparison is valid, before asking which model scores better. This checkpoint
adds an annotation audit and defines the evidence needed for an evaluation run.
It does not change source labels, select a split, or establish benchmark results.

## Run the annotation audit

From the checkout, in an environment with NumPy and h5py:

```sh
python -m sleepkit.recipes.detection.audit \
  --data /path/to/cmidss \
  --events /path/to/cmidss/train_events.csv \
  --output /path/to/new-local-audit.json
```

The output includes local subject identifiers and source-file hashes. Keep it with
local experiment provenance, outside any public model bundle. Terminal output is
aggregate-only. Existing output files are not overwritten; source files are opened
read-only. No TensorFlow, training, or network access is needed.

The audit groups events by series and night rather than pairing separate onset and
wakeup lists by position. A candidate interval requires exactly one event of each
type, present integer steps and timezone-aware timestamps, bounds inside the HDF5
recording, and matching timestamp/step duration. It excludes every conflicting
member of an overlapping interval cluster. Sleep candidates use half-open
`[onset, wakeup)` intervals. Missing, duplicate, reversed, out-of-bounds, and
inconsistent nights remain visible in the report.

Source-step to HDF5-index alignment is an explicit assumption in this audit. Verify
it against the raw series before deriving benchmark labels. Agreement with an HDF5
file created by the same conversion assumption is not independent verification.
The audit does not establish wake labels from absence of a sleep pair.

## Initial local audit

On 2026-09-19 the local snapshot produced these counts:

| Observation | Count |
| --- | ---: |
| HDF5 subjects and event subjects | 277 each |
| Event rows | 14,508 |
| Rows missing event steps | 4,923 |
| Complete paired sleep candidates | 4,790 nights |
| Nights missing a step or timestamp | 2,464 |
| Total HDF5 samples | 127,946,340 |
| Samples inside paired sleep candidates | 29,838,204 |
| HDF5 wake/unknown labels inside those candidates | 0 |
| HDF5 sleep labels outside those candidates | 0 |
| HDF5 wake labels outside those candidates | 98,108,136 |
| Subjects with a non-five-second local-clock transition | 44 |
| Local-clock forward-hour transitions (3,605 seconds) | 17 |
| Local-clock backward-hour transitions (82,805 seconds modulo day) | 27 |

Events SHA-256: `e785115b0772b2953057fa333cc6990c4d9879a6a5321dc6e6cf8b13ef68bf58`.
The local JSON report binds the HDF5 files individually; these counts are not a
portable declaration about every CMIDSS distribution.

The zero disagreements show consistency with the legacy conversion. They do not
prove annotation completeness. The 98,108,136 outside-pair samples include wake
and potentially unobserved intervals; this audit cannot partition those categories.
In particular, missing nights must not silently become confirmed wake or be
entirely relabeled unknown without a defined annotation policy.

All source TS values are finite and in range. The 44 clock transitions may be
local timezone/daylight-saving changes, not missing samples. The current detector's
strict TS cadence check rejects these subjects. Verify raw sample steps and absolute
timestamps to distinguish a clock change from a real gap before revising the reader.
Do not silently remove these subjects and describe the resulting cohort as complete.

## Evidence required before comparison

1. **Source and identity mapping.** Verify raw steps, timestamps, and channel rows
   against HDF5. Document whether a series is one person or one recording; obtain a
   person grouping where available, or explicitly limit claims to series-disjoint
   evaluation. Do not invent a person mapping from filenames.
2. **Annotation policy.** Define supported sleep and wake intervals from raw events,
   handling missing nights, recording edges, ambiguous event ordering, and gaps.
   Keep unknown labels separate. A conservative policy must still provide supported
   examples of both classes; evaluating only known sleep is not a binary benchmark.
   Audit the policy's effect on class counts and excluded durations before training.
3. **Frozen split.** Select subjects/groups without consulting model scores; keep
   all recordings for a known person together. Save the exact split, group mapping,
   source hashes, seed, and rationale locally. Use the same test subjects for every
   eligible model. Retain validation for selection and test for the final comparison.
4. **Historical exposure.** Recover SD-2's training and validation inventory, or
   obtain a cohort independently known not to have been used in training. Without
   that evidence, SD-2 can be a descriptive reference but not a verified held-out
   benchmark. A newly generated split does not establish non-overlap with old training.
5. **Common scoring index.** Persist subject/source hash, context start, feature-end
   sample, target label, and validity for every scored output. Intersect supported
   outputs before scoring; report exclusions and class counts for each model as well
   as the intersection. Identical timestamps alone do not imply identical context.
6. **Native model contracts.** Keep historical weights with their own feature and
   normalization policy; record its whole-record test-subject statistics as an offline
   transductive policy. Keep new weights with their fitted training-only state.
   A comparison of these complete pipelines is not an isolated architecture comparison.
   Inspect historical output semantics before interpreting loss or calibration.

These are evaluation acceptance criteria, not fields in a universal orchestration
config. Implement the evaluator as ordinary Python composition once the policy is
specified and tested. The current audit only measures source-label consistency.

## Separate questions and experiments

- **Pipeline comparison:** compare each trained model with its supported preprocessing
  on the common scoring index. Report native policies and their limitations explicitly.
- **Preprocessing ablation:** train matched new models separately with v1 and v2
  features, identical splits/configs/seeds, and each model's own fitted normalization.
  Do not feed v1 features to v2-trained weights and call that a valid ablation.
- **Architecture comparison:** retrain architectures under a common preparation and
  training protocol. Do not substitute preprocessing beneath historical weights.
- **Label sensitivity:** evaluate declared annotation policies on matched indices;
  report differing denominators instead of attributing a mask change to model quality.
- **Deployment comparison:** establish float32 quality first, then calibrate int8
  using training inputs and measure changes on the frozen test outputs. Keep calibration
  and tuning away from test data.

Report confusion matrices, class recall, macro F1, pooled and per-subject results,
evaluated/excluded counts, and variation across subjects. Compare probability,
calibration, loss, or threshold metrics only once output transforms are verified;
the historical TFLite graph contains two final softmax operators. Never re-use its
published aggregate score as though it was computed under the new protocol.

## Review checkpoints

- **Light review:** small local helpers, report formatting, and documentation;
  check the diff and focused tests.
- **Independent review:** source alignment, label construction, split selection,
  preprocessing state, and evaluation implementation; reviewer verifies evidence
  without assuming the implementation's conclusions.
- **Adversarial review:** before model-quality claims, quantized deployment claims,
  or publication; actively look for train/test overlap, denominator changes,
  unsupported output semantics, and misleading provenance.

Reproduce substantive findings, add a regression check where appropriate, fix the
owning PR, and update any stacked PRs. Avoid repeatedly reviewing unchanged code
or treating review approval as a substitute for source evidence.

Next implementation checkpoint: raw-series alignment and a reviewed annotation
policy. Historical model exposure and model licensing remain unresolved. No model
publication is authorized by producing an audit report.

# Candidate event-supported label policy

Status: implemented candidate policy `sleepkit.event_candidates/v1`, separate from
the active training label policy. No existing HDF5 labels or historical model inputs are rewritten here.
The policy requires verified raw-step/HDF5 alignment before materialization.

## Supported intervals

For each source series, validate events grouped by integer night ID. A complete
night has exactly one onset and one wakeup, finite in-range steps, timezone-aware
timestamps consistent with the source clock, and onset before wakeup. Exclude
all members of conflicting/overlapping groups; do not repair ambiguous pairing
by sorting two event lists independently.

- **Sleep candidate:** `[onset_n, wakeup_n)` for a validated complete night.
- **Wake candidate:** `[wakeup_n, onset_(n+1))` only when both nights are complete,
  their night IDs are consecutive, source order is consistent, and no missing or
  conflicting event group lies between them.
- **Unknown:** all other samples, including recording edges, incomplete nights,
  gaps in night numbering, ambiguous groups, and unsupported source samples.

The wake rule assumes the event sequence describes the task's sleep episodes
between the two annotated nights. It is not proof of continuous clinical wake or
absence of naps. Confirm this interpretation against the source annotation
protocol before calling the candidates ground truth. If that interpretation cannot
be supported, report the limitation or obtain stronger annotations; do not default
the interval to confirmed wake.

An onset boundary belongs to sleep; a wakeup boundary belongs to wake only when
the following interval is supported. There must be no overlap between the two
classes. Keep unknown as `-1` and never convert it to zero for metric convenience.

## Evaluation and feature windows

The current detector targets the last sample of each 60-second feature window.
Align labels at that exact source step. Preserve the full feature/context input
policy separately: a context touching invalid sensor data or unsupported targets
is excluded according to a declared shared scoring policy, and its reason is counted.
Do not stitch remaining frames together after exclusions. Compare model outputs
only on a frozen common index and report native eligibility alongside the intersection.

A sensitivity analysis may use stricter all-window label support, but that changes
the evaluation population and must be reported separately. Historical default-zero
labels remain historical evidence, not the reference mask for this candidate policy.

## Evidence to review before adoption

1. Source alignment confirms every step and UTC sample interval; timezone changes
   affect local time-of-day features but must not be interpreted as physical gaps.
2. Annotation provenance supports the wake-interval interpretation above, with
   missing/ambiguous nights and recording edges counted explicitly.
3. Per-series counts show supported sleep, supported wake, unknown, and the exact
   number of model contexts retained. Both classes must remain represented.
4. A subject/person grouping and split are frozen without consulting model scores.
5. Pure policy tests cover shuffled/duplicate/missing events, skipped night IDs,
   conflicting intervals, boundaries, and daylight-saving transitions.
6. Independent and adversarial reviewers verify the masks before training or
   comparing scores. Materialization must write a new versioned dataset and bind
   raw/events hashes; source HDF5 files remain unchanged.

The first adversarial review identified two implementation gates: compare every
available event timestamp directly with raw UTC at its source step (pair durations
alone cannot detect an equal shift of both timestamps), and test a pure interval
builder that enforces consecutive integer night IDs and excludes incomplete or
conflicting intervening groups. The optional event-clock audit now implements the first check; see
[independent sample clocks](detection-sample-clock.md). The pure interval builder and read-only coverage audit are now implemented.
Annotation-semantics review remains pending before these candidates become
evaluation ground truth.

## Cadence handling boundary

The current HDF5 files store local seconds of day but no independent absolute
sample clock. A source alignment report can establish that a particular snapshot
is continuous in UTC even when local clocks jump. That finding does not justify
accepting arbitrary one-hour jumps from other inputs.

The [verified source adapter](detection-sample-clock.md) supplies an explicit UTC sample
clock alongside local TS. Validate cadence against that clock, use local TS only
for time-of-day features, and carry the clock contract through training and
unlabeled inference. Keep the existing strict check for inputs without sufficient
clock evidence. Do not add a global option that simply disables cadence checks.


## Pure builder and conflict rules

`candidate_labels.build_candidates(nights, sample_count, first_utc_seconds=...)`
accepts one series' CSV event groups and a source clock verified independently.
It returns an int8 label vector, sleep/wake intervals, eligible night IDs, and issue
counts. It performs no I/O and never reads historical labels. UTC equality is
rechecked at each available event; onset and wakeup must identify existing samples,
so a wakeup at `sample_count` is invalid.

Night IDs must be canonical positive integer strings. Ambiguous identifiers such
as `01`, malformed nonmissing step coordinates, malformed timestamps, timestamp-only
evidence, and contradictory step/UTC coordinates invalidate the entire series.
Normal missing step-and-timestamp rows do not invalidate unrelated valid nights.

Every located group participates in chronology checks, including incomplete groups,
duplicates, unexpected event types, and step-only evidence. Exclude all participants
in overlapping or inverted night order before constructing candidates. An incomplete
group whose coordinate bounds touch a complete interval also excludes that interval;
rejected groups can veto a wake interval. Counts of issues can overlap and are not a
partition of nights. No rejected or missing group is silently removed to join distant
nights. Shuffling rows and groups leaves the result unchanged.

## Read-only coverage audit

```sh
uv run python -m sleepkit.recipes.detection.candidate_audit \
  --data /path/to/original-cmidss --events /path/to/train_events.csv \
  --alignment /path/to/passed-event-clock-alignment.json \
  --context 240 --output /path/to/new-candidate-coverage.json
```

The report requires matching, passed raw alignment and event-clock evidence for
all original HDF5 files. It binds events, alignment, source hashes, policy version,
and context rules. Per-subject identifiers remain local. Sources are checked before
and after reading; an existing report is never overwritten. Historical label arrays
are not loaded or changed. Use original files, not the clocked copies with different
hashes, for this audit.

Context coverage uses the same 12-sample windows, stride of six, last-sample target,
and fixed nonoverlapping context groups as preprocessing. It counts nonfinite-sensor
and unknown-target exclusions separately and jointly, preserves source indices,
and drops incomplete tails. It does not run features through a model or select
subjects based on scores.

On the hash-bound local snapshot audited on 2026-09-19:

| Coverage | Samples / intervals |
| --- | ---: |
| Sleep candidates | 29,838,204 samples in 4,790 intervals |
| Wake candidates | 44,959,116 samples in 4,063 intervals |
| Unknown | 53,149,020 samples |
| Total | 127,946,340 samples across 277 subjects |

At 240 feature frames per context, 51,229 of 88,686 complete contexts remain
eligible. Unknown targets exclude 37,457 contexts; none contain nonfinite sensors.
Retained targets include 7,493,186 wake candidates and 4,801,774 sleep candidates.
An additional 39,473 feature windows lie in incomplete context tails. Missing event
values affect 2,464 nights. Both classes remain represented in this candidate
population, but this is not a model benchmark or evidence of label correctness.

## Annotation-semantics gate

The [official competition overview](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states)
defines an event-detection task evaluated by average precision across timestamp
tolerances. This does not by itself establish dense clinical sleep/wake ground truth.
The [retrieved annotation contract and frozen split](detection-frozen-split.md)
now support a derived annotated-period membership target, with explicit inside,
outside-supported-period, and unknown semantics. They do not support clinical
per-sample sleep/wake claims. The builder's historical candidate names remain
unchanged; future training artifacts must declare the adopted target explicitly.
The [annotated dataset adapter and scoring index](detection-annotated-dataset.md)
now implement this contract without changing original labels.

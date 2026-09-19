# Candidate event-supported label policy

Status: proposal for independent/adversarial review, not the active training label
policy. No existing HDF5 labels or historical model inputs are rewritten here.
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
conflicting intervening groups. Neither check is implemented by the source-alignment
verifier, which intentionally audits sensor samples only.

## Cadence handling boundary

The current HDF5 files store local seconds of day but no independent absolute
sample clock. A source alignment report can establish that a particular snapshot
is continuous in UTC even when local clocks jump. That finding does not justify
accepting arbitrary one-hour jumps from other inputs.

A future verified source adapter should supply an explicit UTC/elapsed sample
clock alongside local TS. Validate cadence against that clock, use local TS only
for time-of-day features, and carry the clock contract through training and
unlabeled inference. Keep the existing strict check for inputs without sufficient
clock evidence. Do not add a global option that simply disables cadence checks.

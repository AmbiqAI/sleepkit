# Frozen evaluation assignment and target

The reviewed audit/clock/candidate stack (#22–#25) is integrated into `main`.
This checkpoint defines the derived target and freezes a reproducible split. It
is assignment-only: running the existing recipe with `split.json` still reads
historical HDF5 labels. A target-aware dataset adapter and scoring index must be
implemented before training/evaluating the derived benchmark.

## Target: annotated nightly-period membership

The [official data description](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/data)
was retrieved through Kaggle's public PageService API on 2026-09-19. It describes
annotations of the longest nightly inactivity period while the device is worn.
Periods have a 30-minute minimum and may contain activity lasting up to 30 minutes.
No identifiable period means no annotated pair; nonwear also lacks annotations.
The organizer describes each series as a unique experimental subject.

We therefore adopt a derived **annotated-period membership** target:

- `1`: inside an eligible annotated period.
- `0`: outside supported annotated periods, bounded by eligible consecutive nights.
- `-1`: unknown under the versioned candidate policy.

Neither class asserts clinical sleep/wake at every sample. The outside-period
class may include nonwear or unannotated sleep; it does not assert confirmed wear
or continuous wakefulness. The target is also
separate from the competition's event-detection AP metric. Existing historical
model output names remain unchanged; future dataset/model artifacts must carry
`sleepkit.annotated_period_membership/v1` and its explicit class names.

Source retrieval (public, no credentials):

```sh
curl -s https://www.kaggle.com/api/i/competitions.PageService/ListPages \
  -H 'Content-Type: application/json' --data '{"competitionId":53666}'
```

Select published `data-description`, page ID `239165`, post ID `17663170`.
SHA-256 of its exact UTF-8 `content` string:
`8a2f93fc7f5a9264b14773ef98d0e6a227e6ab5b595edd3a70d0758442813dbf`.
Local evidence retains the source response. We rely on the organizer's subject
identity statement; no independent person-linkage audit was performed.

## Deterministic assignment

```sh
uv run python -m sleepkit.recipes.detection.split \
  --data /path/to/original-cmidss --coverage /path/to/candidate-coverage.json \
  --output /path/to/new-frozen-split --seed 0
```

Every audited subject participates, including subjects with no eligible contexts.
Default groups are series IDs; `--groups` can supply a complete subject-to-group
JSON mapping for sources with known repeated recordings. Duplicate JSON keys,
missing/extra subjects, invalid IDs, and unsupported coverage evidence fail.
Supplied identity grouping is recorded, not independently verified.

Unique groups are sorted by SHA-256 of the versioned algorithm prefix, seed, and
group ID with NUL separators. The first floor(70% of groups) go to training, the
next floor(15%) to validation, and the remainder to test. At least seven groups
are needed. Rows, group-map insertion order, coverage counts, and model scores
cannot influence assignment. No subjects are dropped and no reseeding occurs if
class coverage is inadequate; that condition is reported after assignment.

The new directory contains `split.json`, `groups.json`, `sources.json`, and
`protocol.json`. The protocol binds their hashes, coverage report, supplied group
mapping, target, policy, context rules, and partition summaries. Source/evidence
hashes are checked before and after freezing. Existing destinations are preserved;
publication assumes a single local writer. Keep these identifier-bearing files
with local experiment evidence rather than inside a public model bundle.

## Frozen local cohort

Seed zero, all 277 series, 240 frames per context:

| Partition | Subjects | Eligible contexts | Inside targets | Outside targets |
| --- | ---: | ---: | ---: | ---: |
| Training | 193 | 34,474 | 3,241,944 | 5,031,816 |
| Validation | 41 | 8,173 | 751,356 | 1,210,164 |
| Test | 43 | 8,582 | 808,474 | 1,251,206 |

Every partition contains both classes. Frozen `split.json` SHA-256:
`08c2ac221b3212b5589d29f623015994da8c6fd104708879b40bcb4db60306e7`.
These are candidate-coverage counts, not model performance. Historical SD-2
training exposure remains unknown, so this new split cannot certify that model
as held out. Preserve historical preprocessing and treat its results as descriptive
until exposure evidence exists.

Next: materialize a separate target-versioned dataset from the verified events,
retain unknown masks, bind the frozen assignment, and persist exact scoring
indices. Update exported class metadata for the derived target before training.

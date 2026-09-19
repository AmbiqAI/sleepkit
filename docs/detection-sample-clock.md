# Independent sample clocks

Detection preprocessing v3 accepts local TS/ENMO/ZANGLE values plus an optional
independent `sample_time` vector. It must contain signed int64 Unix seconds, align
one-to-one with source samples, and increase by exactly five seconds. UTC controls
sampling continuity; local TS still supplies the cyclic time-of-day feature. Without
an independent clock, the existing strict local-TS cadence check applies.

```python
from sleepkit.recipes.detection.preprocessing import prepare
from sleepkit.recipes.detection.inference import predict

features = prepare(sensor_data, cache_directory, sample_time=utc_seconds)
result = predict(bundle_directory, sensor_data, sample_time=utc_seconds)
```

Clocks must come from source evidence. Generating a five-second sequence from row
indices alone does not establish continuity. Validation runs before cache lookup;
cache keys bind clock presence, clock values, sensor values, and the preprocessing
specification. Output times and context availability remain recording-relative,
starting with the first feature target at 55 seconds. Nonfinite sensor windows remain
invalid, and finite local TS must still lie in `[0, 86400)`.

Exact v2 fitted states remain supported with their original strict local-clock rule.
They retain v2 when serialized and reject an explicit independent clock. New runs
write v3; v1 remains unsupported because its midnight feature formula differs.

## HDF5 adapter

The recipe reads an optional `sample_time` dataset with a `units="unix_seconds"`
attribute. `read_recording(root, subject, labels=False)` returns `Recording(data,
labels, sample_time)` and does not require annotations. The training generators use
this same reader and preprocessing path. The old two-array `read_subject` interface
remains available; callers needing the clock should use `read_recording`.

Use a fresh source-alignment report containing `first_utc_seconds` to build a separate
clocked dataset. Only a successful report with matching source hashes can support
reconstruction of the regular UTC sequence. Older reports must be regenerated.
The clock materializer preserves historical labels; it does not implement the
[candidate label policy](detection-label-policy.md).

## Event clock evidence

```sh
uv run --extra source-audit python -m sleepkit.recipes.detection.alignment \
  --data /path/to/cmidss --parquet /path/to/train_series.parquet \
  --events /path/to/train_events.csv --output /path/to/new-alignment.json
```

With `--events`, every available event timestamp must equal raw UTC at its actual
source step. A simultaneous shift of onset and wakeup therefore fails even if their
duration agrees. Missing step/timestamp rows remain counted as missing. Malformed,
out-of-range, unknown-subject, unmatched, or ambiguous source-step claims fail the
requested audit; an entirely missing event set does not pass. The report binds the
events file hash and detects file changes during verification.

This verifies event timing, not sleep/wake semantics, independent-person grouping,
or historical model training exposure. The [candidate interval builder and coverage audit](detection-label-policy.md) are
implemented separately. Annotation-protocol review remains a gate before benchmarking.

On the audited local snapshot, all 9,585 available event rows match raw UTC exactly;
4,923 of 14,508 rows have neither step nor timestamp. Raw alignment still passes all
127,946,340 samples across 277 subjects. Events SHA-256:
`e785115b0772b2953057fa333cc6990c4d9879a6a5321dc6e6cf8b13ef68bf58`.
These results apply only to the source hashes in the local report.

To copy explicitly selected subjects with verified clocks:

```sh
uv run python -m sleepkit.recipes.detection.clock_source \
  --data /path/to/cmidss --alignment /path/to/new-alignment.json \
  --output /path/to/new-clocked-source --subjects subject_a subject_b
```

The parent output directory must exist. Publication assumes a single local writer;
existing destinations fail. Original files are hash-checked before and after copying.
The new directory contains local source/output hash provenance in `clock-source.json`
and its checksum. Keep both the alignment report and this provenance with the local
experiment. Reports are trusted local evidence, not signed external attestations.
The regular clock is reconstructed only because the raw scan verified every sample.
Use the resulting directory as the normal recipe `root` with an explicit split.

A local adapter probe copied one forward-shift and one backward-shift recording
(491,040 total samples) and produced all 81,838 feature frames as valid. Sensor
values and historical labels compared equal to the originals. The original
local-clock-only path rejected both recordings, as expected. This is input-path
validation, not a training or model-quality result.

Hash-bound HDF5 evidence requires self-contained files: external links/storage,
soft links, and virtual datasets are rejected by the reader, audit, and materializer.
Their referenced contents can change independently of the HDF5 file checksum.

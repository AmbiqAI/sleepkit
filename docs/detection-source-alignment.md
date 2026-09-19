# Raw-source alignment verification

This check establishes whether the legacy HDF5 sensor rows match the raw Parquet
series, independently of local time-of-day changes. It is separate from annotation
coverage, person identity, and model evaluation.

```sh
uv sync --extra detection --extra source-audit
uv run --extra source-audit python -m sleepkit.recipes.detection.alignment \
  --data /path/to/cmidss --parquet /path/to/cmidss/train_series.parquet \
  --output /path/to/new-local-alignment.json
```

The reader uses optional PyArrow in bounded batches (default 65,536 rows). The
report contains source hashes and per-series details and stays local. Source files
are never written; an existing report is not overwritten. A failed alignment writes
the report and returns a failing exit code. Malformed/null source values fail
explicitly rather than being filled or resampled.

For every series the verifier checks:

- The first source step is zero and subsequent steps equal the HDF5 row indices.
- Absolute UTC timestamps advance exactly five seconds, including batch boundaries.
- Source and HDF5 row counts agree, and no raw series lacks its HDF5 counterpart.
- TS equals local seconds of day from the source timestamp; ENMO and ZANGLE match
  the source after float32 conversion. Matching NaNs count as preserved missing
  sensor values, not as evidence of signal quality.
- Local-clock jumps and timezone-offset changes are counted independently.
- Source hashes remain unchanged throughout verification.

## Verified local snapshot

On 2026-09-19, all **127,946,340 rows across 277 subjects passed**. There were no
step, UTC-cadence, row-count, or sensor-channel mismatches. All 44 local-clock changes
coincided with timezone-offset changes; UTC spacing remained five seconds. Thus the
44 apparent gaps in the annotation audit are local-clock shifts, not physical
sampling gaps in this snapshot.

Raw Parquet SHA-256:
`fd11f8c3af0507757cade38419f5ec34af28dd6e3057ab17ef05b0af717106af`.
The local report binds every HDF5 source individually. Do not generalize these
results to files with different hashes.

This resolves the step-origin and row-order assumption for the audited files. It
does not establish event-label correctness, that every series belongs to a different
person, or that historical model training excluded any particular subject.

## Reader change supported by this evidence

The current detector validates cadence using local TS and rejects these 44 files.
The correct next adapter should preserve raw UTC/elapsed sample time alongside
local TS, validate continuity against the independent clock, and use local TS only
for the time-of-day feature. Carry that input contract through feature caching,
training, and unlabeled inference. Keep strict handling for inputs without verified
clock evidence; do not whitelist arbitrary one-hour jumps or silently omit subjects.

The [candidate event-supported label policy](detection-label-policy.md) is still a
proposal. It must be reviewed against the source annotation protocol before being
materialized as a new training/evaluation dataset.

## Parquet dependency compatibility

The old fastparquet 2023 pin uses a NumPy 1 ABI and fails to import under the existing
NumPy 2 stack. The dependency is raised to `fastparquet>=2024.11.0,<2025`, whose
wheels support the project's Python 3.12/3.13 range. An explicit fastparquet-engine
round trip is tested so optional PyArrow cannot hide a broken legacy reader.

PyArrow is an optional `source-audit` extra, independent of TensorFlow and of the
legacy Parquet reader. CI installs that extra and requires the alignment tests.

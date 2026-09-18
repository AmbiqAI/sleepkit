# SleepKit 1.0 development workflow

The 1.0 alpha introduces composable preprocessing and experiment functions. Historical
0.x artifacts are unchanged. The initial reference recipe uses synthetic two-stage
signals to validate plumbing; its scores are not sleep-monitoring performance claims.

## Install and run

```sh
uv sync --extra train --extra runtime --extra hf
uv run --no-sync sleepkit smoke --output results/v1-smoke --epochs 2
uv run --no-sync sleepkit validate results/v1-smoke/deploy
uv run --no-sync sleepkit publish results/v1-smoke/deploy Ambiq/sleepkit-example-v1
```

Use a new output directory for each run. Publication defaults to a local dry run,
which prints its staging directory. `--upload --license-file <path>` performs an
upload and requires Hub authentication; repositories default to private. Public
publication additionally requires `--public`. Review the generated card, license,
metrics, and provenance before publishing. No upload is performed by training.

For a minimal inference installation:

```sh
pip install 'sleepkit[runtime]'
```

The development source tree must be installed while this alpha is unpublished.
`import sleepkit`, `sleepkit.preprocessing`, and `sleepkit.runtime` do not import
TensorFlow, HeliaEdge, dataset readers, plotting, tracking, or serial support.

The CLI also exposes `train --records <record.npz> ... --preprocessing <definition.json>
--class-names wake sleep --output <new-directory>` for the reference model, and
`evaluate --bundle <deploy-directory> --records <held-out-record.npz> ...` for
independent LiteRT evaluation. Evaluation does not require training dependencies.
Select held-out records using the saved subject split. Save a preprocessing
definition with `json.dumps(preprocess.to_dict())`.

## Assemble an experiment in Python

The recipe is ordinary Python in `sleepkit/recipes/staging.py`. Copy or import the
parts you need; there is no task registration or trainer subclass requirement.

```python
from sleepkit.data import split_subjects
from sleepkit.preprocessing import Channel, WindowFeatures, Standardizer, prepare

# records come from your adapter; each includes signals and optional annotations.
preprocess = WindowFeatures(
    channels=(Channel("movement", "g", "accelerometer", "wrist"),),
    window_seconds=30,
    stride_seconds=30,
)
split = split_subjects(records, seed=7)
normalizer = Standardizer.fit(records, preprocess, split, cache_dir="cache")
prepared = normalizer.transform(prepare(records[0], preprocess, cache_dir="cache"))
```

`WindowFeatures` currently computes mean and standard deviation for each channel.
It is deliberately small and is not the historical FS-W-PA-14 transform. Different
native rates are handled by selecting each channel's samples on a common time
interval, without implicit resampling. Sensor modality, location, and units must
match exactly. Windows with inadequate valid data are masked; incomplete tails
are dropped. Targets are selected at window centers from half-open annotation
intervals. Overlapping target intervals are rejected.

Use `model_windows` to assemble nonoverlapping model contexts. It drops contexts
with invalid feature frames and, for training, missing labels. The same function
supports unlabeled inference with `require_targets=False`. Context assembly must
be called separately for each record so windows never cross recording boundaries.

A custom deterministic transform can be any callable returning `Prepared`, with a
JSON-serializable `to_dict()` describing its implementation version and settings.
The preparation cache fingerprints input content, annotations, identities, signal
metadata, and the transform definition. Changing source data or preprocessing
invalidates it; changing a model does not. Fitted normalization is separate from
the cache, and its saved state checks the preprocessing schema before use.

The initial bundle serializer supports `WindowFeatures`; custom transforms need
an explicit serializer/runtime implementation before they can be exported. They
can still be used freely for training and evaluation.

## Data adapters

`Record` and `Signal` accept NumPy arrays directly. Regular signals retain their
own sampling rates, starts, units, modality, location, and sample-validity masks.
All starts and annotations use one record-relative clock. Irregular signals need
an explicit alignment/resampling adapter; do not hide gaps by concatenating data.

`save_record`/`read_record` in `sleepkit.data.readers` provide an NPZ round trip
without pickle. `read_edf` optionally reads native-rate EDF channels with explicit
channel mappings and checks physical units. `read_nsrr_stages` decodes NSRR stage
annotations using a caller-provided class map. For example:

```python
from sleepkit.data.readers import read_edf, read_nsrr_stages
from sleepkit.preprocessing import Channel

labels = read_nsrr_stages("subject-nsrr.xml", {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1})
record = read_edf(
    "subject.edf",
    dataset="mesa", subject="subject",
    channels={"SpO2": Channel("spo2", "%", "oximetry", "finger")},
    annotations=labels,
)
```

Verify sensor metadata and EDF units against the actual acquisition. For example,
MESA `Leg` and wrist acceleration are different observations. Unmapped annotation
classes remain unlabeled gaps. EDF+D is rejected until discontinuous time handling
is implemented. Dataset ingestion is optional (`sleepkit[datasets]`).

## Deployment boundary

The bundle contains Keras/TFLite models, preprocessing and normalization state,
class labels, a manifest, checksums, metrics, and synthetic reference vectors.
`Predictor.predict_record(record)` runs saved preprocessing and returns logits and
feature-center timestamps, without requiring labels. Window outputs become available
at the feature window end; they should not be presented as zero-lookahead samples.

INT8 conversion uses training calibration inputs and a separate validation set.
Export fails if accuracy degradation exceeds the configured allowance. Conversion
validation is not a replacement for held-out evaluation or hardware profiling.
Checksums detect accidental file changes; they are not cryptographic signatures.

The current runtime supports host raw-record inference. Embedded implementations of
preprocessing are not included. Dataset subject membership stays in local `split.json`;
publishable normalization state records a fingerprint rather than subject identifiers.

Hub loading uses `Predictor.from_pretrained(repo_id, revision="<commit-hash>")`
with both the `runtime` and `hf` extras. Choose an immutable revision for reproducibility.

## Initial compatibility checks

On Linux CPU, TF 2.21.0 / Keras 3.15.1 loaded all six documented baseline Keras
models and produced finite outputs for zero inputs. LiteRT 2.2.0 invoked all six
existing TFLite exports. These checks establish basic loadability, not prediction
parity or accuracy. The additional `ss-4-tcn-lg` Keras 2.15 archive failed to load
because its serialized module path no longer exists; its TFLite artifact invoked.
Historical artifacts were not rewritten. The four-stage TFLite output scale still
needs investigation before a baseline release.

`scripts/inventory_baselines.py` records artifact checksums, Keras metadata, and
historical metrics without loading or modifying weights.

## Migration status

Implemented: optional dependency groups, lazy legacy imports, signal records and
subject splits, window preprocessing and caching, fitted normalization, portable
array/EDF readers, pooled classification metrics, a synthetic Keras recipe, INT8
export validation, standalone LiteRT inference, and dry-run Hub publication.

Still to migrate: physiological FS-W-PA-14 and other historical feature algorithms,
full dataset-specific ingestion recipes, apnea/event-level evaluation, baseline
release cards with reconciled metrics, and embedded preprocessing conformance.
The legacy task implementation remains available through `sleepkit-legacy` with
`uv sync --extra legacy`; it retains its historical behavior and is not the new
recommended API. Avoid using its old split/evaluation behavior for new comparisons.

# A detection pipeline written in Python

The new detection recipe is developed alongside the legacy task CLI. It reads
CMIDSS sensor-channel HDF5 files and owns feature generation, normalization,
training, held-out evaluation, and artifact export. It uses ordinary functions,
a small config, a native Keras model, and the shared artifact package from
[the HF publishing workflow](huggingface-artifacts.md).

This is an experimental major-version building block. It does not reproduce the
historical SD-2-TCN-SM model. Global training-only normalization, invalid-data
handling, context sampling, and model architecture differ deliberately. Use the
unchanged historical bundle as the comparison baseline once its evaluation
provenance is resolved.

## Install and exercise the path

The tested training stack is TensorFlow 2.21, Keras 3.15.1, and LiteRT 2.2 on Linux
CPU. TensorBoard is now an explicit dependency because legacy training uses its
callback and TF 2.21 no longer brings it in transitively. The original SleepKit
training dependencies remain installed; a full dependency split is later work.

```sh
uv sync --extra detection
uv run --extra detection python -m sleepkit.recipes.detection smoke --output /tmp/detection-smoke
```

The smoke command generates synthetic HDF5 sensor records, trains one epoch,
evaluates, saves/reloads Keras, converts to float32 TFLite, replays synthetic
reference vectors, and runs unlabeled sensor-to-prediction inference. Its package
is explicitly marked synthetic. No data is downloaded and nothing is published.
Choose a fresh output directory; existing runs are never overwritten.

A dependency-minimal training environment from this checkout can instead install
`tensorflow==2.21.0 keras==3.15.1 numpy h5py ai-edge-litert==2.2.0` and run the same
module. Data/preprocessing helpers need NumPy and HDF5; inference needs NumPy and
LiteRT, plus HDF5 only when using that reader. Neither inference nor preprocessing
imports TensorFlow, the legacy dataset registry, or HeliaEdge.

## Train with local CMIDSS subjects

Input files must be `<subject>.h5`, directly under the supplied data directory:

| Field | Contract |
| --- | --- |
| `data` | float array `[3, samples]`, ordered TS, ENMO, ZANGLE |
| TS | local seconds of day, contiguous five-second samples; midnight wrapping allowed |
| ENMO / ZANGLE | CMIDSS sensor-derived movement and angle channels, as in the existing HDF5 reader |
| `sleep_stages` | integer `[samples]`: 0 wake, 1 sleep, -1 unknown |
| `sample_rate_hz` attribute | optional for legacy files; must equal 0.2 when supplied |
| `channel_names` attribute | optional for legacy files; must be TS, ENMO, ZANGLE when supplied |

Missing metadata explicitly selects the legacy CMIDSS assumptions (channel order and
0.2 Hz rate), recorded in `recipe.json` under `reader`. New inputs should include
both attributes; conflicting metadata is rejected. This is a CMIDSS-specific reader,
not a generic three-channel HDF5 loader.

These are sensor-derived channels, not raw accelerometer axes or previously
generated five-feature inputs. The existing dataset conversion can produce these
files, but it labels unannotated time as wake. This recipe accepts labels as-is;
annotation policy must be reviewed before claiming a benchmark result. New source
adapters can supply -1 for unknown labels. The recipe does not download datasets,
pair sleep events, or silently resample irregular signals.

Write a local subject split with three nonempty, disjoint partitions:

```json
{
  "train": ["subject-a", "subject-b"],
  "validation": ["subject-c"],
  "test": ["subject-d"]
}
```

Each subject must occur once. Splitting is explicit and no validation/test subject
contributes to fitted normalization. For datasets with multiple recordings per
person, the adapter must group them under the same subject before creating splits;
this CMIDSS adapter treats each source series file as one subject.

```sh
uv run --extra detection python -m sleepkit.recipes.detection train \
  --data /path/to/cmidss --split /path/to/split.json \
  --output /path/to/new-run --cache /path/to/local-feature-cache \
  --context 240 --epochs 5 --batch-size 32 --learning-rate 0.001 --seed 0
```

The five config values above control this recipe's experiment. Paths are function
arguments. Configuration does not name Python classes or describe execution steps.
The training loop uses all complete valid contexts; the final epoch is evaluated
(no automatic best-checkpoint selection). Standard Keras callbacks can implement
other policies explicitly in Python.

## Preprocessing and alignment

A feature window contains 12 source samples (60 seconds), advancing 6 samples
(30 seconds). Features are cosine time of day, ENMO mean/std, ZANGLE mean/std,
with population standard deviations. The time-of-day feature averages the cosine
of each source timestamp: `mean(cos(2*pi*TS/86400))`. Encoding before averaging
keeps windows spanning midnight close to +1 rather than interpreting them as noon.
This corrects the legacy formula and changes the preprocessing contract to v2.
Old v1 caches are bypassed, and v1 fitted states are rejected by this implementation.
Retrain and export a new bundle; do not pair v1 model weights or normalization with
v2 features. Historical baseline packaging and its original feature requirements
remain unchanged.
A window is invalid if any source sample is nonfinite; invalid windows are not
imputed. Fitted normalization uses valid feature windows from training subjects
only, including training features without target labels, and saves mean and
`sqrt(population_variance + 1e-6)` for each feature.

Windows are timestamped at their last source sample (first timestamp 55 seconds),
using seconds relative to the recording start. The feature becomes available at
60 seconds. Labels come from that last source sample. Model contexts are consecutive,
nonoverlapping groups of feature windows; incomplete tails and contexts containing
invalid features or unknown training labels are dropped. Removed contexts are never
stitched together. Inference does not require or use labels.

The convolutional model uses future features inside each context. A prediction is
therefore available only after the complete context, not at its target timestamp.
The API currently processes complete recordings; it is not a streaming state machine.

The optional cache contains only stateless feature arrays, masks, and sample indices.
Its key includes the input sensor bytes and feature contract. Changing labels,
normalization, or model settings does not invalidate feature computation. Labels
are read and aligned separately each time. Cache files contain derived subject data
and stay local. Processing holds one subject and a bounded shuffle buffer in memory;
it does not concatenate the whole dataset. Very large single recordings still need
an incremental reader in a later adapter.

## Compose an experiment

```python
from sleepkit.recipes.detection.recipe import Config, run

run(data_root, split_file, output_dir, Config(epochs=10),
    cache=cache_dir, model_builder=my_model_builder, callbacks=my_keras_callbacks)
```

A builder receives `(context, feature_count)` and returns a float32 Keras model
`[batch, context, 5] -> [batch, context, 2]` with a linear final activation and
WAKE/SLEEP logits. Native serializable Keras layers are supported by the default
safe load path. Custom architectures must pass Keras reload and builtin-only TFLite
conversion; unsupported operators fail rather than silently enabling custom ops.
This logits contract belongs to this recipe, not the generic artifact format.

Experiments can also import and call individual steps directly:

```python
from sleepkit.recipes.detection.data import load_split, subject_features
from sleepkit.recipes.detection.preprocessing import Normalizer

split = load_split(split_file, data_root)
normalizer = Normalizer.fit(subject_features(data_root, split["train"], cache_dir))
# Continue with the recipe's dataset(), build_model(), evaluate(), or your own code.
```

There is no base trainer or plugin registry. Optional integrations are explicit
Python imports; a second recipe will determine which blocks merit extraction.
Bump `SPEC.implementation_version` whenever feature extraction behavior changes,
so caches and saved states cannot silently reuse an older implementation.
The `sleepkit.cmidss_wrist/v2` preprocessing identifier and `sleepkit.detection/v1`
recipe identifier are versioned compatibility contracts. Inference refuses unknown
contracts; manifests never specify arbitrary code to import.

## Run outputs and inference

The run directory retains exact subject splits, per-file source hashes, Keras
history, intermediate SavedModel, and source model files. Its `bundle/` subdirectory
is the self-contained artifact package:

- `model.keras`: trainable checkpoint; safe reload is checked.
- `model.tflite`: fixed-batch float32 deployment model; synthetic conversion parity is checked.
- `preprocessing.json`: supported feature contract and fitted normalization state.
- `recipe.json`: config, context, class order, versions, code hashes, and source/split fingerprints.
- `metrics.json`: held-out Keras confusion matrix, accuracy, macro F1, cross entropy, and number of feature epochs evaluated.
- `reference.npz`: synthetic TFLite input/output vectors.
- Model card, inventory, checksums, and separate validation results.

Subject identifiers, source recordings, and actual subject reference inputs are
excluded from the bundle. Keep local splits/source hashes for reproducibility;
fingerprints in the bundle verify a known local run, but cannot reconstruct its
private data. Metrics report the provided labels, not clinical accuracy. No task
acceptance threshold is implied by successful evaluation or runtime conformance.

```sh
python -m sleepkit.artifacts validate /path/to/run/bundle --profile runnable --runtime
python -m sleepkit.recipes.detection predict \
  --bundle /path/to/run/bundle --data /path/to/cmidss \
  --subject subject-without-labels --output /tmp/predictions.npz
python -m sleepkit.artifacts publish /path/to/run/bundle owner/experiment --profile runnable
```

Prediction output contains target `times`, complete-context `available_at` times,
`logits`, and `probabilities`; softmax is applied
exactly once. The last command is a dry run. These experimental bundles intentionally
have no chosen model license, so publication requires staging a licensed release
using the artifact API. License choice and publication are separate release work.

Current limits: float32 CPU TFLite only, no quantization or MCU validation, no full
historical-baseline parity, no full-cohort benchmark. The next useful experiment is
an agreed subject split and annotation policy, followed by baseline comparison and
int8 conversion with representative training inputs.

The export path follows the native [Keras export API](https://keras.io/api/models/model_saving_apis/export/)
and [TensorFlow-to-LiteRT conversion](https://developers.google.com/edge/litert/conversion/tensorflow/convert_tf).

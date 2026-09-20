# Fixed int8 conversion and release preparation

This step converts the [first membership experiment](detection-first-experiment.md)
without retraining or replacing its fitted preprocessing. It preserves the original
float TFLite and Keras bytes and measures the quantized model on exactly the same
eligible test outputs.

## Run the Python recipe

```python
from sleepkit.recipes.detection.quantization import quantize_run

# source is the same AnnotatedDataset used for the original run.
metrics = quantize_run(
    "/path/to/parent-run",
    source,
    "/path/to/new-int8-run",
    cache="/path/to/stateless-feature-cache",
)
```

The quantization recipe deliberately makes one fixed conversion attempt. It writes
its declaration before selection/conversion and does not use evaluation scores to
change the calibration budget. The separate `collect_calibration` block can support
future declared experiments without introducing an orchestration configuration.

Calibration considers every frozen training series, retaining up to two uniformly
selected eligible native contexts per series with seed zero and no replacement.
The sampler derives a stable per-series seed and uses reservoir sampling; it records
series with zero eligible contexts. It never reads validation/test recordings for
calibration. Unknown targets and invalid features exclude whole contexts, matching
training. The parent global training normalizer is applied without refitting.

Exact selections, source hashes and normalized inputs are saved locally. The recipe
checks their hashes and requires equality between the returned and persisted arrays,
then gives the reloaded array to the converter. Only aggregate counts, policy and
hashes enter the public bundle. Validation/test data are not used to determine ranges.

Conversion follows TensorFlow's [full integer quantization settings](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/performance/post_training_quantization.md):
default optimization, a representative dataset, `TFLITE_BUILTINS_INT8`, and int8 I/O.
The resulting FlatBuffer is inspected across all subgraphs to reject noninteger
tensors or custom operators. This demonstrates an integer graph, not MCU support.

## Matched evaluation and inference

Evaluation rechecks every original native source/context/feature endpoint/target
coordinate, including rejected contexts, and reproduces the original denominator.
It compares saved Keras logits, replayed original float TFLite, and int8 outputs.
Float replay must match Keras within the declared tolerance; int8 differences are
measured rather than asserted to meet that tolerance. Results include confusion,
class support/recall, macro-F1, cross-entropy, per-series results, disagreements,
correct-to-wrong/wrong-to-correct counts, ties and input clipping.

The runtime quantizes using nearest rounding with ties to even, clips to int8,
and dequantizes output logits using the declared per-tensor scale and zero point.
`output_saturated_values` counts encoded -128/127 endpoints; that count alone does
not prove clipping. Probabilities are computed with one softmax after dequantization.
No acceptance threshold or target-hardware result is implied by conversion success.

The existing raw-sensor entry point supports both bundled models:

```python
from sleepkit.recipes.detection.inference import predict

integer = predict("int8-run/bundle", sensor_data, sample_time=utc_seconds)
floating = predict("int8-run/bundle", sensor_data, sample_time=utc_seconds,
                   model_name="model.float.tflite")
```

`sensor_data` is `[3, samples]` in TS/ENMO/ZANGLE order; `utc_seconds` is the optional
aligned signed int64 five-second clock supported by v3. The selected model must be
a declared TFLite artifact with matching shape, dtype, signature and quantization.
Original float-only bundles retain their default behavior. This remains offline,
noncausal inference with a roughly two-hour context for the real experiment.

## Release contents and publication boundary

The new bundle contains the default int8 `model.tflite`, preserved
`model.float.tflite`, `model.keras`, `preprocessing.json`, `recipe.json`, aggregate
`calibration.json` and `metrics.json`, two synthetic runtime reference files, a model
card, manifest, checksums, and separate validation checks. It contains no source
recordings, identifiers, actual calibration arrays or per-series predictions.

The local experiment directory additionally retains the declaration, private
calibration evidence, per-series evaluation, and exact int8/float test arrays.
Staging refuses existing output directories and preserves the original run.

```sh
python -m sleepkit.artifacts validate /path/to/int8-run/bundle \
  --profile runnable --runtime
python -m sleepkit.artifacts publish /path/to/int8-run/bundle \
  AmbiqAI/sleepkit-membership-tcn-seed0 --profile runnable
```

The second command is a local dry run. The repository name is a proposed dedicated
destination, not an existing deployment. Upload requires an explicitly chosen model
artifact license and the `--upload` option; source-code licensing is not assumed to
cover model weights. No upload is part of quantization. Hosting these files on the
Hub would provide downloadable artifacts, not a hosted inference endpoint.

MCU operator compatibility, arena/RAM requirements, device latency and power remain
unmeasured. Any later calibration/model selection must use validation data or a new
evaluation protocol, rather than repeatedly optimize against the inspected test set.

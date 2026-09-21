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

## First fixed conversion result

On 2026-09-20, revision `86b98c2` converted the original five-epoch checkpoint in
one attempt. Calibration selected 372 contexts (89,280 feature frames) from 186
training series; the other seven assigned training series had no eligible context.
All 193 were accounted for, and neither validation nor test features supplied
calibration ranges. Whole-cohort source hashes were still checked for integrity.

On the same 2,059,680 eligible frames from 43 test series:

| Metric | Original Keras / float TFLite | Int8 TFLite |
| --- | ---: | ---: |
| Accuracy | 95.7657% | 95.7560% |
| Macro-F1 | 0.955611 | 0.955520 |
| Outside-period recall | 96.4934% | 96.4449% |
| Inside-period recall | 94.6395% | 94.6899% |
| Unweighted series mean accuracy | 95.8477% | 95.8249% |
| Unweighted series mean macro-F1 | 0.933468 | 0.933174 |
| TFLite file size | 23,212 bytes | 13,568 bytes |

The int8 file is 41.5% smaller, with an accuracy reduction of 0.00966 percentage
points. Its confusion matrix in outside/inside order is
`[[1206725, 44481], [42931, 765543]]`. Small aggregate changes do not imply identical
predictions: 23,343 decisions changed (1.1333%), including 11,771 correct-to-wrong
and 11,572 wrong-to-correct changes. There were 21,649 int8 output ties, resolved
to class zero. Only two of 10,298,400 normalized input values were clipped; 54
encoded output values were at range endpoints.

Int8 mean absolute logit difference from Keras was 0.301505 and the maximum was
12.547834; probability/calibration equivalence is not established. Original float
TFLite reproduced all Keras class decisions and passed full-test logit comparison
at `atol=1e-5, rtol=1e-4` (maximum absolute difference `2.288818359375e-5`). The
relative tolerance matters for that maximum. Evaluation coverage and single-class
series conventions remain those of the first experiment.

The integer graph contains 16 int8 tensors and seven int32 tensors, with builtin
ADD, CONV_2D, FULLY_CONNECTED and RESHAPE operators. Int8 input scale/zero point are
`0.14650388062000275 / -111`; output scale/zero point are
`0.6170711517333984 / 82`. The int8 model SHA-256 is
`46c8cdc4f10461818959c4100b9232c2d6ba09bad7ba94f89a357949d8854a13`.
Calibration, conversion, evaluation and staging took 48.5 seconds locally; this
is not a device-latency benchmark.

Evidence is retained under
`sleepkit-evaluation-evidence/experiments/membership-int8-seed0-20260920/` alongside
the repository. Both synthetic runtime references replayed successfully. The public
raw-sensor API also reproduced saved outputs exactly for the first eligible context
of each test series: 10,320 frames per model, checking both default int8 and optional
float inference, class semantics and timestamps.

Independent review reconstructed all 372 calibration selections and all 89,280
calibration feature frames directly from training sources, reproduced every pooled
and per-series metric, and verified clipping counts, graph contents, preserved
artifacts and bundle privacy. A separate preprocessing/runtime replay checked the
first and last eligible context in each test series through both models: 86 contexts
per model with bitwise-identical outputs. No material finding remained.

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

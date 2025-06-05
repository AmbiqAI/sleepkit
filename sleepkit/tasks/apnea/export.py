import os
import shutil

import keras
import neuralspot_edge as nse
import numpy as np

from ...defines import TaskParams
from ...features import H5Dataloader
from .utils import create_data_pipeline


def export(params: TaskParams):
    """Export sleep apnea model.

    Args:
        params (TaskParams): Task parameters
    """

    os.makedirs(params.job_dir, exist_ok=True)
    logger = nse.utils.setup_logger(__name__, level=params.verbose, file_path=params.job_dir / "export.log")
    logger.debug(f"Creating working directory in {params.job_dir}")

    # target_classes = sorted(list(set(params.class_map.values())))
    # class_names = params.class_names or [f"Class {i}" for i in range(params.num_classes)]

    tfl_model_path = params.job_dir / "model.tflite"
    tflm_model_path = params.job_dir / "model_buffer.h"

    dataloader = H5Dataloader(
        path=params.feature.save_path,
        frame_size=params.frame_size,
        feat_key=params.feature.feat_key,
        label_key=params.feature.label_key,
        mask_key=params.feature.mask_key,
        feat_cols=params.feature.feat_cols,
    )

    feat_shape = dataloader.feature_shape

    test_ds = create_data_pipeline(
        dataloader=dataloader,
        subject_ids=dataloader.test_subject_ids,
        samples_per_subject=params.test_samples_per_subject,
        num_classes=params.num_classes,
        batch_size=params.batch_size,
        cache_size=params.test_size // params.batch_size,
    )

    test_x = np.concatenate([x for x, _ in test_ds.as_numpy_iterator()])
    test_y = np.concatenate([y for _, y in test_ds.as_numpy_iterator()])

    logger.debug("Loading trained model")
    model = nse.models.load_model(params.model_file)
    inputs = keras.Input(shape=feat_shape, batch_size=1, name="input", dtype="float32")
    model(inputs)

    if not params.use_logits and not isinstance(model.layers[-1], keras.layers.Softmax):
        model = nse.models.append_layers(model, layers=[keras.layers.Softmax()], copy_weights=True)
    # END IF

    flops = nse.metrics.flops.get_flops(model, batch_size=1, fpath=params.job_dir / "model_flops.log")
    model.summary(print_fn=logger.debug)

    logger.debug(f"Model requires {flops / 1e6:0.2f} MFLOPS")

    logger.debug(f"Converting model to TFLite (quantization={params.quantization.format})")
    converter = nse.converters.tflite.TfLiteKerasConverter(model=model)

    tflite_content = converter.convert(
        test_x=test_x,
        quantization=params.quantization.format,
        io_type=params.quantization.io_type,
        mode=params.quantization.conversion,
        strict=not params.quantization.fallback,
    )

    if params.quantization.debug:
        quant_df = converter.debug_quantization()
        quant_df.to_csv(params.job_dir / "quant.csv")

    # Save TFLite model
    logger.debug(f"Saving TFLite model to {tfl_model_path}")
    converter.export(tfl_model_path)

    # Save TFLM model
    logger.debug(f"Saving TFL micro model to {tflm_model_path}")
    converter.export_header(tflm_model_path, name=params.tflm_var_name)

    tflite = nse.interpreters.tflite.TfLiteKerasInterpreter(tflite_content)
    tflite.compile()

    y_pred_tf = model.predict(test_x)
    y_pred_tf = np.argmax(y_pred_tf, axis=-1).flatten()

    # Verify TFLite results match TF results on example data
    logger.debug("Validating model results")
    metrics = [
        keras.metrics.CategoricalCrossentropy(name="loss", from_logits=params.use_logits),
        keras.metrics.CategoricalAccuracy(name="acc"),
        keras.metrics.F1Score(name="f1", average="weighted"),
    ]

    y_true = np.argmax(test_y, axis=-1).flatten()
    y_pred_tfl = tflite.predict(x=test_x)
    y_pred_tfl = np.argmax(y_pred_tfl, axis=-1).flatten()

    tf_rst = nse.metrics.compute_metrics(metrics, y_true, y_pred_tf)
    tfl_rst = nse.metrics.compute_metrics(metrics, y_true, y_pred_tfl)
    logger.info("[TF METRICS] " + " ".join([f"{k.upper()}={v:.4f}" for k, v in tf_rst.items()]))
    logger.info("[TFL METRICS] " + " ".join([f"{k.upper()}={v:.4f}" for k, v in tfl_rst.items()]))

    metric_diff = abs(tf_rst[params.test_metric] - tfl_rst[params.test_metric])

    # Check accuracy hit
    if params.test_metric_threshold is not None and metric_diff > params.test_metric_threshold:
        logger.warning(f"TFLite metric dropped by {metric_diff:0.2%}")
    elif params.test_metric_threshold:
        logger.info(f"Validation passed ({metric_diff:0.2%})")

    if params.tflm_file and tflm_model_path != params.tflm_file:
        logger.debug(f"Copying TFLM header to {params.tflm_file}")
        shutil.copyfile(tflm_model_path, params.tflm_file)

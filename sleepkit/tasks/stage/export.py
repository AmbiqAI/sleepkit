"""Sleep Stage Export"""

import logging
import os
import shutil

import keras
import keras_edge as kedge
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

from ...defines import SKExportParams
from ...utils import setup_logger
from .utils import load_dataset, load_test_dataset

logger = setup_logger(__name__)


def export(params: SKExportParams):
    """Export sleep stage model.

    Args:
        params (SKExportParams): Deployment parameters
    """

    logger.info(f"Creating working directory in {params.job_dir}")
    os.makedirs(params.job_dir, exist_ok=True)

    handler = logging.FileHandler(params.job_dir / "export.log", mode="w")
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    # target_classes = sorted(list(set(params.class_map.values())))
    # class_names = params.class_names or [f"Class {i}" for i in range(params.num_classes)]

    tfl_model_path = params.job_dir / "model.tflite"
    tflm_model_path = params.job_dir / "model_buffer.h"

    ds = load_dataset(
        ds_path=params.ds_path,
        frame_size=params.frame_size,
        dataset=params.dataset,
    )
    feat_shape = ds.feature_shape
    class_shape = (params.frame_size, params.num_classes)

    input_spec = (
        tf.TensorSpec(shape=feat_shape, dtype="float32"),
        tf.TensorSpec(shape=class_shape, dtype="int32"),
    )

    test_x, test_y = load_test_dataset(
        ds=ds,
        subject_ids=ds.test_subject_ids,
        samples_per_subject=params.samples_per_subject,
        test_size=params.test_size,
        spec=input_spec,
        class_map=params.class_map,
    )

    # Load model and set fixed batch size of 1
    logger.info("Loading trained model")
    model = kedge.models.load_model(params.model_file)
    inputs = keras.Input(shape=input_spec[0].shape, batch_size=1, name="input", dtype=input_spec[0].dtype)
    outputs = model(inputs)

    if not params.use_logits and not isinstance(model.layers[-1], keras.layers.Softmax):
        outputs = keras.layers.Softmax()(outputs)
        model = keras.Model(inputs, outputs, name=model.name)
        outputs = model(inputs)
    # END IF

    flops = kedge.metrics.flops.get_flops(model, batch_size=1, fpath=params.job_dir / "model_flops.log")
    model.summary(print_fn=logger.info)

    logger.info(f"Model requires {flops/1e6:0.2f} MFLOPS")

    logger.info(f"Converting model to TFLite (quantization={params.quantization.mode})")
    tflite = kedge.converters.tflite.TfLiteKerasConverter(model=model)
    tflite.convert(
        test_x=test_x,
        quantization=params.quantization.mode,
        io_type=params.quantization.io_type,
        use_concrete=params.quantization.concrete,
        strict=not params.quantization.fallback,
    )

    if params.quantization.debug:
        quant_df = tflite.debug_quantization()
        quant_df.to_csv(params.job_dir / "quant.csv")

    # Save TFLite model
    logger.info(f"Saving TFLite model to {tfl_model_path}")
    tflite.export(tfl_model_path)

    # Save TFLM model
    logger.info(f"Saving TFL micro model to {tflm_model_path}")
    tflite.export_header(tflm_model_path, name=params.tflm_var_name)

    y_pred_tf = np.argmax(model.predict(test_x), axis=-1).flatten()

    # Verify TFLite results match TF results on example data
    logger.info("Validating model results")
    y_true = np.argmax(test_y, axis=-1).flatten()

    y_pred_tfl = tflite.predict(x=test_x)
    y_pred_tfl = np.argmax(y_pred_tfl, axis=-1).flatten()

    tf_acc = np.sum(y_true == y_pred_tf) / y_true.size
    tf_f1 = f1_score(y_true, y_pred_tf, average="weighted")
    logger.info(f"[TF SET] ACC={tf_acc:.2%}, F1={tf_f1:.2%}")

    tfl_acc = np.sum(y_true == y_pred_tfl) / y_true.size
    tfl_f1 = f1_score(y_true, y_pred_tfl, average="weighted")
    logger.info(f"[TFL SET] ACC={tfl_acc:.2%}, F1={tfl_f1:.2%}")

    # Check accuracy hit
    tfl_acc_drop = max(0, tf_acc - tfl_acc)
    if params.val_acc_threshold is not None and (1 - tfl_acc_drop) < params.val_acc_threshold:
        logger.warning(f"TFLite accuracy dropped by {tfl_acc_drop:0.2%}")
    elif params.val_acc_threshold:
        logger.info(f"Validation passed ({tfl_acc_drop:0.2%})")

    if params.tflm_file and tflm_model_path != params.tflm_file:
        logger.info(f"Copying TFLM header to {params.tflm_file}")
        shutil.copyfile(tflm_model_path, params.tflm_file)

    tflite.cleanup()

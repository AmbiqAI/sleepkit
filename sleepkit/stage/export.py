"""Sleep Stage Export"""
import logging
import os
import shutil

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from .. import tflite as tfa
from ..defines import SKExportParams
from ..metrics import f1_score
from ..utils import setup_logger
from .defines import get_stage_class_mapping, get_stage_classes
from .utils import load_dataset, load_test_dataset

logger = setup_logger(__name__)


def export(params: SKExportParams):
    """Export sleep stage model.

    Args:
        params (SKExportParams): Deployment parameters
    """
    params.num_sleep_stages = getattr(params, "num_sleep_stages", 3)

    logger.info(f"Creating working directory in {params.job_dir}")
    os.makedirs(params.job_dir, exist_ok=True)

    handler = logging.FileHandler(params.job_dir / "export.log", mode="w")
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    target_classes = get_stage_classes(params.num_sleep_stages)
    class_mapping = get_stage_class_mapping(params.num_sleep_stages)

    tfl_model_path = params.job_dir / "model.tflite"
    tflm_model_path = params.job_dir / "model_buffer.h"

    ds = load_dataset(
        handler=params.ds_handler, ds_path=params.ds_path, frame_size=params.frame_size, params=params.ds_params
    )

    class_shape = (params.frame_size, len(target_classes))

    test_x, test_y = load_test_dataset(
        ds=ds,
        subject_ids=ds.test_subject_ids,
        samples_per_subject=params.samples_per_subject,
        test_size=params.test_size,
        feat_shape=ds.feature_shape,
        class_shape=class_shape,
        class_map=class_mapping,
    )

    # Load model and set fixed batch size of 1
    strategy = tfa.get_strategy()
    with strategy.scope(), tfmot.quantization.keras.quantize_scope():
        logger.info("Loading trained model")
        model = tfa.load_model(params.model_file, custom_objects={"MultiF1Score": tfa.MultiF1Score})

        inputs = tf.keras.layers.Input(ds.feature_shape, dtype=tf.float32, batch_size=1)
        outputs = model(inputs)
        if not params.use_logits and not isinstance(model.layers[-1], tf.keras.layers.Softmax):
            outputs = tf.keras.layers.Softmax()(outputs)
            model = tf.keras.Model(inputs, outputs, name=model.name)
            outputs = model(inputs)
        # END IF
        flops = tfa.get_flops(model, batch_size=1, fpath=params.job_dir / "model_flops.log")
        model.summary(print_fn=logger.info)

        logger.info(f"Model requires {flops/1e6:0.2f} MFLOPS")

        logger.info(f"Converting model to TFLite (quantization={params.quantization})")
        model_func = tf.function(func=model)
        model_cf = model_func.get_concrete_function(
            tf.TensorSpec(shape=(1,)+ds.feature_shape, dtype=tf.float32
        ))

        if params.quantization:
            _, quant_df = tfa.debug_quant_tflite(
                model=model,
                test_x=test_x,
                input_type=tf.int8 if params.quantization else None,
                output_type=tf.int8 if params.quantization else None,
            )
            quant_df.to_csv(params.job_dir / "quant.csv")

        # Following is a workaround for bug (https://github.com/tensorflow/tflite-micro/issues/2319)
        # Default TFLiteConverter generates equivalent graph w/ SpaceToBatchND operations but losses dilation_rate factor.
        # Using concrete function instead of model object to avoid this issue.
        converter = tf.lite.TFLiteConverter.from_concrete_functions([model_cf], model)

        if params.quantization:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            if test_x is not None:
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
                def rep_dataset():
                    for i in range(test_x.shape[0]):
                        yield [test_x[i : i + 1]]
                converter.representative_dataset = rep_dataset
            # END IF
        tflite_model = converter.convert()

        # Save TFLite model
        logger.info(f"Saving TFLite model to {tfl_model_path}")
        with open(tfl_model_path, "wb") as fp:
            fp.write(tflite_model)

        # Save TFLM model
        logger.info(f"Saving TFL micro model to {tflm_model_path}")
        tfa.xxd_c_dump(
            src_path=tfl_model_path,
            dst_path=tflm_model_path,
            var_name=params.tflm_var_name,
            chunk_len=20,
            is_header=True,
        )
    # END WITH

    y_pred_tf = np.argmax(model.predict(test_x), axis=-1).flatten()

    # Verify TFLite results match TF results on example data
    logger.info("Validating model results")
    y_true = np.argmax(test_y, axis=-1).flatten()

    _, y_pred_tfl = tfa.predict_tflite(model_content=tflite_model, test_x=test_x)
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

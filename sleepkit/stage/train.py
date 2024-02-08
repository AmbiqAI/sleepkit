"""Sleep Stage Training"""

import logging
import os
import shutil

import keras
import numpy as np
import sklearn.metrics
import sklearn.model_selection
import sklearn.utils
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

from .. import tflite as tfa
from ..defines import SKTrainParams
from ..metrics import compute_iou, confusion_matrix_plot, f1_score
from ..utils import env_flag, set_random_seed, setup_logger
from .defines import get_stage_class_mapping, get_stage_class_names, get_stage_classes
from .utils import (
    create_model,
    load_dataset,
    load_train_dataset,
    load_validation_dataset,
)

logger = setup_logger(__name__)


def train(params: SKTrainParams):
    """Train sleep stage model.

    Args:
        params (SKTrainParams): Training parameters

    """

    # Custom parameters (add to SKTrainParams for automatic logging)
    params.lr_rate: float = getattr(params, "lr_rate", 1e-3)
    params.lr_cycles: int = getattr(params, "lr_cycles", 3)
    params.steps_per_epoch = params.steps_per_epoch or 100
    params.seed = set_random_seed(params.seed)

    logger.info(f"Creating working directory in {params.job_dir}")
    os.makedirs(params.job_dir, exist_ok=True)

    handler = logging.FileHandler(params.job_dir / "train.log", mode="w")
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    logger.info(f"Random seed {params.seed}")

    with open(params.job_dir / "train_config.json", "w", encoding="utf-8") as fp:
        fp.write(params.model_dump_json(indent=2))

    if env_flag("WANDB"):
        wandb.init(
            project=f"sk-stage-{params.num_classes}",
            entity="ambiq",
            dir=params.job_dir,
        )
        wandb.config.update(params.model_dump())

    target_classes = get_stage_classes(params.num_classes)
    class_names = get_stage_class_names(params.num_classes)
    class_mapping = get_stage_class_mapping(params.num_classes)

    ds = load_dataset(
        handler=params.ds_handler, ds_path=params.ds_path, frame_size=params.frame_size, params=params.ds_params
    )
    feat_shape = ds.feature_shape
    class_shape = (params.frame_size, len(target_classes))

    # Get train/val subject IDs and generators
    train_subject_ids, val_subject_ids = sklearn.model_selection.train_test_split(
        ds.train_subject_ids, test_size=params.val_subjects
    )
    logger.info("Loading training dataset")
    train_ds = load_train_dataset(
        ds=ds,
        subject_ids=train_subject_ids,
        samples_per_subject=params.samples_per_subject,
        buffer_size=params.buffer_size,
        batch_size=params.batch_size,
        feat_shape=feat_shape,
        class_shape=class_shape,
        class_map=class_mapping,
        num_workers=params.data_parallelism,
    )

    logger.info("Loading validation dataset")
    val_ds = load_validation_dataset(
        ds=ds,
        subject_ids=val_subject_ids,
        samples_per_subject=params.val_samples_per_subject,
        batch_size=params.batch_size,
        val_size=params.val_size,
        feat_shape=feat_shape,
        class_shape=class_shape,
        class_map=class_mapping,
    )
    test_labels = [y.numpy() for _, y in val_ds]
    y_true = np.argmax(np.concatenate(test_labels).squeeze(), axis=-1).flatten()
    class_weights = sklearn.utils.compute_class_weight("balanced", classes=np.array(target_classes), y=y_true)
    class_weights = (class_weights + class_weights.mean()) / 2

    strategy = tfa.get_strategy()
    with strategy.scope():
        logger.info("Building model")
        inputs = keras.Input(feat_shape, batch_size=None, dtype=tf.float32)
        model = create_model(inputs, num_classes=len(target_classes), name=params.model, params=params.model_params)
        flops = tfa.get_flops(model, batch_size=1, fpath=params.job_dir / "model_flops.log")

        if params.lr_cycles == 1:
            scheduler = keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=params.lr_rate,
                decay_steps=int(params.steps_per_epoch * params.epochs),
            )
        else:
            scheduler = keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=params.lr_rate,
                first_decay_steps=int(0.1 * params.steps_per_epoch * params.epochs),
                t_mul=1.65 / (0.1 * params.lr_cycles * (params.lr_cycles - 1)),
                m_mul=0.4,
            )
        optimizer = keras.optimizers.Adam(scheduler)
        loss = keras.losses.CategoricalFocalCrossentropy(
            from_logits=True,
            alpha=class_weights,
            label_smoothing=params.label_smoothing,
        )
        metrics = [
            keras.metrics.CategoricalAccuracy(name="acc"),
            tfa.MultiF1Score(name="f1", dtype=tf.float32, average="weighted"),
            keras.metrics.OneHotIoU(
                num_classes=len(target_classes),
                target_class_ids=target_classes,
                name="iou",
            ),
        ]

        if params.weights_file:
            logger.info(f"Loading weights from file {params.weights_file}")
            model.load_weights(params.weights_file)
        params.weights_file = params.job_dir / "model.weights"

        if params.quantization:
            logger.info("Performing QAT...")

            def apply_quantization_to_non_norm(layer):
                if not isinstance(layer, keras.layers.LayerNormalization):
                    return tfmot.quantization.keras.quantize_annotate_layer(layer)
                return layer

            # model = tfmot.quantization.keras.quantize_model(model)
            model = keras.models.clone_model(model, clone_function=apply_quantization_to_non_norm)
            model = tfmot.quantization.keras.quantize_apply(model)

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        model(inputs)
        model.summary(print_fn=logger.info)
        logger.info(f"Model requires {flops/1e6:0.2f} MFLOPS")

        # Remove existing TB logs
        if os.path.exists(params.job_dir / "logs"):
            shutil.rmtree(params.job_dir / "logs")

        ModelCheckpoint = keras.callbacks.ModelCheckpoint
        if env_flag("WANDB"):
            ModelCheckpoint = WandbModelCheckpoint
        model_callbacks = [
            keras.callbacks.EarlyStopping(
                monitor=f"val_{params.val_metric}",
                patience=max(int(0.25 * params.epochs), 1),
                mode="max" if params.val_metric == "f1" else "auto",
                restore_best_weights=True,
            ),
            ModelCheckpoint(
                filepath=params.weights_file,
                monitor=f"val_{params.val_metric}",
                save_best_only=True,
                save_weights_only=True,
                mode="max" if params.val_metric == "f1" else "auto",
                verbose=1,
            ),
            keras.callbacks.CSVLogger(params.job_dir / "history.csv"),
            keras.callbacks.TensorBoard(
                log_dir=params.job_dir / "logs",
                write_steps_per_second=True,
            ),
        ]
        if env_flag("WANDB"):
            model_callbacks.append(WandbMetricsLogger())

        try:
            model.fit(
                train_ds,
                steps_per_epoch=params.steps_per_epoch,
                verbose=2,
                epochs=params.epochs,
                validation_data=val_ds,
                callbacks=model_callbacks,
            )
        except KeyboardInterrupt:
            logger.warning("Stopping training due to keyboard interrupt")

        # Restore best weights from checkpoint
        model.load_weights(params.weights_file)

        # Save full model
        tf_model_path = params.job_dir / "model.tf"
        logger.info(f"Model saved to {tf_model_path}")
        model.save(tf_model_path)

        # Get full validation results
        logger.info("Performing full validation")
        y_pred = np.argmax(model.predict(val_ds).squeeze(), axis=-1).flatten()

        confusion_matrix_plot(
            y_true=y_true,
            y_pred=y_pred,
            labels=class_names,
            save_path=params.job_dir / "confusion_matrix.png",
            normalize="true",
        )
        if env_flag("WANDB"):
            conf_mat = wandb.plot.confusion_matrix(preds=y_pred, y_true=y_true, class_names=class_names)
            wandb.log({"conf_mat": conf_mat})
        # END IF

        # Summarize results
        test_acc = np.sum(y_pred == y_true) / y_true.size
        test_f1 = f1_score(y_true=y_true, y_pred=y_pred, average="weighted")
        test_iou = compute_iou(y_true, y_pred, average="weighted")
        logger.info(f"[TEST SET] ACC={test_acc:.2%}, F1={test_f1:.2%} IoU={test_iou:0.2%}")
    # END WITH

"""Sleep Apnea"""

import logging
import os
import shutil

import keras
import numpy as np
import sklearn.model_selection
import sklearn.utils
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import wandb
from rich.console import Console
from wandb.keras import WandbCallback

from ... import tflite as tfa
from ...defines import SKTrainParams
from ...metrics import compute_iou, confusion_matrix_plot, f1_score
from ...utils import env_flag, set_random_seed, setup_logger
from .utils import (
    create_model,
    load_dataset,
    load_train_dataset,
    load_validation_dataset,
)

console = Console()
logger = setup_logger(__name__)


def train(params: SKTrainParams):
    """Train sleep apnea model.

    Args:
        params (SKTrainParams): Training parameters
    """

    params.seed = set_random_seed(params.seed)
    logger.info(f"Random seed {params.seed}")

    os.makedirs(params.job_dir, exist_ok=True)
    logger.info(f"Creating working directory in {params.job_dir}")

    handler = logging.FileHandler(params.job_dir / "train.log", mode="w")
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    with open(params.job_dir / "train_config.json", "w", encoding="utf-8") as fp:
        fp.write(params.model_dump_json(indent=2))
    # END WITH

    if env_flag("WANDB"):
        wandb.init(
            project=f"sk-apnea-{params.num_classes}",
            entity="ambiq",
            dir=params.job_dir,
        )
        wandb.config.update(params.dict())
    # END IF

    target_classes = sorted(list(set(params.class_map.values())))
    class_names = params.class_names or [f"Class {i}" for i in range(params.num_classes)]

    ds = load_dataset(
        ds_path=params.ds_path,
        frame_size=params.frame_size,
        dataset=params.dataset,
    )

    feat_shape = ds.feature_shape
    class_shape = (params.frame_size, params.num_classes)

    input_spec = (
        tf.TensorSpec(shape=feat_shape, dtype=tf.float32),
        tf.TensorSpec(shape=class_shape, dtype=tf.int32),
    )

    # Get train/val subject IDs and generators
    train_subject_ids, val_subject_ids = sklearn.model_selection.train_test_split(
        ds.subject_ids, test_size=params.val_subjects
    )

    logger.info("Loading training dataset")
    train_ds = load_train_dataset(
        ds=ds,
        subject_ids=train_subject_ids,
        samples_per_subject=params.samples_per_subject,
        buffer_size=params.buffer_size,
        batch_size=params.batch_size,
        spec=input_spec,
        class_map=params.class_map,
        num_workers=params.data_parallelism,
    )

    logger.info("Loading validation dataset")
    val_ds = load_validation_dataset(
        ds=ds,
        subject_ids=val_subject_ids,
        samples_per_subject=params.val_samples_per_subject,
        batch_size=params.batch_size,
        val_size=params.val_size,
        spec=input_spec,
        class_map=params.class_map,
    )

    test_labels = [y.numpy() for _, y in val_ds]
    y_true = np.argmax(np.concatenate(test_labels), axis=-1).flatten()

    class_weights = 0.25
    if params.class_weights == "balanced":
        class_weights = sklearn.utils.compute_class_weight("balanced", classes=np.array(target_classes), y=y_true)
        class_weights = (class_weights + class_weights.mean()) / 2
    # END IF

    with tfa.get_strategy().scope():
        logger.info("Building model")
        inputs = keras.Input(shape=input_spec[0].shape, batch_size=None, name="input", dtype=input_spec[0].dtype)
        if params.resume and params.model_file:
            logger.info(f"Loading model from file {params.model_file}")
            model = tfa.load_model(params.model_file)
        else:
            logger.info("Creating model from scratch")
            model = create_model(inputs, num_classes=params.num_classes, architecture=params.architecture)
        # END IF

        flops = tfa.get_flops(model, batch_size=1, fpath=str(params.job_dir / "model_flops.log"))

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
            tfa.MultiF1Score(name="f1", dtype=tf.float32, average="macro"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.OneHotIoU(
                num_classes=len(target_classes),
                target_class_ids=target_classes,
                name="iou",
            ),
        ]

        if params.resume and params.weights_file:
            logger.info(f"Hydrating model weights from file {params.weights_file}")
            model.load_weights(params.weights_file)
        # END IF

        if params.model_file is None:
            params.model_file = params.job_dir / "model.keras"
        # END IF

        # Perform QAT if requested
        if params.quantization.enabled and params.quantization.qat:
            logger.info("Performing QAT")
            model = tfmot.quantization.keras.quantize_model(model, quantized_layer_name_prefix="q_")
        # END IF

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        model(inputs)
        model.summary(print_fn=logger.info)
        logger.info(f"Model requires {flops/1e6:0.2f} MFLOPS")

        # Remove existing logs
        if os.path.exists(params.job_dir / "logs"):
            shutil.rmtree(params.job_dir / "logs")

        ModelCheckpoint = keras.callbacks.ModelCheckpoint
        model_callbacks = [
            keras.callbacks.EarlyStopping(
                monitor=f"val_{params.val_metric}",
                patience=max(int(0.25 * params.epochs), 1),
                mode="max" if params.val_metric == "f1" else "auto",
                restore_best_weights=True,
            ),
            ModelCheckpoint(
                filepath=str(params.model_file),
                monitor=f"val_{params.val_metric}",
                save_best_only=True,
                mode="max" if params.val_metric == "f1" else "auto",
                verbose=1,
            ),
            keras.callbacks.CSVLogger(params.job_dir / "history.csv"),
        ]
        if env_flag("TENSORBOARD"):
            model_callbacks.append(
                keras.callbacks.TensorBoard(
                    log_dir=params.job_dir / "logs",
                    write_steps_per_second=True,
                )
            )
        if env_flag("WANDB"):
            model_callbacks.append(WandbCallback())

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

        logger.info(f"Model saved to {params.model_file}")

        # Get full validation results
        keras.models.load_model(params.model_file)
        logger.info("Performing full validation")
        y_pred = np.argmax(model.predict(val_ds).squeeze(), axis=-1).flatten()

        cm_path = params.job_dir / "confusion_matrix.png"
        confusion_matrix_plot(
            y_true=y_true,
            y_pred=y_pred,
            labels=class_names,
            save_path=cm_path,
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

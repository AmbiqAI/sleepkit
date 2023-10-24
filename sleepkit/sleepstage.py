"""Sleep Stage"""
import logging
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import sklearn.model_selection
import sklearn.utils
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import wandb
from rich.console import Console
from tqdm import tqdm
from wandb.keras import WandbCallback

from . import tflite as tfa
from .datasets import Hdf5Dataset, SKDataset
from .datasets.utils import create_dataset_from_data
from .defines import (
    SKExportParams,
    SKTestParams,
    SKTrainParams,
    get_sleep_stage_class_mapping,
    get_sleep_stage_class_names,
    get_sleep_stage_classes,
)
from .metrics import (
    compute_iou,
    compute_sleep_efficiency,
    compute_sleep_stage_durations,
    compute_total_sleep_time,
    confusion_matrix_plot,
    f1_score,
)
from .models import (
    EfficientNetParams,
    EfficientNetV2,
    UNet,
    UNetParams,
    UNext,
    UNextBlockParams,
    UNextParams,
)
from .utils import env_flag, set_random_seed, setup_logger

console = Console()
logger = setup_logger(__name__)


def create_model(
    inputs: tf.Tensor, num_classes: int, name: str | None = None, params: dict[str, Any] | None = None
) -> tf.keras.Model:
    """Generate model or use default
    Args:
        inputs (tf.Tensor): Model inputs
        num_classes (int): Number of classes
        name (str | None, optional): Architecture type. Defaults to None.
        params (dict[str, Any] | None, optional): Model parameters. Defaults to None.
    Returns:
        tf.keras.Model: Model
    """
    if name and params is None:
        raise ValueError("Model parameters must be provided if model name is provided")

    if name == "unet":
        return UNet(x=inputs, params=UNetParams.parse_obj(params), num_classes=num_classes)

    if name == "unext":
        return UNext(x=inputs, params=UNextParams.parse_obj(params), num_classes=num_classes)

    if name == "efficientnetv2":
        return EfficientNetV2(x=inputs, params=EfficientNetParams.parse_obj(params), num_classes=num_classes)

    if name == "lstm":
        y = inputs
        y = tf.keras.layers.Reshape((1,) + y.shape[1:])(y)
        y = tf.keras.layers.DepthwiseConv2D(kernel_size=(1, 5), padding="same")(y)
        y = tf.keras.layers.LayerNormalization(axis=[2])(y)
        y = tf.keras.layers.Activation("relu6")(y)

        y = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 5), padding="same")(y)
        y = tf.keras.layers.LayerNormalization(axis=[2])(y)
        y = tf.keras.layers.Activation("relu6")(y)

        y = tf.keras.layers.Reshape(y.shape[2:])(y)

        y = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=40, return_sequences=True))(y)
        y = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=48, return_sequences=True))(y)
        y = tf.keras.layers.LayerNormalization(axis=[1])(y)

        y = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(56))(y)
        y = tf.keras.layers.LayerNormalization(axis=[1])(y)
        y = tf.keras.layers.Activation("relu6")(y)

        y = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_classes))(y)
        model = tf.keras.models.Model(inputs, y)
        return model

    if name:
        raise ValueError(f"No network architecture with name {name}")

    # Otherwise, use default
    return UNext(
        x=inputs,
        params=UNextParams(
            blocks=[
                UNextBlockParams(
                    filters=24, depth=2, kernel=5, pool=2, strides=2, skip=True, expand_ratio=1, se_ratio=1, dropout=0.1
                ),
                UNextBlockParams(
                    filters=32, depth=2, kernel=5, pool=2, strides=2, skip=True, expand_ratio=1, se_ratio=2, dropout=0.1
                ),
                UNextBlockParams(
                    filters=48, depth=2, kernel=5, pool=2, strides=2, skip=True, expand_ratio=1, se_ratio=2, dropout=0.1
                ),
            ],
            output_kernel_size=5,
            include_top=True,
            use_logits=False,
        ),
        num_classes=num_classes,
    )


def prepare(x: tf.Tensor, y: tf.Tensor, num_classes: int, class_map: dict[int, int]) -> tuple[tf.Tensor, tf.Tensor]:
    """Prepare data for training
    Args:
        x (tf.Tensor): Features
        y (tf.Tensor): Labels
        num_classes (int): Number of classes
        class_map (dict[int, int]): Class mapping
    Returns:
        tuple[tf.Tensor, tf.Tensor]: Features and labels
    """
    return (
        x,
        # tf.one_hot(class_map.get(sts.mode(y[-5:]).mode, 0), num_classes)
        tf.one_hot(np.vectorize(class_map.get)(y), num_classes),
    )


def load_dataset(handler: str, ds_path: Path, frame_size: int, params: dict[str, Any]) -> SKDataset:
    """Load dataset(s)
    Args:
        handler (str): Dataset handler
        ds_path (Path): Dataset path
        frame_size (int): Frame size
        params (dict[str, Any]): Dataset arguments
    Returns:
        SKDataset: Dataset
    """
    if handler == "hdf5":
        return Hdf5Dataset(ds_path=ds_path, frame_size=frame_size, **params)
    raise ValueError(f"Unknown dataset handler {handler}")


def load_train_dataset(
    ds: SKDataset,
    subject_ids: list[str],
    samples_per_subject: int,
    buffer_size: int,
    batch_size: int,
    feat_shape: tuple[int, ...],
    class_shape: tuple[int, ...],
    class_map: dict[int, int],
    num_workers: int = 4,
) -> tf.data.Dataset:
    """Load train dataset
    Args:
        ds (SKDataset): Dataset
        subject_ids (list[str]): Subject IDs
        samples_per_subject (int): Samples per subject
        buffer_size (int): Buffer size
        batch_size (int): Batch size
        feat_shape (tuple[int,...]): Feature shape
        class_shape (tuple[int,...]): Class shape
        class_map (dict[int, int]): Class mapping
        num_workers (int, optional): Number of workers. Defaults to 4.
    Returns:
        tf.data.Dataset: Train dataset
    """

    def preprocess(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Preprocess data"""
        xx = x.copy()
        xx = xx + np.random.normal(0, 0.1, size=x.shape)
        # if np.random.rand() < 0.2:
            # xx = np.flip(xx, axis=0)
        return xx

    def train_generator(subject_ids):
        """Train generator per worker"""

        def ds_gen():
            """Worker generator routine"""
            train_subj_gen = ds.uniform_subject_generator(subject_ids)
            return map(
                lambda x_y: prepare(preprocess(x_y[0]), x_y[1], class_shape[-1], class_map),
                ds.signal_generator(train_subj_gen, samples_per_subject=samples_per_subject, normalize=True),
            )

        return tf.data.Dataset.from_generator(
            ds_gen,
            output_signature=(
                tf.TensorSpec(shape=feat_shape, dtype=tf.float32),
                tf.TensorSpec(shape=class_shape, dtype=tf.int32),
            ),
        )

    split = len(subject_ids) // num_workers
    train_datasets = [train_generator(subject_ids[i * split : (i + 1) * split]) for i in range(num_workers)]

    # Create TF datasets (interleave workers)
    train_ds = (
        tf.data.Dataset.from_tensor_slices(train_datasets)
        .interleave(
            lambda x: x,
            cycle_length=num_workers,
            deterministic=False,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .shuffle(
            buffer_size=buffer_size,
            reshuffle_each_iteration=True,
        )
        .batch(
            batch_size=batch_size,
            drop_remainder=False,
        )
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    return train_ds


def load_validation_dataset(
    ds: SKDataset,
    subject_ids: list[str],
    samples_per_subject: int,
    batch_size: int,
    val_size: int,
    feat_shape: tuple[int, ...],
    class_shape: tuple[int, ...],
    class_map: dict[int, int],
) -> tf.data.Dataset:
    """Load validation dataset.
    Args:
        ds (SKDataset): Dataset
        subject_ids (list[str]): Subject IDs
        samples_per_subject (int): Samples per subject
        batch_size (int): Batch size
        val_size (int): Validation size
        feat_shape (tuple[int,...]): Feature shape
        class_shape (tuple[int,...]): Class shape
        class_map (dict[int, int]): Class mapping
    Returns:
        tf.data.Dataset: Validation dataset
    """

    def preprocess(x: npt.NDArray[np.float32]):
        return x

    output_signature = (
        tf.TensorSpec(shape=feat_shape, dtype=tf.float32),
        tf.TensorSpec(shape=class_shape, dtype=tf.int32),
    )

    def val_generator():
        val_subj_gen = ds.uniform_subject_generator(subject_ids)
        return map(
            lambda x_y: prepare(preprocess(x_y[0]), x_y[1], class_shape[-1], class_map),
            ds.signal_generator(val_subj_gen, samples_per_subject=samples_per_subject, normalize=True),
        )

    val_ds = tf.data.Dataset.from_generator(generator=val_generator, output_signature=output_signature)
    val_x, val_y = next(val_ds.batch(val_size).as_numpy_iterator())
    val_ds = create_dataset_from_data(val_x, val_y, output_signature=output_signature).batch(
        batch_size=batch_size,
        drop_remainder=False,
    )

    return val_ds


def load_test_dataset(
    ds: SKDataset,
    subject_ids: list[str],
    samples_per_subject: int,
    test_size: int,
    feat_shape: tuple[int, ...],
    class_shape: tuple[int, ...],
    class_map: dict[int, int],
) -> tuple[npt.NDArray, npt.NDArray]:
    """Load test dataset
    Args:
        params (SKTestParams): Testing parameters
    Returns:
        tuple[npt.NDArray, npt.NDArray]: Test features and labels
    """
    def preprocess(x: npt.NDArray[np.float32]):
        return x

    output_signature = (
        tf.TensorSpec(shape=feat_shape, dtype=tf.float32),
        tf.TensorSpec(shape=class_shape, dtype=tf.int32),
    )

    def test_generator():
        test_subj_gen = ds.uniform_subject_generator(subject_ids)
        return map(
            lambda x_y: prepare(preprocess(x_y[0]), x_y[1], class_shape[-1], class_map),
            ds.signal_generator(test_subj_gen, samples_per_subject=samples_per_subject, normalize=True),
        )

    test_ds = tf.data.Dataset.from_generator(generator=test_generator, output_signature=output_signature)
    test_x, test_y = next(test_ds.batch(test_size).as_numpy_iterator())

    return test_x, test_y


def train(params: SKTrainParams):
    """Train sleep stage model.

    Args:
        params (SKTrainParams): Training parameters
    """

    # Custom parameters (add to SKTrainParams for automatic logging)
    params.num_sleep_stages = getattr(params, "num_sleep_stages", 3)
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
        fp.write(params.json(indent=2))

    if env_flag("WANDB"):
        wandb.init(
            project=f"sk-stage-{params.num_sleep_stages}",
            entity="ambiq",
            dir=params.job_dir,
        )
        wandb.config.update(params.dict())

    target_classes = get_sleep_stage_classes(params.num_sleep_stages)
    class_names = get_sleep_stage_class_names(params.num_sleep_stages)
    class_mapping = get_sleep_stage_class_mapping(params.num_sleep_stages)

    ds = load_dataset(
        handler=params.ds_handler, ds_path=params.ds_path, frame_size=params.frame_size, params=params.ds_params
    )
    feat_shape = ds.feature_shape
    class_shape = (params.frame_size, len(target_classes))

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
    class_weights = sklearn.utils.compute_class_weight("balanced", classes=target_classes, y=y_true)
    class_weights = (class_weights + class_weights.mean()) / 2
    # class_weights = 0.25

    strategy = tfa.get_strategy()
    with strategy.scope():
        logger.info("Building model")
        inputs = tf.keras.Input(feat_shape, batch_size=None, dtype=tf.float32)
        model = create_model(inputs, num_classes=len(target_classes), name=params.model, params=params.model_params)
        flops = tfa.get_flops(model, batch_size=1, fpath=params.job_dir / "model_flops.log")

        if params.lr_cycles == 1:
            scheduler = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=params.lr_rate,
                decay_steps=int(params.steps_per_epoch * params.epochs),
            )
        else:
            scheduler = tf.keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=params.lr_rate,
                first_decay_steps=int(0.1 * params.steps_per_epoch * params.epochs),
                t_mul=1.65 / (0.1 * params.lr_cycles * (params.lr_cycles - 1)),
                m_mul=0.4,
            )
        optimizer = tf.keras.optimizers.Adam(scheduler)
        loss = tf.keras.losses.CategoricalFocalCrossentropy(
            from_logits=True,
            alpha=class_weights,
            label_smoothing=params.label_smoothing,
        )
        metrics = [
            tf.keras.metrics.CategoricalAccuracy(name="acc"),
            tfa.MultiF1Score(name="f1", dtype=tf.float32, average="weighted"),
            tf.keras.metrics.OneHotIoU(
                num_classes=len(target_classes),
                target_class_ids=target_classes,
                name="iou",
            ),
        ]

        if params.weights_file:
            logger.info(f"Loading weights from file {params.weights_file}")
            model.load_weights(params.weights_file)
        params.weights_file = params.job_dir / "model.weights"
        tfmot.quantization.keras.QuantizeConfig
        if params.quantization:
            logger.info("Performing QAT...")

            def apply_quantization_to_non_norm(layer):
                if not isinstance(layer, tf.keras.layers.LayerNormalization):
                    return tfmot.quantization.keras.quantize_annotate_layer(layer)
                return layer
            # model = tfmot.quantization.keras.quantize_model(model)
            model = tf.keras.models.clone_model(model, clone_function=apply_quantization_to_non_norm)
            model = tfmot.quantization.keras.quantize_apply(model)

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        model(inputs)
        model.summary(print_fn=logger.info)
        logger.info(f"Model requires {flops/1e6:0.2f} MFLOPS")

        # Remove existing TB logs
        if os.path.exists(params.job_dir / "logs"):
            shutil.rmtree(params.job_dir / "logs")

        model_callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor=f"val_{params.val_metric}",
                patience=max(int(0.25 * params.epochs), 1),
                mode="max" if params.val_metric == "f1" else "auto",
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=params.weights_file,
                monitor=f"val_{params.val_metric}",
                save_best_only=True,
                save_weights_only=True,
                mode="max" if params.val_metric == "f1" else "auto",
                verbose=1,
            ),
            tf.keras.callbacks.CSVLogger(params.job_dir / "history.csv"),
            tf.keras.callbacks.TensorBoard(
                log_dir=params.job_dir / "logs",
                write_steps_per_second=True,
            ),
        ]
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


def evaluate(params: SKTestParams):
    """Evaluate sleep stage model.
    Args:
        params (SKTestParams): Testing/evaluation parameters
    """
    params.num_sleep_stages = getattr(params, "num_sleep_stages", 3)

    logger.info(f"Creating working directory in {params.job_dir}")
    os.makedirs(params.job_dir, exist_ok=True)

    handler = logging.FileHandler(params.job_dir / "test.log", mode="w")
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    params.seed = set_random_seed(params.seed)
    logger.info(f"Random seed {params.seed}")

    class_names = get_sleep_stage_class_names(params.num_sleep_stages)
    class_mapping = get_sleep_stage_class_mapping(params.num_sleep_stages)

    ds = load_dataset(
        handler=params.ds_handler, ds_path=params.ds_path, frame_size=params.frame_size, params=params.ds_params
    )
    feat_shape = ds.feature_shape
    test_true, test_pred = [], []
    pt_metrics = []

    strategy = tfa.get_strategy()
    with strategy.scope(), tfmot.quantization.keras.quantize_scope():
        logger.info("Loading model")
        model = tfa.load_model(params.model_file, custom_objects={"MultiF1Score": tfa.MultiF1Score})
        flops = tfa.get_flops(model, batch_size=1, fpath=params.job_dir / "model_flops.log")
        model.summary(print_fn=logger.info)
        logger.info(f"Model requires {flops/1e6:0.2f} MFLOPS")

        logger.info("Performing inference")
        for subject_id in tqdm(ds.test_subject_ids, desc="Subject"):
            features, labels, mask = ds.load_subject_data(subject_id=subject_id, normalize=True)
            num_windows = int(features.shape[0] // params.frame_size)
            data_len = params.frame_size * num_windows

            x = features[:data_len, :].reshape((num_windows, params.frame_size) + feat_shape[1:])
            m = mask[:data_len].reshape((num_windows, params.frame_size))
            # m[:, :64] = 0 # Ignore first N samples
            y_prob = tf.nn.softmax(model.predict(x, verbose=0)).numpy()
            y_pred = np.argmax(y_prob, axis=-1).flatten()
            # y_mask = mask[:data_len].flatten()
            y_mask = m.flatten()
            y_true = np.vectorize(class_mapping.get)(labels[:data_len].flatten())
            y_pred = y_pred[y_mask == 1]
            y_true = y_true[y_mask == 1]

            # Get subject specific metrics
            pred_sleep_durations = compute_sleep_stage_durations(y_pred)
            pred_sleep_tst = compute_total_sleep_time(pred_sleep_durations, class_mapping)
            pred_sleep_eff = compute_sleep_efficiency(pred_sleep_durations, class_mapping)
            act_sleep_duration = compute_sleep_stage_durations(y_true)
            act_sleep_tst = compute_total_sleep_time(act_sleep_duration, class_mapping)
            act_sleep_eff = compute_sleep_efficiency(act_sleep_duration, class_mapping)
            pt_acc = np.sum(y_pred == y_true) / y_true.size
            pt_metrics.append([pt_acc, act_sleep_eff, pred_sleep_eff, act_sleep_tst, pred_sleep_tst])
            test_true.append(y_true)
            test_pred.append(y_pred)
        # END FOR

        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)

        df_metrics = pd.DataFrame(pt_metrics, columns=["acc", "act_eff", "pred_eff", "act_tst", "pred_tst"])
        df_metrics.to_csv(params.job_dir / "metrics.csv", header=True, index=False)

        confusion_matrix_plot(
            y_true=test_true,
            y_pred=test_pred,
            labels=class_names,
            save_path=params.job_dir / "confusion_matrix_test.png",
            normalize="true",
        )

        # Summarize results
        logger.info("Testing Results")
        test_acc = np.sum(test_pred == test_true) / test_true.size
        test_f1 = f1_score(y_true=test_true, y_pred=test_pred, average="weighted")
        test_iou = compute_iou(test_true, test_pred, average="weighted")
        logger.info(f"[TEST SET] ACC={test_acc:.2%}, F1={test_f1:.2%} IoU={test_iou:0.2%}")
    # END WITH


def export(params: SKExportParams):
    """Export sleep stage model.

    Args:
        params (SKExportParams): Deployment parameters
    """
    params.num_sleep_stages = getattr(params, "num_sleep_stages", 3)

    target_classes = get_sleep_stage_classes(params.num_sleep_stages)
    class_mapping = get_sleep_stage_class_mapping(params.num_sleep_stages)

    tfl_model_path = params.job_dir / "model.tflite"
    tflm_model_path = params.job_dir / "model_buffer.h"

    # class_names = get_sleep_stage_class_names(params.num_sleep_stages)
    # class_mapping = get_sleep_stage_class_mapping(params.num_sleep_stages)

    ds = load_dataset(
        handler=params.ds_handler, ds_path=params.ds_path, frame_size=params.frame_size, params=params.ds_params
    )
    feat_shape = ds.feature_shape
    class_shape = (params.frame_size, len(target_classes))

    test_x, test_y = load_test_dataset(
        ds=ds,
        subject_ids=ds.test_subject_ids,
        samples_per_subject=params.samples_per_subject,
        test_size=params.test_size,
        feat_shape=feat_shape,
        class_shape=class_shape,
        class_map=class_mapping,
    )

    # Load model and set fixed batch size of 1
    strategy = tfa.get_strategy()
    with strategy.scope(), tfmot.quantization.keras.quantize_scope():
        logger.info("Loading trained model")
        model = tfa.load_model(params.model_file, custom_objects={"MultiF1Score": tfa.MultiF1Score})

        inputs = tf.keras.layers.Input(feat_shape, dtype=tf.float32, batch_size=1)
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

        # if params.quantization:
        #     _, quant_df = tfa.debug_quant_tflite(
        #         model=model,
        #         test_x=test_x,
        #         input_type=tf.int8 if params.quantization else None,
        #         output_type=tf.int8 if params.quantization else None,
        #     )
        #     quant_df.to_csv(params.job_dir / "quant.csv")

        tflite_model = tfa.convert_tflite(
            model=model,
            quantize=params.quantization,
            test_x=test_x,
            input_type=tf.int8 if params.quantization else None,
            output_type=tf.int8 if params.quantization else None,
        )

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

    y_pred_tfl = np.argmax(tfa.predict_tflite(model_content=tflite_model, test_x=test_x), axis=-1).flatten()

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

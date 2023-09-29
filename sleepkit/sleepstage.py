"""Sleep Stage"""
import os
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import tensorflow as tf
import sklearn.model_selection
import sklearn.utils
import wandb
from rich.console import Console
from wandb.keras import WandbCallback

from neuralspot.tflite.metrics import get_flops, MultiF1Score
from neuralspot.tflite.model import get_strategy, load_model

from .defines import (
    SKTrainParams,
    SKTestParams,
    get_sleep_stage_classes,
    get_sleep_stage_class_names,
    get_sleep_stage_class_mapping
)
from .utils import env_flag, set_random_seed, setup_logger
from .datasets import Hdf5Dataset
from .datasets.utils import create_dataset_from_data
from .models import UNet, UNetParams, UNext, UNextBlockParams, UNextParams
from .metrics import compute_iou, confusion_matrix_plot, f1_score

console = Console()
logger = setup_logger(__name__)

def load_model(inputs: tf.Tensor, num_classes: int, name: str | None = None, params: dict[str, Any] | None = None) -> tf.keras.Model:
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

    if name:
        raise ValueError(f"No network architecture with name {name}")

    # Otherwise, use default
    return UNext(
        x=inputs,
        params=UNextParams(
            blocks=[
                UNextBlockParams(filters=24, depth=2, kernel=5, pool=2, strides=2, skip=True, expand_ratio=1, se_ratio=2, dropout=0),
                UNextBlockParams(filters=32, depth=2, kernel=5, pool=2, strides=2, skip=True, expand_ratio=1, se_ratio=2, dropout=0),
                UNextBlockParams(filters=48, depth=2, kernel=5, pool=2, strides=2, skip=True, expand_ratio=1, se_ratio=2, dropout=0),
            ],
            output_kernel_size=5,
            include_top=True,
            use_logits=False,
        ),
        num_classes=num_classes
    )


def prepare(x: tf.Tensor, y: tf.Tensor, num_classes: int, class_map: dict[int, int]) -> tuple[tf.Tensor, tf.Tensor]:
    """Prepare data for training
    Args:
        x (tf.Tensor): Features
        y (tf.Tensor): Labels
        num_classes (int): Number of classes
        class_map (dict[int, int]): Class mapping
    """
    return (
        x,
        #tf.one_hot(class_map.get(sts.mode(y[-5:]).mode, 0), num_classes)
        tf.one_hot(np.vectorize(class_map.get)(y), num_classes)
    )

def load_dataset(ds_path: Path, frame_size: int, feat_cols: list[int] | None = None) -> Hdf5Dataset:
    """Load dataset(s)
    Args:
        ds_path (Path): Dataset path
        frame_size (int): Frame size
        feat_cols (list[int] | None, optional): Feature columns. Defaults to None.
    Returns:
        Hdf5Dataset: Dataset
    """
    ds = Hdf5Dataset(
        ds_path=ds_path,
        frame_size=frame_size,
        mask_key="mask",
        feat_cols=feat_cols
    )
    return ds

def load_train_dataset(
        ds: Hdf5Dataset,
        subject_ids,
        samples_per_subject: int,
        buffer_size: int,
        batch_size: int,
        feat_shape: tuple[int,...],
        class_shape: tuple[int, ...],
        class_map: dict[int, int],
        num_workers: int = 4
    ) -> tf.data.Dataset:
    """Load train dataset
    Args:
        ds (Hdf5Dataset): Dataset
        subject_ids ([type]): Subject IDs
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

    def preprocess(x: npt.NDArray[np.float32]):
        """Preprocess data"""
        return x + np.random.normal(0, 0.1, size=x.shape)

    def train_generator(subject_ids):
        def ds_gen():
            train_subj_gen = ds.uniform_subject_generator(subject_ids)
            return map(
                lambda x_y: prepare(preprocess(x_y[0]), x_y[1], class_shape[-1], class_map),
                ds.signal_generator(
                    train_subj_gen,
                    samples_per_subject=samples_per_subject
                )
            )
        return tf.data.Dataset.from_generator(
            ds_gen,
            output_signature=(
                tf.TensorSpec(shape=feat_shape, dtype=tf.float32),
                tf.TensorSpec(shape=class_shape, dtype=tf.int32),
            ),
        )

    split = len(subject_ids) // num_workers
    train_datasets = [train_generator(
        subject_ids[i * split : (i + 1) * split]
    ) for i in range(num_workers)]

    # Create TF datasets
    train_ds = tf.data.Dataset.from_tensor_slices(
        train_datasets
    ).interleave(
        lambda x: x,
        cycle_length=num_workers,
        deterministic=False,
        num_parallel_calls=tf.data.AUTOTUNE,
    ).shuffle(
        buffer_size=buffer_size,
        reshuffle_each_iteration=True,
    ).batch(
        batch_size=batch_size,
        drop_remainder=False,
    ).prefetch(
        buffer_size=tf.data.AUTOTUNE
    )

    return train_ds

def load_validation_dataset(
        ds: Hdf5Dataset,
        subject_ids: list[str],
        samples_per_subject: int,
        batch_size: int,
        val_size: int,
        feat_shape: tuple[int, ...],
        class_shape: tuple[int, ...],
        class_map: dict[int, int]
    ) -> tf.data.Dataset:
    """Load validation dataset.
    Args:
        ds (Hdf5Dataset): Dataset
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
            ds.signal_generator(
                val_subj_gen,
                samples_per_subject=samples_per_subject
            )
        )

    val_ds = tf.data.Dataset.from_generator(
        generator=val_generator,
        output_signature=output_signature
    )
    val_x, val_y = next(val_ds.batch(val_size).as_numpy_iterator())
    val_ds = create_dataset_from_data(
        val_x, val_y, output_signature=output_signature
    ).batch(
        batch_size=batch_size,
        drop_remainder=False,
    )

    return val_ds

def load_test_dataset(params: SKTestParams) -> tuple[npt.NDArray, npt.NDArray]:
    """Load test dataset
    Args:
        params (SKTestParams): Testing parameters
    Returns:
        tuple[npt.NDArray, npt.NDArray]: Test features and labels
    """
    ds = Hdf5Dataset(
        ds_path=params.ds_path,
        frame_size=params.frame_size,
        mask_key="mask",
    )
    test_x = []
    test_y = []
    for subject_id in ds.test_subject_ids:
        x, y = ds.load_subject_data(subject_id)
        test_x.append(x)
        test_y.append(y)

    return test_x, test_y

def train(params: SKTrainParams):
    """Train sleep stage model.

    Args:
        params (SKTrainParams): Training parameters
    """

    # Custom parameters (add to SKTrainParams for automatic logging)
    params.num_sleep_stages = getattr(params, "num_sleep_stages", 3)
    params.feat_cols = getattr(params, "feat_cols", None)
    params.lr_rate: float = getattr(params, "lr_rate", 1e-3)
    params.lr_cycles: int = getattr(params, "lr_cycles", 3)
    params.steps_per_epoch = params.steps_per_epoch or 1000
    params.seed = set_random_seed(params.seed)

    logger.info(f"Random seed {params.seed}")

    os.makedirs(str(params.job_dir), exist_ok=True)
    logger.info(f"Creating working directory in {params.job_dir}")
    with open(str(params.job_dir / "train_config.json"), "w", encoding="utf-8") as fp:
        fp.write(params.json(indent=2))


    if env_flag("WANDB"):
        wandb.init(
            project=f"sk-stage-{params.num_sleep_stages}",
            entity="ambiq",
            dir=params.job_dir,
        )
        wandb.config.update(params.dict())

    logger.info("Loading dataset(s)")

    target_classes = get_sleep_stage_classes(params.num_sleep_stages)
    class_names = get_sleep_stage_class_names(params.num_sleep_stages)
    class_mapping = get_sleep_stage_class_mapping(params.num_sleep_stages)

    ds = load_dataset(ds_path=params.ds_path, frame_size=params.frame_size, feat_cols=params.feat_cols)
    feat_shape = ds.feature_shape
    class_shape = (ds.frame_size, len(target_classes))

    # Get train/val subject IDs and generators
    train_subject_ids, val_subject_ids = sklearn.model_selection.train_test_split(
        ds.subject_ids, test_size=params.val_subjects
    )

    train_ds = load_train_dataset(
        ds=ds,
        subject_ids=train_subject_ids,
        samples_per_subject=params.samples_per_subject,
        buffer_size=params.buffer_size,
        batch_size=params.batch_size,
        feat_shape=feat_shape,
        class_shape=class_shape,
        class_map=class_mapping,
        num_workers=params.data_parallelism
    )

    val_ds = load_validation_dataset(
        ds=ds,
        subject_ids=val_subject_ids,
        samples_per_subject=params.val_samples_per_subject,
        batch_size=params.batch_size,
        val_size=params.val_size,
        feat_shape=feat_shape,
        class_shape=class_shape,
        class_map=class_mapping
    )

    test_labels = [y.numpy() for _, y in val_ds]
    y_true = np.argmax(np.concatenate(test_labels).squeeze(), axis=-1).flatten()
    class_weights = sklearn.utils.compute_class_weight(
        "balanced",
        classes=target_classes,
        y=y_true
    )

    strategy = get_strategy()
    with strategy.scope():
        logger.info("Building model")
        inputs = tf.keras.Input(feat_shape, batch_size=None, dtype=tf.float32)
        model = load_model(inputs, num_classes=len(target_classes), name=params.model, params=params.model_params)
        flops = get_flops(model, batch_size=1, fpath=str(params.job_dir / "model_flops.log"))

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
            alpha=class_weights
        )
        metrics = [
            tf.keras.metrics.CategoricalAccuracy(name="acc"),
            MultiF1Score(name="f1", dtype=tf.float32, average="weighted"),
            tf.keras.metrics.OneHotIoU(
                num_classes=len(target_classes),
                target_class_ids=target_classes,
                name="iou",
            ),
        ]
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        model(inputs)

        model.summary(print_fn=logger.info)
        logger.info(f"Model requires {flops/1e6:0.2f} MFLOPS")

        if params.weights_file:
            logger.info(f"Loading weights from file {params.weights_file}")
            model.load_weights(str(params.weights_file))
        params.weights_file = str(params.job_dir / "model.weights")

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
            tf.keras.callbacks.CSVLogger(str(params.job_dir / "history.csv")),
            tf.keras.callbacks.TensorBoard(
                log_dir=str(params.job_dir / "logs"),
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
        tf_model_path = str(params.job_dir / "model.tf")
        logger.info(f"Model saved to {tf_model_path}")
        model.save(tf_model_path)

        # Get full validation results
        logger.info("Performing full validation")
        y_pred = np.argmax(model.predict(val_ds).squeeze(), axis=-1).flatten()

        cm_path = str(params.job_dir / "confusion_matrix_test.png")
        confusion_matrix_plot(
            y_true=y_true,
            y_pred=y_pred,
            labels=class_names,
            save_path=cm_path,
            normalize="true",
        )
        if env_flag("WANDB"):
            conf_mat = wandb.plot.confusion_matrix(
                preds=y_pred,
                y_true=y_true,
                class_names=class_names
            )
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
    params.seed = set_random_seed(params.seed)
    logger.info(f"Random seed {params.seed}")

    test_x, test_y = load_test_dataset(params)

    strategy = get_strategy()
    with strategy.scope():
        logger.info("Loading model")
        model = load_model(str(params.model_file))
        flops = get_flops(model, batch_size=1, fpath=str(params.job_dir / "model_flops.log"))
        model.summary(print_fn=logger.info)
        logger.info(f"Model requires {flops/1e6:0.2f} MFLOPS")

        logger.info("Performing inference")
        y_true = np.argmax(test_y, axis=1)
        y_prob = tf.nn.softmax(model.predict(test_x)).numpy()
        y_pred = np.argmax(y_prob, axis=1)

        # Summarize results
        logger.info("Testing Results")
        test_acc = np.sum(y_pred == y_true) / len(y_true)
        test_f1 = f1_score(y_true, y_pred, average="macro")
        logger.info(f"[TEST SET] ACC={test_acc:.2%}, F1={test_f1:.2%}")
        class_names = get_class_names(HeartTask.arrhythmia)
        if len(class_names) == 2:
            roc_path = str(params.job_dir / "roc_auc_test.png")
            roc_auc_plot(y_true, y_prob[:, 1], labels=class_names, save_path=roc_path)
        # END IF

        # If threshold given, only count predictions above threshold
        if params.threshold:
            numel = len(y_true)
            y_prob, y_pred, y_true = threshold_predictions(y_prob, y_pred, y_true, params.threshold)
            drop_perc = 1 - len(y_true) / numel
            test_acc = np.sum(y_pred == y_true) / len(y_true)
            test_f1 = f1_score(y_true, y_pred, average="macro")
            logger.info(f"[TEST SET] THRESH={params.threshold:0.2%}, DROP={drop_perc:.2%}")
            logger.info(f"[TEST SET] ACC={test_acc:.2%}, F1={test_f1:.2%}")
        # END IF
        cm_path = str(params.job_dir / "confusion_matrix_test.png")
        confusion_matrix_plot(y_true, y_pred, labels=class_names, save_path=cm_path, normalize="true")
    # END WITH

"""Sleep Stage"""
import os

import numpy as np
from multiprocessing import Pool
import numpy.typing as npt
import physiokit as pk
import tensorflow as tf
import sklearn.model_selection
import wandb
from rich.console import Console
from wandb.keras import WandbCallback

from neuralspot.tflite.metrics import get_flops, MultiF1Score
from neuralspot.tflite.model import get_strategy, load_model

from .defines import SKTrainParams, SleepStage
from .utils import env_flag, set_random_seed, setup_logger
from .datasets import Hdf5Dataset
from .datasets.utils import create_dataset_from_data
from .models import UNet, UNetParams, UNetBlockParams
from .metrics import compute_iou, confusion_matrix_plot

console = Console()
logger = setup_logger(__name__)

def get_target_classes(nstages: int) -> list[int]:
    """Get target classes for sleep stage classification
    Args:
        nstages (int): Number of sleep stages
    Returns:
        list[int]: Target classes
    """
    if 2 <= nstages <= 5:
        return list(range(nstages))
    raise ValueError(f"Invalid number of stages: {nstages}")

def get_class_mapping(nstages: int) -> dict[int, int]:
    """Get class mapping for sleep stage classification
    Args:
        nstages (int): Number of sleep stages
    Returns:
        dict[int, int]: Class mapping
    """
    if nstages == 2:
        return {
            SleepStage.wake: 0,
            SleepStage.stage1: 1,
            SleepStage.stage2: 1,
            SleepStage.stage3: 1,
            SleepStage.stage4: 1,
            SleepStage.rem: 1,
        }
    if nstages == 3:
        return {
            SleepStage.wake: 0,
            SleepStage.stage1: 1,
            SleepStage.stage2: 1,
            SleepStage.stage3: 1,
            SleepStage.stage4: 1,
            SleepStage.rem: 2,
        }
    if nstages == 4:
        return {
            SleepStage.wake: 0,
            SleepStage.stage1: 1,
            SleepStage.stage2: 1,
            SleepStage.stage3: 2,
            SleepStage.stage4: 2,
            SleepStage.rem: 3,
        }
    if nstages == 5:
        return {
            SleepStage.wake: 0,
            SleepStage.stage1: 1,
            SleepStage.stage2: 2,
            SleepStage.stage3: 3,
            SleepStage.stage4: 3,
            SleepStage.rem: 4,
        }
    raise ValueError(f"Invalid number of stages: {nstages}")


def get_class_names(nstages: int):
    """Get class names for sleep stage classification
    Args:
        nstages (int): Number of sleep stages
    Returns:
        list[str]: Class names
    """
    if nstages == 2:
        return ["WAKE", "SLEEP"]
    if nstages == 3:
        return ["WAKE", "NREM", "REM"]
    if nstages == 4:
        return ["WAKE", "CORE", "DEEP", "REM"]
    if nstages == 5:
        return ["WAKE", "N1", "N2", "N3", "REM"]
    raise ValueError(f"Invalid number of stages: {nstages}")


def load_model(inputs: tf.Tensor, num_classes: int = 2):
    """Load sleep stage model"""
    blocks = [
        UNetBlockParams(filters=24, depth=1, kernel=(1, 5), strides=(1, 2), skip=True, seperable=True),
        UNetBlockParams(filters=32, depth=1, kernel=(1, 5), strides=(1, 2), skip=True, seperable=True),
        UNetBlockParams(filters=40, depth=1, kernel=(1, 5), strides=(1, 2), skip=True, seperable=False),
    ]
    return UNet(
        inputs,
        params=UNetParams(
            blocks=blocks,
            output_kernel_size=(1, 5),
            include_top=True,
            include_rnn=False
        ),
        num_classes=num_classes,
    )

def prepare(x: tf.Tensor, y: tf.Tensor, num_classes: int, class_map: dict[int, int]):
    """Prepare data for training"""
    return (
        x,
        tf.one_hot(np.vectorize(class_map.get)(y), num_classes)
    )

def load_train_datasets(params: SKTrainParams, feat_shape, class_shape, class_map, feat_cols=None):
    """Load train/val datasets"""
    def preprocess(x: npt.NDArray[np.float32]):
        xx = x.copy()
        for i in range(3):
            xx[:, i] = pk.signal.filter_signal(xx[:, i], lowcut=1.0, highcut=30, sample_rate=params.sampling_rate, order=3)  # EEG
        for i in range(3, 5):
            xx[:, i] = pk.signal.filter_signal(xx[:, i], lowcut=0.5, highcut=8, sample_rate=params.sampling_rate, order=4)  # EOG
        for i in range(5):
            xx[:, i] = pk.signal.normalize_signal(xx[:, i], eps=1e-3, axis=None)
        return xx

    output_signature = (
        tf.TensorSpec(shape=feat_shape, dtype=tf.float32),
        tf.TensorSpec(shape=class_shape, dtype=tf.int32),
    )

    # Create dataset(s)
    ds = Hdf5Dataset(
        ds_path=params.ds_path,
        frame_size=params.frame_size,
        mask_key="mask",
        feat_cols=feat_cols
    )

    # Get train/val subject IDs and generators
    train_subject_ids, val_subject_ids = sklearn.model_selection.train_test_split(
        ds.train_subject_ids, test_size=params.val_subjects
    )

    def train_generator(subject_ids):
        def ds_gen():
            train_subj_gen = ds.uniform_subject_generator(subject_ids)
            return map(
                lambda x_y: prepare(preprocess(x_y[0]), x_y[1], class_shape[-1], class_map),
                ds.signal_generator(
                    train_subj_gen,
                    samples_per_subject=params.samples_per_subject
                )
            )
        return tf.data.Dataset.from_generator(
            ds_gen,
            output_signature=output_signature,
        )

    split = len(train_subject_ids) // params.data_parallelism
    train_datasets = [train_generator(
        train_subject_ids[i * split : (i + 1) * split]
    ) for i in range(params.data_parallelism)]

    # Create TF datasets
    train_ds = tf.data.Dataset.from_tensor_slices(
        train_datasets
    ).interleave(
        lambda x: x,
        cycle_length=params.data_parallelism,
        deterministic=False,
        num_parallel_calls=tf.data.AUTOTUNE,
    ).shuffle(
        buffer_size=params.buffer_size,
        reshuffle_each_iteration=True,
    ).batch(
        batch_size=params.batch_size,
        drop_remainder=False,
    ).prefetch(
        buffer_size=tf.data.AUTOTUNE
    )

    def val_generator():
        val_subj_gen = ds.uniform_subject_generator(val_subject_ids)
        return map(
            lambda x_y: prepare(preprocess(x_y[0]), x_y[1], class_shape[-1], class_map),
            ds.signal_generator(
                val_subj_gen,
                samples_per_subject=params.samples_per_subject
            )
        )

    val_ds = tf.data.Dataset.from_generator(
        generator=val_generator,
        output_signature=output_signature
    )
    val_x, val_y = next(val_ds.batch(params.val_size).as_numpy_iterator())
    val_ds = create_dataset_from_data(
        val_x, val_y, output_signature=output_signature
    ).batch(
        batch_size=params.batch_size,
        drop_remainder=False,
    )

    return train_ds, val_ds


def train(params: SKTrainParams):
    """Train sleep stage model.

    Args:
        params (SKTrainParams): Training parameters
    """

    params.seed = set_random_seed(params.seed)
    logger.info(f"Random seed {params.seed}")

    os.makedirs(str(params.job_dir), exist_ok=True)
    logger.info(f"Creating working directory in {params.job_dir}")
    with open(str(params.job_dir / "train_config.json"), "w", encoding="utf-8") as fp:
        fp.write(params.json(indent=2))

    if env_flag("WANDB"):
        wandb.init(
            project=f"sleepkit-sleepstage",
            entity="ambiq",
            dir=params.job_dir,
        )
        wandb.config.update(params.dict())

    num_sleep_stages = 3
    feat_cols = [
        0, 1, 2, 3,
        4, 5, 6, 7,
        8, 9, 10, 11, 12, 13,
        14, 16, 17, 18
    ]
    num_feats = len(feat_cols)
    target_classes = get_target_classes(num_sleep_stages)
    class_names = get_class_names(num_sleep_stages)
    class_mapping = get_class_mapping(num_sleep_stages)
    num_classes = len(target_classes)

    feat_shape = (params.frame_size, num_feats)
    class_shape = (params.frame_size, num_classes)

    train_ds, val_ds = load_train_datasets(
        params,
        feat_shape,
        class_shape,
        class_mapping,
        feat_cols=feat_cols
    )

    strategy = get_strategy()
    with strategy.scope():
        logger.info("Building model")
        inputs = tf.keras.Input(feat_shape, batch_size=None, dtype=tf.float32)
        model = load_model(inputs, num_classes=len(target_classes))
        flops = get_flops(model, batch_size=1, fpath=str(params.job_dir / "model_flops.log"))

        # Grab optional LR parameters
        lr_rate: float = getattr(params, "lr_rate", 1e-3)
        lr_cycles: int = getattr(params, "lr_cycles", 3)
        steps_per_epoch = params.steps_per_epoch or 1000
        scheduler = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=lr_rate,
            first_decay_steps=int(0.1 * steps_per_epoch * params.epochs),
            t_mul=1.65 / (0.1 * lr_cycles * (lr_cycles - 1)),
            m_mul=0.4,
        )
        optimizer = tf.keras.optimizers.Adam(scheduler)
        loss = tf.keras.losses.CategoricalFocalCrossentropy(from_logits=True)
        metrics = [
            tf.keras.metrics.CategoricalAccuracy(name="acc"),
            MultiF1Score(name="f1", dtype=tf.float32, average="macro"),
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
                steps_per_epoch=steps_per_epoch,
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

        test_labels = [y.numpy() for _, y in val_ds]
        y_true = np.argmax(np.concatenate(test_labels).squeeze(), axis=-1)
        y_pred = np.argmax(model.predict(val_ds).squeeze(), axis=-1)

        # Summarize results
        test_acc = np.sum(y_pred == y_true) / y_true.size

        test_iou = compute_iou(y_true, y_pred, average="weighted")
        logger.info(f"[TEST SET] ACC={test_acc:.2%}, IoU={test_iou:.2%}")

        cm_path = str(params.job_dir / "confusion_matrix_test.png")
        confusion_matrix_plot(
            y_true.flatten(),
            y_pred.flatten(),
            labels=class_names,
            save_path=cm_path,
            normalize="true",
        )
    # END WITH

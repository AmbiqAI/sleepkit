"""Sleep detect"""
import os

import numpy as np
import numpy.typing as npt
import physiokit as pk
import tensorflow as tf
import sklearn.model_selection
import wandb
from rich.console import Console
from sklearn.metrics import f1_score
from wandb.keras import WandbCallback

from neuralspot.tflite.metrics import get_flops
from neuralspot.tflite.model import get_strategy, load_model

from .defines import SKTrainParams
from .utils import env_flag, set_random_seed, setup_logger
from .datasets import MesaDataset
from .datasets.utils import create_dataset_from_data
from .models import UNet, UNetParams, UNetBlockParams

console = Console()
logger = setup_logger(__name__)

sd_features = ["EEG1", "EOG-L"]
sd_classes = [0, 1]
sd_sleep_mapping = lambda s: 1 if s in (1, 2, 3, 4, 5) else 0

def load_model(inputs: tf.Tensor, num_classes: int = 2):
    blocks = [
        UNetBlockParams(filters=8, depth=2, kernel=(1, 3), strides=(1, 2), skip=True),
        UNetBlockParams(filters=16, depth=2, kernel=(1, 3), strides=(1, 2), skip=True),
        UNetBlockParams(filters=24, depth=2, kernel=(1, 3), strides=(1, 2), skip=True),
        UNetBlockParams(filters=32, depth=2, kernel=(1, 3), strides=(1, 2), skip=True),
        UNetBlockParams(filters=48, depth=2, kernel=(1, 3), strides=(1, 2), skip=True),
    ]
    return UNet(
        inputs,
        params=UNetParams(
            blocks=blocks,
            output_kernel_size=(1, 3),
            include_top=True,
        ),
        num_classes=num_classes,
    )

def preprocess(x: npt.NDArray[np.float32]):
    xx = x.copy()
    for i in range(x.shape[-1]):
        xx[..., i] = pk.signal.normalize_signal(xx[..., i], eps=1e-3, axis=None)
    return xx

def augment(x):
    return x

def prepare(x, y):
    return (
        # Add empty dimension (1D -> 2D)
        tf.expand_dims(x, axis=0),
        # Add empty dimension (1D -> 2D) and one-hot encode
        tf.one_hot(tf.expand_dims(y, axis=0))
    )

def load_train_datasets(params: SKTrainParams):

    output_signature=(
        tf.TensorSpec(shape=(params.frame_size, len(sd_features)), dtype=tf.float32),
        tf.TensorSpec(shape=(params.frame_size), dtype=tf.int32),
    )

    # Create dataset(s)
    ds = MesaDataset(
        ds_path=params.ds_path,
        frame_size=params.frame_size,
        target_rate=params.sampling_rate,
        is_commercial=True
    )
    ds.set_sleep_mapping(sd_sleep_mapping)

    # Get train/val subject IDs and generators
    train_subject_ids, val_subject_ids = sklearn.model_selection.train_test_split(
        ds.train_subject_ids, test_size=params.val_subjects
    )
    train_subj_gen = ds.uniform_subject_generator(train_subject_ids)
    val_subj_gen = ds.uniform_subject_generator(val_subject_ids)

    # Create TF datasets
    train_ds = tf.data.Dataset.from_generator(
        generator=ds.signal_generator,
        output_signature=output_signature,
        args=(train_subj_gen, sd_features, params.samples_per_subject),
    # Preprocess/augment
    ).map(
        lambda x, y: (augment(preprocess(x)), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    # Prep for model input
    ).map(
        lambda x, y: prepare(x, y),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).shuffle(
        buffer_size=params.buffer_size,
        reshuffle_each_iteration=True,
    ).batch(
        batch_size=params.batch_size,
        drop_remainder=True,
    ).prefetch(
        buffer_size=tf.data.AUTOTUNE,
    )

    val_ds = tf.data.Dataset.from_generator(
        generator=ds.signal_generator,
        output_signature=output_signature,
        args=(val_subj_gen, sd_features, params.val_samples_per_subject),
    # Preprocess
    ).map(
        lambda x, y: (preprocess(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    # Prep for model input
    ).map(
        lambda x, y: prepare(x, y),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).shuffle(
        buffer_size=params.buffer_size,
        reshuffle_each_iteration=True,
    ).batch(
        batch_size=params.batch_size,
        drop_remainder=True,
    )
    val_x, val_y = next(val_ds.batch(params.val_size).as_numpy_iterator())
    val_ds = create_dataset_from_data(val_x, val_y, output_signature=output_signature)

    return train_ds, val_ds


def train(params: SKTrainParams):
    """Train awake/sleep detection model.

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

    train_ds, val_ds = load_train_datasets(params)

    strategy = get_strategy()
    with strategy.scope():
        logger.info("Building model")
        in_shape = (1, params.frame_size, len(sd_features))
        inputs = tf.keras.Input(in_shape, batch_size=None, dtype=tf.float32)
        model = load_model(inputs, num_classes=len(sd_classes))
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
            tf.keras.metrics.OneHotIoU(
                num_classes=len(sd_classes),
                target_class_ids=[0, 1],
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
            tf.keras.callbacks.TensorBoard(log_dir=str(params.job_dir), write_steps_per_second=True),
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
        test_labels = [label.numpy() for _, label in val_ds]
        y_true = np.argmax(np.concatenate(test_labels), axis=1)
        y_pred = np.argmax(model.predict(val_ds), axis=1)

        # Summarize results
        test_acc = np.sum(y_pred == y_true) / len(y_true)
        test_f1 = f1_score(y_true, y_pred, average="macro")
        logger.info(f"[VAL SET] ACC={test_acc:.2%}, F1={test_f1:.2%}")
    # END WITH

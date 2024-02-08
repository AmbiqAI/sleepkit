"""Sleep Apnea Utils"""

from pathlib import Path
from typing import Any

import keras
import numpy as np
import numpy.typing as npt
import tensorflow as tf

from ..datasets import Hdf5Dataset
from ..datasets.utils import create_dataset_from_data
from ..models import UNet, UNetParams, UNext, UNextBlockParams, UNextParams


def create_model(
    inputs: tf.Tensor, num_classes: int, name: str | None = None, params: dict[str, Any] | None = None
) -> keras.Model:
    """Generate model or use default

    Args:
        inputs (tf.Tensor): Model inputs
        num_classes (int): Number of classes
        name (str | None, optional): Architecture type. Defaults to None.
        params (dict[str, Any] | None, optional): Model parameters. Defaults to None.

    Returns:
        keras.Model: Model
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
                UNextBlockParams(
                    filters=24, depth=2, kernel=5, pool=2, strides=2, skip=True, expand_ratio=1, se_ratio=2, dropout=0
                ),
                UNextBlockParams(
                    filters=32, depth=2, kernel=5, pool=2, strides=2, skip=True, expand_ratio=1, se_ratio=2, dropout=0
                ),
                UNextBlockParams(
                    filters=48, depth=2, kernel=5, pool=2, strides=2, skip=True, expand_ratio=1, se_ratio=2, dropout=0
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
        feat_key="features",
        label_key="apnea",
        mask_key="mask",
        feat_cols=feat_cols,
    )
    return ds


def load_train_dataset(
    ds: Hdf5Dataset,
    subject_ids,
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
        ds (Hdf5Dataset): Dataset
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

    def preprocess(x: npt.NDArray[np.float32]):
        """Preprocess data"""
        return x + np.random.normal(0, 0.05, size=x.shape)

    def train_generator(subject_ids):
        """Train generator per worker"""

        def ds_gen():
            """Worker generator routine"""
            train_subj_gen = ds.uniform_subject_generator(subject_ids)
            return map(
                lambda x_y: prepare(preprocess(x_y[0]), x_y[1], class_shape[-1], class_map),
                ds.signal_generator(train_subj_gen, samples_per_subject=samples_per_subject),
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

    # Create TF datasets
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
    ds: Hdf5Dataset,
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
        """Preprocess data"""
        return x

    output_signature = (
        tf.TensorSpec(shape=feat_shape, dtype=tf.float32),
        tf.TensorSpec(shape=class_shape, dtype=tf.int32),
    )

    def val_generator():
        val_subj_gen = ds.uniform_subject_generator(subject_ids)
        return map(
            lambda x_y: prepare(preprocess(x_y[0]), x_y[1], class_shape[-1], class_map),
            ds.signal_generator(val_subj_gen, samples_per_subject=samples_per_subject),
        )

    val_ds = tf.data.Dataset.from_generator(generator=val_generator, output_signature=output_signature)
    val_x, val_y = next(val_ds.batch(val_size).as_numpy_iterator())
    val_ds = create_dataset_from_data(val_x, val_y, output_signature=output_signature).batch(
        batch_size=batch_size,
        drop_remainder=False,
    )
    return val_ds


def load_test_dataset(
    ds: Hdf5Dataset,
    subject_ids: list[str],
    samples_per_subject: int,
    batch_size: int,
    test_size: int,
    feat_shape: tuple[int, ...],
    class_shape: tuple[int, ...],
    class_map: dict[int, int],
) -> tf.data.Dataset:
    """Load test dataset.

    Args:
        ds (Hdf5Dataset): Dataset
        subject_ids (list[str]): Subject IDs
        samples_per_subject (int): Samples per subject
        batch_size (int): Batch size
        test_size (int): Test size
        feat_shape (tuple[int,...]): Feature shape
        class_shape (tuple[int,...]): Class shape
        class_map (dict[int, int]): Class mapping

    Returns:
        tf.data.Dataset: Validation dataset
    """

    def preprocess(x: npt.NDArray[np.float32]):
        """Preprocess data"""
        return x

    output_signature = (
        tf.TensorSpec(shape=feat_shape, dtype=tf.float32),
        tf.TensorSpec(shape=class_shape, dtype=tf.int32),
    )

    def test_generator():
        val_subj_gen = ds.uniform_subject_generator(subject_ids)
        return map(
            lambda x_y: prepare(preprocess(x_y[0]), x_y[1], class_shape[-1], class_map),
            ds.signal_generator(val_subj_gen, samples_per_subject=samples_per_subject),
        )

    test_ds = tf.data.Dataset.from_generator(generator=test_generator, output_signature=output_signature)
    test_x, test_y = next(test_ds.batch(test_size).as_numpy_iterator())
    test_ds = create_dataset_from_data(test_x, test_y, output_signature=output_signature).batch(
        batch_size=batch_size,
        drop_remainder=False,
    )
    return test_ds

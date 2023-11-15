from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import tensorflow as tf

from ..datasets import Hdf5Dataset, SKDataset
from ..datasets.utils import create_dataset_from_data
from ..models import Tcn, TcnBlockParams, TcnParams, generate_model


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
    if name:
        return generate_model(inputs=inputs, num_classes=num_classes, name=name, params=params)

    return Tcn(
        x=inputs,
        params=TcnParams(
            input_kernel=(1, 5),
            input_norm="batch",
            blocks=[
                TcnBlockParams(
                    filters=64, kernel=(1, 5), dilation=(1, 2**d), dropout=0.1, ex_ratio=1, se_ratio=4, norm="batch"
                )
                for d in range(4)
            ],
            output_kernel=(1, 5),
            include_top=True,
            use_logits=True,
            model_name="tcn",
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
        ds (SKDataset): Dataset
        subject_ids (list[str]): Subject IDs
        samples_per_subject (int): Samples per subject
        test_size (int): Test size
        feat_shape (tuple[int,...]): Feature shape
        class_shape (tuple[int,...]): Class shape
        class_map (dict[int, int]): Class mapping

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

"""Sleep Apnea Utils"""

from pathlib import Path

import keras
import numpy as np
import numpy.typing as npt
import tensorflow as tf

from ...datasets import DatasetFactory, SKDataset
from ...datasets.utils import create_dataset_from_data
from ...defines import DatasetParams, ModelArchitecture
from ...models import ModelFactory, UNext, UNextBlockParams, UNextParams


def create_model(inputs: tf.Tensor, num_classes: int, architecture: ModelArchitecture | None) -> keras.Model:
    """Generate model or use default

    Args:
        inputs (tf.Tensor): Model inputs
        num_classes (int): Number of classes
        architecture (ModelArchitecture|None): Model

    Returns:
        keras.Model: Model
    """
    if architecture:
        return ModelFactory.create(
            name=architecture.name,
            params=architecture.params,
            inputs=inputs,
            num_classes=num_classes,
        )

    return _default_model(inputs=inputs, num_classes=num_classes)


def _default_model(
    inputs: tf.Tensor,
    num_classes: int,
) -> keras.Model:
    """Reference model

    Args:
        inputs (tf.Tensor): Model inputs
        num_classes (int): Number of classes

    Returns:
        keras.Model: Model
    """

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


def load_dataset(
    ds_path: Path,
    frame_size: int,
    dataset: DatasetParams,
) -> SKDataset:
    """Load dataset

    Args:
        ds_path (Path): Dataset path
        frame_size (int): Frame size
        dataset (DatasetParams): Dataset parameters

    Returns:
        SKDataset: Dataset
    """
    if not DatasetFactory.has(dataset.name):
        raise ValueError(f"Dataset {dataset.name} not found")

    return DatasetFactory.get(dataset.name)(ds_path=ds_path, frame_size=frame_size, **dataset.params)


def load_train_dataset(
    ds: SKDataset,
    subject_ids: list[str],
    samples_per_subject: int,
    buffer_size: int,
    batch_size: int,
    spec: tuple[tf.TensorSpec, tf.TensorSpec],
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
        spec (tuple[tf.TensorSpec, tf.TensorSpec]): Dataset signature
        class_map (dict[int, int]): Class mapping
        num_workers (int, optional): Number of workers. Defaults to 4.

    Returns:
        tf.data.Dataset: Train dataset
    """

    def preprocess(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Preprocess data"""
        xx = x.copy()
        # xx = xx + np.random.normal(0, 0.1, size=x.shape)
        return xx

    class_shape = spec[1].shape

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
            output_signature=spec,
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
    spec: tuple[tf.TensorSpec, tf.TensorSpec],
    class_map: dict[int, int],
) -> tf.data.Dataset:
    """Load validation dataset.

    Args:
        ds (SKDataset): Dataset
        subject_ids (list[str]): Subject IDs
        samples_per_subject (int): Samples per subject
        batch_size (int): Batch size
        val_size (int): Validation size
        spec (tuple[tf.TensorSpec, tf.TensorSpec]): Dataset signature
        class_map (dict[int, int]): Class mapping

    Returns:
        tf.data.Dataset: Validation dataset
    """

    def preprocess(x: npt.NDArray[np.float32]):
        return x

    class_shape = spec[1].shape

    def val_generator():
        val_subj_gen = ds.uniform_subject_generator(subject_ids)
        return map(
            lambda x_y: prepare(preprocess(x_y[0]), x_y[1], class_shape[-1], class_map),
            ds.signal_generator(val_subj_gen, samples_per_subject=samples_per_subject, normalize=True),
        )

    val_ds = tf.data.Dataset.from_generator(generator=val_generator, output_signature=spec)
    val_x, val_y = next(val_ds.batch(val_size).as_numpy_iterator())
    val_ds = create_dataset_from_data(val_x, val_y, output_signature=spec).batch(
        batch_size=batch_size,
        drop_remainder=False,
    )

    return val_ds


def load_test_dataset(
    ds: SKDataset,
    subject_ids: list[str],
    samples_per_subject: int,
    test_size: int,
    spec: tuple[tf.TensorSpec, tf.TensorSpec],
    class_map: dict[int, int],
) -> tuple[npt.NDArray, npt.NDArray]:
    """Load test dataset

    Args:
        ds (SKDataset): Dataset
        subject_ids (list[str]): Subject IDs
        samples_per_subject (int): Samples per subject
        test_size (int): Test size
        spec (tuple[tf.TensorSpec, tf.TensorSpec]): Dataset signature
        class_map (dict[int, int]): Class mapping

    Returns:
        tuple[npt.NDArray, npt.NDArray]: Test features and labels
    """

    def preprocess(x: npt.NDArray[np.float32]):
        return x

    class_shape = spec[1].shape

    def test_generator():
        test_subj_gen = ds.uniform_subject_generator(subject_ids)
        return map(
            lambda x_y: prepare(preprocess(x_y[0]), x_y[1], class_shape[-1], class_map),
            ds.signal_generator(test_subj_gen, samples_per_subject=samples_per_subject, normalize=True),
        )

    test_ds = tf.data.Dataset.from_generator(generator=test_generator, output_signature=spec)
    test_x, test_y = next(test_ds.batch(test_size).as_numpy_iterator())

    return test_x, test_y

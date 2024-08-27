"""
# Stage Classification Task Utilities API

Functions:
    create_data_pipeline: Create a data pipeline.
    subject_data_preprocessor: Preprocess entire subject data.

"""

import functools

import numpy as np
import tensorflow as tf
import neuralspot_edge as nse

from ...features import H5Dataloader


def subject_data_preprocessor(x, y, mask):
    """Preprocess entire subject data."""

    epsilon = 1e-6
    mask_x = x[mask == 1] if mask is not None else x

    # Impute missing values with median
    if mask is not None:
        x_med = np.nanmedian(mask_x, axis=0)
        x[mask == 0, :] = x_med

    x_mu = np.nanmean(mask_x, axis=0)
    x_var = np.nanvar(mask_x, axis=0)
    x = (x - x_mu) / np.sqrt(x_var + epsilon)
    return x, y, mask


def create_data_pipeline(
    dataloader: H5Dataloader,
    subject_ids: list[str],
    samples_per_subject: int,
    num_classes: int,
    batch_size: int,
    buffer_size: int | None = None,
    cache_size: int | None = None,
) -> tf.data.Dataset:
    """Create a data pipeline.

    Args:

    Returns:
        tf.data.Dataset: Data pipeline.
    """

    # augmenter = create_augmentation_pipeline(augmentations)

    data_gen = functools.partial(
        dataloader.signal_generator,
        subject_generator=nse.utils.uniform_id_generator(subject_ids),
        samples_per_subject=samples_per_subject,
        preprocessor=subject_data_preprocessor,
    )

    sig = nse.utils.get_output_signature_from_gen(data_gen)

    dataloader = tf.data.Dataset.from_generator(data_gen, output_signature=sig).map(
        lambda data, labels: (
            tf.cast(data, "float32"),
            tf.one_hot(labels, num_classes),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if buffer_size:
        dataloader = dataloader.shuffle(
            buffer_size=buffer_size,
            reshuffle_each_iteration=True,
        )
    if batch_size:
        dataloader = dataloader.batch(
            batch_size=batch_size,
            drop_remainder=False,
        )
    if cache_size:
        dataloader = dataloader.take(cache_size).cache()

    return dataloader.prefetch(tf.data.AUTOTUNE)

import math
from typing import Callable

import numpy as np
import tensorflow as tf
from pydantic import BaseModel, Field

def sample_normalize(sample):
    mean = tf.math.reduce_mean(sample)
    std = tf.math.reduce_std(sample)
    sample = tf.math.divide_no_nan(sample-mean, std)

    return sample.numpy()

def get_blocks(
    series,
    columns,
    block_size,
    block_stride
):
    series = series.copy()
    series = series[columns]
    series = series.values
    series = series.astype(np.float32)

    block_count = math.ceil(len(series) / block_size)

    series = np.pad(series, pad_width=[[0, block_count*block_size-len(series)], [0, 0]])

    block_begins = list(range(0, len(series), block_stride))
    block_begins = [x for x in block_begins if x+block_size <= len(series)]

    blocks = []
    for begin in block_begins:
        values = series[begin:begin+block_size]
        blocks.append({'begin': begin, 'end': begin+block_size, 'values': values})
    # END FOR
    return blocks


class FogParams(BaseModel):
    model_dim: int = Field(default=320, description="Model dimension")
    block_size: int = Field(default=15552, description="Block size")
    patch_size: int = Field(default=18, description="Patch size")
    num_encoders: int = Field(default=2, description="# encoder layers")
    num_lstms: int = Field(default=2, description="# LSTM layers")
    num_heads: int = Field(default=2, description="# transformer heads")
    batch_size: int = Field(default=32, description="Batch size")
    training: bool = Field(default=False, description="Training mode")
    dropout: float = Field(default=0, description="Dropout rate")

def encoder(
    num_heads: int = 2,
    model_dim: int = 2,
    dropout: float = 0
) -> Callable[[tf.Tensor], tf.Tensor]:
    def layer(x: tf.Tensor):
        y = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=model_dim, dropout=dropout)(query=x, key=x, value=x)
        y = tf.keras.layers.Add()([x, y])
        y = tf.keras.layers.LayerNormalization()(y)
        ys = y
        y = tf.keras.layers.Dense(model_dim)(y)
        y = tf.keras.layers.Dropout(dropout)(y)
        y = tf.keras.layers.Dense(model_dim)(y)
        y = tf.keras.layers.Dropout(dropout)(y)
        y = tf.keras.layers.Add()([ys, y])
        y = tf.keras.layers.LayerNormalization()(y)
        return y
    return layer

def fog_encoder(
    model_dim: int,
    block_size: int,
    patch_size: int,
    num_encoders: int,
    num_lstms: int,
    num_heads: int,
    batch_size: int,
    training: int,
    dropout: float = 0,
):
    def layer(x: tf.Tensor) -> tf.Tensor:
        sequence_len = block_size / patch_size
        y = tf.keras.layers.Dense(model_dim)(x)
        pos_encoding = tf.Variable(initial_value=tf.random.normal(shape=(1, sequence_len, model_dim), stddev=0.02), trainable=True)
        if training:
            pos_encoder = tf.roll(tf.tile(
                pos_encoding,
                multiples=[batch_size, 1, 1]),
                shift=tf.random.uniform(
                    shape=(batch_size,),
                    minval=-sequence_len,
                    maxval=0,
                    dtype=tf.int32
                ),
                axis=batch_size * [1],
            )
        else:
            pos_encoder = tf.tile(pos_encoding, multiples=[batch_size, 1, 1])

        y = tf.keras.layers.Add()[y, pos_encoder]
        y = tf.keras.layers.Dropout(dropout)(y)
        for _ in range(num_encoders):
            y = encoder(num_heads=num_heads, model_dim=model_dim, dropout=dropout)(y)
        for _ in range(num_lstms):
            y = tf.keras.layers.LSTM(model_dim, return_sequences=True)(y)
            y = tf.keras.layers.Bidirectional()(y)
    return layer

def fog_model(
    inputs: tf.Tensor,
    params: FogParams,
    num_classes: int
):
    y = fog_encoder(
        model_dim=params.model_dim,
        block_size=params.block_size,
        patch_size=params.patch_size,
        num_encoders=params.num_encoders,
        num_lstms=params.num_lstms,
        num_heads=params.num_heads,
        batch_size=params.batch_size,
        training=params.training,
        dropout=params.dropout
    )(inputs)
    y = tf.keras.layers.Dense(num_classes)(y)
    y = tf.keras.layers.Activation(tf.keras.activations.hard_sigmoid)(y)
    model = tf.keras.Model(inputs=inputs, outputs=y)
    return model

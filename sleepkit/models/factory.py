from typing import Any

import keras
import tensorflow as tf

from .efficientnet import EfficientNetParams, EfficientNetV2
from .tcn import Tcn, TcnParams
from .unet import UNet, UNetParams
from .unext import UNext, UNextParams


def generate_model(
    inputs: tf.Tensor,
    num_classes: int,
    name: str,
    params: dict[str, Any],
) -> keras.Model:
    """Model factory: Generates a model based on the provided name and parameters

    Args:
        inputs (tf.Tensor): Input tensor
        num_classes (int): Number of classes
        name (str): Model name
        params (dict[str, Any]): Model parameters

    Returns:
        keras.Model: Generated model
    """
    if params is None:
        raise ValueError("Model parameters must be provided")

    match name:
        case "unet":
            return UNet(x=inputs, params=UNetParams.parse_obj(params), num_classes=num_classes)

        case "unext":
            return UNext(x=inputs, params=UNextParams.parse_obj(params), num_classes=num_classes)

        case "efficientnetv2":
            return EfficientNetV2(x=inputs, params=EfficientNetParams.parse_obj(params), num_classes=num_classes)

        case "tcn":
            return Tcn(x=inputs, params=TcnParams.parse_obj(params), num_classes=num_classes)

        case "rnn":
            y = inputs
            y = keras.layers.Reshape((1,) + y.shape[1:])(y)
            y = keras.layers.DepthwiseConv2D(kernel_size=(1, 5), padding="same")(y)
            y = keras.layers.LayerNormalization(axis=[2])(y)
            y = keras.layers.Activation("relu6")(y)

            y = keras.layers.Conv2D(filters=32, kernel_size=(1, 5), padding="same")(y)
            y = keras.layers.LayerNormalization(axis=[2])(y)
            y = keras.layers.Activation("relu6")(y)

            y = keras.layers.Reshape(y.shape[2:])(y)

            y = keras.layers.Bidirectional(keras.layers.LSTM(units=48, return_sequences=True))(y)
            # y = keras.layers.LSTM(units=48, return_sequences=True)(y)
            y = keras.layers.LayerNormalization(axis=[1])(y)

            y = keras.layers.TimeDistributed(keras.layers.Dense(64))(y)
            y = keras.layers.LayerNormalization(axis=[1])(y)
            y = keras.layers.Activation("relu6")(y)

            y = keras.layers.TimeDistributed(keras.layers.Dense(num_classes))(y)
            model = keras.models.Model(inputs, y)
            return model

        case _:
            raise NotImplementedError()
    # END MATCH

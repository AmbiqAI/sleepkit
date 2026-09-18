"""A small native Keras model. Experiments can replace this ordinary function."""


def build_model(context, features=5):
    import keras

    inputs = keras.Input((context, features), name="features")
    x = keras.layers.Conv1D(16, 1, activation="relu")(inputs)
    for dilation in (1, 2, 4):
        residual = keras.layers.Conv1D(16, 5, padding="same", dilation_rate=dilation, activation="relu")(x)
        x = keras.layers.Add()([x, residual])
    outputs = keras.layers.Dense(2, name="logits")(x)
    return keras.Model(inputs, outputs, name="wrist_detection")

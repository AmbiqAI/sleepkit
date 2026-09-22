"""Small default model; experiments may supply another builder directly to train."""


def build_model():
    import keras

    inputs = keras.Input((240, 14), dtype="float32", name="features")
    hidden = keras.layers.Conv1D(24, 7, padding="same", activation="relu")(inputs)
    hidden = keras.layers.Conv1D(24, 5, padding="same", dilation_rate=2, activation="relu")(hidden)
    logits = keras.layers.Conv1D(3, 1, name="logits")(hidden)
    return keras.Model(inputs, logits, name="staging_conv1d_v1")

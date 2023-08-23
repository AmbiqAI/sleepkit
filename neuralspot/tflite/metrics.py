import tensorflow as tf
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder


def get_flops(model: tf.keras.Model, batch_size: int | None = None, fpath: str | None = None) -> float:
    """
    Calculate FLOPS for tf.keras.Model or tf.keras.Sequential .
    Ignore operations used in only training mode such as Initialization.
    Use tf.profiler of tensorflow v2 api.
    """
    input_signature = [tf.TensorSpec(shape=(batch_size,) + model.input_shape[1:])]
    forward_pass = tf.function(model.call, input_signature=input_signature)
    graph = forward_pass.get_concrete_function().graph
    options = ProfileOptionBuilder.float_operation()
    if fpath:
        options["output"] = f"file:outfile={fpath}"
    graph_info = profile(graph, options=options)
    return float(graph_info.total_float_ops)

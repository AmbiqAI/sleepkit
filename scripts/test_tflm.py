import numpy as np
import tensorflow as tf
import tflite_micro as tflm

INPUT_SHAPE = (8, 1)
DILATION_RATE = 2

# Create example dilated convolutional model
inputs = tf.keras.layers.Input(shape=INPUT_SHAPE)
y = inputs
y = tf.keras.layers.Convolution1D(1, 3, dilation_rate=DILATION_RATE, padding="same")(y)
model = tf.keras.Model(inputs=inputs, outputs=y)


# Convert to TF Lite model
converter = tf.lite.TFLiteConverter.from_keras_model(model=model)
tflite_model = converter.convert()

# Create TFLite and TFLM interpreter
tfl_interpreter = tf.lite.Interpreter(model_content=tflite_model)
tfl_interpreter.allocate_tensors()

tflm_interpreter = tflm.runtime.Interpreter.from_bytes(
    tflite_model, intrepreter_config=tflm.runtime.InterpreterConfig.kPreserveAllTensors
)

# Create input tensor of 1s
x = np.ones((1,) + INPUT_SHAPE, dtype=np.float32)

# Invoke TF, TFLite and TFLM models using same input
y_tf = model(x)

tfl_interpreter.set_tensor(tfl_interpreter.get_input_details()[0]["index"], x)
tfl_interpreter.invoke()
y_tfl = tfl_interpreter.get_tensor(tfl_interpreter.get_output_details()[0]["index"])

tflm_interpreter.set_input(x, 0)
tflm_interpreter.invoke()
y_tflm = tflm_interpreter.get_output(0)

# Below passes
np.testing.assert_almost_equal(y_tf, y_tfl, decimal=3, err_msg="TF and TFLite output do not match.")

# Below fails
# np.testing.assert_almost_equal(y_tf, y_tflm, decimal=3, err_msg="TF and TFLM output do not match.")


tf.lite.experimental.Analyzer.analyze(model_content=tflite_model)

for i in range(0, tfl_interpreter._interpreter.NumTensors(0)):
    print(f"Tensor {i}: {tfl_interpreter.get_tensor_details()[i]['name']}")
    tflm_tensor = tflm_interpreter.GetTensor(i, 0)["tensor_data"]
    try:
        tfl_tensor = tfl_interpreter.get_tensor(i, 0)
    except ValueError:
        print("  TFL: N/A")
        print(f" TFLM: shape={tflm_tensor.shape}, dtype={tflm_tensor.dtype}")
        print("")
        continue

    is_match = np.allclose(tfl_tensor, tflm_tensor, atol=1e-3)
    print(f"  TFL: shape={tfl_tensor.shape}, dtype={tfl_tensor.dtype}")
    print(f" TFLM: shape={tflm_tensor.shape}, dtype={tflm_tensor.dtype}")
    print(f" MATCH: {'YES' if is_match else 'NO'}")
    print("")

# END FOR

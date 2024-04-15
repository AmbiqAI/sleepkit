import numpy as np
import tensorflow as tf
import tflite_micro as tflm

INPUT_SHAPE = (900, 5)
inputs = tf.keras.layers.Input(shape=INPUT_SHAPE)

x = np.ones((1,)+INPUT_SHAPE, dtype=np.int8)


with open("../results/sa-2-tcn-sm/model.tflite", "rb") as f:
    tflite_model = f.read()

# Create TFLite and TFLM interpreter
tfl_interpreter = tf.lite.Interpreter(model_content=tflite_model)
tfl_interpreter.allocate_tensors()

tflm_interpreter = tflm.runtime.Interpreter.from_bytes(tflite_model, intrepreter_config=tflm.runtime.InterpreterConfig.kPreserveAllTensors)

tfl_interpreter.set_tensor(tfl_interpreter.get_input_details()[0]["index"], x)
tfl_interpreter.invoke()
y_tfl = tfl_interpreter.get_tensor(tfl_interpreter.get_output_details()[0]["index"])

tflm_interpreter.set_input(x, 0)
tflm_interpreter.invoke()
y_tflm = tflm_interpreter.get_output(0)


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

"""Small optional LiteRT adapter for float32 or per-tensor int8 detection logits."""

from pathlib import Path

import numpy as np

from sleepkit.artifacts.package import sha256


def _quantization(tensor, integer):
    quant = tensor.get("quantization_parameters", {})
    scales = np.asarray(quant.get("scales", []))
    zeros = np.asarray(quant.get("zero_points", []))
    if scales.ndim != 1 or zeros.ndim != 1:
        raise ValueError("Runtime requires per-tensor quantization metadata")
    if not integer:
        if scales.size or zeros.size:
            raise ValueError("Float32 tensors must not declare quantization")
        return None, None
    if (scales.size != 1 or zeros.size != 1 or scales.dtype.kind not in "fiu"
            or zeros.dtype.kind not in "iu" or not np.isfinite(scales[0]) or scales[0] <= 0
            or not -128 <= int(zeros[0]) <= 127):
        raise ValueError("Int8 tensors require one finite positive scale and an int8 zero point")
    return float(scales[0]), int(zeros[0])


class DetectionRuntime:
    """One context in, float32 logits out; never applies a softmax."""

    def __init__(self, model_path, context, *, artifact=None):
        from ai_edge_litert.interpreter import Interpreter

        if type(context) is not int or context < 1:
            raise ValueError("Context must be a positive integer")
        self.path, self.context = Path(model_path), context
        self.model_path = self.path
        if self.path.is_symlink() or not self.path.is_file():
            raise ValueError("Runtime model must be a regular file")
        self.sha256 = sha256(self.path)
        if artifact is not None and (Path(artifact.source).resolve() != self.path.resolve()
                                     or artifact.role != "model" or artifact.format != "tflite"
                                     or (artifact.expected_sha256 is not None and artifact.expected_sha256 != self.sha256)):
            raise ValueError("Declared artifact differs from the runtime model")
        self.runner = Interpreter(model_path=str(self.path), num_threads=1)
        self.runner.allocate_tensors()
        inputs, outputs = self.runner.get_input_details(), self.runner.get_output_details()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Detection runtime requires one input and one output")
        self.input, self.output = inputs[0], outputs[0]
        self.input_details, self.output_details = self.input, self.output
        if (tuple(self.input["shape"]) != (1, context, 5)
                or tuple(self.output["shape"]) != (1, context, 2)
                or np.dtype(self.input["dtype"]) != np.dtype(self.output["dtype"])
                or np.dtype(self.input["dtype"]) not in (np.dtype("float32"), np.dtype("int8"))):
            raise ValueError("Incompatible runtime tensor contract")
        self.integer = np.dtype(self.input["dtype"]) == np.dtype("int8")
        self.input_scale, self.input_zero = _quantization(self.input, self.integer)
        self.output_scale, self.output_zero = _quantization(self.output, self.integer)
        if artifact is not None:
            for specs, tensor, scale, zero in (
                (artifact.inputs, self.input, self.input_scale, self.input_zero),
                (artifact.outputs, self.output, self.output_scale, self.output_zero),
            ):
                if len(specs) != 1:
                    raise ValueError("Declared artifact needs complete tensor signatures")
                spec = specs[0]
                signature = tuple(None if int(d) == -1 else int(d) for d in tensor["shape_signature"])
                if (spec.name != tensor["name"] or tuple(spec.shape) != signature
                        or spec.dtype != np.dtype(tensor["dtype"]).name
                        or spec.scale != scale or spec.zero_point != zero):
                    raise ValueError("Declared tensor signature or quantization differs from runtime")
        self.verify_unchanged()

    def verify_unchanged(self):
        if self.path.is_symlink() or not self.path.is_file() or sha256(self.path) != self.sha256:
            raise ValueError("Runtime model changed during execution")

    def predict(self, values):
        """Return logits and counters; raw output endpoints need not imply clipping.

        ``clipped_input_values`` counts rounded integers outside the int8 range.
        ``output_saturated_values`` only counts encoded -128/127 endpoints.
        ``raw_output`` retains the exact returned dtype before dequantization.
        """
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (self.context, 5) or not np.isfinite(values).all():
            raise ValueError("Expected finite input with shape (context, 5)")
        clipped = 0
        if self.integer:
            quantized = np.rint(values.astype(np.float64) / self.input_scale + self.input_zero)
            clipped = int(np.count_nonzero((quantized < -128) | (quantized > 127)))
            encoded = np.clip(quantized, -128, 127).astype(np.int8)
        else:
            encoded = values
        self.runner.set_tensor(self.input["index"], encoded[None])
        self.runner.invoke()
        raw = self.runner.get_tensor(self.output["index"])
        if (raw.shape != (1, self.context, 2) or raw.dtype != self.output["dtype"]
                or not np.isfinite(raw).all()):
            raise ValueError("Runtime output differs from its finite tensor contract")
        raw = raw[0].copy()
        logits = ((raw.astype(np.float64) - self.output_zero) * self.output_scale).astype(np.float32) if self.integer else raw.copy()
        if not np.isfinite(logits).all():
            raise ValueError("Runtime produced nonfinite dequantized logits")
        return logits, {
            "input_values": int(values.size), "clipped_input_values": clipped,
            "argmax_ties": int(np.count_nonzero(raw[:, 0] == raw[:, 1])),
            "output_saturated_values": int(np.count_nonzero((raw == -128) | (raw == 127))) if self.integer else 0,
            "raw_output": raw,
        }

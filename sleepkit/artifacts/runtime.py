"""Optional LiteRT conformance checks, independent of training and task metrics."""

from importlib.metadata import version
from pathlib import Path

from .package import sha256, validate_bundle
from .schema import Artifact, Check, TensorSpec


def _interpreter(artifact):
    import numpy as np
    from ai_edge_litert.interpreter import Interpreter

    runner = Interpreter(model_path=str(artifact.source))
    runner.allocate_tensors()
    for specs, details in (
        (artifact.inputs, runner.get_input_details()),
        (artifact.outputs, runner.get_output_details()),
    ):
        if not specs or len(specs) != len(details):
            raise ValueError("Runtime requires complete input and output signatures")
        for spec, tensor in zip(specs, details):
            shape = tuple(None if int(d) == -1 else int(d) for d in tensor["shape_signature"])
            if (
                spec.name != tensor["name"]
                or tuple(spec.shape) != shape
                or spec.dtype != np.dtype(tensor["dtype"]).name
            ):
                raise ValueError(f"Runtime tensor signature differs: {spec.name}")
            quant = tensor["quantization_parameters"]
            scales, zeros = quant["scales"], quant["zero_points"]
            if len(scales) > 1:
                raise ValueError("Runtime adapter supports per-tensor I/O quantization only")
            scale, zero = (float(scales[0]), int(zeros[0])) if len(scales) else (None, None)
            if spec.scale != scale or spec.zero_point != zero:
                raise ValueError(f"Runtime quantization differs: {spec.name}")
    return runner


def create_reference(artifact, destination):
    """Record deterministic synthetic I/O. This establishes no task accuracy."""
    import numpy as np

    runner = _interpreter(artifact)
    rng = np.random.default_rng(0)
    arrays = {}
    for i, (spec, tensor) in enumerate(zip(artifact.inputs, runner.get_input_details())):
        values = rng.uniform(-0.5, 0.5, size=tuple(tensor["shape"]))
        if spec.scale is not None:
            limits = np.iinfo(tensor["dtype"])
            values = np.clip(np.rint(values / spec.scale + spec.zero_point), limits.min, limits.max)
        values = values.astype(tensor["dtype"])
        arrays[f"input_{i}"] = values
        runner.set_tensor(tensor["index"], values)
    runner.invoke()
    for i, tensor in enumerate(runner.get_output_details()):
        values = runner.get_tensor(tensor["index"])
        if not np.isfinite(values).all():
            raise ValueError("Runtime produced nonfinite outputs")
        arrays[f"output_{i}"] = values
    destination = Path(destination)
    with destination.open("wb") as stream:
        np.savez(stream, **arrays)
    return Check(
        f"runtime_conformance:{artifact.name}",
        "passed",
        "Signature and finite synthetic inference verified; no accuracy or preprocessing equivalence claim.",
        {artifact.name: sha256(artifact.source), destination.name: sha256(destination)},
        {
            "runtime": "ai-edge-litert",
            "version": version("ai-edge-litert"),
            "seed": 0,
            "reference": destination.name,
            "integer_atol": 1,
            "float_atol": 1e-6,
            "float_rtol": 1e-5,
        },
    )


def replay_bundle(path):
    """Replay every declared model against its recorded, hash-bound synthetic I/O."""
    import numpy as np

    path = Path(path)
    report = validate_bundle(path, profile="runnable")
    results = []
    for entry in report["manifest"]["artifacts"]:
        if entry["role"] != "model":
            continue
        if entry["format"] != "tflite":
            raise ValueError(f"No runtime adapter for {entry['format']}")
        artifact = Artifact(
            path / entry["path"],
            entry["path"],
            entry["role"],
            entry["format"],
            entry["origin"],
            tuple(TensorSpec(**t) for t in entry["inputs"]),
            tuple(TensorSpec(**t) for t in entry["outputs"]),
        )
        check = next(c for c in report["checks"] if c["name"] == f"runtime_conformance:{artifact.name}")
        reference = check["details"].get("reference")
        if reference not in check["artifacts"] or not any(
            a["path"] == reference and a["role"] == "reference" for a in report["manifest"]["artifacts"]
        ):
            raise ValueError("Runtime reference must be a hash-bound reference artifact")
        runner = _interpreter(artifact)
        with np.load(path / reference, allow_pickle=False) as arrays:
            inputs, outputs = runner.get_input_details(), runner.get_output_details()
            if set(arrays.files) != {f"input_{i}" for i in range(len(inputs))} | {
                f"output_{i}" for i in range(len(outputs))
            }:
                raise ValueError("Reference keys differ from runtime signature")
            for i, tensor in enumerate(inputs):
                values = arrays[f"input_{i}"]
                if (
                    values.shape != tuple(tensor["shape"])
                    or values.dtype != tensor["dtype"]
                    or not np.isfinite(values).all()
                ):
                    raise ValueError("Invalid reference input")
                runner.set_tensor(tensor["index"], values)
            runner.invoke()
            for i, tensor in enumerate(outputs):
                actual, expected = runner.get_tensor(tensor["index"]), arrays[f"output_{i}"]
                if (
                    actual.shape != expected.shape
                    or actual.dtype != expected.dtype
                    or not np.isfinite(actual).all()
                    or not np.isfinite(expected).all()
                ):
                    raise ValueError("Reference output signature differs")
                integer = np.issubdtype(actual.dtype, np.integer)
                if not np.allclose(
                    actual.astype(float),
                    expected.astype(float),
                    atol=1 if integer else 1e-6,
                    rtol=0 if integer else 1e-5,
                    equal_nan=False,
                ):
                    raise ValueError("Runtime output differs from reference")
        results.append(
            {
                "model": artifact.name,
                "status": "passed",
                "runtime": "ai-edge-litert",
                "version": version("ai-edge-litert"),
            }
        )
    return results

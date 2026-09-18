"""Versioned feature/raw-record inference bundles."""

import hashlib
import json
from pathlib import Path

import numpy as np

from sleepkit.preprocessing import Standardizer, WindowFeatures, model_windows, prepare


REQUIRED_FILES = {
    "model.tflite",
    "preprocessing.json",
    "manifest.json",
    "reference_vectors.npz",
    "metrics.json",
    "README.md",
}


def validate_bundle(path):
    path = Path(path)
    checksums = json.loads((path / "checksums.json").read_text())
    if not REQUIRED_FILES <= checksums.keys():
        raise ValueError("Bundle checksums omit required artifacts")
    for name, digest in checksums.items():
        if Path(name).name != name or not (path / name).is_file():
            raise ValueError(f"Invalid bundle artifact: {name}")
        if hashlib.sha256((path / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Checksum mismatch: {name}")
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("output") != "logits":
        raise ValueError("Unsupported bundle contract")
    return manifest


class Predictor:
    @classmethod
    def from_pretrained(cls, repo_id, *, revision):
        """Download an explicitly selected Hub revision, then validate locally."""
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=[*REQUIRED_FILES, "model.keras", "LICENSE", "checksums.json"],
        )
        return cls(path)

    def __init__(self, path):
        from ai_edge_litert.interpreter import Interpreter

        self.path = Path(path)
        self.manifest = validate_bundle(path)
        state = json.loads((self.path / "preprocessing.json").read_text())
        self.preprocessing = WindowFeatures.from_dict(state["transform"])
        self.normalizer = Standardizer.from_dict(state["normalizer"])
        if self.normalizer.mean.shape != (len(self.preprocessing.spec["feature_names"]),):
            raise ValueError("Normalization shape differs from feature contract")
        self.interpreter = Interpreter(model_path=str(self.path / "model.tflite"))
        self.interpreter.allocate_tensors()
        if len(self.interpreter.get_input_details()) != 1 or len(self.interpreter.get_output_details()) != 1:
            raise ValueError("This bundle contract requires one input and one output")
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        expected = [1, self.manifest["context"], len(self.preprocessing.spec["feature_names"])]
        if self.input["shape"].tolist() != expected:
            raise ValueError("Model input shape differs from preprocessing contract")
        if self.output["shape"].tolist() != [1, self.manifest["context"], len(self.manifest["class_names"])]:
            raise ValueError("Model output shape differs from class contract")

    def predict_features(self, values):
        """Infer logits from normalized, ordered model inputs."""
        values = np.asarray(values, dtype=np.float32)
        expected = tuple(self.input["shape"][1:])
        if values.ndim != 3 or values.shape[1:] != expected or not np.isfinite(values).all():
            raise ValueError(f"Expected finite [batch, {expected[0]}, {expected[1]}] input")
        outputs = []
        for frame in values:
            sample = frame[None]
            if np.issubdtype(self.input["dtype"], np.integer):
                scale, zero = self.input["quantization"]
                if scale <= 0:
                    raise ValueError("Invalid input quantization scale")
                limits = np.iinfo(self.input["dtype"])
                sample = np.clip(np.rint(sample / scale + zero), limits.min, limits.max)
            self.interpreter.set_tensor(self.input["index"], sample.astype(self.input["dtype"]))
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output["index"]).astype(np.float32)
            if np.issubdtype(self.output["dtype"], np.integer):
                scale, zero = self.output["quantization"]
                if scale <= 0:
                    raise ValueError("Invalid output quantization scale")
                output = (output - zero) * scale
            outputs.append(output[0])
        return np.asarray(outputs, dtype=np.float32).reshape(len(values), *self.output["shape"][1:])

    def predict_record(self, record):
        """Apply saved preprocessing without annotations; return logits and times."""
        prepared = self.normalizer.transform(prepare(record, self.preprocessing))
        x, _, times = model_windows(prepared, self.manifest["context"], require_targets=False)
        return self.predict_features(x), times

    def check_reference(self):
        with np.load(self.path / "reference_vectors.npz", allow_pickle=False) as vectors:
            actual = self.predict_features(vectors["inputs"])
            if not np.allclose(actual, vectors["outputs"], atol=1e-5, rtol=1e-5):
                raise ValueError("Reference inference mismatch")
        return True

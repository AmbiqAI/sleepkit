"""A small backend-portable signal recipe with explicit artifact contracts.

This is a synthetic contract example.  It exercises one heliaEDGE layer and
custom-object reload without making a domain or quality claim.
"""

from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import tempfile

import numpy as np

from sleepkit.artifacts import Artifact, Check, TensorSpec, stage_bundle
from sleepkit.artifacts.package import sha256, write_json
from sleepkit.recipes._components import ClassificationAccumulator, FileSnapshot, implementation_files


LENGTH = 128
CLASSES = 3
CLASS_ORDER = ["one_cycle", "two_cycles", "three_cycles"]
SPLIT_SIZES = {"train": 75, "validation": 30, "test": 30}
PREPROCESSING = {
    "input": "finite float32 waveform [examples, 128]",
    "output": "per-waveform centered/unit-population-scale float32 features [examples, 128, 1]",
    "minimum_population_std_exclusive": 1e-8,
    "statistics_dtype": "float64",
}


@dataclass(frozen=True)
class Config:
    batch_size: int = 16
    epochs: int = 2
    seed: int = 0

    def __post_init__(self):
        if any(type(value) is not int or value < 1 for value in (self.batch_size, self.epochs)):
            raise ValueError("batch_size and epochs must be positive integers")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")


def make_data(seed):
    """Create deterministic train/validation/test raw waveforms and integer targets."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    time = np.arange(LENGTH, dtype=np.float32) / LENGTH
    data = {}
    for split, count in SPLIT_SIZES.items():
        targets = np.arange(count, dtype=np.int32) % CLASSES
        rng.shuffle(targets)
        phase = rng.uniform(-np.pi, np.pi, count).astype(np.float32)
        frequency = (targets + 1).astype(np.float32)
        raw = np.sin(2 * np.pi * frequency[:, None] * time[None, :] + phase[:, None])
        raw += 0.25 * np.cos(2 * np.pi * (frequency + 1)[:, None] * time[None, :])
        raw += rng.normal(0, 0.06, raw.shape).astype(np.float32)
        data[split] = {"raw": raw.astype(np.float32), "targets": targets}
    return data


def prepare(raw):
    """Convert finite [examples, 128] waveforms into deterministic [examples, 128, 1] features."""
    values = np.asarray(raw, dtype=np.float32)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 2 or values.shape[1] != LENGTH:
        raise ValueError(f"Expected raw signals with shape [examples, {LENGTH}]")
    if not np.isfinite(values).all():
        raise ValueError("Raw signals must be finite")
    mean = values.mean(axis=1, keepdims=True, dtype=np.float64)
    scale = values.std(axis=1, keepdims=True, dtype=np.float64)
    if (scale <= 1e-8).any() or not np.isfinite(scale).all():
        raise ValueError("Raw signals must have nonzero finite variation")
    features = ((values.astype(np.float64) - mean) / scale).astype(np.float32)
    return features[..., None]


def _split(data, name):
    if not isinstance(data, dict) or name not in data:
        raise ValueError(f"Missing data split: {name}")
    item = data[name]
    if not isinstance(item, dict) or set(item) != {"raw", "targets"}:
        raise ValueError(f"Split {name} must contain raw and targets")
    raw = np.asarray(item["raw"], dtype=np.float32)
    targets = np.asarray(item["targets"])
    if raw.ndim != 2 or raw.shape[1] != LENGTH or targets.shape != (len(raw),):
        raise ValueError(f"Split {name} has incompatible shapes")
    if targets.dtype.kind not in "iu" or not np.isin(targets, np.arange(CLASSES)).all():
        raise ValueError(f"Split {name} targets must be integers in [0, {CLASSES})")
    if not np.isfinite(raw).all():
        raise ValueError(f"Split {name} contains nonfinite raw signals")
    return raw, targets.astype(np.int32, copy=False)


def _numpy(value):
    """Materialize a backend tensor without requiring a backend-specific import."""
    if hasattr(value, "detach"):
        value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
    return np.asarray(value)


def _validate_model(model):
    """The archive supports a single variable-batch float32 input and linear logits."""
    import keras

    inputs, outputs = getattr(model, "inputs", []), getattr(model, "outputs", [])
    layers = getattr(model, "layers", [])
    if (
        len(inputs) != 1 or len(outputs) != 1 or not layers
        or tuple(inputs[0].shape) != (None, LENGTH, 1)
        or tuple(outputs[0].shape) != (None, CLASSES)
        or inputs[0].dtype != "float32" or outputs[0].dtype != "float32"
        or getattr(layers[-1], "activation", None) is not keras.activations.linear
    ):
        raise ValueError("Model must map float32 [batch, 128, 1] to float32 [batch, 3] linear logits")


def _reference_logits(model, inputs):
    logits = _numpy(model(inputs, training=False))
    if logits.shape != (len(inputs), CLASSES) or logits.dtype != np.float32 or not np.isfinite(logits).all():
        raise ValueError("Reference output must be finite float32 [batch, 3] logits")
    return logits


def build_model():
    """Build the Keras model while using heliaEDGE's registered PatchLayer2D."""
    import keras
    from helia_edge.layers import PatchLayer2D

    inputs = keras.Input((LENGTH, 1), name="signal")
    image = keras.layers.Reshape((LENGTH, 1, 1), name="signal_image")(inputs)
    patches = PatchLayer2D(LENGTH, 1, 1, 16, 1, name="patches")(image)
    hidden = keras.layers.Dense(16, activation="relu", name="hidden")(keras.layers.Flatten()(patches))
    outputs = keras.layers.Dense(CLASSES, name="logits")(hidden)
    return keras.Model(inputs, outputs, name="portable_signal_classifier")


def train(data, config=Config(), model=None):
    """Train finite epochs with retained tail batches and return ``(model, history)``."""
    import keras

    config = config if isinstance(config, Config) else Config(**config)
    train_raw, train_targets = _split(data, "train")
    validation_raw, validation_targets = _split(data, "validation")
    keras.utils.set_random_seed(config.seed)
    model = build_model() if model is None else model
    _validate_model(model)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    history = model.fit(
        prepare(train_raw),
        train_targets,
        validation_data=(prepare(validation_raw), validation_targets),
        batch_size=config.batch_size,
        epochs=config.epochs,
        shuffle=True,
        verbose=0,
    )
    return model, {key: [float(value) for value in values] for key, values in history.history.items()}


def evaluate(model, data, split="test"):
    """Evaluate one split with the shared unweighted classification accumulator."""
    raw, targets = _split(data, split)
    logits = _numpy(model(prepare(raw), training=False))
    accumulator = ClassificationAccumulator(CLASSES)
    accumulator.update(targets, logits)
    return {"split": split, "model": "model.keras", **accumulator.result()}


def _data_fingerprint(data):
    digest = hashlib.sha256()
    for split in ("train", "validation", "test"):
        raw, targets = _split(data, split)
        digest.update(split.encode())
        digest.update(json.dumps({"raw": list(raw.shape), "targets": list(targets.shape)}, sort_keys=True).encode())
        digest.update(np.ascontiguousarray(raw, dtype="<f4").tobytes())
        digest.update(np.ascontiguousarray(targets, dtype="<i4").tobytes())
    return digest.hexdigest()


def _edge_metadata():
    try:
        distribution = importlib.metadata.distribution("helia-edge")
        direct = distribution.read_text("direct_url.json")
        direct_url = None if direct is None else json.loads(direct)
        edge_version = distribution.version
    except importlib.metadata.PackageNotFoundError:
        direct_url, edge_version = None, None
    packages = {}
    for name in ("keras", "numpy", "tensorflow", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "helia_edge": edge_version,
        "helia_edge_direct_url": direct_url,
        "backend": _keras_backend(),
        "python": platform.python_version(),
        "packages": packages,
    }


def _keras_backend():
    import keras

    return keras.backend.backend()


def _load_model(path):
    """Reload through heliaEDGE so custom-object registration is exercised."""
    from helia_edge.models import load_model

    return load_model(path)


def package(model, data, output, config=Config(), history=None, *, _code_snapshot=None):
    """Save model, metrics, config, and reload vectors into an archive-profile bundle."""
    config = config if isinstance(config, Config) else Config(**config)
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}")
    _validate_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    code_snapshot = _code_snapshot or FileSnapshot.capture(implementation_files(Path(__file__).parent))
    metrics = evaluate(model, data)
    test_raw, test_targets = _split(data, "test")
    reference_inputs = prepare(test_raw[: min(6, len(test_raw))])
    reference_logits = _reference_logits(model, reference_inputs)
    environment = _edge_metadata()
    data_hash = _data_fingerprint(data)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".signals-") as directory:
        stage = Path(directory)
        model_path = stage / "model.keras"
        model.save(model_path)
        reference_npz = stage / "reference.npz"
        np.savez_compressed(reference_npz, inputs=reference_inputs, logits=reference_logits, targets=test_targets[: len(reference_inputs)])
        write_json(
            stage / "reference.json",
            {"count": len(reference_inputs), "input_shape": list(reference_inputs.shape), "output_shape": list(reference_logits.shape), "atol": 1e-5, "rtol": 1e-4, "npz_sha256": sha256(reference_npz)},
        )
        write_json(
            stage / "config.json",
            {"config": asdict(config), "data_fingerprint": data_hash, "environment": environment, "preprocessing": PREPROCESSING,
             "class_order": CLASS_ORDER, "split_counts": {name: len(_split(data, name)[0]) for name in SPLIT_SIZES}},
        )
        write_json(stage / "preprocessing.json", PREPROCESSING)
        write_json(stage / "history.json", history or {})
        restored = _load_model(model_path)
        _validate_model(restored)
        restored_logits = _reference_logits(restored, reference_inputs)
        if not np.allclose(restored_logits, reference_logits, atol=1e-5, rtol=1e-4):
            raise ValueError("heliaEDGE/Keras model reload changed reference logits")
        code_snapshot.verify()
        artifacts = [
            Artifact(model_path, "model.keras", "model", "keras", "Synthetic signal recipe model using heliaEDGE PatchLayer2D", (TensorSpec("signal", (None, LENGTH, 1), "float32", "prepared waveform"),), (TensorSpec("logits", (None, CLASSES), "float32", "class logits"),)),
            Artifact(stage / "metrics.json", "metrics.json", "metrics", "json", "Synthetic held-out evaluation metrics"),
            Artifact(stage / "config.json", "config.json", "config", "json", "Synthetic recipe configuration and environment"),
            Artifact(stage / "preprocessing.json", "preprocessing.json", "preprocessing", "json", "Per-waveform normalization contract"),
            Artifact(stage / "history.json", "history.json", "metrics", "json", "Finite training history with retained tail batches"),
            Artifact(stage / "reference.json", "reference.json", "reference", "json", "Reload reference metadata"),
            Artifact(reference_npz, "reference.npz", "reference", "npz", "Synthetic model reload vectors"),
        ]
        write_json(stage / "metrics.json", {**metrics, "data_fingerprint": data_hash, "environment": environment})
        check = Check(
            "model_reload",
            "passed",
            "Reloaded model matches saved reference logits within tolerance",
            {"model.keras": sha256(model_path), "reference.npz": sha256(reference_npz)},
        )
        card = """# Synthetic signal classification

This integration example uses generated noisy waveforms, not a sleep dataset.
Class indices 0, 1, 2 represent base frequencies of one, two, and three cycles
per 128-sample window. Outputs are three unnormalized logits in that order.
The model consumes float32 [batch, 128, 1] features. Normalize each raw waveform
using its own float64 mean and population standard deviation, rejecting standard
deviations <= 1e-8, then cast to float32. See preprocessing.json.

With the recorded Keras backend selected before imports and the EDGE version
from config.json installed, load the archive with:

```python
from helia_edge.models import load_model
model = load_model("model.keras")
logits = model(prepared_waveforms, training=False)
```

Reference inputs/logits are in reference.npz. This is an archive-profile bundle;
no LiteRT conversion, hardware compatibility, or real-domain quality is claimed.
"""
        metadata = {"recipe": "sleepkit.signals/v1", "config": asdict(config), "data_fingerprint": data_hash, "environment": environment, "implementation_sha256": code_snapshot.hashes()}
        return stage_bundle(output, title="sleepKIT synthetic signal classification", artifacts=artifacts, card=card, checks=[check], metadata=metadata)


def run(output, config=Config()):
    """Generate, train, evaluate, and package one synthetic recipe run."""
    config = config if isinstance(config, Config) else Config(**config)
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}")
    code_snapshot = FileSnapshot.capture(implementation_files(Path(__file__).parent))
    data = make_data(config.seed)
    model, history = train(data, config)
    return package(model, data, output, config, history, _code_snapshot=code_snapshot)

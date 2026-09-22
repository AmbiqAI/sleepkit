"""A fixed saved-feature training flow, composed from independently usable blocks."""

from dataclasses import asdict, dataclass
import importlib.metadata
import math
import os
from pathlib import Path
import platform
import subprocess
import tempfile

import numpy as np

from sleepkit.artifacts import Artifact, Check, TensorSpec, stage_bundle
from sleepkit.artifacts.package import sha256, write_json
from sleepkit.recipes._components import ClassificationAccumulator, FileSnapshot, implementation_files

from .data import prepare_partition
from .evaluation import evaluate_windows, validate_model
from .model import build_model
from .preprocessing import CLASS_ORDER, FEATURE_ORDER
from .split import PARTITIONS, fingerprint, load_split, read_json

SCHEMA = "sleepkit.staging_training/v1"
GOLDEN_SCHEMA = "sleepkit.staging_golden/v1"
PREPROCESSING = {
    "kind": "sleepkit.fs-w-pa-14-60/offline/v1",
    "feature_order": list(FEATURE_ORDER),
    "class_order": list(CLASS_ORDER),
    "feature_window_seconds": 60,
    "epoch_seconds": 30,
    "context_epochs": 240,
    "normalization": "whole-subject float32 median imputation, mean, population variance; epsilon=1e-6",
    "scoring": "quality-valid stages 0..5; map 0->WAKE,1..4->NREM,5->REM; stage6 unknown",
    "windowing": "complete nonoverlapping contexts; preserve gaps; omit incomplete tail",
    "source": "preprocessing.py is the executable authority; raw feature semantics asserted by caller",
}


@dataclass(frozen=True)
class Config:
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.001
    seed: int = 0

    def __post_init__(self):
        if any(type(n) is not int or n < 1 for n in (self.epochs, self.batch_size)):
            raise ValueError("Epochs and batch size must be positive integers")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Seed must be an integer in [0, 2**32)")
        if (
            type(self.learning_rate) not in (float, int)
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError("Learning rate must be finite and positive")


def environment():
    import keras

    backend = keras.backend.backend()
    if backend not in {"tensorflow", "torch"}:
        raise ValueError("This recipe supports the TensorFlow and Torch Keras backends")
    return {
        "python": platform.python_version(),
        "backend": backend,
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "TF_NUM_INTRAOP_THREADS",
                "TF_NUM_INTEROP_THREADS",
                "OMP_NUM_THREADS",
                "TF_DETERMINISTIC_OPS",
                "CUDA_VISIBLE_DEVICES",
            )
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("keras", "numpy", "h5py", "tensorflow" if backend == "tensorflow" else "torch")
        },
    }


def revision():
    """Best-effort checkout identity; implementation hashes remain authoritative."""
    try:
        directory = Path(__file__).parent
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=directory, stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=directory, stderr=subprocess.DEVNULL, text=True
            ).strip()
        )
        return {"commit": head, "dirty": dirty}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def train(training, validation, config=Config(), *, model_builder=build_model):
    """Fit every supervised context once per epoch using Keras array adapters.

    Builder replacement is an experimental Python extension. The promoted run
    below deliberately fixes its builder so its implementation is fully recorded.
    """
    import keras

    config = config if isinstance(config, Config) else Config(**config)
    environment()
    x, y, weights = training.fit_arrays()
    validation_arrays = validation.fit_arrays()
    keras.utils.set_random_seed(config.seed)
    model = model_builder()
    validate_model(model)
    model.compile(
        optimizer=keras.optimizers.Adam(config.learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="mean_with_sample_weight"),
        jit_compile=False,
    )
    history = model.fit(
        x,
        y,
        sample_weight=weights,
        validation_data=validation_arrays,
        epochs=config.epochs,
        batch_size=config.batch_size,
        shuffle=True,
        verbose=0,
    )
    values = {key: [float(v) for v in series] for key, series in history.history.items()}
    if any(len(series) != config.epochs or not np.isfinite(series).all() for series in values.values()):
        raise ValueError("Training did not produce the declared finite epoch history")
    if not values or any(not np.isfinite(np.asarray(keras.ops.convert_to_numpy(w))).all() for w in model.weights):
        raise ValueError("Missing training history or nonfinite model weights")
    return model, values


def evaluate(model, partition, batch_size=32):
    """Pooled scored-epoch metrics; unlike Keras history, not averaged batch means."""
    partition.fit_arrays()  # Validate caller-constructed partitions too.
    accumulator = ClassificationAccumulator(len(CLASS_ORDER))
    evaluate_windows(model, partition, accumulator, batch_size)
    return {"class_order": list(CLASS_ORDER), **accumulator.result()}


def _identity(root, split_file, config):
    manifest_snapshot = FileSnapshot.capture({"split.json": split_file})
    manifest = load_split(split_file)
    source = FileSnapshot.capture(
        {s: Path(root) / f"{s}.h5" for group in manifest["partitions"].values() for s in group}
    )
    code = FileSnapshot.capture(implementation_files(Path(__file__).parent))
    identity = {
        "recipe": SCHEMA,
        "model": "staging_conv1d_v1",
        "config": asdict(config),
        "dataset": manifest["dataset"],
        "split_sha256": fingerprint(manifest),
        "source_inventory_sha256": fingerprint(source.hashes()),
        "implementation_sha256": code.hashes(),
        "environment": environment(),
    }
    return manifest, manifest_snapshot, source, code, identity


def _verify(manifest, source, code):
    for snapshot in (manifest, source, code):
        snapshot.verify()
    if set(implementation_files(Path(__file__).parent)) != set(code.hashes()):
        raise ValueError("Training implementation inventory changed")


def declare_golden(root, split_file, output, config=Config(), *, name):
    """Freeze prospective identities/configuration without training or reading labels."""
    config = config if isinstance(config, Config) else Config(**config)
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Golden name must be nonempty")
    _, manifest, source, code, identity = _identity(root, split_file, config)
    _verify(manifest, source, code)
    value = {"schema": GOLDEN_SCHEMA, "name": name, "expected": identity}
    with Path(output).open("x") as stream:
        import json

        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return value


def _logits(model, inputs):
    import keras

    result = np.asarray(keras.ops.convert_to_numpy(model(inputs, training=False)))
    if result.shape != (*inputs.shape[:2], 3) or result.dtype != np.float32 or not np.isfinite(result).all():
        raise ValueError("Expected finite float32 staging logits")
    return result


def _package(model, partitions, output, metadata, verify):
    import keras

    with tempfile.TemporaryDirectory(dir=Path(output).parent, prefix=".staging-") as directory:
        stage = Path(directory)
        model.save(stage / "model.keras")
        restored = keras.models.load_model(stage / "model.keras", compile=False, safe_mode=True)
        validate_model(restored)
        # Real held-out logits remain in memory; only synthetic vectors enter the bundle.
        max_error = 0.0
        for start in range(0, len(partitions["test"].features), 32):
            inputs = partitions["test"].features[start : start + 32]
            original, reloaded = _logits(model, inputs), _logits(restored, inputs)
            max_error = max(max_error, float(np.max(np.abs(original - reloaded))))
            if not np.allclose(original, reloaded, atol=1e-5, rtol=1e-4):
                raise ValueError("Checkpoint reload changed held-out logits")
        metrics = {name: evaluate(restored, partitions[name]) for name in ("validation", "test")}
        reference = np.random.default_rng(0).uniform(-1, 1, (2, 240, 14)).astype(np.float32)
        np.savez_compressed(stage / "reference.npz", inputs=reference, logits=_logits(restored, reference))
        write_json(stage / "recipe.json", metadata)
        write_json(stage / "metrics.json", metrics)
        write_json(stage / "preprocessing.json", PREPROCESSING)
        write_json(
            stage / "reference.json",
            {
                "kind": "synthetic_normalized_features",
                "atol": 1e-5,
                "rtol": 1e-4,
                "held_out_reload_max_abs_error": max_error,
            },
        )
        import shutil

        shutil.copyfile(Path(__file__).with_name("preprocessing.py"), stage / "preprocessing.py")
        artifacts = [
            Artifact(
                stage / "model.keras",
                "model.keras",
                "model",
                "keras",
                "New fixed-final-epoch staging model",
                (TensorSpec("features", (None, 240, 14), "float32", "offline-normalized FS-W-PA-14-60"),),
                (TensorSpec("logits", (None, 240, 3), "float32", "WAKE, NREM, REM logits"),),
            )
        ]
        for name, role, fmt in (
            ("recipe.json", "recipe_metadata", "json"),
            ("metrics.json", "evaluation", "json"),
            ("preprocessing.json", "preprocessing_state", "json"),
            ("preprocessing.py", "preprocessing", "python"),
            ("reference.npz", "reference", "npz"),
            ("reference.json", "reference", "json"),
        ):
            artifacts.append(
                Artifact(stage / name, name, role, fmt, "Staging recipe evidence; no subject arrays or IDs")
            )
        check = Check(
            "keras_reload",
            "passed",
            "Held-out logits match before and after save/reload",
            {"model.keras": sha256(stage / "model.keras"), "reference.npz": sha256(stage / "reference.npz")},
            {"atol": 1e-5, "rtol": 1e-4, "max_abs_error": max_error},
        )
        card = """# Saved-feature three-class staging

New model trained with explicit disjoint subject membership; fixed final epoch.
Inputs: float32 [batch,240,14] offline-normalized FS-W-PA-14-60 features.
Outputs: float32 [batch,240,3] linear logits, ordered WAKE, NREM, REM.
The caller must supply the documented feature order/cadence; raw extraction
provenance is unverified. Whole-subject normalization uses future epochs/tails.
Subject disjointness relies on the dataset provider's filename identity grouping.
This is not a causal streaming model. Model/dataset usage terms require separate
review; this archive is not an approved HF release or deployment qualification.

Select the recorded Keras backend before importing Keras and install the recorded
packages from recipe.json. From this bundle directory:

```python
import sys
sys.dont_write_bytecode = True  # Keep the checksum-bound archive unchanged.
import keras
from preprocessing import read_subject, prepare_subject, window_subject
model = keras.models.load_model("model.keras", compile=False, safe_mode=True)
windows = window_subject(prepare_subject(read_subject("/private/subject.h5")))
logits = model(windows.features, training=False)
# Score only windows.scoring_mask positions; preserve all input context.
```

reference.npz contains synthetic normalized inputs/logits, not dataset examples.
preprocessing.py is executable; preprocessing.json describes its input contract.
No TFLite/hardware parity, clinical claim or historical-model equivalence claimed.
"""
        verify()
        result = stage_bundle(
            output,
            title="sleepKIT saved-feature staging",
            artifacts=artifacts,
            card=card,
            checks=[check],
            metadata=metadata,
        )
    return result, metrics


def run(root, split_file, output, config=Config(), *, golden=None):
    """Declare, prepare, train, evaluate and archive one fixed-model experiment.

    Output is private run evidence; only its bundle excludes subject identifiers.
    A failed run may leave a declaration/intermediate files, never a completion marker.
    """
    config = config if isinstance(config, Config) else Config(**config)
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    manifest, manifest_snapshot, source, code, identity = _identity(root, split_file, config)
    golden_snapshot = None
    if golden is not None:
        golden_snapshot = FileSnapshot.capture({"golden.json": golden})
        definition = read_json(golden)
        if (
            not isinstance(definition, dict)
            or set(definition) != {"schema", "name", "expected"}
            or definition["schema"] != GOLDEN_SCHEMA
            or definition["expected"] != identity
        ):
            raise ValueError("Golden experiment differs from current config/data/code/environment")
        golden_snapshot.verify()
    output.mkdir(parents=True)
    declaration = {
        **identity,
        "git": revision(),
        "split_file_sha256": manifest_snapshot.hashes()["split.json"],
        "golden_sha256": None if golden_snapshot is None else golden_snapshot.hashes()["golden.json"],
        "selection": "fixed final epoch; no test-driven model selection",
        "preprocessing": PREPROCESSING,
    }
    write_json(output / "declaration.json", declaration)
    write_json(output / "split.json", manifest)
    write_json(output / "sources.json", source.hashes())
    partitions = {name: prepare_partition(root, manifest["partitions"][name]) for name in PARTITIONS}

    private_snapshot = None

    def verify():
        _verify(manifest_snapshot, source, code)
        if golden_snapshot is not None:
            golden_snapshot.verify()
        if private_snapshot is not None:
            private_snapshot.verify()

    verify()
    write_json(output / "windows.json", {name: item.coordinates for name, item in partitions.items()})
    private_snapshot = FileSnapshot.capture(
        {name: output / name for name in ("declaration.json", "split.json", "sources.json", "windows.json")}
    )
    model, history = train(partitions["train"], partitions["validation"], config)
    verify()
    write_json(output / "history.json", history)
    private_snapshot = FileSnapshot.capture(
        {
            name: output / name
            for name in ("declaration.json", "split.json", "sources.json", "windows.json", "history.json")
        }
    )
    metadata = {
        **declaration,
        "coverage": {name: item.coverage for name, item in partitions.items()},
        "history": history,
        "sampling": "each supervised window once per epoch; shuffle; retain partial batch",
        "history_loss": "Keras batch-normalized aggregation; final metrics use pooled scored epochs",
    }
    result, metrics = _package(model, partitions, output / "bundle", metadata, verify)
    verify()
    write_json(
        output / "completed.json",
        {
            "recipe": SCHEMA,
            "metrics": metrics,
            "bundle_manifest_sha256": sha256(output / "bundle/manifest.json"),
            "declaration_sha256": sha256(output / "declaration.json"),
        },
    )
    return result

"""Export validated Keras/LiteRT bundles; publish independently of training."""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from sleepkit.preprocessing import WindowFeatures
from sleepkit.runtime import Predictor, validate_bundle
from sleepkit.runtime.bundle import REQUIRED_FILES


def write_checksums(path):
    checksums = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(path.iterdir())
        if p.is_file() and p.name != "checksums.json"
    }
    (path / "checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")


def export_bundle(
    model,
    destination,
    *,
    preprocessing,
    normalizer,
    class_names,
    calibration,
    validation_inputs,
    validation_targets,
    context,
    metrics,
    int8=True,
    max_accuracy_drop=0.02,
    provenance=None,
):
    """Convert and validate before atomically installing a new bundle directory.

    Calibration must come from training subjects, validation from a disjoint
    validation split. Test evaluation remains separate. Existing bundles are never
    overwritten. No patient-derived reference vectors are published.
    """
    import keras
    import tensorflow as tf
    from sleepkit.preprocessing.core import fingerprint

    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite bundle: {destination}")
    if not 0 <= max_accuracy_drop <= 1:
        raise ValueError("Accuracy drop threshold must be in [0, 1]")
    if normalizer.input_fingerprint != fingerprint(preprocessing.spec):
        raise ValueError("Normalization does not match preprocessing")
    expected = (context, len(preprocessing.spec["feature_names"]))
    for name, inputs in (("calibration", calibration), ("validation", validation_inputs)):
        if inputs.ndim != 3 or inputs.shape[1:] != expected or not len(inputs) or not np.isfinite(inputs).all():
            raise ValueError(f"Invalid {name} inputs")
    if tuple(model.input_shape[1:]) != expected or tuple(model.output_shape[1:]) != (context, len(class_names)):
        raise ValueError("Model shapes do not match bundle contract")
    if validation_targets.shape != validation_inputs.shape[:2]:
        raise ValueError("Validation targets do not match input frames")
    if (
        not np.issubdtype(validation_targets.dtype, np.integer)
        or (validation_targets < 0).any()
        or (validation_targets >= len(class_names)).any()
    ):
        raise ValueError("Validation labels are outside the class range")
    if not isinstance(preprocessing, WindowFeatures):
        raise ValueError("This bundle format currently supports WindowFeatures only")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".export-") as folder:
        stage = Path(folder)
        model.save(stage / "model.keras")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        if int8:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = lambda: ([frame[None].astype(np.float32)] for frame in calibration)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        (stage / "model.tflite").write_bytes(converter.convert())
        (stage / "preprocessing.json").write_text(
            json.dumps(
                {
                    "transform": preprocessing.to_dict(),
                    "normalizer": {
                        **normalizer.to_dict(),
                        "training_subjects": [],
                        "training_subjects_sha256": fingerprint(normalizer.training_subjects),
                    },
                },
                indent=2,
            )
        )
        manifest = {
            "schema_version": 1,
            "context": context,
            "class_names": list(class_names),
            "output": "logits",
            "scope": "raw_record_host",
            "embedded_preprocessing": False,
            "quantization": "int8" if int8 else "float32",
            "versions": {"tensorflow": tf.__version__, "keras": keras.__version__, "numpy": np.__version__},
            "provenance": provenance or {},
            "reference_source": "synthetic_normalized_features",
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2))
        (stage / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
        (stage / "README.md").write_text(
            "---\ntags:\n- sleepkit\n- time-series\n- tflite\n---\n\n"
            "# SleepKit experimental bundle\n\n"
            "Includes raw-record preprocessing for host inference and a LiteRT model. "
            "Embedded preprocessing is not provided. Metrics and their provenance are in metrics.json and manifest.json. "
            "Reference vectors are synthetic conformance inputs, not accuracy evidence. "
            "Confirm the model license and data provenance before public release.\n"
        )
        synthetic = np.random.default_rng(42).normal(0, 0.5, (3, *expected)).astype(np.float32)
        np.savez_compressed(
            stage / "reference_vectors.npz", inputs=synthetic, outputs=np.zeros((3, context, len(class_names)))
        )
        write_checksums(stage)
        predictor = Predictor(stage)
        float_predictions = np.asarray(model(validation_inputs, training=False))
        lite_predictions = predictor.predict_features(validation_inputs)
        if not np.isfinite(float_predictions).all() or not np.isfinite(lite_predictions).all():
            raise ValueError("Nonfinite model output")
        float_accuracy = float(np.mean(np.argmax(float_predictions, axis=-1) == validation_targets))
        lite_accuracy = float(np.mean(np.argmax(lite_predictions, axis=-1) == validation_targets))
        if float_accuracy - lite_accuracy > max_accuracy_drop:
            raise ValueError(f"Quantization accuracy drop {float_accuracy - lite_accuracy:.4f} exceeds threshold")
        manifest["conversion_validation"] = {
            "float_accuracy": float_accuracy,
            "lite_accuracy": lite_accuracy,
            "max_accuracy_drop": max_accuracy_drop,
            "max_absolute_logit_error": float(np.max(np.abs(float_predictions - lite_predictions))),
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2))
        np.savez_compressed(
            stage / "reference_vectors.npz", inputs=synthetic, outputs=predictor.predict_features(synthetic)
        )
        write_checksums(stage)
        Predictor(stage).check_reference()
        stage.rename(destination)
    return destination


def publish_bundle(path, repo_id, *, dry_run=True, private=True, license_file=None):
    """Stage an allowlisted bundle; upload only when explicitly requested."""
    validate_bundle(path)
    Predictor(path).check_reference()
    source = Path(path)
    names = json.loads((source / "checksums.json").read_text())
    if not set(names) <= REQUIRED_FILES | {"model.keras", "LICENSE"}:
        raise ValueError("Bundle contains files outside the publication allowlist")
    if not dry_run and license_file is None:
        raise ValueError("A model license file is required for publication")
    stage = Path(tempfile.mkdtemp(prefix="sleepkit-hf-"))
    try:
        for name in [*names, "checksums.json"]:
            shutil.copy2(source / name, stage / name)
        if license_file:
            shutil.copy2(license_file, stage / "LICENSE")
            write_checksums(stage)
        if dry_run:
            return stage
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
        return api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(stage))
    finally:
        if not dry_run:
            shutil.rmtree(stage)

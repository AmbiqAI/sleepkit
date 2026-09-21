"""Reproducible, offline checkpoint evaluation on an explicit saved-feature cohort."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re

import numpy as np

from sleepkit.recipes._components import ClassificationAccumulator, FileSnapshot, implementation_files

from .preprocessing import CLASS_ORDER, FEATURE_ORDER, prepare_subject, read_subject, window_subject


def read_cohort(path):
    """Read an explicit JSON list of unique filename stems, independent of discovery order."""
    subjects = json.loads(Path(path).read_text())
    if (
        not isinstance(subjects, list)
        or not subjects
        or any(not isinstance(s, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", s) for s in subjects)
        or len(set(subjects)) != len(subjects)
    ):
        raise ValueError("Cohort must be a nonempty JSON list of unique, safe subject stems")
    return sorted(subjects)


def validate_model(model):
    """Require the saved SS-3-TCN-SM float32 sequence-to-logits interface."""
    import keras

    terminal = model.layers[-1]
    if isinstance(terminal, keras.layers.Reshape):
        producer = model.layers[-2]
        if terminal.input is not producer.output:
            raise ValueError("Output reshape must consume the final logit layer directly")
        terminal = producer
    if (
        len(model.inputs) != 1
        or len(model.outputs) != 1
        or tuple(model.input_shape) != (None, 240, 14)
        or tuple(model.output_shape) != (None, 240, 3)
        or str(model.inputs[0].dtype) != "float32"
        or str(model.outputs[0].dtype) != "float32"
        or not hasattr(terminal, "activation")
        or keras.activations.serialize(terminal.activation) != "linear"
    ):
        raise ValueError("Expected float32 [batch,240,14] -> [batch,240,3] with final linear activation")


def evaluate_windows(model, windows, accumulator, batch_size=16):
    """Apply the scoring mask after inference so excluded epochs retain context."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    import keras

    for start in range(0, len(windows.features), batch_size):
        features = windows.features[start : start + batch_size]
        inputs = keras.tree.map_structure(lambda _: features, model.input)
        scores = np.asarray(keras.ops.convert_to_numpy(model(inputs, training=False)))
        if scores.shape != (*features.shape[:2], len(CLASS_ORDER)) or scores.dtype != np.float32:
            raise ValueError("Model must return float32 per-epoch logits with three classes")
        if not np.isfinite(scores).all():
            raise ValueError("Model returned nonfinite logits")
        mask = windows.scoring_mask[start : start + batch_size]
        accumulator.update(windows.targets[start : start + batch_size][mask], scores[mask])


def evaluate(features_dir, checkpoint, cohort, output, *, batch_size=16):
    """Write an aggregate report only after evaluation and input revalidation.

    Subject IDs and source paths stay in the private cohort/input inventory. The
    report binds that inventory by digest; this is replay evidence, not proof of
    historical training membership or raw-feature provenance.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    declarations = FileSnapshot.capture({"checkpoint": checkpoint, "cohort": cohort})
    subjects = read_cohort(cohort)
    records = {subject: Path(features_dir) / f"{subject}.h5" for subject in subjects}
    data = FileSnapshot.capture(records)
    code = FileSnapshot.capture(implementation_files(Path(__file__).parent))
    import keras

    model = keras.models.load_model(checkpoint, compile=False, safe_mode=True)
    validate_model(model)
    accumulator = ClassificationAccumulator(len(CLASS_ORDER))
    coverage = {}
    for subject in subjects:
        windows = window_subject(prepare_subject(read_subject(records[subject])))
        evaluate_windows(model, windows, accumulator, batch_size)
        for key, value in windows.coverage.items():
            coverage[key] = coverage.get(key, 0) + value
    metrics = accumulator.result()
    for snapshot in (declarations, data, code):
        snapshot.verify()
    inventory = json.dumps(data.hashes(), sort_keys=True, separators=(",", ":")).encode()
    report = {
        "schema": "sleepkit.staging-evaluation/v1",
        "class_order": list(CLASS_ORDER),
        "feature_order": list(FEATURE_ORDER),
        "metrics": metrics,
        "coverage": {"subjects": len(subjects), **coverage},
        "protocol": {
            "normalization": "whole-subject quality-valid float32 median/mean/population-variance; epsilon=1e-6",
            "windows": "nonoverlapping 240 epochs; incomplete tail omitted; 30 seconds/epoch",
            "scoring": "quality-valid known stages; stage 6 excluded without removing context",
            "batch_size": batch_size,
            "historical_training_membership": "unverified",
            "raw_feature_provenance": "unverified; caller asserts FS-W-PA-14-60 semantics",
        },
        "provenance": {
            **declarations.hashes(),
            "subject_set_sha256": hashlib.sha256("\n".join(subjects).encode()).hexdigest(),
            "source_inventory_sha256": hashlib.sha256(inventory).hexdigest(),
            "implementation_sha256": code.hashes(),
            "environment": {
                "backend": keras.backend.backend(),
                **{name: importlib.metadata.version(name) for name in ("keras", "numpy", "h5py")},
                "backend_version": importlib.metadata.version(
                    "tensorflow" if keras.backend.backend() == "tensorflow" else keras.backend.backend()
                ),
            },
        },
        "limitations": [
            "Saved-feature replay; no raw feature regeneration, training, or new held-out-performance claim.",
            "Cross entropy and macro F1 differ from historical focal loss and weighted F1.",
            "Whole-subject normalization uses later epochs and the omitted tail; not causal inference.",
            "File snapshots detect changes between checks, not concurrent-write prevention.",
            "No export parity, hardware result, or publication qualification.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cohort", required=True, help="Private JSON list of subject filename stems")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    evaluate(args.features_dir, args.checkpoint, args.cohort, args.output, batch_size=args.batch_size)


if __name__ == "__main__":
    main()

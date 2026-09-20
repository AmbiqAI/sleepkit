"""Small, reproducible calibration sets for the detection runtime.

Calibration is deliberately a source operation rather than a training
operation: the fitted normalizer is supplied by the caller and only the
frozen training partition is read.  The files written here contain model
inputs and provenance, while target values remain outside the calibration
array.
"""

from pathlib import Path
import hashlib
import json
import tempfile

import numpy as np

from .preprocessing import SPEC, fingerprint, prepare


def _bytes_sha256(value):
    return hashlib.sha256(value).hexdigest()


def _stable_seed(seed, subject):
    digest = hashlib.sha256(f"sleepkit-calibration:{seed}:{subject}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _source_digest(source, subject):
    hashes = getattr(source, "source_hashes", None)
    if not isinstance(hashes, dict) or subject not in hashes:
        raise ValueError("Source must expose source_hashes for calibration provenance")
    value = hashes[subject]
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("source_hashes must contain hexadecimal SHA-256 strings")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError("source_hashes must contain hexadecimal SHA-256 strings") from error
    return value


def _check_output_location(source, output_path):
    root = getattr(source, "root", None)
    if root is None:
        return
    root = Path(root).resolve()
    output_path = output_path.resolve()
    try:
        output_path.relative_to(root)
    except ValueError:
        return
    raise ValueError("Calibration output must be outside the source raw-data directory")


def _select(features, labels, context, *, seed, subject, limit):
    values = np.asarray(features.values)
    valid = np.asarray(features.valid)
    ends = np.asarray(features.ends)
    labels = np.asarray(labels)
    if values.ndim != 2 or values.shape[1] != 5 or valid.shape != (len(values),) or valid.dtype != bool:
        raise ValueError("Prepared features have an invalid shape")
    if ends.shape != (len(values),) or not np.issubdtype(ends.dtype, np.integer):
        raise ValueError("Prepared feature ends have an invalid shape")
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Labels must be an integer sample vector")
    if len(ends) and (ends[0] < 0 or ends[-1] >= len(labels)):
        raise ValueError("Labels do not cover all feature ends")
    if not np.isin(labels, [-1, 0, 1]).all():
        raise ValueError("Labels must contain only -1 (unknown), 0 (WAKE), or 1 (SLEEP)")
    target = labels[ends]
    generator = np.random.default_rng(_stable_seed(seed, subject))
    selected = []
    eligible_count = 0
    for start in range(0, len(values) - context + 1, context):
        selection = slice(start, start + context)
        if not valid[selection].all() or (target[selection] < 0).any():
            continue
        eligible_count += 1
        if len(selected) < limit:
            selected.append(start)
        else:
            replacement = int(generator.integers(0, eligible_count))
            if replacement < limit:
                selected[replacement] = start
    selected.sort()
    return target, eligible_count, selected


def _validate_knob(name, value, *, maximum=None):
    if type(value) is not int or value < 1 or (maximum is not None and value > maximum):
        bound = f" and at most {maximum}" if maximum is not None else ""
        raise ValueError(f"{name} must be a positive integer{bound}")


def collect_calibration(source, normalizer, output_path, *, cache=None, seed=0, contexts_per_subject=2):
    """Collect frozen-normalizer model inputs from native train contexts.

    The returned metadata is aggregate-only.  Subject IDs and per-context
    provenance are written to the local report files beside ``calibration.npz``.
    """
    _validate_knob("contexts_per_subject", contexts_per_subject)
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if getattr(normalizer, "spec", None) != SPEC:
        raise ValueError("Calibration requires the fitted v3 preprocessing contract")
    if not hasattr(normalizer, "to_dict") or not callable(normalizer.to_dict):
        raise ValueError("Calibration requires a fitted normalizer")
    output_path = Path(output_path)
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Calibration output already exists: {output_path}")
    _check_output_location(source, output_path)

    verify = getattr(source, "verify_unchanged", None)
    if not callable(verify):
        raise ValueError("Source must provide verify_unchanged()")
    fitted_state_fingerprint = fingerprint(normalizer.to_dict())
    verify()
    split = getattr(source, "split", None)
    if not isinstance(split, dict) or "train" not in split:
        raise ValueError("Source must provide a train split")
    train = split["train"]
    if not isinstance(train, (list, tuple)) or not train or any(not isinstance(s, str) for s in train):
        raise ValueError("Source train split must be a nonempty subject sequence")
    if len(set(train)) != len(train):
        raise ValueError("Source train split contains duplicate subjects")
    for partition in ("validation", "test"):
        values = split.get(partition, ())
        if not isinstance(values, (list, tuple)) or len(set(values)) != len(values):
            raise ValueError(f"Source {partition} split is invalid")
        if set(train) & set(values):
            raise ValueError("Train and held-out source splits must be disjoint")
    subjects = sorted(train)
    context = getattr(source, "context", None)
    _validate_knob("source context", context)

    source_inventory = []
    subject_reports = []
    selected_rows = []
    selected_values = []
    for subject in subjects:
        recording = source.read(subject, labels=True)
        if getattr(recording, "labels", None) is None:
            raise ValueError("Calibration source.read(..., labels=True) returned no labels")
        data = np.asarray(recording.data)
        labels = np.asarray(recording.labels)
        if data.ndim != 2 or data.shape[1] < 1 or labels.shape != (data.shape[1],):
            raise ValueError("Calibration labels must align with every raw sensor sample")
        features = prepare(
            data,
            cache,
            sample_time=getattr(recording, "sample_time", None),
            spec=SPEC,
        )
        normalized = normalizer.transform(features)
        target, eligible_count, starts = _select(
            normalized,
            labels,
            context,
            seed=seed,
            subject=subject,
            limit=contexts_per_subject,
        )
        digest = _source_digest(source, subject)
        source_inventory.append({"source_id": subject, "source_sha256": digest})
        subject_reports.append(
            {
                "source_id": subject,
                "source_sha256": digest,
                "eligible_context_count": eligible_count,
                "selected_context_count": len(starts),
                "feature_frame_count": int(len(normalized.values)),
            }
        )
        for start in starts:
            stop = start + context
            values = np.asarray(normalized.values[start:stop], dtype=np.float32)
            if values.shape != (context, 5) or not np.isfinite(values).all():
                raise ValueError("Selected calibration context is not finite [context, 5]")
            first_end = int(normalized.ends[start])
            last_end = int(normalized.ends[stop - 1])
            target_values = np.asarray(target[start:stop], dtype="<i4")
            selected_values.append(values.copy())
            selected_rows.append(
                {
                    "source_id": subject,
                    "source_sha256": digest,
                    "context_start_frame": int(start),
                    "context_end_frame_exclusive": int(stop),
                    "raw_start_sample": first_end - 11,
                    "raw_end_sample_exclusive": last_end + 1,
                    "first_feature_end_sample": first_end,
                    "last_feature_end_sample": last_end,
                    "valid_targets_sha256": _bytes_sha256(target_values.tobytes()),
                }
            )

    # Detect a source mutation even when a custom reader does not check hashes
    # during its read method.
    verify()
    if fingerprint(normalizer.to_dict()) != fitted_state_fingerprint:
        raise ValueError("Fitted normalizer changed during calibration collection")
    if not selected_values:
        raise ValueError("No eligible train contexts available for calibration")
    array = np.stack(selected_values, axis=0).astype(np.float32, copy=False)
    if array.ndim != 3 or array.shape[1:] != (context, 5) or not np.isfinite(array).all():
        raise ValueError("Calibration array must be finite float32 [N, context, 5]")

    report = {
        "schema": "sleepkit.detection.calibration/v1",
        "seed": seed,
        "context": context,
        "contexts_per_subject": contexts_per_subject,
        "inventory": source_inventory,
        "subjects": subject_reports,
    }
    normalizer_fingerprint = fitted_state_fingerprint
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_path.parent, prefix=f".{output_path.name}.") as directory:
        stage = Path(directory)
        np.savez_compressed(stage / "calibration.npz", contexts=array)
        index_path = stage / "calibration-index.jsonl"
        with index_path.open("w", encoding="utf-8") as stream:
            for row in selected_rows:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        selection_path = stage / "selection.json"
        selection_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        metadata = {
            "policy": "train-only frozen normalizer, native nonoverlapping contexts, uniform per-subject sampling",
            "selection_algorithm": "per-subject SHA-256 seed with reservoir sampling, then ascending native starts",
            "train_subject_count": len(subjects),
            "contributing_subject_count": sum(bool(row["selected_context_count"]) for row in subject_reports),
            "zero_eligible_subject_count": sum(row["eligible_context_count"] == 0 for row in subject_reports),
            "context_count": int(array.shape[0]),
            "frame_count": int(array.shape[0] * array.shape[1]),
            "seed": seed,
            "context": context,
            "contexts_per_subject": contexts_per_subject,
            "fitted_state_fingerprint": normalizer_fingerprint,
            "selected_index_sha256": _bytes_sha256(index_path.read_bytes()),
            "array_sha256": _bytes_sha256((stage / "calibration.npz").read_bytes()),
            "local_report_sha256": _bytes_sha256(selection_path.read_bytes()),
        }
        stage.rename(output_path)
    verify()
    return array, metadata

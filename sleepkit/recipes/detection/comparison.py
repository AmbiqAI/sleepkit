"""Descriptive SD-2 comparison on exact native context/target intersections.

This composes the historical adapter with a verified membership run. It does not
retrain, tune thresholds, recover historical training exposure, or compare losses.
"""

from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np

from sleepkit.recipes._components import summarize_confusion

from sleepkit.artifacts.baselines import HASHES
from sleepkit.artifacts.package import sha256, write_json
from .evaluation import summarize
from .historical import SD2Runtime, read_verified_features
from .preprocessing import fingerprint


POLICY = {
    "kind": "sleepkit.sd2_membership_comparison/v1",
    "selection": "All frozen test series; anchored complete 240-frame contexts; no stitching",
    "intersection": "Equal subject, raw source hash, context start/end, every feature-end sample and candidate target; both contexts eligible",
    "historical_features": "Exact stored float32 FS_W_A_5 features verified against raw source and legacy tail arithmetic",
    "historical_normalization": "Whole-record nanmean/nanvar, including unknown periods and incomplete context tails; offline transductive",
    "historical_classes": ["WAKE", "SLEEP"],
    "comparison_mapping": "WAKE -> outside supported annotated period; SLEEP -> inside annotated period; descriptive proxy only",
    "decision": "Argmax, ties choose index 0; no additional softmax, threshold, loss or calibration comparison",
    "native_denominator": "Each pipeline's own complete contexts eligible under the candidate target and finite-feature policy; historical native metrics do not reproduce original published label/split metrics",
    "scope": "Historical training exposure unresolved. Complete pipelines differ in preprocessing, normalization, training and quantization. No held-out historical, clinical or architecture superiority claim.",
}


def class_metrics(confusion):
    """Fixed two-class descriptive metrics; no probability interpretation."""
    confusion = np.asarray(confusion, dtype=np.int64)
    count = int(confusion.sum())
    if not count:
        return None
    result = summarize_confusion(confusion)
    return {"frames": result.pop("count"), **result}


def match_context(rows, *, subject, source_hash, ends, targets):
    """Reject coordinate or target drift even when shapes and timestamps look plausible."""
    ends, targets = np.asarray(ends), np.asarray(targets)
    if len(rows) != len(ends) or len(ends) != len(targets) or not len(rows):
        raise ValueError("Context lengths differ")
    start, end = int(ends[0]) - 11, int(ends[-1])
    for row, feature_end, target in zip(rows, ends, targets):
        if (row["subject"] != subject or row["source_sha256"] != source_hash
                or row["context_start_sample"] != start or row["context_end_sample"] != end
                or row["feature_end_sample"] != int(feature_end) or row["target"] != int(target)):
            raise ValueError("Native context coordinates or targets differ")


def _write_line(stream, value):
    stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def compare(run_path, source, feature_root, baseline_source, output_path):
    """Run one declared descriptive comparison; output is exclusively local evidence.

    ``source`` is an AnnotatedDataset. Prediction arrays stay in memory; scoring
    indices stream context-by-context. Existing output directories are refused.
    Partial failed runs remain with their declaration and no completed report.
    """
    run, feature_root, baseline_source, output = map(Path, (run_path, feature_root, baseline_source, output_path))
    if output.resolve().is_relative_to(run.resolve()):
        raise ValueError("Comparison output must stay outside the source run")
    if source.context != 240:
        raise ValueError("SD-2 requires the frozen 240-frame context")
    source.verify_unchanged()
    if any(sha256(baseline_source / name) != digest for name, digest in HASHES.items()):
        raise ValueError("Historical source files differ from the pinned release")
    recipe = json.loads((run / "bundle/recipe.json").read_text())
    if (source.provenance != recipe.get("dataset") or fingerprint(source.split) != recipe.get("split_sha256")
            or fingerprint(source.source_hashes) != recipe.get("source_sha256")):
        raise ValueError("Source protocol differs from the evaluated run")
    output.mkdir(parents=True, exist_ok=False)
    verified = summarize(run, output / "new-evaluation.json")
    subjects = source.split["test"]
    feature_hashes = {s: sha256(feature_root / f"{s}.h5") for s in subjects}
    from . import historical

    declaration = {
        "declared_utc": datetime.now(timezone.utc).isoformat(), "policy": POLICY,
        "new_evidence_sha256": verified["evidence_sha256"], "historical_release_sha256": HASHES,
        "historical_feature_sha256": feature_hashes, "source": source.provenance,
        "implementation_sha256": {"comparison.py": sha256(Path(__file__)),
                                  "historical.py": sha256(Path(historical.__file__))},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "ai-edge-litert": version("ai-edge-litert"), "h5py": version("h5py")},
    }
    write_json(output / "declaration.json", declaration)
    runner = SD2Runtime(baseline_source / "model.tflite")
    with np.load(run / "test-predictions.npz", allow_pickle=False) as arrays:
        new_logits, new_targets = arrays["logits"], arrays["targets"]
    native_confusion, common_old, common_new = (np.zeros((2, 2), np.int64) for _ in range(3))
    saved_outputs, saved_targets, common_old_offsets, common_new_offsets = [], [], [], []
    details = {}
    old_offset, ties, clipped, input_values = 0, 0, 0, 0
    with ((run / "test-index.jsonl").open() as index,
          (output / "historical-index.jsonl").open("x") as historical_index,
          (output / "common-index.jsonl").open("x") as common_index):
        for subject in subjects:
            recording = source.read(subject, labels=True)
            features, metadata = read_verified_features(feature_root / f"{subject}.h5", recording.data,
                                                        sample_time=recording.sample_time)
            if sha256(feature_root / f"{subject}.h5") != feature_hashes[subject]:
                raise ValueError("Historical features changed after declaration")
            native_count = len(features.ends) // 240
            new_count = verified["subjects"][subject]["complete_contexts"]
            if native_count > new_count:
                raise ValueError("Legacy tail unexpectedly extends beyond new native contexts")
            counts = {"native_contexts": native_count, "native_eligible_contexts": 0, "common_contexts": 0,
                      "excluded_unknown_target": 0, "excluded_nonfinite_features": 0,
                      "new_only_tail_contexts": new_count - native_count,
                      "historical_tail_feature_frames": len(features.ends) % 240,
                      "input_values": 0, "clipped_input_values": 0, "argmax_ties": 0}
            subject_native, subject_old, subject_new = (np.zeros((2, 2), np.int64) for _ in range(3))
            for context in range(new_count):
                rows = [json.loads(index.readline()) for _ in range(240)]
                if rows[0]["subject"] != subject:
                    raise ValueError("New index subject order differs")
                if context >= native_count:
                    continue
                begin, stop = context * 240, (context + 1) * 240
                ends = features.ends[begin:stop]
                target = recording.labels[ends]
                match_context(rows, subject=subject, source_hash=source.source_hashes[subject], ends=ends, targets=target)
                invalid, unknown = not features.valid[begin:stop].all(), bool((target < 0).any())
                eligible = not invalid and not unknown
                counts["excluded_unknown_target"] += int(unknown)
                counts["excluded_nonfinite_features"] += int(invalid)
                coordinate = {"subject": subject, "source_sha256": source.source_hashes[subject],
                              "context_start_sample": int(ends[0]) - 11, "context_end_sample": int(ends[-1]),
                              "first_feature_end_sample": int(ends[0]), "feature_stride_samples": 6, "frames": 240}
                _write_line(historical_index, {**coordinate, "targets": target.tolist(),
                            "feature_file_sha256": feature_hashes[subject], "eligible": eligible,
                            "exclusion_reasons": (["nonfinite_features"] if invalid else [])
                            + (["unknown_target"] if unknown else []),
                            "prediction_start": old_offset if eligible else None})
                if not eligible:
                    continue
                raw, saturation = runner.predict(features.values[begin:stop])
                prediction = raw.argmax(axis=1)
                tie_count = int((raw[:, 0] == raw[:, 1]).sum())
                counts["native_eligible_contexts"] += 1
                counts["input_values"] += 240 * 5
                counts["clipped_input_values"] += saturation
                counts["argmax_ties"] += tie_count
                np.add.at(subject_native, (target, prediction), 1)
                saved_outputs.append(raw.copy())
                saved_targets.append(target.copy())
                if rows[0]["context_eligible"]:
                    position = rows[0]["eligible_output_index"]
                    if ([r["eligible_output_index"] for r in rows] != list(range(position, position + 240))
                            or not np.array_equal(new_targets[position:position + 240], target)):
                        raise ValueError("Common outputs differ from saved new targets/ordinals")
                    np.add.at(subject_old, (target, prediction), 1)
                    np.add.at(subject_new, (target, new_logits[position:position + 240].argmax(axis=1)), 1)
                    counts["common_contexts"] += 1
                    common_old_offsets.append(old_offset)
                    common_new_offsets.append(position)
                    _write_line(common_index, {**coordinate, "historical_prediction_start": old_offset,
                                "new_prediction_start": position, "targets_sha256": fingerprint(target.tolist())})
                old_offset += 240
            native_confusion += subject_native
            common_old += subject_old
            common_new += subject_new
            ties += counts["argmax_ties"]
            clipped += counts["clipped_input_values"]
            input_values += counts["input_values"]
            details[subject] = {"coverage": counts, "historical_preprocessing": metadata,
                                "historical_native": class_metrics(subject_native),
                                "historical_common": class_metrics(subject_old), "new_common": class_metrics(subject_new)}
        if index.readline():
            raise ValueError("Unexpected trailing new scoring rows")
    if not common_old_offsets:
        raise ValueError("No common eligible contexts")
    np.savez_compressed(output / "historical-predictions.npz", outputs=np.concatenate(saved_outputs),
                        targets=np.concatenate(saved_targets),
                        common_historical_starts=np.asarray(common_old_offsets, dtype=np.int64),
                        common_new_starts=np.asarray(common_new_offsets, dtype=np.int64))
    source.verify_unchanged()
    runner.verify_unchanged()
    if (any(sha256(run / name) != digest for name, digest in verified["evidence_sha256"].items())
            or any(sha256(feature_root / f"{s}.h5") != digest for s, digest in feature_hashes.items())
            or any(sha256(baseline_source / name) != digest for name, digest in HASHES.items())
            or sha256(Path(__file__)) != declaration["implementation_sha256"]["comparison.py"]
            or sha256(Path(historical.__file__)) != declaration["implementation_sha256"]["historical.py"]):
        raise ValueError("Comparison evidence changed during execution")
    means = {}
    for kind in ("historical_native", "historical_common", "new_common"):
        values = [d[kind] for d in details.values() if d[kind] is not None]
        means[kind] = {"contributing_series": len(values), **{
            key: float(np.mean([value[key] for value in values])) if values else None
            for key in ("accuracy", "macro_f1")}}
    report = {
        "schema": POLICY["kind"], "policy": POLICY,
        "target": verified["target"], "class_order": verified["class_order"],
        "coverage": {key: sum(d["coverage"][key] for d in details.values()) for key in counts},
        "series": len(details), "new_native": verified["pooled"], "new_native_coverage": verified["coverage"],
        "historical_native": class_metrics(native_confusion), "historical_common": class_metrics(common_old),
        "new_common": class_metrics(common_new), "unweighted_eligible_series_mean": means,
        "runtime": {"input_values": input_values, "clipped_input_values": clipped, "argmax_ties": ties},
        "evidence_sha256": {p.name: sha256(p) for p in sorted(output.iterdir())}, "subjects": details,
    }
    write_json(output / "comparison.json", report)
    return report

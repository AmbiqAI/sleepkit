"""Read-only held-out summaries from hash-bound local scoring records.

This checks saved evidence and coordinates; it does not reread recordings, replay
models, establish clinical sleep/wake accuracy, or validate participant identity.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from sleepkit.recipes._components import summarize_confusion

from sleepkit.artifacts.package import sha256, validate_bundle
from .output_contract import validate_output
from .preprocessing import SPEC, Normalizer, fingerprint
from .split import PARTITIONS, TARGET


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate scoring JSON key")
        result[key] = value
    return result


def read_bound_json(run, name, hashes):
    """Parse the same bytes whose digest was verified, including transient changes."""
    content = (Path(run) / name).read_bytes()
    if hashlib.sha256(content).hexdigest() != hashes.get(name):
        raise ValueError(f"Run evidence changed before metadata read: {name}")
    value = json.loads(content, object_pairs_hook=_unique)
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in run metadata: {name}")
    return value


def _metrics(confusion, loss):
    count = int(confusion.sum())
    if not count:
        return None
    result = summarize_confusion(confusion)
    return {"feature_frames_evaluated": result.pop("count"), **result, "cross_entropy": float(loss / count)}


def _new_subject():
    return {"complete_contexts": 0, "eligible_contexts": 0, "outputs": 0, "eligible_outputs": 0,
            "excluded_contexts": {"nonfinite_sensor": 0, "unknown_target": 0},
            "confusion": np.zeros((2, 2), dtype=np.int64), "loss": 0.0}


def summarize(run_path, output_path):
    """Verify local run evidence and exclusively write a per-series JSON report.

    The JSONL index is read one context at a time. Saved prediction arrays are
    loaded into memory; no pandas, TensorFlow, source data, or model is loaded.
    """
    run, output = Path(run_path), Path(output_path)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    bundle = run / "bundle"
    if output.resolve().is_relative_to(bundle.resolve()):
        raise ValueError("Evaluation output must stay outside the artifact bundle")
    validated = validate_bundle(bundle)
    if not any(entry["path"] == "model.keras" and entry["format"] == "keras"
               and entry["role"] in {"model", "training_checkpoint"}
               for entry in validated["manifest"]["artifacts"]):
        raise ValueError("Evaluated bundle must contain the declared model.keras checkpoint")
    names = ["split.json", "sources.json", "test-index.jsonl", "test-index-summary.json", "test-predictions.npz"]
    paths = [run / name for name in names] + sorted(bundle.iterdir())
    if any(p.is_symlink() or not p.is_file() for p in paths):
        raise ValueError("Run evidence must be regular files")
    bound = {str(p.relative_to(run)): sha256(p) for p in paths}
    recipe, recorded = (read_bound_json(run, f"bundle/{name}", bound) for name in ("recipe.json", "metrics.json"))
    classes = validate_output(recipe)
    if recipe.get("target") != TARGET or recorded.get("target") != TARGET:
        raise ValueError("An annotated-period target run is required")
    normalizer = Normalizer.from_dict(read_bound_json(run, "bundle/preprocessing.json", bound))
    if normalizer.spec != SPEC:
        raise ValueError("Scoring preprocessing differs from the bundle")
    split, sources, summary = (read_bound_json(run, name, bound) for name in names[:2] + [names[3]])
    if not isinstance(split, dict) or set(split) != set(PARTITIONS):
        raise ValueError("Invalid split")
    if any(not isinstance(group, list) or any(not isinstance(s, str) or not s for s in group)
           for group in split.values()):
        raise ValueError("Invalid split subjects")
    all_subjects = [s for group in split.values() for s in group]
    if len(set(all_subjects)) != len(all_subjects) or set(sources) != set(all_subjects):
        raise ValueError("Split subjects must be disjoint and match source hashes")
    if any(not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
           for digest in sources.values()):
        raise ValueError("Invalid source hashes")
    if recipe.get("split_sha256") != fingerprint(split) or recipe.get("source_sha256") != fingerprint(sources):
        raise ValueError("Split/source fingerprints differ from the recipe")
    if recipe.get("subjects_per_split") != {name: len(group) for name, group in split.items()}:
        raise ValueError("Subject counts differ from the recipe")
    context = recipe.get("context")
    if type(context) is not int or context < 1 or recipe.get("config", {}).get("context") != context:
        raise ValueError("Invalid recipe context")
    expected = {
        "schema": "sleepkit.detection_scoring_index/v1", "format": "jsonl", "context": context,
        "subjects": len(split["test"]), "source_sha256": fingerprint({s: sources[s] for s in split["test"]}),
        "provenance_sha256": fingerprint(recipe.get("dataset")), "target_sha256": fingerprint(TARGET),
        "preprocessing_sha256": fingerprint(SPEC), "sha256": bound["test-index.jsonl"],
        "predictions_sha256": bound["test-predictions.npz"],
    }
    if recipe.get("scoring") != summary or any(summary.get(k) != v for k, v in expected.items()):
        raise ValueError("Scoring metadata or evidence hash mismatch")
    if any(type(summary.get(key)) is not int or summary[key] < 0 for key in
           ("complete_contexts", "eligible_contexts", "outputs", "eligible_outputs",
            "tail_feature_frames", "samples_after_last_feature")):
        raise ValueError("Invalid scoring coverage counts")
    with np.load(run / "test-predictions.npz", allow_pickle=False) as arrays:
        if set(arrays.files) != {"logits", "targets"}:
            raise ValueError("Unexpected prediction arrays")
        logits, targets = arrays["logits"], arrays["targets"]
    if (targets.ndim != 1 or logits.shape != (len(targets), 2) or logits.dtype != np.float64
            or targets.dtype.kind not in "iu" or not np.isin(targets, [0, 1]).all()
            or not np.isfinite(logits).all()):
        raise ValueError("Invalid saved logits or targets")
    details = {s: _new_subject() for s in split["test"]}
    subject_order = {s: i for i, s in enumerate(split["test"])}
    ordinal, eligible_ordinal, previous_subject = 0, 0, -1
    digest = hashlib.sha256()
    with (run / "test-index.jsonl").open("rb") as stream:
        while first := stream.readline():
            lines = [first] + [stream.readline() for _ in range(context - 1)]
            if any(not line for line in lines):
                raise ValueError("Incomplete scoring context")
            for line in lines:
                digest.update(line)
            rows = [json.loads(line, object_pairs_hook=_unique) for line in lines]
            if any(not isinstance(row, dict) for row in rows):
                raise ValueError("Scoring rows must be JSON objects")
            subject = rows[0].get("subject")
            if subject not in details or subject_order[subject] < previous_subject:
                raise ValueError("Scoring subject order differs from test split")
            previous_subject = subject_order[subject]
            detail = details[subject]
            start = detail["complete_contexts"] * context * SPEC["stride_samples"]
            end = start + SPEC["window_samples"] - 1 + (context - 1) * SPEC["stride_samples"]
            reasons = []
            if any(row.get("sensor_valid") is not True for row in rows):
                reasons.append("nonfinite_sensor")
            if any(row.get("target") == -1 for row in rows):
                reasons.append("unknown_target")
            eligible = not reasons
            for frame, row in enumerate(rows):
                target = row.get("target")
                if (type(target) is not int or target not in (-1, 0, 1)
                        or any(type(row.get(key)) is not bool for key in
                               ("sensor_valid", "target_known", "context_eligible"))
                        or (eligible and type(row.get("eligible_output_index")) is not int)):
                    raise ValueError("Invalid scoring target or sensor validity")
                wanted = {
                    "subject": subject, "source_sha256": sources[subject], "output_index": ordinal,
                    "eligible_output_index": eligible_ordinal if eligible else None,
                    "context_start_sample": start, "context_end_sample": end,
                    "feature_end_sample": start + SPEC["window_samples"] - 1 + frame * SPEC["stride_samples"],
                    "target": target, "sensor_valid": row["sensor_valid"], "target_known": target >= 0,
                    "context_eligible": eligible, "exclusion_reasons": reasons,
                }
                if row != wanted or any(type(row.get(k)) is not int for k in
                                        ("output_index", "context_start_sample", "context_end_sample", "feature_end_sample")):
                    raise ValueError("Scoring coordinates or context eligibility mismatch")
                if eligible:
                    if eligible_ordinal >= len(targets) or targets[eligible_ordinal] != target:
                        raise ValueError("Scoring targets differ from saved predictions")
                    eligible_ordinal += 1
                ordinal += 1
            if eligible:
                values = logits[eligible_ordinal - context:eligible_ordinal]
                labels = targets[eligible_ordinal - context:eligible_ordinal]
                np.add.at(detail["confusion"], (labels, values.argmax(axis=1)), 1)
                shifted = values - values.max(axis=1, keepdims=True)
                detail["loss"] += float((np.log(np.exp(shifted).sum(axis=1))
                                         - shifted[np.arange(context), labels]).sum())
            detail["complete_contexts"] += 1
            detail["eligible_contexts"] += int(eligible)
            detail["outputs"] += context
            detail["eligible_outputs"] += context * int(eligible)
            for reason in reasons:
                detail["excluded_contexts"][reason] += 1
    if eligible_ordinal != len(targets) or digest.hexdigest() != bound["test-index.jsonl"]:
        raise ValueError("Scoring evidence changed or prediction count differs")
    coverage = {key: sum(d[key] for d in details.values()) for key in
                ("complete_contexts", "eligible_contexts", "outputs", "eligible_outputs")}
    coverage["excluded_contexts"] = {
        key: sum(d["excluded_contexts"][key] for d in details.values())
        for key in ("nonfinite_sensor", "unknown_target")
    }
    if any(summary.get(k) != v for k, v in coverage.items()):
        raise ValueError("Index coverage differs from saved summary")
    if recipe.get("contexts_per_split", {}).get("test") != coverage["eligible_contexts"]:
        raise ValueError("Test context count differs from recipe")
    pooled = _metrics(sum((d["confusion"] for d in details.values()), np.zeros((2, 2), dtype=np.int64)),
                      sum(d["loss"] for d in details.values()))
    if pooled is None or recorded.get("model") != "model.keras" or recorded.get("split") != "test" or recorded.get("class_order") != classes:
        raise ValueError("Invalid recorded evaluation metadata")
    for key in ("feature_frames_evaluated", "confusion_matrix", "f1_zero_division"):
        if recorded.get(key) != pooled[key]:
            raise ValueError("Recomputed metrics differ from recorded evaluation")
    for key in ("accuracy", "macro_f1", "cross_entropy"):
        if not np.isclose(recorded.get(key, np.nan), pooled[key], rtol=1e-10, atol=1e-12):
            raise ValueError("Recomputed metrics differ from recorded evaluation")
    for detail in details.values():
        detail["metrics"] = _metrics(detail.pop("confusion"), detail.pop("loss"))
    eligible_metrics = [d["metrics"] for d in details.values() if d["metrics"] is not None]
    report = {
        "schema": "sleepkit.detection_evaluation/v1", "target": TARGET, "class_order": classes,
        "model": "model.keras", "split": "test", "evidence_sha256": bound,
        "coverage": {**coverage, "subjects": len(details), "subjects_with_eligible_outputs": len(eligible_metrics),
                     "subjects_without_eligible_outputs": len(details) - len(eligible_metrics),
                     "tail_feature_frames": summary["tail_feature_frames"],
                     "samples_after_last_feature": summary["samples_after_last_feature"]},
        "pooled": pooled,
        "unweighted_eligible_subject_mean": {
            key: float(np.mean([m[key] for m in eligible_metrics])) for key in ("accuracy", "macro_f1")
        },
        "subjects": details,
        "scope": "Saved Keras logits on eligible native contexts; series-level descriptive metrics, not clinical sleep/wake, official event AP, participant-independent validation, or an acceptance claim. Tail counts are recorded evidence, not reconstructed from the index. No source or model replay.",
    }
    if any(sha256(run / name) != digest for name, digest in bound.items()):
        raise ValueError("Run evidence changed during evaluation")
    validate_bundle(bundle)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = summarize(args.run, args.output)
    print(json.dumps({key: report[key] for key in ("schema", "coverage", "pooled", "unweighted_eligible_subject_mean")},
                     sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

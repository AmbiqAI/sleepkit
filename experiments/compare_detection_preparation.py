#!/usr/bin/env python3
"""Compare uncached extraction against the immutable scalar baseline on training data."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sleepkit.artifacts.package import sha256, write_json  # noqa: E402
from sleepkit.recipes.detection import preprocessing  # noqa: E402
from sleepkit.recipes.detection.target_dataset import AnnotatedDataset  # noqa: E402

BASELINE_COMMIT = "042c1eef5608db3bb842f29c6f12b768663f9026"
MODULE = "sleepkit/recipes/detection/preprocessing.py"
BASELINE_SHA256 = "f9b32c300f31818269b157f06904297520d995d92a744b16167d32268146eb27"


def legacy_module():
    code = subprocess.check_output(["git", "show", f"{BASELINE_COMMIT}:{MODULE}"], cwd=ROOT)
    if hashlib.sha256(code).hexdigest() != BASELINE_SHA256:
        raise ValueError("Historical extractor source hash mismatch")
    # This is the pinned, trusted repository implementation, not downloaded model code.
    module = types.ModuleType("_sleepkit_scalar_baseline")
    sys.modules[module.__name__] = module
    exec(compile(code, f"{BASELINE_COMMIT}:{MODULE}", "exec"), module.__dict__)
    return module


def compare(source, output, *, repeats=3):
    if type(repeats) is not int or repeats < 2:
        raise ValueError("At least two repeated subset measurements are required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    legacy = legacy_module()
    if legacy.SPEC != preprocessing.SPEC:
        raise ValueError("The feature specification changed")
    code_hashes = {p.name: sha256(p) for p in sorted((ROOT / "sleepkit/recipes/detection").glob("*.py"))}
    runner_hash = sha256(Path(__file__))
    declaration = {
        "schema": "sleepkit.preparation_equivalence/v1",
        "declared_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_commit": BASELINE_COMMIT,
        "baseline_preprocessing_sha256": BASELINE_SHA256,
        "candidate_code_sha256": code_hashes,
        "runner_sha256": runner_hash,
        "dataset": source.provenance,
        "source_sha256": preprocessing.fingerprint(source.source_hashes),
        "split_sha256": preprocessing.fingerprint(source.split),
        "spec_sha256": preprocessing.fingerprint(preprocessing.SPEC),
        "numpy": version("numpy"),
        "python": platform.python_version(),
        "partition": "train",
        "comparison": "Exact dtype, shape and bytes of values, valid and ends; no cache reads",
        "timing": "perf_counter around extract only; source reads and equality checks excluded",
        "repeated_subset": "First three training series in frozen order, or all if fewer; alternating implementation order",
        "repeats": repeats,
    }
    write_json(output / "declaration.json", declaration)
    declaration_hash = sha256(output / "declaration.json")
    source.verify_unchanged()
    timings = {"scalar": [], "vectorized": []}
    digest = hashlib.sha256()
    frames = 0
    subjects = source.split["train"]

    def timed(fn, data, sample_time):
        start = time.perf_counter()
        result = fn(data, sample_time=sample_time)
        return result, time.perf_counter() - start

    def identical(a, b):
        for name in ("values", "valid", "ends"):
            x, y = getattr(a, name), getattr(b, name)
            if x.shape != y.shape or x.dtype != y.dtype or x.tobytes() != y.tobytes():
                raise ValueError(f"Scalar/vectorized {name} mismatch")

    def features():
        nonlocal frames
        for index, subject in enumerate(subjects):
            recording = source.read(subject, labels=False)
            # Alternate execution order to reduce systematic warm-cache ordering bias.
            order = [("scalar", legacy.extract), ("vectorized", preprocessing.extract)]
            if index % 2:
                order.reverse()
            results = {}
            for name, fn in order:
                results[name], elapsed = timed(fn, recording.data, recording.sample_time)
                timings[name].append(elapsed)
            identical(results["scalar"], results["vectorized"])
            current = results["vectorized"]
            for name in ("values", "valid", "ends"):
                array = getattr(current, name)
                digest.update(str(array.shape).encode() + str(array.dtype).encode() + array.tobytes())
            frames += len(current.values)
            if (index + 1) % 20 == 0 or index + 1 == len(subjects):
                print(f"Compared {index + 1}/{len(subjects)} training series", flush=True)
            yield current

    state = preprocessing.Normalizer.fit(features())
    repeated = {"scalar": [], "vectorized": []}
    for repeat in range(repeats):
        totals = {"scalar": 0.0, "vectorized": 0.0}
        for index, subject in enumerate(subjects[:3]):
            recording = source.read(subject, labels=False)
            order = [("scalar", legacy.extract), ("vectorized", preprocessing.extract)]
            if (repeat + index) % 2:
                order.reverse()
            results = {}
            for name, fn in order:
                results[name], elapsed = timed(fn, recording.data, recording.sample_time)
                totals[name] += elapsed
            identical(results["scalar"], results["vectorized"])
        for name in repeated:
            repeated[name].append(totals[name])
    source.verify_unchanged()
    if (
        sha256(output / "declaration.json") != declaration_hash
        or sha256(Path(__file__)) != runner_hash
        or {p.name: sha256(p) for p in sorted((ROOT / "sleepkit/recipes/detection").glob("*.py"))} != code_hashes
    ):
        raise ValueError("Code or declaration changed during comparison")
    report = {
        "schema": declaration["schema"],
        "status": "passed",
        "declaration_sha256": declaration_hash,
        "training_subjects": len(subjects),
        "feature_frames_compared": frames,
        "feature_arrays_sha256": digest.hexdigest(),
        "normalizer_sha256": preprocessing.fingerprint(state.to_dict()),
        "normalization_frames": state.count,
        "full_partition_extract_seconds": {name: sum(values) for name, values in timings.items()},
        "repeated_subset_extract_seconds": repeated,
        "limitations": [
            "One development machine; no controlled system-load isolation or cross-backend claim.",
            "Subset repeats measure extraction only, not complete preparation or training.",
            "Exact equality is verified for this corpus and environment, not all possible floating-point inputs.",
        ],
    }
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "events", "alignment", "coverage", "frozen", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    source = AnnotatedDataset(args.data, args.events, args.alignment, args.coverage, args.frozen)
    print(json.dumps(compare(source, args.output), indent=2))


if __name__ == "__main__":
    main()

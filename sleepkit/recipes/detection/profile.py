"""Measure this concrete TensorFlow recipe without exporting or selecting a model."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import os
from pathlib import Path
import platform
import resource
import time

import numpy as np

from sleepkit.artifacts.package import sha256, write_json
from .data import examples, subject_features
from .preprocessing import Normalizer, SPEC, fingerprint
from .recipe import Config, dataset, training_model


def _rss():
    # ru_maxrss is a lifetime high-water mark, not the current or per-stage RSS.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def _summary(seconds):
    return {
        "count": len(seconds),
        "total_seconds": float(sum(seconds)),
        "first_seconds": float(seconds[0]),
        "median_seconds": float(np.median(seconds)),
        "p95_seconds": float(np.percentile(seconds, 95)),
    }


def profile_membership(source, output, cfg=Config(), *, resident_steps=20, warmup_steps=2, data_kind="cmidss"):
    """Profile training only, keeping the reader's integrity checks and finite schedule.

    The output is a local workspace containing a private feature cache. Only
    report.json/declaration.json contain aggregate, path-free evidence. Use a fresh
    process for comparisons. A failed run keeps its declaration but has no report.
    """
    if any(type(n) is not int or n < 1 for n in (resident_steps, warmup_steps)):
        raise ValueError("Resident and warmup steps must be positive integers")
    if cfg.context != source.context:
        raise ValueError("Context must match the frozen evaluation protocol")
    if data_kind not in {"cmidss", "synthetic"}:
        raise ValueError("Declare data_kind as cmidss or synthetic")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cache = output / "private-feature-cache"
    cache.mkdir()
    declaration = {
        "schema": "sleepkit.detection_profile/v1",
        "declared_utc": datetime.now(timezone.utc).isoformat(),
        "data_kind": data_kind,
        "config": asdict(cfg),
        "profile_epochs": 1,
        "resident_steps": resident_steps,
        "warmup_steps": warmup_steps,
        "dataset": source.provenance,
        "split_sha256": fingerprint(source.split),
        "source_sha256": fingerprint(source.source_hashes),
        "preprocessing": SPEC,
        "code_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "schedule": {
            "partition": "train",
            "replacement": False,
            "balancing": False,
            "shuffle_buffer": 256,
            "reshuffle_each_iteration": True,
            "drop_partial_batch": False,
            "complete_passes_per_fit": 1,
        },
        "scope": "Performance only; disposable models; no validation/test evaluation or export",
    }
    write_json(output / "declaration.json", declaration)
    declaration_hash = sha256(output / "declaration.json")
    started = time.perf_counter()
    import keras
    import tensorflow as tf

    if keras.backend.backend() != "tensorflow":
        raise ValueError("This profiler measures the TensorFlow detection recipe")
    initialization_seconds = time.perf_counter() - started
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "versions": {name: version(name) for name in ("tensorflow", "keras", "numpy", "h5py")},
        "devices": [{"name": d.name, "type": d.device_type} for d in tf.config.list_physical_devices()],
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "TF_NUM_INTRAOP_THREADS",
                "TF_NUM_INTEROP_THREADS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "TF_DETERMINISTIC_OPS",
                "CUDA_VISIBLE_DEVICES",
                "KERAS_BACKEND",
            )
        },
        "tf_intraop_threads": tf.config.threading.get_intra_op_parallelism_threads(),
        "tf_interop_threads": tf.config.threading.get_inter_op_parallelism_threads(),
    }
    write_json(output / "environment.json", environment)
    environment_hash = sha256(output / "environment.json")
    stages = {}

    def measure(name, fn):
        start = time.perf_counter()
        cpu = time.process_time()
        value = fn()
        stages[name] = {
            "wall_seconds": time.perf_counter() - start,
            "process_cpu_seconds": time.process_time() - cpu,
            "process_lifetime_peak_rss_bytes": _rss(),
        }
        print(f"{name}: {stages[name]['wall_seconds']:.3f}s", flush=True)
        return value

    measure("verify_before", source.verify_unchanged)
    subjects = source.split["train"]
    reader = source.read

    def prepare_training():
        frames = valid = 0
        for features in subject_features(source.root, subjects, cache, reader=reader):
            frames += len(features.values)
            valid += int(features.valid.sum())
        return {"feature_frames": frames, "valid_feature_frames": valid}

    prepared_counts = measure("cold_feature_cache_prepare", prepare_training)
    normalizer = measure(
        "warm_fit_normalizer", lambda: Normalizer.fit(subject_features(source.root, subjects, cache, reader=reader))
    )
    count = measure(
        "count_eligible_contexts",
        lambda: sum(1 for _ in examples(source.root, subjects, normalizer, cfg.context, cache, reader=reader)),
    )
    if not count:
        raise ValueError("Training partition has no eligible contexts")
    expected_batches = (count + cfg.batch_size - 1) // cfg.batch_size

    def batches():
        return dataset(source.root, subjects, normalizer, cfg, cache, training=True, count=count, reader=reader)

    def load_epoch():
        waits = []
        contexts = 0
        resident = None
        iterator = iter(batches())
        for _ in range(expected_batches):
            start = time.perf_counter()
            x, y = next(iterator)
            x, y = x.numpy(), y.numpy()  # Complete host materialization inside the measured wait.
            waits.append(time.perf_counter() - start)
            contexts += len(x)
            if resident is None:
                resident = (x, y)  # Retain only one batch; never materialize an epoch in memory.
        try:
            next(iterator)
        except StopIteration:
            pass
        else:
            raise ValueError("Loader exceeded its declared finite epoch")
        if contexts != count:
            raise ValueError("Loader context count changed")
        return resident, waits

    keras.utils.set_random_seed(cfg.seed)
    resident, waits = measure("warm_loader_epoch", load_epoch)
    stages["warm_loader_epoch"].update(
        batch_wait=_summary(waits), contexts_per_second=count / stages["warm_loader_epoch"]["wall_seconds"]
    )
    model = measure("resident_model_build", lambda: training_model(cfg))
    measure("resident_compile_and_warmup", lambda: [model.train_on_batch(*resident) for _ in range(warmup_steps)])

    def resident_train():
        durations = []
        for _ in range(resident_steps):
            start = time.perf_counter()
            # Returned NumPy logs synchronize execution; transfers from this host batch are included.
            model.train_on_batch(*resident)
            durations.append(time.perf_counter() - start)
        return durations

    durations = measure("resident_model_steps", resident_train)
    stages["resident_model_steps"].update(
        steps=_summary(durations),
        contexts=resident_steps * len(resident[0]),
        contexts_per_second=resident_steps * len(resident[0]) / stages["resident_model_steps"]["wall_seconds"],
        host_batch_bytes=sum(a.nbytes for a in resident),
    )
    del model
    keras.backend.clear_session()
    model = measure("end_to_end_model_build", lambda: training_model(cfg))

    class BatchCounter(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.count = 0

        def on_train_batch_end(self, batch, logs=None):
            self.count += 1

    counter = BatchCounter()
    measure(
        "end_to_end_fit_epoch", lambda: model.fit(batches(), epochs=1, shuffle=False, verbose=0, callbacks=[counter])
    )
    if counter.count != expected_batches:
        raise ValueError("Fit did not consume the declared finite epoch")
    stages["end_to_end_fit_epoch"].update(
        batches=counter.count, contexts_per_second=count / stages["end_to_end_fit_epoch"]["wall_seconds"]
    )
    measure("verify_after", source.verify_unchanged)
    if (
        sha256(output / "declaration.json") != declaration_hash
        or sha256(output / "environment.json") != environment_hash
        or {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))} != declaration["code_sha256"]
    ):
        raise ValueError("Profile declaration, environment, or source code changed during measurement")
    report = {
        "schema": declaration["schema"],
        "status": "complete",
        "declaration_sha256": declaration_hash,
        "environment_sha256": environment_hash,
        "normalizer_sha256": fingerprint(normalizer.to_dict()),
        "resident_batch": {
            "input_sha256": hashlib.sha256(resident[0].tobytes()).hexdigest(),
            "target_sha256": hashlib.sha256(resident[1].tobytes()).hexdigest(),
            "input_shape": list(resident[0].shape),
            "target_shape": list(resident[1].shape),
            "input_dtype": str(resident[0].dtype),
            "target_dtype": str(resident[1].dtype),
        },
        "framework_import_seconds": initialization_seconds,
        "counts": {
            **prepared_counts,
            "normalization_frames": normalizer.count,
            "subjects": len(subjects),
            "contexts": count,
            "feature_frames_trained": count * cfg.context,
            "batches": expected_batches,
            "last_batch_contexts": (count - 1) % cfg.batch_size + 1,
        },
        "stages": stages,
        "limitations": [
            "Cold means an empty feature cache, not a cold operating-system disk cache; preparation is serial.",
            "Reader hashing, evidence validation and target reconstruction remain enabled, including uncached count checks.",
            "Warm loader waits include next() and host materialization; prefetch overlaps work. They are not raw disk latency.",
            "Resident steps repeat one host batch with transfers included; not device-only compute or an epoch schedule.",
            "Fit is a fresh-model single training epoch after resident process/device warmup; includes tracing/compilation, excludes validation, export and model build.",
            "RSS is process lifetime high-water memory, not stage allocation or device memory; no GPU memory measurement.",
            "Single observations, not a speedup claim. Use fresh processes and repeated runs for comparisons.",
        ],
    }
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "events", "alignment", "coverage", "frozen", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resident-steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=2)
    args = parser.parse_args()
    from .target_dataset import AnnotatedDataset

    source = AnnotatedDataset(args.data, args.events, args.alignment, args.coverage, args.frozen)
    profile_membership(
        source,
        args.output,
        Config(context=source.context, batch_size=args.batch_size, seed=args.seed),
        resident_steps=args.resident_steps,
        warmup_steps=args.warmup_steps,
    )


if __name__ == "__main__":
    main()

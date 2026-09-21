"""Readable recipe: prepare → fit → train → evaluate → export. No orchestration config."""

from dataclasses import asdict, dataclass
from pathlib import Path
import tempfile

import numpy as np

from sleepkit.artifacts.package import sha256, write_json
from sleepkit.recipes._components import ClassificationAccumulator, FileSnapshot, implementation_files
from .data import READER_SPEC, examples, load_split, subject_features
from .model import build_model
from .output_contract import output_contract
from .preprocessing import Normalizer, fingerprint


@dataclass(frozen=True)
class Config:
    context: int = 240
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-3
    seed: int = 0

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in (self.context, self.epochs, self.batch_size)):
            raise ValueError("Context, epochs, and batch size must be positive integers")
        if (
            not np.isfinite(self.learning_rate)
            or self.learning_rate <= 0
            or type(self.seed) is not int
            or self.seed < 0
        ):
            raise ValueError("Learning rate must be finite and positive; seed must be a nonnegative integer")


def dataset(root, subjects, normalizer, cfg, cache, *, training, count, reader=None):
    import tensorflow as tf

    data = tf.data.Dataset.from_generator(
        lambda: examples(root, subjects, normalizer, cfg.context, cache, reader=reader),
        output_signature=(tf.TensorSpec((cfg.context, 5), tf.float32), tf.TensorSpec((cfg.context,), tf.int32)),
    )
    data = data.apply(tf.data.experimental.assert_cardinality(count))
    if training:
        data = data.shuffle(256, seed=cfg.seed, reshuffle_each_iteration=True)
    # Bounded prefetch/thread pool; only one subject and a bounded shuffle buffer are loaded.
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    return data.batch(cfg.batch_size).with_options(options).prefetch(1)


def training_model(cfg, model_builder=build_model):
    """Build and compile the recipe's logits model; usable without running the pipeline."""
    import keras

    keras.utils.set_random_seed(cfg.seed)
    model = model_builder(cfg.context, 5)
    if (
        model.input_shape != (None, cfg.context, 5)
        or model.output_shape != (None, cfg.context, 2)
        or getattr(model.layers[-1], "activation", None) is not keras.activations.linear
    ):
        raise ValueError("Builder must return float32 [batch,context,5] → [batch,context,2] logits with linear output")
    model.compile(
        optimizer=keras.optimizers.Adam(cfg.learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def evaluate(model, batches, *, class_names=None, prediction_batches=None):
    accumulator = ClassificationAccumulator(2)
    for x, y in batches:
        logits = np.asarray(model(x, training=False), dtype=np.float64)
        labels = np.asarray(y)
        accumulator.update(labels, logits)
        if prediction_batches is not None:
            prediction_batches.append((logits.reshape(-1, 2).copy(), labels.reshape(-1).astype(np.int32)))
    if not accumulator.confusion.sum():
        raise ValueError("No valid held-out contexts")
    result = accumulator.result()
    return {
        "model": "model.keras", "split": "test",
        "feature_frames_evaluated": result["count"],
        **{key: result[key] for key in ("confusion_matrix", "accuracy", "macro_f1", "cross_entropy", "f1_zero_division")},
        "class_order": ["WAKE", "SLEEP"] if class_names is None else list(class_names),
    }


def _run(
    root,
    split_file,
    output,
    cfg=Config(),
    *,
    cache=None,
    model_builder=build_model,
    callbacks=(),
    data_kind="cmidss",
    prepared=None,
):
    """Compose ordinary functions; replace the logits model builder or Keras callbacks in Python.

    Exact splits and file provenance stay in the local run directory. The release
    contains hashes/counts plus fitted state, not subject identifiers or recordings.
    """
    import keras

    if keras.backend.backend() != "tensorflow":
        raise ValueError("This recipe's conversion adapter requires the TensorFlow Keras backend")
    if data_kind not in {"cmidss", "synthetic"}:
        raise ValueError("Declare data_kind as cmidss or synthetic")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    reader = None if prepared is None else prepared.read
    target = None if prepared is None else prepared.target
    _, class_names = output_contract(target)
    if prepared is not None:
        if cfg.context != prepared.context:
            raise ValueError("Context must match the frozen evaluation protocol")
        prepared.verify_unchanged()
    split = load_split(split_file, root)
    if prepared is not None and split != prepared.split:
        raise ValueError("Split differs from the frozen protocol")
    source_snapshot = FileSnapshot.capture({
        subject: Path(root) / f"{subject}.h5" for group in split.values() for subject in group
    })
    source_hashes = source_snapshot.hashes()
    code_snapshot = FileSnapshot.capture(implementation_files(Path(__file__).parent))
    normalizer = Normalizer.fit(subject_features(root, split["train"], cache, reader=reader))
    counts = {
        name: sum(1 for _ in examples(root, subjects, normalizer, cfg.context, cache, reader=reader))
        for name, subjects in split.items()
    }
    if not all(counts.values()):
        raise ValueError(f"Every partition needs complete valid labeled contexts; found {counts}")
    model = training_model(cfg, model_builder)
    train = dataset(root, split["train"], normalizer, cfg, cache, training=True, count=counts["train"], reader=reader)
    validation = dataset(
        root, split["validation"], normalizer, cfg, cache, training=False, count=counts["validation"], reader=reader
    )
    history = model.fit(
        train, validation_data=validation, epochs=cfg.epochs, callbacks=list(callbacks), shuffle=False, verbose=2
    )
    prediction_batches = [] if prepared is not None else None
    metrics = evaluate(
        model,
        dataset(root, split["test"], normalizer, cfg, cache, training=False, count=counts["test"], reader=reader),
        class_names=class_names,
        prediction_batches=prediction_batches,
    )
    metadata = {
        "config": asdict(cfg),
        "reader": READER_SPEC,
        "data_kind": data_kind,
        "split_sha256": fingerprint(split),
        "subjects_per_split": {name: len(group) for name, group in split.items()},
        "contexts_per_split": counts,
        "source_sha256": fingerprint(source_hashes),
        "code_sha256": code_snapshot.hashes(),
    }
    if prepared is not None:
        metadata.update(target=target, dataset=prepared.provenance)
        metrics["target"] = target
        prepared.verify_unchanged()
    # Do not attach provenance to a run if source files changed during fitting/evaluation.
    source_snapshot.verify()
    code_snapshot.verify()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".detection-", dir=output.parent) as directory:
        directory = Path(directory)
        write_json(directory / "split.json", split)
        write_json(directory / "sources.json", source_hashes)
        write_json(directory / "history.json", history.history)
        if prepared is not None:
            from .scoring import write_index

            evaluated_targets = np.concatenate([y for _, y in prediction_batches])
            index = write_index(
                prepared,
                split["test"],
                cfg.context,
                directory / "test-index.jsonl",
                cache=cache,
                expected_targets=evaluated_targets,
            )
            if index["eligible_outputs"] != metrics["feature_frames_evaluated"]:
                raise ValueError("Scoring index does not match evaluated outputs")
            np.savez_compressed(
                directory / "test-predictions.npz",
                logits=np.concatenate([x for x, _ in prediction_batches]),
                targets=evaluated_targets,
            )
            index["predictions_sha256"] = sha256(directory / "test-predictions.npz")
            write_json(directory / "test-index-summary.json", index)
            metadata["scoring"] = index
            prepared.verify_unchanged()
        from .export import export_bundle

        export_bundle(model, normalizer, directory / "bundle", metadata, metrics)
        if prepared is not None:
            prepared.verify_unchanged()
        source_snapshot.verify()
        code_snapshot.verify()
        if set(implementation_files(Path(__file__).parent)) != set(code_snapshot.hashes()):
            raise ValueError("Recipe implementation inventory changed during the run")
        directory.rename(output)
    return output


def run(
    root, split_file, output, cfg=Config(), *, cache=None, model_builder=build_model, callbacks=(), data_kind="cmidss"
):
    """Historical-label recipe, preserved for existing experiments."""
    return _run(
        root,
        split_file,
        output,
        cfg,
        cache=cache,
        model_builder=model_builder,
        callbacks=callbacks,
        data_kind=data_kind,
    )


def run_membership(
    source, output, cfg=Config(), *, cache=None, model_builder=build_model, callbacks=(), data_kind="cmidss"
):
    """Compose an AnnotatedDataset with training, evaluation, and versioned export."""
    return _run(
        source.root,
        source.split_path,
        output,
        cfg,
        cache=cache,
        model_builder=model_builder,
        callbacks=callbacks,
        data_kind=data_kind,
        prepared=source,
    )

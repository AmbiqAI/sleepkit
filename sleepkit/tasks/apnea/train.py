import os
import shutil

import keras
import neuralspot_edge as nse
import numpy as np
import sklearn.model_selection
import sklearn.utils
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

from ...defines import TaskParams
from ...models import ModelFactory
from ...features import H5Dataloader
from .utils import create_data_pipeline


def train(params: TaskParams):
    """Train sleep apnea model.

    Args:
        params (TaskParams): Task parameters
    """

    os.makedirs(params.job_dir, exist_ok=True)
    logger = nse.utils.setup_logger(__name__, level=params.verbose, file_path=params.job_dir / "train.log")
    logger.debug(f"Creating working directory in {params.job_dir}")

    params.seed = nse.utils.set_random_seed(params.seed)
    logger.debug(f"Random seed {params.seed}")

    with open(params.job_dir / "configuration.json", "w", encoding="utf-8") as fp:
        fp.write(params.model_dump_json(indent=2))
    # END WITH

    if nse.utils.env_flag("WANDB"):
        wandb.init(
            project=params.project,
            entity="ambiq",
            dir=params.job_dir,
        )
        wandb.config.update(params.model_dump())
    # END IF

    target_classes = sorted(set(params.class_map.values()))
    class_names = params.class_names or [f"Class {i}" for i in range(params.num_classes)]

    # Load features
    dataloader = H5Dataloader(
        path=params.feature.save_path,
        frame_size=params.frame_size,
        feat_key=params.feature.feat_key,
        label_key=params.feature.label_key,
        mask_key=params.feature.mask_key,
        feat_cols=params.feature.feat_cols,
        class_map=params.class_map,
    )

    feat_shape = dataloader.feature_shape

    # Get train/val subject IDs and generators
    train_subject_ids, val_subject_ids = sklearn.model_selection.train_test_split(
        dataloader.subject_ids, test_size=params.val_subjects
    )

    logger.debug("Loading training dataset")
    train_ds = create_data_pipeline(
        dataloader=dataloader,
        subject_ids=train_subject_ids,
        samples_per_subject=params.samples_per_subject,
        num_classes=params.num_classes,
        batch_size=params.batch_size,
        buffer_size=params.buffer_size,
    )

    logger.debug("Loading validation dataset")
    val_cache_size = params.val_size // params.batch_size if params.val_size else params.val_steps_per_epoch
    val_ds = create_data_pipeline(
        dataloader=dataloader,
        subject_ids=val_subject_ids,
        samples_per_subject=params.val_samples_per_subject,
        num_classes=params.num_classes,
        batch_size=params.batch_size,
        cache_size=val_cache_size,
    )
    y_true = np.concatenate([y for _, y in val_ds.as_numpy_iterator()])
    y_true = np.argmax(y_true, axis=-1).flatten()

    class_weights = 0.25
    if params.class_weights == "balanced":
        class_weights = sklearn.utils.compute_class_weight("balanced", classes=np.array(target_classes), y=y_true)
        class_weights = (class_weights + class_weights.mean()) / 2
        class_weights = class_weights.tolist()
    # END IF

    logger.debug("Building model")
    inputs = keras.Input(
        shape=feat_shape,
        batch_size=None,
        name="input",
        dtype="float32",
    )
    if params.resume and params.model_file:
        logger.debug(f"Loading model from file {params.model_file}")
        model = nse.models.load_model(params.model_file)
    else:
        logger.debug("Creating model from scratch")
        if params.architecture is None:
            raise ValueError("Model architecture must be specified")
        model = ModelFactory.get(params.architecture.name)(
            inputs=inputs,
            params=params.architecture.params,
            num_classes=params.num_classes,
        )
    # END IF

    flops = nse.metrics.flops.get_flops(model, batch_size=1, fpath=str(params.job_dir / "model_flops.log"))

    t_mul = 1
    first_steps = (params.steps_per_epoch * params.epochs) / (np.power(params.lr_cycles, t_mul) - t_mul + 1)
    scheduler = keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=params.lr_rate,
        first_decay_steps=np.ceil(first_steps),
        t_mul=t_mul,
        m_mul=0.5,
    )

    optimizer = keras.optimizers.Adam(scheduler)
    loss = keras.losses.CategoricalFocalCrossentropy(
        from_logits=True,
        alpha=class_weights,
        label_smoothing=params.label_smoothing,
    )
    metrics = [
        keras.metrics.CategoricalAccuracy(name="acc"),
        nse.metrics.MultiF1Score(name="f1", average="weighted"),
        keras.metrics.AUC(name="auc"),
        keras.metrics.OneHotIoU(
            num_classes=len(target_classes),
            target_class_ids=target_classes,
            name="iou",
        ),
    ]

    if params.resume and params.weights_file:
        logger.debug(f"Hydrating model weights from file {params.weights_file}")
        model.load_weights(params.weights_file)
    # END IF

    if params.model_file is None:
        params.model_file = params.job_dir / "model.keras"
    # END IF

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model(inputs)
    model.summary(print_fn=logger.debug)
    logger.debug(f"Model requires {flops / 1e6:0.2f} MFLOPS")

    # Remove existing logs
    if os.path.exists(params.job_dir / "logs"):
        shutil.rmtree(params.job_dir / "logs")

    ModelCheckpoint = keras.callbacks.ModelCheckpoint
    if nse.utils.env_flag("WANDB"):
        ModelCheckpoint = WandbModelCheckpoint
    model_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor=f"val_{params.val_metric}",
            patience=max(int(0.25 * params.epochs), 1),
            mode="max" if params.val_metric == "f1" else "auto",
            restore_best_weights=True,
        ),
        ModelCheckpoint(
            filepath=str(params.model_file),
            monitor=f"val_{params.val_metric}",
            save_best_only=True,
            mode="max" if params.val_metric == "f1" else "auto",
            verbose=1,
        ),
        keras.callbacks.CSVLogger(params.job_dir / "history.csv"),
    ]
    if nse.utils.env_flag("TENSORBOARD"):
        model_callbacks.append(
            keras.callbacks.TensorBoard(
                log_dir=params.job_dir / "logs",
                write_steps_per_second=True,
            )
        )
    if nse.utils.env_flag("WANDB"):
        model_callbacks.append(WandbMetricsLogger())

    try:
        history = model.fit(
            train_ds,
            steps_per_epoch=params.steps_per_epoch,
            verbose=2,
            epochs=params.epochs,
            validation_data=val_ds,
            callbacks=model_callbacks,
        )
    except KeyboardInterrupt:
        logger.warning("Stopping training due to keyboard interrupt")

    logger.debug(f"Model saved to {params.model_file}")

    nse.plotting.plot_history_metrics(
        history.history,
        metrics=["loss", "f1"],
        save_path=params.job_dir / "history.png",
        title="Training History",
        stack=True,
        figsize=(9, 5),
    )

    logger.debug("Performing full validation")
    rst = model.evaluate(val_ds, verbose=params.verbose, return_dict=True)
    logger.info("[VAL SET] " + ", ".join(f"{k.upper()}={v:.4f}" for k, v in rst.items()))

    y_pred = np.argmax(model.predict(val_ds).squeeze(), axis=-1).flatten()
    cm_path = params.job_dir / "confusion_matrix.png"
    nse.plotting.cm.confusion_matrix_plot(
        y_true=y_true,
        y_pred=y_pred,
        labels=class_names,
        save_path=cm_path,
        normalize="true",
    )
    if nse.utils.env_flag("WANDB"):
        conf_mat = wandb.plot.confusion_matrix(preds=y_pred, y_true=y_true, class_names=class_names)
        wandb.log({"conf_mat": conf_mat})
    # END IF

import os
import json

import numpy as np
import pandas as pd
import keras
import neuralspot_edge as nse
from tqdm import tqdm

from ...defines import TaskParams
from ...features import H5Dataloader
from .metrics import (
    compute_sleep_efficiency,
    compute_sleep_stage_durations,
    compute_total_sleep_time,
)
from .utils import create_data_pipeline, subject_data_preprocessor


def evaluate(params: TaskParams):
    """Evaluate sleep stage model.

    Args:
        params (TaskParams): Testing/evaluation parameters
    """
    os.makedirs(params.job_dir, exist_ok=True)
    logger = nse.utils.setup_logger(__name__, level=params.verbose, file_path=params.job_dir / "test.log")
    logger.debug(f"Creating working directory in {params.job_dir}")

    params.seed = nse.utils.set_random_seed(params.seed)
    logger.debug(f"Random seed {params.seed}")

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

    test_ds = create_data_pipeline(
        dataloader=dataloader,
        subject_ids=dataloader.test_subject_ids,
        samples_per_subject=params.test_samples_per_subject,
        num_classes=params.num_classes,
        batch_size=params.batch_size,
        cache_size=params.test_size // params.batch_size,
    )

    test_true, test_pred, test_prob = [], [], []
    pt_metrics = []

    logger.debug("Loading model")
    model = nse.models.load_model(params.model_file)
    flops = nse.metrics.flops.get_flops(model, batch_size=1, fpath=params.job_dir / "model_flops.log")
    model.summary(print_fn=logger.debug)
    logger.debug(f"Model requires {flops / 1e6:0.2f} MFLOPS")

    logger.debug("Performing full inference")
    for subject_id in tqdm(dataloader.test_subject_ids, desc="Subject"):
        features, labels, mask = dataloader.load_subject_data(
            subject_id=subject_id, preprocessor=subject_data_preprocessor
        )
        num_windows = int(features.shape[0] // params.frame_size)
        data_len = params.frame_size * num_windows

        x = features[:data_len, :].reshape((num_windows, params.frame_size) + dataloader.feature_shape[1:])
        m = mask[:data_len].reshape((num_windows, params.frame_size))
        # m[:, :64] = 0 # Ignore first N samples
        y_prob = keras.ops.softmax(model.predict(x, verbose=0)).numpy()
        y_pred = np.argmax(y_prob, axis=-1).flatten()
        y_prob = y_prob.reshape((-1, y_prob.shape[-1]))
        # y_mask = mask[:data_len].flatten()
        y_mask = m.flatten()
        y_true = labels[:data_len].flatten()
        y_pred = y_pred[y_mask == 1]
        y_true = y_true[y_mask == 1]
        y_prob = y_prob[y_mask == 1]

        # Get subject specific metrics
        pred_sleep_durations = compute_sleep_stage_durations(y_pred)
        pred_sleep_tst = compute_total_sleep_time(pred_sleep_durations, params.class_map)
        pred_sleep_eff = compute_sleep_efficiency(pred_sleep_durations, params.class_map)
        act_sleep_duration = compute_sleep_stage_durations(y_true)
        act_sleep_tst = compute_total_sleep_time(act_sleep_duration, params.class_map)
        act_sleep_eff = compute_sleep_efficiency(act_sleep_duration, params.class_map)
        pt_acc = np.sum(y_pred == y_true) / y_true.size
        pt_metrics.append(
            [
                subject_id,
                pt_acc,
                act_sleep_eff,
                pred_sleep_eff,
                act_sleep_tst,
                pred_sleep_tst,
            ]
        )
        test_true.append(y_true)
        test_pred.append(y_pred)
        test_prob.append(y_prob)
    # END FOR
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_prob = np.vstack(test_prob)

    df_metrics = pd.DataFrame(
        pt_metrics,
        columns=["subject", "acc", "act_eff", "pred_eff", "act_tst", "pred_tst"],
    )
    df_metrics.to_csv(params.job_dir / "metrics.csv", header=True, index=False)

    df_results = pd.DataFrame(dict(y_true=test_true, y_pred=test_pred))
    df_results.to_csv(params.job_dir / "results.csv", header=True, index=False)

    nse.plotting.cm.confusion_matrix_plot(
        y_true=test_true,
        y_pred=test_pred,
        labels=class_names,
        save_path=params.job_dir / "confusion_matrix_test.png",
        normalize="true",
    )

    # Summarize results
    # Summarize results
    logger.info("Testing Results")
    rsts = model.evaluate(test_ds, verbose=params.verbose, return_dict=True)
    logger.info("[TEST SET] " + ", ".join([f"{k}={v:.2%}" for k, v in rsts.items()]))

    rsts["flops"] = flops
    rsts["parameters"] = model.count_params()
    with open(params.job_dir / "metrics.json", "w") as fp:
        json.dump(rsts, fp)

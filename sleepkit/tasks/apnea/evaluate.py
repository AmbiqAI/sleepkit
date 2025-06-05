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
    compute_apnea_efficiency,
    compute_apnea_hypopnea_index,
    compute_sleep_apnea_durations,
)
from .utils import create_data_pipeline, subject_data_preprocessor


def evaluate(params: TaskParams):
    """Evaluate sleep apnea model.

    Args:
        params (TaskParams): Task parameters
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
    )

    test_ds = create_data_pipeline(
        dataloader=dataloader,
        subject_ids=dataloader.test_subject_ids,
        samples_per_subject=params.test_samples_per_subject,
        num_classes=params.num_classes,
        batch_size=params.batch_size,
        cache_size=params.test_size // params.batch_size,
    )

    test_true, test_pred = [], []
    pt_metrics = []

    logger.debug("Loading model")
    model = nse.models.load_model(params.model_file)
    flops = nse.metrics.flops.get_flops(model, batch_size=1, fpath=params.job_dir / "model_flops.log")
    model.summary(print_fn=logger.debug)
    logger.debug(f"Model requires {flops / 1e6:0.2f} MFLOPS")

    logger.debug("Performing inference")
    for subject_id in tqdm(dataloader.test_subject_ids, desc="Subject"):
        features, labels, mask = dataloader.load_subject_data(
            subject_id=subject_id, preprocessor=subject_data_preprocessor
        )
        num_windows = int(features.shape[0] // dataloader.frame_size)
        data_len = dataloader.frame_size * num_windows

        x = features[:data_len, :].reshape((num_windows, dataloader.frame_size) + dataloader.feature_shape[1:])
        y_prob = keras.ops.softmax(model.predict(x, verbose=0)).numpy()
        y_pred = np.argmax(y_prob, axis=-1).flatten()
        y_mask = mask[:data_len].flatten()
        y_true = np.vectorize(params.class_map.get)(labels[:data_len].flatten())
        y_pred = y_pred[y_mask == 1]
        y_true = y_true[y_mask == 1]

        # Get subject specific metrics
        act_apnea_durations = compute_sleep_apnea_durations(y_true)
        pred_apnea_durations = compute_sleep_apnea_durations(y_pred)
        act_eff = compute_apnea_efficiency(act_apnea_durations, class_map=params.class_map)
        pred_eff = compute_apnea_efficiency(pred_apnea_durations, class_map=params.class_map)
        act_ahi = compute_apnea_hypopnea_index(
            y_true,
            min_duration=int(10 * params.sampling_rate),
            sample_rate=params.sampling_rate,
        )
        pred_ahi = compute_apnea_hypopnea_index(
            y_pred,
            min_duration=int(10 * params.sampling_rate),
            sample_rate=params.sampling_rate,
        )
        pt_acc = np.sum(y_pred == y_true) / y_true.size
        pt_metrics.append([subject_id, pt_acc, act_eff, pred_eff, act_ahi, pred_ahi])
        test_true.append(y_true)
        test_pred.append(y_pred)
    # END FOR

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)

    df_metrics = pd.DataFrame(
        pt_metrics,
        columns=["subject_id", "acc", "act_eff", "pred_eff", "act_ahi", "pred_ahi"],
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
    logger.info("Testing Results")
    rsts = model.evaluate(test_ds, verbose=params.verbose, return_dict=True)
    logger.info("[TEST SET] " + ", ".join([f"{k}={v:.2%}" for k, v in rsts.items()]))

    rsts["flops"] = flops
    rsts["parameters"] = model.count_params()
    with open(params.job_dir / "metrics.json", "w") as fp:
        json.dump(rsts, fp)

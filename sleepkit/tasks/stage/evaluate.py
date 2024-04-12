"""Sleep Stage Evaluation"""

import logging
import os

import numpy as np
import pandas as pd
import sklearn.metrics
import sklearn.model_selection
import sklearn.utils
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tqdm import tqdm

from ... import tflite as tfa
from ...defines import SKTestParams
from ...metrics import compute_iou, confusion_matrix_plot, f1_score
from ...utils import set_random_seed, setup_logger
from .metrics import (
    compute_sleep_efficiency,
    compute_sleep_stage_durations,
    compute_total_sleep_time,
)
from .utils import load_dataset

logger = setup_logger(__name__)


def evaluate(params: SKTestParams):
    """Evaluate sleep stage model.

    Args:
        params (SKTestParams): Testing/evaluation parameters
    """
    params.seed = set_random_seed(params.seed)
    logger.info(f"Random seed {params.seed}")

    logger.info(f"Creating working directory in {params.job_dir}")
    os.makedirs(params.job_dir, exist_ok=True)

    handler = logging.FileHandler(params.job_dir / "test.log", mode="w")
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    # target_classes = sorted(list(set(params.class_map.values())))
    class_names = params.class_names or [f"Class {i}" for i in range(params.num_classes)]

    ds = load_dataset(
        ds_path=params.ds_path,
        frame_size=params.frame_size,
        dataset=params.dataset,
    )
    test_true, test_pred, test_prob = [], [], []
    pt_metrics = []

    strategy = tfa.get_strategy()
    with strategy.scope(), tfmot.quantization.keras.quantize_scope():
        logger.info("Loading model")
        model = tfa.load_model(params.model_file, custom_objects={"MultiF1Score": tfa.MultiF1Score})
        flops = tfa.get_flops(model, batch_size=1, fpath=params.job_dir / "model_flops.log")
        model.summary(print_fn=logger.info)
        logger.info(f"Model requires {flops/1e6:0.2f} MFLOPS")

        logger.info("Performing full inference")
        for subject_id in tqdm(ds.test_subject_ids, desc="Subject"):
            features, labels, mask = ds.load_subject_data(subject_id=subject_id, normalize=True)
            num_windows = int(features.shape[0] // params.frame_size)
            data_len = params.frame_size * num_windows

            x = features[:data_len, :].reshape((num_windows, params.frame_size) + ds.feature_shape[1:])
            m = mask[:data_len].reshape((num_windows, params.frame_size))
            # m[:, :64] = 0 # Ignore first N samples
            y_prob = tf.nn.softmax(model.predict(x, verbose=0)).numpy()
            y_pred = np.argmax(y_prob, axis=-1).flatten()
            y_prob = y_prob.reshape((-1, y_prob.shape[-1]))
            # y_mask = mask[:data_len].flatten()
            y_mask = m.flatten()
            y_true = np.vectorize(params.class_map.get)(labels[:data_len].flatten())
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
            pt_metrics.append([subject_id, pt_acc, act_sleep_eff, pred_sleep_eff, act_sleep_tst, pred_sleep_tst])
            test_true.append(y_true)
            test_pred.append(y_pred)
            test_prob.append(y_prob)
        # END FOR
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_prob = np.vstack(test_prob)

        df_metrics = pd.DataFrame(pt_metrics, columns=["subject", "acc", "act_eff", "pred_eff", "act_tst", "pred_tst"])
        df_metrics.to_csv(params.job_dir / "metrics.csv", header=True, index=False)

        df_results = pd.DataFrame(dict(y_true=test_true, y_pred=test_pred))
        df_results.to_csv(params.job_dir / "results.csv", header=True, index=False)

        confusion_matrix_plot(
            y_true=test_true,
            y_pred=test_pred,
            labels=class_names,
            save_path=params.job_dir / "confusion_matrix_test.png",
            normalize="true",
        )

        # Summarize results
        logger.info("Testing Results")
        test_acc = np.sum(test_pred == test_true) / test_true.size
        test_f1 = f1_score(y_true=test_true, y_pred=test_pred, average="weighted")
        y_scores = test_prob[:, 1] if params.num_classes == 2 else test_prob
        test_ap = sklearn.metrics.average_precision_score(y_true=test_true, y_score=y_scores, average="weighted")
        test_iou = compute_iou(test_true, test_pred, average="weighted")
        logger.info(f"[TEST SET] ACC={test_acc:.2%}, F1={test_f1:.2%}, AP={test_ap:0.2%}, IoU={test_iou:0.2%}")
    # END WITH

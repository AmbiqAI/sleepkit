"""Sleep Apnea Evaluation"""

import numpy as np
import pandas as pd
import tensorflow as tf
from rich.console import Console
from tqdm import tqdm

from .. import tflite as tfa
from ..defines import SKTestParams
from ..metrics import compute_iou, confusion_matrix_plot, f1_score
from ..utils import set_random_seed, setup_logger
from .defines import get_sleep_apnea_class_mapping, get_sleep_apnea_class_names
from .metrics import (
    compute_apnea_efficiency,
    compute_apnea_hypopnea_index,
    compute_sleep_apnea_durations,
)
from .utils import load_dataset

console = Console()
logger = setup_logger(__name__)


def evaluate(params: SKTestParams):
    """Evaluate sleep apnea model.

    Args:
        params (SKTestParams): Testing/evaluation parameters
    """
    params.seed = set_random_seed(params.seed)
    logger.info(f"Random seed {params.seed}")

    # target_classes = get_sleep_apnea_classes(params.num_classes)
    class_names = get_sleep_apnea_class_names(params.num_classes)
    class_mapping = get_sleep_apnea_class_mapping(params.num_classes)

    ds = load_dataset(ds_path=params.ds_path, frame_size=params.frame_size, feat_cols=params.feat_cols)
    feat_shape = ds.feature_shape
    test_true, test_pred = [], []
    pt_metrics = []

    strategy = tfa.get_strategy()
    with strategy.scope():
        logger.info("Loading model")
        model = tfa.load_model(params.model_file, custom_objects={"MultiF1Score": tfa.MultiF1Score})
        flops = tfa.get_flops(model, batch_size=1, fpath=params.job_dir / "model_flops.log")
        model.summary(print_fn=logger.info)
        logger.info(f"Model requires {flops/1e6:0.2f} MFLOPS")

        logger.info("Performing inference")
        for subject_id in tqdm(ds.test_subject_ids, desc="Subject"):
            features, labels, mask = ds.load_subject_data(subject_id=subject_id, normalize=True)
            num_windows = int(features.shape[0] // ds.frame_size)
            data_len = ds.frame_size * num_windows

            x = features[:data_len, :].reshape((num_windows, ds.frame_size) + feat_shape[1:])
            y_prob = tf.nn.softmax(model.predict(x, verbose=0)).numpy()
            y_pred = np.argmax(y_prob, axis=-1).flatten()
            y_mask = mask[:data_len].flatten()
            y_true = np.vectorize(class_mapping.get)(labels[:data_len].flatten())
            y_pred = y_pred[y_mask == 1]
            y_true = y_true[y_mask == 1]

            # Get subject specific metrics
            act_apnea_durations = compute_sleep_apnea_durations(y_true)
            pred_apnea_durations = compute_sleep_apnea_durations(y_pred)
            act_eff = compute_apnea_efficiency(act_apnea_durations, class_map=class_mapping)
            pred_eff = compute_apnea_efficiency(pred_apnea_durations, class_map=class_mapping)
            act_ahi = compute_apnea_hypopnea_index(y_true, min_duration=1, sample_rate=params.sampling_rate)
            pred_ahi = compute_apnea_hypopnea_index(y_pred, min_duration=1, sample_rate=params.sampling_rate)
            pt_acc = np.sum(y_pred == y_true) / y_true.size
            pt_metrics.append([subject_id, pt_acc, act_eff, pred_eff, act_ahi, pred_ahi])
            test_true.append(y_true)
            test_pred.append(y_pred)
        # END FOR

        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)

        df_metrics = pd.DataFrame(
            pt_metrics, columns=["subject_id", "acc", "act_eff", "pred_eff", "act_ahi", "pred_ahi"]
        )
        df_metrics.to_csv(params.job_dir / "metrics.csv", header=True, index=False)

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
        test_iou = compute_iou(test_true, test_pred, average="weighted")
        logger.info(f"[TEST SET] ACC={test_acc:.2%}, F1={test_f1:.2%} IoU={test_iou:0.2%}")
    # END WITH

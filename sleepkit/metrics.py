from typing import Literal

import numpy.typing as npt
from sklearn.metrics import jaccard_score


def compute_iou(
    y_true: npt.NDArray,
    y_pred: npt.NDArray,
    average: Literal["micro", "macro", "weighted"] = "micro",
) -> float:
    """Compute IoU

    Args:
        y_true (npt.NDArray): Y true
        y_pred (npt.NDArray): Y predicted

    Returns:
        float: IoU
    """
    return jaccard_score(y_true.flatten(), y_pred.flatten(), average=average)

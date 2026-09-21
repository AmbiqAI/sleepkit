"""Offline unweighted classification counts/losses; NumPy only, no task policy."""

import numpy as np


def summarize_confusion(confusion):
    """Rows are targets, columns predictions; absent-class F1 is zero."""
    matrix = np.asarray(confusion)
    if (
        matrix.ndim != 2
        or matrix.shape[0] < 2
        or matrix.shape[0] != matrix.shape[1]
        or matrix.dtype.kind not in "iu"
        or (matrix < 0).any()
    ):
        raise ValueError("Expected a nonnegative square integer confusion matrix with at least two classes")
    count = sum(map(int, matrix.flat))
    if count > np.iinfo(np.int64).max:
        raise ValueError("Confusion counts exceed int64 range")
    matrix = matrix.astype(np.int64, copy=False)
    if not count:
        raise ValueError("No classification samples")
    support = matrix.sum(axis=1)
    denominator = support.astype(np.float64) + matrix.sum(axis=0)
    f1 = np.divide(
        2 * matrix.diagonal().astype(np.float64), denominator, out=np.zeros(len(matrix)), where=denominator != 0
    )
    return {
        "count": count,
        "confusion_matrix": matrix.tolist(),
        "class_support": support.tolist(),
        "class_recall": [float(matrix[i, i] / n) if n else None for i, n in enumerate(support)],
        "accuracy": float(matrix.trace() / count),
        "macro_f1": float(f1.mean()),
        "f1_zero_division": 0,
    }


class ClassificationAccumulator:
    """Accumulate integer targets and finite logits with an explicit class axis.

    Targets have shape [...], logits [..., classes]. Unknown labels must be
    filtered by the caller. No implicit flattening of mismatched shapes, weights,
    balancing or domain class names. Ties select the lowest class index.
    """

    def __init__(self, classes):
        if type(classes) is not int or classes < 2:
            raise ValueError("Class count must be an integer >= 2")
        self._confusion = np.zeros((classes, classes), dtype=np.int64)
        self._loss_sum = 0.0

    @property
    def confusion(self):
        return self._confusion.copy()

    @property
    def loss_sum(self):
        return self._loss_sum

    def update(self, targets, logits):
        labels = np.asarray(targets)
        scores = np.asarray(logits)
        classes = len(self._confusion)
        if labels.dtype.kind not in "iu" or scores.dtype.kind not in "fiu":
            raise ValueError("Integer targets and real logits are required")
        if scores.shape != (*labels.shape, classes):
            raise ValueError("Logits must match target shape with one trailing class axis")
        if ((labels < 0) | (labels >= classes)).any():
            raise ValueError("Targets outside the declared class range")
        if scores.dtype.kind in "iu" and ((scores < -(2**53)).any() or (scores > 2**53).any()):
            raise ValueError("Integer logits must lie within the exact float64 integer range [-2**53, 2**53]")
        scores = scores.astype(np.float64).reshape(-1, classes)
        labels = labels.reshape(-1)
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite evaluation logits")
        if not len(labels):
            return
        with np.errstate(over="ignore", invalid="ignore"):
            shifted = scores - scores.max(axis=1, keepdims=True)
            loss = float((np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(labels)), labels]).sum())
        if not np.isfinite(loss) or not np.isfinite(self._loss_sum + loss):
            raise ValueError("Classification loss exceeds finite float64 range")
        if int(self._confusion.sum()) + len(labels) > np.iinfo(np.int64).max:
            raise ValueError("Classification counts exceed int64 range")
        delta = np.zeros_like(self._confusion)
        np.add.at(delta, (labels, np.argmax(scores, axis=-1)), 1)
        self._confusion += delta
        self._loss_sum += loss

    def result(self):
        report = summarize_confusion(self._confusion)
        return {**report, "loss_sum": self._loss_sum, "cross_entropy": self._loss_sum / report["count"]}

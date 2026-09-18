"""Metrics accepting predictions from any model implementation."""

import numpy as np


def classification_metrics(targets, predictions, num_classes):
    targets, predictions = np.asarray(targets).ravel(), np.asarray(predictions).ravel()
    if not len(targets) or targets.shape != predictions.shape:
        raise ValueError("Targets and predictions must be nonempty and match")
    if not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError("At least two classes are required")
    for values in (targets, predictions):
        if not np.issubdtype(values.dtype, np.integer) or (values < 0).any() or (values >= num_classes).any():
            raise ValueError("Labels must be integers within the class range")
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (targets, predictions), 1)
    tp = np.diag(cm)
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(num_classes, dtype=float), where=(support + predicted) > 0)
    recall = np.divide(tp, support, out=np.zeros(num_classes, dtype=float), where=support > 0)
    accuracy = float(tp.sum() / cm.sum())
    expected = float(np.sum(support.astype(float) * predicted) / cm.sum() ** 2)
    return {
        "accuracy": accuracy,
        "macro_f1": float(f1.mean()),
        "per_class_f1": f1.tolist(),
        "balanced_accuracy": float(recall[support > 0].mean()),
        "kappa": (accuracy - expected) / (1 - expected) if expected < 1 else None,
        "support": support.tolist(),
        "confusion_matrix": cm.tolist(),
        "aggregation": "pooled_valid_frames",
        "absent_class_f1": 0,
    }


def evaluate_records(predictor, records):
    """Evaluate a deployed bundle on labeled records without TensorFlow.

    Callers select the held-out records explicitly. Preparation and context
    selection use the same saved definition as inference; invalid or unlabeled
    contexts are excluded. Per-record results stay in the caller's local report.
    """
    from sleepkit.preprocessing import model_windows, prepare

    actual, predicted, per_record = [], [], []
    classes = len(predictor.manifest["class_names"])
    for record in records:
        prepared = predictor.normalizer.transform(prepare(record, predictor.preprocessing))
        x, y, _ = model_windows(prepared, predictor.manifest["context"])
        if not len(x):
            raise ValueError(f"No complete valid labeled windows: {record.dataset}/{record.recording}")
        labels = predictor.predict_features(x).argmax(axis=-1)
        actual.append(y.ravel())
        predicted.append(labels.ravel())
        per_record.append(
            {
                "dataset": record.dataset,
                "subject": record.subject,
                "recording": record.recording,
                "metrics": classification_metrics(y, labels, classes),
            }
        )
    if not actual:
        raise ValueError("Evaluation requires at least one record")
    return {
        "pooled": classification_metrics(np.concatenate(actual), np.concatenate(predicted), classes),
        "records": per_record,
    }

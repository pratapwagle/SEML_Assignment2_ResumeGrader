"""Model-quality metrics for multi-class resume role prediction."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import label_binarize


def compute_classification_metrics(
    y_true,
    y_pred,
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> dict[str, Any]:
    """Measure accuracy, weighted F1, multiclass Brier, and top-3 accuracy.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        probabilities: Class probability matrix aligned with ``classes``.
        classes: Ordered class label array from the fitted estimator.

    Returns:
        Dict of float metrics used by training gates and the metrics snapshot.
    """
    true_labels = np.asarray(y_true)
    predicted_labels = np.asarray(y_pred)
    encoded_true = label_binarize(true_labels, classes=classes)
    if encoded_true.shape[1] == 1:
        encoded_true = np.column_stack([1 - encoded_true, encoded_true])

    top_three_indices = np.argsort(probabilities, axis=1)[:, -3:]
    true_indices = np.array(
        [int(np.where(classes == label)[0][0]) for label in true_labels]
    )
    top_three_hits = [
        true_index in row for true_index, row in zip(true_indices, top_three_indices)
    ]

    return {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "weighted_f1": float(
            f1_score(true_labels, predicted_labels, average="weighted")
        ),
        "multiclass_brier": float(
            np.mean(np.sum((probabilities - encoded_true) ** 2, axis=1))
        ),
        "top_3_accuracy": float(np.mean(top_three_hits)),
    }

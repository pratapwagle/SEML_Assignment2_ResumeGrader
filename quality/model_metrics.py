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
    """Measure classification, ranking, and confidence calibration quality."""
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

"""Data-quality measurement and simple text-length drift detection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml.data import REQUIRED_COLUMNS


def compute_data_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    """Return schema, completeness, balance, and length statistics.

    Args:
        frame: Training or evaluation frame. Schema is measured here rather
            than assumed valid, so a missing-column frame still returns a
            report instead of raising.

    Returns:
        Dict suitable for metrics snapshots and the API metrics schema.
    """
    schema_valid = REQUIRED_COLUMNS.issubset(frame.columns)
    if not schema_valid:
        return {
            "schema_valid": False,
            "missing_value_rate": 1.0,
            "duplicate_rows": (
                int(frame.duplicated().sum()) if hasattr(frame, "duplicated") else 0
            ),
            "class_count": 0,
            "minority_class_fraction": 0.0,
            "text_length_mean": 0.0,
            "text_length_std": 0.0,
        }

    required = frame[list(REQUIRED_COLUMNS)]
    missing_or_empty = required.isna() | required.apply(
        lambda column: column.astype(str).str.strip().eq("")
    )
    lengths = frame["resume_text"].astype(str).str.len().to_numpy()
    class_counts = frame["job_role"].value_counts()
    return {
        "schema_valid": True,
        "missing_value_rate": float(missing_or_empty.to_numpy().mean()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "class_count": int(len(class_counts)),
        "minority_class_fraction": float(class_counts.min() / len(frame)),
        "text_length_mean": float(np.mean(lengths)),
        "text_length_std": float(np.std(lengths)),
    }


def detect_text_length_drift(
    reference_mean: float,
    reference_std: float,
    current_lengths: list[int],
) -> dict[str, Any]:
    """Flag a batch whose mean text length is more than three sigma away.

    Args:
        reference_mean: Baseline mean character length.
        reference_std: Baseline standard deviation (0 disables z-score).
        current_lengths: Character lengths for the current batch.

    Returns:
        Dict with z-score and ``drift_detected`` boolean.

    Raises:
        ValueError: If ``current_lengths`` is empty.
    """
    if not current_lengths:
        raise ValueError("current_lengths must not be empty")
    current_mean = float(np.mean(current_lengths))
    z_score = (
        0.0
        if reference_std == 0
        else float((current_mean - reference_mean) / reference_std)
    )
    return {
        "reference_mean": float(reference_mean),
        "current_mean": current_mean,
        "mean_length_z_score": z_score,
        "drift_detected": abs(z_score) > 3.0,
    }

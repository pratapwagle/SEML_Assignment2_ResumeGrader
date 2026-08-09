from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_data_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    """Return measurable schema, completeness, balance, and length statistics."""
    required = frame[["resume_text", "job_role"]]
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
    """Flag a current batch whose mean text length is over three sigma away."""
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

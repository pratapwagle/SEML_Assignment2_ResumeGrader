import pandas as pd
import pytest

from ml.data import build_training_data
from quality.data_metrics import compute_data_metrics, detect_text_length_drift


def test_data_metrics_measure_completeness_balance_and_length():
    metrics = compute_data_metrics(build_training_data())
    assert metrics["schema_valid"] is True
    assert metrics["missing_value_rate"] == 0.0
    assert metrics["duplicate_rows"] == 0
    assert metrics["class_count"] == 6
    assert metrics["minority_class_fraction"] > 0.0
    assert metrics["text_length_mean"] > 0.0


def test_data_metrics_report_invalid_schema_without_raising():
    metrics = compute_data_metrics(pd.DataFrame({"text": ["example"]}))
    assert metrics["schema_valid"] is False
    assert metrics["class_count"] == 0


def test_text_length_drift_detects_large_shift():
    report = detect_text_length_drift(
        reference_mean=100.0,
        reference_std=10.0,
        current_lengths=[150, 160, 170],
    )
    assert report["drift_detected"] is True
    assert report["mean_length_z_score"] > 3.0


def test_text_length_drift_rejects_empty_batch():
    with pytest.raises(ValueError, match="must not be empty"):
        detect_text_length_drift(100.0, 10.0, [])

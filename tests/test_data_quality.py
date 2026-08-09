import pandas as pd
import pytest

from ml.data import build_training_data, validate_training_data


def test_data_schema_and_missing_values():
    report = validate_training_data(build_training_data())
    assert report.schema_valid is True
    assert report.missing_values == 0
    assert report.duplicate_rows == 0
    assert report.class_count == 6


def test_invalid_schema_is_rejected():
    with pytest.raises(ValueError):
        validate_training_data(pd.DataFrame({"text": ["example"]}))


def test_missing_required_values_are_rejected():
    frame = build_training_data()
    frame.loc[0, "resume_text"] = None
    with pytest.raises(ValueError, match="missing"):
        validate_training_data(frame)

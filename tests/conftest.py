from pathlib import Path

import pytest

from ml.data import build_training_data
from ml.trainer import train_model


@pytest.fixture(scope="session", autouse=True)
def trained_model() -> Path:
    path = Path("models/resume_classifier.joblib")
    train_model(build_training_data(), path)
    return path

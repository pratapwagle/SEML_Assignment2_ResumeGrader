from pathlib import Path

import pytest

from ml.data import build_training_data
from ml.trainer import train_model


@pytest.fixture(scope="session")
def trained_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    train_model(build_training_data(), path)
    return path

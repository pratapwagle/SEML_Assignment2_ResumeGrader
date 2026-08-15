from pathlib import Path

import joblib
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from ml.data import build_training_data
from ml.features import FeatureEngineer
from ml.preprocessing import clean_text
from ml.trainer import (
    MIN_ACCURACY,
    MIN_WEIGHTED_F1,
    build_pipeline,
    train_model,
)


def test_training_creates_model_and_quality_metrics(tmp_path: Path):
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    metrics = train_model(build_training_data(), model_path, metrics_path)
    assert model_path.exists()
    assert metrics_path.exists()
    assert metrics["accuracy"] >= MIN_ACCURACY
    assert metrics["weighted_f1"] >= MIN_WEIGHTED_F1
    assert metrics["schema_valid"] is True
    assert metrics["missing_values"] == 0
    assert metrics["duplicate_rows"] == 0
    assert 0.0 <= metrics["multiclass_brier"] <= 2.0
    assert metrics["top_3_accuracy"] >= metrics["accuracy"]
    assert "length_drift" in metrics
    assert "drift_detected" in metrics["length_drift"]


def test_model_can_overfit_small_batch():
    frame = build_training_data().iloc[:8]
    model = build_pipeline().fit(
        frame["resume_text"].map(clean_text), frame["job_role"]
    )
    assert model.score(frame["resume_text"].map(clean_text), frame["job_role"]) >= 0.95


def test_training_log_loss_decreases():
    frame = build_training_data()
    texts = frame["resume_text"].map(clean_text)
    features = FeatureEngineer(ngram_range=(1, 2), min_df=1).fit_transform(texts)
    classifier = LogisticRegression(
        max_iter=1,
        warm_start=True,
        solver="lbfgs",
        random_state=42,
    )
    losses = []
    with pytest.warns(ConvergenceWarning):
        for step in range(1, 16):
            classifier.max_iter = step
            classifier.fit(features, frame["job_role"])
            losses.append(
                log_loss(frame["job_role"], classifier.predict_proba(features))
            )
    assert losses[-1] < losses[0]


def test_quality_gate_rejects_unusable_model(tmp_path: Path):
    frame = build_training_data().copy()
    frame["job_role"] = frame["job_role"].sample(frac=1, random_state=0).to_numpy()
    with pytest.raises(ValueError, match="quality gate failed"):
        train_model(frame, tmp_path / "bad.joblib")


def test_training_is_deterministic():
    frame = build_training_data()
    texts = frame["resume_text"].map(clean_text)
    first = build_pipeline().fit(texts, frame["job_role"])
    second = build_pipeline().fit(texts, frame["job_role"])
    probes = [
        "python machine learning nlp statistics",
        "docker kubernetes terraform aws",
    ]
    assert first.predict(probes).tolist() == second.predict(probes).tolist()


def test_model_persistence_roundtrip(tmp_path: Path):
    frame = build_training_data()
    model = build_pipeline().fit(
        frame["resume_text"].map(clean_text),
        frame["job_role"],
    )
    path = tmp_path / "roundtrip.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    probe = ["selenium playwright testing regression automation"]
    assert restored.predict(probe).tolist() == model.predict(probe).tolist()

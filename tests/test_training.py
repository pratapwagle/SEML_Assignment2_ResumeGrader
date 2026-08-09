from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

from ml.data import build_training_data
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


def test_model_can_overfit_small_batch():
    frame = build_training_data().iloc[:8]
    model = build_pipeline().fit(
        frame["resume_text"].map(clean_text), frame["job_role"]
    )
    assert model.score(frame["resume_text"].map(clean_text), frame["job_role"]) >= 0.95


def test_training_log_loss_decreases():
    frame = build_training_data()
    texts = frame["resume_text"].map(clean_text)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    features = vectorizer.fit_transform(texts)
    labels = LabelEncoder().fit_transform(frame["job_role"])
    classes = np.unique(labels)
    classifier = SGDClassifier(
        loss="log_loss",
        random_state=42,
        learning_rate="constant",
        eta0=0.05,
    )
    losses = []
    for _ in range(10):
        classifier.partial_fit(features, labels, classes=classes)
        losses.append(
            log_loss(
                labels,
                classifier.predict_proba(features),
                labels=classes,
            )
        )
    assert losses[-1] < losses[0]


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

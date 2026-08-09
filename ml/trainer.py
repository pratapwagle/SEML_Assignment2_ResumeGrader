"""Offline training, release gates, and artifact persistence."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml.data import validate_training_data
from ml.preprocessing import clean_text
from quality.data_metrics import compute_data_metrics
from quality.model_metrics import compute_classification_metrics

logger = logging.getLogger(__name__)
MIN_ACCURACY = 0.80
MIN_WEIGHTED_F1 = 0.80


def build_pipeline() -> Pipeline:
    """Construct the TF-IDF + logistic regression inference pipeline.

    Returns:
        Unfitted scikit-learn ``Pipeline`` with a fixed random seed on the
        classifier for reproducible demos.
    """
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def train_model(
    frame: pd.DataFrame,
    model_path: Path,
    metrics_path: Path | None = None,
) -> dict[str, Any]:
    """Train, gate, and persist the resume role classifier.

    Validates the frame, evaluates on a stratified holdout, enforces minimum
    accuracy and weighted F1, then refits on all rows and writes the joblib
    artifact. Optionally writes a metrics JSON snapshot for the API/UI.

    Args:
        frame: Training data with ``resume_text`` and ``job_role`` columns.
        model_path: Destination path for the joblib pipeline.
        metrics_path: Optional path for ``metrics_snapshot.json``.

    Returns:
        Dict of evaluation metrics and embedded data-quality summary.

    Raises:
        TypeError, ValueError: From data validation or failed quality gates.
        Exception: From model fit failures.
    """
    data_quality = validate_training_data(frame)
    cleaned = frame["resume_text"].map(clean_text)
    labels = frame["job_role"]
    x_train, x_test, y_train, y_test = train_test_split(
        cleaned, labels, test_size=0.5, random_state=42, stratify=labels
    )
    pipeline = build_pipeline()
    logger.info("Starting model training with %d training rows", len(x_train))
    try:
        pipeline.fit(x_train, y_train)
    except Exception:
        logger.exception("Model training failed")
        raise
    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)
    model_metrics = compute_classification_metrics(
        y_test,
        predictions,
        probabilities,
        pipeline.classes_,
    )
    data_metrics = compute_data_metrics(frame)
    metrics = {
        **model_metrics,
        "validation_rows": int(len(x_test)),
        "schema_valid": data_quality.schema_valid,
        "missing_values": data_quality.missing_values,
        "duplicate_rows": data_quality.duplicate_rows,
        "data_quality": data_metrics,
    }
    if metrics["accuracy"] < MIN_ACCURACY or metrics["weighted_f1"] < MIN_WEIGHTED_F1:
        logger.error("Model quality gate failed: %s", metrics)
        raise ValueError(
            "Model quality gate failed: "
            f"accuracy must be >= {MIN_ACCURACY:.2f} and weighted_f1 "
            f"must be >= {MIN_WEIGHTED_F1:.2f}"
        )

    final_model = build_pipeline().fit(cleaned, labels)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_path)
    if metrics_path is not None:
        snapshot = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "model_path": str(model_path.name),
            "model_quality": {
                key: metrics[key]
                for key in [
                    "accuracy",
                    "weighted_f1",
                    "multiclass_brier",
                    "top_3_accuracy",
                    "validation_rows",
                ]
            },
            "data_quality": data_metrics,
            "quality_gates": {
                "minimum_accuracy": MIN_ACCURACY,
                "minimum_weighted_f1": MIN_WEIGHTED_F1,
                "passed": True,
            },
        }
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        logger.info("Saved metrics snapshot to %s", metrics_path)
    logger.info("Saved trained model to %s; metrics=%s", model_path, metrics)
    return metrics

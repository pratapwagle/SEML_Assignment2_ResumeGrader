from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib

from ml.preprocessing import clean_text

logger = logging.getLogger(__name__)


class ModelPredictor:
    def __init__(self, model_path: Path):
        if not model_path.exists():
            logger.error("Model artifact not found: %s", model_path)
            raise FileNotFoundError(f"Model artifact not found: {model_path}")
        self.pipeline = joblib.load(model_path)
        logger.info("Loaded model artifact from %s", model_path)

    def predict(self, resume_text: str) -> dict[str, Any]:
        cleaned = clean_text(resume_text)
        try:
            probabilities = self.pipeline.predict_proba([cleaned])[0]
        except Exception:
            logger.exception("Model inference failed")
            raise
        ranking = sorted(
            zip(self.pipeline.classes_, probabilities),
            key=lambda item: item[1],
            reverse=True,
        )
        role = str(ranking[0][0])
        result = {
            "predicted_role": role,
            "confidence": float(max(probabilities)),
            "role_ranking": [
                {"role": str(name), "probability": float(score)}
                for name, score in ranking
            ],
        }
        logger.info(
            "Inference completed; predicted_role=%s confidence=%.3f",
            role,
            result["confidence"],
        )
        return result

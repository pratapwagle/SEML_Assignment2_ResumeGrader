"""Role scoring with keyword-based explanation and human-oversight note."""

from __future__ import annotations

from typing import Any

from ml.predictor import ModelPredictor
from ml.preprocessing import clean_text

KEYWORDS = {
    "Data Scientist": ["machine learning", "python", "statistics", "nlp"],
    "Data Engineer": ["etl", "spark", "kafka", "warehouse"],
    "Backend Developer": ["java", "spring", "api", "microservices"],
    "QA Engineer": ["testing", "selenium", "playwright", "automation"],
    "Frontend Developer": ["react", "angular", "frontend", "typescript"],
    "DevOps Engineer": ["docker", "kubernetes", "terraform", "aws"],
}


class ResumeScoringService:
    """Combine model inference with lightweight keyword explanations.

    Args:
        predictor: Loaded ``ModelPredictor`` instance.
    """

    def __init__(self, predictor: ModelPredictor):
        self.predictor = predictor

    def score(self, resume_text: str) -> dict[str, Any]:
        """Predict role and attach explanation plus advisory decision note.

        Args:
            resume_text: Raw resume text.

        Returns:
            Prediction dict extended with ``explanation`` and ``decision_note``.
            Explanations are heuristic keyword matches for recruiter review,
            not full model interpretability.
        """
        result = self.predictor.predict(resume_text)
        normalized = clean_text(resume_text)
        matches = [
            keyword
            for keyword in KEYWORDS.get(result["predicted_role"], [])
            if keyword in normalized
        ]
        result["explanation"] = (
            "Matched evidence: " + ", ".join(matches)
            if matches
            else ("Limited direct keyword evidence; " "recruiter review is required.")
        )
        result["decision_note"] = (
            "Advisory output only; no automatic rejection is performed."
        )
        return result

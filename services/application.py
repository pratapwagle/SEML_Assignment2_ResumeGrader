from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from services.audit import AuditRepository
from services.scoring import ResumeScoringService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResumeSubmission:
    candidate_name: str
    resume_text: str
    source: str


class ResumeScreeningApplication:
    """Coordinate validation, scoring, and privacy-aware audit persistence."""

    def __init__(
        self,
        scoring_service: ResumeScoringService,
        repository: AuditRepository,
    ):
        self.scoring_service = scoring_service
        self.repository = repository

    def submit(self, submission: ResumeSubmission) -> dict[str, Any]:
        candidate_name = submission.candidate_name.strip()
        if not candidate_name:
            logger.warning("Prediction rejected because candidate name is blank")
            raise ValueError("candidate_name must not be blank")

        result = self.scoring_service.score(submission.resume_text)
        try:
            self.repository.save(
                candidate_name,
                submission.source,
                submission.resume_text,
                result,
            )
        except Exception:
            logger.exception("Prediction audit persistence failed")
            raise
        logger.info(
            "Submission completed; source=%s predicted_role=%s",
            submission.source,
            result["predicted_role"],
        )
        return result

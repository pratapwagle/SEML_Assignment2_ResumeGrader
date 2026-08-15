"""Privacy-preserving audit history for screening decisions."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)
AUDIT_COLUMNS = [
    "timestamp_utc",
    "candidate_name",
    "source",
    "resume_characters",
    "predicted_role",
    "confidence",
    "explanation",
]


class AuditRepository:
    """Persist prediction metadata without storing sensitive resume text.

    Args:
        storage_path: CSV path for append-only audit rows.
    """

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path

    def save(
        self,
        candidate_name: str,
        source: str,
        resume_text: str,
        result: dict[str, Any],
    ) -> None:
        """Append one audit row using resume length instead of raw body.

        Args:
            candidate_name: Candidate display name.
            source: Channel label (``api``, ``upload``, etc.).
            resume_text: Used only to compute ``resume_characters``.
            result: Scoring output with role, confidence, and explanation.
        """
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "candidate_name": candidate_name,
            "source": source,
            "resume_characters": len(resume_text),
            "predicted_role": result["predicted_role"],
            "confidence": result["confidence"],
            "explanation": result["explanation"],
        }
        history = self.list_results()
        new_row = pd.DataFrame([record], columns=AUDIT_COLUMNS)
        updated = (
            new_row
            if history.empty
            else pd.concat([history, new_row], ignore_index=True)
        )
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        updated.to_csv(self.storage_path, index=False)
        logger.info(
            "Saved audit metadata; predicted_role=%s source=%s",
            result["predicted_role"],
            source,
        )

    def list_results(self) -> pd.DataFrame:
        """Return the full audit history, or an empty frame with known columns.

        Returns:
            DataFrame of audit rows ordered as stored on disk.
        """
        if self.storage_path.exists():
            return pd.read_csv(self.storage_path)
        return pd.DataFrame(columns=AUDIT_COLUMNS)

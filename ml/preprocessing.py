from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)
MAX_RESUME_CHARS = 20_000


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        logger.error("Resume input must be a string; received %s", type(text).__name__)
        raise TypeError("resume_text must be a string")
    if not text.strip():
        logger.warning("Empty resume text rejected")
        raise ValueError("resume_text must not be empty")
    if len(text) > MAX_RESUME_CHARS:
        logger.warning("Resume exceeds maximum supported length: %d", len(text))
        raise ValueError(f"resume_text exceeds {MAX_RESUME_CHARS} characters")
    normalized = text.lower()
    normalized = re.sub(r"[^a-z0-9+#. ]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        logger.warning("Resume contains no usable text after normalization")
        raise ValueError("resume_text must contain letters or numbers")
    logger.info("Preprocessed resume: %d -> %d characters", len(text), len(normalized))
    return normalized

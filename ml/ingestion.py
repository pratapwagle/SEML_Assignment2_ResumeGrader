"""Resume file ingestion for TXT, PDF, and DOCX uploads.

Extraction is isolated from scoring so the Streamlit UI can reject bad files
before calling the shared application service.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
SUPPORTED_SUFFIXES = {".txt", ".pdf", ".docx"}
MAX_UPLOAD_BYTES = 5 * 1024 * 1024


def extract_resume_text(filename: str, payload: bytes) -> str:
    """Extract plain text from an uploaded resume file.

    Args:
        filename: Original upload name; suffix selects the parser.
        payload: Raw file bytes (not base64).

    Returns:
        Non-empty extracted text with surrounding whitespace stripped.

    Raises:
        ValueError: Unsupported type, empty/oversized payload, or failed
            extraction with no usable text.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        logger.warning("Unsupported resume file type: %s", suffix or "<none>")
        raise ValueError("Unsupported file type; use TXT, PDF, or DOCX")
    if not payload:
        logger.warning("Empty resume upload rejected")
        raise ValueError("Uploaded resume is empty")
    if len(payload) > MAX_UPLOAD_BYTES:
        logger.warning("Resume upload exceeds %d bytes", MAX_UPLOAD_BYTES)
        raise ValueError("Uploaded resume exceeds the 5 MB limit")

    try:
        if suffix == ".txt":
            text = payload.decode("utf-8", errors="ignore")
        elif suffix == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(payload))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            from docx import Document

            document = Document(io.BytesIO(payload))
            text = "\n".join(
                paragraph.text.strip()
                for paragraph in document.paragraphs
                if paragraph.text.strip()
            )
    except Exception:
        logger.exception("Resume extraction failed for file type %s", suffix)
        raise ValueError("Resume text could not be extracted") from None

    text = text.strip()
    if not text:
        logger.warning("No text was extracted from uploaded %s file", suffix)
        raise ValueError("Uploaded resume contains no extractable text")
    logger.info(
        "Extracted %d characters from %s resume without logging content",
        len(text),
        suffix,
    )
    return text

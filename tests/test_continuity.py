from pathlib import Path

import pytest

from ml.ingestion import extract_resume_text
from services.audit import AuditRepository


def test_txt_resume_ingestion_preserves_assignment_one_capability():
    text = extract_resume_text(
        "candidate.txt",
        b"Python machine learning and statistics experience.",
    )
    assert text.startswith("Python machine learning")


def test_unsupported_resume_file_is_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        extract_resume_text("candidate.exe", b"not a resume")


def test_audit_repository_stores_metadata_without_raw_resume(tmp_path: Path):
    path = tmp_path / "audit.csv"
    repository = AuditRepository(path)
    raw_text = "sensitive candidate resume content"
    repository.save(
        candidate_name="Demo Candidate",
        source="text input",
        resume_text=raw_text,
        result={
            "predicted_role": "Data Scientist",
            "confidence": 0.9,
            "explanation": "Matched evidence: python",
        },
    )

    history = repository.list_results()
    assert len(history) == 1
    assert history.iloc[0]["resume_characters"] == len(raw_text)
    assert raw_text not in path.read_text(encoding="utf-8")

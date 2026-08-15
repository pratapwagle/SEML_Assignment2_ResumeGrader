import io
from pathlib import Path

import pytest

from ml.ingestion import extract_resume_text
from services.audit import AuditRepository


def _pdf_with_text(text: str) -> bytes:
    """Minimal one-page PDF with extractable Helvetica text (no reportlab)."""
    content = f"BT /F1 12 Tf 20 80 Td ({text}) Tj ET\n".encode("latin-1")
    objects = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        (
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        ),
        b"4 0 obj\n<< /Length %d >>\nstream\n" % len(content)
        + content
        + b"endstream\nendobj\n",
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
    ]
    body = b"%PDF-1.4\n"
    for obj in objects:
        body += obj
    xref_pos = len(body)
    # Offsets are computed after the header so pypdf can walk the xref table.
    cursor = len(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(cursor)
        cursor += len(obj)
    xref = b"xref\n0 6\n0000000000 65535 f \n"
    xref += b"".join(
        f"{offset:010d} 00000 n \n".encode("ascii") for offset in offsets[1:]
    )
    trailer = f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n"
    return body + xref + trailer.encode("latin-1")


def test_txt_resume_ingestion_preserves_assignment_one_capability():
    text = extract_resume_text(
        "candidate.txt",
        b"Python machine learning and statistics experience.",
    )
    assert text.startswith("Python machine learning")


def test_pdf_resume_ingestion_extracts_text():
    pytest.importorskip("pypdf")
    text = extract_resume_text(
        "candidate.pdf",
        _pdf_with_text("Python machine learning statistics"),
    )
    assert "Python machine learning" in text


def test_docx_resume_ingestion_extracts_text():
    Document = pytest.importorskip("docx").Document
    buffer = io.BytesIO()
    document = Document()
    document.add_paragraph("Docker kubernetes terraform aws cloud")
    document.save(buffer)
    text = extract_resume_text("candidate.docx", buffer.getvalue())
    assert "Docker kubernetes" in text


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

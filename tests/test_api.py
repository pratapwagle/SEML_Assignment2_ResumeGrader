import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.api as api_module
from app.api import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolate_api_state(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(api_module, "AUDIT_PATH", tmp_path / "audit.csv")
    monkeypatch.setattr(api_module, "_application", None)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["model_loaded"] is True
    assert payload["version"] == "2.1.0"


def test_health_endpoint_reports_corrupt_model(
    monkeypatch,
    tmp_path: Path,
):
    corrupt = tmp_path / "corrupt.joblib"
    corrupt.write_bytes(b"not-a-joblib-model")
    monkeypatch.setattr(api_module, "MODEL_PATH", corrupt)
    monkeypatch.setattr(api_module, "_application", None)
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["detail"] == "Model artifact is unavailable"


def test_health_endpoint_reports_missing_model(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(api_module, "MODEL_PATH", tmp_path / "missing.joblib")
    monkeypatch.setattr(api_module, "_application", None)
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["detail"] == "Model artifact is unavailable"


def test_prediction_endpoint_contract():
    response = client.post(
        "/v1/predictions",
        json={
            "candidate_name": "Demo Candidate",
            "resume_text": (
                "Experienced QA engineer with selenium playwright automation "
                "testing and regression skills."
            ),
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_role"] == "QA Engineer"
    assert 0.0 <= payload["confidence"] <= 1.0
    assert len(payload["role_ranking"]) == 6


def test_predict_alias_uses_shared_application_layer():
    response = client.post(
        "/predict",
        json={
            "candidate_name": "Demo Candidate",
            "resume_text": (
                "React TypeScript CSS frontend accessibility and component design"
            ),
        },
    )
    assert response.status_code == 200
    assert response.json()["predicted_role"] == "Frontend Developer"


def test_prediction_endpoint_persists_privacy_aware_audit():
    resume_text = "Docker Kubernetes Terraform AWS cloud reliability automation"
    response = client.post(
        "/v1/predictions",
        json={"candidate_name": "Audit Candidate", "resume_text": resume_text},
    )
    assert response.status_code == 200
    audit_text = api_module.AUDIT_PATH.read_text(encoding="utf-8")
    assert "Audit Candidate" in audit_text
    assert resume_text not in audit_text


def test_prediction_endpoint_rejects_short_input():
    response = client.post(
        "/v1/predictions",
        json={"candidate_name": "A", "resume_text": "short"},
    )
    assert response.status_code == 422


def test_prediction_endpoint_rejects_semantically_empty_input():
    response = client.post(
        "/v1/predictions",
        json={"candidate_name": "A", "resume_text": "!" * 20},
    )
    assert response.status_code == 422


def test_prediction_endpoint_rejects_blank_candidate_name():
    response = client.post(
        "/v1/predictions",
        json={
            "candidate_name": "   ",
            "resume_text": "python machine learning statistics and nlp",
        },
    )
    assert response.status_code == 422


def test_metrics_endpoint_returns_training_snapshot(monkeypatch, tmp_path: Path):
    metrics_path = tmp_path / "metrics_snapshot.json"
    metrics_path.write_text(
        json.dumps(
            {
                "generated_at_utc": "2026-08-01T00:00:00+00:00",
                "model_path": "resume_classifier.joblib",
                "model_quality": {
                    "accuracy": 1.0,
                    "weighted_f1": 1.0,
                    "multiclass_brier": 0.7,
                    "top_3_accuracy": 1.0,
                    "validation_rows": 12,
                },
                "data_quality": {
                    "schema_valid": True,
                    "missing_value_rate": 0.0,
                    "duplicate_rows": 0,
                    "class_count": 6,
                    "minority_class_fraction": 1 / 6,
                    "text_length_mean": 77.0,
                    "text_length_std": 8.1,
                },
                "quality_gates": {
                    "minimum_accuracy": 0.8,
                    "minimum_weighted_f1": 0.8,
                    "passed": True,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(api_module, "METRICS_PATH", metrics_path)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.json()["model_quality"]["accuracy"] == 1.0


def test_metrics_endpoint_returns_404_when_snapshot_missing(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(api_module, "METRICS_PATH", tmp_path / "missing.json")
    response = client.get("/metrics")
    assert response.status_code == 404


def test_api_lifespan_loads_model_once():
    with TestClient(app) as lifespan_client:
        response = lifespan_client.get("/health")
    assert response.status_code == 200
    assert api_module._application is not None

"""FastAPI transport layer for resume screening inference.

Routes are thin: they validate HTTP contracts, map errors to status codes, and
delegate to ``ResumeScreeningApplication`` so behaviour matches Streamlit.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status

from app.schemas import (
    HealthResponse,
    MetricsSnapshot,
    PredictionResponse,
    ResumeRequest,
)
from config import AUDIT_PATH, METRICS_PATH, MODEL_PATH, MODEL_VERSION
from logging_config import configure_logging
from ml.predictor import ModelPredictor
from services.application import ResumeScreeningApplication, ResumeSubmission
from services.audit import AuditRepository
from services.scoring import ResumeScoringService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Configure logging and eagerly load the model when the artifact exists."""
    configure_logging()
    if MODEL_PATH.exists():
        get_application()
        logger.info("API startup loaded model version %s", MODEL_VERSION)
    else:
        logger.error("API startup could not find the model artifact")
    yield


app = FastAPI(
    title="Group 179 Resume Screening API",
    version=MODEL_VERSION,
    description=(
        "Production-style inference API for Assignment II. "
        "Authentication and authorization are not implemented in this coursework "
        "demo and would be required before production use."
    ),
    lifespan=lifespan,
)

_application: ResumeScreeningApplication | None = None


def get_application() -> ResumeScreeningApplication:
    """Return a process-wide application instance (lazy singleton).

    Returns:
        Wired ``ResumeScreeningApplication`` with predictor, scoring, and audit.
    """
    global _application
    if _application is None:
        scoring_service = ResumeScoringService(ModelPredictor(MODEL_PATH))
        _application = ResumeScreeningApplication(
            scoring_service,
            AuditRepository(AUDIT_PATH),
        )
    return _application


@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    responses={503: {"description": "Model artifact is unavailable"}},
)
def health() -> HealthResponse:
    """Readiness probe: model artifact present and API version.

    Returns:
        Health payload when the joblib file exists.

    Raises:
        HTTPException: 503 if the model artifact is missing.
    """
    if not MODEL_PATH.exists():
        logger.error("Readiness check failed; model artifact is missing")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact is unavailable",
        )
    try:
        get_application()
    except Exception:
        logger.exception("Readiness check failed; model artifact could not be loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact is unavailable",
        ) from None
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        version=MODEL_VERSION,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
)
@app.post(
    "/v1/predictions",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
)
def predict(
    request: ResumeRequest,
) -> PredictionResponse:
    """Score a resume and return ranked roles with explanation.

    ``POST /predict`` and ``POST /v1/predictions`` share this handler.

    Args:
        request: Validated candidate name and resume text.

    Returns:
        Typed prediction response for the OpenAPI contract.

    Raises:
        HTTPException: 422 validation/domain errors, 503 missing model, 500
            unexpected failures.
    """
    try:
        result = get_application().submit(
            ResumeSubmission(
                candidate_name=request.candidate_name,
                resume_text=request.resume_text,
                source="api",
            )
        )
        return PredictionResponse(candidate_name=request.candidate_name, **result)
    except ValueError as exc:
        logger.warning("Invalid prediction request: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        logger.error("Prediction service unavailable: %s", exc)
        raise HTTPException(
            status_code=503, detail="Model artifact is unavailable"
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected prediction API error")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc


@app.get("/metrics", response_model=MetricsSnapshot)
def metrics() -> MetricsSnapshot:
    """Return the last training metrics snapshot from disk.

    Returns:
        Metrics snapshot used for release evidence and UI summaries.

    Raises:
        HTTPException: 404 if the snapshot file is missing.
    """
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="Metrics snapshot is unavailable")
    return MetricsSnapshot.model_validate_json(METRICS_PATH.read_text(encoding="utf-8"))

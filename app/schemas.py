"""Pydantic request and response contracts for the REST API."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ResumeRequest(BaseModel):
    """Inbound prediction payload."""

    model_config = ConfigDict(str_strip_whitespace=True)

    candidate_name: str = Field(default="Candidate", min_length=1, max_length=100)
    resume_text: str = Field(min_length=20, max_length=20_000)

    @field_validator("candidate_name")
    @classmethod
    def validate_candidate_name(cls, value: str) -> str:
        """Reject names that are empty after whitespace stripping."""
        if not value:
            raise ValueError("candidate_name must not be blank")
        return value


class RoleScore(BaseModel):
    """Single class probability in a ranking list."""

    role: str
    probability: float = Field(ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    """Successful inference response returned to API clients."""

    candidate_name: str
    predicted_role: str
    confidence: float = Field(ge=0.0, le=1.0)
    role_ranking: list[RoleScore]
    explanation: str
    decision_note: str


class HealthResponse(BaseModel):
    """API readiness payload."""

    status: str
    model_loaded: bool
    version: str


class ModelQualityMetrics(BaseModel):
    """Holdout classification metrics persisted after training."""

    accuracy: float = Field(ge=0.0, le=1.0)
    weighted_f1: float = Field(ge=0.0, le=1.0)
    multiclass_brier: float = Field(ge=0.0)
    top_3_accuracy: float = Field(ge=0.0, le=1.0)
    validation_rows: int = Field(gt=0)


class DataQualityMetrics(BaseModel):
    """Training-data quality summary embedded in the metrics snapshot."""

    schema_valid: bool
    missing_value_rate: float = Field(ge=0.0, le=1.0)
    duplicate_rows: int = Field(ge=0)
    class_count: int = Field(gt=1)
    minority_class_fraction: float = Field(gt=0.0, le=1.0)
    text_length_mean: float = Field(ge=0.0)
    text_length_std: float = Field(ge=0.0)
    length_drift_detected: bool | None = None
    mean_length_z_score: float | None = None


class QualityGates(BaseModel):
    """Release gate thresholds and pass flag."""

    minimum_accuracy: float = Field(ge=0.0, le=1.0)
    minimum_weighted_f1: float = Field(ge=0.0, le=1.0)
    passed: bool


class MetricsSnapshot(BaseModel):
    """On-disk training snapshot exposed by ``GET /metrics``."""

    generated_at_utc: str
    model_path: str
    model_quality: ModelQualityMetrics
    data_quality: DataQualityMetrics
    quality_gates: QualityGates

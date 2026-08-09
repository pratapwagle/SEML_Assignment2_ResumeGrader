from pydantic import BaseModel, ConfigDict, Field, field_validator


class ResumeRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    candidate_name: str = Field(default="Candidate", min_length=1, max_length=100)
    resume_text: str = Field(min_length=20, max_length=20_000)

    @field_validator("candidate_name")
    @classmethod
    def validate_candidate_name(cls, value: str) -> str:
        if not value:
            raise ValueError("candidate_name must not be blank")
        return value


class RoleScore(BaseModel):
    role: str
    probability: float = Field(ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    candidate_name: str
    predicted_role: str
    confidence: float = Field(ge=0.0, le=1.0)
    role_ranking: list[RoleScore]
    explanation: str
    decision_note: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class ModelQualityMetrics(BaseModel):
    accuracy: float = Field(ge=0.0, le=1.0)
    weighted_f1: float = Field(ge=0.0, le=1.0)
    multiclass_brier: float = Field(ge=0.0)
    top_3_accuracy: float = Field(ge=0.0, le=1.0)
    validation_rows: int = Field(gt=0)


class DataQualityMetrics(BaseModel):
    schema_valid: bool
    missing_value_rate: float = Field(ge=0.0, le=1.0)
    duplicate_rows: int = Field(ge=0)
    class_count: int = Field(gt=1)
    minority_class_fraction: float = Field(gt=0.0, le=1.0)
    text_length_mean: float = Field(ge=0.0)
    text_length_std: float = Field(ge=0.0)


class QualityGates(BaseModel):
    minimum_accuracy: float = Field(ge=0.0, le=1.0)
    minimum_weighted_f1: float = Field(ge=0.0, le=1.0)
    passed: bool


class MetricsSnapshot(BaseModel):
    generated_at_utc: str
    model_path: str
    model_quality: ModelQualityMetrics
    data_quality: DataQualityMetrics
    quality_gates: QualityGates

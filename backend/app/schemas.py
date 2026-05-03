from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class PatientData(BaseModel):
    age: int = Field(..., ge=0, le=130)
    gender: str = Field(..., min_length=1)
    hypertension: int | bool | str
    heart_disease: int | bool | str
    ever_married: str = Field(..., min_length=1)
    work_type: str = Field(..., min_length=1)
    residence_type: str = Field(..., min_length=1)
    avg_glucose_level: float = Field(..., ge=0)
    bmi: float | None = Field(default=None, ge=0)
    smoking_status: str = Field(..., min_length=1)

    @field_validator("hypertension", "heart_disease", mode="before")
    @classmethod
    def parse_binary(cls, value: int | bool | str) -> int:
        if isinstance(value, bool):
            return int(value)

        if isinstance(value, int):
            if value in (0, 1):
                return value
            raise ValueError("Must be 0/1 or yes/no")

        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "yes", "y", "true"}:
                return 1
            if normalized in {"0", "no", "n", "false"}:
                return 0

        raise ValueError("Must be 0/1 or yes/no")

    @field_validator(
        "gender",
        "ever_married",
        "work_type",
        "residence_type",
        "smoking_status",
        mode="before",
    )
    @classmethod
    def strip_text(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("Must be a string")
        normalized = value.strip()
        if not normalized:
            raise ValueError("Cannot be empty")
        return normalized


class KeyFactor(BaseModel):
    feature: str
    contribution: float
    direction: str


class PredictionResponse(BaseModel):
    stroke_probability: float
    risk_percentage: float
    risk_label: str
    key_contributing_factors: list[KeyFactor]


class FeatureImportanceItem(BaseModel):
    feature: str
    feature_label: str
    importance: float


class FeatureImportanceResponse(BaseModel):
    feature_importance: list[FeatureImportanceItem]
    model_metrics: dict[str, float]


class ReportExtractedFields(BaseModel):
    age: int | None = None
    gender: str | None = None
    hypertension: str | None = None
    heart_disease: str | None = None
    avg_glucose_level: float | None = None
    bmi: float | None = None
    smoking_status: str | None = None


class ReportAnalysisResponse(BaseModel):
    extracted_fields: ReportExtractedFields
    prediction: PredictionResponse | None = None
    missing_fields: list[str]


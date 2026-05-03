"""
main.py — NeuroScan AI · Brain Stroke Detection
Single-entry-point FastAPI app for Render deployment.
Run locally:  uvicorn main:app --reload --port 8000
Deploy:       uvicorn main:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import base64
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Environment & Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR.parent / ".env")

MODEL_PATH = BASE_DIR / "models" / "stroke_best_model.pkl"
STATIC_DIR = BASE_DIR / "app" / "static"

TARGET_COLUMN = "stroke"
DEFAULT_RISK_THRESHOLD = 0.3

DISPLAY_NAME_MAP: dict[str, str] = {
    "gender": "Gender",
    "age": "Age",
    "hypertension": "Hypertension",
    "heart_disease": "Heart Disease",
    "ever_married": "Ever Married",
    "work_type": "Work Type",
    "residence_type": "Residence Type",
    "avg_glucose_level": "Avg Glucose Level",
    "bmi": "BMI",
    "smoking_status": "Smoking Status",
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model Service
# ---------------------------------------------------------------------------
class StrokeModelService:
    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        risk_threshold: float = DEFAULT_RISK_THRESHOLD,
    ) -> None:
        self.model_path = model_path
        self.risk_threshold = risk_threshold
        self.loaded = False
        self.model = None
        self.model_name: str = "Unknown"
        self.label_encoders: dict[str, list[str]] = {}
        self.feature_columns: list[str] = []
        self.categorical_columns: list[str] = []
        self.bmi_median: float = 28.1
        self.feature_importance: dict[str, float] = {}
        self.metrics: dict[str, float] = {}

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        artifact = joblib.load(self.model_path)
        self.model = artifact["model"]
        self.model_name = artifact.get("model_name", "Unknown")
        self.label_encoders = artifact["label_encoders"]
        self.feature_columns = artifact["feature_columns"]
        self.categorical_columns = artifact["categorical_columns"]
        self.bmi_median = float(artifact["bmi_median"])
        self.feature_importance = artifact.get("feature_importance", {})
        self.metrics = artifact.get("metrics", {})
        self.loaded = True
        logger.info("Loaded model: %s", self.model_name)

    # ---- preprocessing ----
    def _encode_categorical(self, column: str, value: str) -> int:
        if column not in self.label_encoders:
            raise ValueError(f"Unsupported categorical field: {column}")
        classes = self.label_encoders[column]
        norm = {item.strip().lower(): idx for idx, item in enumerate(classes)}
        key = value.strip().lower()
        if key not in norm:
            raise ValueError(
                f"Invalid value '{value}' for {column}. Allowed: {', '.join(classes)}"
            )
        return norm[key]

    def preprocess_input(self, payload: dict[str, Any]) -> pd.DataFrame:
        if not self.loaded:
            raise RuntimeError("Model not loaded")
        processed: dict[str, Any] = {
            "age": float(payload["age"]),
            "hypertension": int(payload["hypertension"]),
            "heart_disease": int(payload["heart_disease"]),
            "avg_glucose_level": float(payload["avg_glucose_level"]),
        }
        bmi_val = payload.get("bmi")
        processed["bmi"] = self.bmi_median if bmi_val is None else float(bmi_val)
        for col in self.categorical_columns:
            processed[col] = self._encode_categorical(col, str(payload[col]))
        row = pd.DataFrame([processed])
        return row[self.feature_columns]

    # ---- contributing factors ----
    def _contributing_factors(
        self, row: pd.DataFrame, probability: float
    ) -> list[dict[str, Any]]:
        """
        Compute per-feature contributions using feature importance * deviation
        from dataset median (encoded). Returns top-5 sorted by absolute contribution.
        """
        if not self.feature_importance:
            return []

        # Typical median values for the encoded dataset
        MEDIANS: dict[str, float] = {
            "age": 45.0,
            "hypertension": 0.0,
            "heart_disease": 0.0,
            "avg_glucose_level": 91.0,
            "bmi": self.bmi_median,
            "gender": 0.5,
            "ever_married": 0.5,
            "work_type": 2.0,
            "residence_type": 0.5,
            "smoking_status": 1.5,
        }

        factors: list[dict[str, Any]] = []
        for feature in self.feature_columns:
            importance = self.feature_importance.get(feature, 0.0)
            val = float(row[feature].iloc[0])
            median = MEDIANS.get(feature, val)
            deviation = val - median
            contribution = importance * deviation
            direction = "↑ Increases Risk" if contribution > 0 else "↓ Lowers Risk"
            factors.append(
                {
                    "feature": DISPLAY_NAME_MAP.get(feature, feature),
                    "contribution": round(contribution, 5),
                    "direction": direction,
                }
            )

        factors.sort(key=lambda f: abs(f["contribution"]), reverse=True)
        return factors[:5]

    # ---- predict ----
    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.loaded or self.model is None:
            raise RuntimeError("Model not loaded")
        row = self.preprocess_input(payload)
        if hasattr(self.model, "predict_proba"):
            probability = float(self.model.predict_proba(row)[0][1])
        else:
            probability = float(self.model.predict(row)[0])

        return {
            "stroke_probability": probability,
            "risk_percentage": round(probability * 100, 2),
            "risk_label": "High Risk" if probability >= self.risk_threshold else "Low Risk",
            "key_contributing_factors": self._contributing_factors(row, probability),
        }


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    age: float = Field(..., ge=0, le=130)
    gender: str
    hypertension: Any
    heart_disease: Any
    ever_married: str
    work_type: str
    residence_type: str
    avg_glucose_level: float = Field(..., ge=0)
    bmi: float | None = None
    smoking_status: str

    @field_validator("hypertension", "heart_disease", mode="before")
    @classmethod
    def coerce_bool_str(cls, v: Any) -> int:
        if isinstance(v, str):
            return 1 if v.strip().lower() in ("yes", "1", "true") else 0
        return int(v)


# ---------------------------------------------------------------------------
# Lifespan & App
# ---------------------------------------------------------------------------
stroke_service = StrokeModelService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    stroke_service.load()
    yield


app = FastAPI(title="NeuroScan AI — Brain Stroke Detection API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "healthy", "model": stroke_service.model_name}


@app.get("/model-info")
def model_info():
    """Return model name, performance metrics, and feature importances."""
    if not stroke_service.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    m = stroke_service.metrics
    return {
        "model_name": stroke_service.model_name,
        "test_accuracy": round(float(m.get("Test Accuracy", 0)), 4),
        "test_f1": round(float(m.get("Test F1", 0)), 4),
        "roc_auc": round(float(m.get("ROC AUC", 0)), 4),
        "train_accuracy": round(float(m.get("Train Accuracy", 0)), 4),
        "feature_importance": {
            DISPLAY_NAME_MAP.get(k, k): round(v, 5)
            for k, v in sorted(
                stroke_service.feature_importance.items(),
                key=lambda x: -x[1],
            )
        },
    }


@app.post("/predict")
def predict(req: PredictRequest):
    if not stroke_service.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        result = stroke_service.predict(req.model_dump())
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail="Internal prediction error") from exc


@app.post("/analyze-report")
async def analyze_report(file: UploadFile = File(...)):
    """Extract patient data from a PDF/image report using Claude AI."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=503, detail="AI report analysis not configured")

    try:
        import anthropic
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=f"Missing dependency: {exc}") from exc

    contents = await file.read()
    filename = file.filename or ""

    # Extract text / base64 image
    if filename.lower().endswith(".pdf"):
        try:
            doc = fitz.open(stream=contents, filetype="pdf")
            text = "\n".join(page.get_text() for page in doc)
            doc.close()
            media_content: list[dict] = [{"type": "text", "text": text[:8000]}]
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Could not parse PDF: {exc}") from exc
    else:
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "jpeg"
        mime = f"image/{'png' if ext == 'png' else 'jpeg'}"
        b64 = base64.standard_b64encode(contents).decode()
        media_content = [
            {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}}
        ]

    SYSTEM_PROMPT = (
        "You are a medical data extraction assistant. "
        "Extract the following fields from the provided medical report and return ONLY a JSON object "
        "with these exact keys (null if not found): "
        "age (number), gender (Female/Male/Other), hypertension (Yes/No), "
        "heart_disease (Yes/No), avg_glucose_level (number), bmi (number), "
        "smoking_status (never smoked/formerly smoked/smokes/Unknown). "
        "Return only the JSON, no markdown, no explanation."
    )

    client = anthropic.Anthropic(api_key=api_key)
    try:
        message = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": media_content}],
        )
        raw = message.content[0].text.strip()
        # strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        extracted: dict = json.loads(raw)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"AI extraction failed: {exc}") from exc

    missing = [k for k, v in extracted.items() if v is None]

    # Attempt an immediate prediction if all required fields exist
    prediction = None
    required = ["age", "gender", "hypertension", "heart_disease",
                "avg_glucose_level", "smoking_status"]
    if stroke_service.loaded and all(extracted.get(k) is not None for k in required):
        try:
            payload = {**extracted, "ever_married": "Yes", "work_type": "Private",
                       "residence_type": "Urban"}
            prediction = stroke_service.predict(payload)
        except Exception:
            prediction = None

    return {"extracted_fields": extracted, "missing_fields": missing, "prediction": prediction}


# ---------------------------------------------------------------------------
# Static Files & HTML Routes
# ---------------------------------------------------------------------------
if STATIC_DIR.exists():
    # Serve /static/* assets
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def serve_index():
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return html_path.read_text(encoding="utf-8")


@app.get("/report", response_class=HTMLResponse)
def serve_report():
    html_path = STATIC_DIR / "report.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="report.html not found")
    return html_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Local Dev Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

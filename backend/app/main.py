from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .model_service import DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH, StrokeModelService, train_and_save_model
from .report_analyzer import analyze_medical_report_upload
from .schemas import PatientData, PredictionResponse, ReportAnalysisResponse, ReportExtractedFields

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR.parent / ".env")
STATIC_DIR = BASE_DIR / "app" / "static"

app = FastAPI(
    title="Brain Stroke Detection API",
    description="Predict stroke risk using an XGBoost model trained on the Kaggle stroke dataset.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

model_service = StrokeModelService(DEFAULT_MODEL_PATH)


@app.on_event("startup")
def startup_load_model() -> None:
    if not DEFAULT_MODEL_PATH.exists():
        if DEFAULT_DATASET_PATH.exists():
            logger.info("No model found. Training a new model from dataset...")
            train_and_save_model(DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH)
        else:
            logger.warning(
                "Model file not found and dataset missing at %s. Prediction endpoints will be unavailable.",
                DEFAULT_DATASET_PATH,
            )
            return

    model_service.load()
    logger.info("Stroke model loaded from %s", DEFAULT_MODEL_PATH)


@app.get("/", include_in_schema=False)
def serve_home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/report", include_in_schema=False)
def serve_report() -> FileResponse:
    return FileResponse(STATIC_DIR / "report.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok" if model_service.loaded else "model_not_loaded"}


@app.post("/predict", response_model=PredictionResponse)
def predict_stroke(payload: PatientData) -> dict:
    if not model_service.loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not available. Add dataset to backend/data and run backend/train_model.py.",
        )

    try:
        return model_service.predict(payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/analyze-report")
async def analyze_report(file: UploadFile = File(...)) -> dict:
    if not model_service.loaded:
        raise HTTPException(status_code=503, detail="Model is not available.")

    try:
        file_bytes = await file.read()
        extracted_fields = analyze_medical_report_upload(file_bytes, file.filename)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    missing_fields = [
        field_name
        for field_name, value in extracted_fields.items()
        if value is None or (isinstance(value, str) and not value.strip())
    ]

    prediction = None
    if not missing_fields:
        try:
            patient_payload = ReportExtractedFields(**extracted_fields).model_dump(exclude_none=True)
            patient_data = PatientData(**patient_payload)
            prediction = model_service.predict(patient_data.model_dump())
        except ValueError as exc:
            logger.warning("Report analysis produced invalid values: %s", exc)

    # Normalize prediction payload for frontend: risk_percent, risk_label, top_factors
    prediction_summary = None
    if prediction:
        prediction_summary = {
            "risk_percent": float(prediction.get("risk_percentage", 0)),
            "risk_label": prediction.get("risk_label"),
            "top_factors": [
                {
                    "feature": f.get("feature"),
                    "contribution": float(f.get("contribution", 0)),
                    "direction": f.get("direction"),
                }
                for f in prediction.get("key_contributing_factors", [])
            ],
        }

    return {
        "extracted_fields": extracted_fields,
        "prediction": prediction_summary,
        "missing_fields": missing_fields,
    }


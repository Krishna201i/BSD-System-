from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .model_service import DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH, StrokeModelService, train_and_save_model
from .schemas import FeatureImportanceResponse, PatientData, PredictionResponse

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parents[1]
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


@app.get("/dashboard", include_in_schema=False)
def serve_dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "dashboard.html")


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


@app.get("/feature-importance", response_model=FeatureImportanceResponse)
def feature_importance() -> dict:
    if not model_service.loaded:
        raise HTTPException(status_code=503, detail="Model is not available.")
    return model_service.get_feature_importance()


from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = BASE_DIR / "data" / "healthcare-dataset-stroke-data.csv"
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "stroke_xgboost.pkl"

TARGET_COLUMN = "stroke"
DISPLAY_NAME_MAP = {
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


def train_and_save_model(
    dataset_path: Path = DEFAULT_DATASET_PATH, model_path: Path = DEFAULT_MODEL_PATH
) -> dict[str, Any]:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Place healthcare-dataset-stroke-data.csv in backend\\data."
        )

    df = pd.read_csv(dataset_path)
    required_columns = {
        "id",
        "gender",
        "age",
        "hypertension",
        "heart_disease",
        "ever_married",
        "work_type",
        "Residence_type",
        "avg_glucose_level",
        "bmi",
        "smoking_status",
        "stroke",
    }
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {', '.join(missing_columns)}")

    df = df.rename(columns={"Residence_type": "residence_type"})
    df = df.drop(columns=["id"])
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    bmi_median = float(df["bmi"].median())
    df["bmi"] = df["bmi"].fillna(bmi_median)

    categorical_columns = ["gender", "ever_married", "work_type", "residence_type", "smoking_status"]
    label_encoders: dict[str, list[str]] = {}
    for column in categorical_columns:
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column].astype(str).str.strip())
        label_encoders[column] = [str(value) for value in encoder.classes_]

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=2,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
    )
    model.fit(X_train_balanced, y_train_balanced)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    artifact: dict[str, Any] = {
        "model": model,
        "bmi_median": bmi_median,
        "categorical_columns": categorical_columns,
        "label_encoders": label_encoders,
        "feature_columns": list(X.columns),
        "feature_importance": {
            feature: float(importance)
            for feature, importance in zip(X.columns, model.feature_importances_, strict=True)
        },
        "metrics": {
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        },
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    return artifact


class StrokeModelService:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH, risk_threshold: float = 0.5) -> None:
        self.model_path = model_path
        self.risk_threshold = risk_threshold
        self.loaded = False
        self.model: XGBClassifier | None = None
        self.label_encoders: dict[str, list[str]] = {}
        self.feature_columns: list[str] = []
        self.categorical_columns: list[str] = []
        self.bmi_median: float = 0.0
        self.feature_importance: dict[str, float] = {}
        self.metrics: dict[str, float] = {}

    def load(self) -> None:
        artifact = joblib.load(self.model_path)
        self.model = artifact["model"]
        self.label_encoders = artifact["label_encoders"]
        self.feature_columns = artifact["feature_columns"]
        self.categorical_columns = artifact["categorical_columns"]
        self.bmi_median = float(artifact["bmi_median"])
        self.feature_importance = artifact["feature_importance"]
        self.metrics = artifact.get("metrics", {})
        self.loaded = True

    def _encode_categorical(self, column: str, value: str) -> int:
        if column not in self.label_encoders:
            raise ValueError(f"Unsupported categorical field: {column}")

        classes = self.label_encoders[column]
        normalized_map = {item.strip().lower(): idx for idx, item in enumerate(classes)}
        key = value.strip().lower()
        if key not in normalized_map:
            allowed = ", ".join(classes)
            raise ValueError(f"Invalid value '{value}' for {column}. Allowed values: {allowed}")
        return normalized_map[key]

    def preprocess_input(self, payload: dict[str, Any]) -> pd.DataFrame:
        if not self.loaded:
            raise RuntimeError("Model is not loaded")

        processed: dict[str, Any] = {
            "age": float(payload["age"]),
            "hypertension": int(payload["hypertension"]),
            "heart_disease": int(payload["heart_disease"]),
            "avg_glucose_level": float(payload["avg_glucose_level"]),
        }

        bmi_value = payload.get("bmi")
        processed["bmi"] = self.bmi_median if bmi_value is None else float(bmi_value)

        for column in self.categorical_columns:
            processed[column] = self._encode_categorical(column, str(payload[column]))

        row = pd.DataFrame([processed])
        return row[self.feature_columns]

    def _top_contributing_factors(self, row: pd.DataFrame) -> list[dict[str, Any]]:
        if self.model is None:
            raise RuntimeError("Model is not loaded")

        dmatrix = xgb.DMatrix(row)
        contribs = self.model.get_booster().predict(dmatrix, pred_contribs=True)[0]
        feature_contribs = contribs[:-1]
        ranked_idx = np.argsort(np.abs(feature_contribs))[::-1]

        factors: list[dict[str, Any]] = []
        for idx in ranked_idx:
            value = float(feature_contribs[idx])
            if abs(value) < 1e-10:
                continue
            feature = self.feature_columns[idx]
            factors.append(
                {
                    "feature": DISPLAY_NAME_MAP.get(feature, feature),
                    "contribution": value,
                    "direction": "higher risk" if value > 0 else "lower risk",
                }
            )
            if len(factors) == 3:
                break

        return factors

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.loaded or self.model is None:
            raise RuntimeError("Model is not loaded")

        row = self.preprocess_input(payload)
        probability = float(self.model.predict_proba(row)[0][1])
        return {
            "stroke_probability": probability,
            "risk_percentage": probability * 100,
            "risk_label": "High Risk" if probability >= self.risk_threshold else "Low Risk",
            "key_contributing_factors": self._top_contributing_factors(row),
        }

    def get_feature_importance(self) -> dict[str, Any]:
        if not self.loaded:
            raise RuntimeError("Model is not loaded")

        items = sorted(self.feature_importance.items(), key=lambda pair: pair[1], reverse=True)
        return {
            "feature_importance": [
                {
                    "feature": feature,
                    "feature_label": DISPLAY_NAME_MAP.get(feature, feature),
                    "importance": float(importance),
                }
                for feature, importance in items
            ],
            "model_metrics": self.metrics,
        }


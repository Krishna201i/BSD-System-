from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.combine import SMOTETomek
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = BASE_DIR / "data" / "healthcare-dataset-stroke-data.csv"
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "stroke_xgboost.pkl"

TARGET_COLUMN = "stroke"
DEFAULT_RISK_THRESHOLD = 0.3
THRESHOLD_SEARCH_MIN = 0.25
THRESHOLD_SEARCH_MAX = 0.45
RECALL_FLOOR = 0.70
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
    smote_tomek = SMOTETomek(random_state=42)
    X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train, y_train)

    classes = np.array([0, 1], dtype=int)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_balanced)
    class_weight_map = {int(cls): float(weight) for cls, weight in zip(classes, class_weights, strict=True)}
    sample_weights = np.array([class_weight_map[int(label)] for label in y_train_balanced], dtype=float)

    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())
    scale_pos_weight = float(negatives / positives) if positives else 1.0

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        reg_lambda=1.0,
        random_state=42,
    )
    model.fit(X_train_balanced, y_train_balanced, sample_weight=sample_weights)

    y_prob = model.predict_proba(X_test)[:, 1]

    _, _, curve_thresholds = precision_recall_curve(y_test, y_prob)
    threshold_candidates = sorted(
        {
            float(threshold)
            for threshold in curve_thresholds
            if THRESHOLD_SEARCH_MIN <= float(threshold) <= THRESHOLD_SEARCH_MAX
        }
    )
    if not threshold_candidates:
        threshold_candidates = [DEFAULT_RISK_THRESHOLD]

    best_threshold = threshold_candidates[0]
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = -1.0
    found_with_recall_floor = False

    for threshold in threshold_candidates:
        y_pred_candidate = (y_prob >= threshold).astype(int)
        precision = float(precision_score(y_test, y_pred_candidate, zero_division=0))
        recall = float(recall_score(y_test, y_pred_candidate, zero_division=0))
        f1 = float(f1_score(y_test, y_pred_candidate, zero_division=0))

        if recall < RECALL_FLOOR:
            continue

        found_with_recall_floor = True
        if (
            f1 > best_f1
            or (f1 == best_f1 and precision > best_precision)
            or (f1 == best_f1 and precision == best_precision and recall > best_recall)
        ):
            best_threshold = threshold
            best_precision = precision
            best_recall = recall
            best_f1 = f1

    if not found_with_recall_floor:
        for threshold in threshold_candidates:
            y_pred_candidate = (y_prob >= threshold).astype(int)
            precision = float(precision_score(y_test, y_pred_candidate, zero_division=0))
            recall = float(recall_score(y_test, y_pred_candidate, zero_division=0))
            f1 = float(f1_score(y_test, y_pred_candidate, zero_division=0))
            if (
                f1 > best_f1
                or (f1 == best_f1 and precision > best_precision)
                or (f1 == best_f1 and precision == best_precision and recall > best_recall)
            ):
                best_threshold = threshold
                best_precision = precision
                best_recall = recall
                best_f1 = f1

    y_pred = (y_prob >= best_threshold).astype(int)

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
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "decision_threshold": float(best_threshold),
        },
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    return artifact


class StrokeModelService:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH, risk_threshold: float = DEFAULT_RISK_THRESHOLD) -> None:
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
        saved_threshold = self.metrics.get("decision_threshold")
        if isinstance(saved_threshold, (int, float)):
            self.risk_threshold = float(saved_threshold)
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


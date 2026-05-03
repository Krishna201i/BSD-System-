from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

import anthropic
import fitz
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

SYSTEM_PROMPT = (
    "You are a medical report analyzer. Extract these exact values from the report if present: age, gender, "
    "hypertension (yes/no), heart_disease (yes/no), avg_glucose_level (number), bmi (number), smoking_status "
    "(never/formerly smoked/smokes/Unknown). Return ONLY a JSON object with these exact keys. If a value is not "
    "found, use null. Do not add any explanation."
)

REPORT_FIELD_NAMES = (
    "age",
    "gender",
    "hypertension",
    "heart_disease",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
PDF_EXTENSIONS = {".pdf"}
ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | PDF_EXTENSIONS


def _get_client() -> anthropic.Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to the .env file.")
    return anthropic.Anthropic(api_key=api_key)


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_gender(value: Any) -> str | None:
    text = _normalize_text(value)
    if text is None:
        return None

    normalized = text.lower()
    if normalized in {"m", "male"}:
        return "Male"
    if normalized in {"f", "female"}:
        return "Female"
    if normalized in {"other", "non-binary", "nonbinary"}:
        return "Other"
    return text


def _normalize_yes_no(value: Any) -> str | None:
    text = _normalize_text(value)
    if text is None:
        return None

    normalized = text.lower()
    if normalized in {"yes", "y", "true", "1"}:
        return "Yes"
    if normalized in {"no", "n", "false", "0"}:
        return "No"
    return text


def _normalize_smoking_status(value: Any) -> str | None:
    text = _normalize_text(value)
    if text is None:
        return None

    normalized = text.lower()
    if normalized in {"never", "never smoked", "non-smoker", "nonsmoker"}:
        return "never smoked"
    if normalized in {"formerly smoked", "former smoker", "former smoker", "past smoker"}:
        return "formerly smoked"
    if normalized in {"smokes", "smoker", "current smoker"}:
        return "smokes"
    if normalized == "unknown":
        return "Unknown"
    return text


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = _normalize_text(value)
    if text is None:
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    text = _normalize_text(value)
    if text is None:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _normalize_extracted_values(raw_values: dict[str, Any]) -> dict[str, Any]:
    return {
        "age": _coerce_int(raw_values.get("age")),
        "gender": _normalize_gender(raw_values.get("gender")),
        "hypertension": _normalize_yes_no(raw_values.get("hypertension")),
        "heart_disease": _normalize_yes_no(raw_values.get("heart_disease")),
        "avg_glucose_level": _coerce_float(raw_values.get("avg_glucose_level")),
        "bmi": _coerce_float(raw_values.get("bmi")),
        "smoking_status": _normalize_smoking_status(raw_values.get("smoking_status")),
    }


def _extract_response_text(response: Any) -> str:
    parts: list[str] = []
    for block in getattr(response, "content", []):
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts).strip()


def _parse_json_response(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Claude response did not contain valid JSON.")
        parsed = json.loads(text[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("Claude response must be a JSON object.")
    return parsed


def _call_claude_with_image(image_bytes: bytes, media_type: str) -> dict[str, Any]:
    client = _get_client()
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": encoded_image,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Extract the medical values as JSON.",
                    },
                ],
            }
        ],
    )

    response_text = _extract_response_text(response)
    return _parse_json_response(response_text)


def _validate_extension(filename: str | None) -> str:
    extension = Path(filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported file type. Upload a PDF, JPG, or PNG file.")
    return extension


def _pdf_pages_to_images(pdf_bytes: bytes) -> list[bytes]:
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    images: list[bytes] = []

    for page in document:
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        images.append(pixmap.tobytes("png"))

    document.close()
    return images


def analyze_medical_report_upload(file_bytes: bytes, filename: str | None) -> dict[str, Any]:
    extension = _validate_extension(filename)

    image_blobs: list[tuple[bytes, str]]
    if extension in PDF_EXTENSIONS:
        image_blobs = [(blob, "image/png") for blob in _pdf_pages_to_images(file_bytes)]
    else:
        image_blobs = [(file_bytes, "image/jpeg" if extension in {".jpg", ".jpeg"} else "image/png")]

    extracted: dict[str, Any] = {field_name: None for field_name in REPORT_FIELD_NAMES}

    for image_bytes, media_type in image_blobs:
        page_values = _normalize_extracted_values(_call_claude_with_image(image_bytes, media_type))
        for field_name, value in page_values.items():
            if extracted[field_name] is None and value is not None:
                extracted[field_name] = value

        if all(extracted[field_name] is not None for field_name in REPORT_FIELD_NAMES):
            break

    return extracted
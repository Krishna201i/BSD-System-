# Brain Stroke Detection Web App

FastAPI + HTML/CSS/JS web app for stroke risk prediction using the Kaggle stroke dataset.

## Features

- XGBoost model for stroke prediction
- SMOTE for class imbalance handling
- BMI median imputation
- Label encoding for categorical features
- `POST /predict` API returning probability + High/Low risk
- Key contributing factors per prediction
- Dashboard with feature importance chart + model metrics
- Mobile-responsive UI + loading spinner

## Project Structure

```text
backend/
  app/
    main.py
    model_service.py
    schemas.py
    static/
      index.html
      dashboard.html
      styles.css
      app.js
      dashboard.js
  data/
    healthcare-dataset-stroke-data.csv   # add this
  models/
    stroke_xgboost.pkl                   # generated after training
  train_model.py
  requirements.txt
```

## Setup

1. Place Kaggle dataset file at:
   - `backend\data\healthcare-dataset-stroke-data.csv`
2. Create and activate a virtual environment.
3. Install dependencies:
   - `pip install -r backend\requirements.txt`
4. Train model:
   - `python backend\train_model.py`
5. Run app:
   - `uvicorn app.main:app --app-dir backend --reload`
6. Open:
   - `http://127.0.0.1:8000` (prediction UI)
   - `http://127.0.0.1:8000/dashboard` (dashboard)

## API

### `POST /predict`

Request body:

```json
{
  "age": 67,
  "gender": "Male",
  "hypertension": "Yes",
  "heart_disease": "No",
  "ever_married": "Yes",
  "work_type": "Private",
  "residence_type": "Urban",
  "avg_glucose_level": 228.69,
  "bmi": 36.6,
  "smoking_status": "formerly smoked"
}
```

Response:

```json
{
  "stroke_probability": 0.8235,
  "risk_percentage": 82.35,
  "risk_label": "High Risk",
  "key_contributing_factors": [
    { "feature": "Age", "contribution": 0.8421, "direction": "higher risk" },
    { "feature": "Avg Glucose Level", "contribution": 0.5612, "direction": "higher risk" },
    { "feature": "Hypertension", "contribution": 0.3145, "direction": "higher risk" }
  ]
}
```

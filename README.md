# NeuroScan AI 🧠

**NeuroScan AI** is a smart, end-to-end stroke risk detection platform. It uses machine learning to predict a patient's likelihood of experiencing a brain stroke based on their medical history, lifestyle factors, and demographics.

The system features a clean, responsive frontend built with Vanilla HTML/CSS/JS and a robust **FastAPI** backend that serves both the UI and the ML predictions. It also includes an **AI Report Analyzer** that uses Claude API to extract patient data directly from medical reports (PDF/images).

## ✨ Features

- **Stroke Risk Prediction**: Uses a trained Machine Learning model (Gradient Boosting/Random Forest/XGBoost) to evaluate user inputs and provide a risk probability.
- **Key Contributing Factors**: Dynamically calculates and displays which factors (e.g., Age, Glucose Levels, Hypertension) are increasing or decreasing the patient's specific risk.
- **AI Report Analyzer**: Upload a medical report (PDF/Image) and let Claude AI automatically extract the relevant fields to pre-fill the form.
- **Live Model Stats**: Displays the active model's name, Test Accuracy, ROC AUC, and F1 Score dynamically on the frontend.
- **Single-Command Deployment**: The backend and frontend are tightly integrated to run from a single FastAPI server, making it trivial to deploy to platforms like Render.

## 🛠️ Tech Stack

- **Backend**: Python 3.12, FastAPI, Uvicorn, Pandas, Numpy
- **Machine Learning**: Scikit-Learn, XGBoost, Imbalanced-Learn (SMOTETomek)
- **Frontend**: Vanilla HTML5, CSS3 (Custom Properties, Animations), JavaScript
- **AI Integration**: Anthropic Claude Opus (for medical report parsing via PyMuPDF)

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- An Anthropic API Key (if you want to use the AI Report Analyzer)

### Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd BSD-System-
   ```

2. **Set up a virtual environment (optional but recommended):**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Create a `.env` file in the root directory (or `backend/` directory) and add your Anthropic API Key:
   ```ini
   ANTHROPIC_API_KEY=your_api_key_here
   ```

5. **Run the Application:**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

6. **View the App:**
   Open your browser and navigate to `http://localhost:8000`.

## 🧠 Model Training

The predictive model is trained on a healthcare stroke dataset. If you need to re-train the model or if you encounter a pickle version mismatch:

1. Open the `backend/Brain_Stroke_Detection.ipynb` Jupyter Notebook to view the full data pipeline, visualizations (Confusion Matrix, ROC Curve, Feature Importance), and evaluations.
2. Alternatively, run the retraining script to quickly regenerate the `models/stroke_best_model.pkl` file:
   ```bash
   cd backend
   python retrain.py
   ```

## ☁️ Deployment (Render)

This project is configured to be deployed easily on [Render](https://render.com).

1. Push your code to GitHub.
2. Connect your repository to Render.
3. A `render.yaml` Blueprint is included in the root directory. Render will automatically detect it and set up a Web Service.
4. Go to the Render Dashboard for your new Web Service and add your `ANTHROPIC_API_KEY` to the Environment Variables.

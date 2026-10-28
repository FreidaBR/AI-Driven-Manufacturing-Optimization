
from pathlib import Path
import json
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE = Path(__file__).resolve().parents[1]

DATA = BASE / "data" / "phase4_outputs"
MODEL_DIR = BASE / "models"

MODEL_PATH = MODEL_DIR / "lightgbm_model.joblib"
IMPUTER_PATH = MODEL_DIR / "imputer.joblib"
FEATURES_PATH = MODEL_DIR / "feature_columns.json"


# --------------------------------------------------
# CSV FILES
# --------------------------------------------------

FILES = {
    "priorities": "machine_priority_ranking.csv",
    "recommendations": "machine_component_recommendations.csv",
    "components": "historical_component_failure_profile.csv",
    "schedule": "maintenance_inspection_schedule.csv",
    "daily_workload": "daily_maintenance_workload.csv",
    "capacity": "maintenance_capacity_scenarios.csv",
    "failure_matches": "strict_24h_failure_matches.csv",
}


# --------------------------------------------------
# FASTAPI APP
# --------------------------------------------------

app = FastAPI(
    title="AI Manufacturing Optimization API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# --------------------------------------------------
# LOAD MODEL ARTIFACTS
# --------------------------------------------------

model = None
imputer = None
feature_columns = []
model_load_error = None

try:
    model = joblib.load(MODEL_PATH)
    imputer = joblib.load(IMPUTER_PATH)

    with open(FEATURES_PATH, "r") as f:
        feature_columns = json.load(f)

    if len(feature_columns) != 47:
        raise ValueError(
            f"Expected 47 features, found {len(feature_columns)}"
        )

    print("Model loaded successfully!")
    print(f"Number of features: {len(feature_columns)}")

except Exception as e:
    model_load_error = str(e)
    print(f"Model loading failed: {model_load_error}")


# --------------------------------------------------
# PREDICTION REQUEST SCHEMA
# --------------------------------------------------

class PredictionRequest(BaseModel):
    features: dict[str, float | None]

    threshold: float = Field(
        default=0.98,
        ge=0,
        le=1
    )


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def read_csv(key):
    path = DATA / FILES[key]

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def rows(df):
    if df.empty:
        return []

    return (
        df.astype(object)
        .where(pd.notna(df), None)
        .to_dict("records")
    )


# --------------------------------------------------
# HOME
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Manufacturing Optimization API",
        "docs": "/docs"
    }


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/api/health")
def health():

    return {
        "files": {
            k: (DATA / v).exists()
            for k, v in FILES.items()
        },
        "model_loaded": model is not None,
        "feature_count": len(feature_columns),
        "model_error": model_load_error
    }


# --------------------------------------------------
# MODEL HEALTH
# --------------------------------------------------

@app.get("/api/model/health")
def model_health():

    return {
        "model_loaded": model is not None,
        "imputer_loaded": imputer is not None,
        "feature_count": len(feature_columns),
        "prediction_endpoint": "/api/predict"
    }


# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

@app.get("/api/overview")
def overview():

    df = read_csv("priorities")

    counts = (
        df["priority"].astype(str).str.upper().value_counts().to_dict()
        if "priority" in df.columns
        else {}
    )

    return {
        "total_machines": len(df),
        "critical": int(counts.get("CRITICAL", 0)),
        "high": int(counts.get("HIGH", 0)),
        "medium": int(counts.get("MEDIUM", 0)),
        "low": int(counts.get("LOW", 0)),
        "capacity_scenarios": rows(read_csv("capacity")),
        "data_loaded": not df.empty
    }


# --------------------------------------------------
# PREDICT FAILURE RISK
# --------------------------------------------------

@app.post("/api/predict")
def predict(request: PredictionRequest):

    if model is None or imputer is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction model or imputer is not available."
        )

    # Check for unexpected feature names
    unknown_features = (
        set(request.features.keys()) - set(feature_columns)
    )

    if unknown_features:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Unknown features supplied.",
                "unknown_features": list(unknown_features)
            }
        )

    try:
        # Arrange features in the exact training order.
        # Missing features are passed as None for imputation.
        input_data = {
            col: request.features.get(col)
            for col in feature_columns
        }

        input_df = pd.DataFrame(
            [input_data],
            columns=feature_columns
        )

        # Apply the fitted training imputer
        input_imputed = imputer.transform(input_df)

        # Predict probability of failure within 24 hours
        probability = float(
            model.predict_proba(input_imputed)[0][1]
        )

        alert = probability >= request.threshold

        return {
            "failure_probability_24h": round(probability, 6),
            "failure_probability_percentage": round(
                probability * 100, 2
            ),
            "threshold": request.threshold,
            "alert": bool(alert),
            "status": "ALERT" if alert else "NORMAL"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# --------------------------------------------------
# DATASET ENDPOINTS
# --------------------------------------------------

@app.get("/api/{dataset}")
def dataset(dataset: str):

    if dataset not in FILES:
        return {
            "error": "Unknown dataset",
            "items": []
        }

    df = read_csv(dataset)

    return {
        "items": rows(df),
        "count": len(df)
    }

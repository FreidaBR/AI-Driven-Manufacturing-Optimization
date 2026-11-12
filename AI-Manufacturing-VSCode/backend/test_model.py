
import json
import joblib
from pathlib import Path

import pandas as pd

from app.evaluation_data import load_labeled_evaluation_data
from app.model_utils import impute_missing_features, load_imputer_statistics

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"
RAW_DATA_DIR = BASE / "data" / "raw" / "cleaned_dataset"
GENERATED_FEATURES_PATH = (
    BASE / "data" / "generated" / "manufacturing_features_local.parquet"
)

with open(MODEL_DIR / "feature_columns.json", encoding="utf-8") as file:
    feature_columns = json.load(file)

model = joblib.load(MODEL_DIR / "lightgbm_model.joblib")
imputer_statistics = load_imputer_statistics(
    MODEL_DIR / "imputer_statistics.json",
    feature_columns,
)

evaluation_data, labels = load_labeled_evaluation_data(
    RAW_DATA_DIR,
    MODEL_DIR / "feature_columns.json",
    GENERATED_FEATURES_PATH,
)
evaluation_data = evaluation_data.assign(failure_next_24h=labels)

normal_rows = evaluation_data[evaluation_data["failure_next_24h"] == 0].head(5)
failure_rows = evaluation_data[evaluation_data["failure_next_24h"] == 1].head(5)
test_rows = pd.concat([normal_rows, failure_rows])

X = test_rows[feature_columns]
X_imputed = impute_missing_features(
    X,
    feature_columns,
    imputer_statistics,
)
probabilities = model.predict_proba(X_imputed)[:, 1]

results = pd.DataFrame(
    {
    "machineID": test_rows["machineID"].values,
    "datetime": test_rows["datetime"].values,
    "actual_label": test_rows["failure_next_24h"].values,
    "predicted_probability": probabilities
    }
)

results["predicted_percentage"] = (
    results["predicted_probability"] * 100
).round(4)

results["alert_at_0.98"] = (
    results["predicted_probability"] >= 0.98
)

print("\nNORMAL VS FAILURE RECORDS")
print(results.to_string(index=False))

print("\nAverage probability for normal records:",
      round(results[results.actual_label == 0]
            ["predicted_probability"].mean(), 6))

print("Average probability for failure records:",
      round(results[results.actual_label == 1]
            ["predicted_probability"].mean(), 6))

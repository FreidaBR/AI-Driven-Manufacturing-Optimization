
import json
import joblib
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent

df = pd.read_parquet(
    BASE / "data" / "cleaned_dataset" / "manufacturing_features.parquet"
)

model = joblib.load(BASE / "models" / "lightgbm_model.joblib")
imputer = joblib.load(BASE / "models" / "imputer.joblib")

with open(BASE / "models" / "feature_columns.json") as f:
    feature_columns = json.load(f)

# Select normal and failure records
normal_rows = df[df["failure_next_24h"] == 0].head(5)
failure_rows = df[df["failure_next_24h"] == 1].head(5)

test_rows = pd.concat([normal_rows, failure_rows])

# Prepare model features
X = test_rows[feature_columns]

X_imputed = pd.DataFrame(
    imputer.transform(X),
    columns=feature_columns,
    index=X.index
)

# Predict
probabilities = model.predict_proba(X_imputed)[:, 1]

results = pd.DataFrame({
    "machineID": test_rows["machineID"].values,
    "datetime": test_rows["datetime"].values,
    "actual_label": test_rows["failure_next_24h"].values,
    "predicted_probability": probabilities
})

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

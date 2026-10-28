
import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score
)

BASE = Path(__file__).resolve().parent

df = pd.read_parquet(
    BASE / "data" / "cleaned_dataset" / "manufacturing_features.parquet"
)

model = joblib.load(BASE / "models" / "lightgbm_model.joblib")
imputer = joblib.load(BASE / "models" / "imputer.joblib")

with open(BASE / "models" / "feature_columns.json") as f:
    feature_columns = json.load(f)

# Prepare features
X = df[feature_columns]
y = df["failure_next_24h"].astype(int)

# Impute missing values
X_imputed = pd.DataFrame(
    imputer.transform(X),
    columns=feature_columns
)

# Predict in batches to limit memory usage
batch_size = 50000
probabilities = []

for start in range(0, len(X_imputed), batch_size):
    end = min(start + batch_size, len(X_imputed))
    batch = X_imputed.iloc[start:end]
    probabilities.extend(model.predict_proba(batch)[:, 1])

probabilities = pd.Series(probabilities)

# Evaluate at threshold 0.98
threshold = 0.98
predictions = (probabilities >= threshold).astype(int)

print("\n===== MODEL EVALUATION =====")
print("Total records:", len(df))
print("Threshold:", threshold)

print("\nConfusion Matrix:")
print(confusion_matrix(y, predictions))

print("\nClassification Report:")
print(classification_report(y, predictions, digits=4))

print("Average Precision:", average_precision_score(y, probabilities))
print("ROC AUC:", roc_auc_score(y, probabilities))

print("\nAverage probability by actual class:")
print(
    pd.DataFrame({
        "actual": y,
        "probability": probabilities
    }).groupby("actual")["probability"].mean()
)

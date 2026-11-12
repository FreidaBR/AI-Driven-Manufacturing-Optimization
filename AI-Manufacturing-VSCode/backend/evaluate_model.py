
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

from app.evaluation_data import load_labeled_evaluation_data
from app.model_utils import impute_missing_features, load_imputer_statistics

BASE = Path(__file__).resolve().parent
RAW_DATA_DIR = BASE / "data" / "raw" / "cleaned_dataset"
GENERATED_FEATURES_PATH = (
    BASE / "data" / "generated" / "manufacturing_features_local.parquet"
)
MODEL_DIR = BASE / "models"

with open(MODEL_DIR / "feature_columns.json", encoding="utf-8") as f:
    feature_columns = json.load(f)

model = joblib.load(MODEL_DIR / "lightgbm_model.joblib")
imputer_statistics = load_imputer_statistics(
    MODEL_DIR / "imputer_statistics.json",
    feature_columns,
)

df, y = load_labeled_evaluation_data(
    RAW_DATA_DIR,
    MODEL_DIR / "feature_columns.json",
    GENERATED_FEATURES_PATH,
)
y = pd.Series(y, index=df.index, dtype=int)

# Predict in batches to limit memory usage
X = df[feature_columns]
X_imputed = impute_missing_features(
    X,
    feature_columns,
    imputer_statistics,
)
batch_size = 50000
probabilities = []

for start in range(0, len(X_imputed), batch_size):
    end = min(start + batch_size, len(X_imputed))
    batch = X_imputed.iloc[start:end]
    probabilities.extend(model.predict_proba(batch)[:, 1])

probabilities = pd.Series(probabilities, index=y.index)

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

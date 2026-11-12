
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)

from app.evaluation_data import load_labeled_evaluation_data
from app.model_utils import impute_missing_features, load_imputer_statistics


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

GENERATED_DIR = BASE_DIR / "data" / "generated"
MODEL_DIR = BASE_DIR / "models"

FEATURES_PATH = MODEL_DIR / "feature_columns.json"
MODEL_PATH = MODEL_DIR / "lightgbm_model.joblib"
IMPUTER_STATISTICS_PATH = MODEL_DIR / "imputer_statistics.json"
RAW_DATA_DIR = BASE_DIR / "data" / "raw" / "cleaned_dataset"
GENERATED_FEATURES_PATH = GENERATED_DIR / "manufacturing_features_local.parquet"

THRESHOLDS = [0.90, 0.95, 0.97, 0.98, 0.99]


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

print("Loading model artifacts...")

model = joblib.load(MODEL_PATH)

with open(FEATURES_PATH, "r") as f:
    feature_columns = json.load(f)

imputer_statistics = load_imputer_statistics(
    IMPUTER_STATISTICS_PATH,
    feature_columns,
)

print(f"Loaded {len(feature_columns)} model features.")


# --------------------------------------------------
# LOAD ENGINEERED FEATURES
# --------------------------------------------------

evaluation_data, y_true = load_labeled_evaluation_data(
    RAW_DATA_DIR,
    FEATURES_PATH,
    GENERATED_FEATURES_PATH,
)

print(f"Eligible snapshots: {len(evaluation_data):,}")
print(f"Failure-positive snapshots: {int(y_true.sum()):,}")
print(f"Failure-negative snapshots: {int((y_true == 0).sum()):,}")


# --------------------------------------------------
# GENERATE MODEL PROBABILITIES
# --------------------------------------------------

print("Generating model predictions...")

X = evaluation_data[feature_columns].copy()
X_imputed = impute_missing_features(
    X,
    feature_columns,
    imputer_statistics,
)

probabilities = model.predict_proba(X_imputed)[:, 1]

print("Predictions generated.")


# --------------------------------------------------
# EVALUATE THRESHOLDS
# --------------------------------------------------

results = []

for threshold in THRESHOLDS:

    y_pred = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    results.append({
        "threshold": threshold,
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1_score": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "predicted_alerts": int(y_pred.sum()),
    })


results_df = pd.DataFrame(results)

print("\nTHRESHOLD COMPARISON")
print("=" * 100)

print(
    results_df.to_string(
        index=False,
        formatters={
            "precision": "{:.4f}".format,
            "recall": "{:.4f}".format,
            "f1_score": "{:.4f}".format,
        }
    )
)


# --------------------------------------------------
# OVERALL RANKING METRICS
# --------------------------------------------------

print("\nOVERALL MODEL METRICS")
print("=" * 60)

print(
    "Average Precision:",
    round(average_precision_score(y_true, probabilities), 5)
)

print(
    "ROC AUC:",
    round(roc_auc_score(y_true, probabilities), 5)
)


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

output_dir = BASE_DIR / "evaluation_results"
output_dir.mkdir(exist_ok=True)

results_df.to_csv(
    output_dir / "threshold_comparison.csv",
    index=False
)

prediction_output = evaluation_data[
    ["machineID", "datetime"]
].copy()

prediction_output["actual_failure_24h"] = y_true
prediction_output["predicted_failure_probability"] = probabilities

prediction_output.to_csv(
    output_dir / "historical_predictions.csv",
    index=False
)

print("\nEvaluation completed!")
print(f"Results saved in: {output_dir}")

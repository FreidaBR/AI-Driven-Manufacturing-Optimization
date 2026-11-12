"""Train a reproducible LightGBM candidate on historical telemetry."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.evaluation_data import HORIZON_HOURS, load_labeled_evaluation_data
from app.model_utils import impute_missing_features, load_imputer_statistics


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
RAW_DATA_DIR = BASE_DIR / "data" / "raw" / "cleaned_dataset"
GENERATED_FEATURES_PATH = (
    BASE_DIR / "data" / "generated" / "manufacturing_features_local.parquet"
)
CANDIDATE_DIR = MODEL_DIR / "retrained_candidate"
TEST_FRACTION = 0.20
ALERT_THRESHOLD = 0.98
MODEL_PARAMETERS = {
    "objective": "binary",
    "class_weight": "balanced",
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "subsample_freq": 0,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}
SENSOR_FEATURE_COLUMNS = {
    "volt",
    "rotate",
    "pressure",
    "vibration",
}


def calculate_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int | list[list[int]]]:
    predictions = probabilities >= threshold
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "average_precision": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "precision_at_threshold": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall_at_threshold": float(
            recall_score(labels, predictions, zero_division=0)
        ),
        "f1_at_threshold": float(f1_score(labels, predictions, zero_division=0)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "confusion_matrix_labels_0_1": matrix.tolist(),
        "true_negatives": int(matrix[0, 0]),
        "false_positives": int(matrix[0, 1]),
        "false_negatives": int(matrix[1, 0]),
        "true_positives": int(matrix[1, 1]),
        "predicted_alerts": int(predictions.sum()),
    }


def threshold_sensitivity(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, dict[str, float | int | list[list[int]]]]:
    return {
        str(threshold): calculate_metrics(labels, probabilities, threshold)
        for threshold in (0.50, 0.85, ALERT_THRESHOLD)
    }


def main() -> None:
    features_path = MODEL_DIR / "feature_columns.json"
    feature_columns = json.loads(features_path.read_text(encoding="utf-8"))
    if len(feature_columns) != 47:
        raise ValueError(f"Expected 47 model features, got {len(feature_columns)}.")

    data, labels = load_labeled_evaluation_data(
        RAW_DATA_DIR,
        features_path,
        GENERATED_FEATURES_PATH,
    )
    labels = np.asarray(labels, dtype=np.int8)
    timestamps = pd.to_datetime(data["datetime"])
    unique_timestamps = np.sort(timestamps.unique())
    cutoff_index = int(len(unique_timestamps) * (1 - TEST_FRACTION))
    cutoff = pd.Timestamp(unique_timestamps[cutoff_index])
    embargo = pd.Timedelta(hours=HORIZON_HOURS)

    train_mask = timestamps < cutoff - embargo
    test_mask = timestamps >= cutoff
    if not train_mask.any() or not test_mask.any():
        raise ValueError("Chronological split produced an empty train or test set.")

    y_train = labels[train_mask]
    y_test = labels[test_mask]
    if np.unique(y_train).size != 2 or np.unique(y_test).size != 2:
        raise ValueError("Both classes must occur in the train and test partitions.")

    X_train_raw = data.loc[train_mask, feature_columns]
    X_test_raw = data.loc[test_mask, feature_columns]
    medians = X_train_raw.median(axis=0, skipna=True)
    all_missing = medians[medians.isna()].index.tolist()
    if all_missing:
        medians.loc[all_missing] = 0.0
    imputer_statistics = tuple(float(value) for value in medians.to_numpy())
    X_train = impute_missing_features(
        X_train_raw, feature_columns, imputer_statistics
    )
    X_test = impute_missing_features(
        X_test_raw, feature_columns, imputer_statistics
    )

    model = LGBMClassifier(**MODEL_PARAMETERS)
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    candidate_metrics = calculate_metrics(
        y_test,
        probabilities,
        ALERT_THRESHOLD,
    )

    existing_model = joblib.load(MODEL_DIR / "lightgbm_model.joblib")
    existing_statistics = load_imputer_statistics(
        MODEL_DIR / "imputer_statistics.json",
        feature_columns,
    )
    existing_X_test = impute_missing_features(
        X_test_raw,
        feature_columns,
        existing_statistics,
    )
    existing_probabilities = existing_model.predict_proba(existing_X_test)[:, 1]
    existing_metrics = calculate_metrics(
        y_test,
        existing_probabilities,
        ALERT_THRESHOLD,
    )

    sensor_feature_columns = [
        column
        for column in feature_columns
        if column in SENSOR_FEATURE_COLUMNS
        or any(
            column.startswith(f"{sensor}_rolling_")
            for sensor in SENSOR_FEATURE_COLUMNS
        )
    ]
    if not sensor_feature_columns:
        raise ValueError("No sensor-derived features found in the model schema.")
    X_train_sensors_raw = data.loc[train_mask, sensor_feature_columns]
    X_test_sensors_raw = data.loc[test_mask, sensor_feature_columns]
    sensor_medians = X_train_sensors_raw.median(axis=0, skipna=True)
    sensor_all_missing = sensor_medians[sensor_medians.isna()].index.tolist()
    if sensor_all_missing:
        sensor_medians.loc[sensor_all_missing] = 0.0
    sensor_statistics = tuple(float(value) for value in sensor_medians.to_numpy())
    X_train_sensors = impute_missing_features(
        X_train_sensors_raw,
        sensor_feature_columns,
        sensor_statistics,
    )
    X_test_sensors = impute_missing_features(
        X_test_sensors_raw,
        sensor_feature_columns,
        sensor_statistics,
    )
    sensor_model = LGBMClassifier(**MODEL_PARAMETERS)
    sensor_model.fit(X_train_sensors, y_train)
    sensor_probabilities = sensor_model.predict_proba(X_test_sensors)[:, 1]
    sensor_metrics = calculate_metrics(
        y_test,
        sensor_probabilities,
        ALERT_THRESHOLD,
    )

    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, CANDIDATE_DIR / "lightgbm_model.joblib")
    (CANDIDATE_DIR / "feature_columns.json").write_text(
        json.dumps(feature_columns, indent=2) + "\n",
        encoding="utf-8",
    )
    (CANDIDATE_DIR / "imputer_statistics.json").write_text(
        json.dumps(
            {
                "strategy": "median",
                "feature_columns": feature_columns,
                "statistics": list(imputer_statistics),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    predictions = data.loc[test_mask, ["machineID", "datetime"]].copy()
    predictions["actual_failure_24h"] = y_test
    predictions["candidate_probability"] = probabilities
    predictions["candidate_alert"] = probabilities >= ALERT_THRESHOLD
    predictions["existing_model_probability"] = existing_probabilities
    predictions["existing_model_alert"] = (
        existing_probabilities >= ALERT_THRESHOLD
    )
    predictions.to_csv(CANDIDATE_DIR / "holdout_predictions.csv", index=False)

    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "gain_importance": model.booster_.feature_importance(
                importance_type="gain"
            ),
            "split_importance": model.booster_.feature_importance(
                importance_type="split"
            ),
        }
    ).sort_values("gain_importance", ascending=False)
    importance.to_csv(CANDIDATE_DIR / "feature_importance.csv", index=False)

    sensor_candidate_dir = CANDIDATE_DIR / "sensor_only"
    sensor_candidate_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(sensor_model, sensor_candidate_dir / "lightgbm_model.joblib")
    (sensor_candidate_dir / "feature_columns.json").write_text(
        json.dumps(sensor_feature_columns, indent=2) + "\n",
        encoding="utf-8",
    )
    (sensor_candidate_dir / "imputer_statistics.json").write_text(
        json.dumps(
            {
                "strategy": "median",
                "feature_columns": sensor_feature_columns,
                "statistics": list(sensor_statistics),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    sensor_importance = pd.DataFrame(
        {
            "feature": sensor_feature_columns,
            "gain_importance": sensor_model.booster_.feature_importance(
                importance_type="gain"
            ),
            "split_importance": sensor_model.booster_.feature_importance(
                importance_type="split"
            ),
        }
    ).sort_values("gain_importance", ascending=False)
    sensor_importance.to_csv(
        sensor_candidate_dir / "feature_importance.csv",
        index=False,
    )

    report = {
        "model_type": "LightGBM binary classifier",
        "model_parameters": MODEL_PARAMETERS,
        "target": "failure recorded in the next 24 hours",
        "split_method": "chronological with 24-hour label-window embargo",
        "test_fraction_of_timestamps": TEST_FRACTION,
        "alert_threshold": ALERT_THRESHOLD,
        "dataset": {
            "eligible_rows": int(len(data)),
            "positive_rows": int(labels.sum()),
            "positive_rate": float(labels.mean()),
            "train_rows": int(train_mask.sum()),
            "train_positive_rows": int(y_train.sum()),
            "train_start": timestamps[train_mask].min().isoformat(),
            "train_end": timestamps[train_mask].max().isoformat(),
            "test_rows": int(test_mask.sum()),
            "test_positive_rows": int(y_test.sum()),
            "test_start": timestamps[test_mask].min().isoformat(),
            "test_end": timestamps[test_mask].max().isoformat(),
            "embargo_hours": HORIZON_HOURS,
            "all_missing_training_features_filled_with_zero": all_missing,
        },
        "candidate_metrics": candidate_metrics,
        "candidate_threshold_sensitivity": threshold_sensitivity(
            y_test, probabilities
        ),
        "existing_model_same_holdout_metrics": existing_metrics,
        "sensor_only_feature_count": len(sensor_feature_columns),
        "sensor_only_feature_columns": sensor_feature_columns,
        "sensor_only_metrics": sensor_metrics,
        "sensor_only_threshold_sensitivity": threshold_sensitivity(
            y_test, sensor_probabilities
        ),
        "notes": [
            "The existing deployed artifacts were not overwritten.",
            (
                "The sensor-only result is a separate baseline using raw and "
                "rolling sensor features only; it is not the deployed model."
            ),
            (
                "This is a retrospective benchmark on one historical dataset, "
                "not evidence of live-factory performance or reduced downtime."
            ),
            (
                "Class-weighted LightGBM probabilities are not calibrated "
                "failure frequencies."
            ),
        ],
    }
    (CANDIDATE_DIR / "training_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print(f"\nCandidate artifacts saved to: {CANDIDATE_DIR}")


if __name__ == "__main__":
    main()

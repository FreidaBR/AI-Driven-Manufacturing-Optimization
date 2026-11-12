"""
Train and export complete predictive maintenance models:
1. Binary failure classifier (predicts whether machine will fail in 24h)
2. Remaining Useful Life (RUL) regressor (predicts when failure may happen in hours)
3. Component failure classifier (predicts failure mode and component severity)
4. Baseline statistics for model explainability

All models are trained strictly on the real historical Azure PdM dataset using a chronological split.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
RAW_DATA_DIR = BASE_DIR / "data" / "raw" / "cleaned_dataset"
FEATURES_PARQUET = RAW_DATA_DIR / "manufacturing_features.parquet"
FEATURES_PATH = MODEL_DIR / "feature_columns.json"

CUTOFF_TIMESTAMP = pd.Timestamp("2015-10-19 11:00:00")
HORIZON_HOURS = 24


def main():
    print("Loading feature schema and dataset...")
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    df = pd.read_parquet(FEATURES_PARQUET)
    failures = pd.read_csv(RAW_DATA_DIR / "cleaned_failures.csv", parse_dates=["datetime"])
    machines = pd.read_csv(RAW_DATA_DIR / "cleaned_machines.csv")
    telemetry = pd.read_csv(RAW_DATA_DIR / "cleaned_telemetry.csv", parse_dates=["datetime"])

    print(f"Loaded {len(df):,} feature rows and {len(failures):,} failure events.")

    # 1. Compute exact ground truth time-to-failure (RUL in hours) and next failure component
    machine_ids = df["machineID"].to_numpy()
    datetimes = pd.to_datetime(df["datetime"]).to_numpy(dtype="datetime64[ns]")

    hours_to_failure = np.full(len(df), np.nan, dtype=np.float32)
    next_failure_comp = np.full(len(df), "none", dtype=object)

    for mid, group in failures.groupby("machineID"):
        mask = machine_ids == mid
        idx_m = np.flatnonzero(mask)
        times_m = datetimes[idx_m]

        f_times = group["datetime"].to_numpy(dtype="datetime64[ns]")
        f_comps = group["failure"].to_numpy()

        ins = np.searchsorted(f_times, times_m, side="right")
        valid = ins < len(f_times)

        v_idx = idx_m[valid]
        v_ins = ins[valid]

        diff_h = (f_times[v_ins] - times_m[valid]).astype("timedelta64[m]").astype(float) / 60.0
        hours_to_failure[v_idx] = diff_h
        next_failure_comp[v_idx] = f_comps[v_ins]

    # Target RUL capped at 168 hours (7 days) for practical maintenance planning
    rul_target = np.clip(hours_to_failure, 0, 168.0)
    rul_target[np.isnan(rul_target)] = 168.0

    # Chronological partition
    ts = pd.to_datetime(df["datetime"])
    train_mask = ts < (CUTOFF_TIMESTAMP - pd.Timedelta(hours=HORIZON_HOURS))
    test_mask = ts >= CUTOFF_TIMESTAMP

    X_train = df.loc[train_mask, feature_columns]
    y_train_binary = df.loc[train_mask, "failure_next_24h"].to_numpy(dtype=np.int8)

    X_test = df.loc[test_mask, feature_columns]
    y_test_binary = df.loc[test_mask, "failure_next_24h"].to_numpy(dtype=np.int8)

    # Imputer statistics (median values on train)
    medians = X_train.median(axis=0, skipna=True)
    medians = medians.fillna(0.0)
    imputer_stats = tuple(float(val) for val in medians.to_numpy())

    X_train_imputed = X_train.fillna(medians)
    X_test_imputed = X_test.fillna(medians)

    # Compute baseline statistics for nominal (healthy) machine states for explainability
    healthy_train = X_train_imputed[y_train_binary == 0]
    baseline_stats = {
        col: {
            "mean": float(healthy_train[col].mean()),
            "std": float(max(healthy_train[col].std(), 1e-6)),
            "median": float(healthy_train[col].median()),
            "q25": float(healthy_train[col].quantile(0.25)),
            "q75": float(healthy_train[col].quantile(0.75)),
        }
        for col in feature_columns
    }

    # Add physical temperature baseline
    # T = 20.0 + 0.12*volt + 0.04*(rotate/10.0) + 0.25*vibration + 0.08*pressure
    healthy_volt = healthy_train["volt"]
    healthy_rot = healthy_train["rotate"]
    healthy_vib = healthy_train["vibration"]
    healthy_press = healthy_train["pressure"]
    healthy_temp = 20.0 + 0.12 * healthy_volt + 0.04 * (healthy_rot / 10.0) + 0.25 * healthy_vib + 0.08 * healthy_press
    baseline_stats["temperature"] = {
        "mean": float(healthy_temp.mean()),
        "std": float(healthy_temp.std()),
        "median": float(healthy_temp.median()),
        "q25": float(healthy_temp.quantile(0.25)),
        "q75": float(healthy_temp.quantile(0.75)),
    }

    with open(MODEL_DIR / "sensor_baseline_statistics.json", "w", encoding="utf-8") as f:
        json.dump(baseline_stats, f, indent=2)
    print("Saved sensor_baseline_statistics.json")

    # ----------------------------------------------------
    # MODEL 1: Binary Failure Classifier (Likelihood in 24h)
    # ----------------------------------------------------
    print("\n[1/3] Training Binary Failure Classification Model (LightGBM)...")
    clf = LGBMClassifier(
        objective="binary",
        class_weight="balanced",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    clf.fit(X_train_imputed, y_train_binary)
    probs_test = clf.predict_proba(X_test_imputed)[:, 1]

    roc_auc = float(roc_auc_score(y_test_binary, probs_test))
    avg_prec = float(average_precision_score(y_test_binary, probs_test))
    preds_test_98 = probs_test >= 0.98
    prec_98 = float(precision_score(y_test_binary, preds_test_98, zero_division=0))
    rec_98 = float(recall_score(y_test_binary, preds_test_98, zero_division=0))
    f1_98 = float(f1_score(y_test_binary, preds_test_98, zero_division=0))

    print(f"Binary Model: ROC-AUC={roc_auc:.4f}, PR-AUC={avg_prec:.4f}, Precision@0.98={prec_98:.4f}, Recall@0.98={rec_98:.4f}, F1={f1_98:.4f}")
    joblib.dump(clf, MODEL_DIR / "lightgbm_model.joblib")
    print("Exported lightgbm_model.joblib")

    # ----------------------------------------------------
    # MODEL 2: Remaining Useful Life (RUL) Regressor ("When failure may happen")
    # ----------------------------------------------------
    print("\n[2/3] Training RUL Regressor (LightGBM)...")
    y_train_rul = rul_target[train_mask]
    y_test_rul = rul_target[test_mask]

    # Sample active degradation signals and general operations
    sample_mask = (
        (y_train_rul <= 72)
        | (y_train_binary == 1)
        | (np.random.RandomState(42).rand(len(y_train_rul)) < 0.15)
    )

    rul_reg = LGBMRegressor(
        objective="regression",
        n_estimators=200,
        learning_rate=0.06,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    rul_reg.fit(X_train_imputed[sample_mask], y_train_rul[sample_mask])
    pred_rul_test = np.clip(rul_reg.predict(X_test_imputed), 1.0, 168.0)

    urgent_test_mask = y_test_rul <= 24
    rul_mae_urgent = float(mean_absolute_error(y_test_rul[urgent_test_mask], pred_rul_test[urgent_test_mask]))
    print(f"RUL Regressor: MAE in urgent window (<=24h) = {rul_mae_urgent:.2f} hours")
    joblib.dump(rul_reg, MODEL_DIR / "rul_model.joblib")
    print("Exported rul_model.joblib")

    # ----------------------------------------------------
    # MODEL 3: Component Failure Mode Classifier ("How severe the risk is")
    # ----------------------------------------------------
    print("\n[3/3] Training Component Failure Mode Classifier (LightGBM)...")
    # Train on instances where failure occurs within 48h to recognize component failure signatures
    fail_window_train = (hours_to_failure[train_mask] <= 48) & (hours_to_failure[train_mask] > 0)
    fail_window_test = (hours_to_failure[test_mask] <= 48) & (hours_to_failure[test_mask] > 0)

    X_train_comp = X_train_imputed[fail_window_train]
    y_train_comp = next_failure_comp[train_mask][fail_window_train]

    X_test_comp = X_test_imputed[fail_window_test]
    y_test_comp = next_failure_comp[test_mask][fail_window_test]

    comp_clf = LGBMClassifier(
        objective="multiclass",
        n_estimators=150,
        learning_rate=0.06,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    comp_clf.fit(X_train_comp, y_train_comp)
    preds_comp = comp_clf.predict(X_test_comp)
    comp_acc = float(accuracy_score(y_test_comp, preds_comp))
    print(f"Component Classifier Accuracy = {comp_acc:.4f}")
    joblib.dump(comp_clf, MODEL_DIR / "component_model.joblib")
    print("Exported component_model.joblib")

    # Update model metadata
    metadata = {
        "model_type": "Multi-Model Predictive Maintenance Suite",
        "models": {
            "failure_likelihood": {
                "algorithm": "LightGBM Classifier",
                "target": "Failure within next 24 hours",
                "roc_auc": round(roc_auc, 5),
                "pr_auc": round(avg_prec, 5),
                "precision_at_98": round(prec_98, 4),
                "recall_at_98": round(rec_98, 4),
                "f1_at_98": round(f1_98, 4),
            },
            "time_to_failure_rul": {
                "algorithm": "LightGBM Regressor",
                "target": "Remaining Useful Life (hours to next failure event)",
                "mae_urgent_window_hours": round(rul_mae_urgent, 2),
                "prediction_horizon": "1 to 168 hours",
            },
            "failure_severity_component": {
                "algorithm": "LightGBM Multiclass Classifier",
                "target": "Failing component (comp1, comp2, comp3, comp4)",
                "accuracy": round(comp_acc, 4),
                "component_downtime_hours": {
                    "comp1": 4.0,
                    "comp2": 8.0,
                    "comp3": 6.0,
                    "comp4": 12.0,
                },
                "component_risk_weights": {
                    "comp1": 0.45,
                    "comp2": 0.85,
                    "comp3": 0.65,
                    "comp4": 1.00,
                },
            },
        },
        "dataset": "Azure Predictive Maintenance Dataset (100 machines, 876,100 hourly records)",
        "feature_count": len(feature_columns),
        "split_method": "Chronological embargoed split at 2015-10-19 11:00:00",
    }

    with open(MODEL_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print("Saved model_metadata.json")

    # Update imputer_statistics.json
    with open(MODEL_DIR / "imputer_statistics.json", "w", encoding="utf-8") as f:
        json.dump(list(imputer_stats), f)
    print("Saved imputer_statistics.json")
    print("\nAll models trained and exported successfully!")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.evaluation_data import load_labeled_evaluation_data


BACKEND_DIR = Path(__file__).resolve().parent
DATA_DIR = BACKEND_DIR / "data" / "raw" / "cleaned_dataset"
MODEL_DIR = BACKEND_DIR / "models"
OUTPUT_DIR = BACKEND_DIR / "evaluation_results"
REPORT_PATH = MODEL_DIR / "retrained_candidate" / "training_report.json"
DEFAULT_CUTOFF = pd.Timestamp("2015-10-19 11:00:00")
DEFAULT_THRESHOLD = 0.98
THRESHOLDS = (0.50, 0.85, 0.90, 0.95, 0.97, 0.98, 0.99)


def binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()
    specificity = float(tn / (tn + fp)) if tn + fp else 0.0
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "specificity": specificity,
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "predicted_alerts": int(predictions.sum()),
    }


def calibration_bins(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bin_count: int = 10,
) -> list[dict[str, Any]]:
    ordered = np.argsort(probabilities, kind="stable")
    bins = np.array_split(ordered, min(bin_count, len(ordered)))
    return [
        {
            "bin": index + 1,
            "samples": int(len(indices)),
            "mean_predicted_probability": float(probabilities[indices].mean()),
            "observed_failure_rate": float(labels[indices].mean()),
        }
        for index, indices in enumerate(bins)
        if len(indices)
    ]


def _svg_polyline(
    title: str,
    points: list[tuple[float, float]],
    x_label: str,
    y_label: str,
    reference_diagonal: bool = False,
) -> str:
    width, height = 720, 470
    left, right, top, bottom = 78, 28, 54, 72
    plot_width = width - left - right
    plot_height = height - top - bottom

    def coords(x: float, y: float) -> tuple[float, float]:
        return left + x * plot_width, top + (1 - y) * plot_height

    path = " ".join(
        f"{coords(x, y)[0]:.1f},{coords(x, y)[1]:.1f}" for x, y in points
    )
    diagonal = ""
    if reference_diagonal:
        x1, y1 = coords(0, 0)
        x2, y2 = coords(1, 1)
        diagonal = (
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            'stroke="#8291a7" stroke-dasharray="6 5"/>'
        )
    x0, y0 = coords(0, 0)
    x1, y1 = coords(1, 1)
    grid = []
    for tick in range(6):
        value = tick / 5
        x_tick, y_tick = coords(value, value)
        grid.append(
            f'<line x1="{left}" y1="{y_tick:.1f}" x2="{width-right}" '
            f'y2="{y_tick:.1f}" stroke="#e1e7ef"/>'
            f'<text x="{left-12}" y="{y_tick+4:.1f}" text-anchor="end">'
            f"{value:.1f}</text>"
            f'<line x1="{x_tick:.1f}" y1="{top}" x2="{x_tick:.1f}" '
            f'y2="{height-bottom}" stroke="#f0f3f7"/>'
            f'<text x="{x_tick:.1f}" y="{height-bottom+22}" '
            f'text-anchor="middle">{value:.1f}</text>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        f'<text x="{width/2}" y="28" text-anchor="middle" '
        'font-family="Arial" font-size="18" font-weight="bold">'
        f"{title}</text>"
        '<g font-family="Arial" font-size="11" fill="#526176">'
        f'{"".join(grid)}{diagonal}'
        f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" '
        'stroke="#1d66cc" stroke-width="2.5" fill="none"/>'
        f'<polyline points="{path}" fill="none" stroke="#1d66cc" '
        'stroke-width="2.5"/>'
        f'<text x="{left + plot_width/2}" y="{height-18}" '
        f'text-anchor="middle">{x_label}</text>'
        f'<text x="20" y="{top + plot_height/2}" '
        f'transform="rotate(-90 20 {top + plot_height/2})" '
        f'text-anchor="middle">{y_label}</text></g></svg>'
    )


def _write_curves(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: list[dict[str, Any]],
) -> None:
    from sklearn.metrics import precision_recall_curve, roc_curve

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, probabilities)
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    roc_points = list(zip(false_positive_rate, true_positive_rate))
    pr_points = list(zip(recall, precision))
    calibration_points = [
        (
            item["mean_predicted_probability"],
            item["observed_failure_rate"],
        )
        for item in bins
    ]
    for filename, title, points, x_label, y_label, diagonal in [
        (
            "roc_curve.svg",
            "Chronological holdout ROC curve",
            roc_points,
            "False positive rate",
            "True positive rate",
            True,
        ),
        (
            "precision_recall_curve.svg",
            "Chronological holdout precision-recall curve",
            pr_points,
            "Recall",
            "Precision",
            False,
        ),
        (
            "calibration_curve.svg",
            "Holdout reliability (equal-count bins)",
            calibration_points,
            "Mean predicted probability",
            "Observed failure rate",
            True,
        ),
    ]:
        (OUTPUT_DIR / filename).write_text(
            _svg_polyline(title, points, x_label, y_label, diagonal),
            encoding="utf-8",
        )


def run_evaluation() -> dict[str, Any]:
    with REPORT_PATH.open(encoding="utf-8") as file:
        training_report = json.load(file)
    split = training_report["dataset"]
    cutoff = pd.Timestamp(split["test_start"])
    embargo_hours = int(split["embargo_hours"])
    embargo_start = cutoff - pd.Timedelta(hours=embargo_hours)

    features, labels = load_labeled_evaluation_data(
        DATA_DIR,
        MODEL_DIR / "feature_columns.json",
        DATA_DIR / "manufacturing_features.parquet",
    )
    features["datetime"] = pd.to_datetime(features["datetime"], errors="raise")
    train_mask = features["datetime"] < embargo_start
    test_mask = features["datetime"] >= cutoff
    test_features = features.loc[test_mask].reset_index(drop=True)
    y_test = np.asarray(labels)[test_mask.to_numpy()]
    if not train_mask.any() or not test_mask.any():
        raise ValueError("The chronological holdout split is empty.")

    feature_columns = json.loads(
        (MODEL_DIR / "feature_columns.json").read_text(encoding="utf-8")
    )
    imputer = json.loads(
        (MODEL_DIR / "imputer_statistics.json").read_text(encoding="utf-8")
    )
    if feature_columns != imputer["feature_columns"]:
        raise ValueError("Feature and imputer schemas are inconsistent.")
    if len(feature_columns) != 47 or len(imputer["statistics"]) != 47:
        raise ValueError("Expected the saved 47-feature inference schema.")

    model = joblib.load(MODEL_DIR / "lightgbm_model.joblib")
    if hasattr(model, "n_features_in_") and model.n_features_in_ != len(feature_columns):
        raise ValueError("Saved model input width does not match the feature schema.")
    matrix = test_features[feature_columns].to_numpy(dtype=np.float64)
    matrix[~np.isfinite(matrix)] = np.nan
    statistics = np.asarray(imputer["statistics"], dtype=np.float64)
    missing_rows, missing_columns = np.where(np.isnan(matrix))
    matrix[missing_rows, missing_columns] = statistics[missing_columns]
    probabilities = model.predict_proba(matrix)[:, 1]

    default_metrics = binary_metrics(y_test, probabilities, DEFAULT_THRESHOLD)
    thresholds = [
        binary_metrics(y_test, probabilities, threshold)
        for threshold in THRESHOLDS
    ]
    bins = calibration_bins(y_test, probabilities)
    expected = training_report["existing_model_same_holdout_metrics"]
    metric_names = {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "precision_at_threshold": "precision",
        "recall_at_threshold": "recall",
        "f1_at_threshold": "f1",
        "brier_score": "brier_score",
    }
    reference_comparison = {
        source_name: {
            "recorded": float(expected[source_name]),
            "reproduced": float(default_metrics[metric_name]),
            "absolute_difference": abs(
                float(expected[source_name]) - float(default_metrics[metric_name])
            ),
        }
        for source_name, metric_name in metric_names.items()
    }

    test_dates = test_features["datetime"]
    monthly = []
    for month, positions in test_dates.groupby(test_dates.dt.to_period("M")).groups.items():
        index = np.asarray(list(positions), dtype=int)
        monthly.append(
            {
                "month": str(month),
                "rows": int(len(index)),
                "positive_rows": int(y_test[index].sum()),
                "metrics": binary_metrics(
                    y_test[index],
                    probabilities[index],
                    DEFAULT_THRESHOLD,
                ),
            }
        )

    machine_rows = []
    for machine_id, positions in test_features.groupby("machineID").groups.items():
        index = np.asarray(list(positions), dtype=int)
        machine_labels = y_test[index]
        machine_probabilities = probabilities[index]
        metrics = {
            "machineID": int(machine_id),
            "rows": int(len(index)),
            "positive_rows": int(machine_labels.sum()),
            "precision": float(
                precision_score(
                    machine_labels,
                    machine_probabilities >= DEFAULT_THRESHOLD,
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    machine_labels,
                    machine_probabilities >= DEFAULT_THRESHOLD,
                    zero_division=0,
                )
            ),
        }
        if np.unique(machine_labels).size == 2:
            metrics["roc_auc"] = float(
                roc_auc_score(machine_labels, machine_probabilities)
            )
        else:
            metrics["roc_auc"] = None
        machine_rows.append(metrics)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(thresholds).to_csv(
        OUTPUT_DIR / "threshold_comparison.csv",
        index=False,
    )
    pd.DataFrame(monthly).to_csv(
        OUTPUT_DIR / "temporal_subperiod_metrics.csv",
        index=False,
    )
    pd.DataFrame(machine_rows).to_csv(
        OUTPUT_DIR / "per_machine_holdout_metrics.csv",
        index=False,
    )
    pd.DataFrame(bins).to_csv(
        OUTPUT_DIR / "calibration_bins.csv",
        index=False,
    )
    _write_curves(y_test, probabilities, bins)

    report = {
        "evaluation_type": "reproduced chronological holdout; no model retraining",
        "target": "failure recorded in the next 24 hours",
        "model": "existing deployed LightGBM binary classifier",
        "feature_count": len(feature_columns),
        "feature_order_matches_saved_schema": True,
        "cutoff": cutoff.isoformat(),
        "embargo_start_exclusive": embargo_start.isoformat(),
        "embargo_hours": embargo_hours,
        "training_rows_not_rescored": int(train_mask.sum()),
        "training_positive_rows": int(np.asarray(labels)[train_mask.to_numpy()].sum()),
        "holdout_rows": int(len(test_features)),
        "holdout_positive_rows": int(y_test.sum()),
        "holdout_positive_rate": float(y_test.mean()),
        "holdout_start": test_dates.min().isoformat(),
        "holdout_end": test_dates.max().isoformat(),
        "unique_machines_in_holdout": int(test_features["machineID"].nunique()),
        "machine_split": (
            "The same machine IDs occur in training and holdout; this is not "
            "an unseen-machine generalization test."
        ),
        "threshold_metrics": thresholds,
        "default_threshold_metrics": default_metrics,
        "calibration": {
            "brier_score": default_metrics["brier_score"],
            "log_loss": default_metrics["log_loss"],
            "expected_calibration_error_equal_count": float(
                sum(
                    item["samples"]
                    / len(y_test)
                    * abs(
                        item["observed_failure_rate"]
                        - item["mean_predicted_probability"]
                    )
                    for item in bins
                )
            ),
            "bins": bins,
            "interpretation": (
                "Retrospective reliability analysis only; no calibration "
                "transform was fitted."
            ),
        },
        "threshold_comparison": "threshold_comparison.csv",
        "temporal_subperiods": monthly,
        "per_machine_holdout_metrics": "per_machine_holdout_metrics.csv",
        "recorded_training_report_comparison": reference_comparison,
        "software": {
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "limitations": [
            "Training performance is not independently recomputed.",
            "Full-dataset metrics are intentionally not reported as holdout evidence.",
            "Machine-level slices are on seen machines, not a held-out-machine test.",
            "This is a synthetic historical benchmark, not live industrial validation.",
        ],
    }
    (OUTPUT_DIR / "temporal_holdout_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + os.linesep,
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    results = run_evaluation()
    print(json.dumps(results, indent=2))

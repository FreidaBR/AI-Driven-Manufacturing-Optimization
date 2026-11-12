"""
Feature engineering pipeline for the AI-Driven Manufacturing Optimization project.

Reproduces the feature construction used in the Kaggle notebook:
- Current sensor readings and rolling statistics (3, 12, 24 observations)
- Strictly prior error-event history (3, 12, 24 hours)
- Strictly prior maintenance-event history (24 hours, 7 days, 30 days)
- Time since most recent prior error / maintenance
- Most recently maintained component, excluding exact-timestamp events
- One-hot encoding aligned to the exported 47-feature model schema

This module creates features only. It does not create future-failure labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SENSOR_COLUMNS = ["volt", "rotate", "pressure", "vibration"]
ROLLING_WINDOWS = [3, 12, 24]

RAW_FILES = {
    "telemetry": "cleaned_telemetry.csv",
    "errors": "cleaned_errors.csv",
    "maintenance": "cleaned_maintenance.csv",
    "failures": "cleaned_failures.csv",
    "machines": "cleaned_machines.csv",
}

REQUIRED_COLUMNS = {
    "telemetry": {"datetime", "machineID", *SENSOR_COLUMNS},
    "errors": {"datetime", "machineID", "errorID"},
    "maintenance": {"datetime", "machineID", "comp"},
    "machines": {"machineID", "model", "age"},
}


def load_raw_data(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load and validate the cleaned CSV files."""
    data_dir = Path(data_dir)

    data: dict[str, pd.DataFrame] = {}
    for name, filename in RAW_FILES.items():
        path = data_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required dataset not found: {path}")

        # Parse datetime only for event/sensor tables that contain it.
        if name in {"telemetry", "errors", "maintenance", "failures"}:
            df = pd.read_csv(path, parse_dates=["datetime"])
        else:
            df = pd.read_csv(path)

        required = REQUIRED_COLUMNS.get(name)
        if required:
            missing = required.difference(df.columns)
            if missing:
                raise ValueError(
                    f"{filename} is missing required columns: {sorted(missing)}"
                )

        data[name] = df

    # Normalize timestamps and identifiers consistently.
    for name in ("telemetry", "errors", "maintenance", "failures"):
        data[name]["datetime"] = pd.to_datetime(
            data[name]["datetime"], errors="coerce"
        )
        data[name] = data[name].dropna(
            subset=["datetime", "machineID"]
        ).copy()
        data[name]["machineID"] = pd.to_numeric(
            data[name]["machineID"], errors="raise"
        ).astype("int32")

    data["machines"]["machineID"] = pd.to_numeric(
        data["machines"]["machineID"], errors="raise"
    ).astype("int32")

    if data["machines"]["machineID"].duplicated().any():
        raise ValueError("machines.csv contains duplicate machineID values.")

    return data


def _add_event_history_features(
    base: pd.DataFrame,
    event_df: pd.DataFrame,
    prefix: str,
    windows_hours: Iterable[int],
) -> pd.DataFrame:
    """
    Add event counts and recency using strictly prior events.

    For a snapshot at time t, events exactly at t are excluded.
    Count windows use (t - window, t), matching the notebook's
    searchsorted(side='left') implementation for the upper boundary.
    """
    result = base.copy()
    windows_hours = list(windows_hours)

    for window in windows_hours:
        result[f"{prefix}_count_{window}h"] = np.zeros(
            len(result), dtype=np.int32
        )
    result[f"{prefix}_time_since_last_hours"] = np.nan

    event_groups = {
        machine_id: np.sort(
            group["datetime"].to_numpy(dtype="datetime64[ns]")
        )
        for machine_id, group in event_df.groupby("machineID", sort=False)
    }

    for machine_id, row_indices in result.groupby(
        "machineID", sort=False
    ).groups.items():
        event_times = event_groups.get(machine_id)
        if event_times is None or len(event_times) == 0:
            continue

        idx = np.asarray(list(row_indices), dtype=np.int64)
        current_times = result.loc[idx, "datetime"].to_numpy(
            dtype="datetime64[ns]"
        )

        # Strictly before current snapshot: exclude events at exactly t.
        right_now = np.searchsorted(
            event_times, current_times, side="left"
        )

        for window in windows_hours:
            lower_bound = current_times - np.timedelta64(window, "h")
            right_lower = np.searchsorted(
                event_times, lower_bound, side="right"
            )
            counts = right_now - right_lower
            result.loc[idx, f"{prefix}_count_{window}h"] = counts.astype(
                np.int32
            )

        has_previous = right_now > 0
        elapsed = np.full(len(idx), np.nan, dtype=np.float64)

        if has_previous.any():
            last_event_times = event_times[right_now[has_previous] - 1]
            elapsed[has_previous] = (
                current_times[has_previous] - last_event_times
            ) / np.timedelta64(1, "h")

        result.loc[idx, f"{prefix}_time_since_last_hours"] = elapsed

    return result


def build_engineered_features(
    data: dict[str, pd.DataFrame],
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Build the 47 model features for every telemetry snapshot.

    Returns a DataFrame containing machineID, datetime, and the exact
    model feature columns in the exported model's order.
    """
    telemetry = data["telemetry"].copy()
    errors = data["errors"].copy()
    maintenance = data["maintenance"].copy()
    machines = data["machines"].copy()

    # Attach static machine information; prevent accidental row multiplication.
    integrated = telemetry.merge(
        machines[["machineID", "model", "age"]],
        on="machineID",
        how="left",
        validate="many_to_one",
    )

    # Match notebook order and rolling semantics.
    integrated = integrated.sort_values(
        ["machineID", "datetime"]
    ).reset_index(drop=True)

    for sensor in SENSOR_COLUMNS:
        grouped = integrated.groupby("machineID", sort=False)[sensor]

        for window in ROLLING_WINDOWS:
            integrated[f"{sensor}_rolling_mean_{window}h"] = (
                grouped.transform(
                    lambda s, w=window: s.rolling(
                        window=w, min_periods=1
                    ).mean()
                )
            )

            integrated[f"{sensor}_rolling_std_{window}h"] = (
                grouped.transform(
                    lambda s, w=window: s.rolling(
                        window=w, min_periods=2
                    ).std()
                ).fillna(0)
            )

    # Strictly prior event history, preserving notebook feature names.
    integrated = _add_event_history_features(
        integrated, errors, "error", [3, 12, 24]
    )

    integrated = _add_event_history_features(
        integrated, maintenance, "maintenance", [24, 24 * 7, 24 * 30]
    )

    integrated = integrated.rename(
        columns={
            "maintenance_count_168h": "maintenance_count_7d",
            "maintenance_count_720h": "maintenance_count_30d",
        }
    )

    # Find most recent maintenance strictly before each telemetry timestamp.
    maintenance_sorted = maintenance[
        ["machineID", "datetime", "comp"]
    ].rename(
        columns={
            "datetime": "last_maintenance_datetime",
            "comp": "last_maintenance_component",
        }
    )

    base_for_asof = integrated.sort_values(
        ["datetime", "machineID"]
    ).copy()
    maintenance_sorted = maintenance_sorted.sort_values(
        ["last_maintenance_datetime", "machineID"]
    )

    integrated = pd.merge_asof(
        base_for_asof,
        maintenance_sorted,
        left_on="datetime",
        right_on="last_maintenance_datetime",
        by="machineID",
        direction="backward",
        allow_exact_matches=False,
    )

    integrated = integrated.sort_values(
        ["machineID", "datetime"]
    ).reset_index(drop=True)

    # One-hot encode exactly as in the notebook, including NaN categories.
    categorical = [
        col
        for col in ["model", "last_maintenance_component"]
        if col in integrated.columns
    ]

    model_frame = integrated.drop(
        columns=["last_maintenance_datetime", "machineID", "datetime"],
        errors="ignore",
    ).copy()

    model_frame = pd.get_dummies(
        model_frame,
        columns=categorical,
        dummy_na=True,
        dtype="int8",
    )

    model_frame = model_frame.replace([np.inf, -np.inf], np.nan)

    for col in model_frame.columns:
        if model_frame[col].dtype == "object":
            model_frame[col] = pd.to_numeric(
                model_frame[col], errors="coerce"
            )

    # Reindex to the exact exported schema. Missing dummy columns become 0.
    missing_model_features = [
        col for col in feature_columns if col not in model_frame.columns
    ]
    extra_model_features = [
        col for col in model_frame.columns if col not in feature_columns
    ]

    if extra_model_features:
        unexpected = sorted(extra_model_features)
        raise ValueError(
            "Feature pipeline generated unexpected columns: "
            f"{unexpected}"
        )

    if missing_model_features:
        for col in missing_model_features:
            model_frame[col] = 0

    model_frame = model_frame.reindex(columns=feature_columns)

    # Keep identifiers outside the model matrix.
    output = integrated[["machineID", "datetime"]].copy()
    output = pd.concat(
        [output.reset_index(drop=True), model_frame.reset_index(drop=True)],
        axis=1,
    )

    if output[list(feature_columns)].shape[1] != 47:
        raise ValueError(
            f"Expected 47 features, got {output[list(feature_columns)].shape[1]}"
        )

    return output


def build_features_from_directory(
    data_dir: str | Path,
    feature_columns_path: str | Path,
) -> pd.DataFrame:
    """Convenience wrapper to load CSVs and build model-aligned features."""
    with open(feature_columns_path, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    if len(feature_columns) != 47:
        raise ValueError(
            f"Expected 47 exported model features, found {len(feature_columns)}"
        )

    data = load_raw_data(data_dir)
    return build_engineered_features(data, feature_columns)


if __name__ == "__main__":
    project_backend = Path(__file__).resolve().parents[1]
    raw_dir = project_backend / "data" / "raw" / "cleaned_dataset"
    feature_schema = project_backend / "models" / "feature_columns.json"

    features = build_features_from_directory(raw_dir, feature_schema)

    output_dir = project_backend / "data" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "manufacturing_features_local.parquet"

    features.to_parquet(output_path, index=False)

    print("Feature generation completed.")
    print("Rows:", len(features))
    print("Feature count:", len(features.columns) - 2)
    print("Output:", output_path)
    print("Feature columns:", len(features.columns) - 2)

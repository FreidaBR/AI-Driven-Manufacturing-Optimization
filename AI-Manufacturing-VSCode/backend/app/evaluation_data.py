from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.feature_pipeline import build_features_from_directory


HORIZON_HOURS = 24


def load_labeled_evaluation_data(
    raw_data_dir: str | Path,
    feature_columns_path: str | Path,
    generated_features_path: str | Path,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build the feature matrix and 24-hour labels from checked-in raw CSVs."""
    raw_data_dir = Path(raw_data_dir)
    generated_features_path = Path(generated_features_path)

    with open(feature_columns_path, "r", encoding="utf-8") as file:
        feature_columns = json.load(file)

    if generated_features_path.is_file():
        features = pd.read_parquet(generated_features_path)
    else:
        features = build_features_from_directory(
            raw_data_dir,
            feature_columns_path,
        )
        generated_features_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(generated_features_path, index=False)

    required_columns = ["machineID", "datetime", *feature_columns]
    missing_columns = set(required_columns).difference(features.columns)
    if missing_columns:
        raise ValueError(
            "Generated features are missing columns: "
            f"{sorted(missing_columns)}"
        )

    features["datetime"] = pd.to_datetime(
        features["datetime"],
        errors="coerce",
    )
    features = features.dropna(subset=["machineID", "datetime"])
    features = features.sort_values(
        ["machineID", "datetime"]
    ).reset_index(drop=True)

    failures = pd.read_csv(
        raw_data_dir / "cleaned_failures.csv",
        parse_dates=["datetime"],
    )
    failures["datetime"] = pd.to_datetime(
        failures["datetime"],
        errors="coerce",
    )
    failures = failures.dropna(subset=["machineID", "datetime"])
    failure_events = failures[["machineID", "datetime"]].drop_duplicates()

    machine_ids = features["machineID"].to_numpy()
    snapshot_times = features["datetime"].to_numpy(dtype="datetime64[ns]")
    failure_groups = {
        machine_id: group["datetime"].to_numpy(dtype="datetime64[ns]")
        for machine_id, group in failure_events.groupby("machineID")
    }

    labels = np.zeros(len(features), dtype=np.int8)
    eligible = np.zeros(len(features), dtype=bool)
    horizon = np.timedelta64(HORIZON_HOURS, "h")

    for machine_id in features["machineID"].unique():
        machine_positions = np.flatnonzero(machine_ids == machine_id)
        machine_times = snapshot_times[machine_positions]
        machine_end = machine_times.max()
        eligible_positions = machine_positions[
            machine_times + horizon <= machine_end
        ]
        eligible[eligible_positions] = True

        event_times = failure_groups.get(
            machine_id,
            np.array([], dtype="datetime64[ns]"),
        )
        if len(event_times) == 0:
            continue

        eligible_times = snapshot_times[eligible_positions]
        left = np.searchsorted(event_times, eligible_times, side="right")
        right = np.searchsorted(
            event_times,
            eligible_times + horizon,
            side="right",
        )
        labels[eligible_positions] = (right > left).astype(np.int8)

    evaluation_data = features.loc[eligible].copy()
    return evaluation_data, labels[eligible]

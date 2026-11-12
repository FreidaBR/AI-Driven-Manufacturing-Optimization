from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def load_imputer_statistics(
    path: str | Path,
    feature_columns: Sequence[str],
) -> tuple[float, ...]:
    """Load portable median-imputation values aligned to the model schema."""
    with open(path, "r", encoding="utf-8") as file:
        artifact = json.load(file)

    if artifact.get("strategy") != "median":
        raise ValueError("Imputer statistics must use the median strategy.")

    if artifact.get("feature_columns") != list(feature_columns):
        raise ValueError("Imputer statistics do not match the model features.")

    statistics = np.asarray(artifact.get("statistics"), dtype=np.float64)

    if statistics.shape != (len(feature_columns),):
        raise ValueError(
            f"Expected {len(feature_columns)} imputer statistics, "
            f"found {statistics.size}."
        )

    if not np.isfinite(statistics).all():
        raise ValueError("Imputer statistics must all be finite.")

    return tuple(float(value) for value in statistics)


def impute_missing_features(
    features: pd.DataFrame,
    feature_columns: Sequence[str],
    statistics: Sequence[float],
) -> pd.DataFrame:
    """Fill missing numeric feature values while preserving names and index."""
    columns = list(feature_columns)
    if list(features.columns) != columns:
        raise ValueError("Input features do not match the model feature schema.")

    medians = np.asarray(statistics, dtype=np.float64)
    if medians.shape != (len(columns),) or not np.isfinite(medians).all():
        raise ValueError("Imputer statistics do not match the model schema.")

    values = features.to_numpy(
        dtype=np.float64,
        na_value=np.nan,
        copy=True,
    )
    if np.isinf(values).any():
        raise ValueError("Input features must not contain infinite values.")

    missing = np.isnan(values)
    values[missing] = np.broadcast_to(medians, values.shape)[missing]

    return pd.DataFrame(values, columns=columns, index=features.index)

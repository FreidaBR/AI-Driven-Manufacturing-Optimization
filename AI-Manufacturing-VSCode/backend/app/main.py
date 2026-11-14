from pathlib import Path
from functools import lru_cache
import json
import logging
import math
import threading
from typing import Any

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from app.feature_pipeline import build_features_from_directory
from app.model_utils import impute_missing_features, load_imputer_statistics

logger = logging.getLogger(__name__)

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE = Path(__file__).resolve().parents[1]

DATA = BASE / "data" / "phase4_outputs"
RAW_DATA = BASE / "data" / "raw" / "cleaned_dataset"
GENERATED_DATA = BASE / "data" / "generated"

MODEL_DIR = BASE / "models"
MODEL_PATH = MODEL_DIR / "lightgbm_model.joblib"
RUL_MODEL_PATH = MODEL_DIR / "rul_model.joblib"
COMPONENT_MODEL_PATH = MODEL_DIR / "component_model.joblib"
IMPUTER_STATISTICS_PATH = MODEL_DIR / "imputer_statistics.json"
FEATURES_PATH = MODEL_DIR / "feature_columns.json"
BASELINES_PATH = MODEL_DIR / "sensor_baseline_statistics.json"

GENERATED_FEATURES_PATH = (
    GENERATED_DATA / "manufacturing_features_local.parquet"
)


# --------------------------------------------------
# COMPONENT FAILURE LABELS AND UNVALIDATED DURATION ASSUMPTIONS
# --------------------------------------------------

# Used only for planning/scenario estimates; these are not measured site durations.
COMPONENT_DOWNTIME_HOURS = {
    "comp1": 4.0,
    "comp2": 8.0,
    "comp3": 6.0,
    "comp4": 12.0,
}

COMPONENT_SEVERITY_WEIGHTS = {
    "comp1": 0.45,
    "comp2": 0.85,
    "comp3": 0.65,
    "comp4": 1.00,
}

COMPONENT_NAMES = {
    "comp1": "Electrical Inverter / Power Relay",
    "comp2": "Drive Shaft & Coupling",
    "comp3": "Pressure Valve & Pump Seal",
    "comp4": "Bearing Assembly & Spindle",
}

# --------------------------------------------------
# INDUSTRIAL SCHEDULING REALISM CONSTANTS
# --------------------------------------------------

# Illustrative scenario assumption, not a measured site throughput cost.
DOWNTIME_COST_PER_HOUR = 2500.0

SHIFT_CALENDAR = [
    {"shift_id": "SHIFT_1", "name": "Morning Shift", "hours": "06:00 - 14:00", "duration_hours": 8.0},
    {"shift_id": "SHIFT_2", "name": "Afternoon / Swing Shift", "hours": "14:00 - 22:00", "duration_hours": 8.0},
    {"shift_id": "SHIFT_3", "name": "Night Shift", "hours": "22:00 - 06:00", "duration_hours": 8.0},
]

TECHNICIAN_SKILLS = {
    "comp1": "Electrical Specialist (Level III - High Voltage & Inverters)",
    "comp2": "Mechanical Millwright (Torque Transmission & Couplings)",
    "comp3": "Hydraulics & Fluid Power Technician (Pumps & Valves)",
    "comp4": "Precision Vibration & Bearing Specialist",
}

SPARE_PARTS_INVENTORY = {
    "comp1": {
        "sku": "SKU-ELEC-RELAY-101",
        "description": "Solid State Inverter Relay & Board",
        "in_stock": 14,
        "location": "Warehouse A-12",
        "lead_time_days": 1,
    },
    "comp2": {
        "sku": "SKU-MECH-SHAFT-204",
        "description": "High-Torque Drive Shaft & Coupler Kit",
        "in_stock": 7,
        "location": "Warehouse B-04",
        "lead_time_days": 2,
    },
    "comp3": {
        "sku": "SKU-HYDR-VALVE-302",
        "description": "Pressure Regulator & Ceramic Pump Seal",
        "in_stock": 9,
        "location": "Warehouse A-08",
        "lead_time_days": 1,
    },
    "comp4": {
        "sku": "SKU-ROTR-BEAR-409",
        "description": "Heavy-Duty Dual-Row Tapered Roller Bearing Kit",
        "in_stock": 4,
        "location": "Central Depot C-01",
        "lead_time_days": 3,
    },
}

# Advisory project guidance; these bands are not ISO-certified diagnostic rules.
MAINTENANCE_GUIDANCE = {
    "CRITICAL": {
        "zone": "Critical risk band",
        "min_probability": 0.98,
        "max_rul_hours": 24.0,
        "action": "Prompt qualified inspection; follow site safety and shutdown procedures.",
    },
    "HIGH": {
        "zone": "High risk band",
        "min_probability": 0.85,
        "max_rul_hours": 72.0,
        "action": "Prioritize a qualified inspection within the available maintenance capacity.",
    },
    "MEDIUM": {
        "zone": "Medium risk band",
        "min_probability": 0.50,
        "max_rul_hours": 168.0,
        "action": "Plan a preventive inspection when capacity permits.",
    },
    "LOW": {
        "zone": "Low risk band",
        "min_probability": 0.00,
        "max_rul_hours": 9999.0,
        "action": "Continue routine monitoring; this prototype does not replace site procedures.",
    },
}

RISK_CONFIGURATION = {
    "operational_alert_threshold": 0.98,
    "risk_bands": [
        {"category": "CRITICAL", "minimum_probability": 0.98},
        {"category": "HIGH", "minimum_probability": 0.85},
        {"category": "MEDIUM", "minimum_probability": 0.50},
        {"category": "LOW", "minimum_probability": 0.0},
    ],
}

MAINTENANCE_CATEGORY_BY_RISK = {
    "CRITICAL": "URGENT",
    "HIGH": "PREVENTIVE",
    "MEDIUM": "PREVENTIVE",
    "LOW": "ROUTINE",
}


# --------------------------------------------------
# DASHBOARD CSV FILES
# --------------------------------------------------

FILES = {
    "priorities": "machine_priority_ranking.csv",
    "recommendations": "machine_component_recommendations.csv",
    "components": "historical_component_failure_profile.csv",
    "schedule": "maintenance_inspection_schedule.csv",
    "daily_workload": "daily_maintenance_workload.csv",
    "capacity": "maintenance_capacity_scenarios.csv",
    "failure_matches": "strict_24h_failure_matches.csv",
}

REQUIRED_CSV_COLUMNS = {
    "priorities": {"machineID", "priority"},
    "recommendations": {"machineID"},
    "components": {"model", "failure", "failure_count"},
    "schedule": {"machineID"},
    "daily_workload": {"allocation_label", "machines"},
    "capacity": {"machines_per_day", "machines_requiring_inspection"},
    "failure_matches": {"machineID", "failure_datetime"},
}


# --------------------------------------------------
# FASTAPI APP
# --------------------------------------------------

app = FastAPI(
    title="AI Manufacturing Optimization API",
    version="2.5.0",
    description=(
        "Predictive-maintenance prototype for historical benchmark replay. "
        "Predictions and schedule proposals are advisory; no live factory feed is connected."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# LOAD MODEL ARTIFACTS
# --------------------------------------------------

model = None
rul_model = None
component_model = None
baseline_statistics = {}
imputer_statistics = None
feature_columns = []
model_load_error = None

try:
    with open(FEATURES_PATH, "r", encoding="utf-8") as file:
        feature_columns = json.load(file)

    if len(feature_columns) != 47:
        raise ValueError(f"Expected 47 features, found {len(feature_columns)}")

    imputer_statistics = load_imputer_statistics(
        IMPUTER_STATISTICS_PATH,
        feature_columns,
    )
    model = joblib.load(MODEL_PATH)

    if RUL_MODEL_PATH.exists():
        rul_model = joblib.load(RUL_MODEL_PATH)

    if COMPONENT_MODEL_PATH.exists():
        component_model = joblib.load(COMPONENT_MODEL_PATH)

    if BASELINES_PATH.exists():
        with open(BASELINES_PATH, "r", encoding="utf-8") as f:
            baseline_statistics = json.load(f)

    print("All ML models and feature baselines loaded successfully.")
    print(f"Features: {len(feature_columns)}, RUL model: {rul_model is not None}, Component model: {component_model is not None}")

except Exception as error:
    model = None
    rul_model = None
    component_model = None
    imputer_statistics = None
    model_load_error = str(error)
    print(f"Model loading failed: {model_load_error}")


# --------------------------------------------------
# PREDICTION & SIMULATION SCHEMAS
# --------------------------------------------------

class PredictionRequest(BaseModel):
    features: dict[str, float | None]

    @field_validator("features")
    @classmethod
    def reject_non_finite_features(cls, features: dict[str, float | None]):
        if any(value is not None and not math.isfinite(value) for value in features.values()):
            raise ValueError("Feature values must be finite numbers or null.")
        return features


class SimulationStepRequest(BaseModel):
    machines_per_day: int = Field(default=5, ge=1, le=100)


def validate_machines_per_day(machines_per_day: int) -> int:
    if not 1 <= machines_per_day <= 100:
        raise HTTPException(
            status_code=422,
            detail="machines_per_day must be between 1 and 100.",
        )
    return machines_per_day


# --------------------------------------------------
# SIMULATION STATE
# --------------------------------------------------

simulation_lock = threading.Lock()
simulation_position = -1


def classify_risk(probability: float) -> str:
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("Failure probability must be finite and between 0 and 1.")
    for band in RISK_CONFIGURATION["risk_bands"]:
        if probability >= band["minimum_probability"]:
            return band["category"]
    return "LOW"


def maintenance_category_for_risk(risk_category: str) -> str:
    return MAINTENANCE_CATEGORY_BY_RISK[risk_category]


@lru_cache(maxsize=1)
def get_machine_models():
    meta = get_machine_metadata()
    return {mid: info["model"] for mid, info in meta.items()}


def load_required_csv(
    path: Path,
    dataset_name: str,
    required_columns: set[str],
) -> pd.DataFrame:
    if not path.is_file():
        logger.error("Required dataset is missing: %s", path)
        raise HTTPException(
            status_code=503,
            detail=f"Required dataset '{dataset_name}' is unavailable.",
        )
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeError) as error:
        logger.exception("Could not read dataset '%s' from %s", dataset_name, path)
        raise HTTPException(
            status_code=500,
            detail=f"Dataset '{dataset_name}' could not be read.",
        ) from error

    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        logger.error(
            "Dataset '%s' is missing required columns: %s",
            dataset_name,
            sorted(missing_columns),
        )
        raise HTTPException(
            status_code=500,
            detail={
                "message": f"Dataset '{dataset_name}' has an invalid schema.",
                "missing_columns": sorted(missing_columns),
            },
        )
    return frame


def read_csv(key: str) -> pd.DataFrame:
    if key not in FILES:
        raise HTTPException(status_code=404, detail="Unknown dataset.")
    return load_required_csv(
        DATA / FILES[key],
        key,
        REQUIRED_CSV_COLUMNS.get(key, set()),
    )


def to_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_json_safe(value.tolist())
    if isinstance(value, np.generic):
        return to_json_safe(value.item())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def optional_finite_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    return to_json_safe(df.astype(object).where(pd.notna(df), None).to_dict("records"))


def check_model():
    if model is None or imputer_statistics is None:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Prediction model or imputer statistics are unavailable.",
                "error": model_load_error,
            },
        )


# --------------------------------------------------
# SENSOR PHYSICS, HEALTH INDEX & PREDICTIVE HELPERS
# --------------------------------------------------

def calculate_temperature(volt: float | None, rotate: float | None, pressure: float | None, vibration: float | None) -> float:
    """
    Calculate the project's formula-derived temperature proxy in degrees Celsius.
    """
    v = volt if volt is not None and np.isfinite(volt) else 170.0
    r = rotate if rotate is not None and np.isfinite(rotate) else 446.0
    p = pressure if pressure is not None and np.isfinite(pressure) else 100.0
    vib = vibration if vibration is not None and np.isfinite(vibration) else 40.0

    temp = 20.0 + (0.12 * v) + (0.04 * (r / 10.0)) + (0.25 * vib) + (0.08 * p)
    return round(float(temp), 2)


def estimate_downtime_breakdown(component: str, priority: str) -> dict[str, Any]:
    """
    Calculate an illustrative duration and cost estimate from project assumptions.
    """
    active_replacement = COMPONENT_DOWNTIME_HOURS.get(component, 8.0)
    diagnostic_inspection = 1.0 if priority in ("CRITICAL", "HIGH") else 1.5
    testing_and_calibration = 1.0
    total_hours = round(diagnostic_inspection + active_replacement + testing_and_calibration, 1)

    bottleneck_multiplier = 1.5 if priority == "CRITICAL" else 1.0
    financial_loss = round(total_hours * DOWNTIME_COST_PER_HOUR * bottleneck_multiplier, 2)

    return {
        "total_downtime_hours": total_hours,
        "diagnostic_hours": diagnostic_inspection,
        "active_replacement_hours": active_replacement,
        "testing_calibration_hours": testing_and_calibration,
        "line_bottleneck_impact": "Not assessed; production dependency data is unavailable.",
        "estimated_financial_loss_usd": financial_loss,
        "cost_per_hour_rate_usd": DOWNTIME_COST_PER_HOUR,
        "estimate_basis": "Unvalidated component-duration and hourly-cost assumptions.",
    }


def calculate_general_ttf(predicted_rul_hours: float) -> dict[str, Any]:
    """Expose the RUL model output without inventing calibrated uncertainty bounds."""
    rul_hours = round(float(predicted_rul_hours), 1)
    return {
        "ttf_hours": rul_hours,
        "ttf_days": round(rul_hours / 24.0, 1),
        "confidence_interval_text": "No calibrated confidence interval is available.",
        "confidence_interval_calibrated": False,
    }


def calculate_composite_health_index(
    failure_probability: float,
    vibration_z: float,
    temperature_z: float,
    pressure_z: float,
    error_count_24h: int,
    runtime_since_maint_hours: float = 0.0,
) -> dict[str, Any]:
    """
    Calculate the project's 0-100 health index; it is not an ISO-validated standard.
    """
    prob_penalty = round(min(40.0, failure_probability * 40.0), 1)

    vib_excess = max(0.0, vibration_z - 1.0)
    temp_excess = max(0.0, temperature_z - 1.0)
    pres_excess = max(0.0, pressure_z - 1.0)
    sensor_penalty = round(min(30.0, (vib_excess * 7.0) + (temp_excess * 5.0) + (pres_excess * 3.5)), 1)

    error_penalty = round(min(18.0, error_count_24h * 3.6), 1)

    wear_ratio = min(1.0, max(0.0, runtime_since_maint_hours / (45.0 * 24.0)))
    wear_penalty = round(wear_ratio * 12.0, 1)

    total_penalties = round(prob_penalty + sensor_penalty + error_penalty + wear_penalty, 1)
    health_score = round(max(0.0, min(100.0, 100.0 - total_penalties)), 1)

    if health_score >= 85.0:
        grade = "Grade A (Optimal / Pristine)"
    elif health_score >= 70.0:
        grade = "Grade B (Fair / Nominal)"
    elif health_score >= 50.0:
        grade = "Grade C (Degraded / Inspection Required)"
    else:
        grade = "Grade D (Critical / Immediate Action)"

    return {
        "health_score": health_score,
        "health_grade": grade,
        "breakdown": {
            "risk_probability_penalty": prob_penalty,
            "sensor_anomaly_penalty": sensor_penalty,
            "error_burst_penalty": error_penalty,
            "maintenance_wear_penalty": wear_penalty,
            "total_penalties": total_penalties,
        },
    }


def calculate_health_score(
    failure_probability: float,
    vibration_z: float,
    temperature_z: float,
    pressure_z: float,
    error_count_24h: int,
) -> float:
    """Backward-compatible float wrapper for composite health score."""
    chi = calculate_composite_health_index(
        failure_probability, vibration_z, temperature_z, pressure_z, error_count_24h
    )
    return chi["health_score"]


def extract_model_contributions(
    imputed_features: pd.DataFrame,
    row_index: int = 0,
    precomputed: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    booster = getattr(model, "booster_", None)
    if booster is None and precomputed is None:
        return []
    try:
        contributions = np.asarray(precomputed, dtype=np.float64) if precomputed is not None else np.asarray(
            booster.predict(imputed_features, pred_contrib=True),
            dtype=np.float64,
        )
        if contributions.ndim == 1:
            contributions = contributions.reshape(1, -1)
        if (
            contributions.ndim != 2
            or contributions.shape[1] != len(feature_columns) + 1
            or not 0 <= row_index < contributions.shape[0]
        ):
            raise ValueError(
                "Unexpected LightGBM feature-contribution output shape: "
                f"{contributions.shape}"
            )
        values = imputed_features.iloc[row_index]
        ranked = [
            {
                "feature": name,
                "value": float(values[name]),
                "contribution_log_odds": float(contributions[row_index, index]),
                "direction": (
                    "increases"
                    if contributions[row_index, index] > 0
                    else "decreases"
                ),
            }
            for index, name in enumerate(feature_columns)
        ]
        return sorted(
            ranked,
            key=lambda item: abs(item["contribution_log_odds"]),
            reverse=True,
        )[:5]
    except Exception:
        logger.exception("Could not calculate LightGBM feature contributions.")
        return []


def generate_explanation(
    machine_id: int,
    probability: float,
    rul_hours: float,
    component: str,
    sensor_readings: dict[str, float],
    feature_row: dict[str, Any],
    model_contributions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    sensor_indicators = []
    vib = sensor_readings.get("vibration")
    temp = sensor_readings.get("temperature")
    pres = sensor_readings.get("pressure")

    vib_base = baseline_statistics.get("vibration", {"mean": 40.0, "std": 4.5})
    temp_base = baseline_statistics.get("temperature", {"mean": 70.0, "std": 5.0})
    pres_base = baseline_statistics.get("pressure", {"mean": 100.0, "std": 10.0})
    for sensor_name, sensor_value in sensor_readings.items():
        if sensor_value is None:
            sensor_indicators.append(
                f"{sensor_name} was not recorded; the model input was imputed from training statistics."
            )

    vib_z = (
        (vib - vib_base["mean"]) / max(0.1, vib_base["std"])
        if vib is not None
        else 0.0
    )
    temp_z = (
        (temp - temp_base["mean"]) / max(0.1, temp_base["std"])
        if temp is not None
        else 0.0
    )
    pres_z = (
        (pres - pres_base["mean"]) / max(0.1, pres_base["std"])
        if pres is not None
        else 0.0
    )

    if vib_z > 1.5:
        sensor_indicators.append(
            f"Vibration is {vib_z:.1f} standard deviations above its historical baseline."
        )
    if temp_z > 1.5:
        sensor_indicators.append(
            f"Derived temperature is {temp_z:.1f} standard deviations above its historical baseline."
        )
    if pres_z > 1.5:
        sensor_indicators.append(
            f"Pressure is {pres_z:.1f} standard deviations above its historical baseline."
        )

    err_value = optional_finite_float(feature_row.get("error_count_24h"))
    err_24h = int(err_value) if err_value is not None else 0
    if err_24h > 0:
        sensor_indicators.append(
            f"{err_24h} machine error log(s) were recorded in the prior 24 hours."
        )

    maint_hours = feature_row.get("maintenance_time_since_last_hours")
    if maint_hours is not None and pd.notna(maint_hours):
        maint_days = round(float(maint_hours) / 24.0, 1)
        if maint_days > 45:
            sensor_indicators.append(
                f"{maint_days} days have elapsed since the last recorded maintenance event."
            )

    if not sensor_indicators:
        sensor_indicators.append(
            "No configured sensor-deviation indicator exceeded its reporting threshold."
        )

    comp_name = COMPONENT_NAMES.get(component, component)
    downtime_hours = COMPONENT_DOWNTIME_HOURS.get(component, 8.0)
    risk_category = classify_risk(probability)
    guidance = MAINTENANCE_GUIDANCE[risk_category]
    maintenance_category = maintenance_category_for_risk(risk_category)
    summary = (
        f"Machine {machine_id}: the model estimates {probability * 100:.2f}% "
        f"probability of a recorded failure within 24 hours ({risk_category} risk; "
        f"{maintenance_category} recommendation). The component model predicts "
        f"{comp_name}; this is a model prediction, not a confirmed fault. "
        f"Estimated component downtime assumption: {downtime_hours:.1f} hours. "
        f"RUL model output: {rul_hours:.1f} hours."
    )

    return {
        "summary": summary,
        "key_drivers": sensor_indicators,
        "sensor_indicators": sensor_indicators,
        "vibration_z_score": round(float(vib_z), 2),
        "temperature_z_score": round(float(temp_z), 2),
        "pressure_z_score": round(float(pres_z), 2),
        "target_component": component,
        "target_component_name": comp_name,
        "downtime_risk_hours": downtime_hours,
        "model_attribution_method": (
            "LightGBM TreeSHAP (pred_contrib)"
            if model_contributions
            else "unavailable"
        ),
        "model_contributions": model_contributions or [],
        "attribution_note": (
            "Contributions are in raw model log-odds space; they are not causal effects."
        ),
        "maintenance_guidance": guidance,
    }


# --------------------------------------------------
# MULTI-MODEL PREDICTION LOGIC
# --------------------------------------------------

def predict_feature_row(feature_row: dict[str, Any]) -> dict[str, Any]:
    """
    Applies imputer, Binary Classifier, RUL Regressor, and Component Classifier
    to predict whether it fails, when it fails, how severe the risk is, health score, and explanation.
    """
    check_model()

    input_df = pd.DataFrame([feature_row], columns=feature_columns)
    input_imputed_df = impute_missing_features(
        input_df,
        feature_columns,
        imputer_statistics,
    )

    # 1. Whether machine will fail (Failure Probability in 24h)
    probability = float(model.predict_proba(input_imputed_df)[0, 1])
    risk_category = classify_risk(probability)
    maintenance_category = maintenance_category_for_risk(risk_category)

    # 2. When failure may happen (Remaining Useful Life in hours & general TTF)
    if rul_model is not None:
        predicted_rul = float(np.clip(rul_model.predict(input_imputed_df)[0], 1.0, 168.0))
    else:
        predicted_rul = round(max(2.0, (1.0 - probability) * 72.0), 1)

    general_ttf = calculate_general_ttf(predicted_rul)

    # 3. How severe the risk is & which component will fail
    if component_model is not None:
        predicted_component = str(component_model.predict(input_imputed_df)[0])
    else:
        predicted_component = "comp2"

    comp_weight = COMPONENT_SEVERITY_WEIGHTS.get(predicted_component, 0.75)
    downtime_details = estimate_downtime_breakdown(
        predicted_component,
        risk_category,
    )
    predicted_downtime = downtime_details["total_downtime_hours"]

    risk_severity_score = round(probability * 100.0, 1)
    guidance = MAINTENANCE_GUIDANCE[risk_category]
    if risk_category == "CRITICAL":
        schedule_window = f"Highest priority; predicted RUL {predicted_rul:.1f} hours"
    elif risk_category == "HIGH":
        schedule_window = "Prioritize the next available maintenance slot"
    elif risk_category == "MEDIUM":
        schedule_window = "Plan inspection within available maintenance capacity"
    else:
        schedule_window = "Routine monitoring"
    recommended_action = (
        f"{guidance['action']} The component model predicts {predicted_component}; "
        "verify the fault before replacement."
    )

    # Physics Sensor Readings & Runtime
    volt = optional_finite_float(feature_row.get("volt"))
    rotate = optional_finite_float(feature_row.get("rotate"))
    pressure = optional_finite_float(feature_row.get("pressure"))
    vibration = optional_finite_float(feature_row.get("vibration"))
    temperature = (
        calculate_temperature(volt, rotate, pressure, vibration)
        if all(
            value is not None and np.isfinite(value)
            for value in (volt, rotate, pressure, vibration)
        )
        else None
    )

    maint_time_hours = optional_finite_float(
        feature_row.get("maintenance_time_since_last_hours")
    )
    if maint_time_hours is None:
        maint_time_hours = optional_finite_float(
            input_imputed_df.iloc[0].get("maintenance_time_since_last_hours")
        ) or 0.0

    # Sensor z-scores for health score
    vib_base = baseline_statistics.get("vibration", {"mean": 40.0, "std": 4.5})
    temp_base = baseline_statistics.get("temperature", {"mean": 70.0, "std": 5.0})
    pres_base = baseline_statistics.get("pressure", {"mean": 100.0, "std": 10.0})

    vib_z = (vibration - vib_base["mean"]) / vib_base["std"] if vibration is not None else 0.0
    temp_z = (temperature - temp_base["mean"]) / temp_base["std"] if temperature is not None else 0.0
    pres_z = (pressure - pres_base["mean"]) / pres_base["std"] if pressure is not None else 0.0
    err_value = optional_finite_float(feature_row.get("error_count_24h"))
    if err_value is None:
        err_value = optional_finite_float(
            input_imputed_df.iloc[0].get("error_count_24h")
        )
    err_24h = int(err_value) if err_value is not None else 0

    chi = calculate_composite_health_index(
        probability, vib_z, temp_z, pres_z, err_24h, maint_time_hours
    )
    health_score = chi["health_score"]

    sensor_dict = {
        "volt": volt,
        "rotate": rotate,
        "pressure": pressure,
        "vibration": vibration,
        "temperature": temperature,
    }

    explanation = generate_explanation(
        machine_id=int(feature_row.get("machineID") or 0),
        probability=probability,
        rul_hours=predicted_rul,
        component=predicted_component,
        sensor_readings=sensor_dict,
        feature_row={**feature_row, "health_score": health_score},
        model_contributions=extract_model_contributions(input_imputed_df),
    )

    is_alert = probability >= RISK_CONFIGURATION["operational_alert_threshold"]

    return {
        "failure_probability_24h": round(probability, 6),
        "failure_probability_percentage": round(probability * 100.0, 2),
        "predicted_rul_hours": round(predicted_rul, 1),
        "general_time_to_failure": general_ttf,
        "predicted_downtime_hours": round(predicted_downtime, 1),
        "downtime_breakdown": downtime_details,
        "risk_severity_score": risk_severity_score,
        "component_severity_weight": comp_weight,
        "risk_severity_level": risk_category,
        "risk_category": risk_category,
        "health_score": health_score,
        "composite_health_index": chi,
        "maintenance_priority": risk_category,
        "maintenance_category": maintenance_category,
        "recommended_schedule": schedule_window,
        "recommended_action": recommended_action,
        "predicted_component": predicted_component,
        "predicted_component_name": COMPONENT_NAMES.get(predicted_component, predicted_component),
        "threshold": RISK_CONFIGURATION["operational_alert_threshold"],
        "alert": is_alert,
        "status": "ALERT" if is_alert else "NORMAL",
        "explanation": explanation,
        "model_contributions": explanation["model_contributions"],
        "maintenance_guidance": guidance,
        "sensor_readings": sensor_dict,
    }


# --------------------------------------------------
# FEATURE GENERATION & CACHING
# --------------------------------------------------

@lru_cache(maxsize=1)
def get_engineered_features() -> pd.DataFrame:
    """Generate or load the model-aligned feature matrix from historical data."""
    if not RAW_DATA.exists():
        logger.error("Raw dataset directory is missing: %s", RAW_DATA)
        raise HTTPException(
            status_code=503,
            detail="Required historical dataset directory is unavailable.",
        )

    primary_parquet = RAW_DATA / "manufacturing_features.parquet"
    try:
        if primary_parquet.exists():
            features = pd.read_parquet(primary_parquet)
        elif GENERATED_FEATURES_PATH.exists():
            features = pd.read_parquet(GENERATED_FEATURES_PATH)
        else:
            GENERATED_DATA.mkdir(parents=True, exist_ok=True)
            features = build_features_from_directory(RAW_DATA, FEATURES_PATH)
            features.to_parquet(GENERATED_FEATURES_PATH, index=False)
    except (OSError, ValueError, ImportError) as error:
        logger.exception("Could not load or generate engineered features.")
        raise HTTPException(
            status_code=503,
            detail="Historical features could not be loaded or generated.",
        ) from error

    required_columns = ["machineID", "datetime"] + feature_columns
    missing_columns = [col for col in required_columns if col not in features.columns]
    if missing_columns:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Generated feature data has an invalid schema.",
                "missing_columns": missing_columns,
            },
        )

    features["datetime"] = pd.to_datetime(features["datetime"], errors="coerce")
    features["machineID"] = pd.to_numeric(features["machineID"], errors="coerce")
    if features["datetime"].isna().any() or features["machineID"].isna().any():
        logger.error("Generated feature data contains invalid timestamps or machine IDs.")
        raise HTTPException(
            status_code=500,
            detail="Generated feature data contains invalid timestamps or machine IDs.",
        )
    if features.empty:
        raise HTTPException(status_code=503, detail="No historical feature records are available.")
    if features.duplicated(["datetime", "machineID"]).any():
        logger.error("Generated feature data has duplicate machine/timestamp rows.")
        raise HTTPException(
            status_code=500,
            detail="Generated feature data contains duplicate machine snapshots.",
        )
    features = features.sort_values(["datetime", "machineID"]).reset_index(drop=True)
    return features


@lru_cache(maxsize=1)
def get_simulation_timestamp_groups():
    features = get_engineered_features()
    grouped = features.groupby("datetime", sort=True)
    return list(grouped.indices.items())


@lru_cache(maxsize=1)
def get_machine_metadata():
    machines_path = RAW_DATA / "cleaned_machines.csv"
    df = load_required_csv(
        machines_path,
        "cleaned_machines",
        {"machineID", "model", "age"},
    )
    df["machineID"] = pd.to_numeric(df["machineID"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    if (
        df.empty
        or df[["machineID", "age"]].isna().any().any()
        or df["model"].isna().any()
        or not np.equal(df["machineID"] % 1, 0).all()
    ):
        logger.error("cleaned_machines.csv contains invalid machine IDs or ages.")
        raise HTTPException(
            status_code=500,
            detail="Machine metadata contains invalid IDs or ages.",
        )
    if df["machineID"].duplicated().any():
        raise HTTPException(
            status_code=500,
            detail="Machine metadata contains duplicate machine IDs.",
        )
    return {
        int(row["machineID"]): {
            "model": str(row["model"]),
            "age": int(row["age"]),
            "line": f"Line {((int(row['machineID']) - 1) // 20) + 1}",
        }
        for _, row in df.iterrows()
    }


@lru_cache(maxsize=1)
def get_historical_failures_and_downtime():
    failures_path = RAW_DATA / "cleaned_failures.csv"
    failures = load_required_csv(
        failures_path,
        "cleaned_failures",
        {"datetime", "machineID", "failure"},
    )
    failures["datetime"] = pd.to_datetime(failures["datetime"], errors="coerce")
    failures["machineID"] = pd.to_numeric(
        failures["machineID"],
        errors="coerce",
    )
    if failures[["datetime", "machineID"]].isna().any().any():
        logger.error("cleaned_failures.csv contains invalid timestamps or machine IDs.")
        raise HTTPException(
            status_code=500,
            detail="Failure history contains invalid timestamps or machine IDs.",
        )
    unknown_components = set(failures["failure"].dropna()).difference(COMPONENT_DOWNTIME_HOURS)
    if failures["failure"].isna().any() or unknown_components:
        logger.error(
            "Failure history contains unknown components: %s",
            sorted(map(str, unknown_components)),
        )
        raise HTTPException(
            status_code=500,
            detail="Failure history contains unknown component labels.",
        )
    failures["datetime_str"] = failures["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    failures["downtime_hours"] = failures["failure"].map(COMPONENT_DOWNTIME_HOURS)

    machine_downtime = {}
    for mid, group in failures.groupby("machineID"):
        machine_downtime[int(mid)] = {
            "failure_count": len(group),
            "total_downtime_hours": round(float(group["downtime_hours"].sum()), 1),
            "events": [
                {
                    "datetime": row["datetime_str"],
                    "failure_component": row["failure"],
                    "downtime_hours": row["downtime_hours"],
                }
                for _, row in group.iterrows()
            ],
        }

    return machine_downtime, failures


@lru_cache(maxsize=1)
def get_maintenance_events() -> pd.DataFrame:
    events = load_required_csv(
        RAW_DATA / "cleaned_maintenance.csv",
        "cleaned_maintenance",
        {"datetime", "machineID", "comp"},
    )
    events["datetime"] = pd.to_datetime(events["datetime"], errors="coerce")
    events["machineID"] = pd.to_numeric(events["machineID"], errors="coerce")
    if events[["datetime", "machineID"]].isna().any().any():
        logger.error("cleaned_maintenance.csv contains invalid timestamps or machine IDs.")
        raise HTTPException(
            status_code=500,
            detail="Maintenance history contains invalid timestamps or machine IDs.",
        )
    return events


def machines_with_maintenance_at(timestamp: pd.Timestamp) -> set[int]:
    events = get_maintenance_events()
    at_timestamp = events.loc[events["datetime"] == timestamp, "machineID"]
    return {int(machine_id) for machine_id in at_timestamp}


# --------------------------------------------------
# SIMULATION SNAPSHOT BUILDER
# --------------------------------------------------

def build_simulation_snapshot(position: int, machines_per_day: int):
    check_model()
    validate_machines_per_day(machines_per_day)
    timestamp_groups = get_simulation_timestamp_groups()

    if position < 0 or position >= len(timestamp_groups):
        raise HTTPException(
            status_code=404,
            detail="Simulation position is outside the available replay data.",
        )

    timestamp, row_indices = timestamp_groups[position]
    all_features = get_engineered_features()
    snapshot = all_features.iloc[row_indices].sort_values("machineID").reset_index(drop=True)

    machine_meta = get_machine_metadata()
    machine_downtime_dict, _ = get_historical_failures_and_downtime()

    # Vectorized / batch inference for speed
    snapshot_features = impute_missing_features(
        snapshot[feature_columns],
        feature_columns,
        imputer_statistics,
    )
    probabilities = model.predict_proba(snapshot_features)[:, 1]
    model_contributions = None
    try:
        model_contributions = model.booster_.predict(
            snapshot_features,
            pred_contrib=True,
        )
    except Exception:
        logger.exception("Could not calculate replay feature contributions.")

    if rul_model is not None:
        predicted_ruls = np.clip(rul_model.predict(snapshot_features), 1.0, 168.0)
    else:
        predicted_ruls = np.clip((1.0 - probabilities) * 72.0, 2.0, 168.0)

    if component_model is not None:
        predicted_comps = component_model.predict(snapshot_features)
    else:
        predicted_comps = np.array(["comp2"] * len(snapshot))

    vib_base = baseline_statistics.get("vibration", {"mean": 40.0, "std": 4.5})
    temp_base = baseline_statistics.get("temperature", {"mean": 70.0, "std": 5.0})
    pres_base = baseline_statistics.get("pressure", {"mean": 100.0, "std": 10.0})

    machines = []
    maintenance_at_snapshot = machines_with_maintenance_at(pd.Timestamp(timestamp))

    for idx, (row, prob, rul, comp) in enumerate(zip(snapshot.to_dict("records"), probabilities, predicted_ruls, predicted_comps)):
        machine_id = int(row["machineID"])
        prob = float(prob)
        rul = float(rul)
        comp = str(comp)

        meta = machine_meta.get(machine_id)
        if meta is None:
            raise HTTPException(
                status_code=500,
                detail=f"Machine {machine_id} is missing from machine metadata.",
            )
        dt_info = machine_downtime_dict.get(machine_id, {"failure_count": 0, "total_downtime_hours": 0.0, "events": []})

        # Physical Sensor Readings
        volt = optional_finite_float(row["volt"])
        rotate = optional_finite_float(row["rotate"])
        pressure = optional_finite_float(row["pressure"])
        vibration = optional_finite_float(row["vibration"])
        temperature = (
            calculate_temperature(volt, rotate, pressure, vibration)
            if all(
                value is not None and np.isfinite(value)
                for value in (volt, rotate, pressure, vibration)
            )
            else None
        )

        # Runtime
        elapsed_hours = max(0.0, (timestamp - pd.Timestamp("2015-01-01 06:00:00")).total_seconds() / 3600.0)
        total_runtime_hours = round((meta["age"] * 8760.0) + elapsed_hours, 1)
        recorded_runtime_since_maint = optional_finite_float(
            row.get("maintenance_time_since_last_hours")
        )
        runtime_since_maint_for_health = recorded_runtime_since_maint
        if runtime_since_maint_for_health is None:
            runtime_since_maint_for_health = optional_finite_float(
                snapshot_features.iloc[idx].get(
                    "maintenance_time_since_last_hours"
                )
            ) or 0.0

        # Health score & Composite Health Index
        vib_z = (
            (vibration - vib_base["mean"]) / vib_base["std"]
            if vibration is not None
            else 0.0
        )
        temp_z = (
            (temperature - temp_base["mean"]) / temp_base["std"]
            if temperature is not None
            else 0.0
        )
        pres_z = (
            (pressure - pres_base["mean"]) / pres_base["std"]
            if pressure is not None
            else 0.0
        )
        error_value = optional_finite_float(row.get("error_count_24h"))
        if error_value is None:
            error_value = optional_finite_float(
                snapshot_features.iloc[idx].get("error_count_24h")
            )
        err_24h = int(error_value) if error_value is not None else 0

        chi = calculate_composite_health_index(
            prob,
            vib_z,
            temp_z,
            pres_z,
            err_24h,
            runtime_since_maint_for_health,
        )
        health_score = chi["health_score"]

        comp_downtime = COMPONENT_DOWNTIME_HOURS.get(comp, 8.0)
        comp_weight = COMPONENT_SEVERITY_WEIGHTS.get(comp, 0.75)
        severity_score = round(prob * 100.0, 1)

        general_ttf = calculate_general_ttf(rul)
        priority = classify_risk(prob)
        maint_cat = maintenance_category_for_risk(priority)
        alert = prob >= RISK_CONFIGURATION["operational_alert_threshold"]
        guidance = MAINTENANCE_GUIDANCE[priority]

        if priority == "CRITICAL":
            action = f"Prioritize qualified inspection of model-predicted {comp}; verify before replacement."
            sched_window = f"Highest priority; predicted RUL {rul:.1f} hours"
        elif priority == "HIGH":
            action = f"Prioritize inspection of model-predicted {comp}; verify before replacement."
            sched_window = "Prioritize the next available maintenance slot"
        elif priority == "MEDIUM":
            action = f"Plan an inspection of model-predicted {comp}; verify before replacement."
            sched_window = "Plan inspection within available maintenance capacity"
        else:
            action = "Continue routine monitoring; no risk-band-triggered work order."
            sched_window = "Routine monitoring"

        downtime_details = estimate_downtime_breakdown(comp, priority)

        sensor_readings = {
            "volt": volt,
            "rotate": rotate,
            "pressure": pressure,
            "vibration": vibration,
            "temperature": temperature,
        }

        explanation = generate_explanation(
            machine_id=machine_id,
            probability=prob,
            rul_hours=rul,
            component=comp,
            sensor_readings=sensor_readings,
            feature_row={**row, "health_score": health_score},
            model_contributions=extract_model_contributions(
                snapshot_features,
                idx,
                model_contributions,
            ) if model_contributions is not None else [],
        )

        machine_models = get_machine_models()
        model_name = machine_models.get(machine_id, meta.get("model", "Unknown"))

        machines.append({
            "machineID": machine_id,
            "model": model_name,
            "age": meta["age"],
            "production_line": meta["line"],
            "snapshot_datetime": timestamp.isoformat(),
            "priority": priority,
            "risk_category": priority,
            "maintenance_category": maint_cat,
            "rank": 0,
            "failure_probability_24h": round(prob, 4),
            "predicted_rul_hours": round(rul, 1),
            "general_time_to_failure": general_ttf,
            "predicted_downtime_hours": round(comp_downtime, 1),
            "downtime_breakdown": downtime_details,
            "predicted_component": comp,
            "predicted_component_name": COMPONENT_NAMES.get(comp, comp),
            "risk_severity_score": severity_score,
            "component_severity_weight": comp_weight,
            "health_score": health_score,
            "composite_health_index": chi,
            "alert": alert,
            "recommended_schedule": sched_window,
            "recommended_action": action,
            "reason": explanation["summary"],
            "flagged_drivers": explanation["key_drivers"],
            "model_contributions": explanation["model_contributions"],
            "maintenance_guidance": guidance,
            "sensor_readings": sensor_readings,
            "runtime_hours": total_runtime_hours,
            "runtime_since_maintenance_hours": (
                round(recorded_runtime_since_maint, 1)
                if recorded_runtime_since_maint is not None
                else None
            ),
            "historical_failure_count": dt_info["failure_count"],
            "historical_total_downtime_hours": dt_info["total_downtime_hours"],
            "recent_error_count_24h": err_24h,
            "maintenance_status": (
                "MAINTENANCE_RECORDED_AT_SNAPSHOT"
                if machine_id in maintenance_at_snapshot
                else "NOT_RECORDED_AT_SNAPSHOT"
            ),
        })

    # Rank machines by failure probability and risk severity (who needs attention first)
    machines.sort(key=lambda m: (-m["failure_probability_24h"], -m["risk_severity_score"], m["predicted_rul_hours"]))
    for rank, m in enumerate(machines, start=1):
        m["rank"] = rank

    counts = {
        "CRITICAL": sum(m["priority"] == "CRITICAL" for m in machines),
        "HIGH": sum(m["priority"] == "HIGH" for m in machines),
        "MEDIUM": sum(m["priority"] == "MEDIUM" for m in machines),
        "LOW": sum(m["priority"] == "LOW" for m in machines),
    }

    urgent_count = sum(m["maintenance_category"] == "URGENT" for m in machines)
    preventive_count = sum(m["maintenance_category"] == "PREVENTIVE" for m in machines)
    routine_count = sum(m["maintenance_category"] == "ROUTINE" for m in machines)

    attention_machines = [
        machine
        for machine in machines
        if machine["priority"] != "LOW"
        and machine["maintenance_status"] != "MAINTENANCE_RECORDED_AT_SNAPSHOT"
    ]
    avg_health = round(float(np.mean([m["health_score"] for m in machines])), 1)

    schedule = []
    for index, machine in enumerate(attention_machines):
        day_offset = index // machines_per_day
        slot_in_day = index % machines_per_day
        shift_index = slot_in_day % len(SHIFT_CALENDAR)
        slot_start_hours = day_offset * 24 + shift_index * 8
        repair_duration = COMPONENT_DOWNTIME_HOURS.get(
            machine["predicted_component"],
            8.0,
        )
        feasible = (
            slot_start_hours + repair_duration
            <= machine["predicted_rul_hours"]
        )
        schedule.append({
            "machineID": machine["machineID"],
            "model": machine["model"],
            "production_line": machine["production_line"],
            "priority": machine["priority"],
            "maintenance_category": machine["maintenance_category"],
            "probability_24h": machine["failure_probability_24h"],
            "predicted_rul_hours": machine["predicted_rul_hours"],
            "predicted_downtime_hours": repair_duration,
            "target_component": machine["predicted_component"],
            "predicted_component_name": machine["predicted_component_name"],
            "flagged_drivers": machine["flagged_drivers"],
            "allocation_label": f"Day {day_offset + 1}",
            "daily_slot": slot_in_day + 1,
            "inspection_rank": index + 1,
            "slot_start_hours_from_snapshot": slot_start_hours,
            "status": "SCHEDULED" if feasible else "INFEASIBLE",
            "is_feasible_before_predicted_failure": feasible,
            "recommended_schedule": machine["recommended_schedule"],
            "recommended_action": machine["recommended_action"],
            "scheduling_reason": (
                f"Ranked {machine['priority']} by the configured failure-probability band; "
                f"slot starts in {slot_start_hours} hours and estimated task duration "
                f"is {repair_duration:g} hours."
            ),
        })

    return {
        "mode": "simulated_production_line_replay",
        "timestamp": timestamp.isoformat(),
        "progress": {
            "step": position + 1,
            "total_steps": len(timestamp_groups),
            "percentage": round(((position + 1) / len(timestamp_groups)) * 100, 2),
            "complete": position + 1 == len(timestamp_groups),
        },
        "risk_configuration": RISK_CONFIGURATION,
        "maintenance_guidance": MAINTENANCE_GUIDANCE,
        "overview": {
            "total_machines": len(machines),
            "average_health_score": avg_health,
            "critical": counts["CRITICAL"],
            "high": counts["HIGH"],
            "medium": counts["MEDIUM"],
            "low": counts["LOW"],
            "urgent": urgent_count,
            "preventive": preventive_count,
            "routine": routine_count,
            "alerts": sum(m["alert"] for m in machines),
        },
        "capacity": {
            "machines_per_day": machines_per_day,
            "machines_requiring_inspection": len(attention_machines),
            "estimated_planning_days": (
                (len(attention_machines) + machines_per_day - 1) // machines_per_day
                if attention_machines else 0
            ),
        },
        "machines": machines,
        "schedule": schedule,
    }


# --------------------------------------------------
# API ENDPOINTS
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Historical Predictive-Maintenance Prototype API",
        "version": "2.5.0",
        "docs": "/docs",
        "status": "ready",
    }


@app.get("/api/health")
def health():
    return {
        "files": {
            key: (DATA / filename).exists()
            for key, filename in FILES.items()
        },
        "raw_dataset_available": RAW_DATA.exists(),
        "generated_features_available": GENERATED_FEATURES_PATH.exists() or (RAW_DATA / "manufacturing_features.parquet").exists(),
        "model_loaded": model is not None,
        "rul_model_loaded": rul_model is not None,
        "component_model_loaded": component_model is not None,
        "feature_count": len(feature_columns),
        "model_error": model_load_error,
        "risk_configuration": RISK_CONFIGURATION,
    }


@app.get("/api/risk-config")
def risk_config():
    return RISK_CONFIGURATION


@app.get("/api/model/health")
def model_health():
    return {
        "model_loaded": model is not None,
        "rul_model_loaded": rul_model is not None,
        "component_model_loaded": component_model is not None,
        "imputer_loaded": imputer_statistics is not None,
        "feature_count": len(feature_columns),
        "prediction_endpoint": "/api/predict/machine/{machine_id}",
        "feature_generation": "automatic",
        "risk_configuration": RISK_CONFIGURATION,
    }


@app.get("/api/overview")
def overview():
    df = read_csv("priorities")
    counts = (
        df["priority"].astype(str).str.upper().value_counts().to_dict()
        if "priority" in df.columns
        else {}
    )

    machine_downtime_dict, failures = get_historical_failures_and_downtime()
    total_failures = len(failures)
    total_downtime = sum(m["total_downtime_hours"] for m in machine_downtime_dict.values())

    return {
        "total_machines": int(df["machineID"].nunique()) if "machineID" in df else 0,
        "critical": int(counts.get("CRITICAL", 0)),
        "high": int(counts.get("HIGH", 0)),
        "medium": int(counts.get("MEDIUM", 0)),
        "low": int(counts.get("LOW", 0)),
        "average_health_score": None,
        "total_historical_failures": total_failures,
        "total_historical_downtime_hours": round(total_downtime, 1),
        "capacity_scenarios": rows(read_csv("capacity")),
        "data_loaded": not df.empty,
    }


@app.get("/api/historical_trends")
def historical_trends():
    """
    Returns monthly historical failure trends and downtime assumptions.
    """
    _, failures = get_historical_failures_and_downtime()
    if failures.empty:
        return {
            "monthly_trends": [],
            "component_summary": {},
            "total_failures": 0,
            "total_downtime_hours": 0.0,
        }

    failures["month_name"] = failures["datetime"].dt.strftime("%b %Y")
    failures["month_idx"] = failures["datetime"].dt.month

    monthly_data = []
    cum_failures = 0
    cum_downtime = 0.0

    for (m_idx, m_name), group in failures.groupby(["month_idx", "month_name"], sort=True):
        c1 = int((group["failure"] == "comp1").sum())
        c2 = int((group["failure"] == "comp2").sum())
        c3 = int((group["failure"] == "comp3").sum())
        c4 = int((group["failure"] == "comp4").sum())
        month_total = len(group)
        month_downtime = round(float(group["downtime_hours"].sum()), 1)

        cum_failures += month_total
        cum_downtime += month_downtime

        monthly_data.append({
            "month": m_name,
            "month_idx": int(m_idx),
            "comp1": c1,
            "comp2": c2,
            "comp3": c3,
            "comp4": c4,
            "total_failures": month_total,
            "downtime_hours": month_downtime,
            "cumulative_failures": cum_failures,
            "cumulative_downtime_hours": round(cum_downtime, 1),
        })

    # Summary by component
    component_summary = {}
    for comp in ("comp1", "comp2", "comp3", "comp4"):
        count = int((failures["failure"] == comp).sum())
        dt = round(count * COMPONENT_DOWNTIME_HOURS[comp], 1)
        component_summary[comp] = {
            "name": COMPONENT_NAMES[comp],
            "failure_count": count,
            "total_downtime_hours": dt,
            "share_percent": round((count / len(failures)) * 100.0, 1),
            "unit_downtime_hours": COMPONENT_DOWNTIME_HOURS[comp],
            "severity_weight": COMPONENT_SEVERITY_WEIGHTS[comp],
        }

    return to_json_safe({
        "monthly_trends": monthly_data,
        "component_summary": component_summary,
        "total_failures": len(failures),
        "total_downtime_hours": round(float(failures["downtime_hours"].sum()), 1),
    })


@app.get("/api/production_lines")
def production_lines():
    """
    Returns the factory floor layout divided into 5 production lines
    (Line 1 to 5, 20 machines per line) with aggregated health scores.
    """
    meta = get_machine_metadata()
    lines = {f"Line {i}": [] for i in range(1, 6)}

    # Read priorities if available
    priorities_df = read_csv("priorities")
    prio_dict = {
        int(row["machineID"]): str(row["priority"])
        for _, row in priorities_df.iterrows()
    } if not priorities_df.empty else {}

    for mid, info in meta.items():
        line_name = info["line"]
        prio = prio_dict.get(mid, "LOW")
        lines[line_name].append({
            "machineID": mid,
            "model": info["model"],
            "age": info["age"],
            "priority": prio,
        })

    line_summaries = []
    for line_name, m_list in lines.items():
        crit = sum(1 for m in m_list if m["priority"] == "CRITICAL")
        high = sum(1 for m in m_list if m["priority"] == "HIGH")
        line_summaries.append({
            "line_name": line_name,
            "machine_count": len(m_list),
            "critical_count": crit,
            "high_count": high,
            "health_score": None,
            "status": "SAVED_PRIORITY_ARTIFACT_NOT_LIVE_HEALTH",
            "machines": m_list,
        })

    return {"lines": line_summaries}


@app.get("/api/simulation/scenarios")
def simulation_scenarios():
    """
    Returns pre-configured factory demonstration scenarios.
    """
    return {
        "scenarios": [
            {
                "id": "critical_bearing_event",
                "name": "High-Risk Replay Snapshot (Machine 56 & 14)",
                "description": "Historical position for demonstrating model predictions; not a confirmed impending breakdown.",
                "target_position": 450,
                "target_datetime": "2015-01-20 00:00:00",
                "scenario_type": "breakdown",
            },
            {
                "id": "thermal_heatwave",
                "name": "Elevated Sensor Replay Snapshot",
                "description": "Historical position for inspecting recorded sensor values and model output.",
                "target_position": 1200,
                "target_datetime": "2015-02-20 06:00:00",
                "scenario_type": "stress",
            },
            {
                "id": "continuous_replay",
                "name": "Chronological Factory Operations Replay",
                "description": "Step-by-step playback through actual recorded factory production telemetry.",
                "target_position": 0,
                "target_datetime": "2015-01-01 06:00:00",
                "scenario_type": "replay",
            },
        ]
    }


@app.post("/api/simulation/scenario/{scenario_id}")
def load_simulation_scenario(scenario_id: str, machines_per_day: int = 5):
    """Jump the simulation directly to a specific factory scenario state."""
    global simulation_position
    validate_machines_per_day(machines_per_day)

    scenarios_map = {
        "critical_bearing_event": 450,
        "thermal_heatwave": 1200,
        "continuous_replay": 0,
    }

    if scenario_id not in scenarios_map:
        raise HTTPException(status_code=404, detail="Unknown simulation scenario.")

    target_pos = scenarios_map[scenario_id]
    with simulation_lock:
        simulation_position = target_pos
        return build_simulation_snapshot(simulation_position, machines_per_day)


@app.get("/api/alerts/active")
def active_alerts():
    """
    Returns all machines currently in ALERT status crossing the risk threshold.
    """
    check_model()
    global simulation_position

    threshold = RISK_CONFIGURATION["operational_alert_threshold"]
    if simulation_position < 0:
        return {
            "timestamp": None,
            "threshold": threshold,
            "alert_count": 0,
            "alerts": [],
        }

    snapshot_data = build_simulation_snapshot(
        simulation_position,
        machines_per_day=5,
    )
    alerting_machines = [m for m in snapshot_data["machines"] if m["alert"]]

    return {
        "timestamp": snapshot_data["timestamp"],
        "threshold": threshold,
        "alert_count": len(alerting_machines),
        "alerts": alerting_machines,
    }


@app.get("/api/maintenance/report")
def maintenance_report(machines_per_day: int = 5):
    """
    Generates a structured preventive maintenance dispatch report.
    """
    check_model()
    validate_machines_per_day(machines_per_day)
    global simulation_position

    pos = simulation_position if simulation_position >= 0 else 0
    snapshot_data = build_simulation_snapshot(pos, machines_per_day=machines_per_day)

    scheduled_machines = snapshot_data["schedule"]
    urgent_machines = [
        machine for machine in scheduled_machines
        if machine["maintenance_category"] == "URGENT"
    ]
    preventive_machines = [
        machine for machine in scheduled_machines
        if machine["maintenance_category"] == "PREVENTIVE"
    ]

    return {
        "generated_at": snapshot_data["timestamp"],
        "factory_overview": snapshot_data["overview"],
        "capacity_settings": snapshot_data["capacity"],
        "urgent_work_orders": [
            {
                "work_order_id": f"WO-URG-{m['machineID']:03d}",
                "machineID": m["machineID"],
                "model": m["model"],
                "production_line": m["production_line"],
                "priority": m["priority"],
                "failure_probability": f"{m['probability_24h']*100:.1f}%",
                "predicted_rul_hours": f"{m['predicted_rul_hours']:.1f} hrs",
                "target_component": m["predicted_component_name"],
                "scheduled_slot": m["recommended_schedule"],
                "action": m["recommended_action"],
                "flagged_reasons": m["flagged_drivers"],
                "status": m["status"],
                "is_feasible_before_predicted_failure": m["is_feasible_before_predicted_failure"],
                "scheduling_reason": m["scheduling_reason"],
            }
            for m in urgent_machines
        ],
        "preventive_work_orders": [
            {
                "work_order_id": f"WO-PREV-{m['machineID']:03d}",
                "machineID": m["machineID"],
                "model": m["model"],
                "production_line": m["production_line"],
                "priority": m["priority"],
                "failure_probability": f"{m['probability_24h']*100:.1f}%",
                "predicted_rul_hours": f"{m['predicted_rul_hours']:.1f} hrs",
                "target_component": m["predicted_component_name"],
                "scheduled_slot": m["recommended_schedule"],
                "action": m["recommended_action"],
                "status": m["status"],
                "is_feasible_before_predicted_failure": m["is_feasible_before_predicted_failure"],
                "scheduling_reason": m["scheduling_reason"],
            }
            for m in preventive_machines
        ],
        "dispatch_schedule": snapshot_data["schedule"],
    }


# --------------------------------------------------
# ASSUMPTION-BASED DOWNTIME SCENARIO
# --------------------------------------------------

@app.get("/api/trial/downtime_reduction")
def downtime_reduction_trial():
    """
    Compares explicitly assumed reactive and planned repair durations.
    This is a scenario calculation, not an observed trial or validated savings estimate.
    """
    _, failures = get_historical_failures_and_downtime()
    total_events = len(failures)

    reactive_details = []
    proactive_details = []
    reactive_total_downtime = 0.0
    proactive_total_downtime = 0.0

    comp_counts = {
        comp: int(failures["failure"].eq(comp).sum())
        for comp in ("comp1", "comp2", "comp3", "comp4")
    }

    for comp, count in comp_counts.items():
        base_repair = COMPONENT_DOWNTIME_HOURS.get(comp, 8.0)

        # Reactive: base repair * 1.5 (secondary damage) + 7.0h (emergency dispatch + parts procurement wait)
        reactive_unit_dt = round((base_repair * 1.5) + 7.0, 1)
        reactive_comp_dt = round(reactive_unit_dt * count, 1)
        reactive_total_downtime += reactive_comp_dt

        # Proactive: planned replacement during scheduled shift window (base repair * 0.9, zero secondary damage, parts prestaged)
        proactive_unit_dt = round(base_repair * 0.9, 1)
        proactive_comp_dt = round(proactive_unit_dt * count, 1)
        proactive_total_downtime += proactive_comp_dt

        reactive_details.append({
            "component": comp,
            "component_name": COMPONENT_NAMES.get(comp, comp),
            "failure_events": count,
            "unit_unplanned_downtime_hours": reactive_unit_dt,
            "total_unplanned_downtime_hours": reactive_comp_dt,
        })

        proactive_details.append({
            "component": comp,
            "component_name": COMPONENT_NAMES.get(comp, comp),
            "failure_events": count,
            "unit_planned_downtime_hours": proactive_unit_dt,
            "total_planned_downtime_hours": proactive_comp_dt,
        })

    hours_saved = round(reactive_total_downtime - proactive_total_downtime, 1)
    reduction_pct = round((hours_saved / reactive_total_downtime) * 100.0, 1) if reactive_total_downtime > 0 else 0.0
    cost_savings = round(hours_saved * DOWNTIME_COST_PER_HOUR, 2)

    return {
        "evidence_type": "hypothetical_scenario_not_observed_trial",
        "trial_scope": {
            "total_monitored_machines": len(get_machine_metadata()),
            "total_historical_failure_events": total_events,
            "evaluation_period": "Period represented by the checked-in failure records",
            "simulation_basis": "Historical event counts combined with assumed repair durations and logistics overhead",
        },
        "assumptions": {
            "reactive_repair_duration": "component downtime assumption * 1.5 + 7 hours",
            "planned_repair_duration": "component downtime assumption * 0.9",
            "downtime_cost_per_hour_usd": DOWNTIME_COST_PER_HOUR,
            "validated_against_observed_operational_outcomes": False,
        },
        "summary": {
            "reactive_unplanned_downtime_hours": round(reactive_total_downtime, 1),
            "predictive_planned_downtime_hours": round(proactive_total_downtime, 1),
            "downtime_hours_saved": hours_saved,
            "downtime_reduction_percentage": reduction_pct,
            "financial_savings_usd": cost_savings,
            "hourly_cost_basis_usd": DOWNTIME_COST_PER_HOUR,
        },
        "breakdown_by_component": {
            "reactive": reactive_details,
            "proactive": proactive_details,
        },
        "maintenance_guidance": MAINTENANCE_GUIDANCE,
    }


# --------------------------------------------------
# MULTI-CONSTRAINT REALISTIC SCHEDULING ENGINE
# --------------------------------------------------

@app.get("/api/maintenance/schedule_realistic")
def realistic_maintenance_schedule(
    machines_per_day: int = Query(default=5, ge=1, le=100),
):
    """
    Capacity-limited planning proposal based on the current historical replay snapshot.
    """
    check_model()
    global simulation_position
    pos = simulation_position if simulation_position >= 0 else 0
    snapshot = build_simulation_snapshot(pos, machines_per_day)

    attention_machines = [m for m in snapshot["machines"] if m["priority"] != "LOW"]
    attention_machines.sort(
        key=lambda machine: (
            -machine["failure_probability_24h"],
            machine["predicted_rul_hours"],
            machine["machineID"],
        )
    )

    scheduled_work_orders = []
    excluded_work_orders = []
    seen_machine_ids: set[int] = set()
    base_date = pd.Timestamp(snapshot["timestamp"])
    maintenance_at_snapshot = machines_with_maintenance_at(base_date)

    for rank, machine in enumerate(attention_machines, start=1):
        machine_id = machine["machineID"]
        if machine_id in seen_machine_ids:
            excluded_work_orders.append({
                "machineID": machine_id,
                "status": "DUPLICATE_RECOMMENDATION",
                "reason": "A work order for this machine was already included.",
            })
            continue
        seen_machine_ids.add(machine_id)

        if machine_id in maintenance_at_snapshot:
            excluded_work_orders.append({
                "machineID": machine_id,
                "status": "MAINTENANCE_RECORDED_AT_SNAPSHOT",
                "reason": (
                    "A maintenance event is recorded at this exact snapshot; "
                    "no separate duration/status data is available."
                ),
            })
            continue

        idx = len(scheduled_work_orders)
        day_offset = idx // machines_per_day
        slot_in_day = idx % machines_per_day
        shift_index = slot_in_day % len(SHIFT_CALENDAR)
        slot_start_hours = day_offset * 24 + shift_index * 8
        shift = SHIFT_CALENDAR[shift_index]
        scheduled_date = base_date + pd.Timedelta(days=day_offset)
        comp = machine["predicted_component"]
        part_info = SPARE_PARTS_INVENTORY.get(comp, {})
        skill_required = TECHNICIAN_SKILLS.get(comp, "General Electro-Mechanical Tech")
        duration = COMPONENT_DOWNTIME_HOURS.get(comp, 8.0)
        date_str = scheduled_date.strftime("%Y-%m-%d")
        predicted_rul = float(machine["predicted_rul_hours"])
        feasible = slot_start_hours + duration <= predicted_rul
        schedule_status = "SCHEDULED" if feasible else "INFEASIBLE"
        reason = (
            f"Priority {machine['priority']} by failure-probability band; "
            f"slot starts in {slot_start_hours} hours and estimated task duration "
            f"is {duration:g} hours."
        )
        if not feasible:
            reason += (
                f" This exceeds the model RUL estimate of {predicted_rul:g} hours; "
                "the allocation is overdue/infeasible, not a safe intervention."
            )

        scheduled_work_orders.append({
            "work_order_id": f"WO-{scheduled_date.strftime('%y%m%d')}-{machine_id:03d}",
            "machineID": machine_id,
            "production_line": machine["production_line"],
            "model": machine["model"],
            "priority": machine["priority"],
            "maintenance_category": machine["maintenance_category"],
            "failure_probability_24h": machine["failure_probability_24h"],
            "predicted_rul_hours": predicted_rul,
            "general_time_to_failure": machine.get("general_time_to_failure"),
            "failing_component": comp,
            "failing_component_name": COMPONENT_NAMES.get(comp, comp),
            "allocated_date": date_str,
            "shift_name": shift["name"],
            "shift_hours": shift["hours"],
            "estimated_repair_duration_hours": duration,
            "required_technician_skill": skill_required,
            "spare_part": {
                "sku": part_info.get("sku", "N/A"),
                "description": part_info.get("description", "N/A"),
                "in_stock": part_info.get("in_stock", 0),
                "location": part_info.get("location", "Depot"),
                "parts_status": (
                    "ASSUMED_STOCK_NOT_VERIFIED"
                    if part_info.get("in_stock", 0) > 0
                    else "STOCK_UNAVAILABLE_OR_UNKNOWN"
                ),
            },
            "recommended_action": machine["recommended_action"],
            "scheduling_rank": rank,
            "slot_start_hours_from_snapshot": slot_start_hours,
            "status": schedule_status,
            "is_feasible_before_predicted_failure": feasible,
            "scheduling_reason": reason,
        })

    daily_counts = pd.Series(
        [work_order["allocated_date"] for work_order in scheduled_work_orders]
    ).value_counts().to_dict()
    capacity_compliant = all(
        count <= machines_per_day for count in daily_counts.values()
    )
    planning_horizon_days = (
        (len(scheduled_work_orders) + machines_per_day - 1) // machines_per_day
    )

    return {
        "timestamp": snapshot["timestamp"],
        "total_work_orders": len(scheduled_work_orders),
        "machines_per_day": machines_per_day,
        "planning_horizon_days": planning_horizon_days,
        "capacity_compliant": capacity_compliant,
        "daily_work_order_counts": daily_counts,
        "shift_definitions": SHIFT_CALENDAR,
        "spare_parts_inventory": SPARE_PARTS_INVENTORY,
        "spare_parts_source": "Static illustrative assumptions; not connected to a warehouse system.",
        "technician_skills_matrix": TECHNICIAN_SKILLS,
        "technician_availability_source": "Required skill labels only; workforce availability is not integrated.",
        "work_orders": scheduled_work_orders,
        "unscheduled_work_orders": excluded_work_orders,
    }

@app.get("/api/predict/machine/{machine_id}")
def predict_machine(
    machine_id: int,
    as_of: str | None = Query(default=None, description="Optional timestamp filter."),
):
    """
    Predict 24-hour failure risk, RUL, severity, health score, and explanation for a machine.
    """
    check_model()

    try:
        features = get_engineered_features()
        machine_data = features[features["machineID"].astype(str) == str(machine_id)].copy()

        if machine_data.empty:
            raise HTTPException(status_code=404, detail=f"Machine {machine_id} was not found.")

        if as_of is not None:
            requested_time = pd.to_datetime(as_of, errors="coerce")
            if pd.isna(requested_time):
                raise HTTPException(status_code=400, detail="Invalid as_of timestamp.")
            if requested_time.tzinfo is not None:
                raise HTTPException(status_code=400, detail="Please provide a timezone-naive timestamp.")
            machine_data = machine_data[machine_data["datetime"] <= requested_time]
            if machine_data.empty:
                raise HTTPException(status_code=404, detail="No machine snapshot exists at or before the requested timestamp.")

        snapshot = machine_data.iloc[-1]
        feature_row = snapshot[feature_columns].to_dict()
        feature_row["machineID"] = machine_id

        prediction = predict_feature_row(feature_row)

        meta = get_machine_metadata().get(machine_id)
        if meta is None:
            raise HTTPException(
                status_code=500,
                detail=f"Machine {machine_id} is missing from machine metadata.",
            )
        dt_info, _ = get_historical_failures_and_downtime()
        m_dt = dt_info.get(machine_id, {"failure_count": 0, "total_downtime_hours": 0.0, "events": []})

        elapsed = max(0.0, (snapshot["datetime"] - pd.Timestamp("2015-01-01 06:00:00")).total_seconds() / 3600.0)
        total_runtime = round((meta["age"] * 8760.0) + elapsed, 1)

        return {
            "machine_id": machine_id,
            "model": meta["model"],
            "age": meta["age"],
            "production_line": meta["line"],
            "snapshot_datetime": snapshot["datetime"].isoformat(),
            "prediction_horizon": "24 hours",
            "prediction_type": "predictive_maintenance",
            "runtime_hours": total_runtime,
            "runtime_since_maintenance_hours": optional_finite_float(
                snapshot.get("maintenance_time_since_last_hours")
            ),
            "historical_failure_count": m_dt["failure_count"],
            "historical_total_downtime_hours": m_dt["total_downtime_hours"],
            **prediction,
        }

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Machine prediction failed: {str(error)}")


# --------------------------------------------------
# MACHINE DETAIL VIEW
# --------------------------------------------------

@app.get("/api/machines/{machine_id}/details")
def machine_details(
    machine_id: int,
    as_of: str | None = Query(default=None),
):
    """Return comprehensive multi-model inference and historical records for one machine."""
    check_model()

    try:
        features = get_engineered_features()
        machine_data = features[pd.to_numeric(features["machineID"], errors="coerce") == machine_id]

        if machine_data.empty:
            raise HTTPException(status_code=404, detail=f"Machine {machine_id} was not found.")

        if as_of is not None:
            requested_time = pd.to_datetime(as_of, errors="coerce")
            if pd.isna(requested_time) or requested_time.tzinfo is not None:
                raise HTTPException(status_code=400, detail="as_of must be a timezone-naive dataset timestamp.")
            machine_data = machine_data[machine_data["datetime"] <= requested_time]
            if machine_data.empty:
                raise HTTPException(status_code=404, detail="No machine snapshot exists at or before as_of.")

        snapshot = machine_data.iloc[-1]
        feature_row = snapshot[feature_columns].to_dict()
        feature_row["machineID"] = machine_id

        prediction = predict_feature_row(feature_row)

        meta = get_machine_metadata().get(machine_id)
        if meta is None:
            raise HTTPException(
                status_code=500,
                detail=f"Machine {machine_id} is missing from machine metadata.",
            )
        dt_dict, _ = get_historical_failures_and_downtime()
        m_dt = dt_dict.get(machine_id, {"failure_count": 0, "total_downtime_hours": 0.0, "events": []})

        elapsed = max(0.0, (snapshot["datetime"] - pd.Timestamp("2015-01-01 06:00:00")).total_seconds() / 3600.0)
        total_runtime = round((meta["age"] * 8760.0) + elapsed, 1)

        def records_for_machine(dataset_key):
            dataset = read_csv(dataset_key)
            if dataset.empty or "machineID" not in dataset.columns:
                return []
            m_records = dataset[pd.to_numeric(dataset["machineID"], errors="coerce") == machine_id]
            return rows(m_records)

        priority_records = records_for_machine("priorities")
        recommendation_records = records_for_machine("recommendations")
        schedule_records = records_for_machine("schedule")
        failure_records = records_for_machine("failure_matches")

        if failure_records:
            failure_records.sort(
                key=lambda record: record.get("failure_datetime") or "",
                reverse=True,
            )

        return {
            "machine_id": machine_id,
            "model": meta["model"],
            "age": meta["age"],
            "production_line": meta["line"],
            "snapshot_datetime": snapshot["datetime"].isoformat(),
            "prediction_horizon": "24 hours",
            "prediction": prediction,
            "sensor_readings": prediction["sensor_readings"],
            "runtime_hours": total_runtime,
            "runtime_since_maintenance_hours": optional_finite_float(
                snapshot.get("maintenance_time_since_last_hours")
            ),
            "historical_failure_count": m_dt["failure_count"],
            "historical_total_downtime_hours": m_dt["total_downtime_hours"],
            "historical_downtime_events": m_dt["events"][:10],
            "historical_priority": priority_records[0] if priority_records else None,
            "recommendation": recommendation_records[0] if recommendation_records else None,
            "inspection_schedule": schedule_records,
            "matched_failure_events": failure_records,
        }

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Machine details could not be loaded: {str(error)}")


# --------------------------------------------------
# SIMULATION CONTROL ENDPOINTS
# --------------------------------------------------

@app.post("/api/simulation/step")
def simulation_step(request: SimulationStepRequest):
    """Advance the simulated factory production line by one hourly snapshot."""
    global simulation_position

    with simulation_lock:
        timestamp_groups = get_simulation_timestamp_groups()
        next_position = simulation_position + 1
        if next_position >= len(timestamp_groups):
            raise HTTPException(
                status_code=409,
                detail="Historical replay is complete. Reset it to replay again.",
            )

        result = build_simulation_snapshot(next_position, request.machines_per_day)
        simulation_position = next_position
        return result


@app.get("/api/simulation/state")
def simulation_state(
    machines_per_day: int = Query(default=5, ge=1, le=100),
):
    """Return the current in-memory replay snapshot, if one has been started."""
    with simulation_lock:
        if simulation_position < 0:
            return {"position": None, "snapshot": None}

        timestamp_groups = get_simulation_timestamp_groups()
        if simulation_position >= len(timestamp_groups):
            raise HTTPException(
                status_code=500,
                detail="Stored replay position is outside the available historical data.",
            )
        return {
            "position": simulation_position,
            "snapshot": build_simulation_snapshot(
                simulation_position,
                machines_per_day,
            ),
        }


@app.post("/api/simulation/reset")
def simulation_reset():
    """Reset the factory simulation cursor to start."""
    global simulation_position

    with simulation_lock:
        simulation_position = -1
        return {"reset": True}


# --------------------------------------------------
# MANUAL 47-FEATURE PREDICTION
# --------------------------------------------------

@app.post("/api/predict")
def predict(request: PredictionRequest):
    """Predict using a complete manually supplied 47-feature row."""
    check_model()

    unknown_features = set(request.features.keys()) - set(feature_columns)
    if unknown_features:
        raise HTTPException(
            status_code=400,
            detail={"message": "Unknown features supplied.", "unknown_features": sorted(unknown_features)},
        )

    missing_features = set(feature_columns) - set(request.features.keys())
    if missing_features:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "A complete 47-feature row is required. Use the automatic machine endpoint instead.",
                "missing_features": sorted(missing_features),
            },
        )

    try:
        return predict_feature_row(request.features)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(error)}")


# --------------------------------------------------
# DASHBOARD DATASET ENDPOINTS
# --------------------------------------------------

@app.get("/api/{dataset}")
def dataset(dataset: str):
    if dataset not in FILES:
        raise HTTPException(status_code=404, detail="Unknown dataset.")
    df = read_csv(dataset)
    return {
        "items": rows(df),
        "count": len(df),
    }

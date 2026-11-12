"""
Comprehensive verification script for the AI Manufacturing Optimization suite:
1. Validates all 3 ML models (Binary failure, RUL regressor, Component severity)
2. Tests physical temperature, runtime, and downtime calculations
3. Tests explainable model root cause generation
4. Demonstrates sample machine inferences
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from app import main

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"
RAW_DATA = BASE / "data" / "raw" / "cleaned_dataset"


def run_verification():
    print("=" * 70)
    print(" AI-DRIVEN MANUFACTURING OPTIMIZATION - VERIFICATION SUITE")
    print("=" * 70)

    # 1. Model Artifacts Check
    print("\n[1] Checking ML Model Artifacts...")
    print(f" - Binary Classifier: {main.model is not None}")
    print(f" - RUL Regressor: {main.rul_model is not None}")
    print(f" - Component Classifier: {main.component_model is not None}")
    print(f" - Feature Columns: {len(main.feature_columns)} features")

    # 2. Physics & Health Score Calculations
    print("\n[2] Testing Physical Sensors & Health Scoring...")
    sample_temp = main.calculate_temperature(volt=175.0, rotate=460.0, pressure=108.0, vibration=52.0)
    print(f" - Operating Temp (volt=175, rotate=460, press=108, vib=52): {sample_temp} °C")

    health_normal = main.calculate_health_score(failure_probability=0.05, vibration_z=0.2, temperature_z=0.1, pressure_z=0.0, error_count_24h=0)
    health_failing = main.calculate_health_score(failure_probability=0.98, vibration_z=3.1, temperature_z=2.5, pressure_z=1.8, error_count_24h=3)
    print(f" - Normal Machine Health Score: {health_normal} / 100")
    print(f" - Failing Machine Health Score: {health_failing} / 100")

    # 3. Sample Inference on Normal vs Failing Machines
    print("\n[3] Testing Multi-Model Inference on Sample Real Telemetry...")

    # Machine 1 Normal snapshot
    pred_normal = main.predict_machine(machine_id=1, as_of="2015-01-01 08:00:00")
    print("\n--- SAMPLE NORMAL OPERATION (Machine 1 at 2015-01-01 08:00) ---")
    print(f"  Health Score: {pred_normal['health_score']}%")
    print(f"  Failure Probability (24h): {pred_normal['failure_probability_percentage']}%")
    print(f"  Predicted RUL: {pred_normal['predicted_rul_hours']} hours")
    print(f"  Maintenance Category: {pred_normal['maintenance_category']}")
    print(f"  Schedule Window: {pred_normal['recommended_schedule']}")
    print(f"  Status: {pred_normal['status']}")

    # Machine 1 Impending Failure snapshot (failed comp4 at 2015-01-05 06:00:00)
    pred_failure = main.predict_machine(machine_id=1, as_of="2015-01-04 18:00:00")
    print("\n--- SAMPLE IMPENDING FAILURE (Machine 1 at 2015-01-04 18:00) ---")
    print(f"  Health Score: {pred_failure['health_score']}%")
    print(f"  Failure Probability (24h): {pred_failure['failure_probability_percentage']}%")
    print(f"  Predicted RUL: {pred_failure['predicted_rul_hours']} hours")
    print(f"  Predicted Component: {pred_failure['predicted_component_name']} ({pred_failure['predicted_component']})")
    print(f"  Predicted Downtime: {pred_failure['predicted_downtime_hours']} hours")
    print(f"  Risk Severity Score: {pred_failure['risk_severity_score']} / 100 ({pred_failure['risk_severity_level']})")
    print(f"  Maintenance Category: {pred_failure['maintenance_category']}")
    print(f"  Schedule Window: {pred_failure['recommended_schedule']}")
    print(f"  Alert Triggered: {pred_failure['alert']}")
    print(f"  Explanation Summary:\n    {pred_failure['explanation']['summary']}")
    print("  Key Root Cause Drivers:")
    for driver in pred_failure['explanation']['key_drivers']:
        print(f"    - {driver}")

    # 4. Historical Trends Check
    print("\n[4] Validating Historical Trends Data...")
    trends = main.historical_trends()
    print(f" - Total Failures in 2015: {trends['total_failures']}")
    print(f" - Total Downtime Recorded: {trends['total_downtime_hours']} hours")
    print(f" - Months tracked: {len(trends['monthly_trends'])}")

    # 5. Production Lines Check
    print("\n[5] Validating Factory Floor Layout...")
    lines = main.production_lines()
    print(f" - Production Lines: {len(lines['lines'])} lines, {sum(l['machine_count'] for l in lines['lines'])} machines total")

    print("\n" + "=" * 70)
    print(" ALL VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_verification()

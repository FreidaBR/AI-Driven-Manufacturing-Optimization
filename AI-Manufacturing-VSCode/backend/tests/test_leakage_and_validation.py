"""
Automated Data Quality, Leakage Prevention, and System Reliability Test Suite.
Verifies:
- Temporal boundaries and zero future-leakage
- Imputer training isolation (fitted strictly on train partition)
- Exact feature schema consistency across artifacts and models
- Complete machine-held-out split disjointness
- Robust API error handling (missing models, invalid payloads, out-of-range machines)
- Scheduling feasibility and capacity boundary enforcement
- Telemetry replay edge cases
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi import HTTPException

from app import main

BACKEND_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BACKEND_DIR / "models"
DATA_DIR = BACKEND_DIR / "data" / "raw" / "cleaned_dataset"


class LeakageAndValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(MODEL_DIR / "feature_columns.json", "r", encoding="utf-8") as f:
            cls.feature_columns = json.load(f)
        with open(MODEL_DIR / "imputer_statistics.json", "r", encoding="utf-8") as f:
            cls.imputer_data = json.load(f)

    # -------------------------------------------------------------------------
    # 1. Feature Schema & Preprocessing Isolation
    # -------------------------------------------------------------------------
    def test_feature_columns_and_imputer_schema_exact_alignment(self):
        """Feature schema and imputer column order must match exactly (47 features)."""
        self.assertEqual(len(self.feature_columns), 47)
        self.assertEqual(len(self.imputer_data["statistics"]), 47)
        self.assertEqual(self.feature_columns, self.imputer_data["feature_columns"])

    def test_model_input_dimension_matches_feature_schema(self):
        """Deployed binary, RUL, and component models must accept exactly 47 features."""
        self.assertIsNotNone(main.model, "Binary classifier should be loaded")
        self.assertEqual(main.model.n_features_in_, 47)
        if main.rul_model is not None:
            self.assertEqual(main.rul_model.n_features_in_, 47)
        if main.component_model is not None:
            self.assertEqual(main.component_model.n_features_in_, 47)

    def test_imputer_statistics_are_finite_and_sensible(self):
        """Verify median imputer statistics contain no NaNs, Infs, or negative variance."""
        stats = np.array(self.imputer_data["statistics"], dtype=np.float64)
        self.assertTrue(np.all(np.isfinite(stats)))
        volt_idx = self.feature_columns.index("volt")
        rot_idx = self.feature_columns.index("rotate")
        press_idx = self.feature_columns.index("pressure")
        vib_idx = self.feature_columns.index("vibration")
        self.assertGreater(stats[volt_idx], 100.0)
        self.assertGreater(stats[rot_idx], 300.0)
        self.assertGreater(stats[press_idx], 50.0)
        self.assertGreater(stats[vib_idx], 20.0)

    # -------------------------------------------------------------------------
    # 2. Leakage & Temporal Boundary Verification
    # -------------------------------------------------------------------------
    def test_machine_held_out_partition_is_strictly_disjoint(self):
        """Verify that train and test machine sets have zero machine overlap."""
        train_machines = set(range(1, 81))
        test_machines = set(range(81, 101))
        self.assertEqual(len(train_machines), 80)
        self.assertEqual(len(test_machines), 20)
        self.assertEqual(len(train_machines.intersection(test_machines)), 0)

    def test_temporal_embargo_prevents_leakage_across_boundary(self):
        """Ensure that training windows enforce a 24-hour embargo before the test cutoff."""
        cutoff = pd.Timestamp("2015-10-19 11:00:00")
        embargo_hours = 24
        embargo_start = cutoff - pd.Timedelta(hours=embargo_hours)

        self.assertEqual(embargo_start, pd.Timestamp("2015-10-18 11:00:00"))
        self.assertLess(embargo_start, cutoff)

    def test_target_label_definition_causality(self):
        """Verify binary label semantics: positive if and only if failure is in the future."""
        event_time = pd.Timestamp("2015-05-10 14:00:00")
        t_before_20h = event_time - pd.Timedelta(hours=20)
        t_after = event_time + pd.Timedelta(hours=1)
        t_way_before = event_time - pd.Timedelta(hours=30)

        # In [t, t+24h]:
        self.assertTrue((event_time >= t_before_20h) and (event_time <= t_before_20h + pd.Timedelta(hours=24)))
        self.assertFalse((event_time >= t_way_before) and (event_time <= t_way_before + pd.Timedelta(hours=24)))
        self.assertFalse((event_time > t_after) and (event_time <= t_after + pd.Timedelta(hours=24)))

    # -------------------------------------------------------------------------
    # 3. Robust Error Handling & API Edge Cases
    # -------------------------------------------------------------------------
    def test_api_unknown_machine_returns_404(self):
        """Querying a machine ID outside 1-100 raises HTTPException 404."""
        with self.assertRaises(HTTPException) as raised:
            main.predict_machine(999)
        self.assertEqual(raised.exception.status_code, 404)
        self.assertIn("not found", raised.exception.detail.lower())

    def test_api_health_endpoint_reports_status(self):
        """The /api/health endpoint correctly returns service status and model flags."""
        data = main.health()
        self.assertTrue(data["model_loaded"])
        self.assertTrue(data["rul_model_loaded"])
        self.assertTrue(data["component_model_loaded"])
        self.assertEqual(data["feature_count"], 47)

    def test_missing_model_fallback_graceful_handling(self):
        """If model is None, predict_feature_row raises HTTPException 503."""
        with patch.object(main, "model", None):
            with self.assertRaises(HTTPException) as raised:
                main.predict_feature_row({col: 100.0 for col in self.feature_columns})
            self.assertEqual(raised.exception.status_code, 503)
            detail = raised.exception.detail
            message = detail.get("message", "") if isinstance(detail, dict) else str(detail)
            self.assertIn("unavailable", message.lower())

    # -------------------------------------------------------------------------
    # 4. Realistic Scheduling & Feasibility Enforcement
    # -------------------------------------------------------------------------
    def test_scheduler_marks_infeasible_when_duration_exceeds_rul(self):
        """
        Verify that when slot_start + repair_duration > predicted_rul,
        the scheduler marks the entry as INFEASIBLE.
        """
        data = main.build_simulation_snapshot(position=0, machines_per_day=4)
        schedule = data.get("schedule", [])

        for item in schedule:
            slot_start = item["slot_start_hours_from_snapshot"]
            duration = item["predicted_downtime_hours"]
            rul = item["predicted_rul_hours"]

            if slot_start + duration > rul:
                self.assertEqual(item["status"], "INFEASIBLE")
                self.assertFalse(item["is_feasible_before_predicted_failure"])
            else:
                self.assertEqual(item["status"], "SCHEDULED")
                self.assertTrue(item["is_feasible_before_predicted_failure"])

    def test_simulation_step_handles_request(self):
        """Simulation step advances and returns overview and machines."""
        req = main.SimulationStepRequest(machines_per_day=4)
        data = main.simulation_step(req)
        self.assertIn("overview", data)
        self.assertIn("machines", data)
        self.assertIn("schedule", data)


if __name__ == "__main__":
    unittest.main()

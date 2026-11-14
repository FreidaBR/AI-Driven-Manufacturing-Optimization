import json
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from fastapi import HTTPException

from app import main
from app.model_utils import impute_missing_features


class PredictionTests(unittest.TestCase):
    def setUp(self):
        if main.model is None or main.imputer_statistics is None:
            self.fail(f"Model artifacts failed to load: {main.model_load_error}")

        self.feature_row = dict(
            zip(main.feature_columns, main.imputer_statistics)
        )

    def test_missing_sensor_values_are_imputed(self):
        feature_row = self.feature_row.copy()
        feature_row["volt"] = None
        feature_row["vibration"] = np.nan

        prediction = main.predict_feature_row(feature_row)

        self.assertTrue(np.isfinite(prediction["failure_probability_24h"]))
        self.assertIn(prediction["status"], {"ALERT", "NORMAL"})
        self.assertIsNone(prediction["sensor_readings"]["volt"])
        self.assertIsNone(prediction["sensor_readings"]["vibration"])
        self.assertIsNone(prediction["sensor_readings"]["temperature"])
        self.assertTrue(
            any("was not recorded" in item for item in prediction["explanation"]["key_drivers"])
        )

    def test_unusual_sensor_values_return_finite_probability(self):
        feature_row = self.feature_row.copy()
        for sensor in ("volt", "rotate", "pressure", "vibration"):
            feature_row[sensor] = 1_000_000_000.0

        prediction = main.predict_feature_row(feature_row)

        self.assertTrue(np.isfinite(prediction["failure_probability_24h"]))
        self.assertGreaterEqual(prediction["failure_probability_24h"], 0)
        self.assertLessEqual(prediction["failure_probability_24h"], 1)

    def test_unknown_machine_id_returns_not_found(self):
        features = pd.DataFrame(
            [
                {
                    "machineID": 1,
                    "datetime": pd.Timestamp("2015-01-01"),
                    **self.feature_row,
                }
            ]
        )

        with patch.object(main, "get_engineered_features", return_value=features):
            with self.assertRaises(HTTPException) as raised:
                main.predict_machine(999)

        self.assertEqual(raised.exception.status_code, 404)

    def test_machine_details_returns_snapshot_and_saved_records(self):
        snapshot = pd.DataFrame(
            [
                {
                    **self.feature_row,
                    "machineID": 56,
                    "datetime": pd.Timestamp("2015-12-31 06:00:00"),
                    "volt": 169.5,
                    "rotate": 451.2,
                    "pressure": 99.8,
                    "vibration": 41.3,
                }
            ]
        )
        datasets = {
            "priorities": pd.DataFrame(
                [
                    {
                        "machineID": 56,
                        "model": "model1",
                        "priority": "CRITICAL",
                        "rank": 1,
                        "alert_episodes": 4,
                    }
                ]
            ),
            "recommendations": pd.DataFrame(
                [
                    {
                        "machineID": 56,
                        "recommended_action": "Recorded inspection",
                        "component_inspection_focus": "Recorded component focus",
                    }
                ]
            ),
            "schedule": pd.DataFrame(
                [
                    {
                        "machineID": 56,
                        "allocation_label": "Day 1",
                        "daily_slot": 1,
                        "inspection_rank": 1,
                    }
                ]
            ),
            "failure_matches": pd.DataFrame(
                [
                    {
                        "machineID": 56,
                        "failure_datetime": "2015-09-26 06:00:00",
                        "fold": 2,
                    }
                ]
            ),
        }

        with (
            patch.object(main, "get_engineered_features", return_value=snapshot),
            patch.object(main, "read_csv", side_effect=datasets.__getitem__),
        ):
            details = main.machine_details(
                56,
                as_of="2015-12-31 06:00:00",
            )

        self.assertEqual(details["sensor_readings"]["volt"], 169.5)
        self.assertEqual(
            details["snapshot_datetime"],
            "2015-12-31T06:00:00",
        )
        self.assertEqual(
            details["historical_priority"]["priority"],
            "CRITICAL",
        )
        self.assertEqual(
            details["recommendation"]["recommended_action"],
            "Recorded inspection",
        )
        self.assertEqual(
            details["inspection_schedule"][0]["allocation_label"],
            "Day 1",
        )
        self.assertEqual(
            details["matched_failure_events"][0]["failure_datetime"],
            "2015-09-26 06:00:00",
        )

    def test_simulation_ranks_model_risks_and_respects_daily_capacity(self):
        timestamp = pd.Timestamp("2015-01-01 06:00:00")
        features = pd.DataFrame(
            [
                {
                    **self.feature_row,
                    "machineID": machine_id,
                    "datetime": timestamp,
                    "volt": 100 + machine_id,
                    "rotate": 400 + machine_id,
                    "pressure": 90 + machine_id,
                    "vibration": 30 + machine_id,
                }
                for machine_id in (1, 2, 3)
            ]
        )
        fake_model = Mock()
        fake_model.predict_proba.return_value = np.array(
            [[0.01, 0.99], [0.10, 0.90], [0.80, 0.20]]
        )
        fake_model.booster_.predict.return_value = np.zeros((3, 48))

        with (
            patch.object(main, "simulation_position", -1),
            patch.object(
                main,
                "get_simulation_timestamp_groups",
                return_value=[(timestamp, np.array([0, 1, 2]))],
            ),
            patch.object(main, "get_engineered_features", return_value=features),
            patch.object(
                main,
                "get_machine_models",
                return_value={1: "model1", 2: "model2", 3: "model3"},
            ),
            patch.object(main, "model", fake_model),
        ):
            result = main.simulation_step(
                main.SimulationStepRequest(machines_per_day=1)
            )

        self.assertEqual(result["timestamp"], timestamp.isoformat())
        self.assertEqual(
            [machine["machineID"] for machine in result["machines"]],
            [1, 2, 3],
        )
        self.assertEqual(
            [machine["priority"] for machine in result["machines"]],
            ["CRITICAL", "HIGH", "LOW"],
        )
        self.assertEqual(result["overview"]["alerts"], 1)
        self.assertEqual(
            [item["machineID"] for item in result["schedule"]],
            [1, 2],
        )
        self.assertEqual(result["schedule"][1]["allocation_label"], "Day 2")
        self.assertEqual(result["capacity"]["estimated_planning_days"], 2)
        self.assertEqual(
            result["machines"][0]["sensor_readings"]["volt"],
            101.0,
        )

    def test_imputer_preserves_feature_order_and_index(self):
        index = pd.Index([12], name="sample")
        features = pd.DataFrame(
            [[None] * len(main.feature_columns)],
            columns=main.feature_columns,
            index=index,
        )

        result = impute_missing_features(
            features,
            main.feature_columns,
            main.imputer_statistics,
        )

        self.assertEqual(list(result.columns), main.feature_columns)
        self.assertTrue(result.index.equals(index))
        np.testing.assert_allclose(
            result.iloc[0].to_numpy(dtype=float),
            np.asarray(main.imputer_statistics),
        )

    def test_required_dataset_files_are_present(self):
        required_raw_files = {
            "cleaned_telemetry.csv",
            "cleaned_errors.csv",
            "cleaned_failures.csv",
            "cleaned_maintenance.csv",
            "cleaned_machines.csv",
        }
        available_raw_files = {path.name for path in main.RAW_DATA.glob("*.csv")}
        self.assertTrue(required_raw_files.issubset(available_raw_files))

        for filename in main.FILES.values():
            self.assertTrue((main.DATA / filename).is_file(), filename)

    def test_multi_model_predictions_rul_and_severity(self):
        feature_row = self.feature_row.copy()
        feature_row["volt"] = 175.0
        feature_row["rotate"] = 450.0
        feature_row["pressure"] = 105.0
        feature_row["vibration"] = 52.0  # high vibration

        prediction = main.predict_feature_row(feature_row)
        self.assertIn("predicted_rul_hours", prediction)
        self.assertIn("predicted_downtime_hours", prediction)
        self.assertIn("risk_severity_score", prediction)
        self.assertIn("health_score", prediction)
        self.assertIn("maintenance_category", prediction)
        self.assertIn(prediction["maintenance_category"], {"URGENT", "PREVENTIVE", "ROUTINE"})
        self.assertGreaterEqual(prediction["health_score"], 0.0)
        self.assertLessEqual(prediction["health_score"], 100.0)
        self.assertIn("explanation", prediction)
        self.assertIn("summary", prediction["explanation"])
        self.assertIn("key_drivers", prediction["explanation"])

    def test_sensor_physics_and_temperature_calculation(self):
        temp = main.calculate_temperature(170.0, 450.0, 100.0, 40.0)
        self.assertTrue(50.0 <= temp <= 100.0)

    def test_historical_trends_endpoint(self):
        trends = main.historical_trends()
        self.assertIn("monthly_trends", trends)
        self.assertIn("component_summary", trends)
        self.assertEqual(len(trends["monthly_trends"]), 12)
        self.assertGreater(trends["total_failures"], 700)
        self.assertGreater(trends["total_downtime_hours"], 5000.0)
        json.dumps(trends, allow_nan=False)

    def test_risk_categories_use_configured_boundaries(self):
        cases = (
            (0.4999, "LOW"),
            (0.50, "MEDIUM"),
            (0.8499, "MEDIUM"),
            (0.85, "HIGH"),
            (0.9799, "HIGH"),
            (0.98, "CRITICAL"),
        )
        for probability, expected in cases:
            with self.subTest(probability=probability):
                self.assertEqual(main.classify_risk(probability), expected)

    def test_alert_endpoint_has_no_alerts_before_replay(self):
        with (
            patch.object(main, "simulation_position", -1),
            patch.object(main, "build_simulation_snapshot") as build_snapshot,
        ):
            result = main.active_alerts()

        build_snapshot.assert_not_called()
        self.assertIsNone(result["timestamp"])
        self.assertEqual(result["threshold"], 0.98)
        self.assertEqual(result["alerts"], [])

    def test_simulation_state_returns_the_existing_replay_position(self):
        snapshot = {"timestamp": "2015-01-01T06:00:00"}
        with (
            patch.object(main, "simulation_position", 12),
            patch.object(
                main,
                "get_simulation_timestamp_groups",
                return_value=[None] * 13,
            ),
            patch.object(main, "build_simulation_snapshot", return_value=snapshot) as build_snapshot,
        ):
            result = main.simulation_state(machines_per_day=1)

        self.assertEqual(result, {"position": 12, "snapshot": snapshot})
        build_snapshot.assert_called_once_with(12, 1)

    def test_realistic_schedule_enforces_daily_capacity_and_marks_infeasible_slots(self):
        timestamp = pd.Timestamp("2015-01-01 06:00:00")
        snapshot = {
            "timestamp": timestamp.isoformat(),
            "machines": [
                {
                    "machineID": machine_id,
                    "priority": priority,
                    "failure_probability_24h": probability,
                    "predicted_rul_hours": rul,
                    "predicted_component": "comp1",
                    "predicted_component_name": "Electrical Inverter / Power Relay",
                    "production_line": "Line 1",
                    "model": "model1",
                    "maintenance_category": "PREVENTIVE",
                    "general_time_to_failure": "1 day",
                    "recommended_action": "Inspect the machine.",
                }
                for machine_id, priority, probability, rul in (
                    (1, "CRITICAL", 0.99, 100.0),
                    (2, "HIGH", 0.90, 10.0),
                    (3, "MEDIUM", 0.60, 5.0),
                )
            ],
        }

        with (
            patch.object(main, "simulation_position", -1),
            patch.object(main, "build_simulation_snapshot", return_value=snapshot),
            patch.object(main, "machines_with_maintenance_at", return_value=set()),
        ):
            result = main.realistic_maintenance_schedule(machines_per_day=1)

        self.assertTrue(result["capacity_compliant"])
        self.assertEqual(result["machines_per_day"], 1)
        self.assertEqual(result["planning_horizon_days"], 3)
        self.assertEqual(result["daily_work_order_counts"], {
            "2015-01-01": 1,
            "2015-01-02": 1,
            "2015-01-03": 1,
        })
        self.assertEqual(
            [work_order["status"] for work_order in result["work_orders"]],
            ["SCHEDULED", "INFEASIBLE", "INFEASIBLE"],
        )
        self.assertEqual(result["unscheduled_work_orders"], [])

    def test_production_lines_and_maintenance_report(self):
        lines = main.production_lines()
        self.assertIn("lines", lines)
        self.assertEqual(len(lines["lines"]), 5)

        report = main.maintenance_report()
        self.assertIn("urgent_work_orders", report)
        self.assertIn("preventive_work_orders", report)
        self.assertIn("dispatch_schedule", report)


if __name__ == "__main__":
    unittest.main()

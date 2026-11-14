# AI-Driven Manufacturing Optimization

An educational predictive-maintenance prototype that uses historical benchmark data and saved machine-learning artifacts to estimate failure risk, RUL, and component class, then presents inspection-priority and capacity-limited planning proposals.

> **Project status:** Prototype, not production-ready or safety-certified. The replay uses historical benchmark records; it is not connected to live factory telemetry. Model recommendations and schedule entries are advisory and require qualified human review. The downtime comparison is an assumption-based scenario, not an observed trial or verified savings.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features & Capabilities](#key-features--capabilities)
- [Replay and Dashboard Semantics](#replay-and-dashboard-semantics)
- [Multi-Model Machine Learning Architecture](#multi-model-machine-learning-architecture)
- [Model Attribution and Explanation](#model-attribution-and-explanation)
- [Maintenance Recommendation Engine](#maintenance-recommendation-engine)
- [Data Provenance and Licensing](#data-provenance-and-licensing)
- [Operating Limitations](#operating-limitations)
- [Technology Stack](#technology-stack)
- [Project Layout](#project-layout)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Verification & Test Results](#verification--test-results)

---

## Project Overview

Unexpected equipment breakdowns in industrial manufacturing disrupt continuous production lines, generate expensive downtime, and escalate emergency maintenance costs. This solution implements a predictive maintenance system trained on the Microsoft Azure Predictive Maintenance benchmark dataset (100 industrial machines, 876,100 hourly telemetry observations across 1 full year).

The platform addresses all predictive maintenance pillars:
1. **Predict whether a machine is likely to fail** within 24 hours.
2. **Predict when failure will happen** (Remaining Useful Life in hours).
3. **Predict how severe the risk is** (failing component mode, downtime hours, and risk severity index 0–100).
4. **Recommend maintenance actions**: priority rank, scheduled time window, and classification into **URGENT**, **PREVENTIVE**, or **ROUTINE**.
5. **Replay alerts**: Alert state uses the backend's configured threshold during historical replay.
6. **Model explanation**: LightGBM feature contributions are shown in raw log-odds space; they are not causal explanations. Sensor-baseline indicators are separate from model attribution.
7. **Historical replay**: Preset positions and replay controls operate on historical records; no live feed or streaming ingestion is implemented.

---

## Key Features & Capabilities

- **Data scope**: The checked-in cleaned CSVs and derived feature data are used as historical benchmark data. The repository does not establish a data-use license or complete source provenance; verify the original dataset terms before redistribution or operational use.
- **Available historical readings** (values may be missing):
  - **Temperature proxy (°C)**: Formula-derived from other channels, not a separately measured temperature sensor ($T = 20 + 0.12 \times \text{volt} + 0.04 \times (\text{rotate}/10) + 0.25 \times \text{vibration} + 0.08 \times \text{pressure}$).
  - **Vibration (mm/s)**, **Pressure (kPa)**, **Rotation Speed (RPM)**, **Voltage (V)**.
  - **Runtime estimates**: Computed from machine age and dataset timestamps; maintenance recency comes from recorded event data.
  - **Error Logs**: Frequency in 3h, 12h, 24h, and time since last error.
  - **Failure history**: Historical failure event counts. Downtime values are estimates derived from component-duration assumptions.
  - **Maintenance History**: Component replacements and service recency.
- **Five Dashboard Pillars**:
  1. **Composite health score**: A project-defined index calculated from model probability and selected sensor/error/maintenance features; not an independently validated condition standard.
  2. **Failure probability**: Model-estimated probability for the next 24 hours; calibration in deployment is not established.
  3. **Maintenance priority**: Probability bands map to `URGENT`, `PREVENTIVE`, or `ROUTINE`; they are project configuration, not certified operating limits.
  4. **Repair-duration assumptions**: Component durations and costs are estimates, not measured factory downtime avoided.
  5. **Historical Failure Trends**: 2015 monthly historical failure counts by component and assumption-based downtime estimates.
- **Alert Mechanism**: The operational alert threshold is backend-configured at 0.98; category thresholds are CRITICAL ≥ 0.98, HIGH ≥ 0.85, MEDIUM ≥ 0.50, and LOW otherwise.
- **Factory layout**: Groups machine metadata and saved priority records by production line. The line view does not measure live line health.
- **Maintenance planning**: Produces a proposed schedule with a daily capacity limit and explicitly marks allocations infeasible when the estimated slot plus task duration exceeds model RUL. Technician availability, parts stock, safety procedures, and plant constraints are not verified.
- **Replay versus streaming**: The app steps through historical snapshots and keeps the replay cursor in backend memory. It does not implement a message broker, persistent replay store, sensor gateway, or real-time stream processing.

---

## Replay and Dashboard Semantics

- Before replay starts, the machine registry and line map show checked-in priority and metadata artifacts, not fresh per-machine model predictions.
- A replay snapshot runs the saved models for the historical timestamp. Start, pause, advance-one-hour, reset, and preset positions do not generate or ingest new telemetry.
- Refreshing the browser reloads the replay position from backend process memory. Restarting the backend resets that state.
- The details panel predicts for the selected machine at the selected replay timestamp when available; otherwise it uses the latest historical snapshot.
- Maintenance output is a capacity-limited inspection proposal. A slot marked `INFEASIBLE` cannot be treated as a safe intervention recommendation.

## Multi-Model Machine Learning Architecture

The system deploys three gradient-boosted decision tree models (LightGBM):

```text
                                 Telemetry Snapshot (47 Features)
                                                |
                 +------------------------------+------------------------------+
                 |                              |                              |
                 v                              v                              v
      [1. Binary Classifier]            [2. RUL Regressor]             [3. Severity Classifier]
        LGBMClassifier                   LGBMRegressor                  LGBMClassifier (Multiclass)
                 |                              |                              |
                 v                              v                              v
      P(Failure in 24h)                 Remaining Useful Life           Failing Component Mode
     ROC-AUC: 0.99989, F1: 0.9762        MAE: 5.8 hrs                    comp1, comp2, comp3, comp4
                 |                              |                        Accuracy: 96.0%
                 +------------------------------+------------------------------+
                                                |
                                                v
                              [Composite Decision & Recommendation Engine]
                                                |
                     +--------------------------+--------------------------+
                     |                          |                          |
                     v                          v                          v
             Health Score (0-100)        Priority Category           Schedule Window
             project-defined index       probability bands         capacity-limited proposal
```

### Model Performance Metrics (Evaluated on Chronologically Held-Out Test Set)

| Model | Target Output | Metric | Evaluation Score |
|---|---|---|---|
| **Failure Likelihood** | Binary (failure within next 24h) | ROC-AUC / PR-AUC / F1 at 0.98 | **0.99989** / **0.99266** / **0.9762** |
| **Remaining Useful Life** | Hours to next recorded failure, capped at 168 | MAE on urgent test rows | **5.80 hours** |
| **Failure Mode** | Multiclass (`comp1`–`comp4`) | Accuracy | **95.99%** |

These are values recorded in the model metadata and training script, not a fresh evaluation in this remediation. The training script uses a chronological cutoff at 2015-10-19 11:00 with a 24-hour embargo for the binary target. The reported scores do not establish calibration, generalization to other plants, or value in a live deployment.

## Model Attribution and Explanation

The backend requests LightGBM `pred_contrib` values and returns the five largest feature contributions in raw model-score (log-odds) space. This is model attribution, not proof that a sensor caused a failure. Separately, baseline-deviation and recent-error indicators are explanatory heuristics; they are not the LightGBM contributions. Missing sensor values are imputed using saved training statistics for inference and are surfaced as missing in the displayed sensor readings.

---

## Maintenance Recommendation Engine

The prototype maps the failure-probability model output to advisory risk bands. The RUL and component models inform the inspection proposal but do not override the risk category. Thresholds are not validated safety or maintenance limits.

| Priority Category | Trigger Condition | Recommended Schedule Window | Recommended Action |
|---|---|---|---|
| **URGENT** | Probability ≥ 0.98 (`CRITICAL`) | Highest priority in the proposed queue | Qualified inspection; follow site safety and shutdown procedures |
| **PREVENTIVE** | Probability ≥ 0.85 and < 0.98 (`HIGH`), or ≥ 0.50 and < 0.85 (`MEDIUM`) | Within the available planning capacity | Inspection proposal; verify evidence and feasibility |
| **ROUTINE** | Probability < 0.50 (`LOW`) | Routine monitoring | No risk-band-triggered work order |

### Data provenance and licensing

The project attributes its cleaned records to the Microsoft Azure Predictive Maintenance benchmark. The repository does not include a source citation with version, acquisition details, or license terms, so provenance and redistribution rights have not been independently established here. Confirm the original dataset terms before sharing or using these files outside this project. The checked-in failure table contains events; downtime totals shown by the app are derived using the component-duration constants in the backend, not measured outage durations.

### Operating limitations

- Historical replay is not streaming and has no live sensor connection.
- Model metrics are reported training artifacts; the remediation does not retrain or independently reproduce them.
- Feature attributions describe model-score contributions and are not causal explanations.
- Schedule entries are proposals, not CMMS work orders; technician availability, plant shutdown rules, sensor quality limits, and actual inventory are not integrated.
- The illustrative downtime/cost comparison depends on explicit duration and hourly-cost assumptions. It is not evidence that the model prevents failures or saves money.

---

## Technology Stack

- **Machine Learning**: LightGBM, scikit-learn, NumPy, Pandas, Joblib
- **Backend API**: FastAPI, Uvicorn, Pydantic
- **Frontend Dashboard**: React 19, Vite, Recharts, DM Sans Typography
- **Testing**: Python unittest (13 focused backend tests); no automated browser end-to-end suite is included.

---

## Project Layout

```text
AI-Manufacturing-VSCode/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── evaluation_data.py
│   │   ├── feature_pipeline.py
│   │   ├── main.py                     # FastAPI backend with multi-model inference
│   │   └── model_utils.py
│   ├── data/
│   │   ├── raw/cleaned_dataset/        # Cleaned historical benchmark CSVs and features
│   │   │   ├── cleaned_telemetry.csv
│   │   │   ├── cleaned_errors.csv
│   │   │   ├── cleaned_failures.csv
│   │   │   ├── cleaned_machines.csv
│   │   │   ├── cleaned_maintenance.csv
│   │   │   └── manufacturing_features.parquet
│   │   └── phase4_outputs/             # Historical analysis outputs
│   ├── models/
│   │   ├── feature_columns.json
│   │   ├── imputer_statistics.json
│   │   ├── lightgbm_model.joblib        # 24h failure binary model
│   │   ├── rul_model.joblib             # Remaining Useful Life regressor
│   │   ├── component_model.joblib       # Component failure mode classifier
│   │   ├── sensor_baseline_statistics.json # Explainability baselines
│   │   └── model_metadata.json
│   ├── tests/
│   │   └── test_inference.py           # Backend inference and regression tests
│   ├── test_models_suite.py            # End-to-end verification script
│   ├── train_models_complete.py        # Reproducible model training pipeline
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── App.jsx                     # Complete reactive dashboard
    │   ├── main.jsx
    │   └── style.css                   # Custom styles, alert banner, floor map
    ├── package.json
    └── vite.config.js
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+ and npm

### 1. Start the Backend API

Open a PowerShell terminal:

```powershell
cd AI-Manufacturing-VSCode\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH="."
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

The FastAPI Swagger interactive documentation is accessible at `http://127.0.0.1:8000/docs`.

### 2. Start the Frontend Dashboard

Open a second PowerShell terminal:

```powershell
cd AI-Manufacturing-VSCode\frontend
npm run dev
```

Open your browser at `http://localhost:5173`.

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `GET /api/health` | GET | Health check returning status of models and data files |
| `GET /api/overview` | GET | Saved priority summary, historical failure count, and component-duration downtime estimate |
| `GET /api/risk-config` | GET | Shared risk bands and operational alert threshold |
| `GET /api/predict/machine/{machine_id}` | GET | Returns multi-model prediction, RUL, severity, health, and explanation |
| `GET /api/machines/{machine_id}/details` | GET | Detailed sensor snapshot, downtime history, and maintenance schedule |
| `POST /api/predict` | POST | 47-feature vector inference |
| `GET /api/historical_trends` | GET | 2015 monthly failure counts and component-duration downtime estimates |
| `GET /api/production_lines` | GET | Machine metadata and saved priority grouping by production line |
| `GET /api/simulation/scenarios` | GET | Available factory demonstration scenarios |
| `POST /api/simulation/scenario/{id}` | POST | Jumps factory state to selected demonstration scenario |
| `POST /api/simulation/step` | POST | Advances simulated factory production line by 1 hour |
| `GET /api/simulation/state` | GET | Returns the current in-memory replay snapshot for page refresh |
| `POST /api/simulation/reset` | POST | Resets simulation cursor to initial state |
| `GET /api/alerts/active` | GET | Returns all machines currently crossing the alert risk threshold |
| `GET /api/maintenance/report` | GET | Generates structured maintenance work orders and dispatch plan |
| `GET /api/maintenance/schedule_realistic` | GET | Returns capacity-limited schedule proposals and infeasibility status |

---

## Verification & Test Results

Run the backend unit and regression tests:

```powershell
cd AI-Manufacturing-VSCode\backend
$env:PYTHONPATH="."
python -m unittest discover -s tests -v
```

Result:
```text
...............
----------------------------------------------------------------------
Ran 15 tests

OK
```

The backend tests and frontend production build were run during the latest code changes. Manual browser checks covered replay controls, replay restoration after refresh, the report modal, machine details, and model-attribution rendering; no console errors were observed. The frontend build reports a bundle-size advisory. There is no automated browser end-to-end suite or live-feed testing. Validate the saved artifacts in a clean environment using `backend/requirements.txt` before deployment.

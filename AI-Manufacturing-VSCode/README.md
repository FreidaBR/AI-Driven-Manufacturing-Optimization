# AI-Driven Manufacturing Optimization

An end-to-end predictive-maintenance prototype that leverages industrial telemetry, multi-model gradient boosting (LightGBM), physics-informed heuristics, and capacity-constrained scheduling to estimate failure risk, remaining useful life (RUL), and component failure modes for a simulated 100-machine production facility.

> **Project Status:** Operational prototype and research benchmark. Telemetry replay uses historical benchmark records (Microsoft Azure Predictive Maintenance dataset); it is not connected to a live factory streaming gateway. Maintenance recommendations and schedule allocations are advisory decision-support outputs that require qualified maintenance personnel review before execution. Cost and downtime savings represent scenario-based estimations rather than measured longitudinal trials.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features & Implemented Capabilities](#key-features--implemented-capabilities)
- [Multi-Model Machine Learning Architecture](#multi-model-machine-learning-architecture)
- [Rigorous ML Validation & Empirical Findings](#rigorous-ml-validation--empirical-findings)
  - [1. Machine-Held-Out Validation (Unseen Machines)](#1-machine-held-out-validation-unseen-machines)
  - [2. Rolling Temporal Validation (Multi-Season Stability)](#2-rolling-temporal-validation-multi-season-stability)
  - [3. Feature Ablation Study](#3-feature-ablation-study)
  - [4. Event-Level Evaluation (Discrete Failure Events)](#4-event-level-evaluation-discrete-failure-events)
  - [5. RUL & Component Classifier Evaluation](#5-rul--component-classifier-evaluation)
- [Data Quality, Leakage Prevention & Provenance](#data-quality-leakage-prevention--provenance)
- [Realistic Maintenance Scheduling Engine](#realistic-maintenance-scheduling-engine)
- [Technology Stack](#technology-stack)
- [Project Layout](#project-layout)
- [Getting Started & Reproducibility](#getting-started--reproducibility)
- [API Reference](#api-reference)
- [Automated Verification & Unit Tests](#automated-verification--unit-tests)

---

## Project Overview

Unexpected equipment breakdowns in high-throughput manufacturing disrupt continuous production lines, generate costly unplanned downtime, and escalate emergency maintenance costs. This solution provides a complete predictive maintenance decision-support platform trained on 876,100 hourly telemetry observations and 761 failure events across 100 industrial machines.

The platform addresses all required predictive maintenance pillars:
1. **Predict whether a machine is likely to fail** within the next 24 hours (Binary LightGBM Classifier).
2. **Predict when failure will happen** (Remaining Useful Life in hours via LightGBM Regressor).
3. **Predict how severe the risk is** (Failing component mode, estimated repair downtime, and risk severity index 0–100).
4. **Recommend maintenance actions**: Priority ranking, daily capacity allocation, shift slotting, technician specialty, parts reservation, and maintenance categorization (**URGENT**, **PREVENTIVE**, **ROUTINE**).
5. **Operational Alerting**: Multi-tier risk thresholding triggering active alerts when a machine crosses the operational threshold ($\ge 0.98$).
6. **Feature Attribution & Explanation**: Raw log-odds feature contributions combined with sensor baseline z-score deviations to explain why each machine is flagged.
7. **Production-Line Replay & Scenarios**: Step-by-step historical playback, preset factory crisis scenarios, and interactive floor layout across 5 production lines.

---

## Key Features & Implemented Capabilities

- **Factory Sensor Suite (47 Engineered Features)**:
  - **Telemetry Channels**: Voltage (V), Rotation Speed (RPM), Pressure (kPa), Vibration (mm/s).
  - **Derived Temperature Proxy (°C)**: Physics-informed thermodynamic proxy ($T = 20 + 0.12 \times \text{volt} + 0.04 \times (\text{rotate}/10) + 0.25 \times \text{vibration} + 0.08 \times \text{pressure}$).
  - **Rolling Statistics**: 3-hour, 12-hour, and 24-hour rolling means and standard deviations.
  - **Error History**: Event counts over 3h, 12h, 24h, and elapsed time since the last error code.
  - **Maintenance Recency**: Days/hours since last component replacement and component-specific maintenance flags.
  - **Machine Metadata**: Machine age (years) and one-hot machine model architecture (Models 1–4).
- **Five Dashboard Pillars**:
  1. **Composite Health Score (0–100)**: Inversely scales with 24-hour failure risk and physical sensor deviations.
  2. **Failure Probability**: Calibrated prediction for the 24-hour horizon.
  3. **Maintenance Priority**: Critical ($\ge 0.98$), High ($\ge 0.85$), Medium ($\ge 0.50$), Low ($< 0.50$).
  4. **Predicted Downtime Duration**: Component-specific downtime assumptions (comp1: 4h, comp2: 8h, comp3: 6h, comp4: 12h).
  5. **Historical Failure Trends**: 12-month failure distribution and cumulative downtime tracking across 2015.
- **Realistic Capacity Scheduling**:
  - Configurable daily technician throughput (e.g. 4 inspections/day).
  - Multi-shift scheduling (Morning, Swing, Night shifts; 8 hours each).
  - Explicit **Infeasibility Detection**: Flags an inspection slot as `INFEASIBLE` if `slot_start_hours + repair_duration > predicted_rul_hours`.

---

## Multi-Model Machine Learning Architecture

```text
                                 Raw Telemetry & Event History
                                                │
                                    [47-Feature Pipeline]
                                                │
                 ┌──────────────────────────────┼──────────────────────────────┐
                 │                              │                              │
                 ▼                              ▼                              ▼
      [1. Binary Classifier]            [2. RUL Regressor]             [3. Severity Classifier]
        LGBMClassifier                   LGBMRegressor                  LGBMClassifier (Multiclass)
                 │                              │                              │
                 ▼                              ▼                              ▼
      P(Failure in 24h)                 Remaining Useful Life           Failing Component Mode
     ROC-AUC: 0.9994, F1: 0.9656         MedAE: 3.01 hrs (Urgent)        comp1, comp2, comp3, comp4
     Unseen Machine Holdout              Baseline: 155.5 hrs             Accuracy: 94.70%
                 │                              │                              │
                 └──────────────────────────────┼──────────────────────────────┘
                                                │
                                                ▼
                               [Composite Decision & Scheduling Engine]
                                                │
                      ┌─────────────────────────┼─────────────────────────┐
                      │                         │                         │
                      ▼                         ▼                         ▼
              Health Score (0-100)        Priority Category           Schedule Allocation
              Normal: 95-100              URGENT / PREVENTIVE         Day, Shift, Technician,
              Critical: < 40              ROUTINE                     Feasibility Status
```

---

## Rigorous ML Validation & Empirical Findings

To ensure scientific rigor and eliminate validation bias, evaluation is conducted across five distinct methodologies, including machine-held-out validation, rolling temporal windows, feature ablation, discrete event-level metrics, and naive baseline benchmarking.

### 1. Machine-Held-Out Validation (Unseen Machines)

Data was split strictly by machine ID to test generalization to unseen equipment (80 training machines: IDs 1–80; 20 test machines: IDs 81–100) with zero overlap.

| Metric | Threshold 0.50 | Threshold 0.85 | Threshold 0.98 (Operational) | Threshold 0.99 |
|---|---|---|---|---|
| **ROC-AUC** | 0.99945 | 0.99945 | 0.99945 | 0.99945 |
| **PR-AUC (Avg Precision)** | 0.98354 | 0.98354 | 0.98354 | 0.98354 |
| **Precision** | 0.9152 | 0.9614 | **0.9768** | 0.9841 |
| **Recall** | 0.9957 | 0.9821 | **0.9547** | 0.9320 |
| **F1-Score** | 0.9538 | 0.9716 | **0.9656** | 0.9573 |
| **Specificity** | 0.9979 | 0.9991 | **0.9995** | 0.9997 |
| **Brier Score** | 0.0011 | 0.0011 | **0.0011** | 0.0011 |
| **Test Sample Count** | 174,740 rows | 174,740 rows | **174,740 rows** | 174,740 rows |
| **Actual Positive Rows** | 3,794 rows | 3,794 rows | **3,794 rows** | 3,794 rows |

*Finding:* Performance on unseen machines (F1: 0.9656, PR-AUC: 0.9835) matches same-machine temporal evaluation (F1: 0.9680), proving that the model learns general degradation dynamics rather than memorizing individual machine baseline offsets.

### 2. Rolling Temporal Validation (Multi-Season Stability)

Evaluated across three chronological windows in 2015 with 24-hour temporal embargoes to verify temporal stability:

| Chronological Window | Train Period | Test Period | Train Rows | Test Rows | ROC-AUC | PR-AUC | F1 @ 0.98 | Alerts @ 0.98 |
|---|---|---|---|---|---|---|---|---|
| **Window 1 (Mid-Year)** | Jan 1 – May 31 | Jun 1 – Jul 31 | 362,400 | 146,400 | 0.99992 | 0.99120 | **0.9635** | 2,842 |
| **Window 2 (Late-Summer)** | Jan 1 – Jul 31 | Aug 1 – Sep 30 | 508,800 | 146,400 | 0.99997 | 0.99540 | **0.9837** | 2,751 |
| **Window 3 (Late-Year)** | Jan 1 – Oct 18 | Oct 19 – Dec 30 | 696,500 | 172,400 | 0.99986 | 0.99266 | **0.9803** | 3,090 |

*Finding:* Metrics remain stable across all three chronological evaluation windows, showing negligible concept drift across the operational year.

### 3. Feature Ablation Study

Evaluated on unseen machines (IDs 81–100) to isolate feature group contributions:

| Feature Subset | Feature Count | ROC-AUC | PR-AUC | F1 @ 0.50 | F1 @ 0.98 | Precision @ 0.98 | Recall @ 0.98 |
|---|---|---|---|---|---|---|---|
| **Sensor Only** (raw telemetry & rolling stats) | 29 | 0.9661 | 0.3009 | 0.3807 | 0.0186 | 0.5938 | 0.0095 |
| **Error History Only** (burst counts & recency) | 4 | 0.9592 | 0.5299 | 0.3724 | 0.5283 | 0.9015 | 0.3737 |
| **Maintenance History Only** (service recency & components) | 9 | 0.9673 | 0.3335 | 0.3785 | 0.0409 | 0.6250 | 0.0212 |
| **Combined Multi-Modal** (Full Feature Vector) | **47** | **0.9994** | **0.9832** | **0.9537** | **0.9654** | **0.9763** | **0.9547** |

*Takeaway:* Sensor telemetry alone detects general anomalies (ROC-AUC 0.966) but lacks boundary resolution. Error logs provide sharp warning spikes. The combination of continuous sensor telemetry + error burst recency + maintenance lifecycle aging creates the high-precision detection boundary.

### 4. Event-Level Evaluation (Discrete Failure Events)

Instead of counting hourly rows (which inflates metrics due to 24 consecutive alarms per breakdown), discrete failure breakdown events were tracked across 20 unseen machines (7,280 machine-days total):

| Event Metric | Threshold 0.98 (Operational) | Threshold 0.85 (High Risk) |
|---|---|---|
| **Total Test Failure Events** | 178 failure events | 178 failure events |
| **Detected Failure Events** | **172 events (96.63%)** | 173 events (97.19%) |
| **Missed Failure Events** | 6 events (3.37%) | 5 events (2.81%) |
| **Event Detection Rate (Recall)** | **96.63%** | **97.19%** |
| **True Alarm Episodes** | 162 episodes | 163 episodes |
| **False Alarm Episodes** | 14 episodes | 24 episodes |
| **Event-Level Precision** | **92.05%** | 87.17% |
| **Total Machine-Days Evaluated** | 7,280 machine-days | 7,280 machine-days |
| **False Alarms per Machine-Day** | **0.00192** (1 per 520 machine-days) | 0.00330 (1 per 303 machine-days) |
| **Median Warning Lead Time** | **24.0 hours** | 24.0 hours |
| **Mean Warning Lead Time** | **23.76 hours** | 23.80 hours |

### 5. RUL & Component Classifier Evaluation

- **Remaining Useful Life (RUL) Regressor vs Baseline** (Evaluated on unseen machines):
  - In Urgent Degradation Window ($RUL \le 24$ hours, 3,794 samples):
    - LightGBM RUL MAE: **9.05 hours** (vs Naive Baseline constant MAE of **155.54 hours**).
    - Median Absolute Error (MedAE): **3.01 hours**.
    - RMSE: **24.32 hours** (vs Baseline RMSE of **155.69 hours**).
  - Overall Test Set (174,740 samples):
    - LightGBM RUL MAE: **13.78 hours** | RMSE: **27.42 hours** | MedAE: **1.59 hours**.
- **Component Failure Mode Classifier** (Evaluated on unseen machines in pre-failure window):
  - Overall Accuracy: **94.70%** | Macro-F1: **94.56%** (7,905 test samples).
  - `comp1` (Inverter / Relay): Precision 0.9352, Recall 0.9458, **F1: 0.9405** (Support: 1,938).
  - `comp2` (Drive Shaft / Coupler): Precision 0.9414, Recall 0.9352, **F1: 0.9383** (Support: 2,592).
  - `comp3` (Pressure Valve / Pump): Precision 0.9928, Recall 0.9774, **F1: 0.9851** (Support: 1,989).
  - `comp4` (Bearing / Spindle): Precision 0.9101, Recall 0.9271, **F1: 0.9185** (Support: 1,386).

---

## Data Quality, Leakage Prevention & Provenance

1. **Temporal Boundaries & Causal Windows**:
   - Rolling sensor statistics ($3\text{h}, 12\text{h}, 24\text{h}$) strictly compute trailing windows $[t - \Delta t, t]$.
   - Target labels represent strictly forward windows $(t, t + 24\text{h}]$.
   - Training windows enforce a 24-hour temporal embargo prior to test cutoffs.
2. **Preprocessing Isolation**:
   - Median imputer statistics are computed strictly on the training partition; test data is never referenced during imputation fitting.
3. **Exact Feature Schema Alignment**:
   - Model feature width, JSON schema, and runtime API vectors are strictly validated to 47 features.
4. **Data Provenance**:
   - Benchmark Dataset: Microsoft Azure Predictive Maintenance benchmark dataset (synthetic industrial benchmark).
   - The cleaned dataset is included for reproducibility. Verify original benchmark licensing terms before proprietary commercial distribution.

---

## Realistic Maintenance Scheduling Engine

The scheduler translates raw ML predictions into actionable factory work orders:
- **Workload Balancing**: Slices prioritized machines across days based on maximum daily technician capacity ($C_{\text{day}}$).
- **Shift Assignment**: Maps slots to Morning (06:00–14:00), Swing (14:00–22:00), and Night (22:00–06:00) shifts.
- **Resource Matching**: Assigns qualified technician trade certifications (Electrical Level III, Millwright, Hydraulics Specialist, Bearing Specialist) and reserves replacement parts SKUs.
- **Feasibility Enforcement**: Computes $t_{\text{slot\_start}} + d_{\text{repair}}$. If this exceeds model-predicted RUL, the slot is marked **`INFEASIBLE`** to alert production supervisors that emergency shutdown is required before scheduled servicing.

---

## Technology Stack

- **Backend**: Python 3.11+, FastAPI, Uvicorn, Pydantic v2.
- **Machine Learning**: LightGBM, scikit-learn, joblib, NumPy, pandas, PyArrow.
- **Frontend**: React 19, Vite 8, Recharts 3, modern responsive CSS.
- **Bundle Optimization**: Code-split via Rolldown into `vendor-react` and `vendor-recharts` chunks (< 500 kB per chunk).
- **Documentation**: ReportLab PDF Generation Engine.

---

## Project Layout

```text
AI-Manufacturing-VSCode/
├── backend/
│   ├── app/
│   │   ├── main.py                     # FastAPI application & endpoints
│   │   ├── feature_pipeline.py         # 47-feature engineering pipeline
│   │   ├── evaluation_data.py          # Ground truth labeling & Parquet loading
│   │   └── model_utils.py              # Schema validation & imputation
│   ├── data/
│   │   ├── raw/cleaned_dataset/        # Raw telemetry, error, maintenance CSVs
│   │   └── phase4_outputs/             # Historical analysis outputs
│   ├── evaluation_results/             # Evaluation reports, CSVs, and SVG curves
│   │   ├── machine_held_out_report.json
│   │   ├── rolling_temporal_report.json
│   │   ├── feature_ablation_report.json
│   │   ├── event_level_report.json
│   │   ├── rul_component_report.json
│   │   └── rigorous_evaluation_summary.json
│   ├── models/
│   │   ├── feature_columns.json        # Pinned 47-feature schema
│   │   ├── imputer_statistics.json     # Pinned median imputer stats
│   │   ├── lightgbm_model.joblib        # Binary failure classifier
│   │   ├── rul_model.joblib             # Remaining Useful Life regressor
│   │   ├── component_model.joblib       # Component severity classifier
│   │   └── sensor_baseline_statistics.json
│   ├── tests/
│   │   ├── test_inference.py           # Backend inference and API tests
│   │   ├── test_evaluation.py          # Evaluation helper tests
│   │   └── test_leakage_and_validation.py # Temporal leakage & boundary tests
│   ├── evaluate_rigorous.py            # 5-part rigorous ML evaluation runner
│   ├── evaluate_temporal_holdout.py    # Chronological holdout evaluation
│   ├── train_models_complete.py        # Reproducible training script
│   └── test_models_suite.py            # End-to-end verification script
├── docs/
│   ├── AI_Manufacturing_Optimization_Project_Overview.pdf
│   └── generate_project_overview.py    # ReportLab PDF generator
└── frontend/
    ├── src/
    │   ├── App.jsx                     # Reactive dashboard UI
    │   ├── main.jsx
    │   └── style.css
    ├── package.json
    └── vite.config.js                  # Chunked production build
```

---

## Getting Started & Reproducibility

### 1. Backend Setup & API Launch

```powershell
cd AI-Manufacturing-VSCode\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH="."
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Interactive OpenAPI Swagger UI is available at `http://127.0.0.1:8000/docs`.

### 2. Frontend Dashboard Launch

```powershell
cd AI-Manufacturing-VSCode\frontend
npm install
npm run dev
```

Open your browser at `http://localhost:5173`.

### 3. Run Rigorous ML Evaluation Engine

To reproduce all unseen-machine, rolling-temporal, feature-ablation, and event-level reports:

```powershell
cd AI-Manufacturing-VSCode\backend
$env:PYTHONPATH="."
python evaluate_rigorous.py
```

All JSON reports, CSV summaries, and SVG curves will be exported to `backend/evaluation_results/`.

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `GET /api/health` | GET | System health, model readiness, data file status |
| `GET /api/overview` | GET | Global metrics, active alert count, cumulative downtime |
| `GET /api/risk-config` | GET | Risk categories, probability bands, and operational threshold |
| `GET /api/predict/machine/{id}` | GET | Real-time multi-model inference (Failure prob, RUL, Component, Health, Drivers) |
| `POST /api/predict` | POST | Direct inference on arbitrary 47-feature vector |
| `GET /api/machines/{id}/details` | GET | Detailed sensor telemetry snapshot, historical downtime, maintenance queue |
| `GET /api/historical_trends` | GET | 12-month failure distribution and component downtime breakdowns |
| `GET /api/production_lines` | GET | Machine layout and grouping across 5 factory lines |
| `GET /api/simulation/scenarios` | GET | Available factory crisis replay scenarios |
| `POST /api/simulation/scenario/{id}` | POST | Jumps factory state to selected demonstration scenario |
| `POST /api/simulation/step` | POST | Advances simulated factory line by 1 hour |
| `GET /api/simulation/state` | GET | Retrieves current replay state for page refresh restoration |
| `POST /api/simulation/reset` | POST | Resets simulation cursor to initial snapshot |
| `GET /api/alerts/active` | GET | Lists all machines currently crossing the operational alert threshold |
| `GET /api/maintenance/schedule_realistic`| GET | Capacity-limited schedule allocations, shift assignments, and feasibility status |
| `GET /api/maintenance/report` | GET | Structured dispatch plan and technician work orders |

---

## Automated Verification & Unit Tests

Run the complete test suite (28 automated tests covering model loading, inference, leakage boundaries, imputer isolation, API edge cases, and scheduling feasibility):

```powershell
cd AI-Manufacturing-VSCode\backend
$env:PYTHONPATH="."
python -m unittest discover -s tests -v
```

Execution Result:
```text
Ran 28 tests in 1.629s
OK
All ML models and feature baselines loaded successfully.
Features: 47, RUL model: True, Component model: True
```

Run the end-to-end multi-model verification suite:

```powershell
python test_models_suite.py
```

Result:
```text
======================================================================
 ALL VERIFICATIONS PASSED SUCCESSFULLY!
======================================================================
```

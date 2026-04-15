# PM2.5 Prediction ML System

A machine learning system for predicting PM2.5 (fine particulate matter) concentration levels using historical air quality monitoring data from Thailand's Pollution Control Department (กรมควบคุมมลพิษ).

**Objective:** Predict next-day PM2.5 (µg/m³) for Station 10T, Bangkok using lag features, rolling statistics, and calendar features. Includes automated monitoring and retraining when model performance degrades.

Group: Software ML
---

## Quick Start

### Step 1 — Start all services

```bash
docker compose up -d
```

Wait ~60 seconds for all services to initialize.

| Service | URL | Credentials |
|---------|-----|-------------|
| Airflow UI | http://localhost:8080 | admin / admin |
| MLflow UI | http://localhost:5001 | — |
| Prediction API | http://localhost:8001 | — |
| API Docs | http://localhost:8001/docs | — |

### Step 2 — Train the initial model

Go to **Airflow UI → DAGs → `pm25_training_pipeline` → trigger ▶**

Or via CLI:
```bash
curl -X POST http://localhost:8080/api/v1/dags/pm25_training_pipeline/dagRuns \
  -u admin:admin -H "Content-Type: application/json" \
  -d '{"dag_run_id": "initial_train"}'
```

Training runs: feature engineering → Ridge, Random Forest, XGBoost, LSTM → evaluate → export ONNX.
Check progress in Airflow grid view (~5–10 min). Results appear in MLflow at http://localhost:5001.

### Step 3 — Test the full pipeline with mock data

```bash
# Normal mode — model stays healthy, no retrain triggered
python scripts/mock_pipeline.py --mode normal --days 25

# Degraded mode — MAE spikes 2–3×, auto-retrain triggered
python scripts/mock_pipeline.py --mode degraded --days 25

# Drift mode — gradual PM2.5 shift, PSI rises, may trigger retrain
python scripts/mock_pipeline.py --mode drift --days 30
```

The script automatically:
1. Generates mock PM2.5 data
2. Sends predictions (`/predict`) + actuals (`/actual`) to the API
3. Triggers `pm25_pipeline` monitoring DAG via Airflow REST API
4. Polls until complete and prints MAE / PSI summary

### Step 4 — Check monitoring results

```bash
cat results/monitoring_results.csv
```

Or in Airflow UI → `pm25_pipeline` → grid → `check_mae_and_psi` task → **XCom** tab.

---

## Dataset

| Item | Detail |
|------|--------|
| Source | กรมควบคุมมลพิษ (Pollution Control Department, Thailand) |
| Files | `PM2.5(2024).xlsx` (training), `PM2.5(2025).xlsx` (testing) |
| Stations | 96 monitoring stations across Thailand |
| Granularity | Daily PM2.5 per station |
| Selected Station | **10T** — เคหะชุมชนคลองจั่น, เขตบางกะปิ, กทม. |

---

## Repository Structure

```
pm25-prediction-ml-system/
├── configs/
│   └── config.yaml                   # All parameters: models, monitoring, paths
├── dags/
│   ├── pm25_training_dag.py          # Training pipeline DAG
│   └── pm25_pipeline_dag.py          # Unified monitoring + auto-retraining DAG
├── data/
│   ├── raw/                          # PM2.5(2024).xlsx, PM2.5(2025).xlsx
│   └── processed/                    # Parquet files shared between Airflow tasks
├── docker/
│   └── init-db.sql                   # Postgres DB init (airflow + mlflow schemas)
├── models/                           # Saved .joblib models, lstm.pt, feature_columns.json
│   └── onnx/                         # ONNX-exported models
├── results/
│   ├── experiment_results.csv        # Model comparison metrics
│   ├── predictions_log.csv           # Logged predictions (written by /predict)
│   ├── actuals_log.csv               # Logged actuals (written by /actual)
│   └── monitoring_results.csv        # Monitoring run history
├── scripts/
│   ├── mock_pipeline.py              # All-in-one end-to-end test script
│   ├── run_mock_pipeline.py          # CSV-based pipeline test
│   └── generate_mock_data.py         # Generate mock CSV files
├── src/
│   ├── api.py                        # FastAPI service (/predict /actual /retrain)
│   ├── data_loader.py                # Load Excel data per station
│   ├── preprocessing.py              # ffill/bfill, clip [0, 500] µg/m³
│   ├── feature_engineering.py        # 17 features with shift(1) to prevent leakage
│   ├── train.py                      # GridSearchCV + TimeSeriesSplit training
│   ├── evaluate.py                   # MAE, RMSE, R²
│   ├── predict.py                    # CLI inference
│   ├── monitor.py                    # MAE + PSI monitoring
│   ├── lstm_model.py                 # PyTorch LSTM with skorch
│   ├── export_onnx.py                # Export all models to ONNX
│   └── predict_onnx.py               # ONNX inference
├── Dockerfile                        # Airflow image
├── Dockerfile.api                    # Lightweight API image
├── docker-compose.yml                # Full stack
└── tests/
    └── test_preprocessing.py
```

---

## API Reference

### `GET /health`
Liveness check.
```json
{"status": "ok"}
```

### `GET /model/info`
Returns loaded model name and feature list.

### `POST /predict`
Send ≥15 days of PM2.5 history, get next-day forecast. Logs prediction to `predictions_log.csv`.

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "history": [
      {"date": "2025-06-01", "pm25": 42.1},
      {"date": "2025-06-02", "pm25": 38.5},
      ...
      {"date": "2025-06-15", "pm25": 36.8}
    ]
  }'
```

```json
{
  "prediction_date": "2025-06-16",
  "predicted_pm25": 34.21,
  "unit": "µg/m³",
  "model": "random_forest"
}
```

### `POST /actual`
Record ground truth after measurement. Logs to `actuals_log.csv` and returns absolute error against the matched prediction.

```bash
curl -X POST http://localhost:8001/actual \
  -H "Content-Type: application/json" \
  -d '{"date": "2025-06-16", "pm25_actual": 36.5}'
```

### `POST /retrain`
Joins prediction+actual logs, computes MAE, triggers Airflow DAG if MAE > threshold.

```bash
curl -X POST http://localhost:8001/retrain \
  -H "Content-Type: application/json" \
  -d '{"threshold": 6.0, "min_pairs": 7}'
```

---

## Monitoring & Auto-Retraining

The `pm25_pipeline` DAG runs daily at 01:00 UTC. It checks two metrics on the rolling 30-day window of matched prediction+actual pairs:

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| MAE | > 6.0 µg/m³ | Prediction accuracy degraded |
| PSI | > 0.2 | Significant distribution shift |

**PSI thresholds:**
- PSI < 0.1 — stable
- PSI 0.1–0.2 — moderate shift (monitor)
- PSI > 0.2 — significant shift → retrain

**DAG flow:**
```
pm25_pipeline (daily @ 01:00 UTC)
  └── export_data
        └── check_mae_and_psi
              ├── needs_retrain → pm25_training_pipeline → clear_logs
              └── healthy       → (no action)
```

After retraining, both log files are cleared so stale degraded data doesn't re-trigger retraining on the next run.

Configure thresholds in `configs/config.yaml`:
```yaml
monitoring:
  rolling_window_days: 30
  min_evaluation_pairs: 7
  mae:
    enabled: true
    threshold: 6.0
  psi:
    enabled: true
    threshold: 0.2
    bins: 10
```

---

## Experiment Results

Station **10T** | Train: 2024 (359 days) → Test: 2025 (174 days)

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression (Baseline) | 5.1348 | 6.7493 | 0.7726 |
| Ridge Regression | 4.8286 | 6.5294 | 0.7871 |
| **Random Forest** ⭐ | **4.5869** | 6.6809 | 0.7772 |
| XGBoost | 4.9735 | 7.3464 | 0.7305 |
| LSTM | 6.2156 | 8.2195 | 0.6627 |

**Best model:** Random Forest (MAE = 4.59 µg/m³) — used by default in the API.

---

## Features (17 total)

| Category | Features |
|----------|----------|
| Lag | `pm25_lag_1/2/3/5/7` |
| Rolling mean | `pm25_rolling_mean_3/7/14` |
| Rolling std | `pm25_rolling_std_3/7/14` |
| Time | `day_of_week`, `month`, `day_of_year`, `is_weekend` |
| Change | `pm25_diff_1`, `pm25_pct_change_1` |

All features use `shift(1)` to prevent data leakage.

---

## Local Development (without Docker)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
brew install libomp          # macOS only (XGBoost)

# Start MLflow
mlflow server --host 127.0.0.1 --port 5001 --backend-store-uri mlruns/ &

# Train
PYTHONPATH=src python src/train.py

# Start API
PYTHONPATH=src uvicorn src.api:app --host 0.0.0.0 --port 8001

# Run tests
python tests/test_preprocessing.py
```

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| ML | scikit-learn, XGBoost, PyTorch, skorch |
| Serving | FastAPI, uvicorn |
| Orchestration | Apache Airflow (LocalExecutor) |
| Experiment tracking | MLflow |
| Infrastructure | Docker Compose, PostgreSQL |
| Export | ONNX, onnxruntime |

---

## License

This project is for academic purposes (ML Systems course).

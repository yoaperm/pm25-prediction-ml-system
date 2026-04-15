# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Local Development
```bash
# Install dependencies
pip install -r requirements-base.txt -r requirements-ml.txt

# Run tests
pytest tests/ -v

# Run a single test
pytest tests/test_preprocessing.py::test_handle_missing_ffill -v

# Lint
ruff check src/ tests/ --select E,F,W

# Run FastAPI server
uvicorn src.api:app --reload --port 8001

# Run Streamlit dashboard
streamlit run streamlit_app.py
```

### Docker
```bash
# Start full stack (Postgres, MLflow, Airflow, Triton, API, Streamlit)
docker compose up --build

# Services: Airflow UI @ :8080, MLflow @ :5001, API @ :8001, Dashboard @ :8501, Triton @ :8010
```

### ML Pipeline
```bash
# Train single-station (station 10T, outputs to models/ and logs to MLflow)
PYTHONPATH=src python -c "from src.train import train_all_models; from src.data_loader import load_config; train_all_models(load_config('configs/config.yaml'))"

# Train multi-station (stations 63-67, daily aggregated from hourly CSVs)
PYTHONPATH=src python scripts/train_multi_station.py
PYTHONPATH=src python scripts/train_multi_station.py --stations 63 65

# Train 24-hour simultaneous forecast (multi-output, hourly data)
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_DEVICE=cpu \
  python scripts/train_24h_forecast.py --stations 63

# Export models to ONNX
PYTHONPATH=src python src/export_onnx.py
```

## Architecture

PM2.5 air quality prediction system for Thai monitoring stations. Predicts next-day PM2.5 using daily data (station 10T pipeline) or next 24 hours simultaneously using hourly data (multi-station pipeline).

### System Components

**FastAPI service** (`src/api.py`) is the inference core:
- Loads active model at startup via `models/active_model.json` (ONNX or Triton)
- Requires `X-API-Key` header for all non-health endpoints
- `POST /predict` — takes ≥15 days of PM2.5 history, returns next-day forecast; logs to `results/predictions_log.csv`
- `POST /actual` — logs ground truth to `results/actuals_log.csv`
- `POST /retrain` — computes rolling 30-day MAE and PSI; triggers Airflow DAG `pm25_training_pipeline` if MAE > 6.0 or PSI > 0.2
- Inference backend: Triton (default in Docker) or onnxruntime (local)

**Streamlit dashboard** (`streamlit_app.py`) calls the FastAPI backend. Three pages: Predict, Results (model comparison), Monitoring (MAE/PSI trends).

**Airflow DAGs** (`dags/`):
- `pm25_training_dag.py` — manual trigger; trains all 5 models for station 10T; each task saves a temp ONNX `_tmp_{key}.onnx`, then evaluate → compare_and_deploy
- `pm25_station_training_dag.py` — same pipeline parameterized for stations 63–67 via Airflow UI (`Param(type="integer", enum=[63,64,65,66,67])`)
- `pm25_pipeline_dag.py` — daily at 01:00 UTC; monitors drift and conditionally retrains
- `pm25_hourly_ingest_dag.py` — hourly ingestion from AirBKK API into PostgreSQL

### ML Pipeline Flow

```
data/raw/ (Excel for station 10T, long CSV for stations 63-67)
  → src/data_loader.py          load per-station data
  → src/preprocessing.py        ffill/bfill, clip [0,500]
  → src/feature_engineering.py  build features (see below)
  → src/train.py                GridSearchCV + TimeSeriesSplit(3)
  → src/evaluate.py             MAE, RMSE, R²
  → models/onnx/*.onnx + models/active_model.json
```

**Station 10T features (17):** lags [1,2,3,5,7 days], rolling mean/std [3,7,14 days], month, day_of_year, day_of_week. All use `shift(1)` to prevent leakage.

**24h forecast features (19):** lags [1,2,3,6,12,24h], rolling mean/std [6,12,24h], diff [1h,24h], time features (hour, day_of_week, month, day_of_year, is_weekend). Targets are `pm25_h1`…`pm25_h24` (shift(-1) to shift(-24)).

### Deployment Pointer — `active_model.json`

Every model directory has an `active_model.json`:
```json
{"onnx_file": "xgboost_2020-01-02_2024-12-31.onnx", "model_key": "xgboost",
 "train_start": "...", "train_end": "...", "is_lstm": false}
```
- Station 10T: `models/active_model.json`
- Per-station: `models/station_{id}/active_model.json`
- 24h forecast: `models/station_{id}_24h/active_model.json`

New models deploy only if MAE improves over the current production ONNX. Old versioned ONNX files are kept (not deleted).

### Key Design Decisions

- **ONNX-only deployment** — no joblib/pt files at inference time; all models exported to ONNX
- **`shift(1)` on all features** — critical for preventing leakage in time-series
- **LSTM** uses PyTorch via skorch wrapper (`src/lstm_model.py`); export uses `torch.onnx.export`
- **XGBoost multi-output** (24h pipeline) uses native 2D-Y support, exported via `onnxmltools.convert_xgboost`
- **OMP thread conflict** — always run training scripts with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` to prevent hang between XGBoost GridSearchCV workers and PyTorch LSTM training
- **Apple Silicon MPS crash** — set `PYTORCH_DEVICE=cpu` env var; handled in `src/lstm_model.py`
- **GRID_N_JOBS** env var controls GridSearchCV parallelism (default -1); reduce to 2 on memory-constrained machines
- **Monitoring** uses PSI (Population Stability Index) for feature drift detection, not just accuracy
- `configs/config.yaml` is the single source of truth for all pipeline parameters

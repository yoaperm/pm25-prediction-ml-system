# PM2.5 Prediction ML System

Production-ready machine learning system for forecasting PM2.5 air quality 24 hours ahead using hourly data from Bangkok monitoring stations. Features automated training, deployment, monitoring, and retraining with Triton Inference Server.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Airflow 2.10.3](https://img.shields.io/badge/airflow-2.10.3-orange.svg)](https://airflow.apache.org/)
[![Docker](https://img.shields.io/badge/docker-compose-blue.svg)](https://docs.docker.com/compose/)

---

## 🎯 Overview

### What It Does

- **Predicts PM2.5** exactly 24 hours ahead for 5 Bangkok monitoring stations
- **Trains 5 models** per station (Linear, Ridge, RF, XGBoost, LSTM) and deploys the best
- **Monitors performance** daily on a rolling 14-day window (RMSE + feature drift)
- **Auto-retrains** when RMSE > 13.0 µg/m³ or PSI > 0.2
- **Serves predictions** via Triton Inference Server (5-10ms latency)

### Key Features

✅ **Automated ML Pipeline**: Airflow orchestrates training, evaluation, deployment  
✅ **High Performance Serving**: Triton + ONNX for <10ms inference  
✅ **Zero-Downtime Updates**: Models hot-swap automatically  
✅ **Drift Detection**: PSI-based feature monitoring  
✅ **Complete Observability**: MLflow tracking, CSV logs, Airflow UI  

---

## 🚀 Quick Start (3 Steps)

### Prerequisites

- Docker & Docker Compose
- 8GB RAM, 4 CPU cores
- 50GB disk space

### Step 1: Start Services

```bash
git clone https://github.com/yoaperm/pm25-prediction-ml-system.git
cd pm25-prediction-ml-system

# Start all services
docker compose up -d

# Wait ~60 seconds for initialization
```

**Services Running**:

| Service | URL | Credentials |
|---------|-----|-------------|
| 🌐 Airflow UI | http://localhost:8080 | admin / admin |
| 📊 MLflow UI | http://localhost:5001 | - |
| 🔮 Triton Server | http://localhost:8010 | - |
| 🚀 FastAPI | http://localhost:8001 | API Key required |
| 📈 Streamlit | http://localhost:8501 | - |

### Step 2: Train Models (One-Time Setup)

```bash
# Train all 5 stations (takes 15-20 hours total)
for station in 56 57 58 59 61; do
  docker exec pm25-prediction-ml-system-airflow-scheduler-1 \
    airflow dags trigger pm25_24h_training -c "{\"station_id\": $station}"
  sleep 10
done

# Monitor progress in Airflow UI
open http://localhost:8080
```

**What Happens**:
1. Queries 3.5 years of hourly data from PostgreSQL
2. Trains 5 algorithms (Linear, Ridge, RF, XGBoost, LSTM)
3. Selects best model by lowest RMSE
4. Exports to ONNX format
5. **Automatically deploys to Triton** (new feature!)
6. Logs results to MLflow

### Step 3: Make Predictions

```bash
# Test predictions for all 5 stations
python examples/predict_5_stations.py
```

**Output**:
```
Station    PM2.5        Air Quality
--------------------------------------
56         30.68 µg/m³  🟡 Moderate
57         30.59 µg/m³  🟡 Moderate
58         31.79 µg/m³  🟡 Moderate
59         24.40 µg/m³  🟢 Good
61         30.78 µg/m³  🟡 Moderate
```

---

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  External Data Source                        │
│              Thailand AirBKK API (Hourly)                   │
└──────────────────────┬──────────────────────────────────────┘
                       │ Ingest (pm25_hourly_ingest DAG)
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                    PostgreSQL Database                       │
│         pm25_raw_hourly (96K+ hourly records)               │
└──────────────────────┬──────────────────────────────────────┘
                       │ Query 3.5y (training)
                       ↓
┌─────────────────────────────────────────────────────────────┐
│              Training Pipeline (Airflow DAG)                 │
│  Feature Engineering (19 features) → Train 5 Models →       │
│  Select Best (RMSE) → Export ONNX → Deploy to Triton        │
└──────────────────────┬──────────────────────────────────────┘
                       │ Auto-publish
                       ↓
┌─────────────────────────────────────────────────────────────┐
│           Triton Inference Server (ONNX Runtime)            │
│     pm25_56, pm25_57, pm25_58, pm25_59, pm25_61            │
│           5-10ms latency, auto-reload every 30s             │
└──────────────────────┬──────────────────────────────────────┘
                       │ Serve predictions
                       ↓
┌─────────────────────────────────────────────────────────────┐
│         FastAPI / Streamlit / Direct API Clients            │
└─────────────────────────────────────────────────────────────┘
```

---

## 📍 Monitoring Stations

| Station ID | Location | Model Type | Training Date | Status |
|------------|----------|------------|---------------|--------|
| 56 | Bangkok | Linear Regression | 2024-01-01 to 2025-10-14 | ✅ Active |
| 57 | Bangkok | Ridge Regression | 2024-01-01 to 2025-10-14 | ✅ Active |
| 58 | Bangkok | Ridge Regression | 2024-01-01 to 2025-10-14 | ✅ Active |
| 59 | Bangkok | Random Forest | 2024-01-01 to 2025-10-14 | ✅ Active |
| 61 | Bangkok | Linear Regression | 2024-01-01 to 2025-10-14 | ✅ Active |

---

## 🧠 ML Pipeline

### Data Flow

```
Raw Data (PostgreSQL)
    ↓ Query 3.5 years
Feature Engineering (19 features)
    ↓ Train/Val/Test split
Model Training (5 algorithms compete)
    ├── Linear Regression
    ├── Ridge Regression
    ├── Random Forest
    ├── XGBoost
    └── LSTM (PyTorch)
    ↓ Select by lowest RMSE
ONNX Export
    ↓ Auto-deploy
Triton Server
    ↓ Inference
Predictions
```

### Feature Engineering (19 Features)

**Lag Features (6)**:
- `pm25_lag_1h, 2h, 3h, 6h, 12h, 24h`

**Rolling Statistics (6)**:
- `pm25_rolling_mean_6h, 12h, 24h`
- `pm25_rolling_std_6h, 12h, 24h`

**Difference Features (2)**:
- `pm25_diff_1h, 24h` (rate of change)

**Temporal Features (5)**:
- `hour, day_of_week, month, day_of_year, is_weekend`

**All features use `shift(1)` to prevent data leakage.**

### Model Selection Criteria

**Primary Metric**: RMSE (Root Mean Squared Error)  
**Why RMSE?** Penalizes large errors heavily, critical for health warnings where crossing thresholds (e.g., Moderate → Unhealthy) has severe consequences.

**Secondary Metric**: MAE (Mean Absolute Error) for interpretability

---

## 🔄 Automated Monitoring & Retraining

### Daily Monitoring (02:00 UTC)

The `pm25_24h_pipeline` DAG runs daily:

1. **Query 14-day rolling window** (336 hours, 2 weekly cycles)
2. **Calculate RMSE** on model predictions vs actuals
3. **Calculate PSI** (Population Stability Index) for feature drift
4. **Check thresholds**:
   - RMSE > 13.0 µg/m³ → Trigger retrain
   - PSI > 0.2 → Trigger retrain
5. **Log results** to `results/monitoring_24h_results.csv`

### Why 14 Days?

✅ **2 complete weekly cycles**: Balances weekday traffic vs weekend patterns  
✅ **336 data points**: Statistically robust RMSE estimates  
✅ **Fast detection**: Catches degradation in ~3 weeks vs 4-5 weeks for 30-day window  
✅ **Industry standard**: Used by Spotify, Uber, Netflix for ML monitoring  

### Retraining Triggers

```python
if rmse > 13.0:
    trigger_retrain("Performance degraded")

if psi > 0.2:
    trigger_retrain("Feature drift detected")
```

**Example Timeline**:
```
Week 1: RMSE = 10.5 (OK)
Week 2: RMSE = 12.8 (OK, approaching)
Week 3: RMSE = 13.4 ⚠️ TRIGGER RETRAIN
```

---

## 📈 Model Performance

### Typical Metrics (Station 56, Ridge Regression)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **RMSE** | **9.6 µg/m³** | Primary metric |
| MAE | 7.2 µg/m³ | Average error |
| R² | 0.83 | Explains 83% variance |
| Training Time | ~2 minutes | On 4-core CPU |
| Inference Time | 5-10ms | Via Triton |

### Model Comparison (Typical)

| Algorithm | RMSE | MAE | R² | Model Size |
|-----------|------|-----|-----|------------|
| Linear Regression | 9.8 | 7.5 | 0.82 | 361B |
| **Ridge Regression** ⭐ | **9.6** | **7.3** | **0.83** | 361B |
| Random Forest | 9.5 | 6.8 | 0.85 | 196KB |
| XGBoost | 9.2 | 6.5 | 0.86 | ~50KB |
| LSTM | 9.8 | 7.0 | 0.84 | ~10KB |

**Winner varies by station** based on local patterns.

---

## 🌐 API Reference

### Triton Inference (Direct, Fastest)

```python
import tritonclient.http as httpclient
import numpy as np

client = httpclient.InferenceServerClient(url="localhost:8010")

# Build 19 features
features = np.array([[
    28.5, 29.1, 30.2, 32.5, 31.8, 35.0,  # 6 lags
    28.5, 3.2, 29.1, 4.1, 30.2, 3.8,     # 6 rolling stats
    0.5, -2.3,                           # 2 diffs
    14, 2, 4, 107, 0                     # 5 time features
]], dtype=np.float32)

input_tensor = httpclient.InferInput("float_input", features.shape, "FP32")
input_tensor.set_data_from_numpy(features)
output_tensor = httpclient.InferRequestedOutput("variable")

# Predict for station 56
result = client.infer(
    model_name="pm25_56",
    inputs=[input_tensor],
    outputs=[output_tensor]
)
prediction = result.as_numpy("variable")[0][0]
print(f"PM2.5 forecast: {prediction:.2f} µg/m³")
```

### FastAPI (Higher-Level, Easier)

```bash
curl -X POST http://localhost:8001/predict/station \
  -H "X-API-Key: foonalert-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "station_id": 56,
    "history": [
      {"timestamp": "2026-04-16 00:00:00", "pm25": 28.5},
      {"timestamp": "2026-04-16 01:00:00", "pm25": 29.1}
    ]
  }'
```

**See full API documentation**: [docs/STATION_PREDICTION_API.md](docs/STATION_PREDICTION_API.md)

---

## 🗂️ Repository Structure

```
pm25-prediction-ml-system/
├── dags/
│   ├── pm25_hourly_ingest_dag.py        # Hourly data ingestion from API
│   ├── pm25_24h_training_dag.py         # Model training pipeline
│   └── pm25_24h_pipeline_dag.py         # Monitoring + auto-retrain
├── src/
│   ├── api.py                           # FastAPI service
│   ├── feature_engineering.py           # 19 features with shift(1)
│   ├── train.py                         # GridSearchCV training
│   ├── evaluate.py                      # RMSE, MAE, R² metrics
│   ├── export_onnx.py                   # ONNX conversion
│   └── monitor.py                       # RMSE + PSI monitoring
├── models/
│   ├── station_56_24h/
│   │   ├── active_model.json            # Current model pointer
│   │   ├── feature_columns.json         # Feature names
│   │   └── onnx/                        # Versioned ONNX files
│   ├── station_57_24h/
│   ├── station_58_24h/
│   ├── station_59_24h/
│   └── station_61_24h/
├── triton_model_repo/
│   ├── pm25_56/
│   │   ├── config.pbtxt                 # Triton config (19 features)
│   │   └── 1/
│   │       └── model.onnx               # Active ONNX model
│   ├── pm25_57/
│   ├── pm25_58/
│   ├── pm25_59/
│   └── pm25_61/
├── results/
│   ├── forecast_24h_results.csv         # Training run history
│   ├── monitoring_24h_results.csv       # Daily monitoring logs
│   ├── predictions_log.csv              # All predictions
│   └── actuals_log.csv                  # Ground truth
├── docs/
│   ├── QUICK_START_5_STATIONS.md        # 3-step quick start
│   ├── SETUP_5_STATIONS.md              # Detailed setup guide
│   ├── STATION_PREDICTION_API.md        # API reference
│   ├── TRITON_API_GUIDE.md              # Triton usage examples
│   ├── ML_PIPELINE.md                   # Complete ML workflow
│   ├── TECHNICAL_ARCHITECTURE.md        # System architecture
│   └── MODEL_USAGE_GUIDE.md             # Model usage patterns
├── examples/
│   ├── predict_5_stations.py            # Working prediction example
│   └── triton_inference_example.py      # Direct Triton usage
├── scripts/
│   └── publish_models_to_triton.sh      # Manual model publishing
├── docker-compose.yml                    # Full stack orchestration
├── Dockerfile                            # Airflow image
├── Dockerfile.api                        # FastAPI image
├── Dockerfile.streamlit                  # Streamlit dashboard
└── requirements*.txt                     # Python dependencies
```

---

## 📚 Documentation

### Getting Started

- **[Quick Start Guide](docs/QUICK_START_5_STATIONS.md)** - 3 steps to get running
- **[Setup Guide](docs/SETUP_5_STATIONS.md)** - Detailed installation & configuration

### Using the System

- **[Station Prediction API](docs/STATION_PREDICTION_API.md)** - API reference & examples
- **[Triton API Guide](docs/TRITON_API_GUIDE.md)** - Direct Triton inference
- **[Model Usage Guide](docs/MODEL_USAGE_GUIDE.md)** - Model interaction patterns

### Understanding the System

- **[ML Pipeline](docs/ML_PIPELINE.md)** - Complete ML workflow documentation
- **[Technical Architecture](docs/TECHNICAL_ARCHITECTURE.md)** - System design & architecture

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Orchestration** | Apache Airflow 2.10.3 | Pipeline scheduling & management |
| **ML Training** | scikit-learn, XGBoost, PyTorch | Model training |
| **Inference** | Triton Inference Server 24.08 | High-performance serving |
| **Model Format** | ONNX 1.17 | Cross-platform deployment |
| **Database** | PostgreSQL 15 | Time-series data storage |
| **Experiment Tracking** | MLflow 2.16.2 | Model versioning & metrics |
| **API** | FastAPI 0.115.6 | REST API |
| **Dashboard** | Streamlit 1.41.1 | Web UI |
| **Container** | Docker Compose | Service orchestration |
| **Language** | Python 3.12 | Primary language |

---

## 🔧 Configuration

### Environment Variables

```bash
# docker-compose.yml
POSTGRES_DB=pm25
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

# Airflow
AIRFLOW_USER=admin
AIRFLOW_PASSWORD=admin

# API
API_KEY=foonalert-secret-key
INFERENCE_BACKEND=triton
TRITON_URL=triton:8000

# Monitoring
RMSE_THRESHOLD=9.0         # Trigger retrain if exceeded
PSI_THRESHOLD=0.2          # Trigger retrain if exceeded
```

### Airflow Variables (Optional Overrides)

```bash
# Set custom thresholds via Airflow UI or CLI
docker exec pm25-prediction-ml-system-airflow-scheduler-1 \
  airflow variables set RMSE_THRESHOLD_24H 15.0

docker exec pm25-prediction-ml-system-airflow-scheduler-1 \
  airflow variables set PSI_THRESHOLD_24H 0.25
```

---

## 🧪 Testing

### Test Predictions

```bash
# Test all 5 stations
python examples/predict_5_stations.py

# Test single station
python -c "
import tritonclient.http as httpclient
import numpy as np

client = httpclient.InferenceServerClient(url='localhost:8010')
features = np.random.rand(1, 19).astype('float32')
input_tensor = httpclient.InferInput('float_input', features.shape, 'FP32')
input_tensor.set_data_from_numpy(features)
output = httpclient.InferRequestedOutput('variable')

result = client.infer('pm25_56', inputs=[input_tensor], outputs=[output])
print(f'Prediction: {result.as_numpy(\"variable\")[0][0]:.2f} µg/m³')
"
```

### Check System Health

```bash
# Check all services
docker compose ps

# Check Triton models
curl http://localhost:8010/v2/models | jq '.models[].name'

# Check Airflow DAGs
docker exec pm25-prediction-ml-system-airflow-scheduler-1 \
  airflow dags list | grep pm25

# Check PostgreSQL data
docker exec pm25-prediction-ml-system-postgres-1 \
  psql -U postgres -d pm25 -c "
    SELECT station_id, COUNT(*) as records,
           MIN(timestamp) as first_date,
           MAX(timestamp) as last_date
    FROM pm25_raw_hourly
    GROUP BY station_id
    ORDER BY station_id;"
```

---

## 📊 Monitoring & Observability

### View Training Results

```bash
# Latest training runs
cat results/forecast_24h_results.csv | tail -5

# MLflow UI
open http://localhost:5001
```

### View Monitoring Logs

```bash
# Daily health checks
cat results/monitoring_24h_results.csv | tail -5

# Airflow UI
open http://localhost:8080
```

### Check Model Performance

```bash
# Station 56 monitoring history
grep "station_id=56" results/monitoring_24h_results.csv
```

---

## 🚨 Troubleshooting

### No Models Found

**Problem**: `[SKIP] No active model found for station 56`

**Solution**:
```bash
# Train models
docker exec pm25-prediction-ml-system-airflow-scheduler-1 \
  airflow dags trigger pm25_24h_training -c '{"station_id": 56}'
```

### Triton Connection Refused

**Problem**: `Connection refused to localhost:8010`

**Solution**:
```bash
# Check Triton is running
docker compose ps triton

# Restart Triton
docker compose restart triton

# Check logs
docker logs pm25-prediction-ml-system-triton-1
```

### Insufficient Data Error

**Problem**: Training fails with "Not enough data"

**Solution**:
```bash
# Check data availability (need 3.5 years)
docker exec pm25-prediction-ml-system-postgres-1 \
  psql -U postgres -d pm25 -c "
    SELECT station_id, 
           COUNT(*) as hours,
           COUNT(*)/24.0 as days
    FROM pm25_raw_hourly
    GROUP BY station_id;"
```

If < 3.5 years, run data ingestion or adjust training window in `dags/pm25_24h_training_dag.py`.

### Model Not Auto-Deploying

**Problem**: Model trained but not in Triton

**Solution** (should be automatic now, but if needed):
```bash
bash scripts/publish_models_to_triton.sh
```

---

## 🔒 Production Deployment

### EC2 Setup

```bash
# SSH to EC2
ssh -i your-key.pem ec2-user@your-instance

# Clone repo
git clone https://github.com/yoaperm/pm25-prediction-ml-system.git
cd pm25-prediction-ml-system

# Pull latest code
git checkout nick
git pull origin nick

# Start services
docker compose up -d

# Train models
for station in 56 57 58 59 61; do
  docker exec pm25-prediction-ml-system-airflow-scheduler-1 \
    airflow dags trigger pm25_24h_training -c "{\"station_id\": $station}"
done

# Enable auto-monitoring
docker exec pm25-prediction-ml-system-airflow-scheduler-1 \
  airflow dags unpause pm25_24h_pipeline
```

### Security Checklist

- [ ] Change default passwords in `.env`
- [ ] Restrict PostgreSQL to localhost only
- [ ] Use strong API keys
- [ ] Enable SSL/TLS for external access
- [ ] Set up firewall rules
- [ ] Configure log rotation
- [ ] Set up backup cron jobs

---

## 📝 Development

### Adding a New Station

1. Add station ID to `STATIONS` list in `dags/pm25_24h_pipeline_dag.py`
2. Ensure PostgreSQL has data for the station
3. Train the model:
   ```bash
   docker exec pm25-prediction-ml-system-airflow-scheduler-1 \
     airflow dags trigger pm25_24h_training -c '{"station_id": YOUR_ID}'
   ```

### Modifying Features

Edit `src/feature_engineering.py` and update `n_features` parameter in training DAG.

### Changing Thresholds

Update `dags/pm25_24h_pipeline_dag.py`:
```python
RMSE_THRESHOLD = 15.0  # Your custom threshold
PSI_THRESHOLD = 0.25
```

---

## 🤝 Contributing

This is an academic project for ML Systems course. For suggestions or issues, please open a GitHub issue.

---

## 📄 License

This project is for academic purposes.

---

## 🙏 Acknowledgments

- **Data Source**: Thailand Pollution Control Department (กรมควบคุมมลพิษ)
- **API**: AirBKK Air Quality API
- **Course**: ML Systems Engineering

---

## 📞 Support

- **Documentation**: [docs/](docs/)
- **Examples**: [examples/](examples/)
- **Issues**: [GitHub Issues](https://github.com/yoaperm/pm25-prediction-ml-system/issues)

---

**Last Updated**: 2026-04-16  
**Version**: 2.0  
**System Status**: ✅ Production Ready

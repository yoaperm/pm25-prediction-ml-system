# Technical Architecture Documentation

Complete system architecture for PM2.5 prediction ML system.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Component Details](#component-details)
4. [Data Architecture](#data-architecture)
5. [ML System Architecture](#ml-system-architecture)
6. [Deployment Architecture](#deployment-architecture)
7. [API Architecture](#api-architecture)
8. [Infrastructure](#infrastructure)
9. [Security](#security)
10. [Scalability](#scalability)

---

## System Overview

### High-Level Description

PM2.5 Prediction ML System is an end-to-end machine learning platform for forecasting air quality (PM2.5 levels) 24 hours ahead for 5 monitoring stations in Bangkok, Thailand. The system handles:

- **Data ingestion**: Hourly data from AirBKK API
- **ML training**: 5 competing algorithms per station
- **Model deployment**: Automatic ONNX export and Triton serving
- **Monitoring**: Performance tracking and drift detection
- **Auto-retraining**: Triggered by degradation or drift

### Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Orchestration** | Apache Airflow 2.10.3 | Pipeline scheduling and management |
| **ML Framework** | scikit-learn, XGBoost, PyTorch | Model training |
| **Inference** | Triton Inference Server 24.08 | Low-latency model serving |
| **Model Format** | ONNX | Cross-platform model deployment |
| **Database** | PostgreSQL 15 | Time-series data storage |
| **Experiment Tracking** | MLflow 2.16.2 | Model versioning and metrics |
| **API** | FastAPI 0.115.6 | REST API for predictions |
| **Dashboard** | Streamlit 1.41.1 | Web UI for monitoring |
| **Container** | Docker Compose | Service orchestration |
| **Language** | Python 3.12 | Primary language |

---

## Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              External Layer                                 │
│  ┌──────────────┐         ┌──────────────┐         ┌──────────────┐       │
│  │  AirBKK API  │         │   Web Users  │         │  API Clients │       │
│  └──────┬───────┘         └──────┬───────┘         └──────┬───────┘       │
│         │                        │                        │                │
└─────────┼────────────────────────┼────────────────────────┼────────────────┘
          │                        │                        │
┌─────────┼────────────────────────┼────────────────────────┼────────────────┐
│         │                Application Layer                │                │
│         ↓                        ↓                        ↓                │
│  ┌──────────────┐        ┌──────────────┐        ┌──────────────┐        │
│  │   Airflow    │◄───────┤  Streamlit   │───────►│   FastAPI    │        │
│  │  (Scheduler) │        │  Dashboard   │        │   (Inference)│        │
│  └──────┬───────┘        └──────┬───────┘        └──────┬───────┘        │
│         │                       │                       │                 │
│         │ Trigger               │ Query                 │ Inference       │
│         │ DAGs                  │ Metrics               │ Request         │
│         ↓                       ↓                       ↓                 │
│  ┌──────────────┐        ┌──────────────┐        ┌──────────────┐        │
│  │   Airflow    │───────►│    MLflow    │        │    Triton    │        │
│  │   Workers    │ Log    │   Tracking   │        │   Server     │        │
│  │              │ Metrics│              │        │              │        │
│  └──────┬───────┘        └──────────────┘        └──────┬───────┘        │
│         │                                               │                 │
└─────────┼───────────────────────────────────────────────┼─────────────────┘
          │                                               │
┌─────────┼───────────────────────────────────────────────┼─────────────────┐
│         │              Data & Model Layer               │                 │
│         ↓                                               ↓                 │
│  ┌──────────────────────────────────────────────────────────────┐        │
│  │                      PostgreSQL                               │        │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │        │
│  │  │pm25_raw      │  │predictions   │  │actuals       │       │        │
│  │  │_hourly       │  │_log          │  │_log          │       │        │
│  │  └──────────────┘  └──────────────┘  └──────────────┘       │        │
│  └───────────────────────────────────────────────────────────────┘        │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────┐        │
│  │                   Shared Volumes (Docker)                     │        │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │        │
│  │  │ models/      │  │triton_model  │  │ results/     │       │        │
│  │  │              │  │_repo/        │  │              │       │        │
│  │  │station_56_24h│  │pm25_56/      │  │*.csv logs    │       │        │
│  │  │station_57_24h│  │pm25_57/      │  │              │       │        │
│  │  │station_58_24h│  │pm25_58/      │  │              │       │        │
│  │  │station_59_24h│  │pm25_59/      │  │              │       │        │
│  │  │station_61_24h│  │pm25_61/      │  │              │       │        │
│  │  └──────────────┘  └──────────────┘  └──────────────┘       │        │
│  └───────────────────────────────────────────────────────────────┘        │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Apache Airflow

**Purpose**: Orchestrates all ML pipelines

**Services**:
- `airflow-scheduler`: Triggers DAGs on schedule
- `airflow-webserver`: Web UI (port 8080)
- `airflow-init`: Database initialization

**Key DAGs**:
- `pm25_hourly_ingest`: Hourly data ingestion
- `pm25_24h_training`: Model training (manual trigger)
- `pm25_24h_pipeline`: Daily monitoring and auto-retrain

**Configuration**:
```yaml
services:
  airflow-scheduler:
    image: apache/airflow:2.10.3-python3.12
    environment:
      AIRFLOW__CORE__EXECUTOR: LocalExecutor
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql://...
      AIRFLOW__CORE__DAGS_FOLDER: /app/dags
    volumes:
      - ./dags:/app/dags
      - ./models:/app/models
      - ./triton_model_repo:/app/triton_model_repo
```

### 2. Triton Inference Server

**Purpose**: High-performance model serving

**Configuration**:
```yaml
services:
  triton:
    image: nvcr.io/nvidia/tritonserver:24.08-py3
    command: >
      tritonserver
      --model-repository=/models
      --model-control-mode=poll
      --repository-poll-secs=30
    ports:
      - "8010:8000"  # HTTP
      - "8011:8001"  # gRPC
      - "8012:8002"  # Metrics
```

**Features**:
- Dynamic model loading (30s poll interval)
- ONNX runtime backend
- Dynamic batching (max 32 samples)
- Zero-downtime updates

**Model Repository Structure**:
```
triton_model_repo/
├── pm25_56/
│   ├── 1/
│   │   └── model.onnx
│   └── config.pbtxt
├── pm25_57/
│   ├── 1/
│   │   └── model.onnx
│   └── config.pbtxt
...
```

### 3. PostgreSQL

**Purpose**: Time-series data storage

**Schema**:
```sql
-- Raw hourly PM2.5 data
CREATE TABLE pm25_raw_hourly (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    station_id INTEGER NOT NULL,
    pm25 FLOAT NOT NULL,
    UNIQUE(timestamp, station_id)
);

CREATE INDEX idx_timestamp_station ON pm25_raw_hourly(timestamp, station_id);

-- Predictions log
CREATE TABLE predictions_log (
    id SERIAL PRIMARY KEY,
    station_id INTEGER NOT NULL,
    prediction_date TIMESTAMP NOT NULL,
    predicted_pm25 FLOAT NOT NULL,
    model VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Actuals log (for monitoring)
CREATE TABLE actuals_log (
    id SERIAL PRIMARY KEY,
    station_id INTEGER NOT NULL,
    actual_date TIMESTAMP NOT NULL,
    actual_pm25 FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);
```

**Configuration**:
```yaml
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: pm25
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
```

### 4. MLflow

**Purpose**: Experiment tracking and model registry

**Storage**:
- **Backend**: PostgreSQL
- **Artifacts**: Local filesystem (`./mlruns`)

**Tracked Metrics**:
- MAE, RMSE, R², MAPE
- Training time
- Hyperparameters
- Model artifacts (ONNX files)

**Configuration**:
```yaml
services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.16.2
    command: >
      mlflow server
      --backend-store-uri postgresql://...
      --default-artifact-root /mlflow/artifacts
      --host 0.0.0.0
      --port 5000
    ports:
      - "5001:5000"
```

### 5. FastAPI

**Purpose**: REST API for predictions

**Endpoints**:
```python
GET  /health              # Health check
POST /predict             # Single prediction
POST /predict/batch       # Batch predictions
POST /predict/station     # Station-specific prediction
POST /actual              # Log ground truth
POST /retrain             # Trigger retraining
```

**Configuration**:
```yaml
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    environment:
      INFERENCE_BACKEND: triton
      TRITON_URL: triton:8000
      API_KEY: ${API_KEY}
    ports:
      - "8001:8000"
```

### 6. Streamlit Dashboard

**Purpose**: Web UI for monitoring and predictions

**Pages**:
1. **Predict**: Interactive prediction form
2. **Results**: Model comparison charts
3. **Monitoring**: MAE/PSI trends over time

**Configuration**:
```yaml
services:
  streamlit:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    environment:
      API_URL: http://api:8000
      API_KEY: ${API_KEY}
    ports:
      - "8501:8501"
```

---

## Data Architecture

### Data Flow

```
AirBKK API (External)
    ↓ [HTTP GET every hour]
Airflow DAG (pm25_hourly_ingest)
    ↓ [Parse JSON]
PostgreSQL (pm25_raw_hourly)
    ↓ [Query 3.5 years]
Feature Engineering (19 features)
    ↓ [Train/Val/Test split]
Model Training (5 algorithms)
    ↓ [Select best]
ONNX Export
    ↓ [Deploy if better]
Triton Model Repository
    ↓ [Load in 30s]
Triton Inference Server
    ↓ [Serve predictions]
FastAPI / Streamlit
```

### Data Retention

| Data Type | Retention | Storage | Size |
|-----------|-----------|---------|------|
| Raw hourly data | Indefinite | PostgreSQL | ~10KB per station per day |
| Predictions log | 1 year | PostgreSQL | ~1KB per station per day |
| Model artifacts | 6 months | Filesystem | ~200KB per model |
| MLflow experiments | 1 year | PostgreSQL + FS | ~1MB per experiment |
| Monitoring logs | 1 year | CSV | ~100KB per year |

### Data Quality

**Validation Rules**:
```python
# Range check
assert 0 <= pm25 <= 500, "PM2.5 out of valid range"

# Timestamp check
assert timestamp <= datetime.now(), "Future timestamp not allowed"

# Station check
assert station_id in [56, 57, 58, 59, 61], "Invalid station ID"

# Missing data threshold
assert missing_ratio < 0.2, "Too much missing data (>20%)"
```

---

## ML System Architecture

### Model Lifecycle

```
1. Training Phase (3-4 hours)
   ├── Data Loading (PostgreSQL query)
   ├── Preprocessing (ffill, clip, drop nulls)
   ├── Feature Engineering (19 features)
   ├── Train/Val/Test Split (3y/3m/3m)
   ├── Model Training
   │   ├── Linear Regression (GridSearchCV)
   │   ├── Ridge Regression (GridSearchCV)
   │   ├── Random Forest (GridSearchCV)
   │   ├── XGBoost (GridSearchCV)
   │   └── LSTM (PyTorch, early stopping)
   ├── Evaluation (MAE, RMSE, R², MAPE)
   └── Model Selection (lowest MAE)

2. Deployment Phase (< 1 minute)
   ├── ONNX Export
   ├── Compare with Production (if exists)
   ├── Deploy if Better
   ├── Update active_model.json
   ├── Copy to Triton Repository
   ├── Create config.pbtxt
   └── Log to CSV

3. Serving Phase (5-10ms latency)
   ├── Client Request → FastAPI
   ├── FastAPI → Triton HTTP
   ├── Triton → ONNX Runtime
   ├── ONNX Runtime → Inference
   └── Response → Client

4. Monitoring Phase (daily at 02:00 UTC)
   ├── Calculate Rolling RMSE (14 days, 2 weekly cycles)
   ├── Calculate PSI (feature drift)
   ├── Check Thresholds
   │   ├── RMSE > 13.0 → Trigger Retrain
   │   └── PSI > 0.2 → Trigger Retrain
   └── Log Results
```

### Model Registry Pattern

```python
# Each station has its own model directory
models/
├── station_56_24h/
│   ├── active_model.json          # Pointer to active model
│   ├── feature_columns.json        # Feature names
│   └── onnx/
│       ├── linear_regression_2024-01-01_2025-10-14.onnx
│       ├── ridge_regression_2024-01-01_2025-10-14.onnx
│       └── xgboost_2024-01-01_2025-10-14.onnx
├── station_57_24h/
│   └── ...
...

# active_model.json structure
{
  "onnx_file": "ridge_regression_2024-01-01_2025-10-14.onnx",
  "model_key": "ridge_regression",
  "station_id": 56,
  "train_start": "2024-01-01",
  "train_end": "2025-10-14",
  "is_lstm": false,
  "forecast_hour": 24,
  "n_features": 19
}
```

### Feature Store (Implicit)

Features computed on-demand during training, not pre-computed:
- **Pros**: Always fresh, no staleness
- **Cons**: Recomputed each training (acceptable for weekly/monthly retrains)

**Future Optimization**: Materialize features in PostgreSQL for faster training

---

## Deployment Architecture

### Docker Compose Services

```yaml
services:
  postgres:        # Data storage
  mlflow:          # Experiment tracking
  airflow-init:    # One-time DB setup
  airflow-scheduler: # DAG scheduling
  airflow-webserver: # Web UI
  triton:          # Model serving
  api:             # REST API
  streamlit:       # Dashboard
```

### Volume Mounts

```yaml
volumes:
  # Shared between Airflow and API
  - ./models:/app/models                      # Model artifacts
  - ./triton_model_repo:/app/triton_model_repo  # Triton models
  - ./results:/app/results                    # CSV logs
  
  # Airflow-specific
  - ./dags:/app/dags                          # DAG definitions
  - ./src:/app/src                            # Python modules
  
  # Data persistence
  - postgres_data:/var/lib/postgresql/data    # Database
  - mlflow_data:/mlflow/artifacts             # MLflow artifacts
```

### Network Architecture

```
Docker Network: pm25-prediction-ml-system_default

Services communicate via service names:
- airflow-scheduler → postgres:5432
- airflow-scheduler → mlflow:5000
- api → triton:8000
- streamlit → api:8000

External access (host machine):
- Airflow UI:    localhost:8080
- MLflow UI:     localhost:5001
- FastAPI:       localhost:8001
- Streamlit:     localhost:8501
- Triton HTTP:   localhost:8010
- Triton gRPC:   localhost:8011
- PostgreSQL:    localhost:5432
```

---

## API Architecture

### FastAPI Structure

```python
# src/api.py
from fastapi import FastAPI, HTTPException, Header
import tritonclient.http as httpclient

app = FastAPI(title="PM2.5 Prediction API")

# Dependency: API Key validation
def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=401, detail="Invalid API Key")

# Inference client (initialized on startup)
triton_client = httpclient.InferenceServerClient(
    url=os.getenv("TRITON_URL", "localhost:8000")
)

@app.post("/predict/station")
def predict_station(
    station_id: int,
    features: List[float],
    api_key: str = Depends(verify_api_key)
):
    # Build input tensor
    input_data = np.array([features], dtype=np.float32)
    input_tensor = httpclient.InferInput("float_input", input_data.shape, "FP32")
    input_tensor.set_data_from_numpy(input_data)
    
    # Inference
    result = triton_client.infer(
        model_name=f"pm25_{station_id}",
        inputs=[input_tensor],
        outputs=[httpclient.InferRequestedOutput("variable")]
    )
    
    prediction = result.as_numpy("variable")[0][0]
    return {"station_id": station_id, "predicted_pm25": prediction}
```

### Authentication

**Method**: API Key (Header-based)

```bash
# Environment variable
API_KEY=foonalert-secret-key

# Request header
X-API-Key: foonalert-secret-key
```

**Future**: JWT tokens, OAuth2

### Rate Limiting

**Current**: None (internal system)  
**Future**: Redis-backed rate limiter (e.g., 100 req/min per key)

---

## Infrastructure

### Resource Requirements

| Service | CPU | Memory | Disk |
|---------|-----|--------|------|
| PostgreSQL | 1 core | 512MB | 10GB |
| Airflow Scheduler | 2 cores | 2GB | 5GB |
| Airflow Webserver | 1 core | 1GB | 1GB |
| Triton Server | 2 cores | 2GB | 5GB |
| MLflow | 1 core | 512MB | 20GB |
| FastAPI | 1 core | 512MB | 1GB |
| Streamlit | 1 core | 512MB | 1GB |
| **Total** | **9 cores** | **7GB** | **43GB** |

**Recommended Hardware**:
- **Development**: 4-core CPU, 8GB RAM, 50GB SSD
- **Production**: 8-core CPU, 16GB RAM, 100GB SSD, GPU optional

### Scaling Strategy

#### Horizontal Scaling

**Current**: Single-node Docker Compose  
**Future**: Kubernetes with:
- Multiple Airflow workers (CeleryExecutor)
- Triton replicas (load balanced)
- PostgreSQL read replicas
- Redis for Airflow queue

#### Vertical Scaling

**Training**: Increase `GRID_N_JOBS` for more parallel CV folds  
**Inference**: Increase Triton `max_batch_size` and instance count

### High Availability

**Current**: Single point of failure (all services on one host)

**Future HA Setup**:
```
Load Balancer (NGINX)
    ↓
┌───────────────┬───────────────┬───────────────┐
│   Triton-1    │   Triton-2    │   Triton-3    │
└───────────────┴───────────────┴───────────────┘
                      ↓
┌───────────────┬───────────────┬───────────────┐
│  PostgreSQL   │  PostgreSQL   │  PostgreSQL   │
│  Primary      │  Standby-1    │  Standby-2    │
└───────────────┴───────────────┴───────────────┘
```

---

## Security

### Current Security Measures

1. **API Key**: Required for all non-health endpoints
2. **Network Isolation**: Services communicate via internal Docker network
3. **No Public Exposure**: Postgres not exposed to internet (tunnel required)
4. **Environment Variables**: Secrets in `.env` file (gitignored)

### Security Gaps (Production Improvements Needed)

1. **No HTTPS**: All traffic is HTTP (need SSL/TLS)
2. **No Secrets Management**: Plain text in `.env` (use Vault/AWS Secrets)
3. **No RBAC**: Single API key for all operations
4. **No Audit Logging**: Who accessed what, when
5. **No Input Validation**: Malformed requests could crash services

### Production Security Checklist

- [ ] Enable HTTPS (Let's Encrypt)
- [ ] Implement JWT-based auth with expiry
- [ ] Use secrets manager (HashiCorp Vault, AWS Secrets Manager)
- [ ] Add input validation (Pydantic models)
- [ ] Enable Airflow RBAC (users, roles, permissions)
- [ ] Add audit logging to PostgreSQL
- [ ] Implement rate limiting (Redis + FastAPI middleware)
- [ ] Regular security scans (Trivy, Snyk)
- [ ] Encrypt data at rest (PostgreSQL encryption)
- [ ] Network policies (restrict inter-service communication)

---

## Scalability

### Current Bottlenecks

1. **Training**: Single-threaded GridSearchCV (can parallelize)
2. **Database**: Single Postgres instance (need read replicas)
3. **Inference**: Single Triton instance (can replicate)
4. **Storage**: Local filesystem (need S3/GCS for distributed setup)

### Scaling Scenarios

#### 1. More Stations (5 → 50 stations)

**Impact**:
- Training time: 5 stations × 3-4 hours = 15-20 hours total
- Disk: 50 stations × 200KB = 10MB models
- Inference: 50 models in Triton (still fast)

**Solution**:
- Parallel training (5 stations at a time)
- Shared feature engineering (cache preprocessed data)

#### 2. Higher Frequency (hourly → every 5 minutes)

**Impact**:
- Data volume: 12× more rows in PostgreSQL
- Training: More data → longer training
- Inference: Same (still 5-10ms per request)

**Solution**:
- Downsample training data (e.g., hourly aggregates)
- Incremental training (sliding window)
- Time-series compression (e.g., TimescaleDB)

#### 3. More Models (5 → 20 algorithms)

**Impact**:
- Training time: 4× longer
- Storage: 4× more ONNX files
- Triton: Still handles it (models loaded on-demand)

**Solution**:
- Parallel model training (separate Airflow workers)
- Early stopping (skip bad models after initial eval)

---

## Monitoring & Observability

### Current Monitoring

1. **Airflow UI**: DAG run status, task logs
2. **MLflow UI**: Experiment metrics, model registry
3. **Triton Metrics**: Prometheus endpoint (port 8012)
4. **CSV Logs**: Training results, monitoring results

### Production Monitoring Stack

```
Prometheus (Metrics Collection)
    ├── Triton Server metrics
    ├── FastAPI metrics (via prometheus_client)
    ├── PostgreSQL metrics (via postgres_exporter)
    └── Airflow metrics (via statsd)
        ↓
Grafana (Visualization)
    ├── Model Performance Dashboard
    ├── System Health Dashboard
    └── Alerting Rules
        ↓
AlertManager (Notifications)
    ├── Email
    ├── Slack
    └── PagerDuty
```

### Key Metrics to Monitor

| Metric | Type | Threshold |
|--------|------|-----------|
| Model RMSE | Business | Alert if > 13.0 µg/m³ |
| Feature PSI | Business | Alert if > 0.2 |
| Inference latency | System | Alert if > 100ms (p99) |
| Triton error rate | System | Alert if > 1% |
| PostgreSQL connections | System | Alert if > 80% |
| Disk usage | System | Alert if > 85% |
| Training success rate | Business | Alert if < 95% |

---

## Disaster Recovery

### Backup Strategy

**PostgreSQL**:
```bash
# Daily backup (cron job)
0 3 * * * pg_dump -U postgres pm25 | gzip > /backups/pm25_$(date +%Y%m%d).sql.gz

# Retention: 30 days
find /backups -name "pm25_*.sql.gz" -mtime +30 -delete
```

**Model Artifacts**:
```bash
# Weekly backup to S3
0 2 * * 0 aws s3 sync /app/models s3://pm25-models/backups/$(date +%Y%m%d)/
```

**Recovery Time Objective (RTO)**: 2 hours  
**Recovery Point Objective (RPO)**: 24 hours

### Failover Plan

1. **Primary site down**: Switch to backup EC2 instance
2. **Database failure**: Restore from latest pg_dump
3. **Triton failure**: API falls back to onnxruntime (slower but works)
4. **Complete data loss**: Retrain all models from scratch (72 hours)

---

## Cost Optimization

### Current Costs (AWS EC2)

| Component | Instance Type | Monthly Cost |
|-----------|---------------|--------------|
| EC2 (t3.xlarge) | 4 vCPU, 16GB | $120 |
| EBS Storage (100GB) | gp3 | $10 |
| Data Transfer | Minimal | $5 |
| **Total** | | **$135/month** |

### Optimization Strategies

1. **Spot Instances**: Training on spot instances (70% savings)
2. **Auto-scaling**: Scale down Triton during off-peak hours
3. **S3 Intelligent-Tiering**: Archive old model artifacts
4. **Reserved Instances**: 1-year commitment for 40% discount
5. **Model Compression**: Quantization (INT8) for smaller models

---

## Technology Decisions

### Why ONNX?

✅ **Pros**:
- Cross-platform (CPU, GPU, mobile)
- Faster inference (5-10× vs Python)
- Framework agnostic (scikit-learn, PyTorch, TensorFlow)
- Smaller file size vs joblib

❌ **Cons**:
- Extra export step
- Not all scikit-learn features supported

### Why Triton over TensorFlow Serving?

✅ **Pros**:
- Multi-framework (ONNX, TensorFlow, PyTorch)
- Better batching and queuing
- More flexible configuration
- Better documentation

❌ **Cons**:
- NVIDIA-specific (though runs on CPU)
- Larger docker image

### Why Airflow over Prefect/Dagster?

✅ **Pros**:
- Industry standard (large community)
- Mature ecosystem (integrations)
- Proven at scale (Airbnb, Netflix)
- Self-hosted (no vendor lock-in)

❌ **Cons**:
- Heavier infrastructure
- Steeper learning curve

---

## Future Roadmap

### Short-term (3 months)

- [ ] Add ensemble models (weighted average)
- [ ] Implement A/B testing framework
- [ ] Build Grafana monitoring dashboard
- [ ] Add Slack/Email alerting
- [ ] Feature importance tracking

### Mid-term (6 months)

- [ ] Multi-region deployment (HA)
- [ ] Kubernetes migration
- [ ] Online learning (incremental updates)
- [ ] Feature store (materialized features)
- [ ] Model explainability (SHAP values)

### Long-term (12 months)

- [ ] Real-time predictions (streaming)
- [ ] Multi-variate forecasting (weather, traffic)
- [ ] AutoML for hyperparameter optimization
- [ ] Edge deployment (mobile app)
- [ ] Federated learning (privacy-preserving)

---

## References

### Documentation Links

- [Apache Airflow Docs](https://airflow.apache.org/docs/)
- [Triton Inference Server](https://github.com/triton-inference-server/server)
- [ONNX Runtime](https://onnxruntime.ai/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [MLflow](https://mlflow.org/docs/latest/index.html)

### Internal Docs

- [ML Pipeline Guide](./ML_PIPELINE.md)
- [Quick Start Guide](./QUICK_START_5_STATIONS.md)
- [API Reference](./STATION_PREDICTION_API.md)
- [Triton API Guide](./TRITON_API_GUIDE.md)

---

**Last Updated**: 2026-04-16  
**Version**: 1.0  
**Authors**: ML Platform Team

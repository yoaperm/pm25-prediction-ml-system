# ML Pipeline Documentation

Complete technical documentation for the PM2.5 prediction ML pipeline.

---

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Data Flow](#data-flow)
3. [Feature Engineering](#feature-engineering)
4. [Model Training](#model-training)
5. [Model Evaluation](#model-evaluation)
6. [Model Deployment](#model-deployment)
7. [Monitoring & Drift Detection](#monitoring--drift-detection)
8. [Auto-Retraining Logic](#auto-retraining-logic)
9. [Airflow DAGs](#airflow-dags)
10. [Error Handling](#error-handling)

---

## Pipeline Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Data Ingestion                               │
│  AirBKK API → PostgreSQL (pm25_raw_hourly table)                    │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      Feature Engineering                             │
│  • Time-series features (lags, rolling stats, diffs)                │
│  • Temporal features (hour, day, month, is_weekend)                 │
│  • Feature count: 19 features per station                           │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        Model Training                                │
│  5 Algorithms Compete:                                              │
│  • Linear Regression    • Ridge Regression                          │
│  • Random Forest        • XGBoost                                   │
│  • LSTM (PyTorch)                                                   │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         Evaluation                                   │
│  Metrics: MAE, RMSE, R², MAPE                                       │
│  Test set: Last 3 months of data                                    │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    Model Selection & Deployment                      │
│  • Select best model (lowest MAE)                                   │
│  • Compare with production (if exists)                              │
│  • Deploy if MAE improved                                           │
│  • Export to ONNX                                                   │
│  • Publish to Triton (automatic)                                    │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    Inference (Triton Server)                         │
│  • REST API: http://localhost:8010                                  │
│  • gRPC API: http://localhost:8011                                  │
│  • Response time: ~5-10ms per prediction                            │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                 Monitoring & Auto-Retraining                         │
│  Daily checks (02:00 UTC):                                          │
│  • Rolling 30-day MAE                                               │
│  • PSI (Population Stability Index) for drift                       │
│  • Triggers: MAE > 9.0 OR PSI > 0.2                                 │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Time-series integrity**: All features use `shift(1)` to prevent data leakage
2. **ONNX-first deployment**: No joblib/pickle at inference time
3. **Model versioning**: ONNX files named `{model}_{start_date}_{end_date}.onnx`
4. **Automated deployment**: Best model automatically deployed when MAE improves
5. **Zero-downtime updates**: Triton polls every 30s and hot-swaps models

---

## Data Flow

### 1. Data Ingestion

**Source**: Thailand AirBKK API  
**Schedule**: Hourly (via `pm25_hourly_ingest` DAG)  
**Destination**: PostgreSQL table `pm25_raw_hourly`

#### Table Schema

```sql
CREATE TABLE pm25_raw_hourly (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    station_id INTEGER NOT NULL,
    pm25 FLOAT NOT NULL,
    UNIQUE(timestamp, station_id)
);

CREATE INDEX idx_timestamp_station ON pm25_raw_hourly(timestamp, station_id);
```

#### Stations Tracked

| Station ID | Location | Data Availability |
|------------|----------|-------------------|
| 56 | Bangkok | 2024-01-01 onwards |
| 57 | Bangkok | 2024-01-01 onwards |
| 58 | Bangkok | 2024-01-01 onwards |
| 59 | Bangkok | 2024-01-01 onwards |
| 61 | Bangkok | 2024-01-01 onwards |

### 2. Data Preprocessing

**Location**: `src/preprocessing.py`

```python
def preprocess_data(df):
    """
    1. Sort by timestamp
    2. Forward fill missing values (max 3 hours)
    3. Backward fill remaining gaps
    4. Clip outliers to [0, 500] µg/m³
    5. Drop any remaining nulls
    """
    df = df.sort_values("timestamp")
    df["pm25"] = df["pm25"].ffill(limit=3).bfill(limit=3)
    df["pm25"] = df["pm25"].clip(0, 500)
    df = df.dropna(subset=["pm25"])
    return df
```

**Quality Checks**:
- Outlier detection: Values > 500 µg/m³ clipped
- Gap handling: Max 3-hour forward fill
- Missing data threshold: Training fails if < 80% data available

### 3. Train/Val/Test Split

**Dynamic splits relative to today**:

```python
today = datetime.utcnow().date()

# Train: 3 years (today - 3.5y → today - 6m)
train_start = today - relativedelta(years=3, months=6)
train_end   = today - relativedelta(months=6)

# Validation: 3 months (today - 6m → today - 3m)
val_start   = train_end
val_end     = today - relativedelta(months=3)

# Test: 3 months (today - 3m → today)
test_start  = val_end
test_end    = today
```

**Rationale**:
- Train: 3 years for seasonal patterns
- Val: 3 months for early stopping (LSTM)
- Test: 3 months for recent performance
- No random shuffle (time-series order preserved)

---

## Feature Engineering

**Location**: `src/feature_engineering.py`

### Feature Set (19 features)

#### 1. Lag Features (6)
Historical PM2.5 values:
```python
df['pm25_lag_1h']  = df['pm25'].shift(1)   # 1 hour ago
df['pm25_lag_2h']  = df['pm25'].shift(2)   # 2 hours ago
df['pm25_lag_3h']  = df['pm25'].shift(3)   # 3 hours ago
df['pm25_lag_6h']  = df['pm25'].shift(6)   # 6 hours ago
df['pm25_lag_12h'] = df['pm25'].shift(12)  # 12 hours ago
df['pm25_lag_24h'] = df['pm25'].shift(24)  # 24 hours ago (yesterday)
```

#### 2. Rolling Statistics (6)
Moving averages and standard deviations:
```python
# 6-hour window
df['pm25_rolling_mean_6h'] = df['pm25'].shift(1).rolling(6).mean()
df['pm25_rolling_std_6h']  = df['pm25'].shift(1).rolling(6).std()

# 12-hour window
df['pm25_rolling_mean_12h'] = df['pm25'].shift(1).rolling(12).mean()
df['pm25_rolling_std_12h']  = df['pm25'].shift(1).rolling(12).std()

# 24-hour window (daily pattern)
df['pm25_rolling_mean_24h'] = df['pm25'].shift(1).rolling(24).mean()
df['pm25_rolling_std_24h']  = df['pm25'].shift(1).rolling(24).std()
```

#### 3. Difference Features (2)
Rate of change:
```python
df['pm25_diff_1h']  = df['pm25'].shift(1) - df['pm25'].shift(2)   # hourly change
df['pm25_diff_24h'] = df['pm25'].shift(1) - df['pm25'].shift(25)  # daily change
```

#### 4. Temporal Features (5)
Time-based patterns:
```python
df['hour']         = df['timestamp'].dt.hour           # 0-23
df['day_of_week']  = df['timestamp'].dt.dayofweek      # 0-6 (Mon=0)
df['month']        = df['timestamp'].dt.month          # 1-12
df['day_of_year']  = df['timestamp'].dt.dayofyear      # 1-366
df['is_weekend']   = (df['day_of_week'] >= 5).astype(int)  # 0 or 1
```

### Target Variable

```python
# Predict PM2.5 exactly 24 hours ahead
df['target'] = df['pm25'].shift(-24)
```

### Leakage Prevention

**Critical**: All lag and rolling features use `.shift(1)` to ensure only past data is used:

```python
# ❌ WRONG - uses current hour data
df['lag_1h'] = df['pm25']

# ✅ CORRECT - uses previous hour data
df['lag_1h'] = df['pm25'].shift(1)
```

---

## Model Training

### Training Configuration

**Location**: `dags/pm25_24h_training_dag.py`

```python
FORECAST_HOUR = 24
RANDOM_STATE = 42
GRID_N_JOBS = int(os.environ.get("GRID_N_JOBS", "-1"))  # GridSearchCV parallelism
```

### 1. Linear Regression (Baseline)

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

param_grid = {
    'fit_intercept': [True, False],
}

model = GridSearchCV(
    LinearRegression(),
    param_grid,
    cv=TimeSeriesSplit(n_splits=3),
    scoring='neg_mean_absolute_error',
    n_jobs=GRID_N_JOBS
)
model.fit(X_train, y_train)
```

**Strengths**: Fast, interpretable  
**Weaknesses**: Linear relationships only

### 2. Ridge Regression (Regularized)

```python
from sklearn.linear_model import Ridge

param_grid = {
    'alpha': [0.1, 1.0, 10.0, 100.0],
    'fit_intercept': [True, False],
}

model = GridSearchCV(
    Ridge(),
    param_grid,
    cv=TimeSeriesSplit(n_splits=3),
    scoring='neg_mean_absolute_error',
    n_jobs=GRID_N_JOBS
)
model.fit(X_train, y_train)
```

**Strengths**: Handles multicollinearity, prevents overfitting  
**Weaknesses**: Still linear

### 3. Random Forest (Ensemble)

```python
from sklearn.ensemble import RandomForestRegressor

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
}

model = GridSearchCV(
    RandomForestRegressor(random_state=RANDOM_STATE),
    param_grid,
    cv=TimeSeriesSplit(n_splits=3),
    scoring='neg_mean_absolute_error',
    n_jobs=GRID_N_JOBS
)
model.fit(X_train, y_train)
```

**Strengths**: Non-linear, feature importance  
**Weaknesses**: Large model size (196KB ONNX)

### 4. XGBoost (Gradient Boosting)

```python
from xgboost import XGBRegressor

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
}

model = GridSearchCV(
    XGBRegressor(random_state=RANDOM_STATE, tree_method='hist'),
    param_grid,
    cv=TimeSeriesSplit(n_splits=3),
    scoring='neg_mean_absolute_error',
    n_jobs=GRID_N_JOBS
)
model.fit(X_train, y_train)
```

**Strengths**: Best performance on many stations  
**Weaknesses**: Slower training

### 5. LSTM (Deep Learning)

```python
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden_size, num_layers,
            batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# Training loop
model = LSTMModel(n_features=19)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(50):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    # Validation for early stopping
    val_loss = validate(model, val_loader)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')
```

**Strengths**: Captures temporal dependencies  
**Weaknesses**: Longer training time, requires GPU for production

### Environment Variables for Training

```bash
# Prevent OpenMP/MKL conflicts with PyTorch
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1

# Force CPU for PyTorch (Apple Silicon MPS crashes)
PYTORCH_DEVICE=cpu

# GridSearchCV parallelism (default: all cores)
GRID_N_JOBS=-1
```

---

## Model Evaluation

**Location**: `src/evaluate.py`

### Metrics Computed

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (avoid division by zero)
    mask = y_true > 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape
    }
```

### Benchmark Performance (Typical)

| Model | MAE (µg/m³) | RMSE (µg/m³) | R² | Training Time |
|-------|-------------|--------------|-----|---------------|
| Linear Regression | 7.5 | 10.2 | 0.82 | 1s |
| Ridge Regression | 7.3 | 10.1 | 0.83 | 2s |
| Random Forest | 6.8 | 9.5 | 0.85 | 45s |
| XGBoost | 6.5 | 9.2 | 0.86 | 90s |
| LSTM | 7.0 | 9.8 | 0.84 | 300s |

**Typical winner**: Ridge Regression or XGBoost (varies by station)

---

## Model Deployment

### 1. ONNX Export

**Location**: `src/export_onnx.py`

All models exported to ONNX for unified inference:

```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def export_sklearn(model, name, onnx_dir, n_features=19):
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=17
    )
    
    output_path = f"{onnx_dir}/{name}.onnx"
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    
    return output_path
```

**LSTM Export** (PyTorch):
```python
import torch.onnx

def export_lstm(model, n_features, output_path):
    dummy_input = torch.randn(1, 1, n_features)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["float_input"],
        output_names=["variable"],
        dynamic_axes={"float_input": {0: "batch_size"}}
    )
```

### 2. Model Registry

**Location**: `models/station_{id}_24h/active_model.json`

```json
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

### 3. Deployment Decision Logic

```python
def compare_and_deploy(new_model, prod_model, test_X, test_y):
    # Evaluate new model
    new_mae = evaluate_model(test_y, new_model.predict(test_X))["MAE"]
    
    # Load production model
    if prod_model exists:
        prod_mae = evaluate_model(test_y, prod_model.predict(test_X))["MAE"]
    else:
        prod_mae = None  # First deployment
    
    # Deploy if better OR first time
    if prod_mae is None or new_mae < prod_mae:
        deploy(new_model)
        publish_to_triton(new_model)
        return "DEPLOYED"
    else:
        return "NOT_DEPLOYED"
```

### 4. Triton Publishing (Automatic)

**Location**: `dags/pm25_24h_training_dag.py:_publish_to_triton()`

```python
def _publish_to_triton(onnx_path, station_id, n_features, triton_repo="/app/triton_model_repo"):
    model_name = f"pm25_{station_id}"
    
    # Create directory structure
    version_dir = f"{triton_repo}/{model_name}/1"
    os.makedirs(version_dir, exist_ok=True)
    
    # Copy ONNX file
    shutil.copy2(onnx_path, f"{version_dir}/model.onnx")
    
    # Create config.pbtxt
    config = f'''name: "{model_name}"
backend: "onnxruntime"
max_batch_size: 32

input [
  {{
    name: "float_input"
    data_type: TYPE_FP32
    dims: [ {n_features} ]
  }}
]

output [
  {{
    name: "variable"
    data_type: TYPE_FP32
    dims: [ 1 ]
  }}
]

dynamic_batching {{ }}
'''
    
    with open(f"{triton_repo}/{model_name}/config.pbtxt", "w") as f:
        f.write(config)
    
    print(f"✓ Model '{model_name}' published to Triton (loads in ~30s)")
```

**Triton Auto-Loading**:
- Poll interval: 30 seconds
- Zero-downtime: Old model serves requests until new model ready
- Health check: `GET /v2/health/ready`

---

## Monitoring & Drift Detection

### Daily Monitoring Pipeline

**DAG**: `pm25_24h_pipeline` (runs at 02:00 UTC daily)

### 1. Performance Monitoring

```python
def calculate_rolling_mae(station_id, window_days=30):
    """
    Calculate MAE over last 30 days of predictions vs actuals.
    """
    query = """
    SELECT 
        p.predicted_pm25,
        a.actual_pm25
    FROM predictions_log p
    JOIN actuals_log a 
        ON p.station_id = a.station_id 
        AND p.prediction_date = a.actual_date
    WHERE p.station_id = %s
        AND p.created_at >= NOW() - INTERVAL '30 days'
    """
    
    df = pd.read_sql(query, conn, params=[station_id])
    mae = mean_absolute_error(df['actual_pm25'], df['predicted_pm25'])
    
    return mae
```

### 2. Feature Drift Detection (PSI)

**Population Stability Index (PSI)**:

```python
def calculate_psi(expected, actual, bins=10):
    """
    Calculate PSI between training and recent feature distributions.
    
    PSI < 0.1  : No significant change
    PSI < 0.2  : Small change
    PSI >= 0.2 : Large change (trigger retrain)
    """
    expected_pct = np.histogram(expected, bins=bins)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=bins)[0] / len(actual)
    
    # Avoid division by zero
    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
    actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)
    
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    
    return psi
```

### 3. Retrain Decision Logic

```python
def should_retrain(station_id):
    mae = calculate_rolling_mae(station_id)
    psi = calculate_feature_drift_psi(station_id)
    
    # Thresholds
    MAE_THRESHOLD = 9.0  # µg/m³
    PSI_THRESHOLD = 0.2
    
    if mae > MAE_THRESHOLD:
        return True, f"MAE degraded: {mae:.2f} > {MAE_THRESHOLD}"
    
    if psi > PSI_THRESHOLD:
        return True, f"Feature drift detected: PSI={psi:.3f} > {PSI_THRESHOLD}"
    
    return False, "Model performing well"
```

---

## Auto-Retraining Logic

### Trigger Conditions

1. **Performance Degradation**: Rolling 30-day MAE > 9.0 µg/m³
2. **Feature Drift**: PSI > 0.2 on any feature
3. **Manual Trigger**: Via Airflow UI or API

### Retraining Flow

```
Daily Monitor (02:00 UTC)
    ↓
Check MAE & PSI
    ↓
[If trigger conditions met]
    ↓
Trigger pm25_24h_training DAG
    ↓
Train 5 models (3-4 hours)
    ↓
Evaluate on test set
    ↓
[If new MAE < prod MAE]
    ↓
Deploy to models/ directory
    ↓
Publish to Triton
    ↓
Log to monitoring_24h_results.csv
```

### Monitoring Logs

**Location**: `results/monitoring_24h_results.csv`

```csv
station_id,check_date,rolling_mae,psi,status,action
56,2026-04-16,7.2,0.08,OK,none
57,2026-04-16,8.9,0.15,OK,none
58,2026-04-16,9.5,0.12,DEGRADED,retrain_triggered
59,2026-04-16,6.8,0.25,DRIFT,retrain_triggered
61,2026-04-16,7.5,0.09,OK,none
```

---

## Airflow DAGs

### 1. `pm25_hourly_ingest`

**Purpose**: Ingest hourly data from AirBKK API  
**Schedule**: Every hour (`0 * * * *`)  
**Tasks**:
1. Fetch data from API
2. Parse JSON
3. Insert into PostgreSQL (upsert on conflict)

### 2. `pm25_24h_training`

**Purpose**: Train models for one station  
**Schedule**: Manual trigger (or weekly: `0 2 * * 0`)  
**Parameters**: `station_id` (56, 57, 58, 59, 61)  

**Task Graph**:
```
feature_engineering
    ├── train_linear
    ├── train_ridge
    ├── train_random_forest
    ├── train_xgboost
    └── train_lstm
            └── evaluate
                └── compare_and_deploy
```

### 3. `pm25_24h_pipeline`

**Purpose**: Daily monitoring and auto-retraining  
**Schedule**: Daily at 02:00 UTC (`0 2 * * *`)  
**Tasks**:
1. `monitor_station_56` - Check MAE & PSI
2. `monitor_station_57` - Check MAE & PSI
3. `monitor_station_58` - Check MAE & PSI
4. `monitor_station_59` - Check MAE & PSI
5. `monitor_station_61` - Check MAE & PSI

Each monitoring task:
- Calculates rolling MAE
- Calculates PSI for feature drift
- Triggers training DAG if thresholds exceeded

---

## Error Handling

### Training Failures

```python
try:
    model.fit(X_train, y_train)
except Exception as e:
    logging.error(f"Training failed for {model_name}: {e}")
    # Save temp marker file to skip in evaluate task
    with open(f"/tmp/failed_{model_key}.txt", "w") as f:
        f.write(str(e))
    raise
```

### ONNX Export Failures

```python
try:
    export_onnx(model, output_path, n_features=19)
except Exception as e:
    logging.error(f"ONNX export failed: {e}")
    # Continue with other models (don't block entire pipeline)
    return None
```

### Triton Publishing Failures

```python
try:
    _publish_to_triton(onnx_path, station_id, n_features)
except Exception as e:
    # Model still deployed to models/ directory
    # Can manually publish later
    logging.warning(f"Triton publish failed (model still deployed): {e}")
```

### Insufficient Data

```python
if len(df) < MIN_REQUIRED_HOURS:
    raise ValueError(
        f"Insufficient data: {len(df)} hours < {MIN_REQUIRED_HOURS} required"
    )
```

**MIN_REQUIRED_HOURS**: 3.5 years × 365 days × 24 hours × 0.8 = ~24,528 hours

---

## Performance Optimization

### Training Speed

1. **GridSearchCV parallelism**: `n_jobs=-1` (all cores)
2. **XGBoost tree method**: `tree_method='hist'` (faster than exact)
3. **Early stopping**: LSTM uses validation set
4. **Incremental training**: Only retrain when needed (not daily)

### Inference Speed

1. **ONNX optimization**: 5-10x faster than native Python
2. **Triton batching**: `max_batch_size=32` (dynamic batching enabled)
3. **Model size**: Linear/Ridge = 361B, Random Forest = 196KB

### Storage Optimization

1. **ONNX versioning**: Old models kept, can be pruned
2. **PostgreSQL indexing**: `(timestamp, station_id)` index
3. **Result logs**: CSV (append-only, can archive monthly)

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `dags/pm25_24h_training_dag.py` | Main training pipeline |
| `dags/pm25_24h_pipeline_dag.py` | Monitoring & auto-retrain |
| `dags/pm25_hourly_ingest_dag.py` | Data ingestion |
| `src/feature_engineering.py` | Feature creation |
| `src/preprocessing.py` | Data cleaning |
| `src/train.py` | Model training functions |
| `src/evaluate.py` | Evaluation metrics |
| `src/export_onnx.py` | ONNX export |
| `models/station_{id}_24h/active_model.json` | Model registry |
| `results/forecast_24h_results.csv` | Training logs |
| `results/monitoring_24h_results.csv` | Monitoring logs |

---

## Next Steps

- [ ] Add ensemble models (weighted average of top 3)
- [ ] Implement A/B testing for gradual rollout
- [ ] Add feature importance tracking
- [ ] Integrate with alerting system (Slack/Email)
- [ ] Build Grafana dashboard for real-time monitoring

---

**Last Updated**: 2026-04-16

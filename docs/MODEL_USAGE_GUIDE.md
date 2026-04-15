# How to Use Your 5 Models

## Understanding the "5 Models"

You have **TWO concepts of "5 models"**:

### 1️⃣ **5 Model TYPES** (Training Algorithms)

During each training run, the pipeline trains **5 different algorithms**:

```
Training Pipeline
─────────────────────────────────────────────────
Data (3.5 years) → Feature Engineering (19 features)
                            ↓
        ┌───────────┬─────────┬─────────────┬──────────┬────────┐
        │           │         │             │          │        │
   Linear      Ridge    Random Forest   XGBoost     LSTM
   Baseline    L2 Reg   Tree Ensemble   Boosting    Neural Net
        │           │         │             │          │        │
   GridSearch  GridSearch  GridSearch  GridSearch  Early Stop
        │           │         │             │          │        │
        └───────────┴─────────┴─────────────┴──────────┴────────┘
                            ↓
                    Evaluate on Test Set
                            ↓
                  Select BEST (lowest MAE)
                            ↓
                Deploy to Triton & FastAPI
```

**Key Point**: Only the BEST model gets deployed per station!

---

### 2️⃣ **6 Station MODELS** (Deployed in Production)

Each station has ONE deployed model (the best of the 5 types):

| Station | Model Name | Deployed Algorithm | MAE | Status |
|---------|------------|-------------------|-----|--------|
| 10T | `pm25` | (varies) | - | ✓ Active |
| 63 | `pm25_63` | (varies) | - | ✓ Active |
| 64 | `pm25_64` | (varies) | - | ✓ Active |
| 65 | `pm25_65` | (varies) | - | ✓ Active |
| 66 | `pm25_66` | (varies) | - | ✓ Active |
| 67 | `pm25_67` | (varies) | - | ✓ Active |

**Example**: Station 56 might use Ridge, Station 57 might use XGBoost, etc.

---

## How to Use Your Models

### ✅ **METHOD 1: Triton API** (Production - FASTEST)

```python
import numpy as np
import tritonclient.http as httpclient

# Connect
client = httpclient.InferenceServerClient(url="localhost:8010")

# Prepare features (17 for daily, 19 for hourly)
features = np.array([[
    28.5, 29.1, 30.2, 32.5, 35.0,  # lags
    28.5, 3.2, 29.1, 4.1,           # rolling stats  
    4, 107, 2, 2026, 2, 15, 16, 0   # time features
]], dtype=np.float32)

# Predict
input_tensor = httpclient.InferInput("float_input", features.shape, "FP32")
input_tensor.set_data_from_numpy(features)
output_tensor = httpclient.InferRequestedOutput("variable")

# Station 10T
result = client.infer(
    model_name="pm25",
    inputs=[input_tensor],
    outputs=[output_tensor]
)
prediction = result.as_numpy("variable")[0][0]
print(f"Station 10T: {prediction:.2f} µg/m³")

# Station 63
result_63 = client.infer(
    model_name="pm25_63",
    inputs=[input_tensor],
    outputs=[output_tensor]
)
prediction_63 = result_63.as_numpy("variable")[0][0]
print(f"Station 63: {prediction_63:.2f} µg/m³")
```

**Performance**: ~2ms latency, 500+ req/s

---

### ✅ **METHOD 2: FastAPI** (Development - EASIER)

```python
import requests

API_URL = "http://localhost:8001"
headers = {"X-API-Key": "foonalert-secret-key"}

# Prepare history (last 15 days)
data = {
    "history": [
        {"date": "2026-04-01", "pm25": 28.5},
        {"date": "2026-04-02", "pm25": 29.1},
        # ... 13 more days
        {"date": "2026-04-15", "pm25": 30.7},
    ]
}

# Predict
response = requests.post(f"{API_URL}/predict", json=data, headers=headers)
result = response.json()

print(f"Prediction: {result['prediction']} µg/m³")
print(f"Next date: {result['next_date']}")
print(f"Model: {result['model_name']}")
```

**When to use**: Easier integration, handles feature engineering for you

---

### ✅ **METHOD 3: View Training Results** (Research)

#### A. Check CSV Results

```bash
cat results/forecast_24h_results.csv
```

**Output**:
```csv
station_id,train_start,train_end,best_model,new_mae,prod_mae,mae_delta,status,run_date
56,2024-01-01,2025-10-14,Linear Regression,7.3287,7.3287,0.0,NOT_DEPLOYED,2026-04-15 16:58
57,2024-01-01,2025-10-14,Ridge Regression,6.8022,6.8022,0.0,NOT_DEPLOYED,2026-04-15 16:59
58,2024-01-01,2025-10-14,Ridge Regression,7.1586,7.1587,0.0001,DEPLOYED,2026-04-15 16:59
```

Shows which of the 5 models won for each station!

#### B. MLflow UI

```bash
# Open in browser
http://localhost:5001
```

Navigate to:
- Experiment: `pm25_24h_station_56`
- See all 5 runs with MAE, RMSE, R² comparisons
- View hyperparameters used
- Compare models side-by-side

**Example MLflow Results**:
```
Run Name             MAE      RMSE    R²      Params
────────────────────────────────────────────────────
LinearRegression_24h  7.3287  9.45   0.823   -
Ridge_24h             6.8022  8.91   0.847   alpha=10.0
RandomForest_24h      7.9821  10.23  0.789   n_estimators=100, depth=10
XGBoost_24h           7.1234  9.12   0.841   lr=0.1, depth=5
LSTM_24h              8.2341  10.67  0.771   hidden=64, epochs=12
```

---

### ✅ **METHOD 4: Compare All 5 During Training** (Advanced)

During training, temporary ONNX files exist:

```bash
# Inside Airflow container or during training
models/station_56_24h/
├── _tmp_linear_regression.onnx      ← Temp model 1
├── _tmp_ridge_regression.onnx       ← Temp model 2
├── _tmp_random_forest.onnx          ← Temp model 3
├── _tmp_xgboost.onnx                ← Temp model 4
├── _tmp_lstm.onnx                   ← Temp model 5
├── active_model.json                ← Points to winner
└── onnx/
    └── ridge_regression_2024-01-01_2025-10-14.onnx  ← Deployed (best)
```

**Access temp models**:
```python
import onnxruntime as rt

# Load any temp model
session = rt.InferenceSession(
    "models/station_56_24h/_tmp_xgboost.onnx",
    providers=["CPUExecutionProvider"]
)

# Make prediction
features = np.array([[...]], dtype=np.float32)
pred = session.run(None, {"float_input": features})[0][0]
print(f"XGBoost prediction: {pred:.2f}")
```

**Note**: Temp files are deleted after deployment!

---

### ✅ **METHOD 5: Switch Active Model** (Manual Override)

To use a different model type (rare):

```python
import json
from src.triton_utils import publish_to_triton

# 1. Update active_model.json
active_model = {
    "onnx_file": "xgboost_2024-01-01_2025-10-14.onnx",  # Change this
    "model_key": "xgboost",
    "station_id": 56,
    "train_start": "2024-01-01",
    "train_end": "2025-10-14",
    "is_lstm": false,
    "forecast_hour": 24,
    "n_features": 19
}

with open("models/station_56_24h/active_model.json", "w") as f:
    json.dump(active_model, f, indent=2)

# 2. Republish to Triton
publish_to_triton(
    onnx_path="models/station_56_24h/onnx/xgboost_2024-01-01_2025-10-14.onnx",
    triton_repo="triton_model_repo",
    is_lstm=False
)

# 3. Restart API (if using FastAPI backend)
# docker compose restart api
```

---

## Which Model Type is Best?

### Current Best Models (from MLflow):

| Station | Best Model | MAE |
|---------|-----------|-----|
| 56 | Linear Regression | 7.33 |
| 57 | Ridge Regression | 6.80 |
| 58 | Ridge Regression | 7.16 |
| 59 | Random Forest | 8.44 |
| 61 | Linear Regression | 7.66 |

**Observations**:
- Ridge performs best overall (stations 57, 58)
- Linear is competitive for some stations (56, 61)
- Complex models (LSTM, XGBoost) don't always win
- Best model varies by station's data characteristics

---

## Quick Reference

### Predict with Triton (All Stations):

```python
import tritonclient.http as httpclient
import numpy as np

client = httpclient.InferenceServerClient(url="localhost:8010")

# Sample features
features = np.random.rand(1, 17).astype(np.float32) * 50

input_tensor = httpclient.InferInput("float_input", features.shape, "FP32")
input_tensor.set_data_from_numpy(features)
output_tensor = httpclient.InferRequestedOutput("variable")

# Predict all stations
for model_name in ["pm25", "pm25_63", "pm25_64", "pm25_65", "pm25_66", "pm25_67"]:
    result = client.infer(model_name=model_name, inputs=[input_tensor], outputs=[output_tensor])
    pred = result.as_numpy("variable")[0][0]
    print(f"{model_name}: {pred:.2f} µg/m³")
```

### View Training History:

```bash
# CSV results
cat results/forecast_24h_results.csv | column -t -s,

# MLflow UI
open http://localhost:5001
```

### Check Which Model is Active:

```bash
# For each station
cat models/station_56_24h/active_model.json | jq -r '.model_key'
cat models/station_57_24h/active_model.json | jq -r '.model_key'
# ... etc
```

---

## Summary

✅ **Production**: Use Triton API with deployed models (one per station)

✅ **Development**: Use FastAPI `/predict` endpoint

✅ **Research**: View MLflow UI to compare all 5 model types

✅ **Automation**: Training pipeline automatically selects the best model

❌ **Don't manually manage the 5 types** - the pipeline handles it!

---

## Example Outputs

Run the complete example:

```bash
python examples/use_all_models.py
```

Run Triton-only example:

```bash
python examples/triton_inference_example.py
```

View documentation:

```bash
cat docs/TRITON_API_GUIDE.md
```

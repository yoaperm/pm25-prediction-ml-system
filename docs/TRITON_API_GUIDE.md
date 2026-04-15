# Triton Inference Server API Guide

Complete guide for using Triton Inference Server to make predictions with PM2.5 models.

## Triton Endpoints

- **HTTP Inference**: http://localhost:8010 (mapped from container port 8000)
- **gRPC Inference**: http://localhost:8011 (mapped from container port 8001)
- **Metrics**: http://localhost:8012 (mapped from container port 8002)

---

## Available Models

| Model Name | Station | Features | Input Shape | Output Shape |
|------------|---------|----------|-------------|--------------|
| `pm25` | Station 10T | 17 | [batch, 17] | [batch, 1] |
| `pm25_63` | Station 63 | 17 | [batch, 17] | [batch, 1] |
| `pm25_64` | Station 64 | 17 | [batch, 17] | [batch, 1] |
| `pm25_65` | Station 65 | 17 | [batch, 17] | [batch, 1] |
| `pm25_66` | Station 66 | 17 | [batch, 17] | [batch, 1] |
| `pm25_67` | Station 67 | 17 | [batch, 17] | [batch, 1] |

All models:
- **Backend**: ONNX Runtime
- **Max batch size**: 32
- **Input**: `float_input` (FP32)
- **Output**: `variable` (FP32)
- **Dynamic batching**: Enabled

---

## 1. Check Triton Health

```bash
# Check if Triton is ready
curl http://localhost:8010/v2/health/ready

# Check if Triton is alive
curl http://localhost:8010/v2/health/live
```

**Response**: Empty body with HTTP 200 if healthy

---

## 2. List All Models

```bash
curl http://localhost:8010/v2/models | jq '.'
```

**Response**:
```json
{
  "models": [
    {"name": "pm25", "versions": ["1"]},
    {"name": "pm25_63", "versions": ["1"]},
    {"name": "pm25_64", "versions": ["1"]},
    {"name": "pm25_65", "versions": ["1"]},
    {"name": "pm25_66", "versions": ["1"]},
    {"name": "pm25_67", "versions": ["1"]}
  ]
}
```

---

## 3. Get Model Metadata

```bash
# Get metadata for station 10T model
curl http://localhost:8010/v2/models/pm25 | jq '.'

# Get metadata for station 63 model
curl http://localhost:8010/v2/models/pm25_63 | jq '.'
```

**Response**:
```json
{
  "name": "pm25",
  "versions": ["1"],
  "platform": "onnxruntime_onnx",
  "inputs": [
    {
      "name": "float_input",
      "datatype": "FP32",
      "shape": [-1, 17]
    }
  ],
  "outputs": [
    {
      "name": "variable",
      "datatype": "FP32",
      "shape": [-1, 1]
    }
  ]
}
```

---

## 4. Get Model Config

```bash
curl http://localhost:8010/v2/models/pm25/config | jq '.'
```

---

## 5. Make Predictions

### A. Using HTTP REST API (curl)

```bash
# Example: Predict next-day PM2.5 for station 10T
curl -X POST http://localhost:8010/v2/models/pm25/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "float_input",
        "datatype": "FP32",
        "shape": [1, 17],
        "data": [
          25.3, 28.1, 30.2, 32.5, 35.0,
          28.5, 3.2, 29.1, 4.1,
          2, 92, 3, 92, 0,
          5, 15, 2
        ]
      }
    ]
  }' | jq '.outputs[0].data[0]'
```

**Response**:
```json
{
  "model_name": "pm25",
  "model_version": "1",
  "outputs": [
    {
      "name": "variable",
      "datatype": "FP32",
      "shape": [1, 1],
      "data": [29.45]
    }
  ]
}
```

### B. Using Python (tritonclient)

```python
import numpy as np
import tritonclient.http as httpclient

# Connect to Triton
triton_client = httpclient.InferenceServerClient(url="localhost:8010")

# Prepare input data (17 features)
features = np.array([
    [25.3, 28.1, 30.2, 32.5, 35.0,  # lags: 1,2,3,5,7 days
     28.5, 3.2,                     # rolling_mean_3, rolling_std_3
     29.1, 4.1,                     # rolling_mean_7, rolling_std_7
     2, 92, 3, 92, 0,               # month, day_of_year, day_of_week, etc.
     5, 15, 2]                      # time features
], dtype=np.float32)

# Create input/output objects
input_tensor = httpclient.InferInput("float_input", features.shape, "FP32")
input_tensor.set_data_from_numpy(features)

output_tensor = httpclient.InferRequestedOutput("variable")

# Make prediction
result = triton_client.infer(
    model_name="pm25",
    inputs=[input_tensor],
    outputs=[output_tensor]
)

# Get prediction
prediction = result.as_numpy("variable")[0][0]
print(f"Predicted PM2.5: {prediction:.2f} µg/m³")
```

---

## 6. Batch Predictions

Triton supports batch inference (up to 32 samples per request):

```python
import numpy as np
import tritonclient.http as httpclient

triton_client = httpclient.InferenceServerClient(url="localhost:8010")

# Batch of 5 samples (5 different time points)
features_batch = np.array([
    [25.3, 28.1, 30.2, 32.5, 35.0, 28.5, 3.2, 29.1, 4.1, 2, 92, 3, 92, 0, 5, 15, 2],
    [26.1, 25.3, 28.1, 30.2, 32.5, 27.8, 2.9, 28.3, 3.8, 2, 93, 4, 93, 0, 6, 15, 2],
    [27.5, 26.1, 25.3, 28.1, 30.2, 26.4, 2.5, 27.1, 3.2, 2, 94, 5, 94, 0, 0, 15, 2],
    [29.3, 27.5, 26.1, 25.3, 28.1, 27.1, 2.8, 27.8, 3.5, 2, 95, 6, 95, 1, 1, 15, 2],
    [31.2, 29.3, 27.5, 26.1, 25.3, 28.5, 3.1, 28.7, 3.9, 2, 96, 0, 96, 1, 2, 15, 2],
], dtype=np.float32)

input_tensor = httpclient.InferInput("float_input", features_batch.shape, "FP32")
input_tensor.set_data_from_numpy(features_batch)

output_tensor = httpclient.InferRequestedOutput("variable")

result = triton_client.infer(
    model_name="pm25",
    inputs=[input_tensor],
    outputs=[output_tensor]
)

predictions = result.as_numpy("variable").flatten()
print(f"Batch predictions: {predictions}")
# Output: [29.45, 28.32, 27.89, 29.01, 30.15]
```

---

## 7. Using Different Models (Stations)

```python
# Predict for Station 63
result_63 = triton_client.infer(
    model_name="pm25_63",  # Change model name
    inputs=[input_tensor],
    outputs=[output_tensor]
)

# Predict for Station 65
result_65 = triton_client.infer(
    model_name="pm25_65",  # Change model name
    inputs=[input_tensor],
    outputs=[output_tensor]
)

# Parallel predictions for all stations
stations = [10, 63, 64, 65, 66, 67]
model_names = ["pm25", "pm25_63", "pm25_64", "pm25_65", "pm25_66", "pm25_67"]

predictions = {}
for station, model_name in zip(stations, model_names):
    result = triton_client.infer(
        model_name=model_name,
        inputs=[input_tensor],
        outputs=[output_tensor]
    )
    predictions[station] = result.as_numpy("variable")[0][0]
    print(f"Station {station}: {predictions[station]:.2f} µg/m³")
```

---

## 8. Using gRPC (Faster for Production)

```python
import numpy as np
import tritonclient.grpc as grpcclient

# Connect via gRPC (faster than HTTP)
triton_client = grpcclient.InferenceServerClient(url="localhost:8011")

features = np.array([[25.3, 28.1, 30.2, 32.5, 35.0, 28.5, 3.2, 
                       29.1, 4.1, 2, 92, 3, 92, 0, 5, 15, 2]], 
                     dtype=np.float32)

# gRPC API is similar but different classes
input_tensor = grpcclient.InferInput("float_input", features.shape, "FP32")
input_tensor.set_data_from_numpy(features)

output_tensor = grpcclient.InferRequestedOutput("variable")

result = triton_client.infer(
    model_name="pm25",
    inputs=[input_tensor],
    outputs=[output_tensor]
)

prediction = result.as_numpy("variable")[0][0]
print(f"Predicted PM2.5: {prediction:.2f} µg/m³")
```

**gRPC is ~2-3x faster** than HTTP for high-throughput scenarios.

---

## 9. Monitoring & Metrics

```bash
# Get Triton metrics (Prometheus format)
curl http://localhost:8012/metrics

# Key metrics:
# - nv_inference_request_success: successful requests
# - nv_inference_request_failure: failed requests
# - nv_inference_queue_duration_us: queue time
# - nv_inference_compute_infer_duration_us: inference time
# - nv_inference_request_duration_us: total request time
```

---

## 10. Model Management

### Check Model Status

```bash
curl http://localhost:8010/v2/models/pm25/ready
```

### Load/Unload Models (if using explicit mode)

```bash
# Load a model
curl -X POST http://localhost:8010/v2/repository/models/pm25/load

# Unload a model
curl -X POST http://localhost:8010/v2/repository/models/pm25/unload
```

**Note**: Current setup uses **poll mode** (auto-reload every 30s), so models are loaded automatically.

---

## 11. Error Handling

```python
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

try:
    result = triton_client.infer(
        model_name="pm25",
        inputs=[input_tensor],
        outputs=[output_tensor]
    )
    prediction = result.as_numpy("variable")[0][0]
except InferenceServerException as e:
    print(f"Triton error: {e}")
except Exception as e:
    print(f"Error: {e}")
```

---

## 12. Feature Engineering Helper

The 17 features expected by the models:

```python
def build_features(pm25_history):
    """
    Build 17 features from PM2.5 history (≥15 days).
    
    Parameters
    ----------
    pm25_history : array-like
        Daily PM2.5 values (newest last)
    
    Returns
    -------
    features : np.ndarray shape (17,)
    """
    import numpy as np
    import pandas as pd
    
    df = pd.DataFrame({"pm25": pm25_history})
    
    # Lag features (5)
    lags = [1, 2, 3, 5, 7]
    lag_values = [df["pm25"].iloc[-lag] for lag in lags]
    
    # Rolling statistics (4)
    rolling_3_mean = df["pm25"].iloc[-3:].mean()
    rolling_3_std = df["pm25"].iloc[-3:].std()
    rolling_7_mean = df["pm25"].iloc[-7:].mean()
    rolling_7_std = df["pm25"].iloc[-7:].std()
    
    # Time features (8) - example for today
    import datetime
    today = datetime.date.today()
    month = today.month
    day_of_year = today.timetuple().tm_yday
    day_of_week = today.weekday()
    year = today.year
    quarter = (month - 1) // 3 + 1
    week_of_year = today.isocalendar()[1]
    day_of_month = today.day
    is_weekend = 1 if day_of_week >= 5 else 0
    
    features = np.array(
        lag_values + 
        [rolling_3_mean, rolling_3_std, rolling_7_mean, rolling_7_std] +
        [month, day_of_year, day_of_week, year, quarter, 
         week_of_year, day_of_month, is_weekend]
    , dtype=np.float32)
    
    return features.reshape(1, 17)
```

---

## 13. Complete Example: End-to-End Prediction

```python
import numpy as np
import pandas as pd
import tritonclient.http as httpclient
from datetime import datetime, timedelta

# 1. Connect to Triton
triton_client = httpclient.InferenceServerClient(url="localhost:8010")

# 2. Prepare historical data (last 15 days of PM2.5)
pm25_history = [
    28.5, 29.1, 30.2, 25.3, 27.8,  # 15 days ago
    26.4, 28.9, 31.2, 29.5, 27.1,  # 10 days ago
    30.5, 28.3, 29.8, 32.1, 30.7   # 5 days ago → yesterday
]

# 3. Build features (helper function from above)
features = build_features(pm25_history)

# 4. Create Triton input
input_tensor = httpclient.InferInput("float_input", features.shape, "FP32")
input_tensor.set_data_from_numpy(features)

output_tensor = httpclient.InferRequestedOutput("variable")

# 5. Make prediction for each station
stations = {
    "Station 10T": "pm25",
    "Station 63": "pm25_63",
    "Station 64": "pm25_64",
    "Station 65": "pm25_65",
    "Station 66": "pm25_66",
    "Station 67": "pm25_67",
}

print(f"Predictions for {datetime.now().date() + timedelta(days=1)}:")
print("-" * 50)

for station_name, model_name in stations.items():
    result = triton_client.infer(
        model_name=model_name,
        inputs=[input_tensor],
        outputs=[output_tensor]
    )
    
    prediction = result.as_numpy("variable")[0][0]
    
    # Classify air quality
    if prediction <= 25:
        quality = "Good (Green)"
    elif prediction <= 37:
        quality = "Moderate (Yellow)"
    elif prediction <= 50:
        quality = "Unhealthy for Sensitive (Orange)"
    else:
        quality = "Unhealthy (Red)"
    
    print(f"{station_name:15} {prediction:6.2f} µg/m³  [{quality}]")
```

**Output**:
```
Predictions for 2026-04-17:
--------------------------------------------------
Station 10T     29.45 µg/m³  [Moderate (Yellow)]
Station 63      31.23 µg/m³  [Moderate (Yellow)]
Station 64      28.87 µg/m³  [Moderate (Yellow)]
Station 65      33.12 µg/m³  [Moderate (Yellow)]
Station 66      30.54 µg/m³  [Moderate (Yellow)]
Station 67      32.98 µg/m³  [Moderate (Yellow)]
```

---

## 14. Performance Benchmarking

```python
import time
import numpy as np
import tritonclient.http as httpclient

triton_client = httpclient.InferenceServerClient(url="localhost:8010")

# Warm-up
features = np.random.rand(1, 17).astype(np.float32)
input_tensor = httpclient.InferInput("float_input", features.shape, "FP32")
input_tensor.set_data_from_numpy(features)
output_tensor = httpclient.InferRequestedOutput("variable")

for _ in range(10):
    triton_client.infer(model_name="pm25", inputs=[input_tensor], outputs=[output_tensor])

# Benchmark
n_requests = 100
start = time.time()

for _ in range(n_requests):
    result = triton_client.infer(
        model_name="pm25",
        inputs=[input_tensor],
        outputs=[output_tensor]
    )

elapsed = time.time() - start
print(f"Requests: {n_requests}")
print(f"Total time: {elapsed:.2f}s")
print(f"Avg latency: {elapsed/n_requests*1000:.2f}ms")
print(f"Throughput: {n_requests/elapsed:.2f} req/s")
```

**Typical Results** (CPU):
- Latency: 2-5ms per request
- Throughput: 200-500 req/s (single client)

---

## 15. Deployment Considerations

### EC2 Deployment

Update Triton URL in `.env` or docker-compose.yml:

```bash
# If Triton is on same EC2 (docker-compose)
TRITON_URL=triton:8000

# If Triton is on separate server
TRITON_URL=triton.example.com:8000
```

### Load Balancing

For high traffic, run multiple Triton instances behind a load balancer:

```
                 ┌─> Triton Instance 1
Client → LB  ──┼─> Triton Instance 2
                 └─> Triton Instance 3
```

### GPU Acceleration

To use GPU (if available):

```yaml
# docker-compose.yml
triton:
  image: nvcr.io/nvidia/tritonserver:24.08-py3
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

---

## Summary

- **6 models** deployed: `pm25`, `pm25_63`, `pm25_64`, `pm25_65`, `pm25_66`, `pm25_67`
- **Input**: 17 features (float32)
- **Output**: Single PM2.5 prediction (float32)
- **Batch support**: Up to 32 samples
- **Auto-reload**: Models updated every 30s from `triton_model_repo/`
- **Protocols**: HTTP (port 8010) and gRPC (port 8011)
- **Production**: Use gRPC for 2-3x better performance

# Station-Specific Prediction API Guide

How to make predictions for stations 56, 57, 58, 59, 61 only.

---

## Current Setup

### Stations in PostgreSQL:
```
56, 57, 58, 59, 61
```

### Deployed Models (after cleanup):
```
pm25_56 → Station 56
pm25_57 → Station 57
pm25_58 → Station 58
pm25_59 → Station 59
pm25_61 → Station 61
```

---

## Method 1: Triton API (Direct - Fastest)

### Single Station Prediction

```python
import numpy as np
import tritonclient.http as httpclient

# Connect to Triton
client = httpclient.InferenceServerClient(url="localhost:8010")

# Prepare features (19 features for hourly 24h forecast)
features = np.array([[
    # Lags (6): pm25 for t-1, t-2, t-3, t-6, t-12, t-24 hours
    28.5, 29.1, 30.2, 32.5, 31.8, 35.0,
    
    # Rolling statistics (6): mean/std for 6h, 12h, 24h windows
    28.5, 3.2,  # rolling_mean_6h, rolling_std_6h
    29.1, 4.1,  # rolling_mean_12h, rolling_std_12h
    30.2, 3.8,  # rolling_mean_24h, rolling_std_24h
    
    # Diffs (2)
    0.5,   # pm25_diff_1h
    -2.3,  # pm25_diff_24h
    
    # Time features (5)
    14,    # hour (14:00)
    2,     # day_of_week (Wednesday)
    4,     # month (April)
    107,   # day_of_year
    0      # is_weekend
]], dtype=np.float32)

# Create input tensor
input_tensor = httpclient.InferInput("float_input", features.shape, "FP32")
input_tensor.set_data_from_numpy(features)

# Request output
output_tensor = httpclient.InferRequestedOutput("variable")

# Predict for Station 56
result = client.infer(
    model_name="pm25_56",
    inputs=[input_tensor],
    outputs=[output_tensor]
)

prediction = result.as_numpy("variable")[0][0]
print(f"Station 56 prediction (next 24h): {prediction:.2f} µg/m³")
```

### Predict All 5 Stations

```python
import numpy as np
import tritonclient.http as httpclient

def predict_all_stations(features):
    """
    Make predictions for all 5 stations.
    
    Parameters
    ----------
    features : np.ndarray shape (1, 19)
        Engineered features for prediction
    
    Returns
    -------
    predictions : dict
        {station_id: prediction_value}
    """
    client = httpclient.InferenceServerClient(url="localhost:8010")
    
    # Station IDs and model names
    stations = {
        56: "pm25_56",
        57: "pm25_57",
        58: "pm25_58",
        59: "pm25_59",
        61: "pm25_61",
    }
    
    predictions = {}
    
    # Prepare tensors
    input_tensor = httpclient.InferInput("float_input", features.shape, "FP32")
    input_tensor.set_data_from_numpy(features)
    output_tensor = httpclient.InferRequestedOutput("variable")
    
    # Predict each station
    for station_id, model_name in stations.items():
        try:
            result = client.infer(
                model_name=model_name,
                inputs=[input_tensor],
                outputs=[output_tensor]
            )
            prediction = result.as_numpy("variable")[0][0]
            predictions[station_id] = round(float(prediction), 2)
        except Exception as e:
            print(f"Error predicting station {station_id}: {e}")
            predictions[station_id] = None
    
    return predictions


# Usage
features = np.array([[
    28.5, 29.1, 30.2, 32.5, 31.8, 35.0,  # lags
    28.5, 3.2, 29.1, 4.1, 30.2, 3.8,     # rolling stats
    0.5, -2.3,                           # diffs
    14, 2, 4, 107, 0                     # time features
]], dtype=np.float32)

predictions = predict_all_stations(features)

print("\nPredictions for next 24 hours:")
print(f"{'Station':<12} {'PM2.5 (µg/m³)':<15} {'Air Quality'}")
print("-" * 50)

for station_id in [56, 57, 58, 59, 61]:
    pred = predictions[station_id]
    if pred is not None:
        if pred <= 25:
            quality = "🟢 Good"
        elif pred <= 37:
            quality = "🟡 Moderate"
        elif pred <= 50:
            quality = "🟠 Unhealthy"
        else:
            quality = "🔴 Very Unhealthy"
        
        print(f"Station {station_id:<5} {pred:>6.2f}          {quality}")
    else:
        print(f"Station {station_id:<5} {'ERROR':<15}")
```

**Output**:
```
Predictions for next 24 hours:
Station      PM2.5 (µg/m³)   Air Quality
--------------------------------------------------
Station 56    29.45          🟡 Moderate
Station 57    27.83          🟡 Moderate
Station 58    31.22          🟡 Moderate
Station 59    33.15          🟡 Moderate
Station 61    28.91          🟡 Moderate
```

---

## Method 2: FastAPI Endpoint (Easier - With Feature Engineering)

### Single Station

```python
import requests

API_URL = "http://localhost:8001"
API_KEY = "foonalert-secret-key"

headers = {"X-API-Key": API_KEY}

# Request body
data = {
    "station_id": 56,  # Specify station
    "history": [
        # Last 24+ hours of PM2.5 data
        {"timestamp": "2026-04-16 00:00:00", "pm25": 28.5},
        {"timestamp": "2026-04-16 01:00:00", "pm25": 29.1},
        {"timestamp": "2026-04-16 02:00:00", "pm25": 30.2},
        # ... more hours
        {"timestamp": "2026-04-16 23:00:00", "pm25": 30.7},
    ]
}

response = requests.post(
    f"{API_URL}/predict/station",
    json=data,
    headers=headers
)

if response.status_code == 200:
    result = response.json()
    print(f"Station {result['station_id']}: {result['prediction']} µg/m³")
    print(f"Forecast time: {result['forecast_time']}")
    print(f"Model: {result['model_key']}")
else:
    print(f"Error: {response.status_code} - {response.text}")
```

### All 5 Stations

```python
import requests
from datetime import datetime, timedelta

API_URL = "http://localhost:8001"
API_KEY = "foonalert-secret-key"

headers = {"X-API-Key": API_KEY}

# Generate sample history (last 48 hours)
def generate_sample_history(base_value=30.0):
    history = []
    start_time = datetime.now() - timedelta(hours=48)
    
    for i in range(48):
        timestamp = start_time + timedelta(hours=i)
        # Simulate varying PM2.5 values
        pm25 = base_value + np.random.uniform(-5, 5)
        history.append({
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "pm25": round(pm25, 2)
        })
    
    return history

# Predict all stations
stations = [56, 57, 58, 59, 61]
predictions = {}

print(f"{'Station':<12} {'Prediction':<15} {'Model Type':<20} {'Status'}")
print("-" * 70)

for station_id in stations:
    data = {
        "station_id": station_id,
        "history": generate_sample_history(base_value=28.0 + station_id % 5)
    }
    
    try:
        response = requests.post(
            f"{API_URL}/predict/station",
            json=data,
            headers=headers,
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            predictions[station_id] = result['prediction']
            print(f"Station {station_id:<5} {result['prediction']:>6.2f} µg/m³    "
                  f"{result['model_key']:<20} ✓")
        else:
            print(f"Station {station_id:<5} {'ERROR':<15} {response.status_code:<20} ✗")
            
    except Exception as e:
        print(f"Station {station_id:<5} {'ERROR':<15} {str(e):<20} ✗")

# Summary
print(f"\n{'='*70}")
print(f"Successfully predicted: {len(predictions)}/5 stations")
if predictions:
    avg_pm25 = sum(predictions.values()) / len(predictions)
    print(f"Average PM2.5 forecast: {avg_pm25:.2f} µg/m³")
```

---

## Method 3: Batch Prediction API

### Create a batch endpoint in your FastAPI:

```python
# In src/api.py (add this endpoint)

@app.post("/predict/batch")
async def predict_batch(
    request: BatchPredictionRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Batch predictions for multiple stations.
    
    Request body:
    {
        "stations": [56, 57, 58, 59, 61],
        "history": {
            "56": [...],  # hourly data for station 56
            "57": [...],  # hourly data for station 57
            ...
        }
    }
    """
    results = {}
    
    for station_id in request.stations:
        if str(station_id) not in request.history:
            results[station_id] = {"error": "No history provided"}
            continue
        
        try:
            # Load model for this station
            model_path = f"models/station_{station_id}_24h/active_model.json"
            # ... feature engineering ...
            # ... prediction ...
            
            results[station_id] = {
                "prediction": prediction,
                "forecast_time": forecast_time,
                "model_key": model_key
            }
        except Exception as e:
            results[station_id] = {"error": str(e)}
    
    return {
        "predictions": results,
        "timestamp": datetime.utcnow().isoformat()
    }
```

### Use batch endpoint:

```python
import requests

data = {
    "stations": [56, 57, 58, 59, 61],
    "history": {
        "56": [...],  # 48 hours of data
        "57": [...],
        "58": [...],
        "59": [...],
        "61": [...]
    }
}

response = requests.post(
    f"{API_URL}/predict/batch",
    json=data,
    headers=headers
)

results = response.json()
for station_id, result in results['predictions'].items():
    if 'error' not in result:
        print(f"Station {station_id}: {result['prediction']} µg/m³")
```

---

## Method 4: Query PostgreSQL + Predict

### Automatically fetch data from PostgreSQL:

```python
import psycopg2
import numpy as np
from datetime import datetime, timedelta
import tritonclient.http as httpclient

def fetch_and_predict_station(station_id, db_url):
    """
    Fetch recent data from PostgreSQL and make prediction.
    
    Parameters
    ----------
    station_id : int
        Station ID (56, 57, 58, 59, or 61)
    db_url : str
        PostgreSQL connection string
    
    Returns
    -------
    prediction : float
        PM2.5 prediction for next 24 hours
    """
    # 1. Connect to database
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    
    # 2. Query last 48 hours of data
    query = """
        SELECT timestamp, pm25
        FROM pm25_raw_hourly
        WHERE station_id = %s
          AND timestamp >= NOW() - INTERVAL '48 hours'
        ORDER BY timestamp DESC
        LIMIT 48
    """
    cur.execute(query, (station_id,))
    data = cur.fetchall()
    cur.close()
    conn.close()
    
    if len(data) < 30:
        raise ValueError(f"Insufficient data for station {station_id}")
    
    # 3. Build features (19 features for 24h forecast)
    # ... feature engineering logic ...
    features = build_features_from_history(data)
    
    # 4. Connect to Triton
    client = httpclient.InferenceServerClient(url="localhost:8010")
    
    # 5. Make prediction
    input_tensor = httpclient.InferInput("float_input", features.shape, "FP32")
    input_tensor.set_data_from_numpy(features)
    output_tensor = httpclient.InferRequestedOutput("variable")
    
    result = client.infer(
        model_name=f"pm25_{station_id}",
        inputs=[input_tensor],
        outputs=[output_tensor]
    )
    
    prediction = result.as_numpy("variable")[0][0]
    return round(float(prediction), 2)


# Usage
DB_URL = "postgresql://postgres:postgres@localhost:5432/pm25"

for station_id in [56, 57, 58, 59, 61]:
    try:
        prediction = fetch_and_predict_station(station_id, DB_URL)
        print(f"Station {station_id}: {prediction} µg/m³")
    except Exception as e:
        print(f"Station {station_id}: ERROR - {e}")
```

---

## Complete Working Example

```python
#!/usr/bin/env python3
"""
Predict all 5 stations using Triton API.
"""
import numpy as np
import tritonclient.http as httpclient
from datetime import datetime, timedelta

def build_sample_features():
    """Build realistic 19 features for 24h forecast."""
    now = datetime.now()
    
    return np.array([[
        # Lags (6): last 1,2,3,6,12,24 hours
        28.5, 29.1, 30.2, 32.5, 31.8, 35.0,
        
        # Rolling stats (6)
        28.5, 3.2,   # mean_6h, std_6h
        29.1, 4.1,   # mean_12h, std_12h
        30.2, 3.8,   # mean_24h, std_24h
        
        # Diffs (2)
        0.5,    # diff_1h
        -2.3,   # diff_24h
        
        # Time features (5)
        now.hour,
        now.weekday(),
        now.month,
        now.timetuple().tm_yday,
        1 if now.weekday() >= 5 else 0
    ]], dtype=np.float32)


def predict_all_stations():
    """Predict PM2.5 for all 5 stations."""
    print("="*70)
    print("PM2.5 Predictions for All Stations")
    print("="*70)
    
    # Connect to Triton
    client = httpclient.InferenceServerClient(url="localhost:8010")
    
    # Build features
    features = build_sample_features()
    
    # Prepare input
    input_tensor = httpclient.InferInput("float_input", features.shape, "FP32")
    input_tensor.set_data_from_numpy(features)
    output_tensor = httpclient.InferRequestedOutput("variable")
    
    # Predict all stations
    stations = [56, 57, 58, 59, 61]
    forecast_time = datetime.now() + timedelta(hours=24)
    
    print(f"\nForecast for: {forecast_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"\n{'Station ID':<12} {'PM2.5 (µg/m³)':<15} {'Air Quality':<20} {'Model'}")
    print("-"*70)
    
    for station_id in stations:
        try:
            # Make prediction
            result = client.infer(
                model_name=f"pm25_{station_id}",
                inputs=[input_tensor],
                outputs=[output_tensor]
            )
            
            prediction = result.as_numpy("variable")[0][0]
            
            # Classify air quality
            if prediction <= 25:
                quality = "🟢 Good"
            elif prediction <= 37:
                quality = "🟡 Moderate"
            elif prediction <= 50:
                quality = "🟠 Unhealthy (Sensitive)"
            elif prediction <= 90:
                quality = "🔴 Unhealthy"
            else:
                quality = "🟣 Very Unhealthy"
            
            # Get model metadata
            model_info = client.get_model_metadata(f"pm25_{station_id}")
            platform = model_info.get('platform', 'onnx')
            
            print(f"{station_id:<12} {prediction:>6.2f}          {quality:<20} {platform}")
            
        except Exception as e:
            print(f"{station_id:<12} {'ERROR':<15} {str(e)}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    predict_all_stations()
```

**Save as**: `examples/predict_5_stations.py`

**Run**:
```bash
python examples/predict_5_stations.py
```

---

## Summary

### Quick Reference

| Method | Use Case | Complexity | Performance |
|--------|----------|------------|-------------|
| **Triton Direct** | Production API | Medium | Fastest (2ms) |
| **FastAPI** | Easy integration | Low | Medium (10-20ms) |
| **Batch API** | Multiple stations | Low | Good |
| **PostgreSQL + Triton** | Automated | High | Fast |

### Station Model Names

```python
STATION_MODELS = {
    56: "pm25_56",
    57: "pm25_57",
    58: "pm25_58",
    59: "pm25_59",
    61: "pm25_61",
}
```

### Features Required

- **19 features** for 24h hourly forecast
- **17 features** for daily forecast (if using station 10T model)

### Endpoints

- **Triton**: http://localhost:8010/v2/models/pm25_{id}/infer
- **FastAPI**: http://localhost:8001/predict/station
- **Health**: http://localhost:8010/v2/health/ready

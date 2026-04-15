# Quick Start: Predict PM2.5 for 5 Stations

**Stations**: 56, 57, 58, 59, 61

---

## 🚀 Quick Setup (3 Steps)

### 1. Train Models (One-Time Setup)

```bash
# Train all 5 stations
for station in 56 57 58 59 61; do
  docker exec pm25-prediction-ml-system-airflow-scheduler-1 \
    airflow dags trigger pm25_24h_training -c "{\"station_id\": $station}"
done
```

⏱️ Takes ~5-10 minutes per station

### 2. Verify Models are Ready

```bash
curl http://localhost:8010/v2/models | jq -r '.models[].name'
```

✅ Should show: `pm25_56`, `pm25_57`, `pm25_58`, `pm25_59`, `pm25_61`

### 3. Make Predictions

```bash
python examples/predict_5_stations.py
```

---

## 📊 Usage Examples

### Python (Triton API)

```python
import numpy as np
import tritonclient.http as httpclient

client = httpclient.InferenceServerClient(url="localhost:8010")

# Build features (19 features)
features = np.array([[
    28.5, 29.1, 30.2, 32.5, 31.8, 35.0,  # 6 lags
    28.5, 3.2, 29.1, 4.1, 30.2, 3.8,     # 6 rolling stats
    0.5, -2.3,                           # 2 diffs
    14, 2, 4, 107, 0                     # 5 time features
]], dtype=np.float32)

input_tensor = httpclient.InferInput("float_input", features.shape, "FP32")
input_tensor.set_data_from_numpy(features)
output_tensor = httpclient.InferRequestedOutput("variable")

# Predict each station
for station_id in [56, 57, 58, 59, 61]:
    result = client.infer(
        model_name=f"pm25_{station_id}",
        inputs=[input_tensor],
        outputs=[output_tensor]
    )
    pred = result.as_numpy("variable")[0][0]
    print(f"Station {station_id}: {pred:.2f} µg/m³")
```

### cURL (REST API)

```bash
curl -X POST http://localhost:8010/v2/models/pm25_56/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [{
      "name": "float_input",
      "datatype": "FP32",
      "shape": [1, 19],
      "data": [28.5, 29.1, 30.2, 32.5, 31.8, 35.0, 28.5, 3.2, 29.1, 4.1, 30.2, 3.8, 0.5, -2.3, 14, 2, 4, 107, 0]
    }]
  }' | jq '.outputs[0].data[0]'
```

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `examples/predict_5_stations.py` | **Complete working example** |
| `docs/STATION_PREDICTION_API.md` | Full API documentation |
| `docs/SETUP_5_STATIONS.md` | Detailed setup guide |
| `docs/TRITON_API_GUIDE.md` | Triton API reference |

---

## 🔧 Station Model Names

```python
STATIONS = {
    56: "pm25_56",
    57: "pm25_57",
    58: "pm25_58",
    59: "pm25_59",
    61: "pm25_61",
}
```

---

## 📈 Monitoring

```bash
# View training results
cat results/forecast_24h_results.csv | tail -5

# MLflow UI
open http://localhost:5001

# Airflow UI
open http://localhost:8080
```

---

## 🔄 Auto-Retraining

```bash
# Enable daily monitoring (runs at 02:00 UTC)
docker exec pm25-prediction-ml-system-airflow-scheduler-1 \
  airflow dags unpause pm25_24h_pipeline
```

**Auto-retrains when**:
- MAE > 9.0 µg/m³
- PSI > 0.2 (feature drift)

---

## 🌐 Endpoints

| Service | URL | Purpose |
|---------|-----|---------|
| **Triton HTTP** | http://localhost:8010 | Predictions (fastest) |
| **Triton gRPC** | http://localhost:8011 | Predictions (2-3x faster) |
| **MLflow** | http://localhost:5001 | Experiment tracking |
| **Airflow** | http://localhost:8080 | Pipeline management |
| **FastAPI** | http://localhost:8001 | Alternative API |
| **Streamlit** | http://localhost:8501 | Dashboard |

---

## ✅ Quick Reference

### Train one station
```bash
docker exec pm25-prediction-ml-system-airflow-scheduler-1 \
  airflow dags trigger pm25_24h_training -c '{"station_id": 56}'
```

### Check model status
```bash
curl http://localhost:8010/v2/models/pm25_56
```

### Predict (Python)
```python
import tritonclient.http as httpclient
client = httpclient.InferenceServerClient(url="localhost:8010")
# ... (see full example above)
```

### Predict (Script)
```bash
python examples/predict_5_stations.py
```

---

## 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| No models found | Run training (step 1) |
| Connection refused | `docker compose up triton` |
| Insufficient data | Check PostgreSQL has 3.5y of data |
| Wrong predictions | Verify feature engineering (19 features) |

---

## 📚 Documentation

- **Setup Guide**: `docs/SETUP_5_STATIONS.md`
- **API Reference**: `docs/STATION_PREDICTION_API.md`
- **Triton Guide**: `docs/TRITON_API_GUIDE.md`
- **Model Usage**: `docs/MODEL_USAGE_GUIDE.md`

---

## 💡 Key Concepts

### 5 Model Types (Trained)
During training, 5 algorithms compete:
1. Linear Regression
2. Ridge Regression
3. Random Forest
4. XGBoost
5. LSTM

**Only the best gets deployed per station!**

### 5 Stations (Deployed)
Each station has ONE active model:
- pm25_56 (might be Ridge)
- pm25_57 (might be XGBoost)
- pm25_58 (might be Linear)
- pm25_59 (might be Random Forest)
- pm25_61 (might be Ridge)

**Different stations use different algorithms based on their data!**

---

## 🎯 Next Steps

1. ✅ Train models for all 5 stations
2. ✅ Run `python examples/predict_5_stations.py`
3. ✅ Enable auto-monitoring
4. ✅ Integrate predictions into your application

**Done!** 🎉

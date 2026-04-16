# Setup and Use 5 Stations (56, 57, 58, 59, 61)

Complete guide to train and use models for stations 56, 57, 58, 59, 61 only.

---

## Step 1: Ensure Data is in PostgreSQL

### Check Current Data

```bash
docker exec pm25-prediction-ml-system-postgres-1 psql -U postgres -d pm25 -c "
SELECT station_id, COUNT(*) as rows, 
       MIN(timestamp) as first, MAX(timestamp) as last 
FROM pm25_raw_hourly 
GROUP BY station_id 
ORDER BY station_id;
"
```

**Expected output**:
```
station_id | rows  | first               | last
-----------+-------+---------------------+---------------------
56         | 19948 | 2024-01-01 00:00:00 | 2026-04-15 23:00:00
57         | 19996 | 2024-01-01 00:00:00 | 2026-04-15 23:00:00
58         | 19872 | 2024-01-01 00:00:00 | 2026-04-15 23:00:00
59         | 16123 | 2024-01-01 00:00:00 | 2026-04-15 21:00:00
61         | 20064 | 2024-01-01 00:00:00 | 2026-04-15 23:00:00
```

### If No Data: Load Historical Data

```bash
# Option A: Backfill from CSV (if you have historical data)
docker exec pm25-prediction-ml-system-airflow-scheduler-1 \
  airflow dags trigger pm25_backfill_snapshot

# Option B: Start hourly ingestion (for new data going forward)
docker exec pm25-prediction-ml-system-airflow-scheduler-1 \
  airflow dags unpause pm25_hourly_ingest
```

---

## Step 2: Train Models for All 5 Stations

### Option A: Manual Trigger (Recommended for first time)

```bash
# Train each station one by one
for station in 56 57 58 59 61; do
  echo "Training station $station..."
  docker exec pm25-prediction-ml-system-airflow-scheduler-1 \
    airflow dags trigger pm25_24h_training \
    -c "{\"station_id\": $station}"
  sleep 5
done
```

**This will**:
- Query 3.5 years of data from PostgreSQL
- Train 5 models (Linear, Ridge, Random Forest, XGBoost, LSTM)
- Evaluate and deploy the best model
- Save to `models/station_{id}_24h/`
- Publish to Triton as `pm25_{id}`

### Option B: Enable Automatic Monitoring (for ongoing)

```bash
# Unpause the monitoring DAG (runs daily at 02:00 UTC)
docker exec pm25-prediction-ml-system-airflow-scheduler-1 \
  airflow dags unpause pm25_24h_pipeline
```

**This will automatically**:
- Check model performance daily (14-day rolling window: RMSE & PSI)
- Retrain if RMSE > 13.0 or PSI > 0.2

### Monitor Training Progress

```bash
# Watch Airflow UI
open http://localhost:8080

# Check logs
docker compose logs -f airflow-scheduler | grep "pm25_24h_training"

# View MLflow experiments
open http://localhost:5001
```

---

## Step 3: Verify Models are Deployed

### Check Triton Models

```bash
curl http://localhost:8010/v2/models | jq '.models[].name'
```

**Expected output**:
```
"pm25_56"
"pm25_57"
"pm25_58"
"pm25_59"
"pm25_61"
```

### Check Model Metadata

```bash
# Check each station's active model
for station in 56 57 58 59 61; do
  echo "=== Station $station ==="
  curl -s http://localhost:8010/v2/models/pm25_$station | jq -r '
    "Name: " + .name + 
    "\nInput: " + .inputs[0].name + " " + (.inputs[0].shape | tostring) + 
    "\nOutput: " + .outputs[0].name + " " + (.outputs[0].shape | tostring)'
  echo
done
```

---

## Step 4: Clean Up Old Models (Optional)

### Remove models for stations 10T, 63-67

```bash
# Stop Triton
docker compose stop triton

# Remove old models from Triton repository
rm -rf triton_model_repo/pm25      # Station 10T
rm -rf triton_model_repo/pm25_63
rm -rf triton_model_repo/pm25_64
rm -rf triton_model_repo/pm25_65
rm -rf triton_model_repo/pm25_66
rm -rf triton_model_repo/pm25_67

# Remove old model directories (if they exist)
rm -rf models/station_63
rm -rf models/station_64
rm -rf models/station_65
rm -rf models/station_66
rm -rf models/station_67

# Restart Triton
docker compose start triton
```

---

## Step 5: Use Predictions

### Method 1: Direct Triton API (Fastest)

```python
import numpy as np
import tritonclient.http as httpclient

# Connect
client = httpclient.InferenceServerClient(url="localhost:8010")

# Build features (19 features for 24h forecast)
features = np.array([[
    28.5, 29.1, 30.2, 32.5, 31.8, 35.0,  # lags (6)
    28.5, 3.2, 29.1, 4.1, 30.2, 3.8,     # rolling stats (6)
    0.5, -2.3,                           # diffs (2)
    14, 2, 4, 107, 0                     # time features (5)
]], dtype=np.float32)

# Prepare tensors
input_tensor = httpclient.InferInput("float_input", features.shape, "FP32")
input_tensor.set_data_from_numpy(features)
output_tensor = httpclient.InferRequestedOutput("variable")

# Predict for each station
for station_id in [56, 57, 58, 59, 61]:
    result = client.infer(
        model_name=f"pm25_{station_id}",
        inputs=[input_tensor],
        outputs=[output_tensor]
    )
    
    prediction = result.as_numpy("variable")[0][0]
    print(f"Station {station_id}: {prediction:.2f} µg/m³")
```

**Output**:
```
Station 56: 29.45 µg/m³
Station 57: 27.83 µg/m³
Station 58: 31.22 µg/m³
Station 59: 33.15 µg/m³
Station 61: 28.91 µg/m³
```

### Method 2: Run Example Script

```bash
# Use the complete example
python examples/predict_5_stations.py
```

**Output**:
```
================================================================================
PM2.5 24-Hour Forecast for Stations 56, 57, 58, 59, 61
================================================================================

1. Connecting to Triton Inference Server...
   ✓ Connected to Triton at localhost:8010

2. Checking available models...
   ✓ pm25_56 ready
   ✓ pm25_57 ready
   ✓ pm25_58 ready
   ✓ pm25_59 ready
   ✓ pm25_61 ready

3. Building features...
   Feature shape: (1, 19)
   Sample features: [28.5, 29.1, 30.2, ...]

4. Making predictions...
   Forecast for: 2026-04-17 14:30

   Station    PM2.5        Air Quality               Category
   ----------------------------------------------------------------------
   56          29.45 µg/m³  🟡 Moderate               yellow
   57          27.83 µg/m³  🟡 Moderate               yellow
   58          31.22 µg/m³  🟡 Moderate               yellow
   59          33.15 µg/m³  🟡 Moderate               yellow
   61          28.91 µg/m³  🟡 Moderate               yellow

   ----------------------------------------------------------------------

5. Summary:
   Stations predicted: 5/5
   Average PM2.5: 30.11 µg/m³
   Highest: Station 59 (33.15 µg/m³)
   Lowest:  Station 57 (27.83 µg/m³)
   Overall air quality: 🟡 Moderate

================================================================================
✓ Predictions complete
================================================================================
```

### Method 3: REST API (If using FastAPI)

```bash
# Single station
curl -X POST http://localhost:8001/predict/station \
  -H "X-API-Key: foonalert-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "station_id": 56,
    "history": [
      {"timestamp": "2026-04-16 00:00:00", "pm25": 28.5},
      {"timestamp": "2026-04-16 01:00:00", "pm25": 29.1},
      ...
    ]
  }' | jq '.'
```

---

## Step 6: Monitor Performance

### View Training Results

```bash
# Check which models were deployed
cat results/forecast_24h_results.csv | column -t -s,

# View latest run for each station
tail -n 5 results/forecast_24h_results.csv
```

**Example output**:
```csv
station_id,train_start,train_end,best_model,new_mae,prod_mae,mae_delta,status,run_date
56,2022-10-16,2025-10-16,Ridge Regression,7.32,,,DEPLOYED,2026-04-16 10:00
57,2022-10-16,2025-10-16,Ridge Regression,6.80,,,DEPLOYED,2026-04-16 10:05
58,2022-10-16,2025-10-16,Linear Regression,7.15,,,DEPLOYED,2026-04-16 10:10
59,2022-10-16,2025-10-16,Random Forest,8.44,,,DEPLOYED,2026-04-16 10:15
61,2022-10-16,2025-10-16,XGBoost,7.65,,,DEPLOYED,2026-04-16 10:20
```

### View MLflow Experiments

```bash
# Open MLflow UI
open http://localhost:5001

# Or check via API
python3 -c "
import mlflow
mlflow.set_tracking_uri('http://localhost:5001')

for station in [56, 57, 58, 59, 61]:
    exp_name = f'pm25_24h_station_{station}'
    exp = mlflow.get_experiment_by_name(exp_name)
    if exp:
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], max_results=1)
        if not runs.empty:
            print(f'Station {station}: {runs.iloc[0][\"metrics.MAE\"]:.2f} MAE')
"
```

### Check Monitoring Logs

```bash
# View drift monitoring results
cat results/monitoring_24h_results.csv | column -t -s,

# Check recent monitoring
tail -n 5 results/monitoring_24h_results.csv
```

---

## Step 7: Automate Everything (Production Setup)

### Enable All Automated Pipelines

```bash
# 1. Hourly data ingestion (from AirBKK API)
docker exec pm25-prediction-ml-system-airflow-scheduler-1 \
  airflow dags unpause pm25_hourly_ingest

# 2. Daily monitoring and auto-retraining
docker exec pm25-prediction-ml-system-airflow-scheduler-1 \
  airflow dags unpause pm25_24h_pipeline
```

**This gives you**:
- ✅ Hourly data ingestion (runs every hour)
- ✅ Daily model monitoring (02:00 UTC)
- ✅ Auto-retraining when performance degrades
- ✅ Automatic deployment to Triton
- ✅ Logging to PostgreSQL & CSV

---

## Troubleshooting

### No models in Triton

**Check**:
```bash
docker compose logs triton | grep -i error
docker exec pm25-prediction-ml-system-triton-1 ls /models/
```

**Fix**: Re-run training (Step 2)

### "Insufficient data" error during training

**Cause**: Need at least 3.5 years of data

**Fix**: 
```bash
# Check data availability
docker exec pm25-prediction-ml-system-postgres-1 psql -U postgres -d pm25 -c "
SELECT station_id, 
       COUNT(*) as hours,
       COUNT(*)/24.0 as days,
       MIN(timestamp), MAX(timestamp)
FROM pm25_raw_hourly 
GROUP BY station_id;
"
```

If < 3.5 years, adjust training window in `dags/pm25_24h_training_dag.py`:
```python
# Change line 75 from:
data_start = today - relativedelta(years=3, months=6)
# To:
data_start = today - relativedelta(years=2, months=0)  # Use 2 years instead
```

### Connection errors

```bash
# Check all services are running
docker compose ps

# Restart if needed
docker compose restart triton api postgres
```

---

## Summary

### Final Setup

**Stations**: 56, 57, 58, 59, 61 ✅

**Models**: `pm25_56`, `pm25_57`, `pm25_58`, `pm25_59`, `pm25_61` ✅

**Endpoints**:
- Triton: http://localhost:8010
- MLflow: http://localhost:5001
- Airflow: http://localhost:8080

**Usage**:
```python
# Simple prediction
import tritonclient.http as httpclient
client = httpclient.InferenceServerClient(url="localhost:8010")

# Or use the example script
python examples/predict_5_stations.py
```

**Automation**:
- Hourly data ingestion: `pm25_hourly_ingest` DAG
- Daily monitoring: `pm25_24h_pipeline` DAG
- Auto-retraining: Triggered by monitoring

Done! 🎉

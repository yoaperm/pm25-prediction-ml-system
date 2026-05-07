# FoonAlert Demo — Data Flow & Integration Guide

> **TL;DR:** Olf/Perm train model → run insert script → predictions go to DB → demo picks up automatically

---

## 🔄 Current Data Flow

```
┌──────────────────────────────────────────────────────────────┐
│                    PRODUCTION PIPELINE                       │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  AirBKK API  →  Airflow  →  PostgreSQL (pm25_raw_hourly)    │
│  (hourly)      hourly        5 stations, 96k+ hourly         │
│               ingest         records (2023-01 → now)         │
│                                                               │
│  ML Training                                                  │
│  ────────────                                                 │
│  Check station health (drift) → Trigger retrain if needed    │
│  Train 5 models → Save predictions → Store in DB             │
│                                                               │
│  Inference (via API)                                          │
│  ──────────────                                               │
│  FastAPI → Triton Inference → ONNX models                    │
│                                                               │
│  Monitoring & Reporting                                       │
│  ──────────────────────                                       │
│  Save metrics/reports → CSV (drift, MAE, spike events)       │
│                                                               │
└──────────────────────────────────────────────────────────────┘
                            ↓
                       Streamlit Demo
                   (Read DB + CSV reports)
```

---

## 📊 Database Tables

### 1. `pm25_raw_hourly` (Input Data)

Hourly measurements from AirBKK API:

```sql
SELECT * FROM pm25_raw_hourly LIMIT 1;

-- Columns:
-- id | station_id | station_name | station_name_en | timestamp | pm25 | pm10 | temp | rh | ws | wd | ingestion_time
-- 1  | 56         | เขตดินแดง    | Din Daeng       | 2023-01-01 | 42.5 | ...  | ...  | ...| ...|...|...
```

Current state: **3 years data** (96k+ rows)

### 2. `pm25_predicted_hourly` (Model Predictions)

**This is where YOUR predictions go!**

```sql
SELECT DISTINCT model, COUNT(*) FROM pm25_predicted_hourly GROUP BY model;

-- Current:
-- model             | count
-- xgboost           | 34,870
-- ridge_regression  | 8,751
-- sarima            | 0         ← Olf adds this
-- lstm              | TBD       ← Update existing
-- transformer       | 0         ← Perm adds this
```

**Schema:**
```sql
CREATE TABLE pm25_predicted_hourly (
    id SERIAL PRIMARY KEY,
    station_id INT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    model TEXT NOT NULL,
    predicted_pm25 FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(station_id, timestamp, model)
);
```

### 3. `pm25_api_daily_predictions` (Optional, for daily forecasts)

---

## 🎯 What Demo Expects

### Live Dashboard (Page 1)

**Data source:** PostgreSQL `pm25_predicted_hourly` table

```python
# Pseudo-code of what demo does
current_pm25 = query_actual_pm25(station_id, now)
predictions = query_predictions(station_id, now, models=['sarima','lstm','transformer'])
# → Shows in model battle table
```

### Spike Replay (Page 2)

**Data source:** Mock CSV in `demo_data/replay_station{id}_{date}.csv`

- Contains actual PM2.5 + simulated predictions for demo narrative
- **Temporary** — used until real predictions cover historical spike dates

### Model Battle (Page 3)

**Data source:** CSV files
- `demo_data/model_metrics.csv` — MAE, RMSE, Spike Recall per model
- `demo_data/error_by_horizon.csv` — error breakdown by forecast horizon
- `demo_data/error_by_severity.csv` — error by PM2.5 level

---

## 📋 Integration Checklist

### For Olf (SARIMA)

1. **Train SARIMA model** on all 5 stations
   - Input: Last 3 years hourly PM2.5 from `pm25_raw_hourly`
   - Output: predictions for all stations (can be any date range, but recommend recent 6-12 months for faster testing)

2. **Export to CSV**
   ```
   Columns: timestamp (UTC datetime), predicted_pm25 (float)
   File: results/predictions_sarima_56.csv (per station)
   ```

3. **Insert into DB** using helper script
   ```bash
   python scripts/insert_predictions_to_db.py results/predictions_sarima_56.csv \
     --station 56 --model sarima
   python scripts/insert_predictions_to_db.py results/predictions_sarima_57.csv \
     --station 57 --model sarima
   # ... repeat for 58, 59, 61
   ```

4. **Verify in DB**
   ```sql
   SELECT COUNT(*) FROM pm25_predicted_hourly WHERE model='sarima';
   -- Should be > 0
   ```

5. **Test demo**
   - Refresh http://EC2:8502
   - Live Dashboard → SARIMA should appear in model battle table ✅
   - Predictions should match your CSV ✅

### For Perm (Transformer)

**Exact same steps, but `--model transformer`**

### For Sunta (LSTM Update)

If you have new LSTM predictions:

```bash
python scripts/insert_predictions_to_db.py results/predictions_lstm_56.csv \
  --station 56 --model lstm
# This will UPDATE existing LSTM predictions if any, or INSERT new ones
```

---

## 🚀 How to Use the Helper Script

### Basic Usage

```bash
python scripts/insert_predictions_to_db.py <CSV-PATH> --station <ID> --model <NAME>
```

### Examples

```bash
# SARIMA
python scripts/insert_predictions_to_db.py \
  results/my_sarima_predictions_56.csv \
  --station 56 --model sarima

# Transformer
python scripts/insert_predictions_to_db.py \
  results/transformer_station59_2025.csv \
  --station 59 --model transformer

# Custom DB host (if DB is on different machine)
python scripts/insert_predictions_to_db.py pred.csv \
  --station 56 --model sarima \
  --db-host db.example.com \
  --db-user myuser \
  --db-pass mypass
```

### Output

```
✅ Connected to localhost:5432/pm25
📖 Loading CSV: results/predictions_sarima_56.csv
   Rows to insert: 12,450
   Date range: 2025-01-01 00:00:00+00:00 → 2025-12-01 00:00:00+00:00
   Station: 56, Model: sarima

   Processing 1000/12450...
   Processing 2000/12450...
   ...

✅ Success!
   Inserted: 12,450 new rows
   Updated: 0 existing rows
   Total: 12,450 rows affected
   DB total for sarima station 56: 12,450 rows
```

---

## CSV Template for Predictions

Your model should output predictions in this format:

### Format 1: Simple (Recommended)

```csv
timestamp,predicted_pm25
2025-01-01 01:00:00+07:00,45.3
2025-01-01 02:00:00+07:00,47.1
2025-01-01 03:00:00+07:00,49.8
...
```

### Format 2: With Confidence

```csv
timestamp,predicted_pm25,lower_bound,upper_bound
2025-01-01 01:00:00+07:00,45.3,41.0,50.2
2025-01-01 02:00:00+07:00,47.1,42.1,52.5
...
```

(Optional columns like `lower_bound` are ignored; script only needs `timestamp` + `predicted_pm25`)

### Requirements

- **Datetime format:** ISO 8601 with timezone (e.g., `2025-01-01 01:00:00+07:00`)
  - Or just UTC: `2025-01-01 01:00:00+00:00` (will be converted)
  - Or naive (assumed UTC): `2025-01-01 01:00:00`
- **Numeric format:** float or int (e.g., `45.3`, `47`)
- **No missing values:** All rows must have both timestamp and predicted_pm25
- **Column order:** Doesn't matter, script reads by column name

---

## Why Not Keep CSV?

**Current state (mock CSV in `demo_data/`):**
- ✅ Good for: Testing UI before models are ready, demo narrative/storytelling
- ❌ Bad for: Real-time inference, production monitoring, accuracy verification

**Once predictions in DB:**
- ✅ Good for: Live inference, real-time monitoring, automatic retraining triggers
- ✅ Consistent: Single source of truth (PostgreSQL)
- ✅ Queryable: Can join with actual PM2.5 for metrics

---

## Airflow Integration (Advanced)

If you want fully automated pipeline (model → DB insert in Airflow DAG):

### Option 1: Python Operator

```python
from airflow.operators.python import PythonOperator
import sys
sys.path.insert(0, "/app/scripts")
from insert_predictions_to_db import insert_predictions

insert_task = PythonOperator(
    task_id="insert_sarima_predictions",
    python_callable=insert_predictions,
    op_kwargs={
        "csv_path": "/app/results/predictions_sarima_56.csv",
        "station_id": 56,
        "model_name": "sarima",
        "db_host": "postgres",  # Docker service name
        "db_port": 5432,
        "db_name": "pm25",
        "db_user": "postgres",
        "db_pass": "postgres",
    },
    dag=dag
)

# Add dependency
train_sarima_task >> insert_task
```

### Option 2: Bash Operator

```python
insert_task = BashOperator(
    task_id="insert_sarima_predictions",
    bash_command="""
    python /app/scripts/insert_predictions_to_db.py \
      /app/results/predictions_sarima_56.csv \
      --station 56 \
      --model sarima \
      --db-host postgres \
      --db-port 5432 \
      --db-name pm25 \
      --db-user postgres \
      --db-pass postgres
    """,
    dag=dag
)
```

---

## Troubleshooting

### "Connection error: could not connect to server"

**Cause:** PostgreSQL not running or unreachable  
**Fix:**
```bash
# Check if Docker container is running
docker ps | grep postgres

# Or if using local postgres
systemctl status postgresql
```

### "CSV file not found"

**Cause:** Wrong file path  
**Fix:** Check absolute path:
```bash
ls -la results/predictions_sarima_56.csv
# Adjust path if needed
```

### "Table pm25_predicted_hourly does not exist"

**Cause:** DB schema not initialized  
**Fix:** Create table:
```sql
CREATE TABLE pm25_predicted_hourly (
    id SERIAL PRIMARY KEY,
    station_id INT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    model TEXT NOT NULL,
    predicted_pm25 FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(station_id, timestamp, model)
);
```

### "Column 'timestamp' or 'predicted_pm25' not found in CSV"

**Cause:** CSV has different column names  
**Fix:** Rename columns in CSV or update script

### "Invalid timestamp format"

**Cause:** Timestamp not in ISO 8601 format  
**Fix:** Convert using pandas:
```python
df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize('UTC').dt.isoformat()
```

---

## Useful Queries

### Check what's in the DB

```sql
-- All predictions by model
SELECT model, COUNT(*) FROM pm25_predicted_hourly GROUP BY model;

-- Latest predictions for each model
SELECT DISTINCT model, MAX(timestamp) 
FROM pm25_predicted_hourly 
GROUP BY model;

-- Predictions for specific station
SELECT model, COUNT(*) 
FROM pm25_predicted_hourly 
WHERE station_id = 56 
GROUP BY model;

-- Compare actual vs predicted for one station
SELECT 
  r.timestamp,
  r.pm25 as actual,
  p.model,
  p.predicted_pm25,
  ABS(r.pm25 - p.predicted_pm25) as error
FROM pm25_raw_hourly r
LEFT JOIN pm25_predicted_hourly p 
  ON r.station_id = p.station_id 
  AND r.timestamp = p.timestamp
WHERE r.station_id = 56
  AND r.timestamp > NOW() - INTERVAL '30 days'
ORDER BY r.timestamp DESC
LIMIT 100;
```

---

## Timeline

| Date | Task | Owner |
|------|------|-------|
| Now | Demo running on mock CSV | YG |
| -3 days before presentation | SARIMA predictions in DB | Olf |
| -3 days before presentation | Transformer predictions in DB | Perm |
| Day of | Demo uses live DB predictions | Demo runs automatically |

---

## Questions?

See [docs/ADD_PREDICTIONS_TO_DB.md](ADD_PREDICTIONS_TO_DB.md) for step-by-step guide.

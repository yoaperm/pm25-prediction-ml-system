# How to Add SARIMA & Transformer Predictions to DB

> **Goal:** Train your model → Save predictions to PostgreSQL → Demo picks up automatically

---

## What the Demo Expects

The FoonAlert demo reads predictions from **PostgreSQL table `pm25_predicted_hourly`**:

```sql
-- Table structure (already exists)
CREATE TABLE pm25_predicted_hourly (
    id SERIAL,
    station_id INT,
    timestamp TIMESTAMP WITH TIME ZONE,
    model TEXT,           -- 'xgboost', 'ridge_regression', 'sarima', 'lstm', 'transformer'
    predicted_pm25 FLOAT,
    ...
);
```

Current content:
- ✅ xgboost: 34,870 rows
- ✅ ridge_regression: 8,751 rows
- ❌ sarima: 0 rows (Olf to add)
- ❌ lstm: needs update
- ❌ transformer: 0 rows (Perm to add)

**Once you insert predictions into this table → Demo automatically uses them. No code changes needed!**

---

## Step-by-Step for Olf (SARIMA)

### 1. Train SARIMA Model

Use your existing pipeline. Train on one station first (e.g., station 56).

```bash
# In Airflow DAG or local training script
# Your train code outputs: predictions_sarima_56.csv

# Example output columns:
# timestamp, predicted_pm25
# 2026-05-01 01:00:00+07, 45.3
# 2026-05-01 02:00:00+07, 47.1
# ...
```

### 2. Insert Predictions into DB

Create a simple Python script to load your CSV and insert into PostgreSQL:

```python
import pandas as pd
import psycopg2
from datetime import datetime

# Connection
conn = psycopg2.connect(
    host="localhost",
    database="pm25",
    user="postgres",
    password="postgres",
    port=5432
)
cur = conn.cursor()

# Load your predictions CSV
station_id = 56
model_name = "sarima"
predictions_df = pd.read_csv(f"results/predictions_sarima_56.csv")

# Ensure timestamp is datetime
predictions_df["timestamp"] = pd.to_datetime(predictions_df["timestamp"], utc=True)

# Insert into DB
for _, row in predictions_df.iterrows():
    cur.execute("""
        INSERT INTO pm25_predicted_hourly 
        (station_id, timestamp, model, predicted_pm25)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (station_id, timestamp, model) DO UPDATE
        SET predicted_pm25 = %s
    """, (
        station_id,
        row["timestamp"],
        model_name,
        row["predicted_pm25"],
        row["predicted_pm25"]
    ))

conn.commit()
cur.close()
conn.close()

print(f"✅ Inserted {len(predictions_df)} SARIMA predictions for station {station_id}")
```

### 3. Verify

```sql
SELECT DISTINCT model, COUNT(*) 
FROM pm25_predicted_hourly 
GROUP BY model;

-- Should show:
-- xgboost | 34870
-- ridge_regression | 8751
-- sarima | XXXX  ← your new count
```

### 4. Test Demo

Navigate to http://EC2:8502 → Live Dashboard → Station 56  
Should now show SARIMA in the model battle table ✅

---

## Step-by-Step for Perm (Transformer)

**Exactly same process as Olf, but with `model="transformer"`**

```python
# Just change:
model_name = "transformer"
predictions_df = pd.read_csv(f"results/predictions_transformer_56.csv")
```

---

## Integration with Airflow (Optional - for Production)

If you want to automate this (no manual CSV insertion):

Add a task to your training DAG that inserts directly:

```python
# In your pm25_*_training_dag.py

from airflow.operators.python import PythonOperator

def insert_predictions_to_db(station_id, model_name, predictions_csv_path, **context):
    import pandas as pd
    import psycopg2
    
    conn = psycopg2.connect(
        host="postgres",  # Docker service name
        database="pm25",
        user="postgres",
        password="postgres",
        port=5432
    )
    cur = conn.cursor()
    
    df = pd.read_csv(predictions_csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    
    for _, row in df.iterrows():
        cur.execute("""
            INSERT INTO pm25_predicted_hourly 
            (station_id, timestamp, model, predicted_pm25)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (station_id, timestamp, model) DO UPDATE
            SET predicted_pm25 = %s
        """, (
            station_id, row["timestamp"], model_name, row["predicted_pm25"], row["predicted_pm25"]
        ))
    
    conn.commit()
    cur.close()
    conn.close()

# In DAG definition:
insert_sarima_task = PythonOperator(
    task_id="insert_sarima_predictions",
    python_callable=insert_predictions_to_db,
    op_kwargs={
        "station_id": 56,
        "model_name": "sarima",
        "predictions_csv_path": "/app/results/predictions_sarima_56.csv"
    },
    dag=dag
)

# Add dependency
train_sarima_task >> insert_sarima_task
```

---

## Demo Data CSV (For Testing Before DB Integration)

Currently, demo uses mock CSV files in `demo_data/` folder.

These are **temporary** — once you insert real predictions into DB, they're not needed anymore.

- `demo_data/replay_station56_2024-04-30.csv` — mock with real + simulated predictions
- `demo_data/model_metrics.csv` — mock performance metrics

**Once SARIMA/Transformer are in DB:**
1. Remove or rename `demo_data/` folder
2. Streamlit switches to reading from `pm25_predicted_hourly` table
3. Demo uses live predictions ✅

---

## Checklist

### For Olf (SARIMA)

- [ ] Train SARIMA on stations 56-61 using hourly data
- [ ] Export predictions to CSV (columns: `timestamp`, `predicted_pm25`)
- [ ] Run insert script to add to `pm25_predicted_hourly` table
- [ ] Verify: `SELECT COUNT(*) FROM pm25_predicted_hourly WHERE model='sarima'` → should be > 0
- [ ] Test demo: reload http://EC2:8502 → check SARIMA appears in model battle

### For Perm (Transformer)

- [ ] Train Transformer on stations 56-61 using hourly data
- [ ] Export predictions to CSV
- [ ] Run insert script to add to `pm25_predicted_hourly` table
- [ ] Verify in DB
- [ ] Test demo

---

## Questions?

**Q: Can I insert predictions for multiple stations at once?**  
A: Yes! Just loop: `for station_id in [56, 57, 58, 59, 61]:`

**Q: What if I want to retrain and update old predictions?**  
A: Use `ON CONFLICT ... DO UPDATE` (already in the script above) — it replaces old values

**Q: Can I test with just 1 station first?**  
A: Yes! Train station 56, insert, test demo, then do 57-61

**Q: Why not just mock CSV forever?**  
A: Because CSV isn't real-time. PostgreSQL is the source of truth for production.

---

## Example: Full Insert Script

Save as `scripts/insert_predictions_to_db.py`:

```python
#!/usr/bin/env python3
"""Insert model predictions into PostgreSQL"""

import argparse
from pathlib import Path
import pandas as pd
import psycopg2

def insert_predictions(csv_path, station_id, model_name):
    conn = psycopg2.connect(
        host="localhost",
        database="pm25",
        user="postgres",
        password="postgres",
        port=5432
    )
    cur = conn.cursor()
    
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    
    inserted = 0
    for _, row in df.iterrows():
        cur.execute("""
            INSERT INTO pm25_predicted_hourly 
            (station_id, timestamp, model, predicted_pm25)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (station_id, timestamp, model) DO UPDATE
            SET predicted_pm25 = EXCLUDED.predicted_pm25
        """, (station_id, row["timestamp"], model_name, row["predicted_pm25"]))
        inserted += 1
    
    conn.commit()
    cur.close()
    conn.close()
    
    print(f"✅ Inserted {inserted} {model_name} predictions for station {station_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=str, help="Path to predictions CSV")
    parser.add_argument("--station", type=int, required=True)
    parser.add_argument("--model", type=str, required=True, choices=["sarima", "transformer", "lstm"])
    args = parser.parse_args()
    
    insert_predictions(args.csv, args.station, args.model)

# Usage:
# python scripts/insert_predictions_to_db.py results/predictions_sarima_56.csv --station 56 --model sarima
```

Run it:
```bash
python scripts/insert_predictions_to_db.py results/predictions_sarima_56.csv --station 56 --model sarima
```

Done! ✅

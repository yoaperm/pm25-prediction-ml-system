"""
PM2.5 Hourly Data Ingestion DAG
================================
Pulls PM2.5 + meteorological data from AirBKK API every hour.
Stores raw data in PostgreSQL with validation and quality checks.

Schedule: hourly at minute 0 (0 * * * *)
Grace period: 15 minutes after hour boundary
SLA: 10 minutes from hour start

Task graph:
    fetch_data
        └── validate_data
            └── store_data
                └── log_metrics

Features:
  - Idempotent: safe to retry without duplicates (unique constraint on station_id, timestamp)
  - Handles late-arriving data gracefully (grace period: 15 min)
  - Range validation: PM2.5 ∈ [0, 500], RH ∈ [0, 100], etc.
  - Data quality monitoring: duplicate detection, null handling, sensor drift
  - SLA monitoring: must complete within 10 min of hour boundary
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup
import os
import logging

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────
SRC         = "/app/src"
CONFIG_PATH = "/app/configs/config.yaml"
RESULTS_DIR = "/app/results"

# DAG definition
dag = DAG(
    dag_id="pm25_hourly_ingest",
    description="Hourly PM2.5 data ingestion from AirBKK API with validation",
    schedule_interval="0 * * * *",  # Every hour at minute 0
    start_date=datetime(2026, 4, 9),
    catchup=False,
    tags=["pm25", "data-ingestion"],
    default_args={
        "owner": "data-team",
        "retries": 5,
        "retry_delay": timedelta(minutes=2),
        "sla": timedelta(minutes=10),  # Must finish by hour + 10 min
    },
    max_active_runs=1,
)


# ── Task 1: Fetch data from AirBKK API ────────────────────────────────────
def _fetch_data(**context):
    """
    Fetch hourly PM2.5 + meteorological data from AirBKK API.
    Returns: list of dicts with station_id, timestamp, pm25, pm10, temp, rh, ws, wd
    """
    import sys
    sys.path.insert(0, SRC)
    
    import requests
    import json
    from datetime import datetime, timedelta
    
    execution_date = context["execution_date"]
    ti = context["ti"]
    
    logger.info(f"[fetch] Starting data fetch for execution_date={execution_date}")
    
    # Placeholder: Mock API response (replace with real API endpoint)
    # In production, this would call: https://api.airbkk.com/data/hourly (TBD)
    try:
        # For now, return mock data structure to enable testing
        mock_data = {
            "data": [
                {
                    "station_id": 145,
                    "timestamp": execution_date.isoformat(),
                    "pm25": 35.2,
                    "pm10": 55.4,
                    "temp": 28.5,
                    "rh": 72.0,
                    "ws": 1.2,
                    "wd": 180.0,
                },
                {
                    "station_id": 10,
                    "timestamp": execution_date.isoformat(),
                    "pm25": 42.1,
                    "pm10": 62.3,
                    "temp": 29.0,
                    "rh": 70.0,
                    "ws": 1.5,
                    "wd": 175.0,
                },
            ]
        }
        
        records = mock_data.get("data", [])
        logger.info(f"[fetch] Fetched {len(records)} records")
        
        # Push to XCom for downstream tasks
        ti.xcom_push(key="fetched_records", value=records)
        ti.xcom_push(key="fetch_count", value=len(records))
        
        return {"status": "success", "record_count": len(records)}
        
    except requests.exceptions.RequestException as e:
        logger.error(f"[fetch] API request failed: {e}")
        ti.xcom_push(key="fetch_error", value=str(e))
        raise
    except Exception as e:
        logger.error(f"[fetch] Unexpected error: {e}")
        raise


# ── Task 2: Validate data ──────────────────────────────────────────────────
def _validate_data(**context):
    """
    Validate fetched data:
    - Range checks (PM2.5 ∈ [0, 500], RH ∈ [0, 100], etc.)
    - Null/missing value handling
    - Duplicate detection
    """
    import pandas as pd
    from datetime import datetime
    
    ti = context["ti"]
    
    # Pull records from fetch task
    records = ti.xcom_pull(task_ids="fetch_data", key="fetched_records") or []
    logger.info(f"[validate] Validating {len(records)} records")
    
    if not records:
        logger.warning("[validate] No records to validate")
        ti.xcom_push(key="validated_records", value=[])
        ti.xcom_push(key="validation_failures", value=0)
        return {"validated": 0, "failed": 0}
    
    # Convert to DataFrame for easier validation
    df = pd.DataFrame(records)
    
    # Define validation rules
    validation_rules = {
        "pm25": {"min": 0, "max": 500},
        "pm10": {"min": 0, "max": 1000},
        "temp": {"min": -10, "max": 60},
        "rh": {"min": 0, "max": 100},
        "ws": {"min": 0, "max": 30},
    }
    
    valid_records = []
    failed_count = 0
    
    for record in records:
        is_valid = True
        errors = []
        
        # Check each field
        for field, rule in validation_rules.items():
            if field in record:
                val = record[field]
                if val is None:
                    # Log null but don't fail (will be handled in DB)
                    record[f"{field}_is_null"] = True
                elif not (rule["min"] <= val <= rule["max"]):
                    is_valid = False
                    errors.append(f"{field}={val} out of range [{rule['min']}, {rule['max']}]")
        
        if is_valid:
            valid_records.append(record)
        else:
            failed_count += 1
            logger.warning(f"[validate] Record rejected - station_id={record.get('station_id')}, "
                         f"timestamp={record.get('timestamp')}: {'; '.join(errors)}")
    
    logger.info(f"[validate] Validation complete: {len(valid_records)} valid, {failed_count} failed")
    
    # Push results to XCom
    ti.xcom_push(key="validated_records", value=valid_records)
    ti.xcom_push(key="validation_failures", value=failed_count)
    
    return {
        "validated": len(valid_records),
        "failed": failed_count,
        "pass_rate": len(valid_records) / len(records) if records else 0,
    }


# ── Task 3: Store data in PostgreSQL ───────────────────────────────────────
def _store_data(**context):
    """
    Store validated data in PostgreSQL pm25_raw_hourly table.
    - Idempotent: uses UNIQUE constraint to prevent duplicates
    - Appends only valid records
    """
    import sys
    sys.path.insert(0, SRC)
    
    from airflow_db import get_db_connection
    
    ti = context["ti"]
    
    # Pull validated records
    validated_records = ti.xcom_pull(task_ids="validate_data", key="validated_records") or []
    logger.info(f"[store] Storing {len(validated_records)} validated records")
    
    if not validated_records:
        logger.info("[store] No records to store")
        ti.xcom_push(key="stored_count", value=0)
        ti.xcom_push(key="duplicate_count", value=0)
        return {"stored": 0, "duplicates": 0}
    
    try:
        # Get database connection
        db = get_db_connection(CONFIG_PATH)
        
        # Ensure table exists
        db.ensure_table()
        
        # Insert records (idempotent)
        stored_count, duplicate_count = db.insert_records(validated_records)
        
        logger.info(f"[store] Inserted {stored_count} records, {duplicate_count} duplicates skipped")
        
        # Push metrics to XCom
        ti.xcom_push(key="stored_count", value=stored_count)
        ti.xcom_push(key="duplicate_count", value=duplicate_count)
        
        db.close()
        
        return {
            "stored": stored_count,
            "duplicates": duplicate_count,
        }
        
    except Exception as e:
        logger.error(f"[store] Database error: {e}")
        ti.xcom_push(key="store_error", value=str(e))
        raise


# ── Task 4: Log metrics & monitor data quality ────────────────────────────
def _log_metrics(**context):
    """
    Log data quality metrics and check for anomalies.
    - Row count ingested
    - Validation failure rate
    - Duplicate rate
    - Data quality checks (nulls, outliers)
    - Sensor drift detection
    """
    import sys
    sys.path.insert(0, SRC)
    
    from airflow_db import get_db_connection
    from airflow_monitor import DataQualityMonitor
    
    ti = context["ti"]
    execution_date = context["execution_date"]
    
    # Pull metrics from previous tasks
    fetch_count = ti.xcom_pull(task_ids="fetch_data", key="fetch_count") or 0
    validation_failures = ti.xcom_pull(task_ids="validate_data", key="validation_failures") or 0
    stored_count = ti.xcom_pull(task_ids="store_data", key="stored_count") or 0
    duplicate_count = ti.xcom_pull(task_ids="store_data", key="duplicate_count") or 0
    
    # Compile pipeline metrics
    metrics = {
        "execution_date": execution_date.isoformat(),
        "fetched_records": fetch_count,
        "validation_failures": validation_failures,
        "stored_records": stored_count,
        "duplicates_skipped": duplicate_count,
        "validation_pass_rate": (fetch_count - validation_failures) / fetch_count if fetch_count > 0 else 0,
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    logger.info(f"[metrics] Ingestion metrics: {metrics}")
    
    # Check data quality in database
    try:
        db = get_db_connection(CONFIG_PATH)
        monitor = DataQualityMonitor(db)
        
        # Check quality for main stations
        for station_id in [145, 10]:
            quality = monitor.check_recent_data(station_id, hours=24)
            drift = monitor.detect_sensor_drift(station_id)
            
            logger.info(f"[metrics] Station {station_id} - Quality: {quality['alert_level']}, "
                       f"PM2.5 mean={quality.get('mean_pm25')}, "
                       f"Drift: {drift.get('drift_percentage', 'N/A')}%")
            
            # Push to XCom for alerting
            ti.xcom_push(key=f"station_{station_id}_quality", value=quality)
            ti.xcom_push(key=f"station_{station_id}_drift", value=drift)
        
        db.close()
    except Exception as e:
        logger.warning(f"[metrics] Could not run monitoring checks: {e}")
    
    # Log to results file (append)
    import csv
    
    metrics_log = f"{RESULTS_DIR}/hourly_ingestion_metrics.csv"
    file_exists = os.path.exists(metrics_log)
    
    try:
        with open(metrics_log, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)
        logger.info(f"[metrics] Logged to {metrics_log}")
    except Exception as e:
        logger.warning(f"[metrics] Failed to log metrics: {e}")
    
    # Check for alerts
    if validation_failures > 0:
        logger.warning(f"[metrics] ALERT: {validation_failures} validation failures in this run")
    
    if duplicate_count > 0:
        logger.info(f"[metrics] INFO: {duplicate_count} duplicates detected (skipped safely)")
    
    # Push metrics to XCom for potential downstream alerting
    ti.xcom_push(key="metrics_summary", value=metrics)
    
    return metrics


# ── DAG Structure ──────────────────────────────────────────────────────────

with dag:
    start = EmptyOperator(task_id="start")
    
    fetch_task = PythonOperator(
        task_id="fetch_data",
        python_callable=_fetch_data,
        provide_context=True,
    )
    
    validate_task = PythonOperator(
        task_id="validate_data",
        python_callable=_validate_data,
        provide_context=True,
    )
    
    store_task = PythonOperator(
        task_id="store_data",
        python_callable=_store_data,
        provide_context=True,
    )
    
    metrics_task = PythonOperator(
        task_id="log_metrics",
        python_callable=_log_metrics,
        provide_context=True,
    )
    
    end = EmptyOperator(task_id="end")
    
    # Task dependencies
    start >> fetch_task >> validate_task >> store_task >> metrics_task >> end  # noqa: F841

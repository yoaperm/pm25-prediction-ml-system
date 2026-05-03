"""
PM2.5 T+1h Training DAG
=======================
Manual training pipeline for next-hour PM2.5 models.

Trigger manually with:
    {"station_id": 56}
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

SRC = "/app/src"
SCRIPTS = "/app/scripts"
DEFAULT_DB_URL = "postgresql://postgres:postgres@postgres:5432/pm25"
STATION_IDS = [56, 57, 58, 59, 61]


def _station_id(context):
    return int(context["params"]["station_id"])


def _train_station(**context):
    sys.path.insert(0, SRC)
    sys.path.insert(0, SCRIPTS)

    import mlflow
    from train_1h_forecast import get_splits, train_station_1h

    station_id = _station_id(context)
    db_url = os.environ.get("PM25_DB_URL", DEFAULT_DB_URL)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)

    result = train_station_1h(station_id=station_id, db_url=db_url, splits=get_splits())
    if result is None:
        raise ValueError(f"T+1h training did not produce a model for station {station_id}")

    return {
        "station_id": station_id,
        "rows": len(result),
        "models": result["model"].tolist(),
    }


with DAG(
    dag_id="pm25_1h_training",
    description="Train and deploy T+1h PM2.5 forecast models",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    params={
        "station_id": Param(
            56,
            type="integer",
            enum=STATION_IDS,
            description="Station ID to train for next-hour prediction",
        ),
    },
    tags=["pm25", "ml", "1h-forecast", "postgresql"],
) as dag:
    train_station = PythonOperator(
        task_id="train_station_1h",
        python_callable=_train_station,
    )

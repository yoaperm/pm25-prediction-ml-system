"""
PM2.5 T+1h Training DAG
=======================
Daily training pipeline for next-hour PM2.5 models.
Each station task runs the full T+1h flow from scripts/train_1h_forecast.py:
feature engineering, regression/LSTM training, evaluation, compare/deploy,
and Triton publish.

Scheduled runs train all configured stations. Trigger manually with:
    {"station_id": 56}
"""

from __future__ import annotations

import os
import sys

import pendulum
from airflow import DAG
from airflow.exceptions import AirflowSkipException
from airflow.models.param import Param
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule

SRC = "/app/src"
SCRIPTS = "/app/scripts"
DEFAULT_DB_URL = "postgresql://postgres:postgres@postgres:5432/pm25"
BANGKOK_TIMEZONE_NAME = "Asia/Bangkok"
STATION_IDS = [56, 57, 58, 59, 61]


def _manual_station_id(context):
    dag_run = context.get("dag_run")
    if not dag_run:
        return None

    run_type = getattr(dag_run.run_type, "value", dag_run.run_type)
    if str(run_type).lower() != "manual":
        return None

    conf = dag_run.conf or {}
    station_id = conf.get("station_id", context["params"].get("station_id"))
    if station_id is None:
        return None

    station_id = int(station_id)
    if station_id not in STATION_IDS:
        raise ValueError(f"Unsupported station_id={station_id}; expected one of {STATION_IDS}")
    return station_id


def _train_station(station_id: int, **context):
    sys.path.insert(0, SRC)
    sys.path.insert(0, SCRIPTS)

    import mlflow
    from train_1h_forecast import get_splits, train_station_1h

    selected_station_id = _manual_station_id(context)
    if selected_station_id is not None and selected_station_id != station_id:
        raise AirflowSkipException(
            f"Manual run selected station {selected_station_id}; skipping station {station_id}"
        )

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
    schedule="0 0 * * *",
    start_date=pendulum.datetime(2024, 1, 1, tz=BANGKOK_TIMEZONE_NAME),
    catchup=False,
    max_active_runs=1,
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
    start = EmptyOperator(task_id="start")
    done = EmptyOperator(task_id="done", trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)

    train_tasks = [
        PythonOperator(
            task_id=f"train_station_{station_id}_1h",
            python_callable=_train_station,
            op_kwargs={"station_id": station_id},
        )
        for station_id in STATION_IDS
    ]

    start >> train_tasks >> done

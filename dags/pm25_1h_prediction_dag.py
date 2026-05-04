"""
PM2.5 24-Hour Hourly Prediction DAG
===================================
Daily DAG that loads the active T+1h ONNX model recursively to generate
PM2.5 predictions for the next 24 hours, then stores the 24 forecast rows
per station in PostgreSQL table pm25_predicted_hourly.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import logging
import os
import sys

import pendulum
from airflow import DAG
from airflow.models.param import Param
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

SRC = "/app/src"
BANGKOK_TIMEZONE_NAME = "Asia/Bangkok"
BANGKOK_TZ = pendulum.timezone(BANGKOK_TIMEZONE_NAME)
MODELS_BASE = os.environ.get("MODELS_DIR", "/app/models")
DEFAULT_DB_URL = "postgresql://postgres:postgres@postgres:5432/pm25"
DEFAULT_PREDICTION_TABLE = "pm25_predicted_hourly"
STATION_IDS = (56, 57, 58, 59, 61)
HISTORY_HOURS = 72
MIN_HISTORY_HOURS = 25
FORECAST_HOURS = 24
BACKFILL_DAYS = 365
BATCH_SIZE = 1000


def _load_latest_hourly_for_station(station_id: int, db_url: str):
    import pandas as pd
    import sqlalchemy

    engine = sqlalchemy.create_engine(db_url)
    query = sqlalchemy.text("""
        SELECT (timestamp AT TIME ZONE 'Asia/Bangkok') AS datetime, pm25
        FROM pm25_raw_hourly
        WHERE station_id = :station_id
          AND pm25 IS NOT NULL
        ORDER BY timestamp DESC
        LIMIT :limit
    """)
    try:
        with engine.connect() as conn:
            result = conn.execute(query, {"station_id": station_id, "limit": HISTORY_HOURS})
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
    finally:
        engine.dispose()

    if df.empty:
        return df

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    full_range = pd.date_range(df["datetime"].min(), df["datetime"].max(), freq="h")
    df = df.set_index("datetime").reindex(full_range).rename_axis("datetime").reset_index()
    df["pm25_observed"] = df["pm25"]
    df["pm25"] = df["pm25"].ffill().bfill().clip(lower=0, upper=500)
    return df


def _load_hourly_until_for_station(station_id: int, db_url: str, end_timestamp):
    import pandas as pd
    import sqlalchemy

    engine = sqlalchemy.create_engine(db_url)
    query = sqlalchemy.text("""
        SELECT (timestamp AT TIME ZONE 'Asia/Bangkok') AS datetime, pm25
        FROM pm25_raw_hourly
        WHERE station_id = :station_id
          AND (timestamp AT TIME ZONE 'Asia/Bangkok') <= :end_timestamp
          AND pm25 IS NOT NULL
        ORDER BY timestamp DESC
        LIMIT :limit
    """)
    try:
        with engine.connect() as conn:
            result = conn.execute(
                query,
                {
                    "station_id": station_id,
                    "end_timestamp": end_timestamp,
                    "limit": HISTORY_HOURS,
                },
            )
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
    finally:
        engine.dispose()

    if df.empty:
        return df

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    full_range = pd.date_range(df["datetime"].min(), df["datetime"].max(), freq="h")
    df = df.set_index("datetime").reindex(full_range).rename_axis("datetime").reset_index()
    df["pm25_observed"] = df["pm25"]
    df["pm25"] = df["pm25"].ffill().bfill().clip(lower=0, upper=500)
    return df


def _load_hourly_bounds_for_station(station_id: int, db_url: str):
    import pandas as pd
    import sqlalchemy

    engine = sqlalchemy.create_engine(db_url)
    query = sqlalchemy.text("""
        SELECT
            MIN(timestamp AT TIME ZONE 'Asia/Bangkok') AS min_ts,
            MAX(timestamp AT TIME ZONE 'Asia/Bangkok') AS max_ts
        FROM pm25_raw_hourly
        WHERE station_id = :station_id
          AND pm25 IS NOT NULL
    """)
    try:
        with engine.connect() as conn:
            row = conn.execute(query, {"station_id": station_id}).mappings().first()
    finally:
        engine.dispose()

    if not row or row["min_ts"] is None or row["max_ts"] is None:
        return None, None
    return pd.Timestamp(row["min_ts"]), pd.Timestamp(row["max_ts"])


def _prediction_table_has_rows(db_url: str) -> bool:
    import sqlalchemy

    engine = sqlalchemy.create_engine(db_url)
    try:
        inspector = sqlalchemy.inspect(engine)
        if not inspector.has_table(DEFAULT_PREDICTION_TABLE):
            return False
        query = sqlalchemy.text("SELECT EXISTS (SELECT 1 FROM pm25_predicted_hourly LIMIT 1)")
        with engine.connect() as conn:
            return bool(conn.execute(query).scalar())
    finally:
        engine.dispose()


def _daily_backfill_anchors(min_ts, max_ts, backfill_days: int = BACKFILL_DAYS):
    import pandas as pd

    first_possible = min_ts + pd.Timedelta(hours=MIN_HISTORY_HOURS - 1)
    last_anchor = max_ts.floor("D")
    lookback_anchor = last_anchor - pd.Timedelta(days=backfill_days)
    first_anchor = max(first_possible.ceil("D"), lookback_anchor)
    if first_anchor > last_anchor:
        return []
    return [ts.to_pydatetime() for ts in pd.date_range(first_anchor, last_anchor, freq="D")]


def _get_model_info(station_id: int) -> dict:
    models_dir = os.path.join(MODELS_BASE, f"station_{station_id}_1h")
    registry = os.path.join(models_dir, "active_model.json")
    feature_cols_path = os.path.join(models_dir, "feature_columns.json")

    if not os.path.exists(registry):
        raise FileNotFoundError(f"active_model.json not found: {registry}")
    if not os.path.exists(feature_cols_path):
        raise FileNotFoundError(f"feature_columns.json not found: {feature_cols_path}")

    with open(registry) as f:
        info = json.load(f)
    with open(feature_cols_path) as f:
        info["feature_cols"] = json.load(f)
    info["onnx_path"] = os.path.join(models_dir, "onnx", info["onnx_file"])
    if not os.path.exists(info["onnx_path"]):
        raise FileNotFoundError(f"active ONNX model not found: {info['onnx_path']}")
    return info


def _prepare_prediction_jobs(**context):
    db_url = os.environ.get("PM25_DB_URL", DEFAULT_DB_URL)
    jobs = []
    skipped = []
    force_backfill = bool(context["params"].get("force_backfill", False))
    has_prediction_rows = _prediction_table_has_rows(db_url)
    should_backfill = force_backfill or not has_prediction_rows

    for station_id in STATION_IDS:
        try:
            info = _get_model_info(station_id)

            if should_backfill:
                min_ts, max_ts = _load_hourly_bounds_for_station(station_id, db_url)
                if min_ts is None:
                    skipped.append({"station_id": station_id, "reason": "no_hourly_data"})
                    continue
                anchors = _daily_backfill_anchors(min_ts, max_ts)
                if not anchors:
                    skipped.append({"station_id": station_id, "reason": "insufficient_backfill_range"})
                    continue
                run_type = "manual_backfill" if force_backfill and has_prediction_rows else "initial_backfill"
            else:
                hourly_df = _load_latest_hourly_for_station(station_id, db_url)
                if hourly_df.empty or len(hourly_df) < MIN_HISTORY_HOURS:
                    skipped.append({"station_id": station_id, "reason": "insufficient_history"})
                    continue
                anchors = [hourly_df["datetime"].iloc[-1].to_pydatetime()]
                run_type = "scheduled"

            for anchor_ts in anchors:
                jobs.append({
                    "source_station_id": station_id,
                    "history_end_timestamp": anchor_ts.isoformat(),
                    "feature_cols": info["feature_cols"],
                    "input_shape": info.get("input_shape", "2d"),
                    "input_name": info.get("input_name", "lstm_input" if info.get("input_shape") == "3d" else "float_input"),
                    "output_name": info.get("output_name", "output" if info.get("input_shape") == "3d" else "variable"),
                    "model_name": info.get("model_key", f"pm25_{station_id}_1h"),
                    "onnx_path": info["onnx_path"],
                    "run_type": run_type,
                })
        except Exception as exc:
            logger.warning("[prepare] station %s skipped: %s", station_id, exc)
            skipped.append({"station_id": station_id, "reason": str(exc)})

    if not jobs:
        raise ValueError(f"No 24-hour hourly prediction jobs prepared; skipped={skipped}")

    ti = context["ti"]
    ti.xcom_push(key="prediction_jobs", value=jobs)
    ti.xcom_push(key="skipped_stations", value=skipped)
    ti.xcom_push(key="should_backfill", value=should_backfill)
    ti.xcom_push(key="force_backfill", value=force_backfill)
    logger.info(
        "[prepare] Prepared %s 24-hour hourly jobs; backfill=%s force_backfill=%s skipped=%s",
        len(jobs),
        should_backfill,
        force_backfill,
        len(skipped),
    )
    return {
        "prediction_jobs": len(jobs),
        "backfill": should_backfill,
        "force_backfill": force_backfill,
        "backfill_days": BACKFILL_DAYS if should_backfill else None,
        "skipped_stations": skipped,
    }


def _load_onnx_session(job: dict):
    import onnxruntime as rt

    session = rt.InferenceSession(job["onnx_path"], providers=["CPUExecutionProvider"])
    return {
        "session": session,
        "input_name": session.get_inputs()[0].name,
        "output_name": session.get_outputs()[0].name,
    }


def _infer_onnx(job: dict, features: list[float], model_runtime: dict) -> float:
    import numpy as np

    feature_count = len(job["feature_cols"])
    shape = (1, 1, feature_count) if job["input_shape"] == "3d" else (1, feature_count)
    inputs = np.asarray(features, dtype=np.float32).reshape(shape)
    outputs = model_runtime["session"].run(
        [model_runtime["output_name"]],
        {model_runtime["input_name"]: inputs},
    )
    return round(float(outputs[0].flatten()[0]), 2)


def _run_and_store_predictions(**context):
    sys.path.insert(0, SRC)
    import pandas as pd
    from airflow_db import get_db_connection
    from hourly_forecast import build_next_hour_inference_features

    ti = context["ti"]
    jobs = ti.xcom_pull(task_ids="prepare_prediction_jobs", key="prediction_jobs") or []
    generated_at = datetime.now(timezone.utc).isoformat()
    task_run_at = context["logical_date"].isoformat()
    db_url = os.environ.get("PM25_DB_URL", DEFAULT_DB_URL)
    records = []
    runtime_cache = {}

    for job in jobs:
        history_df = _load_hourly_until_for_station(
            station_id=job["source_station_id"],
            db_url=db_url,
            end_timestamp=job["history_end_timestamp"],
        )
        if history_df.empty or len(history_df) < MIN_HISTORY_HOURS:
            logger.warning(
                "[history] station=%s anchor=%s insufficient history",
                job["source_station_id"],
                job["history_end_timestamp"],
            )
            continue

        history_start_timestamp = history_df["datetime"].iloc[0].isoformat()
        history_end_timestamp = history_df["datetime"].iloc[-1].isoformat()
        history_hours = int(len(history_df))
        filled_history_hours = int(history_df["pm25_observed"].isna().sum()) if "pm25_observed" in history_df else 0

        history_df = history_df[["datetime", "pm25"]].copy()
        model_runtime = runtime_cache.setdefault(job["onnx_path"], _load_onnx_session(job))
        for horizon_hour in range(1, FORECAST_HOURS + 1):
            try:
                feat_df = build_next_hour_inference_features(history_df, job["feature_cols"])
                if feat_df.empty:
                    logger.error("[features] %s horizon=%s produced no feature row", job["model_name"], horizon_hour)
                    break

                prediction = _infer_onnx(
                    job=job,
                    features=feat_df.iloc[0].astype(float).tolist(),
                    model_runtime=model_runtime,
                )
            except Exception as exc:
                logger.error("[onnx] %s horizon=%s failed: %s", job["model_name"], horizon_hour, exc)
                break

            prediction_ts = history_df["datetime"].iloc[-1] + timedelta(hours=1)
            records.append({
                "prediction_timestamp": prediction_ts.isoformat(),
                "predicted_pm25": prediction,
                "unit": "µg/m³",
                "model": job["model_name"],
                "source_station_id": job["source_station_id"],
                "history_hours": history_hours,
                "history_start_timestamp": history_start_timestamp,
                "history_end_timestamp": history_end_timestamp,
                "filled_history_hours": filled_history_hours,
                "prediction_generated_at": generated_at,
                "run_type": job["run_type"],
                "dag_id": context["dag"].dag_id,
                "dag_run_id": context["run_id"],
                "task_run_at": task_run_at,
            })

            history_df = pd.concat(
                [
                    history_df,
                    pd.DataFrame([{"datetime": prediction_ts, "pm25": prediction}]),
                ],
                ignore_index=True,
            )

    if not records:
        raise ValueError("[onnx] No 24-hour hourly predictions generated")

    prediction_df = pd.DataFrame(records)
    stored_count = 0
    db = get_db_connection()
    try:
        db.ensure_hourly_prediction_table()
        for start_idx in range(0, len(prediction_df), BATCH_SIZE):
            chunk = prediction_df.iloc[start_idx : start_idx + BATCH_SIZE].to_dict("records")
            stored_count += db.insert_hourly_prediction_records(records=chunk)
    finally:
        db.close()

    ti.xcom_push(key="prediction_rows", value=len(records))
    ti.xcom_push(key="stored_rows", value=stored_count)
    logger.info("[onnx] Generated %s hourly predictions", len(records))
    logger.info("[store] Stored %s rows in %s", stored_count, DEFAULT_PREDICTION_TABLE)
    return {
        "prediction_rows": len(records),
        "stored": stored_count,
        "table_name": DEFAULT_PREDICTION_TABLE,
    }


with DAG(
    dag_id="pm25_24h_hourly_prediction",
    description="Generate hourly PM2.5 predictions for the next 24 hours via local ONNX",
    schedule="0 2 * * *",
    start_date=pendulum.datetime(2026, 4, 15, tz=BANGKOK_TIMEZONE_NAME),
    catchup=False,
    max_active_runs=1,
    params={
        "force_backfill": Param(
            False,
            type="boolean",
            description="When true, run the 365-day hourly prediction backfill even if pm25_predicted_hourly already has rows.",
        ),
    },
    tags=["pm25", "prediction", "24h-hourly-forecast", "postgresql"],
    default_args={
        "owner": "data-team",
        "retries": 2,
        "retry_delay": timedelta(minutes=2),
    },
) as dag:
    start = EmptyOperator(task_id="start")
    prepare_prediction_jobs = PythonOperator(
        task_id="prepare_prediction_jobs",
        python_callable=_prepare_prediction_jobs,
    )
    run_and_store_predictions = PythonOperator(
        task_id="run_and_store_predictions",
        python_callable=_run_and_store_predictions,
    )
    done = EmptyOperator(task_id="done")

    start >> prepare_prediction_jobs >> run_and_store_predictions >> done

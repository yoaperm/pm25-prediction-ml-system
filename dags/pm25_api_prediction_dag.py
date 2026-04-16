"""
Hourly DAG that calls the FastAPI /predict endpoint and stores results in PostgreSQL.

Manual trigger example:
{
  "start_date": "2026-04-01",
  "end_date": "2026-04-07"
}
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
import logging
import os
from urllib import error, request

import pendulum
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

SRC = "/app/src"
BANGKOK_TIMEZONE_NAME = "Asia/Bangkok"
BANGKOK_TZ = pendulum.timezone(BANGKOK_TIMEZONE_NAME)
DEFAULT_API_URL = os.environ.get("PM25_API_URL", "http://api:8000")
DEFAULT_API_KEY = os.environ.get("API_KEY", "foonalert-secret-key")
DEFAULT_PREDICTION_TABLE = os.environ.get(
    "PM25_PREDICTIONS_TABLE",
    "pm25_api_daily_predictions",
)
STATION_IDS = (56, 57, 58, 59, 61)
DEFAULT_HISTORY_DAYS = 15
FILL_LOOKBACK_DAYS = 7
SCHEDULE_BUFFER_DAYS = 30


def _parse_date_value(value) -> date:
    if isinstance(value, datetime):
        return value.astimezone(BANGKOK_TZ).date() if value.tzinfo else value.date()

    if isinstance(value, date):
        return value

    if value is None:
        raise ValueError("Date value is required")

    text = str(value).strip()
    if not text:
        raise ValueError("Date value must not be empty")

    for fmt in (
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue

    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError as exc:
        raise ValueError(
            "Unsupported date format. Use YYYY-MM-DD, DD/MM/YYYY, or ISO datetime."
        ) from exc


def _get_runtime_value(context, key: str, default=None):
    dag_run = context.get("dag_run")
    dag_run_conf = dag_run.conf if dag_run and dag_run.conf else {}
    params = context.get("params", {})
    return dag_run_conf.get(key, params.get(key, default))


def _resolve_run_mode(context) -> dict:
    raw_start_date = _get_runtime_value(context, "start_date")
    raw_end_date = _get_runtime_value(context, "end_date")
    has_start = raw_start_date not in (None, "")
    has_end = raw_end_date not in (None, "")

    if has_start != has_end:
        raise ValueError("start_date and end_date must be provided together")

    if has_start:
        start_date = _parse_date_value(raw_start_date)
        end_date = _parse_date_value(raw_end_date)
        if start_date > end_date:
            raise ValueError("start_date must be less than or equal to end_date")
        return {
            "run_type": "manual",
            "start_date": start_date,
            "end_date": end_date,
        }

    return {
        "run_type": "scheduled",
        "start_date": None,
        "end_date": None,
    }


def _load_station_daily_df(station_id: int, query_start: date, query_end: date):
    import pandas as pd
    import sqlalchemy

    engine = sqlalchemy.create_engine(
        os.environ.get("PM25_DB_URL", "postgresql://postgres:postgres@postgres:5432/pm25")
    )
    query = """
        SELECT
            DATE(timestamp AT TIME ZONE 'Asia/Bangkok') AS reading_date,
            ROUND(AVG(pm25)::numeric, 2) AS pm25,
            COUNT(pm25) AS hourly_points
        FROM pm25_raw_hourly
        WHERE station_id = :station_id
          AND DATE(timestamp AT TIME ZONE 'Asia/Bangkok') BETWEEN :start_date AND :end_date
          AND pm25 IS NOT NULL
        GROUP BY 1
        ORDER BY 1
    """

    try:
        with engine.connect() as conn:
            result = conn.execute(
                sqlalchemy.text(query),
                {
                    "station_id": station_id,
                    "start_date": query_start,
                    "end_date": query_end,
                },
            )
            daily_df = pd.DataFrame(result.fetchall(), columns=result.keys())
    finally:
        engine.dispose()

    if daily_df.empty:
        return daily_df

    daily_df["reading_date"] = pd.to_datetime(daily_df["reading_date"]).dt.date
    calendar_df = pd.DataFrame(
        {"reading_date": pd.date_range(query_start, query_end, freq="D").date}
    )
    merged_df = calendar_df.merge(daily_df, on="reading_date", how="left")
    merged_df["pm25_observed"] = merged_df["pm25"]
    merged_df["pm25"] = merged_df["pm25"].ffill()
    merged_df["was_filled"] = merged_df["pm25_observed"].isna() & merged_df["pm25"].notna()
    return merged_df


def _prepare_prediction_jobs(**context):
    runtime = _resolve_run_mode(context)
    history_days = DEFAULT_HISTORY_DAYS
    today_bangkok = pendulum.now(BANGKOK_TIMEZONE_NAME).date()

    if runtime["run_type"] == "manual":
        requested_start = runtime["start_date"]
        requested_end = runtime["end_date"]
        query_start = requested_start - timedelta(days=history_days + FILL_LOOKBACK_DAYS)
        query_end = requested_end - timedelta(days=1)
    else:
        requested_start = None
        requested_end = None
        query_start = today_bangkok - timedelta(
            days=history_days + FILL_LOOKBACK_DAYS + SCHEDULE_BUFFER_DAYS
        )
        query_end = today_bangkok

    jobs = []
    skipped_stations = []
    station_target_dates = {}

    for station_id in STATION_IDS:
        merged_df = _load_station_daily_df(
            station_id=station_id,
            query_start=query_start,
            query_end=query_end,
        )
        if merged_df.empty:
            skipped_stations.append(
                {
                    "station_id": station_id,
                    "reason": "no_daily_data",
                }
            )
            continue

        observed_dates = merged_df.loc[merged_df["pm25_observed"].notna(), "reading_date"]
        if observed_dates.empty:
            skipped_stations.append(
                {
                    "station_id": station_id,
                    "reason": "no_observed_rows",
                }
            )
            continue

        latest_available_date = max(observed_dates)

        if runtime["run_type"] == "scheduled":
            station_target_start = latest_available_date + timedelta(days=1)
            station_target_end = station_target_start
        else:
            station_target_start = requested_start
            station_target_end = requested_end
            latest_predictable_date = latest_available_date + timedelta(days=1)
            if station_target_start > latest_predictable_date:
                skipped_stations.append(
                    {
                        "station_id": station_id,
                        "reason": "requested_start_beyond_latest_history",
                        "latest_predictable_date": latest_predictable_date.isoformat(),
                    }
                )
                continue
            station_target_end = min(station_target_end, latest_predictable_date)

        target_dates = []
        current_date = station_target_start
        while current_date <= station_target_end:
            target_dates.append(current_date)
            current_date += timedelta(days=1)

        if not target_dates:
            skipped_stations.append(
                {
                    "station_id": station_id,
                    "reason": "no_target_dates",
                }
            )
            continue

        station_target_dates[station_id] = [d.isoformat() for d in target_dates]

        station_jobs = 0
        for target_date in target_dates:
            history_start = target_date - timedelta(days=history_days)
            history_end = target_date - timedelta(days=1)
            history_window = merged_df[
                (merged_df["reading_date"] >= history_start)
                & (merged_df["reading_date"] <= history_end)
            ].copy()

            if len(history_window) != history_days:
                skipped_stations.append(
                    {
                        "station_id": station_id,
                        "target_date": target_date.isoformat(),
                        "reason": "insufficient_history_window",
                        "expected_history_days": history_days,
                        "actual_history_days": int(len(history_window)),
                    }
                )
                continue

            if history_window["pm25"].isna().any():
                skipped_stations.append(
                    {
                        "station_id": station_id,
                        "target_date": target_date.isoformat(),
                        "reason": "history_contains_missing_pm25",
                    }
                )
                continue

            jobs.append(
                {
                    "target_date": target_date.isoformat(),
                    "source_station_id": station_id,
                    "history_days": history_days,
                    "history_start_date": history_start.isoformat(),
                    "history_end_date": history_end.isoformat(),
                    "filled_history_days": int(history_window["was_filled"].sum()),
                    "history": [
                        {
                            "date": row.reading_date.isoformat(),
                            "pm25": float(row.pm25),
                        }
                        for row in history_window.itertuples(index=False)
                    ],
                }
            )
            station_jobs += 1

        if station_jobs == 0:
            logger.warning(
                "[prepare] Station %s produced no valid prediction jobs in requested range",
                station_id,
            )

    if not jobs:
        raise ValueError(
            "No prediction jobs could be prepared for the requested range and station set"
        )

    prediction_dates = sorted({job["target_date"] for job in jobs})
    prepared_station_ids = sorted({int(job["source_station_id"]) for job in jobs})
    ti = context["ti"]
    ti.xcom_push(key="prediction_jobs", value=jobs)
    ti.xcom_push(key="target_start_date", value=prediction_dates[0])
    ti.xcom_push(key="target_end_date", value=prediction_dates[-1])
    ti.xcom_push(key="run_type", value=runtime["run_type"])
    ti.xcom_push(key="station_ids", value=prepared_station_ids)
    ti.xcom_push(key="history_days", value=history_days)
    ti.xcom_push(key="skipped_stations", value=skipped_stations)

    logger.info(
        "[prepare] Prepared %s prediction job(s) for stations=%s target_range=%s..%s run_type=%s skipped=%s",
        len(jobs),
        prepared_station_ids,
        prediction_dates[0],
        prediction_dates[-1],
        runtime["run_type"],
        len(skipped_stations),
    )
    if skipped_stations:
        logger.warning("[prepare] Skipped station/date combinations: %s", skipped_stations)

    return {
        "prediction_jobs": len(jobs),
        "target_start_date": prediction_dates[0],
        "target_end_date": prediction_dates[-1],
        "station_ids": prepared_station_ids,
        "run_type": runtime["run_type"],
        "skipped_stations": skipped_stations,
        "station_target_dates": station_target_dates,
    }


def _call_prediction_api(**context):
    ti = context["ti"]
    jobs = ti.xcom_pull(task_ids="prepare_prediction_jobs", key="prediction_jobs") or []
    run_type = ti.xcom_pull(task_ids="prepare_prediction_jobs", key="run_type")
    api_url = DEFAULT_API_URL.rstrip("/")
    api_key = DEFAULT_API_KEY
    task_run_at = context["logical_date"].isoformat()
    generated_at = datetime.now(timezone.utc).isoformat()

    if not jobs:
        ti.xcom_push(key="prediction_rows", value=[])
        return {"prediction_rows": 0}

    prediction_rows = []

    for job in jobs:
        payload = json.dumps({"history": job["history"]}).encode("utf-8")
        http_request = request.Request(
            f"{api_url}/predict",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "X-API-Key": api_key,
            },
            method="POST",
        )

        try:
            with request.urlopen(http_request, timeout=60) as response:
                response_body = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            response_text = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(
                f"API /predict failed for target_date={job['target_date']} "
                f"status={exc.code} body={response_text}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(f"API /predict is unreachable at {api_url}") from exc

        if response_body.get("prediction_date") != job["target_date"]:
            raise ValueError(
                "API returned an unexpected prediction_date. "
                f"expected={job['target_date']} actual={response_body.get('prediction_date')}"
            )

        prediction_rows.append(
            {
                "prediction_date": response_body["prediction_date"],
                "predicted_pm25": float(response_body["predicted_pm25"]),
                "unit": response_body.get("unit", "ug/m3"),
                "model": response_body.get("model", "unknown"),
                "source_station_id": job["source_station_id"],
                "history_days": job["history_days"],
                "history_start_date": job["history_start_date"],
                "history_end_date": job["history_end_date"],
                "filled_history_days": job["filled_history_days"],
                "prediction_generated_at": generated_at,
                "run_type": run_type,
                "dag_id": context["dag"].dag_id,
                "dag_run_id": context["run_id"],
                "task_run_at": task_run_at,
            }
        )

    ti.xcom_push(key="prediction_rows", value=prediction_rows)
    logger.info(
        "[predict] API returned %s prediction row(s) from %s",
        len(prediction_rows),
        api_url,
    )

    return {
        "prediction_rows": len(prediction_rows),
        "api_url": api_url,
    }


def _store_predictions(**context):
    import sys

    sys.path.insert(0, SRC)

    from airflow_db import get_db_connection

    ti = context["ti"]
    prediction_rows = ti.xcom_pull(task_ids="call_prediction_api", key="prediction_rows") or []
    target_start_date = ti.xcom_pull(task_ids="prepare_prediction_jobs", key="target_start_date")
    target_end_date = ti.xcom_pull(task_ids="prepare_prediction_jobs", key="target_end_date")
    station_ids = ti.xcom_pull(task_ids="prepare_prediction_jobs", key="station_ids") or []

    if not prediction_rows:
        logger.info("[store] No prediction rows to store")
        return {"deleted": 0, "stored": 0}

    db = get_db_connection()
    try:
        db.ensure_api_prediction_table(DEFAULT_PREDICTION_TABLE)
        deleted_count, stored_count = db.replace_api_prediction_records_for_range(
            records=prediction_rows,
            start_date=target_start_date,
            end_date=target_end_date,
            table_name=DEFAULT_PREDICTION_TABLE,
            station_ids=[int(station_id) for station_id in station_ids],
        )
    finally:
        db.close()

    ti.xcom_push(key="deleted_count", value=deleted_count)
    ti.xcom_push(key="stored_count", value=stored_count)

    logger.info(
        "[store] Replaced %s rows with %s rows in %s for station_ids=%s range=%s..%s",
        deleted_count,
        stored_count,
        DEFAULT_PREDICTION_TABLE,
        station_ids,
        target_start_date,
        target_end_date,
    )

    return {
        "deleted": deleted_count,
        "stored": stored_count,
        "table_name": DEFAULT_PREDICTION_TABLE,
    }


with DAG(
    dag_id="pm25_api_prediction_hourly",
    description="Hourly API-based PM2.5 prediction DAG with optional date-range backfill",
    schedule="10 * * * *",
    start_date=pendulum.datetime(2026, 4, 15, tz=BANGKOK_TIMEZONE_NAME),
    catchup=False,
    max_active_runs=1,
    tags=["pm25", "prediction", "api", "postgresql"],
    default_args={
        "owner": "data-team",
        "retries": 2,
        "retry_delay": timedelta(minutes=2),
    },
    params={
        "start_date": None,
        "end_date": None,
    },
) as dag:
    start = EmptyOperator(task_id="start")

    prepare_prediction_jobs = PythonOperator(
        task_id="prepare_prediction_jobs",
        python_callable=_prepare_prediction_jobs,
    )

    call_prediction_api = PythonOperator(
        task_id="call_prediction_api",
        python_callable=_call_prediction_api,
    )

    store_predictions = PythonOperator(
        task_id="store_predictions",
        python_callable=_store_predictions,
    )

    done = EmptyOperator(task_id="done")

    start >> prepare_prediction_jobs >> call_prediction_api >> store_predictions >> done

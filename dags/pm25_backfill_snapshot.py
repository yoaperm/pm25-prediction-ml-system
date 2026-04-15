"""
On-demand backfill DAG for replacing a Bangkok-local date range in pm25_raw_hourly.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
import logging
import pendulum

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

SRC = "/app/src"
CONFIG_PATH = "/app/configs/config.yaml"
BANGKOK_TIMEZONE_NAME = "Asia/Bangkok"
BANGKOK_TZ = pendulum.timezone(BANGKOK_TIMEZONE_NAME)
DEFAULT_DATE_TEXT = pendulum.now(BANGKOK_TIMEZONE_NAME).to_date_string()
DEFAULT_CHUNK_DAYS = 30


def _parse_backfill_date(value) -> date:
    if isinstance(value, datetime):
        return value.astimezone(BANGKOK_TZ).date() if value.tzinfo else value.date()

    if isinstance(value, date):
        return value

    if value is None:
        raise ValueError("start_date/end_date must be provided")

    text = str(value).strip()
    if not text:
        raise ValueError("start_date/end_date must not be empty")

    for fmt in (
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%d%b%Y",
        "%d %b %Y",
        "%d%B%Y",
        "%d %B %Y",
    ):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue

    raise ValueError(
        "Unsupported date format. Use YYYY-MM-DD, DD/MM/YYYY, or values like 1Jan2024."
    )


def _resolve_requested_date_range(context) -> tuple[date, date]:
    dag_run = context.get("dag_run")
    dag_run_conf = dag_run.conf if dag_run and dag_run.conf else {}
    params = context.get("params", {})

    start_date = _parse_backfill_date(dag_run_conf.get("start_date", params.get("start_date")))
    end_date = _parse_backfill_date(dag_run_conf.get("end_date", params.get("end_date")))

    if start_date > end_date:
        raise ValueError("start_date must be less than or equal to end_date")

    return start_date, end_date


def _resolve_chunk_days(context) -> int:
    dag_run = context.get("dag_run")
    dag_run_conf = dag_run.conf if dag_run and dag_run.conf else {}
    params = context.get("params", {})
    raw_chunk_days = dag_run_conf.get("chunk_days", params.get("chunk_days", DEFAULT_CHUNK_DAYS))

    try:
        chunk_days = int(raw_chunk_days)
    except (TypeError, ValueError) as exc:
        raise ValueError("chunk_days must be a positive integer") from exc

    if chunk_days <= 0:
        raise ValueError("chunk_days must be a positive integer")

    return chunk_days


def _bangkok_day_bounds(start_date: date, end_date: date) -> tuple[datetime, datetime, datetime]:
    start_dt = datetime.combine(start_date, datetime.min.time(), tzinfo=BANGKOK_TZ)
    end_dt = datetime.combine(end_date, datetime.max.time().replace(microsecond=0), tzinfo=BANGKOK_TZ)
    end_exclusive = datetime.combine(end_date + timedelta(days=1), datetime.min.time(), tzinfo=BANGKOK_TZ)
    return start_dt, end_dt, end_exclusive


def _build_fetch_chunks(start_dt: datetime, end_dt: datetime, chunk_days: int) -> list[tuple[datetime, datetime]]:
    chunks: list[tuple[datetime, datetime]] = []
    chunk_start = start_dt

    while chunk_start <= end_dt:
        chunk_end = min(
            chunk_start + timedelta(days=chunk_days) - timedelta(seconds=1),
            end_dt,
        )
        chunks.append((chunk_start, chunk_end))
        chunk_start = chunk_end + timedelta(seconds=1)

    return chunks


def _validate_data(**context):
    ti = context["ti"]
    start_date, end_date = _resolve_requested_date_range(context)
    chunk_days = _resolve_chunk_days(context)
    range_start, range_end, range_end_exclusive = _bangkok_day_bounds(start_date, end_date)
    fetch_chunks = _build_fetch_chunks(range_start, range_end, chunk_days)

    ti.xcom_push(key="backfill_start_date", value=start_date.isoformat())
    ti.xcom_push(key="backfill_end_date", value=end_date.isoformat())
    ti.xcom_push(key="range_start", value=range_start.isoformat())
    ti.xcom_push(key="range_end", value=range_end.isoformat())
    ti.xcom_push(key="range_end_exclusive", value=range_end_exclusive.isoformat())
    ti.xcom_push(key="chunk_days", value=chunk_days)
    ti.xcom_push(key="chunk_count", value=len(fetch_chunks))

    logger.info(
        "[validate] Backfill range resolved to %s through %s Bangkok time",
        range_start.isoformat(),
        range_end.isoformat(),
    )
    logger.info(
        "[validate] Fetch will run in %s chunk(s) using chunk_days=%s",
        len(fetch_chunks),
        chunk_days,
    )

    return {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "range_start": range_start.isoformat(),
        "range_end": range_end.isoformat(),
        "range_end_exclusive": range_end_exclusive.isoformat(),
        "chunk_days": chunk_days,
        "chunk_count": len(fetch_chunks),
    }


def _validate_records(records):
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

        for field, rule in validation_rules.items():
            if field not in record:
                continue

            value = record[field]
            if value is None:
                continue

            if not (rule["min"] <= value <= rule["max"]):
                is_valid = False
                errors.append(f"{field}={value} out of range [{rule['min']}, {rule['max']}]")

        if is_valid:
            valid_records.append(record)
        else:
            failed_count += 1
            logger.warning(
                "[fetch] Backfill record rejected - station_id=%s timestamp=%s errors=%s",
                record.get("station_id"),
                record.get("timestamp"),
                "; ".join(errors),
            )

    return valid_records, failed_count


def _fetch_backfill_data(**context):
    import sys

    sys.path.insert(0, SRC)

    from airbkk_client import AirBKKClient, REQUIRED_STATION_IDS

    ti = context["ti"]
    range_start = datetime.fromisoformat(ti.xcom_pull(task_ids="validate_data", key="range_start"))
    range_end = datetime.fromisoformat(ti.xcom_pull(task_ids="validate_data", key="range_end"))
    chunk_days = int(ti.xcom_pull(task_ids="validate_data", key="chunk_days"))
    fetch_chunks = _build_fetch_chunks(range_start, range_end, chunk_days)

    logger.info(
        "[fetch] Starting backfill fetch for %s through %s",
        range_start.isoformat(),
        range_end.isoformat(),
    )
    logger.info(
        "[fetch] Station filter=%s total_chunks=%s chunk_days=%s",
        REQUIRED_STATION_IDS,
        len(fetch_chunks),
        chunk_days,
    )

    client = AirBKKClient(timeout=120)
    all_valid_records = []
    total_fetched = 0
    total_rejected = 0

    for chunk_index, (chunk_start, chunk_end) in enumerate(fetch_chunks, start=1):
        logger.info(
            "[fetch] Chunk %s/%s requesting %s through %s",
            chunk_index,
            len(fetch_chunks),
            chunk_start.isoformat(),
            chunk_end.isoformat(),
        )
        chunk_records = client.get_records_for_range(
            start_dt=chunk_start,
            end_dt=chunk_end,
            station_ids=REQUIRED_STATION_IDS,
        )
        valid_chunk_records, rejected_chunk_records = _validate_records(chunk_records)
        total_fetched += len(chunk_records)
        total_rejected += rejected_chunk_records
        all_valid_records.extend(valid_chunk_records)

        logger.info(
            "[fetch] Chunk %s/%s completed raw=%s valid=%s rejected=%s cumulative_valid=%s",
            chunk_index,
            len(fetch_chunks),
            len(chunk_records),
            len(valid_chunk_records),
            rejected_chunk_records,
            len(all_valid_records),
        )

    logger.info(
        "[fetch] Backfill fetch completed raw=%s valid=%s rejected=%s",
        total_fetched,
        len(all_valid_records),
        total_rejected,
    )

    ti.xcom_push(key="fetched_records", value=all_valid_records)
    ti.xcom_push(key="fetch_count", value=total_fetched)
    ti.xcom_push(key="validation_failures", value=total_rejected)

    return {
        "fetched_records": total_fetched,
        "valid_records": len(all_valid_records),
        "validation_failures": total_rejected,
    }


def _replace_backfill_range(**context):
    import sys

    sys.path.insert(0, SRC)

    from airflow_db import get_db_connection
    from airbkk_client import REQUIRED_STATION_IDS

    ti = context["ti"]
    records = ti.xcom_pull(task_ids="fetch_backfill_data", key="fetched_records") or []
    range_start = datetime.fromisoformat(ti.xcom_pull(task_ids="validate_data", key="range_start"))
    range_end_exclusive = datetime.fromisoformat(
        ti.xcom_pull(task_ids="validate_data", key="range_end_exclusive")
    )

    db = get_db_connection(CONFIG_PATH)

    try:
        db.ensure_table()
        cur = db.conn.cursor()
        cur.execute(
            """
            SELECT COUNT(*)
            FROM pm25_raw_hourly
            WHERE timestamp >= %s
              AND timestamp < %s
              AND station_id = ANY(%s)
            """,
            (range_start, range_end_exclusive, list(REQUIRED_STATION_IDS)),
        )
        existing_count = cur.fetchone()[0]
        cur.close()
        logger.info(
            "[replace] Preparing DB replace for %s through %s station_ids=%s existing_rows=%s new_records=%s",
            range_start.isoformat(),
            range_end_exclusive.isoformat(),
            REQUIRED_STATION_IDS,
            existing_count,
            len(records),
        )
        deleted_count, stored_count, duplicate_count = db.replace_records_for_range(
            records=records,
            start_timestamp=range_start,
            end_timestamp=range_end_exclusive,
            station_ids=REQUIRED_STATION_IDS,
        )
    finally:
        db.close()

    ti.xcom_push(key="deleted_count", value=deleted_count)
    ti.xcom_push(key="stored_count", value=stored_count)
    ti.xcom_push(key="duplicate_count", value=duplicate_count)

    logger.info(
        "[replace] Deleted %s rows and inserted %s rows (%s duplicates skipped)",
        deleted_count,
        stored_count,
        duplicate_count,
    )

    return {
        "deleted": deleted_count,
        "stored": stored_count,
        "duplicates": duplicate_count,
    }


def _log_summary(**context):
    ti = context["ti"]

    summary = {
        "start_date": ti.xcom_pull(task_ids="validate_data", key="backfill_start_date"),
        "end_date": ti.xcom_pull(task_ids="validate_data", key="backfill_end_date"),
        "range_start": ti.xcom_pull(task_ids="validate_data", key="range_start"),
        "range_end_exclusive": ti.xcom_pull(task_ids="validate_data", key="range_end_exclusive"),
        "fetched_records": ti.xcom_pull(task_ids="fetch_backfill_data", key="fetch_count") or 0,
        "validation_failures": ti.xcom_pull(task_ids="fetch_backfill_data", key="validation_failures") or 0,
        "deleted_records": ti.xcom_pull(task_ids="replace_backfill_range", key="deleted_count") or 0,
        "stored_records": ti.xcom_pull(task_ids="replace_backfill_range", key="stored_count") or 0,
        "duplicate_records": ti.xcom_pull(task_ids="replace_backfill_range", key="duplicate_count") or 0,
        "logged_at": pendulum.now(BANGKOK_TIMEZONE_NAME).isoformat(),
    }

    logger.info("[summary] Backfill summary: %s", summary)
    return summary


dag = DAG(
    dag_id="pm25_backfill_snapshot",
    description="On-demand PM2.5 backfill into pm25_raw_hourly for a Bangkok-local date range",
    schedule_interval=None,
    start_date=pendulum.datetime(2026, 4, 15, tz=BANGKOK_TIMEZONE_NAME),
    catchup=False,
    params={
        "start_date": DEFAULT_DATE_TEXT,
        "end_date": DEFAULT_DATE_TEXT,
        "chunk_days": DEFAULT_CHUNK_DAYS,
    },
    tags=["pm25", "backfill", "on-demand"],
    default_args={
        "owner": "data-team",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    max_active_runs=1,
)


with dag:
    start = EmptyOperator(task_id="start")

    validate_task = PythonOperator(
        task_id="validate_data",
        python_callable=_validate_data,
        provide_context=True,
    )

    fetch_task = PythonOperator(
        task_id="fetch_backfill_data",
        python_callable=_fetch_backfill_data,
        provide_context=True,
    )

    replace_task = PythonOperator(
        task_id="replace_backfill_range",
        python_callable=_replace_backfill_range,
        provide_context=True,
    )

    summary_task = PythonOperator(
        task_id="log_summary",
        python_callable=_log_summary,
        provide_context=True,
    )

    end = EmptyOperator(task_id="end")

    start >> validate_task >> fetch_task >> replace_task >> summary_task >> end  # noqa: F841

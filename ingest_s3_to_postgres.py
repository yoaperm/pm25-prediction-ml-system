#!/usr/bin/env python3
"""
ingest_s3_to_postgres.py
========================
Ingest PM2.5 station data from S3 JSON files into a PostgreSQL database
running inside Docker on EC2.

Usage:
    pip install pandas sqlalchemy psycopg2-binary s3fs boto3
    python ingest_s3_to_postgres.py

Prerequisites:
    - AWS credentials configured (IAM role on EC2, or ~/.aws/credentials, or env vars)
    - PostgreSQL running in Docker with port 5432 exposed to the host
    - Database 'pm25' and table 'pm25_raw_hourly' already exist
"""

import sys
import logging
import datetime

import pandas as pd
from sqlalchemy import create_engine, text, types

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# If the script runs directly on the EC2 host, use localhost (port must be
# published in docker-compose: "5432:5432").
# If the script runs inside the same Docker network, replace localhost with
# the container name, e.g. "pm25-prediction-ml-system-postgres-1".
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "pm25"
DB_USER = "postgres"
DB_PASS = "postgres"

DB_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

TABLE_NAME = "pm25_raw_hourly"

TIMEZONE = datetime.timezone(datetime.timedelta(hours=7))  # Asia/Bangkok

S3_FILES = [
    "s3://seml-pm-airbkk-dataset-616105295716-ap-southeast-7-an/raw/station_56/data.json",
    "s3://seml-pm-airbkk-dataset-616105295716-ap-southeast-7-an/raw/station_57/data.json",
    "s3://seml-pm-airbkk-dataset-616105295716-ap-southeast-7-an/raw/station_58/data.json",
    "s3://seml-pm-airbkk-dataset-616105295716-ap-southeast-7-an/raw/station_59/data.json",
    "s3://seml-pm-airbkk-dataset-616105295716-ap-southeast-7-an/raw/station_61/data.json",
]

# Map DataFrame columns → SQL types (matches the table schema in the image)
DTYPE_MAP = {
    "station_id":       types.Integer,
    "station_name":     types.String,
    "station_name_en":  types.String,
    "timestamp":        types.DateTime(timezone=True),
    "pm25":             types.Float,
    "pm10":             types.Float,
    "temp":             types.Float,
    "rh":               types.Float,
    "ws":               types.Float,
    "wd":               types.Float,
    "ingestion_time":   types.DateTime(timezone=True),
}

# Expected column order (must match the DB table; 'id' is auto-generated)
EXPECTED_COLUMNS = [
    "station_id", "station_name", "station_name_en",
    "timestamp", "pm25", "pm10", "temp", "rh", "ws", "wd",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_engine():
    """Create and test a SQLAlchemy engine."""
    engine = create_engine(DB_URL, pool_pre_ping=True)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    log.info("Connected to PostgreSQL at %s/%s", DB_HOST, DB_NAME)
    return engine


def get_latest_timestamp(engine, station_id: int):
    """Return the latest timestamp already in the DB for a given station,
    or None if no rows exist yet. Used to avoid duplicate inserts."""
    query = text(
        f"SELECT MAX(timestamp) FROM {TABLE_NAME} WHERE station_id = :sid"
    )
    with engine.connect() as conn:
        result = conn.execute(query, {"sid": station_id}).scalar()
    return result


def read_s3_json(s3_path: str) -> pd.DataFrame:
    """Read a JSON file from S3 into a DataFrame.
    Requires the 's3fs' package and valid AWS credentials."""
    df = pd.read_json(s3_path)
    log.info("  Read %d rows from %s", len(df), s3_path)
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate columns, cast types, add ingestion_time."""
    # Drop 'id' if present — Postgres auto-generates it
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Check that all expected columns exist
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in JSON: {missing}")

    # Convert timestamp to timezone-aware (Asia/Bangkok = UTC+7)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True) \
                        .dt.tz_convert("Asia/Bangkok")

    # Add ingestion timestamp
    df["ingestion_time"] = datetime.datetime.now(tz=TIMEZONE)

    # Keep only the columns we need, in the right order
    df = df[EXPECTED_COLUMNS + ["ingestion_time"]]

    return df


def deduplicate(df: pd.DataFrame, engine, station_id: int) -> pd.DataFrame:
    """Remove rows that are already in the database (based on timestamp)."""
    latest_ts = get_latest_timestamp(engine, station_id)
    if latest_ts is not None:
        before = len(df)
        # Make comparison timezone-aware if needed
        if df["timestamp"].dt.tz is not None and latest_ts.tzinfo is None:
            import pytz
            latest_ts = pytz.UTC.localize(latest_ts)
        df = df[df["timestamp"] > latest_ts]
        skipped = before - len(df)
        if skipped:
            log.info("  Skipped %d existing rows (up to %s)", skipped, latest_ts)
    return df


def ingest_file(s3_path: str, engine):
    """Full pipeline for one S3 file: read → clean → deduplicate → insert."""
    log.info("Processing %s", s3_path)

    # 1. Read from S3
    df = read_s3_json(s3_path)
    if df.empty:
        log.warning("  Empty file, skipping.")
        return

    # 2. Clean & validate
    df = clean_dataframe(df)

    # 3. Deduplicate against existing DB rows
    station_id = int(df["station_id"].iloc[0])
    df = deduplicate(df, engine, station_id)
    if df.empty:
        log.info("  No new rows to insert.")
        return

    # 4. Insert into PostgreSQL
    rows = df.to_sql(
        TABLE_NAME,
        engine,
        if_exists="append",
        index=False,
        dtype=DTYPE_MAP,
        method="multi",      # batch insert for speed
        chunksize=1000,
    )
    log.info("  ✅ Inserted %d new rows (station_id=%d)", len(df), station_id)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=" * 60)
    log.info("S3 → PostgreSQL ingestion — start")
    log.info("=" * 60)

    engine = get_engine()

    success, failed = 0, 0
    for s3_path in S3_FILES:
        try:
            ingest_file(s3_path, engine)
            success += 1
        except Exception:
            log.exception("❌ Failed: %s", s3_path)
            failed += 1

    log.info("-" * 60)
    log.info("Done.  success=%d  failed=%d", success, failed)
    log.info("-" * 60)

    # Verify row counts
    with engine.connect() as conn:
        total = conn.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME}")).scalar()
        ts_range = conn.execute(
            text(f"SELECT MIN(timestamp), MAX(timestamp) FROM {TABLE_NAME}")
        ).fetchone()
    log.info("Total rows in %s: %d", TABLE_NAME, total)
    log.info("Timestamp range: %s → %s", ts_range[0], ts_range[1])

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

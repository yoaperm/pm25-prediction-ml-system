#!/usr/bin/env python3
"""ingest_s3.py — Read S3 JSON files and insert via PM25Database."""

import json
import math
import datetime
import boto3
from airflow_db import PM25Database

S3_BUCKET = "seml-pm-airbkk-dataset-616105295716-ap-southeast-7-an"
S3_KEYS = [
    "raw/station_56/data.json",
    "raw/station_57/data.json",
    "raw/station_58/data.json",
    "raw/station_59/data.json",
    "raw/station_61/data.json",
]

FIELD_MAP = {
    "StationID":   "station_id",
    "StationName": "station_name",
    "Date_Time":   "timestamp",
    "PM2.5":       "pm25",
    "PM10":        "pm10",
    "Temp":        "temp",
    "RH":          "rh",
    "WS":          "ws",
    "WD":          "wd",
}

TIMEZONE = datetime.timezone(datetime.timedelta(hours=7))

def clean_value(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return v

def clamp(value, min_val, max_val):
    """Clamp value to range, return None if invalid."""
    if value is None:
        return None
    if value < min_val or value > max_val:
        return min(max(value, min_val), max_val)
    return value

def transform_record(raw: dict, ingestion_time) -> dict:
    record = {
        db_col: clean_value(raw.get(json_key))
        for json_key, db_col in FIELD_MAP.items()
    }
    # Clamp to match DB CHECK constraints
    record["pm25"] = clamp(record.get("pm25"), 0, 500)
    record["rh"]   = clamp(record.get("rh"), 0, 100)
    record["ws"]   = clamp(record.get("ws"), 0, 200)
    if record.get("pm10") is not None and record["pm10"] < 0:
        record["pm10"] = 0
    record["ingestion_time"] = ingestion_time
    return record

def main():
    s3 = boto3.client("s3")
    db = PM25Database(host="localhost")
    db.connect()
    db.ensure_table()

    ingestion_time = datetime.datetime.now(tz=TIMEZONE)
    print(f"Ingestion time: {ingestion_time}")

    for key in S3_KEYS:
        print(f"Processing s3://{S3_BUCKET}/{key}")
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
        raw_records = json.loads(obj["Body"].read())

        records = [transform_record(r, ingestion_time) for r in raw_records]

        inserted, dupes = db.insert_records(records)
        print(f"  ✅ {inserted} inserted, {dupes} duplicates skipped")

    total = db.get_row_count()
    print(f"\nTotal rows in pm25_raw_hourly: {total}")
    db.close()

if __name__ == "__main__":
    main()

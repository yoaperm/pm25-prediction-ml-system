"""
AirBKK hourly data client.

This module wraps the AirBKK report endpoints used in the Colab notebook and
normalizes hourly snapshots into the schema expected by the ingestion DAGs.
"""

from __future__ import annotations

from datetime import datetime
import json
import logging
import time
from typing import Dict, Iterable, List, Optional, Sequence
from urllib import error, parse, request
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

BANGKOK_TZ = ZoneInfo("Asia/Bangkok")
BANGKOK_TIMEZONE_NAME = "Asia/Bangkok"

REQUIRED_STATION_IDS = (56, 57, 58, 59, 61)
STATION_NAME_EN_OVERRIDES = {
    58: "Rat Burana District",
}

BASE_URL = "https://official.airbkk.com"
REPORT_BASE_URL = f"{BASE_URL}/airbkk/Report"

TARGET_PARAMETER_TAGS = ("PM2.5", "PM10", "Temp", "RH", "WS", "WD")
PARAMETER_FIELD_MAP = {
    "PM2.5": "pm25",
    "PM10": "pm10",
    "Temp": "temp",
    "RH": "rh",
    "WS": "ws",
    "WD": "wd",
}


def parse_thai_buddhist_datetime(value: str) -> datetime:
    """Convert AirBKK timestamps like 14/04/2569 02:00 to Bangkok-aware time."""
    date_part, time_part = value.strip().split(" ")
    day, month, buddhist_year = (int(part) for part in date_part.split("/"))
    hour, minute = (int(part) for part in time_part.split(":"))
    return datetime(buddhist_year - 543, month, day, hour, minute, tzinfo=BANGKOK_TZ)


def format_airbkk_datetime(target_dt: datetime) -> str:
    """Format a datetime for AirBKK hourly requests in Bangkok local time."""
    if target_dt.tzinfo is not None:
        local_dt = target_dt.astimezone(BANGKOK_TZ).replace(tzinfo=None)
    else:
        local_dt = target_dt
    return local_dt.strftime("%d/%m/%Y %H:%M")


def _coerce_float(value: object) -> Optional[float]:
    if value is None:
        return None

    text = str(value).strip().replace(",", "")
    if not text or text in {"-", "None", "nan", "NaN"}:
        return None

    try:
        return float(text)
    except ValueError:
        return None


class AirBKKClient:
    """Minimal client for the AirBKK report endpoints."""

    def __init__(
        self,
        timeout: int = 60,
        max_retries: int = 3,
        retry_backoff_sec: float = 1.0,
    ) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff_sec = retry_backoff_sec
        self.headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": f"{BASE_URL}/airbkk/th/report",
            "Origin": BASE_URL,
        }

    def _post_json(self, path: str, data: Iterable[tuple[str, str]]) -> object:
        url = f"{REPORT_BASE_URL}/{path}"
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                encoded = parse.urlencode(list(data)).encode("utf-8")
                req = request.Request(url, data=encoded, headers=self.headers)
                with request.urlopen(req, timeout=self.timeout) as response:
                    payload = json.load(response)

                if isinstance(payload, dict):
                    status = payload.get("status")
                    if status not in (None, "success"):
                        raise RuntimeError(f"AirBKK returned status={status!r}")

                return payload
            except (error.URLError, json.JSONDecodeError, RuntimeError) as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break

                sleep_for = self.retry_backoff_sec * attempt
                logger.warning(
                    "AirBKK request failed for %s (attempt %s/%s): %s",
                    path,
                    attempt,
                    self.max_retries,
                    exc,
                )
                time.sleep(sleep_for)

        raise RuntimeError(f"AirBKK request failed for {path}") from last_error

    def get_measurements(self) -> List[Dict[str, object]]:
        payload = self._post_json("getMeasurements", [("Moo", "all")])
        if not isinstance(payload, list):
            raise RuntimeError("Unexpected getMeasurements payload")
        return payload

    def _build_station_lookup(
        self,
        stations: Sequence[Dict[str, object]],
    ) -> Dict[int, Dict[str, Optional[str]]]:
        station_lookup: Dict[int, Dict[str, Optional[str]]] = {}
        for station in stations:
            raw_station_id = station.get("MeasIndex")
            if raw_station_id is None:
                continue

            station_id = int(raw_station_id)
            station_lookup[station_id] = {
                "station_name": station.get("District"),
                "station_name_en": STATION_NAME_EN_OVERRIDES.get(
                    station_id,
                    station.get("District_en"),
                ),
            }

        return station_lookup

    def _resolve_station_ids(
        self,
        stations: Sequence[Dict[str, object]],
        station_ids: Optional[Sequence[int]] = None,
    ) -> List[int]:
        available_station_ids = {
            int(station["MeasIndex"])
            for station in stations
            if station.get("MeasIndex") is not None
        }
        requested_station_ids = station_ids or REQUIRED_STATION_IDS
        return [
            station_id
            for station_id in requested_station_ids
            if station_id in available_station_ids
        ]

    def get_hourly_records(
        self,
        target_dt: datetime,
        station_ids: Optional[Sequence[int]] = None,
    ) -> List[Dict[str, object]]:
        """Fetch one hourly snapshot for the required stations."""
        stations = self.get_measurements()
        station_lookup = self._build_station_lookup(stations)
        resolved_station_ids = self._resolve_station_ids(stations, station_ids)

        if not resolved_station_ids:
            return []

        target_text = format_airbkk_datetime(target_dt)
        payload: List[tuple[str, str]] = [("groupid", "all")]
        payload.extend(("MeasIndex[]", str(station_id)) for station_id in resolved_station_ids)
        payload.extend(("parameterTags[]", tag) for tag in TARGET_PARAMETER_TAGS)
        payload.extend(
            [
                ("data_type", "hourly"),
                ("date_s", target_text),
                ("date_e", target_text),
                ("display_type", "table"),
            ]
        )

        response = self._post_json("getData", payload)
        if not isinstance(response, dict):
            raise RuntimeError("Unexpected getData payload")

        return self._normalize_snapshot(
            response=response,
            station_lookup=station_lookup,
            station_ids=resolved_station_ids,
        )

    def get_records_for_range(
        self,
        start_dt: datetime,
        end_dt: datetime,
        station_ids: Optional[Sequence[int]] = None,
    ) -> List[Dict[str, object]]:
        """Fetch a Bangkok-local hourly range for the required stations."""
        stations = self.get_measurements()
        station_lookup = self._build_station_lookup(stations)
        resolved_station_ids = self._resolve_station_ids(stations, station_ids)

        if not resolved_station_ids:
            return []

        payload: List[tuple[str, str]] = [("groupid", "all")]
        payload.extend(("MeasIndex[]", str(station_id)) for station_id in resolved_station_ids)
        payload.extend(("parameterTags[]", tag) for tag in TARGET_PARAMETER_TAGS)
        payload.extend(
            [
                ("data_type", "hourly"),
                ("date_s", format_airbkk_datetime(start_dt)),
                ("date_e", format_airbkk_datetime(end_dt)),
                ("display_type", "table"),
            ]
        )

        response = self._post_json("getData", payload)
        if not isinstance(response, dict):
            raise RuntimeError("Unexpected getData payload")

        return self._normalize_snapshot(
            response=response,
            station_lookup=station_lookup,
            station_ids=resolved_station_ids,
        )

    def _normalize_snapshot(
        self,
        response: Dict[str, object],
        station_lookup: Dict[int, Dict[str, Optional[str]]],
        station_ids: Sequence[int],
    ) -> List[Dict[str, object]]:
        arr_parameter = response.get("arrParameter") or []
        arr_data = response.get("arrData") or []
        allowed_station_ids = set(station_ids)

        alias_map: Dict[str, Dict[str, str]] = {}
        for item in arr_parameter:
            if not isinstance(item, dict):
                continue

            alias = item.get("Alias")
            station_id = item.get("MeasIndex")
            parameter = item.get("ShortName")
            field_name = PARAMETER_FIELD_MAP.get(str(parameter))

            if alias and station_id is not None and field_name and int(station_id) in allowed_station_ids:
                alias_map[str(alias)] = {
                    "station_id": str(station_id),
                    "field_name": field_name,
                }

        records_by_key: Dict[tuple[int, str], Dict[str, object]] = {}

        for row in arr_data:
            if not isinstance(row, dict):
                continue

            date_text = row.get("Date_Time")
            if not date_text:
                continue

            timestamp = parse_thai_buddhist_datetime(str(date_text))
            timestamp_text = timestamp.isoformat()

            for alias, raw_value in row.items():
                if alias == "Date_Time":
                    continue

                meta = alias_map.get(alias)
                if not meta:
                    continue

                station_id = int(meta["station_id"])
                station_meta = station_lookup.get(station_id, {})
                record_key = (station_id, timestamp_text)
                record = records_by_key.setdefault(
                    record_key,
                    {
                        "station_id": station_id,
                        "station_name": station_meta.get("station_name"),
                        "station_name_en": station_meta.get("station_name_en"),
                        "timestamp": timestamp_text,
                        "pm25": None,
                        "pm10": None,
                        "temp": None,
                        "rh": None,
                        "ws": None,
                        "wd": None,
                    },
                )
                record[meta["field_name"]] = _coerce_float(raw_value)

        records = [
            record
            for _, record in sorted(records_by_key.items())
            if any(record[field] is not None for field in PARAMETER_FIELD_MAP.values())
        ]
        return records

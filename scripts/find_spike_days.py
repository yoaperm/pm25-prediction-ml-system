#!/usr/bin/env python3
"""
find_spike_days.py
==================
Scan PM2.5 hourly data and identify days with significant spikes.
Outputs a curated list of "demo-worthy" spike days for FoonAlert replay mode.

Usage:
    python scripts/find_spike_days.py
    python scripts/find_spike_days.py --station 59
    python scripts/find_spike_days.py --export-csv demo_data/spike_days.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np


DATA_PATH = Path("data/backup/pm25_raw_hourly_20240101_20250415.csv")
STATION_NAMES = {
    56: "Din Daeng",
    57: "Bang Khun Thian",
    58: "Khlong Toei",
    59: "Wang Thonglang",
    61: "Lat Phrao",
}


def load_data(path: Path) -> pd.DataFrame:
    """Load and prepare hourly PM2.5 data."""
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    return df


def find_spikes(df: pd.DataFrame, station_id: int = None, top_n: int = 10) -> pd.DataFrame:
    """
    Find spike days defined by:
    1. Large PM2.5 increase within 6 hours (spike_score)
    2. High daily range (max - min)
    3. At least 20 hours of data (quality filter)
    4. Max PM2.5 < 250 (filter sensor errors)
    """
    if station_id:
        df = df[df["station_id"] == station_id]

    results = []

    for sid in df["station_id"].unique():
        station_df = df[df["station_id"] == sid].copy()
        station_df["date"] = station_df["timestamp"].dt.date

        # Compute 6-hour forward spike
        station_df["pm25_6h_ahead"] = station_df["pm25"].shift(-6)
        station_df["spike_6h"] = station_df["pm25_6h_ahead"] - station_df["pm25"]

        # Daily aggregates
        daily = station_df.groupby("date").agg(
            pm25_min=("pm25", "min"),
            pm25_max=("pm25", "max"),
            pm25_mean=("pm25", "mean"),
            max_spike_6h=("spike_6h", "max"),
            hour_count=("pm25", "count"),
            null_count=("pm25", lambda x: x.isna().sum()),
        ).reset_index()

        daily["range"] = daily["pm25_max"] - daily["pm25_min"]
        daily["station_id"] = sid
        daily["station_name"] = STATION_NAMES.get(sid, f"Station {sid}")

        # Quality filters
        daily = daily[
            (daily["hour_count"] >= 20)
            & (daily["pm25_max"] < 250)
            & (daily["pm25_max"] > 50)  # Must have elevated PM2.5
            & (daily["range"] > 30)  # Must have meaningful variation
        ]

        # Composite spike score
        daily["spike_score"] = (
            daily["max_spike_6h"].clip(lower=0) * 0.5
            + daily["range"] * 0.3
            + daily["pm25_max"] * 0.2
        )

        results.append(daily)

    if not results:
        return pd.DataFrame()

    all_spikes = pd.concat(results, ignore_index=True)
    all_spikes = all_spikes.sort_values("spike_score", ascending=False)

    if station_id:
        return all_spikes.head(top_n)
    else:
        # Top N per station
        return all_spikes.groupby("station_id").head(top_n // len(df["station_id"].unique()) + 1)


def classify_spike_pattern(df: pd.DataFrame, station_df: pd.DataFrame, date) -> str:
    """Classify the spike pattern for description."""
    day = station_df[station_df["timestamp"].dt.date == date].sort_values("timestamp")
    if day.empty:
        return "Unknown"

    pm25_vals = day["pm25"].dropna().values
    if len(pm25_vals) < 12:
        return "Incomplete"

    peak_hour = day["pm25"].idxmax()
    peak_time = day.loc[peak_hour, "timestamp"].hour if peak_hour in day.index else 12

    if peak_time < 9:
        return "Morning spike"
    elif peak_time < 15:
        return "Midday spike"
    elif peak_time < 21:
        return "Evening spike"
    else:
        return "Night spike"


def main():
    parser = argparse.ArgumentParser(description="Find PM2.5 spike days for demo replay")
    parser.add_argument("--station", type=int, help="Filter by station ID")
    parser.add_argument("--top", type=int, default=15, help="Top N results")
    parser.add_argument("--export-csv", type=str, help="Export results to CSV")
    parser.add_argument("--data-path", type=str, default=str(DATA_PATH))
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)

    print(f"Loading data from {data_path}...")
    df = load_data(data_path)
    print(f"  {len(df)} records, stations: {sorted(df['station_id'].unique())}")

    print(f"\nFinding spike days (top {args.top})...")
    spikes = find_spikes(df, station_id=args.station, top_n=args.top)

    if spikes.empty:
        print("No spike days found matching criteria.")
        sys.exit(0)

    # Display results
    display_cols = [
        "station_id", "station_name", "date",
        "pm25_min", "pm25_max", "range", "max_spike_6h", "spike_score"
    ]
    print("\n" + "=" * 90)
    print("TOP SPIKE DAYS FOR DEMO REPLAY")
    print("=" * 90)
    print(spikes[display_cols].to_string(index=False, float_format="%.1f"))
    print("=" * 90)

    # Recommended demo days
    print("\n📌 RECOMMENDED DEMO DAYS:")
    top3 = spikes.nlargest(5, "spike_score")
    for _, row in top3.iterrows():
        print(f"  Station {row['station_id']} ({row['station_name']}) — {row['date']}")
        print(f"    PM2.5: {row['pm25_min']:.0f} → {row['pm25_max']:.0f} µg/m³ "
              f"(range: {row['range']:.0f}, spike_6h: {row['max_spike_6h']:.0f})")

    if args.export_csv:
        out_path = Path(args.export_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        spikes[display_cols].to_csv(out_path, index=False)
        print(f"\n✅ Exported to {out_path}")


if __name__ == "__main__":
    main()

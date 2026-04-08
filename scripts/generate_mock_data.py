"""
Generate Mock PM2.5 Data
========================
Creates a CSV file with fake PM2.5 readings to simulate new incoming data.

Modes:
  normal    low noise  → model stays healthy (MAE stays low)
  degraded  high spike → model looks bad (MAE jumps, triggers retrain)
  drift     gradual shift upward over time

Usage:
    python scripts/generate_mock_data.py --mode degraded --days 20
    python scripts/generate_mock_data.py --mode normal   --days 20
    python scripts/generate_mock_data.py --mode drift    --days 20
"""

import argparse
import numpy as np
import pandas as pd
from datetime import date, timedelta

OUTPUT_FILE = "data/mock/mock_pm25.csv"
START_DATE  = date(2025, 6, 1)
BASE_VALUES = [42.1, 38.5, 35.2, 40.0, 44.3, 39.8, 36.1,
               33.7, 41.2, 45.6, 38.9, 37.4, 43.0, 40.5,
               36.8, 39.2, 41.8, 37.6, 43.5, 40.1, 38.7,
               35.9, 42.4, 39.6, 44.1, 37.8, 40.3, 38.1,
               43.7, 41.5]


def generate(mode: str, days: int, seed: int = 42):
    np.random.seed(seed)
    rows = []

    for i in range(days):
        day       = START_DATE + timedelta(days=i)
        base      = BASE_VALUES[i % len(BASE_VALUES)]

        if mode == "normal":
            pm25 = base + np.random.normal(0, 2)

        elif mode == "degraded":
            # Sudden spike — 2–3x normal values
            pm25 = base * np.random.uniform(2.0, 3.0)

        elif mode == "drift":
            # Gradual upward drift
            drift = (i / days) * 30
            pm25  = base + drift + np.random.normal(0, 3)

        rows.append({"date": str(day), "pm25": round(float(np.clip(pm25, 0, 500)), 2)})

    df = pd.DataFrame(rows)
    import os; os.makedirs("data/mock", exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved {len(df)} rows to {OUTPUT_FILE}")
    print(f"Mode : {mode}")
    print(f"PM2.5 mean={df['pm25'].mean():.1f}  std={df['pm25'].std():.1f}  "
          f"min={df['pm25'].min():.1f}  max={df['pm25'].max():.1f}")
    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",  choices=["normal", "degraded", "drift"], default="degraded")
    parser.add_argument("--days",  type=int, default=25)
    args = parser.parse_args()
    generate(args.mode, args.days)

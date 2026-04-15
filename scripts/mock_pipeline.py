"""
All-in-One Mock Pipeline
=========================
Generates mock data → sends predictions + actuals → triggers pm25_pipeline DAG
→ polls until done → prints summary.

Modes:
  normal    small noise  → MAE low, PSI low  → no retrain
  degraded  2–3x spike   → MAE high, PSI high → retrain
  drift     gradual shift → PSI rises gradually → retrain

Usage:
    python scripts/mock_pipeline.py --mode degraded
    python scripts/mock_pipeline.py --mode normal
    python scripts/mock_pipeline.py --mode drift --days 30
"""

import argparse
import sys
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

API_URL      = "http://localhost:8001"
AIRFLOW_URL  = "http://localhost:8080"
AIRFLOW_AUTH = ("admin", "admin")
HISTORY_DAYS = 15

BASE_PM25 = [42.1, 38.5, 35.2, 40.0, 44.3, 39.8, 36.1,
             33.7, 41.2, 45.6, 38.9, 37.4, 43.0, 40.5,
             36.8, 39.2, 41.8, 37.6, 43.5, 40.1, 38.7,
             35.9, 42.4, 39.6, 44.1, 37.8, 40.3, 38.1,
             43.7, 41.5]


def divider(title=""):
    print("\n" + "─" * 55)
    if title:
        print(f"  {title}")


def check_services():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        assert r.json()["status"] == "ok"
    except Exception:
        print("ERROR: API not reachable. Run `docker compose up -d` first.")
        sys.exit(1)


def generate_mock(mode: str, days: int, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    rows = []
    start = date(2025, 6, 1)
    for i in range(days):
        base = BASE_PM25[i % len(BASE_PM25)]
        if mode == "normal":
            pm25 = base + np.random.normal(0, 2)
        elif mode == "degraded":
            pm25 = base * np.random.uniform(2.0, 3.0)
        elif mode == "drift":
            drift = (i / days) * 35
            pm25  = base + drift + np.random.normal(0, 3)
        rows.append({"date": start + timedelta(days=i),
                     "pm25": round(float(np.clip(pm25, 0, 500)), 2)})
    return pd.DataFrame(rows)


def send_predictions_and_actuals(df: pd.DataFrame):
    divider("Step 1+2 — Predictions + Actuals")
    print(f"  {'Day':<4} {'Date':<12} {'Predicted':>10} {'Actual':>8} {'Error':>8}")
    print(f"  {'─'*4} {'─'*12} {'─'*10} {'─'*8} {'─'*8}")

    for i in range(HISTORY_DAYS, len(df)):
        history = df.iloc[i - HISTORY_DAYS:i].copy()

        # /predict
        payload = {"history": [{"date": str(r["date"]), "pm25": float(r["pm25"])}
                                for _, r in history.iterrows()]}
        pred = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        pred.raise_for_status()
        pred = pred.json()

        # /actual
        actual_pm25 = float(df.iloc[i]["pm25"])
        actual = requests.post(
            f"{API_URL}/actual",
            json={"date": pred["prediction_date"], "pm25_actual": actual_pm25},
            timeout=10,
        )
        actual.raise_for_status()
        actual = actual.json()

        error = actual.get("absolute_error", "—")
        print(f"  {i - HISTORY_DAYS + 1:<4} {pred['prediction_date']:<12} "
              f"{pred['predicted_pm25']:>10.2f} {actual_pm25:>8.2f} {str(error):>8}")


def trigger_dag_and_poll():
    divider("Step 3 — Trigger pm25_pipeline DAG")

    # Ensure DAG is unpaused
    requests.patch(
        f"{AIRFLOW_URL}/api/v1/dags/pm25_pipeline",
        auth=AIRFLOW_AUTH,
        json={"is_paused": False},
        timeout=10,
    )

    run_id = f"mock_{int(time.time())}"
    resp = requests.post(
        f"{AIRFLOW_URL}/api/v1/dags/pm25_pipeline/dagRuns",
        auth=AIRFLOW_AUTH,
        json={"dag_run_id": run_id},
        timeout=10,
    )
    resp.raise_for_status()
    print(f"  DAG run ID : {run_id}")
    print(f"  Track in UI: {AIRFLOW_URL}/dags/pm25_pipeline/grid")

    divider("Step 4 — Polling DAG")
    url = f"{AIRFLOW_URL}/api/v1/dags/pm25_pipeline/dagRuns/{run_id}"
    while True:
        state = requests.get(url, auth=AIRFLOW_AUTH, timeout=10).json().get("state", "unknown")
        print(f"  State: {state}    ", end="\r")
        if state in ("success", "failed"):
            print(f"  State: {state}    ")
            return state
        time.sleep(15)


def print_summary():
    divider("Summary")
    try:
        df = pd.read_csv("results/monitoring_results.csv")
        last = df.iloc[-1]
        print(f"  Evaluated pairs  : {int(last['evaluated_pairs'])}")
        print(f"  MAE              : {last['mae']}  (threshold={last['mae_threshold']})  degraded={last['mae_degraded']}")
        print(f"  PSI              : {last['psi']}  (threshold={last['psi_threshold']})  status={last['psi_status']}  degraded={last['psi_degraded']}")
        print(f"  Retrain triggered: {last['needs_retraining']}")
    except Exception:
        print("  (monitoring_results.csv not found yet)")


def main(mode: str, days: int):
    print("\n" + "=" * 55)
    print(f"  Mock Pipeline  mode={mode}  days={days}")
    print("=" * 55)
    check_services()

    # Generate mock data
    df = generate_mock(mode, days)
    total_pairs = len(df) - HISTORY_DAYS
    print(f"\n  Generated {len(df)} rows → {total_pairs} prediction+actual pairs")
    print(f"  PM2.5 mean={df['pm25'].mean():.1f}  std={df['pm25'].std():.1f}  "
          f"min={df['pm25'].min():.1f}  max={df['pm25'].max():.1f}")

    if total_pairs < 7:
        print("\n  ERROR: need ≥ 22 days to get 7 pairs (15 history + 7). "
              "Use --days 25 or more.")
        sys.exit(1)

    # Step 1+2: send to API
    send_predictions_and_actuals(df)

    # Step 3+4: trigger DAG and poll
    state = trigger_dag_and_poll()

    # Summary
    print_summary()

    print("\n" + "=" * 55)
    if state == "success":
        print("  Pipeline complete.")
        print("  MLflow: http://localhost:5001")
        print(f"  Airflow: {AIRFLOW_URL}/dags/pm25_pipeline/grid")
    else:
        print("  DAG failed — check Airflow UI for details.")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",  choices=["normal", "degraded", "drift"], default="degraded")
    parser.add_argument("--days",  type=int, default=25)
    args = parser.parse_args()
    main(args.mode, args.days)

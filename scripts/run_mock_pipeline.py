"""
Run Mock Pipeline Test
======================
Reads mock CSV → calls /predict + /actual for each row → calls /retrain.

Usage:
    python scripts/run_mock_pipeline.py
    python scripts/run_mock_pipeline.py --file data/mock/mock_pm25.csv --threshold 6.0
"""

import argparse
import sys
import time
import requests
import pandas as pd

API_URL      = "http://localhost:8001"
AIRFLOW_URL  = "http://localhost:8080"
AIRFLOW_AUTH = ("admin", "admin")
HISTORY_DAYS = 15   # rows used as history to build features for each prediction


def check_services():
    try:
        assert requests.get(f"{API_URL}/health", timeout=5).json()["status"] == "ok"
        print(f"  API     : OK ({API_URL})")
    except Exception:
        print(f"  API     : NOT REACHABLE — run `docker compose up -d`")
        sys.exit(1)


def main(csv_file: str, threshold: float):
    print("\n" + "=" * 55)
    print("  Mock Pipeline Test")
    print("=" * 55)
    check_services()

    df = pd.read_csv(csv_file, parse_dates=["date"])
    print(f"\n  Loaded {len(df)} rows from {csv_file}")
    print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  PM2.5 mean={df['pm25'].mean():.1f}  std={df['pm25'].std():.1f}\n")

    if len(df) < HISTORY_DAYS + 1:
        print(f"  ERROR: need at least {HISTORY_DAYS + 1} rows (got {len(df)})")
        sys.exit(1)

    # ── Step 1+2: /predict then /actual for each row after the history window ──
    print("─" * 55)
    print("  Step 1+2 — Sending predictions + actuals")
    print("─" * 55)
    print(f"  {'Day':<4} {'Date':<12} {'Predicted':>10} {'Actual':>8} {'Error':>8}")
    print(f"  {'─'*4} {'─'*12} {'─'*10} {'─'*8} {'─'*8}")

    for i in range(HISTORY_DAYS, len(df)):
        history_df   = df.iloc[i - HISTORY_DAYS:i][["date", "pm25"]].copy()
        history_df["date"] = history_df["date"].dt.date

        # /predict
        payload = {
            "history": [
                {"date": str(r["date"]), "pm25": float(r["pm25"])}
                for _, r in history_df.iterrows()
            ]
        }
        pred_resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        pred_resp.raise_for_status()
        pred      = pred_resp.json()

        # /actual — use the real pm25 from mock CSV as the "actual"
        actual_pm25  = float(df.iloc[i]["pm25"])
        actual_resp  = requests.post(
            f"{API_URL}/actual",
            json={"date": pred["prediction_date"], "pm25_actual": actual_pm25},
            timeout=10,
        )
        actual_resp.raise_for_status()
        actual = actual_resp.json()

        error = actual.get("absolute_error", "—")
        print(f"  {i - HISTORY_DAYS + 1:<4} {pred['prediction_date']:<12} "
              f"{pred['predicted_pm25']:>10.2f} {actual_pm25:>8.2f} {str(error):>8}")

    # ── Step 3: /retrain ───────────────────────────────────────────────────────
    print("\n" + "─" * 55)
    print("  Step 3 — Calling /retrain")
    print("─" * 55)

    actual_pairs = len(df) - HISTORY_DAYS
    retrain_resp = requests.post(
        f"{API_URL}/retrain",
        json={"threshold": threshold, "min_pairs": actual_pairs},
        timeout=10,
    )
    retrain_resp.raise_for_status()
    result = retrain_resp.json()

    print(f"  Evaluated pairs  : {result['evaluated_pairs']}")
    print(f"  MAE              : {result['mae']}")
    print(f"  Threshold        : {result['threshold']}")
    print(f"  Retrain triggered: {result['retrain_triggered']}")
    print(f"  Reason           : {result['reason']}")

    # ── Step 4: Poll Airflow ───────────────────────────────────────────────────
    if result["retrain_triggered"]:
        dag_run_id = result["dag_run_id"]
        print(f"\n  DAG run ID : {dag_run_id}")
        print(f"  Track in UI: {AIRFLOW_URL}/dags/pm25_training_pipeline/grid")
        print("\n  Polling Airflow...")

        url = f"{AIRFLOW_URL}/api/v1/dags/pm25_training_pipeline/dagRuns/{dag_run_id}"
        while True:
            state = requests.get(url, auth=AIRFLOW_AUTH, timeout=10).json().get("state", "unknown")
            print(f"  State: {state}    ", end="\r")
            if state in ("success", "failed"):
                print(f"\n\n  Training {state.upper()}.")
                break
            time.sleep(20)

        if state == "success":
            print("\n" + "=" * 55)
            print("  Pipeline complete — model retrained")
            print(f"  MLflow : http://localhost:5001")
            print("=" * 55)
    else:
        print("\n" + "=" * 55)
        print("  Pipeline complete — model is healthy, no retrain needed")
        print("=" * 55)

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",      default="data/mock/mock_pm25.csv")
    parser.add_argument("--threshold", type=float, default=6.0)
    args = parser.parse_args()
    main(args.file, args.threshold)

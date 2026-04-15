# ── DISABLED ─────────────────────────────────────────────────────────────────
# Not in use — replaced by pm25_24h_training + pm25_24h_pipeline (PostgreSQL).
# Listed in dags/.airflowignore so Airflow will not load this file.
# ─────────────────────────────────────────────────────────────────────────────
"""
PM2.5 Unified Pipeline DAG
===========================
Single DAG that handles monitoring AND retraining in one flow.

Schedule: daily at 01:00 UTC (can also be triggered manually)

Task graph:
    export_data
        └── check_mae_and_psi  (BranchPythonOperator)
                ├── needs_retrain → trigger_training → done
                └── healthy       → done

Checks (on rolling 30-day window of prediction vs actual pairs):
  1. MAE  > threshold (default 6.0 µg/m³)   → retrain
  2. PSI  > threshold (default 0.2)          → retrain

XCom keys pushed by check_mae_and_psi:
  evaluated_pairs, mae, mae_degraded, psi, psi_status, psi_degraded, needs_retraining
"""

from datetime import datetime
from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.empty import EmptyOperator
import os

SRC         = "/app/src"
CONFIG_PATH = "/app/configs/config.yaml"
RESULTS_DIR = "/app/results"


# ── Task 1: Export — load & join prediction+actual logs ───────────────────────
def _export_data(**context):
    import sys
    import os
    import pandas as pd
    sys.path.insert(0, SRC)

    predictions_log = f"{RESULTS_DIR}/predictions_log.csv"
    actuals_log     = f"{RESULTS_DIR}/actuals_log.csv"

    if not os.path.exists(predictions_log) or not os.path.exists(actuals_log):
        print("[export] Logs not found — nothing to export yet")
        context["ti"].xcom_push(key="evaluated_pairs", value=0)
        return

    preds_df   = pd.read_csv(predictions_log,  parse_dates=["prediction_date"])
    actuals_df = pd.read_csv(actuals_log, parse_dates=["date"])
    preds_df   = preds_df.sort_values("created_at").drop_duplicates("prediction_date", keep="last")
    actuals_df = actuals_df.sort_values("recorded_at").drop_duplicates("date", keep="last")

    merged = preds_df.merge(
        actuals_df, left_on="prediction_date", right_on="date", how="inner"
    ).sort_values("prediction_date")

    n = len(merged)
    print(f"[export] {n} matched prediction+actual pairs found")
    context["ti"].xcom_push(key="evaluated_pairs", value=n)

    if n > 0:
        print(f"[export] Date range: {merged['prediction_date'].min().date()} "
              f"→ {merged['prediction_date'].max().date()}")
        print(f"[export] Predicted  — mean={merged['predicted_pm25'].mean():.2f}  "
              f"std={merged['predicted_pm25'].std():.2f}")
        print(f"[export] Actual     — mean={merged['pm25_actual'].mean():.2f}  "
              f"std={merged['pm25_actual'].std():.2f}")


# ── Task 2: Check MAE + PSI → branch ─────────────────────────────────────────
def _check_mae_and_psi(**context):
    import sys
    sys.path.insert(0, SRC)
    from data_loader import load_config
    import monitor

    monitor.PREDICTIONS_LOG = f"{RESULTS_DIR}/predictions_log.csv"
    monitor.ACTUALS_LOG     = f"{RESULTS_DIR}/actuals_log.csv"

    config = load_config(CONFIG_PATH)
    config.setdefault("monitoring", {})
    config["monitoring"]["predictions_log"] = f"{RESULTS_DIR}/predictions_log.csv"
    config["monitoring"]["actuals_log"]     = f"{RESULTS_DIR}/actuals_log.csv"
    config["monitoring"]["results_file"]    = f"{RESULTS_DIR}/monitoring_results.csv"

    result = monitor.run_monitoring(config)

    ti = context["ti"]
    ti.xcom_push(key="evaluated_pairs", value=result["evaluated_pairs"])
    ti.xcom_push(key="mae",             value=result["mae"])
    ti.xcom_push(key="mae_degraded",    value=result["mae_degraded"])
    ti.xcom_push(key="psi",             value=result["psi"])
    ti.xcom_push(key="psi_status",      value=result["psi_status"])
    ti.xcom_push(key="psi_degraded",    value=result["psi_degraded"])
    ti.xcom_push(key="needs_retraining",value=result["needs_retraining"])

    if result["needs_retraining"]:
        reasons = []
        if result["mae_degraded"]:
            reasons.append(f"MAE={result['mae']} > {result['mae_threshold']}")
        if result["psi_degraded"]:
            reasons.append(f"PSI={result['psi']} ({result['psi_status']}) > {result['psi_threshold']}")
        print(f"[check] Retraining needed — {', '.join(reasons)}")
        return "needs_retrain"

    print(f"[check] Model healthy — "
          f"MAE={result['mae']} PSI={result['psi']} pairs={result['evaluated_pairs']}")
    return "healthy"


# ── DAG ───────────────────────────────────────────────────────────────────────
with DAG(
    dag_id="pm25_pipeline",
    schedule="0 1 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["pm25", "monitoring", "retraining"],
) as dag:

    export_task = PythonOperator(
        task_id="export_data",
        python_callable=_export_data,
    )

    check_task = BranchPythonOperator(
        task_id="check_mae_and_psi",
        python_callable=_check_mae_and_psi,
    )

    def _clear_logs(**context):
        """Delete prediction+actual logs after retraining so stale degraded
        data doesn't re-trigger another retrain on the next monitoring run."""
        for path in [f"{RESULTS_DIR}/predictions_log.csv",
                     f"{RESULTS_DIR}/actuals_log.csv"]:
            if os.path.exists(path):
                os.remove(path)
                print(f"[clear_logs] Removed {path}")

    retrain_task = TriggerDagRunOperator(
        task_id="needs_retrain",
        trigger_dag_id="pm25_training_pipeline",
        wait_for_completion=True,
        reset_dag_run=True,
    )

    clear_logs_task = PythonOperator(
        task_id="clear_logs",
        python_callable=_clear_logs,
    )

    healthy_task = EmptyOperator(task_id="healthy")

    export_task >> check_task >> [retrain_task, healthy_task]
    retrain_task >> clear_logs_task

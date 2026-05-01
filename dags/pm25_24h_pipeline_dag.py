"""
PM2.5 T+24h Monitoring & Auto-retrain Pipeline DAG
====================================================
Daily health check for stations 56/57/58/59/61.
Queries PostgreSQL, runs active ONNX model, computes RMSE + PSI
on a rolling 14-day window (2 weekly cycles), then auto-triggers
pm25_24h_training for any station that is degraded.

Schedule: daily at 02:00 UTC

Task graph:
    check_station_56 ──┐
    check_station_57 ──┤
    check_station_58 ──┼── trigger_retraining
    check_station_59 ──┤
    check_station_61 ──┘

Thresholds (overridable via Airflow Variables):
    RMSE_THRESHOLD_24H  default 13.0  µg/m³
    PSI_THRESHOLD_24H   default 0.2

Trigger manually:
    No config needed — checks all stations automatically.
"""

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator

SRC         = "/app/src"
MODELS_DIR  = "/app/models"
RESULTS_DIR = "/app/results"

DEFAULT_DB_URL    = "postgresql://postgres:postgres@postgres:5432/pm25"
STATIONS          = [56, 57, 58, 59, 61]
FORECAST_HOUR     = 24
ROLLING_DAYS      = 14      # evaluation window (2 weekly cycles)
MIN_PAIRS         = 168     # minimum hours needed (7 days)
RMSE_THRESHOLD    = 13.0    # µg/m³  — retrain if exceeded
PSI_THRESHOLD     = 1     # retrain if exceeded


# ── Shared helpers ────────────────────────────────────────────────────────────
def _db_url():
    return os.environ.get("PM25_DB_URL", DEFAULT_DB_URL)


def _get_threshold(name, default):
    """Read threshold from Airflow Variable, fall back to default."""
    try:
        from airflow.models import Variable
        return float(Variable.get(name, default_var=default))
    except Exception:
        return default


def _load_hourly_from_pg(station_id, db_url, data_start):
    import pandas as pd
    import sqlalchemy
    from sqlalchemy import text

    engine = sqlalchemy.create_engine(db_url)
    query  = text("""
        SELECT timestamp AS datetime, pm25
        FROM   pm25_raw_hourly
        WHERE  station_id = :station_id
          AND  timestamp  >= :data_start
          AND  pm25       IS NOT NULL
        ORDER  BY timestamp
    """)
    with engine.connect() as conn:
        result = conn.execute(query, {"station_id": station_id, "data_start": data_start})
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    engine.dispose()

    if df.empty:
        return df

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(None)
    df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)

    full_range = pd.date_range(df["datetime"].min(), df["datetime"].max(), freq="h")
    df = (df.set_index("datetime")
            .reindex(full_range)
            .rename_axis("datetime")
            .reset_index())
    df["pm25"] = df["pm25"].ffill().bfill().clip(lower=0, upper=500)
    return df


def _build_features_24h(df):
    import pandas as pd
    df = df.copy().sort_values("datetime").reset_index(drop=True)
    feature_cols = []

    for lag in [1, 2, 3, 6, 12, 24]:
        col = f"pm25_lag_{lag}"
        df[col] = df["pm25"].shift(lag)
        feature_cols.append(col)

    shifted = df["pm25"].shift(1)
    for window in [6, 12, 24]:
        df[f"pm25_rolling_mean_{window}h"] = shifted.rolling(window).mean()
        df[f"pm25_rolling_std_{window}h"]  = shifted.rolling(window).std()
        feature_cols += [f"pm25_rolling_mean_{window}h", f"pm25_rolling_std_{window}h"]

    df["pm25_diff_1h"]  = df["pm25"].shift(1).diff(1)
    df["pm25_diff_24h"] = df["pm25"].shift(1).diff(24)
    feature_cols += ["pm25_diff_1h", "pm25_diff_24h"]

    df["hour"]        = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"]       = df["datetime"].dt.month
    df["day_of_year"] = df["datetime"].dt.dayofyear
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    feature_cols += ["hour", "day_of_week", "month", "day_of_year", "is_weekend"]

    target_col     = "pm25_h24"
    df[target_col] = df["pm25"].shift(-FORECAST_HOUR)

    df = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
    return df, feature_cols, target_col


def _compute_psi(predicted, actual, bins=10):
    import numpy as np
    breakpoints = np.percentile(
        np.concatenate([predicted, actual]),
        np.linspace(0, 100, bins + 1),
    )
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 3:
        return 0.0
    breakpoints[0]  = -np.inf
    breakpoints[-1] =  np.inf

    exp_pct = np.histogram(predicted, bins=breakpoints)[0] / len(predicted)
    act_pct = np.histogram(actual,    bins=breakpoints)[0] / len(actual)
    exp_pct = np.where(exp_pct == 0, 1e-4, exp_pct)
    act_pct = np.where(act_pct == 0, 1e-4, act_pct)
    return round(float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))), 4)


# ── Per-station health check ──────────────────────────────────────────────────
def _check_station_health(station_id, **context):
    import json
    import numpy as np
    import pandas as pd
    import onnxruntime as rt

    ti         = context["ti"]
    db_url     = _db_url()
    rmse_thresh = _get_threshold("RMSE_THRESHOLD_24H", RMSE_THRESHOLD)
    psi_thresh = _get_threshold("PSI_THRESHOLD_24H", PSI_THRESHOLD)

    print(f"\n{'─'*50}")
    print(f"  Checking station {station_id}")
    print(f"  RMSE threshold={rmse_thresh}  PSI threshold={psi_thresh}")

    # ── Check active model exists ──
    models_dir = f"{MODELS_DIR}/station_{station_id}_24h"
    registry   = f"{models_dir}/active_model.json"

    if not os.path.exists(registry):
        print(f"  [SKIP] No active model found for station {station_id}")
        ti.xcom_push(key=f"health_{station_id}", value={
            "station_id": station_id, "status": "no_model",
            "needs_retraining": False, "rmse": None, "mae": None, "psi": None,
        })
        return

    with open(registry) as f:
        model_info = json.load(f)

    onnx_path = f"{models_dir}/onnx/{model_info['onnx_file']}"
    if not os.path.exists(onnx_path):
        print(f"  [SKIP] ONNX file missing: {onnx_path}")
        ti.xcom_push(key=f"health_{station_id}", value={
            "station_id": station_id, "status": "missing_onnx",
            "needs_retraining": True, "rmse": None, "mae": None, "psi": None,
        })
        return

    # ── Query rolling window from PostgreSQL ──
    data_start = (datetime.utcnow() - timedelta(days=ROLLING_DAYS + 5)).strftime("%Y-%m-%d")
    hourly = _load_hourly_from_pg(station_id, db_url, data_start)

    if hourly.empty or len(hourly) < MIN_PAIRS + FORECAST_HOUR:
        print(f"  [SKIP] Not enough data for station {station_id}: {len(hourly)} rows")
        ti.xcom_push(key=f"health_{station_id}", value={
            "station_id": station_id, "status": "insufficient_data",
            "needs_retraining": False, "rmse": None, "mae": None, "psi": None,
        })
        return

    # ── Build features ──
    feat_df, feature_cols, target_col = _build_features_24h(hourly)

    # Keep only rolling window (last ROLLING_DAYS days)
    cutoff = feat_df["datetime"].max() - pd.Timedelta(days=ROLLING_DAYS)
    window_df = feat_df[feat_df["datetime"] >= cutoff].reset_index(drop=True)

    if len(window_df) < MIN_PAIRS:
        print(f"  [SKIP] Only {len(window_df)} rows in rolling window (need {MIN_PAIRS})")
        ti.xcom_push(key=f"health_{station_id}", value={
            "station_id": station_id, "status": "insufficient_window",
            "needs_retraining": False, "rmse": None, "mae": None, "psi": None,
        })
        return

    X = window_df[feature_cols].values.astype("float32")
    y_actual = window_df[target_col].values

    # ── Run ONNX model ──
    sess     = rt.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name  = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    if model_info.get("input_shape") == "3d":
        n = model_info.get("n_features", X.shape[1])
        X_in = X.reshape(-1, 1, n)
    else:
        X_in = X

    y_pred = sess.run([out_name], {in_name: X_in})[0].flatten()

    if y_pred.shape[0] != y_actual.shape[0]:
        print(f"  [WARN] Shape mismatch pred={y_pred.shape} actual={y_actual.shape} — skipping")
        ti.xcom_push(key=f"health_{station_id}", value={
            "station_id": station_id, "status": "shape_mismatch",
            "needs_retraining": True, "rmse": None, "mae": None, "psi": None,
        })
        return

    # ── Compute RMSE + MAE + PSI ──
    rmse = round(float(np.sqrt(np.mean((y_actual - y_pred) ** 2))), 4)
    mae = round(float(np.mean(np.abs(y_actual - y_pred))), 4)
    psi = _compute_psi(y_pred, y_actual)

    rmse_degraded = rmse > rmse_thresh
    psi_degraded = psi > psi_thresh
    needs_retrain = rmse_degraded or psi_degraded

    rmse_status = "DEGRADED" if rmse_degraded else "OK"
    psi_label  = "stable" if psi < 0.1 else ("moderate" if psi < 0.2 else "significant")
    psi_status = "DEGRADED" if psi_degraded else "OK"

    print(f"  RMSE={rmse:.4f}  threshold={rmse_thresh}  [{rmse_status}]")
    print(f"  MAE={mae:.4f}  (secondary metric)")
    print(f"  PSI={psi:.4f}  threshold={psi_thresh}  status={psi_label}  [{psi_status}]")
    print(f"  Pairs={len(window_df)}  needs_retraining={needs_retrain}")

    result = {
        "station_id":       station_id,
        "status":           "degraded" if needs_retrain else "healthy",
        "needs_retraining": needs_retrain,
        "rmse":             rmse,
        "rmse_threshold":   rmse_thresh,
        "rmse_degraded":    rmse_degraded,
        "mae":              mae,
        "psi":              psi,
        "psi_threshold":    psi_thresh,
        "psi_status":       psi_label,
        "psi_degraded":     psi_degraded,
        "evaluated_pairs":  len(window_df),
        "model_key":        model_info.get("model_key"),
        "train_end":        model_info.get("train_end"),
        "checked_at":       datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
    }

    # Append to monitoring log
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = f"{RESULTS_DIR}/monitoring_24h_results.csv"
    import pandas as pd
    exists = os.path.exists(report_path)
    pd.DataFrame([result]).to_csv(report_path, mode="a", header=not exists, index=False)

    ti.xcom_push(key=f"health_{station_id}", value=result)


# ── Trigger retraining for degraded stations ──────────────────────────────────
def _trigger_retraining(**context):
    from airflow.api.common.trigger_dag import trigger_dag

    ti = context["ti"]
    degraded = []

    for sid in STATIONS:
        health = ti.xcom_pull(key=f"health_{sid}")
        if health and health.get("needs_retraining"):
            degraded.append(sid)
            print(f"  Station {sid} DEGRADED  RMSE={health.get('rmse')}  MAE={health.get('mae')}  PSI={health.get('psi')}")

    if not degraded:
        print("  All stations healthy — no retraining needed.")
        return

    print(f"\n  Triggering pm25_24h_training for stations: {degraded}")
    run_ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    for sid in degraded:
        run_id = f"monitor_retrain_s{sid}_{run_ts}"
        trigger_dag(
            dag_id="pm25_24h_training",
            run_id=run_id,
            conf={"station_id": sid},
            replace_microseconds=False,
        )
        print(f"  Triggered run_id={run_id} for station {sid}")


# ── Summary ───────────────────────────────────────────────────────────────────
def _print_summary(**context):
    ti = context["ti"]
    print(f"\n{'═'*65}")
    print("  T+24h MONITORING SUMMARY")
    print(f"{'═'*65}")
    print(f"  {'Station':<10} {'Status':<12} {'RMSE':>7} {'MAE':>7} {'PSI':>7} {'Model':<20}")
    print(f"  {'─'*65}")
    for sid in STATIONS:
        h = ti.xcom_pull(key=f"health_{sid}") or {}
        rmse  = f"{h['rmse']:.4f}"  if h.get("rmse")  is not None else "N/A"
        mae   = f"{h['mae']:.4f}"   if h.get("mae")   is not None else "N/A"
        psi   = f"{h['psi']:.4f}"   if h.get("psi")   is not None else "N/A"
        model = h.get("model_key", "N/A")
        status= h.get("status", "unknown").upper()
        print(f"  {sid:<10} {status:<12} {rmse:>7} {mae:>7} {psi:>7} {model:<20}")


# ── DAG definition ────────────────────────────────────────────────────────────
with DAG(
    dag_id="pm25_24h_pipeline",
    schedule="0 2 * * *",        # daily at 02:00 UTC
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args={
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["pm25", "monitoring", "24h-forecast", "postgresql"],
) as dag:

    start = EmptyOperator(task_id="start")

    # One health-check task per station (run in parallel)
    check_tasks = []
    for sid in STATIONS:
        t = PythonOperator(
            task_id=f"check_station_{sid}",
            python_callable=_check_station_health,
            op_kwargs={"station_id": sid},
        )
        check_tasks.append(t)

    trigger_task = PythonOperator(
        task_id="trigger_retraining",
        python_callable=_trigger_retraining,
    )

    summary_task = PythonOperator(
        task_id="summary",
        python_callable=_print_summary,
        trigger_rule="all_done",   # run even if some checks fail
    )

    start >> check_tasks >> trigger_task >> summary_task

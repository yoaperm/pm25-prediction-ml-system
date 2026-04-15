"""
Multi-Station Training Script
==============================
Train and deploy PM2.5 models for stations 63-67.

Data  : data/raw/archived/station_{id}_long.csv  (long format, hourly)
Train : 2020-01-01 → 2024-12-31
Test  : 2025-01-01 → latest

Per station output:
  models/station_{id}/                         joblib + ONNX
  models/station_{id}/active_model.json        current model pointer
  triton_model_repo/pm25_{id}/                 Triton model
  results/multi_station_comparison.csv         deployment history

Usage:
    PYTHONPATH=src python scripts/train_multi_station.py
    PYTHONPATH=src python scripts/train_multi_station.py --stations 63 65
"""

import argparse
import inspect
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime

import mlflow
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluate import evaluate_model, print_metrics
from feature_engineering import build_features, get_feature_columns
from lstm_model import train_lstm_with_tuning
from preprocessing import preprocess_pipeline
from train import train_baseline, train_with_tuning, _setup_mlflow
from triton_utils import publish_to_triton

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT, "data", "raw", "archived")
MODELS_DIR = os.path.join(ROOT, "models")
RESULTS_DIR= os.path.join(ROOT, "results")
TRITON_REPO= os.path.join(ROOT, "triton_model_repo")

TRAIN_END  = "2024-12-31"
TEST_START = "2025-01-01"

LAG_DAYS        = [1, 2, 3, 5, 7]
ROLLING_WINDOWS = [3, 7, 14]
RANDOM_STATE    = 42
VAL_START       = "2024-08-01"


# ── Data loading ──────────────────────────────────────────────────────────────
def load_station_daily(station_id: int) -> pd.DataFrame:
    """
    Load long-format CSV, filter PM2.5, aggregate hourly → daily mean.
    Returns DataFrame with columns [date, pm25].
    """
    path = os.path.join(DATA_DIR, f"station_{station_id}_long.csv")
    df   = pd.read_csv(path, parse_dates=["Date_Time"])
    pm   = df[df["Parameter"] == "PM2.5"].copy()
    pm["date"] = pm["Date_Time"].dt.normalize()
    daily = (
        pm.groupby("date")["Value"]
        .mean()
        .reset_index()
        .rename(columns={"Value": "pm25"})
    )
    daily = daily[daily["pm25"] > 0].reset_index(drop=True)
    return daily


# ── Model save + ONNX export ──────────────────────────────────────────────────
def _station_models_dir(station_id: int) -> str:
    d = os.path.join(MODELS_DIR, f"station_{station_id}")
    os.makedirs(os.path.join(d, "onnx"), exist_ok=True)
    return d


def _load_active_model(station_id: int, X_test, y_test):
    """Load current production model via ONNX Runtime. Returns (None, mae, info) or (None, None, None)."""
    import onnxruntime as rt

    registry = os.path.join(_station_models_dir(station_id), "active_model.json")
    if not os.path.exists(registry):
        return None, None, None
    with open(registry) as f:
        info = json.load(f)
    onnx_path = os.path.join(_station_models_dir(station_id), "onnx", info["onnx_file"])
    if not os.path.exists(onnx_path):
        return None, None, None

    session     = rt.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name  = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    if info.get("is_lstm"):
        X_in = X_test.reshape(X_test.shape[0], 1, X_test.shape[1]).astype(np.float32)
    else:
        X_in = X_test.astype(np.float32)

    preds = session.run([output_name], {input_name: X_in})[0].flatten()
    mae   = evaluate_model(y_test, preds)["MAE"]
    return None, mae, info


def _save_station_model(station_id: int, model, key: str,
                        train_start: str, train_end: str,
                        is_lstm: bool, old_info: dict) -> str:
    """
    Save the best model with date range in filename, export to ONNX,
    publish to Triton, update active_model.json.
    Previous versioned files are kept.
    """
    from export_onnx import export_lstm as export_lstm_onnx
    from export_onnx import export_sklearn, export_xgboost

    models_dir = _station_models_dir(station_id)
    onnx_dir   = os.path.join(models_dir, "onnx")

    onnx_filename = f"{key}_{train_start}_{train_end}.onnx"
    onnx_path     = os.path.join(onnx_dir, onnx_filename)

    # Export ONNX
    print(f"    Exporting ONNX → {onnx_path}")
    if is_lstm:
        export_lstm_onnx(model, onnx_dir, output_path=onnx_path)
    elif key == "xgboost":
        export_xgboost(model, onnx_dir, output_path=onnx_path)
    else:
        export_sklearn(model, key, onnx_dir, output_path=onnx_path)

    info = {
        "onnx_file":   onnx_filename,
        "model_key":   key,
        "station_id":  station_id,
        "train_start": train_start,
        "train_end":   train_end,
        "is_lstm":     is_lstm,
    }

    # Publish to Triton
    triton_model_name = f"pm25_{station_id}"
    station_triton    = os.path.join(TRITON_REPO, triton_model_name)
    # Temporarily set the model name for triton_utils
    import triton_utils as _tu
    orig = _tu.TRITON_MODEL_NAME
    _tu.TRITON_MODEL_NAME = triton_model_name
    publish_to_triton(onnx_path, TRITON_REPO, is_lstm)
    _tu.TRITON_MODEL_NAME = orig

    with open(os.path.join(models_dir, "active_model.json"), "w") as f:
        json.dump(info, f, indent=2)

    return onnx_path


# ── Per-station training ──────────────────────────────────────────────────────
def train_station(station_id: int, config: dict):
    print(f"\n{'═'*60}")
    print(f"  STATION {station_id}")
    print(f"{'═'*60}")

    # ---- Load data ----
    daily = load_station_daily(station_id)
    daily["date"] = pd.to_datetime(daily["date"])

    train_df = daily[daily["date"] <= TRAIN_END].copy()
    test_df  = daily[daily["date"] >= TEST_START].copy()

    if len(train_df) < 100 or len(test_df) < 30:
        print(f"  Skipping station {station_id}: not enough data (train={len(train_df)}, test={len(test_df)})")
        return None

    print(f"  Train: {len(train_df)} days  ({train_df['date'].min().date()} → {train_df['date'].max().date()})")
    print(f"  Test : {len(test_df)} days  ({test_df['date'].min().date()} → {test_df['date'].max().date()})")

    # ---- Preprocess + feature engineering ----
    train_clean = preprocess_pipeline(train_df)
    test_clean  = preprocess_pipeline(test_df)
    train_feat  = build_features(train_clean, LAG_DAYS, ROLLING_WINDOWS)
    test_feat   = build_features(test_clean,  LAG_DAYS, ROLLING_WINDOWS)
    feature_cols = get_feature_columns(train_feat)

    X_train = train_feat[feature_cols].values
    y_train = train_feat["pm25"].values
    X_test  = test_feat[feature_cols].values
    y_test  = test_feat["pm25"].values

    train_start = train_feat["date"].min().strftime("%Y-%m-%d")
    train_end   = train_feat["date"].max().strftime("%Y-%m-%d")
    print(f"  Feature range: {train_start} → {train_end}  ({len(feature_cols)} features)")

    # ---- Train 5 models in memory ----
    trained = {}   # key -> (display_name, model, metrics, is_lstm)
    results = []
    exp_name = f"pm25_station_{station_id}"
    mlflow.set_experiment(exp_name)

    # 1. Baseline
    with mlflow.start_run(run_name="LinearRegression"):
        m_obj, _ = train_baseline(X_train, y_train, config)
        m = evaluate_model(y_test, m_obj.predict(X_test))
        mlflow.log_params({"model_type": "LinearRegression", "station": station_id})
        mlflow.log_metrics(m)
        trained["baseline_linear_regression"] = ("Linear Regression", m_obj, m, False)
        results.append({"model": "Linear Regression", **m})

    # 2. Ridge
    with mlflow.start_run(run_name="Ridge"):
        m_obj, best = train_with_tuning("Ridge", config["models"]["ridge"]["params"],
                                        X_train, y_train, RANDOM_STATE)
        m = evaluate_model(y_test, m_obj.predict(X_test))
        mlflow.log_params({**best, "station": station_id}); mlflow.log_metrics(m)
        trained["ridge_regression"] = ("Ridge Regression", m_obj, m, False)
        results.append({"model": "Ridge Regression", **m})

    # 3. Random Forest
    rf_params = config["models"]["random_forest"]["params"].copy()
    rf_params.pop("random_state", None)
    with mlflow.start_run(run_name="RandomForest"):
        m_obj, best = train_with_tuning("RandomForestRegressor", rf_params,
                                        X_train, y_train, RANDOM_STATE)
        m = evaluate_model(y_test, m_obj.predict(X_test))
        mlflow.log_params({**best, "station": station_id}); mlflow.log_metrics(m)
        trained["random_forest"] = ("Random Forest", m_obj, m, False)
        results.append({"model": "Random Forest", **m})

    # 4. XGBoost
    xgb_params = config["models"]["xgboost"]["params"].copy()
    xgb_params.pop("random_state", None)
    with mlflow.start_run(run_name="XGBoost"):
        m_obj, best = train_with_tuning("XGBRegressor", xgb_params,
                                        X_train, y_train, RANDOM_STATE)
        m = evaluate_model(y_test, m_obj.predict(X_test))
        mlflow.log_params({**best, "station": station_id}); mlflow.log_metrics(m)
        trained["xgboost"] = ("XGBoost", m_obj, m, False)
        results.append({"model": "XGBoost", **m})

    # 5. LSTM
    with mlflow.start_run(run_name="LSTM"):
        lstm_obj, best = train_lstm_with_tuning(X_train, y_train,
                                                config["models"]["lstm"]["params"],
                                                RANDOM_STATE)
        X_test_3d = X_test.reshape(X_test.shape[0], 1, X_test.shape[1]).astype(np.float32)
        m = evaluate_model(y_test, lstm_obj.predict(X_test_3d).flatten())
        mlflow.log_params({**best, "station": station_id}); mlflow.log_metrics(m)
        trained["lstm"] = ("LSTM", lstm_obj, m, True)
        results.append({"model": "LSTM", **m})

    # ---- Summary ----
    results_df = pd.DataFrame(results)
    print(f"\n  {'Model':<30} {'MAE':>7} {'RMSE':>7} {'R2':>7}")
    print(f"  {'─'*52}")
    for _, row in results_df.iterrows():
        print(f"  {row['model']:<30} {row['MAE']:>7.4f} {row['RMSE']:>7.4f} {row['R2']:>7.4f}")

    # ---- Find best ----
    best_key                              = min(trained, key=lambda k: trained[k][2]["MAE"])
    best_name, best_model, best_m, is_lstm = trained[best_key]
    new_mae = best_m["MAE"]
    print(f"\n  Best new model: {best_name}  MAE={new_mae:.4f}")

    # ---- Compare vs production ----
    _, prod_mae, old_info = _load_active_model(station_id, X_test, y_test)
    if prod_mae is None:
        prod_str = "N/A (first deploy)"
    else:
        prod_str = f"MAE={prod_mae:.4f}  ({old_info['train_start']} → {old_info['train_end']})"

    print(f"  Prod: {prod_str}")

    if prod_mae is None or new_mae < prod_mae:
        onnx_path = _save_station_model(station_id, best_model, best_key,
                                        train_start, train_end, is_lstm, old_info)
        status = "DEPLOYED"
        delta  = round(prod_mae - new_mae, 4) if prod_mae is not None else None
        print(f"  → DEPLOYED  {onnx_path}")
    else:
        status = "NOT_DEPLOYED"
        delta  = round(prod_mae - new_mae, 4)
        print(f"  → NOT DEPLOYED  new {new_mae:.4f} >= prod {prod_mae:.4f}")

    # ---- Save feature columns ----
    with open(os.path.join(_station_models_dir(station_id), "feature_columns.json"), "w") as f:
        json.dump(feature_cols, f)

    # ---- Append comparison report ----
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(RESULTS_DIR, "multi_station_comparison.csv")
    row = {
        "station_id":  station_id,
        "train_start": train_start,
        "train_end":   train_end,
        "best_model":  best_name,
        "new_mae":     new_mae,
        "new_rmse":    best_m["RMSE"],
        "new_r2":      best_m["R2"],
        "prod_mae":    prod_mae,
        "mae_delta":   delta,
        "status":      status,
    }
    exists = os.path.exists(report_path)
    pd.DataFrame([row]).to_csv(report_path, mode="a", header=not exists, index=False)

    return results_df


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stations", nargs="+", type=int, default=[63, 64, 65, 66, 67])
    args = parser.parse_args()

    # Load config for model hyperparameters
    from data_loader import load_config
    config = load_config(os.path.join(ROOT, "configs", "config.yaml"))
    _setup_mlflow(config)

    all_results = {}
    for sid in args.stations:
        df = train_station(sid, config)
        if df is not None:
            all_results[sid] = df

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("MULTI-STATION DEPLOYMENT SUMMARY")
    print(f"{'═'*60}")
    report = pd.read_csv(os.path.join(RESULTS_DIR, "multi_station_comparison.csv"))
    latest = report.sort_values("train_start").groupby("station_id").tail(1)
    print(latest[["station_id", "best_model", "new_mae", "prod_mae", "status"]].to_string(index=False))
    print(f"\nTriton models: {[f'pm25_{s}' for s in args.stations]}")
    print(f"Report: {os.path.join(RESULTS_DIR, 'multi_station_comparison.csv')}")

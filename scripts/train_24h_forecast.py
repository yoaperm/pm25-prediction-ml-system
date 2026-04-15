"""
24-Hour PM2.5 Forecasting Script
==================================
Trains models to predict the next 24 hours of PM2.5 simultaneously
using hourly data from monitoring stations.

Data   : data/raw/archived/station_{id}_long.csv
Output : models/station_{id}_24h/onnx/{model}_{start}_{end}.onnx
         models/station_{id}_24h/active_model.json
         results/forecast_24h_results.csv

Usage:
    PYTHONPATH=src python scripts/train_24h_forecast.py
    PYTHONPATH=src python scripts/train_24h_forecast.py --stations 63 65
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import mlflow
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluate import evaluate_model

# ── Constants ─────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(ROOT, "data", "raw", "archived")
MODELS_DIR  = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")

TRAIN_END       = "2024-12-31"
TEST_START      = "2025-01-01"
LOOKBACK_HOURS  = 24       # hours of history used as input features
FORECAST_HOURS  = 24       # hours ahead to predict simultaneously
RANDOM_STATE    = 42

# ── Preprocessing ──────────────────────────────────────────────────────────────
def load_station_hourly(station_id: int) -> pd.DataFrame:
    """
    Load long-format CSV, filter PM2.5, fill missing hours with forward fill.
    Returns DataFrame with columns [datetime, pm25] at hourly frequency.
    """
    path = os.path.join(DATA_DIR, f"station_{station_id}_long.csv")
    df   = pd.read_csv(path, parse_dates=["Date_Time"])
    pm   = df[df["Parameter"] == "PM2.5"][["Date_Time", "Value"]].copy()
    pm   = pm.rename(columns={"Date_Time": "datetime", "Value": "pm25"})
    pm   = pm.sort_values("datetime").drop_duplicates("datetime")

    # Reindex to complete hourly range — fill gaps with forward fill then back fill
    full_range = pd.date_range(pm["datetime"].min(), pm["datetime"].max(), freq="h")
    pm = pm.set_index("datetime").reindex(full_range).rename_axis("datetime").reset_index()
    pm["pm25"] = pm["pm25"].fillna(method="ffill").fillna(method="bfill")

    # Clip to valid range
    pm["pm25"] = pm["pm25"].clip(lower=0, upper=500)

    return pm


# ── Feature Engineering ────────────────────────────────────────────────────────
def build_features_24h(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Build sliding-window features for 24-hour ahead forecasting.

    Features (from t-LOOKBACK to t, no leakage):
      - PM2.5 lags: lag_1 to lag_48 (past 48 hours)
      - Rolling mean/std: 6h, 12h, 24h (computed on shifted series)
      - Time features: hour, day_of_week, month, day_of_year, is_weekend
      - Diff features: pm25_diff_1, pm25_diff_24 (hour-over-hour, day-over-day)

    Targets:
      - pm25_h1 to pm25_h24 (next 24 hours)

    All features use shift(1) → no leakage into future targets.
    """
    df = df.copy().sort_values("datetime").reset_index(drop=True)

    feature_cols = []
    target_cols  = []

    # Lag features: key hours rather than every hour to reduce dimensionality
    for lag in [1, 2, 3, 6, 12, 24]:
        col = f"pm25_lag_{lag}"
        df[col] = df["pm25"].shift(lag)
        feature_cols.append(col)

    # Rolling statistics on lagged series (shift(1) prevents leakage)
    shifted = df["pm25"].shift(1)
    for window in [6, 12, 24]:
        col_mean = f"pm25_rolling_mean_{window}h"
        col_std  = f"pm25_rolling_std_{window}h"
        df[col_mean] = shifted.rolling(window).mean()
        df[col_std]  = shifted.rolling(window).std()
        feature_cols += [col_mean, col_std]

    # Diff features
    df["pm25_diff_1h"]  = df["pm25"].shift(1).diff(1)
    df["pm25_diff_24h"] = df["pm25"].shift(1).diff(24)
    feature_cols += ["pm25_diff_1h", "pm25_diff_24h"]

    # Time features
    df["hour"]        = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"]       = df["datetime"].dt.month
    df["day_of_year"] = df["datetime"].dt.dayofyear
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    feature_cols += ["hour", "day_of_week", "month", "day_of_year", "is_weekend"]

    # Targets: pm25 for next 1..FORECAST_HOURS hours
    for h in range(1, FORECAST_HOURS + 1):
        col = f"pm25_h{h}"
        df[col] = df["pm25"].shift(-h)
        target_cols.append(col)

    df = df.dropna(subset=feature_cols + target_cols).reset_index(drop=True)
    return df, feature_cols, target_cols


# ── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate_24h(y_true: np.ndarray, y_pred: np.ndarray,
                 target_cols: list[str]) -> dict:
    """
    Evaluate 24-hour forecast.
    Returns overall metrics + per-horizon MAE array.
    """
    mae_per_h  = np.mean(np.abs(y_true - y_pred), axis=0)
    rmse_per_h = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))

    overall_mae  = float(np.mean(mae_per_h))
    overall_rmse = float(np.mean(rmse_per_h))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    overall_r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "MAE":         overall_mae,
        "RMSE":        overall_rmse,
        "R2":          overall_r2,
        "mae_per_h":   mae_per_h.tolist(),
        "rmse_per_h":  rmse_per_h.tolist(),
    }


# ── ONNX Export ────────────────────────────────────────────────────────────────
def export_onnx_24h(model, model_key: str, n_features: int,
                    n_targets: int, output_path: str, is_lstm: bool = False):
    """Export a 24-output model to ONNX."""
    if is_lstm:
        import torch
        pytorch_model = model.module_
        pytorch_model.eval()
        dummy = torch.zeros(1, 1, n_features)
        torch.onnx.export(
            pytorch_model, dummy, output_path,
            input_names=["lstm_input"],
            output_names=["forecast_24h"],
            dynamic_axes={"lstm_input": {0: "batch"}, "forecast_24h": {0: "batch"}},
            opset_version=17,
        )
    elif model_key == "xgboost":
        # Native multi-output XGBRegressor: export directly via onnxmltools
        from onnxmltools.convert import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType as OFT
        init_type  = [("float_input", OFT([None, n_features]))]
        onnx_model = convert_xgboost(model, initial_types=init_type)
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
    else:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        init_type  = [("float_input", FloatTensorType([None, n_features]))]
        onnx_model = convert_sklearn(model, initial_types=init_type, target_opset=17)
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

    print(f"  ONNX saved → {output_path}")


# ── Training ───────────────────────────────────────────────────────────────────
def train_station_24h(station_id: int):
    print(f"\n{'═'*60}")
    print(f"  STATION {station_id}  —  24-Hour Forecast")
    print(f"{'═'*60}")

    # ── Load & preprocess ──
    hourly = load_station_hourly(station_id)
    print(f"  Hourly rows: {len(hourly)}  "
          f"({hourly['datetime'].min().date()} → {hourly['datetime'].max().date()})")

    # ── Feature engineering ──
    feat_df, feature_cols, target_cols = build_features_24h(hourly)
    n_features = len(feature_cols)
    print(f"  Features: {n_features}  |  Targets: {len(target_cols)} (h+1 … h+24)")

    # ── Train / test split ──
    train_df = feat_df[feat_df["datetime"] <= TRAIN_END]
    test_df  = feat_df[feat_df["datetime"] >= TEST_START]

    if len(train_df) < 500 or len(test_df) < 48:
        print(f"  Skipping: not enough data (train={len(train_df)}, test={len(test_df)})")
        return None

    X_train = train_df[feature_cols].values.astype(np.float32)
    Y_train = train_df[target_cols].values.astype(np.float32)
    X_test  = test_df[feature_cols].values.astype(np.float32)
    Y_test  = test_df[target_cols].values.astype(np.float32)

    train_start = train_df["datetime"].min().strftime("%Y-%m-%d")
    train_end   = train_df["datetime"].max().strftime("%Y-%m-%d")

    print(f"  Train: {X_train.shape}  Test: {X_test.shape}")
    print(f"  Date range: {train_start} → {train_end}")

    # 3-D version for LSTM inference (defined here so production comparison can use it)
    X_test_3d = X_test.reshape(X_test.shape[0], 1, n_features)

    mlflow.set_experiment(f"pm25_24h_station_{station_id}")

    tscv    = TimeSeriesSplit(n_splits=3)
    n_jobs  = int(os.environ.get("GRID_N_JOBS", "-1"))
    trained = {}   # key → (name, model, metrics, is_lstm)
    results = []

    # ── 1. Linear Regression ──
    with mlflow.start_run(run_name="LinearRegression_24h"):
        model = LinearRegression()
        model.fit(X_train, Y_train)
        preds   = model.predict(X_test)
        metrics = evaluate_24h(Y_test, preds, target_cols)
        mlflow.log_params({"model": "LinearRegression", "station": station_id,
                           "forecast_hours": FORECAST_HOURS})
        mlflow.log_metrics({"MAE": metrics["MAE"], "RMSE": metrics["RMSE"], "R2": metrics["R2"]})
        trained["linear_regression"] = ("Linear Regression", model, metrics, False)
        results.append({"model": "Linear Regression", **{k: v for k, v in metrics.items()
                                                          if k not in ("mae_per_h", "rmse_per_h")}})
        print(f"  Linear Regression  MAE={metrics['MAE']:.4f}")

    # ── 2. Ridge ──
    with mlflow.start_run(run_name="Ridge_24h"):
        grid = GridSearchCV(Ridge(), {"alpha": [0.1, 1.0, 10.0, 100.0]},
                            cv=tscv, scoring="neg_mean_absolute_error",
                            n_jobs=n_jobs, verbose=0)
        grid.fit(X_train, Y_train)
        model   = grid.best_estimator_
        preds   = model.predict(X_test)
        metrics = evaluate_24h(Y_test, preds, target_cols)
        mlflow.log_params({"best_alpha": grid.best_params_["alpha"], "station": station_id})
        mlflow.log_metrics({"MAE": metrics["MAE"], "RMSE": metrics["RMSE"], "R2": metrics["R2"]})
        trained["ridge_regression"] = ("Ridge Regression", model, metrics, False)
        results.append({"model": "Ridge Regression", **{k: v for k, v in metrics.items()
                                                         if k not in ("mae_per_h", "rmse_per_h")}})
        print(f"  Ridge Regression   MAE={metrics['MAE']:.4f}  alpha={grid.best_params_['alpha']}")

    # ── 3. Random Forest ──
    with mlflow.start_run(run_name="RandomForest_24h"):
        rf_grid = {
            "n_estimators":    [100],
            "max_depth":       [5, 10],
            "min_samples_leaf":[4],
        }
        grid = GridSearchCV(RandomForestRegressor(random_state=RANDOM_STATE),
                            rf_grid, cv=tscv, scoring="neg_mean_absolute_error",
                            n_jobs=n_jobs, verbose=0)
        grid.fit(X_train, Y_train)
        model   = grid.best_estimator_
        preds   = model.predict(X_test)
        metrics = evaluate_24h(Y_test, preds, target_cols)
        mlflow.log_params({**grid.best_params_, "station": station_id})
        mlflow.log_metrics({"MAE": metrics["MAE"], "RMSE": metrics["RMSE"], "R2": metrics["R2"]})
        trained["random_forest"] = ("Random Forest", model, metrics, False)
        results.append({"model": "Random Forest", **{k: v for k, v in metrics.items()
                                                      if k not in ("mae_per_h", "rmse_per_h")}})
        print(f"  Random Forest      MAE={metrics['MAE']:.4f}  params={grid.best_params_}")

    # ── 4. XGBoost (native multi-output) ──
    with mlflow.start_run(run_name="XGBoost_24h"):
        xgb_grid = {
            "n_estimators":  [100],
            "max_depth":     [3, 5],
            "learning_rate": [0.1],
        }
        xgb_base = XGBRegressor(random_state=RANDOM_STATE, verbosity=0, n_jobs=1)
        grid = GridSearchCV(xgb_base, xgb_grid, cv=tscv,
                            scoring="neg_mean_absolute_error",
                            n_jobs=n_jobs, verbose=0)
        grid.fit(X_train, Y_train)
        model   = grid.best_estimator_
        preds   = model.predict(X_test)
        metrics = evaluate_24h(Y_test, preds, target_cols)
        mlflow.log_params({**grid.best_params_, "station": station_id})
        mlflow.log_metrics({"MAE": metrics["MAE"], "RMSE": metrics["RMSE"], "R2": metrics["R2"]})
        trained["xgboost"] = ("XGBoost", model, metrics, False)
        results.append({"model": "XGBoost", **{k: v for k, v in metrics.items()
                                                if k not in ("mae_per_h", "rmse_per_h")}})
        print(f"  XGBoost            MAE={metrics['MAE']:.4f}  params={grid.best_params_}")

    # ── 5. LSTM (24 outputs) ──
    with mlflow.start_run(run_name="LSTM_24h"):
        import torch
        import torch.nn as nn
        from skorch import NeuralNetRegressor
        from sklearn.model_selection import RandomizedSearchCV

        class LSTMForecast(nn.Module):
            """LSTM → 24-hour output head."""
            def __init__(self, input_size=58, hidden_size=128,
                         num_layers=2, dropout=0.2, forecast_hours=24):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                    batch_first=True,
                                    dropout=dropout if num_layers > 1 else 0.0)
                self.fc   = nn.Linear(hidden_size, forecast_hours)

            def forward(self, x):
                # x: (batch, 1, input_size)
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])  # (batch, forecast_hours)

        device = "cpu" if os.environ.get("PYTORCH_DEVICE") == "cpu" else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        # Prevent OMP thread pool conflicts with XGBoost/sklearn parallel workers
        torch.set_num_threads(1)
        print(f"  LSTM device: {device}")

        # Subsample every 4th row for LSTM — consecutive rows are highly correlated,
        # so stride=4 still covers the distribution while cutting training time 4x
        stride = 4
        X_lstm = X_train[::stride]
        Y_lstm = Y_train[::stride].astype(np.float32)

        X_train_3d = X_lstm.reshape(X_lstm.shape[0], 1, n_features)

        net = NeuralNetRegressor(
            module=LSTMForecast,
            module__input_size=n_features,
            module__forecast_hours=FORECAST_HOURS,
            criterion=nn.L1Loss,
            optimizer=torch.optim.Adam,
            max_epochs=10,
            batch_size=512,
            train_split=None,
            device=device,
            verbose=1,
            iterator_train__num_workers=0,
            iterator_valid__num_workers=0,
        )

        # Fixed config — no grid search to keep training fast
        net.set_params(module__hidden_size=64, module__dropout=0.1, optimizer__lr=0.001)
        net.fit(X_train_3d, Y_lstm)
        lstm_model = net
        preds      = lstm_model.predict(X_test_3d)
        metrics    = evaluate_24h(Y_test, preds, target_cols)
        mlflow.log_params({"hidden_size": 64, "dropout": 0.1, "lr": 0.001, "station": station_id})
        mlflow.log_metrics({"MAE": metrics["MAE"], "RMSE": metrics["RMSE"], "R2": metrics["R2"]})
        trained["lstm"] = ("LSTM", lstm_model, metrics, True)
        results.append({"model": "LSTM", **{k: v for k, v in metrics.items()
                                             if k not in ("mae_per_h", "rmse_per_h")}})
        print(f"  LSTM               MAE={metrics['MAE']:.4f}")

    # ── Summary table ──
    results_df = pd.DataFrame(results)
    print(f"\n  {'Model':<25} {'MAE':>7} {'RMSE':>7} {'R2':>7}")
    print(f"  {'─'*46}")
    for _, row in results_df.iterrows():
        print(f"  {row['model']:<25} {row['MAE']:>7.4f} {row['RMSE']:>7.4f} {row['R2']:>7.4f}")

    # ── Per-horizon MAE of best model ──
    best_key  = min(trained, key=lambda k: trained[k][2]["MAE"])
    best_name, best_model, best_metrics, best_is_lstm = trained[best_key]
    print(f"\n  Best: {best_name}  MAE={best_metrics['MAE']:.4f}")
    mae_arr = best_metrics["mae_per_h"]
    print(f"\n  Per-horizon MAE (h+1 … h+24):")
    row_str = "  "
    for h, mae in enumerate(mae_arr, 1):
        row_str += f"h{h:02d}:{mae:.2f}  "
        if h % 6 == 0:
            print(row_str.rstrip())
            row_str = "  "

    # ── Load production model for comparison ──
    models_dir = os.path.join(MODELS_DIR, f"station_{station_id}_24h")
    os.makedirs(os.path.join(models_dir, "onnx"), exist_ok=True)
    registry   = os.path.join(models_dir, "active_model.json")

    prod_mae  = None
    old_info  = None
    if os.path.exists(registry):
        with open(registry) as f:
            old_info = json.load(f)
        onnx_prod = os.path.join(models_dir, "onnx", old_info["onnx_file"])
        if os.path.exists(onnx_prod):
            import onnxruntime as rt
            sess    = rt.InferenceSession(onnx_prod, providers=["CPUExecutionProvider"])
            in_name = sess.get_inputs()[0].name
            out_name= sess.get_outputs()[0].name
            X_in    = X_test_3d if old_info.get("is_lstm") else X_test
            preds_prod = sess.run([out_name], {in_name: X_in})[0]
            prod_mae   = evaluate_24h(Y_test, preds_prod, target_cols)["MAE"]

    prod_str = (f"MAE={prod_mae:.4f}  ({old_info['train_start']} → {old_info['train_end']})"
                if prod_mae is not None else "N/A (first deploy)")
    print(f"\n  Prod: {prod_str}")

    # ── Deploy if better ──
    new_mae = best_metrics["MAE"]
    if prod_mae is None or new_mae < prod_mae:
        onnx_filename = f"{best_key}_{train_start}_{train_end}.onnx"
        onnx_path     = os.path.join(models_dir, "onnx", onnx_filename)
        export_onnx_24h(best_model, best_key, n_features,
                        FORECAST_HOURS, onnx_path, best_is_lstm)

        info = {
            "onnx_file":      onnx_filename,
            "model_key":      best_key,
            "station_id":     station_id,
            "train_start":    train_start,
            "train_end":      train_end,
            "is_lstm":        best_is_lstm,
            "forecast_hours": FORECAST_HOURS,
            "n_features":     n_features,
        }
        with open(registry, "w") as f:
            json.dump(info, f, indent=2)

        with open(os.path.join(models_dir, "feature_columns.json"), "w") as f:
            json.dump(feature_cols, f)

        status = "DEPLOYED"
        delta  = round(prod_mae - new_mae, 4) if prod_mae is not None else None
        print(f"  → DEPLOYED  {onnx_path}")
    else:
        status = "NOT_DEPLOYED"
        delta  = round(prod_mae - new_mae, 4)
        print(f"  → NOT DEPLOYED  new {new_mae:.4f} >= prod {prod_mae:.4f}")

    # ── Save per-horizon breakdown ──
    os.makedirs(RESULTS_DIR, exist_ok=True)
    horizon_rows = []
    for key, (name, _, m, _) in trained.items():
        for h, (mae_h, rmse_h) in enumerate(zip(m["mae_per_h"], m["rmse_per_h"]), 1):
            horizon_rows.append({
                "station_id": station_id,
                "model":      name,
                "horizon_h":  h,
                "MAE":        round(mae_h, 4),
                "RMSE":       round(rmse_h, 4),
            })
    horizon_df   = pd.DataFrame(horizon_rows)
    horizon_path = os.path.join(RESULTS_DIR, f"forecast_24h_horizon_station_{station_id}.csv")
    horizon_df.to_csv(horizon_path, index=False)

    # ── Append summary report ──
    report_path = os.path.join(RESULTS_DIR, "forecast_24h_results.csv")
    row = {
        "station_id":  station_id,
        "train_start": train_start,
        "train_end":   train_end,
        "best_model":  best_name,
        "new_mae":     new_mae,
        "prod_mae":    prod_mae,
        "mae_delta":   delta,
        "status":      status,
    }
    exists = os.path.exists(report_path)
    pd.DataFrame([row]).to_csv(report_path, mode="a", header=not exists, index=False)

    print(f"  Horizon breakdown → {horizon_path}")
    return results_df


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 24-hour PM2.5 forecasting models")
    parser.add_argument("--stations", nargs="+", type=int, default=[56, 57, 58, 59, 61])
    args = parser.parse_args()

    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    )

    for sid in args.stations:
        train_station_24h(sid)

    print(f"\n{'═'*60}")
    print("24-HOUR FORECAST DEPLOYMENT SUMMARY")
    print(f"{'═'*60}")
    report_path = os.path.join(RESULTS_DIR, "forecast_24h_results.csv")
    if os.path.exists(report_path):
        report = pd.read_csv(report_path)
        latest = report.sort_values("train_start").groupby("station_id").tail(1)
        print(latest[["station_id", "best_model", "new_mae", "prod_mae", "status"]].to_string(index=False))
    print(f"\nReport: {report_path}")

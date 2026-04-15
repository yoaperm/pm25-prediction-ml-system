"""
24-Hour Ahead PM2.5 Forecasting Script
=======================================
Trains models to predict PM2.5 exactly 24 hours ahead (single output).
Run every hour to get a rolling h+24 forecast.

Data   : data/raw/archived/station_{id}_long.csv
Output : models/station_{id}_24h/onnx/{model}_{start}_{end}.onnx
         models/station_{id}_24h/active_model.json
         results/forecast_24h_results.csv

Usage:
    PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_DEVICE=cpu \\
      python scripts/train_24h_forecast.py
    PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_DEVICE=cpu \\
      python scripts/train_24h_forecast.py --stations 63 65

Inference (every hour):
    Load active_model.json → run ONNX with current hour's 19 features → 1 value = PM2.5 at T+24
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
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluate import evaluate_model

# ── Constants ─────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(ROOT, "data", "raw", "archived")
MODELS_DIR  = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")

TRAIN_END      = "2024-12-31"
TEST_START     = "2025-01-01"
FORECAST_HOUR  = 24          # single horizon: predict T+24
RANDOM_STATE   = 42


# ── Data Loading ───────────────────────────────────────────────────────────────
def load_station_hourly(station_id: int) -> pd.DataFrame:
    """
    Load long-format CSV, filter PM2.5, fill missing hours.
    Returns DataFrame with columns [datetime, pm25] at hourly frequency.
    """
    path = os.path.join(DATA_DIR, f"station_{station_id}_long.csv")
    df   = pd.read_csv(path, parse_dates=["Date_Time"])
    pm   = df[df["Parameter"] == "PM2.5"][["Date_Time", "Value"]].copy()
    pm   = pm.rename(columns={"Date_Time": "datetime", "Value": "pm25"})
    pm   = pm.sort_values("datetime").drop_duplicates("datetime")

    full_range = pd.date_range(pm["datetime"].min(), pm["datetime"].max(), freq="h")
    pm = pm.set_index("datetime").reindex(full_range).rename_axis("datetime").reset_index()
    pm["pm25"] = pm["pm25"].ffill().bfill().clip(lower=0, upper=500)

    return pm


# ── Feature Engineering ────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], str]:
    """
    Build features for T+24 prediction.

    Input features (19, all shifted so no leakage):
      pm25_lag_1/2/3/6/12/24   — recent PM2.5 history
      pm25_rolling_mean/std_6/12/24h — smoothed trends
      pm25_diff_1h / pm25_diff_24h   — rate of change
      hour / day_of_week / month / day_of_year / is_weekend

    Target:
      pm25_h24  — PM2.5 exactly 24 hours ahead (shift(-24))
    """
    df = df.copy().sort_values("datetime").reset_index(drop=True)

    feature_cols = []

    # Lag features
    for lag in [1, 2, 3, 6, 12, 24]:
        col = f"pm25_lag_{lag}"
        df[col] = df["pm25"].shift(lag)
        feature_cols.append(col)

    # Rolling statistics (on shifted series — no leakage)
    shifted = df["pm25"].shift(1)
    for window in [6, 12, 24]:
        df[f"pm25_rolling_mean_{window}h"] = shifted.rolling(window).mean()
        df[f"pm25_rolling_std_{window}h"]  = shifted.rolling(window).std()
        feature_cols += [f"pm25_rolling_mean_{window}h", f"pm25_rolling_std_{window}h"]

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

    # Single target: PM2.5 exactly 24 hours ahead
    target_col = "pm25_h24"
    df[target_col] = df["pm25"].shift(-FORECAST_HOUR)

    df = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
    return df, feature_cols, target_col


# ── ONNX Export ────────────────────────────────────────────────────────────────
def export_onnx(model, model_key: str, n_features: int,
                output_path: str, is_lstm: bool = False):
    """Export a single-output model to ONNX."""
    if is_lstm:
        import torch
        pytorch_model = model.module_
        pytorch_model.eval()
        dummy = torch.zeros(1, 1, n_features)
        torch.onnx.export(
            pytorch_model, dummy, output_path,
            input_names=["lstm_input"],
            output_names=["output"],
            dynamic_axes={"lstm_input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=17,
        )
    elif model_key == "xgboost":
        from onnxmltools.convert import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType as OFT
        onnx_model = convert_xgboost(model, initial_types=[("float_input", OFT([None, n_features]))])
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
    else:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        onnx_model = convert_sklearn(
            model,
            initial_types=[("float_input", FloatTensorType([None, n_features]))],
            target_opset=17,
        )
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

    print(f"  ONNX saved → {output_path}")


# ── Training ───────────────────────────────────────────────────────────────────
def train_station_24h(station_id: int):
    print(f"\n{'═'*60}")
    print(f"  STATION {station_id}  —  T+24h Forecast")
    print(f"{'═'*60}")

    # ── Load & engineer features ──
    hourly = load_station_hourly(station_id)
    print(f"  Hourly rows: {len(hourly)}  "
          f"({hourly['datetime'].min().date()} → {hourly['datetime'].max().date()})")

    feat_df, feature_cols, target_col = build_features(hourly)
    n_features = len(feature_cols)
    print(f"  Features: {n_features}  |  Target: {target_col}")

    # ── Train / test split ──
    train_df = feat_df[feat_df["datetime"] <= TRAIN_END]
    test_df  = feat_df[feat_df["datetime"] >= TEST_START]

    if len(train_df) < 500 or len(test_df) < 48:
        print(f"  Skipping: not enough data (train={len(train_df)}, test={len(test_df)})")
        return None

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df[target_col].values.astype(np.float32)
    X_test  = test_df[feature_cols].values.astype(np.float32)
    y_test  = test_df[target_col].values.astype(np.float32)

    train_start = train_df["datetime"].min().strftime("%Y-%m-%d")
    train_end   = train_df["datetime"].max().strftime("%Y-%m-%d")

    print(f"  Train: {X_train.shape}  Test: {X_test.shape}")
    print(f"  Date range: {train_start} → {train_end}")

    # 3-D for LSTM (defined here so production comparison can use it)
    X_test_3d = X_test.reshape(X_test.shape[0], 1, n_features)

    mlflow.set_experiment(f"pm25_24h_station_{station_id}")

    tscv   = TimeSeriesSplit(n_splits=3)
    n_jobs = int(os.environ.get("GRID_N_JOBS", "-1"))
    trained = {}   # key → (name, model, metrics, is_lstm)
    results = []

    # ── 1. Linear Regression ──
    with mlflow.start_run(run_name="LinearRegression_24h"):
        model   = LinearRegression().fit(X_train, y_train)
        metrics = evaluate_model(y_test, model.predict(X_test))
        mlflow.log_params({"model": "LinearRegression", "station": station_id})
        mlflow.log_metrics(metrics)
        trained["linear_regression"] = ("Linear Regression", model, metrics, False)
        results.append({"model": "Linear Regression", **metrics})
        print(f"  Linear Regression  MAE={metrics['MAE']:.4f}")

    # ── 2. Ridge ──
    with mlflow.start_run(run_name="Ridge_24h"):
        grid = GridSearchCV(Ridge(), {"alpha": [0.1, 1.0, 10.0, 100.0]},
                            cv=tscv, scoring="neg_mean_absolute_error",
                            n_jobs=n_jobs, verbose=0)
        grid.fit(X_train, y_train)
        model   = grid.best_estimator_
        metrics = evaluate_model(y_test, model.predict(X_test))
        mlflow.log_params({"alpha": grid.best_params_["alpha"], "station": station_id})
        mlflow.log_metrics(metrics)
        trained["ridge_regression"] = ("Ridge Regression", model, metrics, False)
        results.append({"model": "Ridge Regression", **metrics})
        print(f"  Ridge Regression   MAE={metrics['MAE']:.4f}  alpha={grid.best_params_['alpha']}")

    # ── 3. Random Forest ──
    with mlflow.start_run(run_name="RandomForest_24h"):
        grid = GridSearchCV(
            RandomForestRegressor(random_state=RANDOM_STATE),
            {"n_estimators": [100], "max_depth": [5, 10], "min_samples_leaf": [4]},
            cv=tscv, scoring="neg_mean_absolute_error", n_jobs=n_jobs, verbose=0,
        )
        grid.fit(X_train, y_train)
        model   = grid.best_estimator_
        metrics = evaluate_model(y_test, model.predict(X_test))
        mlflow.log_params({**grid.best_params_, "station": station_id})
        mlflow.log_metrics(metrics)
        trained["random_forest"] = ("Random Forest", model, metrics, False)
        results.append({"model": "Random Forest", **metrics})
        print(f"  Random Forest      MAE={metrics['MAE']:.4f}  params={grid.best_params_}")

    # ── 4. XGBoost ──
    with mlflow.start_run(run_name="XGBoost_24h"):
        grid = GridSearchCV(
            XGBRegressor(random_state=RANDOM_STATE, verbosity=0, n_jobs=1),
            {"n_estimators": [100], "max_depth": [3, 5], "learning_rate": [0.1]},
            cv=tscv, scoring="neg_mean_absolute_error", n_jobs=n_jobs, verbose=0,
        )
        grid.fit(X_train, y_train)
        model   = grid.best_estimator_
        metrics = evaluate_model(y_test, model.predict(X_test))
        mlflow.log_params({**grid.best_params_, "station": station_id})
        mlflow.log_metrics(metrics)
        trained["xgboost"] = ("XGBoost", model, metrics, False)
        results.append({"model": "XGBoost", **metrics})
        print(f"  XGBoost            MAE={metrics['MAE']:.4f}  params={grid.best_params_}")

    # ── 5. LSTM (single output: T+24) ──
    with mlflow.start_run(run_name="LSTM_24h"):
        import torch
        import torch.nn as nn
        from skorch import NeuralNetRegressor

        class LSTMForecast(nn.Module):
            """LSTM → single value (T+24)."""
            def __init__(self, input_size=19, hidden_size=64,
                         num_layers=2, dropout=0.1):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                    batch_first=True,
                                    dropout=dropout if num_layers > 1 else 0.0)
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])   # (batch, 1)

        device = "cpu" if os.environ.get("PYTORCH_DEVICE") == "cpu" else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        torch.set_num_threads(1)
        print(f"  LSTM device: {device}")

        # Subsample every 4th row — consecutive rows are highly correlated
        X_lstm = X_train[::4].reshape(-1, 1, n_features)
        y_lstm = y_train[::4].reshape(-1, 1)

        net = NeuralNetRegressor(
            module=LSTMForecast,
            module__input_size=n_features,
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
        net.set_params(module__hidden_size=64, module__dropout=0.1, optimizer__lr=0.001)
        net.fit(X_lstm, y_lstm)

        preds   = net.predict(X_test_3d).flatten()
        metrics = evaluate_model(y_test, preds)
        mlflow.log_params({"hidden_size": 64, "dropout": 0.1, "lr": 0.001, "station": station_id})
        mlflow.log_metrics(metrics)
        trained["lstm"] = ("LSTM", net, metrics, True)
        results.append({"model": "LSTM", **metrics})
        print(f"  LSTM               MAE={metrics['MAE']:.4f}")

    # ── Summary ──
    results_df = pd.DataFrame(results)
    print(f"\n  {'Model':<25} {'MAE':>7} {'RMSE':>7} {'R2':>7}")
    print(f"  {'─'*46}")
    for _, row in results_df.iterrows():
        print(f"  {row['model']:<25} {row['MAE']:>7.4f} {row['RMSE']:>7.4f} {row['R2']:>7.4f}")

    best_key                              = min(trained, key=lambda k: trained[k][2]["MAE"])
    best_name, best_model, best_m, is_lstm = trained[best_key]
    new_mae = best_m["MAE"]
    print(f"\n  Best: {best_name}  MAE={new_mae:.4f}")

    # ── Compare vs production ──
    models_dir = os.path.join(MODELS_DIR, f"station_{station_id}_24h")
    os.makedirs(os.path.join(models_dir, "onnx"), exist_ok=True)
    registry = os.path.join(models_dir, "active_model.json")

    prod_mae = None
    old_info = None
    if os.path.exists(registry):
        with open(registry) as f:
            old_info = json.load(f)
        onnx_prod = os.path.join(models_dir, "onnx", old_info["onnx_file"])
        if os.path.exists(onnx_prod):
            import onnxruntime as rt
            sess     = rt.InferenceSession(onnx_prod, providers=["CPUExecutionProvider"])
            in_name  = sess.get_inputs()[0].name
            out_name = sess.get_outputs()[0].name
            X_in     = X_test_3d if old_info.get("is_lstm") else X_test
            preds_prod = sess.run([out_name], {in_name: X_in})[0].flatten()
            prod_mae   = evaluate_model(y_test, preds_prod)["MAE"]

    prod_str = (f"MAE={prod_mae:.4f}  ({old_info['train_start']} → {old_info['train_end']})"
                if prod_mae is not None else "N/A (first deploy)")
    print(f"  Prod: {prod_str}")

    # ── Deploy if better ──
    if prod_mae is None or new_mae < prod_mae:
        onnx_filename = f"{best_key}_{train_start}_{train_end}.onnx"
        onnx_path     = os.path.join(models_dir, "onnx", onnx_filename)
        export_onnx(best_model, best_key, n_features, onnx_path, is_lstm)

        info = {
            "onnx_file":     onnx_filename,
            "model_key":     best_key,
            "station_id":    station_id,
            "train_start":   train_start,
            "train_end":     train_end,
            "is_lstm":       is_lstm,
            "forecast_hour": FORECAST_HOUR,
            "n_features":    n_features,
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

    # ── Append summary report ──
    os.makedirs(RESULTS_DIR, exist_ok=True)
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

    return results_df


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train T+24h PM2.5 forecasting models")
    parser.add_argument("--stations", nargs="+", type=int, default=[63, 64, 65, 66, 67])
    args = parser.parse_args()

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))

    for sid in args.stations:
        train_station_24h(sid)

    print(f"\n{'═'*60}")
    print("T+24h FORECAST DEPLOYMENT SUMMARY")
    print(f"{'═'*60}")
    report_path = os.path.join(RESULTS_DIR, "forecast_24h_results.csv")
    if os.path.exists(report_path):
        report = pd.read_csv(report_path)
        latest = report.sort_values("train_start").groupby("station_id").tail(1)
        print(latest[["station_id", "best_model", "new_mae", "prod_mae", "status"]].to_string(index=False))
    print(f"\nReport: {report_path}")

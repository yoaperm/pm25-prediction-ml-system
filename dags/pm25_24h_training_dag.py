"""
PM2.5 T+24h Training DAG  (PostgreSQL-based, dynamic date splits)
==================================================================
Trains regression, neural, and statistical models to predict PM2.5 exactly
24 hours ahead for stations
that store data in PostgreSQL (pm25_raw_hourly table).

Stations : 56, 57, 58, 59, 61
Data     : PostgreSQL table pm25_raw_hourly  (timestamp, station_id, pm25)
Splits   : dynamic, relative to today
  ├── Train : today - 3y6m → today - 6m   (3 years)
  ├── Val   : today - 6m   → today - 3m   (3 months, LSTM early-stop)
  └── Test  : today - 3m   → today        (3 months, evaluation)

Task graph:
    feature_engineering
        ├── train_baseline
        ├── train_ridge
        ├── train_random_forest
        ├── train_xgboost
        ├── train_lstm
        ├── train_transformer
        └── train_sarima_24h
                └── (all join) ── evaluate ── compare_and_deploy

Trigger manually via Airflow UI with:
    {"station_id": 56}

Environment variable (set in docker-compose or Airflow Connections):
    PM25_DB_URL  (default: postgresql://postgres:postgres@postgres:5432/pm25)
"""

import os
from datetime import datetime

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

# ── Shared paths inside Docker container ─────────────────────────────────────
SRC         = "/app/src"
CONFIG_PATH = "/app/configs/config.yaml"
MODELS_DIR  = "/app/models"
RESULTS_DIR = "/app/results"
PROCESSED   = "/app/data/processed"

DEFAULT_DB_URL = "postgresql://postgres:postgres@postgres:5432/pm25"
FORECAST_HOUR  = 24
RANDOM_STATE   = 42


# ── Helpers ───────────────────────────────────────────────────────────────────
def _db_url():
    return os.environ.get("PM25_DB_URL", DEFAULT_DB_URL)


def _station_id(context):
    return context["params"]["station_id"]


def _models_dir(station_id):
    d = f"{MODELS_DIR}/station_{station_id}_24h"
    os.makedirs(f"{d}/onnx", exist_ok=True)
    return d


def _tmp_onnx(station_id, key):
    return f"{_models_dir(station_id)}/_tmp_{key}.onnx"


def _processed_dir(station_id):
    d = f"{PROCESSED}/station_{station_id}_24h"
    os.makedirs(d, exist_ok=True)
    return d


def _get_splits():
    """Dynamic 3y6m window relative to today."""
    from dateutil.relativedelta import relativedelta
    today      = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    data_start = today - relativedelta(years=3, months=6)
    val_start  = today - relativedelta(months=6)
    test_start = today - relativedelta(months=3)
    return {
        "data_start":  data_start.strftime("%Y-%m-%d"),
        "train_end":   (val_start  - relativedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
        "val_start":   val_start.strftime("%Y-%m-%d"),
        "val_end":     (test_start - relativedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
        "test_start":  test_start.strftime("%Y-%m-%d"),
        "test_end":    today.strftime("%Y-%m-%d"),
    }


def _load_hourly_from_pg(station_id, db_url, data_start):
    """Query pm25_raw_hourly, reindex to full hourly range, fill gaps."""
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
    """
    19 features for T+24h prediction (no leakage).
    Target: pm25_h24 = pm25.shift(-24)
    """
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


def _onnx_predict(onnx_path, X):
    import onnxruntime as rt
    sess      = rt.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name   = sess.get_inputs()[0].name
    out_name  = sess.get_outputs()[0].name
    return sess.run([out_name], {in_name: X.astype("float32")})[0].flatten()


def _setup_mlflow(station_id):
    import mlflow
    uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(f"pm25_24h_station_{station_id}")


def _load_arrays(station_id):
    import json
    import pandas as pd
    base = _processed_dir(station_id)
    X_train = pd.read_parquet(f"{base}/X_train.parquet").values
    y_train = pd.read_parquet(f"{base}/y_train.parquet")["target"].values
    X_val   = pd.read_parquet(f"{base}/X_val.parquet").values
    y_val   = pd.read_parquet(f"{base}/y_val.parquet")["target"].values
    X_test  = pd.read_parquet(f"{base}/X_test.parquet").values
    y_test  = pd.read_parquet(f"{base}/y_test.parquet")["target"].values
    with open(f"{base}/meta.json") as f:
        meta = json.load(f)
    return X_train, y_train, X_val, y_val, X_test, y_test, meta


# ── Task 1: Load from PostgreSQL + feature engineering ───────────────────────
def _feature_engineering(**context):
    import json
    import numpy as np
    import pandas as pd

    station_id = _station_id(context)
    splits     = _get_splits()
    db_url     = _db_url()

    print(f"Station {station_id} — querying from {splits['data_start']} to {splits['test_end']}")
    print(f"  Train: …→ {splits['train_end']}")
    print(f"  Val  : {splits['val_start']} → {splits['val_end']}")
    print(f"  Test : {splits['test_start']} → {splits['test_end']}")

    hourly = _load_hourly_from_pg(station_id, db_url, splits["data_start"])
    if hourly.empty:
        raise ValueError(f"No data found in PostgreSQL for station {station_id}")
    print(f"  Loaded {len(hourly)} hourly rows "
          f"({hourly['datetime'].min().date()} → {hourly['datetime'].max().date()})")

    feat_df, feature_cols, target_col = _build_features_24h(hourly)
    n_features = len(feature_cols)
    print(f"  Features: {n_features}  |  Target: {target_col}  |  Rows: {len(feat_df)}")

    train_df = feat_df[feat_df["datetime"] <  splits["val_start"]]
    val_df   = feat_df[(feat_df["datetime"] >= splits["val_start"]) &
                       (feat_df["datetime"] <  splits["test_start"])]
    test_df  = feat_df[feat_df["datetime"] >= splits["test_start"]]

    print(f"  Split  → train:{len(train_df)}  val:{len(val_df)}  test:{len(test_df)}")
    if len(train_df) < 200 or len(test_df) < 48:
        raise ValueError(f"Not enough data: train={len(train_df)} test={len(test_df)}")

    base = _processed_dir(station_id)
    for name, arr in [("X_train", train_df[feature_cols].values),
                      ("X_val",   val_df[feature_cols].values),
                      ("X_test",  test_df[feature_cols].values)]:
        pd.DataFrame(arr).to_parquet(f"{base}/{name}.parquet", index=False)
    for name, arr in [("y_train", train_df[target_col].values),
                      ("y_val",   val_df[target_col].values),
                      ("y_test",  test_df[target_col].values)]:
        pd.DataFrame(arr, columns=["target"]).to_parquet(f"{base}/{name}.parquet", index=False)

    # Raw pm25 series for SARIMA (endog only — no shifting, no feature engineering)
    for name, df_split in [("pm25_raw_train", train_df),
                            ("pm25_raw_val",   val_df),
                            ("pm25_raw_test",  test_df)]:
        pd.DataFrame({"pm25": df_split["pm25"].values}).to_parquet(
            f"{base}/{name}.parquet", index=False
        )

    meta = {
        "feature_cols": feature_cols,
        "n_features":   n_features,
        "train_start":  str(train_df["datetime"].min().date()),
        "train_end":    str(train_df["datetime"].max().date()),
        "splits":       splits,
    }
    with open(f"{base}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved parquets → {base}")


# ── Task 2a: Linear Regression ───────────────────────────────────────────────
def _train_linear(**context):
    import mlflow
    from sklearn.linear_model import LinearRegression
    import sys; sys.path.insert(0, SRC)
    from evaluate import evaluate_model

    station_id = _station_id(context)
    _setup_mlflow(station_id)
    X_train, y_train, _, _, X_test, y_test, meta = _load_arrays(station_id)

    with mlflow.start_run(run_name="LinearRegression_24h"):
        model = LinearRegression().fit(X_train, y_train)
        metrics = evaluate_model(y_test, model.predict(X_test))
        mlflow.log_params({"model": "LinearRegression", "station": station_id})
        mlflow.log_metrics(metrics)

    import sys as _sys; _sys.path.insert(0, SRC)
    from export_onnx import export_sklearn
    export_sklearn(model, "linear_regression",
                   _models_dir(station_id),
                   output_path=_tmp_onnx(station_id, "linear_regression"),
                   n_features=meta["n_features"])
    print(f"  LinearRegression  RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}")


# ── Task 2b: Ridge ────────────────────────────────────────────────────────────
def _train_ridge(**context):
    import mlflow
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    import sys; sys.path.insert(0, SRC)
    from evaluate import evaluate_model
    from export_onnx import export_sklearn

    station_id = _station_id(context)
    _setup_mlflow(station_id)
    X_train, y_train, _, _, X_test, y_test, meta = _load_arrays(station_id)

    n_jobs = int(os.environ.get("GRID_N_JOBS", "-1"))
    grid   = GridSearchCV(Ridge(), {"alpha": [0.1, 1.0, 10.0, 100.0]},
                          cv=TimeSeriesSplit(3), scoring="neg_mean_absolute_error",
                          n_jobs=n_jobs, verbose=0)
    grid.fit(X_train, y_train)
    model   = grid.best_estimator_
    metrics = evaluate_model(y_test, model.predict(X_test))

    with mlflow.start_run(run_name="Ridge_24h"):
        mlflow.log_params({**grid.best_params_, "station": station_id})
        mlflow.log_metrics(metrics)

    export_sklearn(model, "ridge_regression",
                   _models_dir(station_id),
                   output_path=_tmp_onnx(station_id, "ridge_regression"),
                   n_features=meta["n_features"])
    print(f"  Ridge  RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}  alpha={grid.best_params_['alpha']}")


# ── Task 2c: Random Forest ────────────────────────────────────────────────────
def _train_random_forest(**context):
    import mlflow
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    import sys; sys.path.insert(0, SRC)
    from evaluate import evaluate_model
    from export_onnx import export_sklearn

    station_id = _station_id(context)
    _setup_mlflow(station_id)
    X_train, y_train, _, _, X_test, y_test, meta = _load_arrays(station_id)

    n_jobs = int(os.environ.get("GRID_N_JOBS", "-1"))
    grid   = GridSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE),
        {"n_estimators": [100], "max_depth": [5, 10], "min_samples_leaf": [4]},
        cv=TimeSeriesSplit(3), scoring="neg_mean_absolute_error",
        n_jobs=n_jobs, verbose=0,
    )
    grid.fit(X_train, y_train)
    model   = grid.best_estimator_
    metrics = evaluate_model(y_test, model.predict(X_test))

    with mlflow.start_run(run_name="RandomForest_24h"):
        mlflow.log_params({**grid.best_params_, "station": station_id})
        mlflow.log_metrics(metrics)

    export_sklearn(model, "random_forest",
                   _models_dir(station_id),
                   output_path=_tmp_onnx(station_id, "random_forest"),
                   n_features=meta["n_features"])
    print(f"  RandomForest  RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}  params={grid.best_params_}")


# ── Task 2d: XGBoost ─────────────────────────────────────────────────────────
def _train_xgboost(**context):
    import mlflow
    from xgboost import XGBRegressor
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    import sys; sys.path.insert(0, SRC)
    from evaluate import evaluate_model

    station_id = _station_id(context)
    _setup_mlflow(station_id)
    X_train, y_train, _, _, X_test, y_test, meta = _load_arrays(station_id)

    n_jobs = int(os.environ.get("GRID_N_JOBS", "-1"))
    grid   = GridSearchCV(
        XGBRegressor(random_state=RANDOM_STATE, verbosity=0, n_jobs=1),
        {"n_estimators": [100], "max_depth": [3, 5], "learning_rate": [0.1]},
        cv=TimeSeriesSplit(3), scoring="neg_mean_absolute_error",
        n_jobs=n_jobs, verbose=0,
    )
    grid.fit(X_train, y_train)
    model   = grid.best_estimator_
    metrics = evaluate_model(y_test, model.predict(X_test))

    with mlflow.start_run(run_name="XGBoost_24h"):
        mlflow.log_params({**grid.best_params_, "station": station_id})
        mlflow.log_metrics(metrics)

    # Export via onnxmltools
    from onnxmltools.convert import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType as OFT
    onnx_model = convert_xgboost(
        model,
        initial_types=[("float_input", OFT([None, meta["n_features"]]))]
    )
    with open(_tmp_onnx(station_id, "xgboost"), "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"  XGBoost  RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}  params={grid.best_params_}")


# ── Task 2e: LSTM (single output T+24, val-based early stopping) ──────────────
def _train_lstm(**context):
    import mlflow
    import torch
    import torch.nn as nn
    import numpy as np
    from torch.utils.data import DataLoader, TensorDataset
    import sys; sys.path.insert(0, SRC)
    from evaluate import evaluate_model

    station_id = _station_id(context)
    _setup_mlflow(station_id)
    X_train, y_train, X_val, y_val, X_test, y_test, meta = _load_arrays(station_id)

    torch.set_num_threads(1)
    device_str = "cpu"
    n_features = meta["n_features"]

    class LSTMForecast(nn.Module):
        def __init__(self, input_size=19, hidden_size=64, num_layers=2, dropout=0.1):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True,
                                dropout=dropout if num_layers > 1 else 0.0)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze(-1)   # (batch,)

    # Subsample training rows (consecutive hours are highly correlated)
    X_tr = torch.FloatTensor(X_train[::4].reshape(-1, 1, n_features))
    y_tr = torch.FloatTensor(y_train[::4])
    X_vl = torch.FloatTensor(X_val.reshape(-1, 1, n_features))
    y_vl = torch.FloatTensor(y_val)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=512, shuffle=False)

    lstm_model = LSTMForecast(input_size=n_features).to(device_str)
    optimizer  = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
    criterion  = nn.L1Loss()

    best_val_loss = float("inf")
    patience_left = 5
    best_state    = None
    epochs_done   = 0

    for epoch in range(30):
        lstm_model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device_str), yb.to(device_str)
            optimizer.zero_grad()
            loss = criterion(lstm_model(xb), yb)
            loss.backward()
            optimizer.step()

        lstm_model.eval()
        with torch.no_grad():
            val_loss = criterion(lstm_model(X_vl.to(device_str)), y_vl.to(device_str)).item()

        epochs_done = epoch + 1
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in lstm_model.state_dict().items()}
            patience_left = 5
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    lstm_model.load_state_dict(best_state)

    X_test_3d = torch.FloatTensor(X_test.reshape(-1, 1, n_features)).to(device_str)
    lstm_model.eval()
    with torch.no_grad():
        y_pred = lstm_model(X_test_3d).cpu().numpy().flatten()
    metrics = evaluate_model(y_test, y_pred)

    with mlflow.start_run(run_name="LSTM_24h"):
        mlflow.log_params({"hidden_size": 64, "dropout": 0.1, "lr": 0.001,
                           "patience": 5, "station": station_id})
        mlflow.log_metrics(metrics)

    # Export LSTM to ONNX
    lstm_model.eval()
    dummy = torch.zeros(1, 1, n_features)
    torch.onnx.export(
        lstm_model, dummy, _tmp_onnx(station_id, "lstm"),
        input_names=["lstm_input"],
        output_names=["output"],
        dynamic_axes={"lstm_input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    print(f"  LSTM  RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}  epochs={epochs_done}  device={device_str}")


# ── Task 2f: Transformer (tabular feature-token encoder) ─────────────────────
def _train_transformer(**context):
    import mlflow
    import sys; sys.path.insert(0, SRC)
    from data_loader import load_config
    from evaluate import evaluate_model
    from transformer_model import (
        export_transformer_onnx,
        predict_transformer,
        train_transformer_regressor,
    )

    station_id = _station_id(context)
    _setup_mlflow(station_id)
    X_train, y_train, X_val, y_val, X_test, y_test, meta = _load_arrays(station_id)
    config = load_config(CONFIG_PATH)

    model, best = train_transformer_regressor(
        X_train,
        y_train,
        X_val,
        y_val,
        config.get("models", {}).get("transformer", {}).get("params", {}),
        random_state=RANDOM_STATE,
    )
    y_pred = predict_transformer(model, X_test)
    val_pred = predict_transformer(model, X_val)
    metrics = evaluate_model(y_test, y_pred)
    val_mae = evaluate_model(y_val, val_pred)["MAE"]

    with mlflow.start_run(run_name="Transformer_24h"):
        mlflow.log_params({**best, "station": station_id})
        mlflow.log_metrics({**metrics, "val_MAE": val_mae})

    export_transformer_onnx(
        model,
        _tmp_onnx(station_id, "transformer"),
        n_features=meta["n_features"],
    )
    print(
        f"  Transformer  RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}  "
        f"val_MAE={val_mae:.4f}  epochs={best['epochs_done']}  device={best['device']}"
    )


# ── Task 2g: SARIMA (statistical time-series benchmark, m=24) ────────────────
def _train_sarima_24h(**context):
    import json
    import pandas as pd
    import mlflow
    import sys; sys.path.insert(0, SRC)
    from sarima_model import train_sarima_with_tuning, predict_sarima_n_ahead_rolling
    from evaluate import evaluate_model

    station_id = _station_id(context)
    _setup_mlflow(station_id)
    base       = _processed_dir(station_id)
    models_dir = _models_dir(station_id)

    y_test         = pd.read_parquet(f"{base}/y_test.parquet")["target"].values
    pm25_raw_train = pd.read_parquet(f"{base}/pm25_raw_train.parquet")["pm25"].values
    pm25_raw_test  = pd.read_parquet(f"{base}/pm25_raw_test.parquet")["pm25"].values

    # Limit auto_arima input to last 90 days — full 3yr series takes 20+ min and
    # hits the Airflow heartbeat timeout. 2160 obs covers 90 daily cycles (m=24),
    # enough for stable parameter estimation. Rolling predictions start from here.
    MAX_SARIMA_OBS = 90 * 24
    pm25_raw_train_sarima = pm25_raw_train[-MAX_SARIMA_OBS:]

    model, params = train_sarima_with_tuning(
        pm25_raw_train_sarima,
        seasonal_period=24,
        max_p=2, max_q=2, max_P=1, max_Q=1, max_d=1, max_D=1,
    )
    y_pred  = predict_sarima_n_ahead_rolling(model, pm25_raw_test, n_ahead=FORECAST_HOUR)
    metrics = evaluate_model(y_test, y_pred)

    with mlflow.start_run(run_name="SARIMA_24h"):
        mlflow.log_params({**params, "station": station_id, "seasonal_period": 24})
        mlflow.log_metrics(metrics)

    result = {"model": "SARIMA", **metrics,
              "order": params["order"], "seasonal_order": params["seasonal_order"]}
    with open(f"{models_dir}/_tmp_sarima_result.json", "w") as f:
        json.dump(result, f)
    print(f"  SARIMA  RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}  "
          f"order={params['order']} seasonal={params['seasonal_order']}")


# ── Task 3: Evaluate all 6 temp models ───────────────────────────────────────
def _evaluate(**context):
    import json
    import pandas as pd
    import sys; sys.path.insert(0, SRC)
    from evaluate import evaluate_model, print_metrics

    station_id = _station_id(context)
    _, _, _, _, X_test, y_test, meta = _load_arrays(station_id)
    n = meta["n_features"]
    X_f  = X_test.astype("float32")
    X_3d = X_test.reshape(-1, 1, n).astype("float32")

    tmp_keys = {
        "Linear Regression": ("linear_regression",  False),
        "Ridge Regression":  ("ridge_regression",   False),
        "Random Forest":     ("random_forest",       False),
        "XGBoost":           ("xgboost",             False),
        "LSTM":              ("lstm",                True),
        "Transformer":       ("transformer",         False),
    }

    results = []
    for name, (key, is_lstm) in tmp_keys.items():
        path = _tmp_onnx(station_id, key)
        if not os.path.exists(path):
            print(f"  [skip] {name} — tmp file not found")
            continue
        preds   = _onnx_predict(path, X_3d if is_lstm else X_f)
        metrics = evaluate_model(y_test, preds)
        print_metrics(name, metrics)
        results.append({"model": name, **metrics})

    # SARIMA result (no ONNX file — read from JSON written by _train_sarima_24h)
    sarima_result_path = f"{_models_dir(station_id)}/_tmp_sarima_result.json"
    if os.path.exists(sarima_result_path):
        with open(sarima_result_path) as f:
            sr = json.load(f)
        sarima_metrics = {k: sr[k] for k in ("RMSE", "MAE", "R2") if k in sr}
        print_metrics("SARIMA", sarima_metrics)
        results.append({"model": "SARIMA", **sarima_metrics})

    df = pd.DataFrame(results)
    best_idx = df['RMSE'].idxmin()
    print(f"\nStation {station_id} best: {df.loc[best_idx, 'model']} (RMSE={df.loc[best_idx, 'RMSE']:.4f})")


# ── Helper: Publish to Triton ─────────────────────────────────────────────────
def _publish_to_triton(onnx_path, station_id, n_features, triton_repo="/app/triton_model_repo"):
    """
    Publish ONNX model to Triton repository by:
    1. Creating triton_model_repo/pm25_{station_id}/1/ directory
    2. Copying ONNX file to model.onnx
    3. Creating config.pbtxt with correct input dimensions
    """
    import os
    import shutil

    model_name = f"pm25_{station_id}"
    model_dir = os.path.join(triton_repo, model_name)
    version_dir = os.path.join(model_dir, "1")

    # Create directory structure
    os.makedirs(version_dir, exist_ok=True)

    # Copy ONNX file
    dest_onnx = os.path.join(version_dir, "model.onnx")
    shutil.copy2(onnx_path, dest_onnx)
    print(f"  ✓ Copied ONNX to Triton: {dest_onnx}")

    # Create config.pbtxt
    config_content = f'''name: "{model_name}"
backend: "onnxruntime"
max_batch_size: 32

input [
  {{
    name: "float_input"
    data_type: TYPE_FP32
    dims: [ {n_features} ]
  }}
]

output [
  {{
    name: "variable"
    data_type: TYPE_FP32
    dims: [ 1 ]
  }}
]

dynamic_batching {{ }}
'''

    config_path = os.path.join(model_dir, "config.pbtxt")
    with open(config_path, "w") as f:
        f.write(config_content)
    print(f"  ✓ Created Triton config: {config_path}")
    print(f"  ✓ Model '{model_name}' published to Triton (will load in ~30s)")


# ── Task 4: Compare vs production, deploy if better ──────────────────────────
def _compare_and_deploy(**context):
    import json
    import shutil
    import numpy as np
    import pandas as pd
    import sys; sys.path.insert(0, SRC)
    from evaluate import evaluate_model

    station_id = _station_id(context)
    _, _, _, _, X_test, y_test, meta = _load_arrays(station_id)
    n    = meta["n_features"]
    X_f  = X_test.astype("float32")
    X_3d = X_test.reshape(-1, 1, n).astype("float32")

    train_start = meta["train_start"]
    train_end   = meta["train_end"]
    models_dir  = _models_dir(station_id)
    base        = _processed_dir(station_id)

    tmp_keys = {
        "linear_regression": ("Linear Regression", False),
        "ridge_regression":  ("Ridge Regression",  False),
        "random_forest":     ("Random Forest",      False),
        "xgboost":           ("XGBoost",            False),
        "lstm":              ("LSTM",               True),
        "transformer":       ("Transformer",         False),
    }

    trained = {}
    for key, (name, is_lstm) in tmp_keys.items():
        path = _tmp_onnx(station_id, key)
        if not os.path.exists(path):
            continue
        preds = _onnx_predict(path, X_3d if is_lstm else X_f)
        metrics = evaluate_model(y_test, preds)
        trained[key] = (name, metrics["RMSE"], metrics["MAE"], is_lstm)

    if not trained:
        raise RuntimeError("No trained models found — all tmp ONNX missing")

    # Best ONNX candidate
    best_key = min(trained, key=lambda k: trained[k][1])
    best_name, new_rmse, new_mae, best_is_lstm = trained[best_key]

    # Load SARIMA candidate (written by _train_sarima_24h)
    sarima_tmp_path = f"{models_dir}/_tmp_sarima_result.json"
    sarima_candidate = None
    if os.path.exists(sarima_tmp_path):
        with open(sarima_tmp_path) as f:
            sc = json.load(f)
        sarima_candidate = {"rmse": sc["RMSE"], "mae": sc["MAE"],
                            "order": sc["order"], "seasonal_order": sc["seasonal_order"]}

    # Check if SARIMA beats the best ONNX model
    sarima_wins = sarima_candidate is not None and sarima_candidate["rmse"] < new_rmse
    if sarima_wins:
        new_rmse  = sarima_candidate["rmse"]
        new_mae   = sarima_candidate["mae"]
        best_name = "SARIMA"
    print(f"Best new model: {best_name}  RMSE={new_rmse:.4f}  MAE={new_mae:.4f}")

    # Load production model for comparison
    registry  = f"{models_dir}/active_model.json"
    prod_rmse = None
    prod_mae  = None
    old_info  = None

    if os.path.exists(registry):
        with open(registry) as f:
            old_info = json.load(f)

        if old_info.get("backend") == "sarima":
            # Prod is SARIMA — recompute its RMSE by refitting with saved order
            sarima_order_path = f"{models_dir}/sarima_order.json"
            if os.path.exists(sarima_order_path):
                from sarima_model import fit_sarima, predict_sarima_n_ahead_rolling
                with open(sarima_order_path) as f:
                    so = json.load(f)
                pm25_raw_train = pd.read_parquet(f"{base}/pm25_raw_train.parquet")["pm25"].values
                pm25_raw_test  = pd.read_parquet(f"{base}/pm25_raw_test.parquet")["pm25"].values
                prod_model  = fit_sarima(tuple(so["order"]), tuple(so["seasonal_order"]), pm25_raw_train)
                preds_prod  = predict_sarima_n_ahead_rolling(prod_model, pm25_raw_test, FORECAST_HOUR)
                prod_metrics = evaluate_model(y_test, preds_prod)
                prod_rmse = prod_metrics["RMSE"]
                prod_mae  = prod_metrics["MAE"]
        else:
            # Prod is ONNX
            onnx_prod = f"{models_dir}/onnx/{old_info.get('onnx_file', '')}"
            if os.path.exists(onnx_prod):
                X_in       = X_3d if old_info.get("input_shape") == "3d" else X_f
                preds_prod = _onnx_predict(onnx_prod, X_in)
                if preds_prod.shape[0] != y_test.shape[0]:
                    print("  Prod model incompatible shape — treating as first deploy")
                    old_info = None
                else:
                    prod_metrics = evaluate_model(y_test, preds_prod)
                    prod_rmse = prod_metrics["RMSE"]
                    prod_mae  = prod_metrics["MAE"]

    prod_str = (f"RMSE={prod_rmse:.4f}  MAE={prod_mae:.4f}  ({old_info['train_start']} → {old_info['train_end']})"
                if prod_rmse is not None else "N/A (first deploy)")
    print(f"  Prod: {prod_str}")

    if prod_rmse is None or new_rmse < prod_rmse:
        delta_rmse = round(prod_rmse - new_rmse, 4) if prod_rmse is not None else None
        delta_mae  = round(prod_mae  - new_mae,  4) if prod_mae  is not None else None
        status = "DEPLOYED"

        if sarima_wins:
            # SARIMA deployment — no ONNX file, no Triton publish
            with open(f"{models_dir}/sarima_order.json", "w") as f:
                json.dump({"order": sarima_candidate["order"],
                           "seasonal_order": sarima_candidate["seasonal_order"]}, f)
            info = {
                "model_key":     "sarima",
                "station_id":    station_id,
                "train_start":   train_start,
                "train_end":     train_end,
                "backend":       "sarima",
                "forecast_hour": FORECAST_HOUR,
            }
            with open(registry, "w") as f:
                json.dump(info, f, indent=2)
            with open(f"{models_dir}/feature_columns.json", "w") as f:
                json.dump(meta["feature_cols"], f)
            print(f"  → DEPLOYED (SARIMA — no Triton publish)")
        else:
            # ONNX deployment
            onnx_filename = f"{best_key}_{train_start}_{train_end}.onnx"
            onnx_dest     = f"{models_dir}/onnx/{onnx_filename}"
            shutil.copy2(_tmp_onnx(station_id, best_key), onnx_dest)
            info = {
                "onnx_file":     onnx_filename,
                "model_key":     best_key,
                "station_id":    station_id,
                "train_start":   train_start,
                "train_end":     train_end,
                "backend":       "onnx",
                "input_shape":   "3d" if best_is_lstm else "2d",
                "forecast_hour": FORECAST_HOUR,
                "n_features":    n,
            }
            with open(registry, "w") as f:
                json.dump(info, f, indent=2)
            with open(f"{models_dir}/feature_columns.json", "w") as f:
                json.dump(meta["feature_cols"], f)
            print(f"  → DEPLOYED  {onnx_dest}")
            try:
                _publish_to_triton(
                    onnx_path=onnx_dest,
                    station_id=station_id,
                    n_features=n,
                    triton_repo="/app/triton_model_repo"
                )
            except Exception as e:
                print(f"  ⚠ Triton publish failed (model still deployed to models/): {e}")
    else:
        status = "NOT_DEPLOYED"
        delta_rmse = round(prod_rmse - new_rmse, 4)
        delta_mae  = round(prod_mae  - new_mae,  4)
        print(f"  → NOT DEPLOYED  new RMSE {new_rmse:.4f} >= prod {prod_rmse:.4f}")

    # Clean up temp ONNX files and SARIMA result JSON
    for key in tmp_keys:
        p = _tmp_onnx(station_id, key)
        if os.path.exists(p):
            os.remove(p)
    if os.path.exists(sarima_tmp_path):
        os.remove(sarima_tmp_path)

    # Append to report
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = f"{RESULTS_DIR}/forecast_24h_results.csv"
    row = {
        "station_id":   station_id,
        "train_start":  train_start,
        "train_end":    train_end,
        "best_model":   best_name,
        "new_rmse":     new_rmse,
        "prod_rmse":    prod_rmse,
        "rmse_delta":   delta_rmse,
        "new_mae":      new_mae,
        "prod_mae":     prod_mae,
        "mae_delta":    delta_mae,
        "status":       status,
        "run_date":     datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
    }
    exists = os.path.exists(report_path)
    pd.DataFrame([row]).to_csv(report_path, mode="a", header=not exists, index=False)
    print(f"  Report: {report_path}")


# ── DAG definition ────────────────────────────────────────────────────────────
with DAG(
    dag_id="pm25_24h_training",
    schedule=None,            # manual trigger only (or set "0 2 * * 0" for weekly)
    start_date=datetime(2024, 1, 1),
    catchup=False,
    params={
        "station_id": Param(
            56,
            type="integer",
            enum=[56, 57, 58, 59, 61],
            description="Station ID to train (56–61 via PostgreSQL)",
        ),
    },
    tags=["pm25", "ml", "24h-forecast", "postgresql"],
) as dag:

    feat_task   = PythonOperator(task_id="feature_engineering",  python_callable=_feature_engineering)
    linear_task = PythonOperator(task_id="train_linear",         python_callable=_train_linear)
    ridge_task  = PythonOperator(task_id="train_ridge",          python_callable=_train_ridge)
    rf_task     = PythonOperator(task_id="train_random_forest",  python_callable=_train_random_forest)
    xgb_task    = PythonOperator(task_id="train_xgboost",        python_callable=_train_xgboost)
    lstm_task   = PythonOperator(task_id="train_lstm",           python_callable=_train_lstm)
    trans_task  = PythonOperator(task_id="train_transformer",    python_callable=_train_transformer)
    sarima_task = PythonOperator(task_id="train_sarima_24h",     python_callable=_train_sarima_24h)
    eval_task   = PythonOperator(task_id="evaluate",             python_callable=_evaluate)
    deploy_task = PythonOperator(task_id="compare_and_deploy",   python_callable=_compare_and_deploy)

    feat_task >> [linear_task, ridge_task, rf_task, xgb_task, lstm_task, trans_task, sarima_task]
    [linear_task, ridge_task, rf_task, xgb_task, lstm_task, trans_task, sarima_task] >> eval_task
    eval_task >> deploy_task

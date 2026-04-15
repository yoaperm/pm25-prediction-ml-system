"""
PM2.5 T+24h Training DAG  (PostgreSQL-based, dynamic date splits)
==================================================================
Trains 5 models to predict PM2.5 exactly 24 hours ahead for stations
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
        └── train_lstm
                └── (all 5 join) ── evaluate ── compare_and_deploy

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

    engine = sqlalchemy.create_engine(db_url)
    query  = """
        SELECT timestamp AS datetime, pm25
        FROM   pm25_raw_hourly
        WHERE  station_id = %(station_id)s
          AND  timestamp  >= %(data_start)s
          AND  pm25       IS NOT NULL
        ORDER  BY timestamp
    """
    df = pd.read_sql(query, engine,
                     params={"station_id": station_id, "data_start": data_start})
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
    X_train, y_train, _, _, X_test, y_test, _ = _load_arrays(station_id)

    with mlflow.start_run(run_name="LinearRegression_24h"):
        model = LinearRegression().fit(X_train, y_train)
        metrics = evaluate_model(y_test, model.predict(X_test))
        mlflow.log_params({"model": "LinearRegression", "station": station_id})
        mlflow.log_metrics(metrics)

    import sys as _sys; _sys.path.insert(0, SRC)
    from export_onnx import export_sklearn
    export_sklearn(model, "linear_regression",
                   _models_dir(station_id),
                   output_path=_tmp_onnx(station_id, "linear_regression"))
    print(f"  LinearRegression  MAE={metrics['MAE']:.4f}")


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
    X_train, y_train, _, _, X_test, y_test, _ = _load_arrays(station_id)

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
                   output_path=_tmp_onnx(station_id, "ridge_regression"))
    print(f"  Ridge  MAE={metrics['MAE']:.4f}  alpha={grid.best_params_['alpha']}")


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
    X_train, y_train, _, _, X_test, y_test, _ = _load_arrays(station_id)

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
                   output_path=_tmp_onnx(station_id, "random_forest"))
    print(f"  RandomForest  MAE={metrics['MAE']:.4f}  params={grid.best_params_}")


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
    print(f"  XGBoost  MAE={metrics['MAE']:.4f}  params={grid.best_params_}")


# ── Task 2e: LSTM (single output T+24, val-based early stopping) ──────────────
def _train_lstm(**context):
    import mlflow
    import torch
    import torch.nn as nn
    import numpy as np
    from skorch import NeuralNetRegressor
    from skorch.callbacks import EarlyStopping
    from skorch.dataset import Dataset
    import sys; sys.path.insert(0, SRC)
    from evaluate import evaluate_model

    station_id = _station_id(context)
    _setup_mlflow(station_id)
    X_train, y_train, X_val, y_val, X_test, y_test, meta = _load_arrays(station_id)

    torch.set_num_threads(1)
    device = "cpu" if os.environ.get("PYTORCH_DEVICE") == "cpu" else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
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
            return self.fc(out[:, -1, :])   # (batch, 1)

    # Subsample training rows (consecutive hours are highly correlated)
    X_lstm     = X_train[::4].reshape(-1, 1, n_features).astype("float32")
    y_lstm     = y_train[::4].reshape(-1, 1).astype("float32")
    X_val_3d   = X_val.reshape(-1, 1, n_features).astype("float32")
    y_val_3d   = y_val.reshape(-1, 1).astype("float32")

    val_dataset = Dataset(X_val_3d, y_val_3d)

    net = NeuralNetRegressor(
        module=LSTMForecast,
        module__input_size=n_features,
        criterion=nn.L1Loss,
        optimizer=torch.optim.Adam,
        optimizer__lr=0.001,
        max_epochs=30,
        batch_size=512,
        train_split=lambda X, y: (Dataset(X, y), val_dataset),
        device=device,
        verbose=0,
        iterator_train__num_workers=0,
        iterator_valid__num_workers=0,
        callbacks=[EarlyStopping(patience=5, monitor="valid_loss")],
    )
    net.fit(X_lstm, y_lstm)

    X_test_3d = X_test.reshape(-1, 1, n_features).astype("float32")
    metrics   = evaluate_model(y_test, net.predict(X_test_3d).flatten())

    with mlflow.start_run(run_name="LSTM_24h"):
        mlflow.log_params({"hidden_size": 64, "dropout": 0.1, "lr": 0.001,
                           "patience": 5, "station": station_id})
        mlflow.log_metrics(metrics)

    # Export LSTM to ONNX
    pytorch_model = net.module_
    pytorch_model.eval()
    dummy = torch.zeros(1, 1, n_features)
    torch.onnx.export(
        pytorch_model, dummy, _tmp_onnx(station_id, "lstm"),
        input_names=["lstm_input"],
        output_names=["output"],
        dynamic_axes={"lstm_input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    print(f"  LSTM  MAE={metrics['MAE']:.4f}  epochs={len(net.history)}  device={device}")


# ── Task 3: Evaluate all 5 temp models ───────────────────────────────────────
def _evaluate(**context):
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

    df = pd.DataFrame(results)
    print(f"\nStation {station_id} best: {df.loc[df['MAE'].idxmin(), 'model']}")


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

    tmp_keys = {
        "linear_regression": ("Linear Regression", False),
        "ridge_regression":  ("Ridge Regression",  False),
        "random_forest":     ("Random Forest",      False),
        "xgboost":           ("XGBoost",            False),
        "lstm":              ("LSTM",               True),
    }

    trained = {}
    for key, (name, is_lstm) in tmp_keys.items():
        path = _tmp_onnx(station_id, key)
        if not os.path.exists(path):
            continue
        preds = _onnx_predict(path, X_3d if is_lstm else X_f)
        mae   = evaluate_model(y_test, preds)["MAE"]
        trained[key] = (name, mae, is_lstm)

    if not trained:
        raise RuntimeError("No trained models found — all tmp ONNX missing")

    best_key           = min(trained, key=lambda k: trained[k][1])
    best_name, new_mae, best_is_lstm = trained[best_key]
    print(f"Best new model: {best_name}  MAE={new_mae:.4f}")

    # Load production model for comparison
    models_dir = _models_dir(station_id)
    registry   = f"{models_dir}/active_model.json"
    prod_mae   = None
    old_info   = None

    if os.path.exists(registry):
        with open(registry) as f:
            old_info = json.load(f)
        onnx_prod = f"{models_dir}/onnx/{old_info['onnx_file']}"
        if os.path.exists(onnx_prod):
            X_in       = X_3d if old_info.get("is_lstm") else X_f
            preds_prod = _onnx_predict(onnx_prod, X_in)
            if preds_prod.shape[0] != y_test.shape[0]:
                print("  Prod model incompatible shape — treating as first deploy")
                old_info = None
            else:
                prod_mae = evaluate_model(y_test, preds_prod)["MAE"]

    prod_str = (f"MAE={prod_mae:.4f}  ({old_info['train_start']} → {old_info['train_end']})"
                if prod_mae is not None else "N/A (first deploy)")
    print(f"  Prod: {prod_str}")

    if prod_mae is None or new_mae < prod_mae:
        onnx_filename = f"{best_key}_{train_start}_{train_end}.onnx"
        onnx_dest     = f"{models_dir}/onnx/{onnx_filename}"
        shutil.copy2(_tmp_onnx(station_id, best_key), onnx_dest)

        info = {
            "onnx_file":     onnx_filename,
            "model_key":     best_key,
            "station_id":    station_id,
            "train_start":   train_start,
            "train_end":     train_end,
            "is_lstm":       best_is_lstm,
            "forecast_hour": FORECAST_HOUR,
            "n_features":    n,
        }
        with open(registry, "w") as f:
            json.dump(info, f, indent=2)
        with open(f"{models_dir}/feature_columns.json", "w") as f:
            json.dump(meta["feature_cols"], f)

        status = "DEPLOYED"
        delta  = round(prod_mae - new_mae, 4) if prod_mae is not None else None
        print(f"  → DEPLOYED  {onnx_dest}")
    else:
        status = "NOT_DEPLOYED"
        delta  = round(prod_mae - new_mae, 4)
        print(f"  → NOT DEPLOYED  new {new_mae:.4f} >= prod {prod_mae:.4f}")

    # Clean up temp ONNX files
    for key in tmp_keys:
        p = _tmp_onnx(station_id, key)
        if os.path.exists(p):
            os.remove(p)

    # Append to report
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = f"{RESULTS_DIR}/forecast_24h_results.csv"
    row = {
        "station_id":  station_id,
        "train_start": train_start,
        "train_end":   train_end,
        "best_model":  best_name,
        "new_mae":     new_mae,
        "prod_mae":    prod_mae,
        "mae_delta":   delta,
        "status":      status,
        "run_date":    datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
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

    feat_task  = PythonOperator(task_id="feature_engineering",  python_callable=_feature_engineering)
    linear_task= PythonOperator(task_id="train_linear",         python_callable=_train_linear)
    ridge_task = PythonOperator(task_id="train_ridge",          python_callable=_train_ridge)
    rf_task    = PythonOperator(task_id="train_random_forest",  python_callable=_train_random_forest)
    xgb_task   = PythonOperator(task_id="train_xgboost",        python_callable=_train_xgboost)
    lstm_task  = PythonOperator(task_id="train_lstm",           python_callable=_train_lstm)
    eval_task  = PythonOperator(task_id="evaluate",             python_callable=_evaluate)
    deploy_task= PythonOperator(task_id="compare_and_deploy",   python_callable=_compare_and_deploy)

    feat_task >> [linear_task, ridge_task, rf_task, xgb_task, lstm_task]
    [linear_task, ridge_task, rf_task, xgb_task, lstm_task] >> eval_task
    eval_task >> deploy_task

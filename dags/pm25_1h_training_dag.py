"""
PM2.5 T+1h Training DAG
=======================
Daily training pipeline for next-hour PM2.5 models.

This DAG is self-contained like pm25_24h_training_dag.py: it loads
pm25_raw_hourly, builds T+1h features, trains regression and LSTM models,
compares against the active ONNX model, deploys the best ONNX artifact,
and publishes it to Triton.

Scheduled and default manual runs train all configured stations. Manual runs
can optionally pass {"station_ids": [56, 57]} or {"station_id": 56}.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import pendulum
from airflow import DAG
from airflow.exceptions import AirflowSkipException
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule

SRC = "/app/src"
MODELS_DIR = "/app/models"
RESULTS_DIR = "/app/results"
DEFAULT_DB_URL = "postgresql://postgres:postgres@postgres:5432/pm25"
BANGKOK_TIMEZONE_NAME = "Asia/Bangkok"
STATION_IDS = [56, 57, 58, 59, 61]
FORECAST_HOUR = 1
RANDOM_STATE = 42


def _manual_station_ids(context) -> set[int] | None:
    dag_run = context.get("dag_run")
    if not dag_run:
        return None

    run_type = getattr(dag_run.run_type, "value", dag_run.run_type)
    if str(run_type).lower() != "manual":
        return None

    conf = dag_run.conf or {}
    if "station_ids" in conf:
        selected = conf["station_ids"]
    elif "station_id" in conf:
        selected = [conf["station_id"]]
    else:
        return None

    if isinstance(selected, (str, int)):
        selected = [selected]

    station_ids = {int(station_id) for station_id in selected}
    unsupported = sorted(station_ids - set(STATION_IDS))
    if unsupported:
        raise ValueError(f"Unsupported station_ids={unsupported}; expected subset of {STATION_IDS}")
    return station_ids


def _get_splits():
    from dateutil.relativedelta import relativedelta

    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    data_start = today - relativedelta(years=3, months=6)
    val_start = today - relativedelta(months=6)
    test_start = today - relativedelta(months=3)
    return {
        "data_start": data_start.strftime("%Y-%m-%d"),
        "train_end": (val_start - relativedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
        "val_start": val_start.strftime("%Y-%m-%d"),
        "val_end": (test_start - relativedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
        "test_start": test_start.strftime("%Y-%m-%d"),
        "test_end": today.strftime("%Y-%m-%d"),
    }


def _load_station_hourly(station_id: int, db_url: str, data_start: str):
    import pandas as pd
    import sqlalchemy

    engine = sqlalchemy.create_engine(db_url)
    query = sqlalchemy.text("""
        SELECT timestamp AS datetime, pm25
        FROM pm25_raw_hourly
        WHERE station_id = :station_id
          AND timestamp >= :data_start
          AND pm25 IS NOT NULL
        ORDER BY timestamp
    """)
    try:
        with engine.connect() as conn:
            result = conn.execute(
                query,
                {"station_id": station_id, "data_start": data_start},
            )
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
    finally:
        engine.dispose()

    if df.empty:
        return df

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(None)
    df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    full_range = pd.date_range(df["datetime"].min(), df["datetime"].max(), freq="h")
    df = df.set_index("datetime").reindex(full_range).rename_axis("datetime").reset_index()
    df["pm25"] = df["pm25"].ffill().bfill().clip(lower=0, upper=500)
    return df


def _export_onnx(model, model_key: str, n_features: int, output_path: str, is_lstm: bool = False):
    if is_lstm:
        import torch

        model.eval()
        dummy = torch.zeros(1, 1, n_features)
        torch.onnx.export(
            model,
            dummy,
            output_path,
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

    print(f"  ONNX saved -> {output_path}")


def _active_model_rmse(registry: str, models_dir: str, X_test, y_test, n_features: int):
    if not os.path.exists(registry):
        return None

    with open(registry) as f:
        old_info = json.load(f)

    onnx_prod = os.path.join(models_dir, "onnx", old_info["onnx_file"])
    if not os.path.exists(onnx_prod):
        return None

    import onnxruntime as rt
    import numpy as np

    sys.path.insert(0, SRC)
    from evaluate import evaluate_model

    sess = rt.InferenceSession(onnx_prod, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    X_in = X_test.reshape(-1, 1, n_features) if old_info.get("input_shape") == "3d" else X_test
    preds_prod = sess.run([out_name], {in_name: X_in.astype(np.float32)})[0].flatten()
    if preds_prod.shape[0] != y_test.shape[0]:
        return None
    return evaluate_model(y_test, preds_prod)["RMSE"]


def _publish_to_triton(
    onnx_path: str,
    station_id: int,
    n_features: int,
    is_lstm: bool,
    triton_repo: str = "/app/triton_model_repo",
):
    import shutil

    model_name = f"pm25_{station_id}_1h"
    model_dir = os.path.join(triton_repo, model_name)
    version_dir = os.path.join(model_dir, "1")
    os.makedirs(version_dir, exist_ok=True)

    dest_onnx = os.path.join(version_dir, "model.onnx")
    shutil.copy2(onnx_path, dest_onnx)
    print(f"  Copied ONNX to Triton: {dest_onnx}")

    input_name = "lstm_input" if is_lstm else "float_input"
    output_name = "output" if is_lstm else "variable"
    dims = f"[ 1, {n_features} ]" if is_lstm else f"[ {n_features} ]"
    config_content = f'''name: "{model_name}"
backend: "onnxruntime"
max_batch_size: 32

input [
  {{
    name: "{input_name}"
    data_type: TYPE_FP32
    dims: {dims}
  }}
]

output [
  {{
    name: "{output_name}"
    data_type: TYPE_FP32
    dims: [ 1 ]
  }}
]

dynamic_batching {{ }}
'''

    config_path = os.path.join(model_dir, "config.pbtxt")
    with open(config_path, "w") as f:
        f.write(config_content)
    print(f"  Created Triton config: {config_path}")
    print(f"  Model '{model_name}' published to Triton")


def _train_station(station_id: int, **context):
    sys.path.insert(0, SRC)

    import mlflow
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    from xgboost import XGBRegressor

    from evaluate import evaluate_model
    from hourly_forecast import build_hourly_supervised_features

    selected_station_ids = _manual_station_ids(context)
    if selected_station_ids is not None and station_id not in selected_station_ids:
        raise AirflowSkipException(
            f"Manual run selected stations {sorted(selected_station_ids)}; skipping station {station_id}"
        )

    print(f"\n{'=' * 60}")
    print(f"  STATION {station_id} - T+1h Forecast")
    print(f"{'=' * 60}")

    db_url = os.environ.get("PM25_DB_URL", DEFAULT_DB_URL)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(f"pm25_1h_station_{station_id}")

    splits = _get_splits()
    hourly = _load_station_hourly(station_id, db_url, splits["data_start"])
    if hourly.empty:
        raise ValueError(f"No data in pm25_raw_hourly for station {station_id}")

    feat_df, feature_cols, target_col = build_hourly_supervised_features(
        hourly,
        forecast_hour=FORECAST_HOUR,
    )
    n_features = len(feature_cols)

    train_df = feat_df[feat_df["datetime"] < splits["val_start"]]
    val_df = feat_df[(feat_df["datetime"] >= splits["val_start"]) & (feat_df["datetime"] < splits["test_start"])]
    test_df = feat_df[feat_df["datetime"] >= splits["test_start"]]

    print(f"  Rows -> train:{len(train_df)} val:{len(val_df)} test:{len(test_df)}")
    print(f"  Features: {n_features} | Target: {target_col}")
    if len(train_df) < 200 or len(val_df) < 48 or len(test_df) < 48:
        raise ValueError(f"Not enough data: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df[target_col].values.astype(np.float32)
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df[target_col].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df[target_col].values.astype(np.float32)

    train_start = train_df["datetime"].min().strftime("%Y-%m-%d")
    train_end = train_df["datetime"].max().strftime("%Y-%m-%d")
    tscv = TimeSeriesSplit(n_splits=3)
    n_jobs = int(os.environ.get("GRID_N_JOBS", "-1"))
    trained = {}
    results = []

    with mlflow.start_run(run_name="LinearRegression_1h"):
        model = LinearRegression().fit(X_train, y_train)
        metrics = evaluate_model(y_test, model.predict(X_test))
        val_mae = evaluate_model(y_val, model.predict(X_val))["MAE"]
        mlflow.log_params({"model": "LinearRegression", "station": station_id, "forecast_hour": FORECAST_HOUR})
        mlflow.log_metrics({**metrics, "val_MAE": val_mae})
        trained["linear_regression"] = ("Linear Regression", model, metrics, False)
        results.append({"model": "Linear Regression", **metrics, "val_MAE": val_mae})

    with mlflow.start_run(run_name="Ridge_1h"):
        grid = GridSearchCV(
            Ridge(),
            {"alpha": [0.1, 1.0, 10.0, 100.0]},
            cv=tscv,
            scoring="neg_mean_absolute_error",
            n_jobs=n_jobs,
        )
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        metrics = evaluate_model(y_test, model.predict(X_test))
        val_mae = evaluate_model(y_val, model.predict(X_val))["MAE"]
        mlflow.log_params({**grid.best_params_, "station": station_id, "forecast_hour": FORECAST_HOUR})
        mlflow.log_metrics({**metrics, "val_MAE": val_mae})
        trained["ridge_regression"] = ("Ridge Regression", model, metrics, False)
        results.append({"model": "Ridge Regression", **metrics, "val_MAE": val_mae})

    with mlflow.start_run(run_name="RandomForest_1h"):
        grid = GridSearchCV(
            RandomForestRegressor(random_state=RANDOM_STATE),
            {"n_estimators": [100], "max_depth": [5, 10], "min_samples_leaf": [4]},
            cv=tscv,
            scoring="neg_mean_absolute_error",
            n_jobs=n_jobs,
        )
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        metrics = evaluate_model(y_test, model.predict(X_test))
        val_mae = evaluate_model(y_val, model.predict(X_val))["MAE"]
        mlflow.log_params({**grid.best_params_, "station": station_id, "forecast_hour": FORECAST_HOUR})
        mlflow.log_metrics({**metrics, "val_MAE": val_mae})
        trained["random_forest"] = ("Random Forest", model, metrics, False)
        results.append({"model": "Random Forest", **metrics, "val_MAE": val_mae})

    with mlflow.start_run(run_name="XGBoost_1h"):
        grid = GridSearchCV(
            XGBRegressor(random_state=RANDOM_STATE, verbosity=0, n_jobs=1),
            {"n_estimators": [100], "max_depth": [3, 5], "learning_rate": [0.1]},
            cv=tscv,
            scoring="neg_mean_absolute_error",
            n_jobs=n_jobs,
        )
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        metrics = evaluate_model(y_test, model.predict(X_test))
        val_mae = evaluate_model(y_val, model.predict(X_val))["MAE"]
        mlflow.log_params({**grid.best_params_, "station": station_id, "forecast_hour": FORECAST_HOUR})
        mlflow.log_metrics({**metrics, "val_MAE": val_mae})
        trained["xgboost"] = ("XGBoost", model, metrics, False)
        results.append({"model": "XGBoost", **metrics, "val_MAE": val_mae})

    with mlflow.start_run(run_name="LSTM_1h"):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        class LSTMForecast(nn.Module):
            def __init__(self, input_size=n_features, hidden_size=64, num_layers=2, dropout=0.1):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

        device = "cpu" if os.environ.get("PYTORCH_DEVICE") == "cpu" else ("mps" if torch.backends.mps.is_available() else "cpu")
        torch.set_num_threads(1)
        X_tr = torch.FloatTensor(X_train[::4].reshape(-1, 1, n_features))
        y_tr = torch.FloatTensor(y_train[::4])
        X_va = torch.FloatTensor(X_val.reshape(-1, 1, n_features))
        y_va = torch.FloatTensor(y_val)
        loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=512, shuffle=False)

        model = LSTMForecast().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.L1Loss()
        best_state = None
        best_val_loss = float("inf")
        patience_left = 5
        epochs_done = 0

        for epoch in range(30):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb).squeeze(-1), yb)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_va.to(device)).squeeze(-1), y_va.to(device)).item()
            epochs_done = epoch + 1
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_left = 5
            else:
                patience_left -= 1
                if patience_left == 0:
                    break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            preds = model(torch.FloatTensor(X_test.reshape(-1, 1, n_features)).to(device)).cpu().numpy()
            val_preds = model(X_va.to(device)).cpu().numpy()
        metrics = evaluate_model(y_test, preds)
        val_mae = evaluate_model(y_val, val_preds)["MAE"]
        mlflow.log_params({"hidden_size": 64, "dropout": 0.1, "lr": 0.001, "patience": 5, "station": station_id, "forecast_hour": FORECAST_HOUR})
        mlflow.log_metrics({**metrics, "val_MAE": val_mae})
        trained["lstm"] = ("LSTM", model.cpu(), metrics, True)
        results.append({"model": "LSTM", **metrics, "val_MAE": val_mae, "epochs": epochs_done})

    results_df = pd.DataFrame(results)
    best_key = min(trained, key=lambda k: trained[k][2]["RMSE"])
    best_name, best_model, best_metrics, is_lstm = trained[best_key]
    new_rmse = best_metrics["RMSE"]
    new_mae = best_metrics["MAE"]
    print(results_df[["model", "MAE", "RMSE", "R2", "val_MAE"]].to_string(index=False))
    print(f"  Best: {best_name} RMSE={new_rmse:.4f} MAE={new_mae:.4f}")

    models_dir = os.path.join(MODELS_DIR, f"station_{station_id}_1h")
    os.makedirs(os.path.join(models_dir, "onnx"), exist_ok=True)
    registry = os.path.join(models_dir, "active_model.json")
    prod_rmse = _active_model_rmse(registry, models_dir, X_test, y_test, n_features)

    if prod_rmse is None or new_rmse < prod_rmse:
        onnx_filename = f"{best_key}_{train_start}_{train_end}.onnx"
        onnx_path = os.path.join(models_dir, "onnx", onnx_filename)
        _export_onnx(best_model, best_key, n_features, onnx_path, is_lstm)
        info = {
            "onnx_file": onnx_filename,
            "model_key": best_key,
            "station_id": station_id,
            "train_start": train_start,
            "train_end": train_end,
            "backend": "onnx",
            "input_shape": "3d" if is_lstm else "2d",
            "forecast_hour": FORECAST_HOUR,
            "n_features": n_features,
            "input_name": "lstm_input" if is_lstm else "float_input",
            "output_name": "output" if is_lstm else "variable",
        }
        with open(registry, "w") as f:
            json.dump(info, f, indent=2)
        with open(os.path.join(models_dir, "feature_columns.json"), "w") as f:
            json.dump(feature_cols, f)
        _publish_to_triton(onnx_path, station_id, n_features, is_lstm)
        status = "DEPLOYED"
        delta_rmse = round(prod_rmse - new_rmse, 4) if prod_rmse is not None else None
    else:
        status = "NOT_DEPLOYED"
        delta_rmse = round(prod_rmse - new_rmse, 4)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(RESULTS_DIR, "forecast_1h_results.csv")
    row = {
        "station_id": station_id,
        "train_start": train_start,
        "train_end": train_end,
        "best_model": best_name,
        "new_rmse": new_rmse,
        "new_mae": new_mae,
        "prod_rmse": prod_rmse,
        "rmse_delta": delta_rmse,
        "status": status,
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    exists = os.path.exists(report_path)
    pd.DataFrame([row]).to_csv(report_path, mode="a", header=not exists, index=False)
    print(f"  Report: {report_path}")
    return {
        "station_id": station_id,
        "rows": len(results_df),
        "models": results_df["model"].tolist(),
        "best_model": best_name,
        "status": status,
    }


with DAG(
    dag_id="pm25_1h_training",
    description="Train and deploy T+1h PM2.5 forecast models",
    schedule="0 0 * * *",
    start_date=pendulum.datetime(2024, 1, 1, tz=BANGKOK_TIMEZONE_NAME),
    catchup=False,
    max_active_runs=1,
    tags=["pm25", "ml", "1h-forecast", "postgresql"],
) as dag:
    start = EmptyOperator(task_id="start")
    done = EmptyOperator(task_id="done", trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)

    train_tasks = [
        PythonOperator(
            task_id=f"train_station_{station_id}_1h",
            python_callable=_train_station,
            op_kwargs={"station_id": station_id},
        )
        for station_id in STATION_IDS
    ]

    start >> train_tasks >> done

# ── DISABLED ─────────────────────────────────────────────────────────────────
# Not in use — replaced by pm25_24h_training + pm25_24h_pipeline (PostgreSQL).
# Listed in dags/.airflowignore so Airflow will not load this file.
# ─────────────────────────────────────────────────────────────────────────────
"""
PM2.5 Per-Station Training DAG
================================
Trains 5 models for a single monitoring station and deploys the best one.

Trigger with:
    {"station_id": 63}          -- train station 63
    {"station_id": 65}          -- train station 65

Task graph:
    feature_engineering
        ├── train_baseline
        ├── train_ridge
        ├── train_random_forest
        ├── train_xgboost
        └── train_lstm
                └── (all 5 join) ── evaluate ── compare_and_deploy

Data   : /app/data/raw/archived/station_{id}_long.csv
Models : /app/models/station_{id}/onnx/
Triton : pm25_{id}
Report : /app/results/multi_station_comparison.csv
"""

from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models.param import Param

SRC        = "/app/src"
DATA_DIR   = "/app/data/raw/archived"
MODELS_DIR = "/app/models"
RESULTS_DIR= "/app/results"
TRITON_REPO= "/app/triton_model_repo"
PROCESSED  = "/app/data/processed"

TRAIN_END  = "2024-12-31"
TEST_START = "2025-01-01"
LAG_DAYS        = [1, 2, 3, 5, 7]
ROLLING_WINDOWS = [3, 7, 14]
RANDOM_STATE    = 42


# ── Helpers ───────────────────────────────────────────────────────────────────
def _station_dir(station_id):
    import os
    d = f"{MODELS_DIR}/station_{station_id}"
    os.makedirs(f"{d}/onnx", exist_ok=True)
    return d


def _tmp_onnx(station_id, key):
    return f"{MODELS_DIR}/station_{station_id}/_tmp_{key}.onnx"


def _get_station_id(context):
    return context["params"]["station_id"]


def _onnx_predict(onnx_path, X):
    import onnxruntime as rt
    session     = rt.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name  = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session.run([output_name], {input_name: X.astype("float32")})[0].flatten()


def _setup_mlflow(station_id):
    import os, mlflow
    uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(f"pm25_station_{station_id}")


def _load_station_data(station_id):
    import sys, pandas as pd, numpy as np
    sys.path.insert(0, SRC)
    from preprocessing import preprocess_pipeline
    from feature_engineering import build_features, get_feature_columns

    path  = f"{DATA_DIR}/station_{station_id}_long.csv"
    df    = pd.read_csv(path, parse_dates=["Date_Time"])
    pm    = df[df["Parameter"] == "PM2.5"].copy()
    pm["date"] = pm["Date_Time"].dt.normalize()
    daily = (pm.groupby("date")["Value"].mean()
               .reset_index().rename(columns={"Value": "pm25"}))
    daily = daily[daily["pm25"] > 0].reset_index(drop=True)
    daily["date"] = pd.to_datetime(daily["date"])

    train_df = daily[daily["date"] <= TRAIN_END].copy()
    test_df  = daily[daily["date"] >= TEST_START].copy()

    train_clean = preprocess_pipeline(train_df)
    test_clean  = preprocess_pipeline(test_df)
    train_feat  = build_features(train_clean, LAG_DAYS, ROLLING_WINDOWS)
    test_feat   = build_features(test_clean,  LAG_DAYS, ROLLING_WINDOWS)
    feature_cols = get_feature_columns(train_feat)

    X_train = train_feat[feature_cols].values
    y_train = train_feat["pm25"].values
    X_test  = test_feat[feature_cols].values
    y_test  = test_feat["pm25"].values
    train_start = str(train_feat["date"].min().date())
    train_end   = str(train_feat["date"].max().date())

    return X_train, y_train, X_test, y_test, feature_cols, train_start, train_end


# ── Task 1: Feature engineering ──────────────────────────────────────────────
def _feature_engineering(**context):
    import sys, os, json
    sys.path.insert(0, SRC)
    station_id = _get_station_id(context)

    X_train, y_train, X_test, y_test, feature_cols, train_start, train_end = \
        _load_station_data(station_id)

    os.makedirs(f"{PROCESSED}/station_{station_id}", exist_ok=True)
    import pandas as pd, numpy as np
    pd.DataFrame(X_train).to_parquet(f"{PROCESSED}/station_{station_id}/X_train.parquet", index=False)
    pd.DataFrame(y_train, columns=["pm25"]).to_parquet(f"{PROCESSED}/station_{station_id}/y_train.parquet", index=False)
    pd.DataFrame(X_test).to_parquet(f"{PROCESSED}/station_{station_id}/X_test.parquet",  index=False)
    pd.DataFrame(y_test,  columns=["pm25"]).to_parquet(f"{PROCESSED}/station_{station_id}/y_test.parquet",  index=False)

    meta = {"feature_cols": feature_cols, "train_start": train_start, "train_end": train_end}
    with open(f"{PROCESSED}/station_{station_id}/meta.json", "w") as f:
        json.dump(meta, f)

    print(f"Station {station_id}: {len(feature_cols)} features, "
          f"train {X_train.shape}, test {X_test.shape}")
    print(f"Training date range: {train_start} → {train_end}")


def _load_arrays(station_id):
    import json, pandas as pd, numpy as np
    base = f"{PROCESSED}/station_{station_id}"
    X_train = pd.read_parquet(f"{base}/X_train.parquet").values
    y_train = pd.read_parquet(f"{base}/y_train.parquet")["pm25"].values
    X_test  = pd.read_parquet(f"{base}/X_test.parquet").values
    y_test  = pd.read_parquet(f"{base}/y_test.parquet")["pm25"].values
    with open(f"{base}/meta.json") as f:
        meta = json.load(f)
    return X_train, y_train, X_test, y_test, meta


# ── Task 2a–2e: Train each model ──────────────────────────────────────────────
def _train_baseline(**context):
    import sys, mlflow
    sys.path.insert(0, SRC)
    from sklearn.linear_model import LinearRegression
    from evaluate import evaluate_model, print_metrics
    from export_onnx import export_sklearn

    station_id = _get_station_id(context)
    _setup_mlflow(station_id)
    X_train, y_train, X_test, y_test, meta = _load_arrays(station_id)

    with mlflow.start_run(run_name="LinearRegression"):
        model = LinearRegression()
        model.fit(X_train, y_train)
        metrics = evaluate_model(y_test, model.predict(X_test))
        print_metrics("LinearRegression", metrics)
        mlflow.log_params({"model_type": "LinearRegression", "station": station_id})
        mlflow.log_metrics(metrics)

    export_sklearn(model, "baseline_linear_regression", _station_dir(station_id),
                   output_path=_tmp_onnx(station_id, "baseline_linear_regression"))


def _train_ridge(**context):
    import sys, mlflow
    sys.path.insert(0, SRC)
    from data_loader import load_config
    from train import train_with_tuning
    from evaluate import evaluate_model, print_metrics
    from export_onnx import export_sklearn

    station_id = _get_station_id(context)
    _setup_mlflow(station_id)
    X_train, y_train, X_test, y_test, meta = _load_arrays(station_id)
    config = load_config(f"/app/configs/config.yaml")

    with mlflow.start_run(run_name="Ridge"):
        model, best = train_with_tuning("Ridge", config["models"]["ridge"]["params"],
                                        X_train, y_train, RANDOM_STATE)
        metrics = evaluate_model(y_test, model.predict(X_test))
        print_metrics("Ridge", metrics)
        mlflow.log_params({**best, "station": station_id})
        mlflow.log_metrics(metrics)

    export_sklearn(model, "ridge_regression", _station_dir(station_id),
                   output_path=_tmp_onnx(station_id, "ridge_regression"))


def _train_random_forest(**context):
    import sys, mlflow
    sys.path.insert(0, SRC)
    from data_loader import load_config
    from train import train_with_tuning
    from evaluate import evaluate_model, print_metrics
    from export_onnx import export_sklearn

    station_id = _get_station_id(context)
    _setup_mlflow(station_id)
    X_train, y_train, X_test, y_test, meta = _load_arrays(station_id)
    config = load_config(f"/app/configs/config.yaml")
    rf_params = config["models"]["random_forest"]["params"].copy()
    rf_params.pop("random_state", None)

    with mlflow.start_run(run_name="RandomForest"):
        model, best = train_with_tuning("RandomForestRegressor", rf_params,
                                        X_train, y_train, RANDOM_STATE)
        metrics = evaluate_model(y_test, model.predict(X_test))
        print_metrics("RandomForest", metrics)
        mlflow.log_params({**best, "station": station_id})
        mlflow.log_metrics(metrics)

    export_sklearn(model, "random_forest", _station_dir(station_id),
                   output_path=_tmp_onnx(station_id, "random_forest"))


def _train_xgboost(**context):
    import sys, mlflow
    sys.path.insert(0, SRC)
    from data_loader import load_config
    from train import train_with_tuning
    from evaluate import evaluate_model, print_metrics
    from export_onnx import export_xgboost

    station_id = _get_station_id(context)
    _setup_mlflow(station_id)
    X_train, y_train, X_test, y_test, meta = _load_arrays(station_id)
    config = load_config(f"/app/configs/config.yaml")
    xgb_params = config["models"]["xgboost"]["params"].copy()
    xgb_params.pop("random_state", None)

    with mlflow.start_run(run_name="XGBoost"):
        model, best = train_with_tuning("XGBRegressor", xgb_params,
                                        X_train, y_train, RANDOM_STATE)
        metrics = evaluate_model(y_test, model.predict(X_test))
        print_metrics("XGBoost", metrics)
        mlflow.log_params({**best, "station": station_id})
        mlflow.log_metrics(metrics)

    export_xgboost(model, _station_dir(station_id),
                   output_path=_tmp_onnx(station_id, "xgboost"))


def _train_lstm(**context):
    import sys, mlflow, numpy as np
    sys.path.insert(0, SRC)
    from data_loader import load_config
    from lstm_model import train_lstm_with_tuning
    from evaluate import evaluate_model, print_metrics
    from export_onnx import export_lstm as export_lstm_onnx

    station_id = _get_station_id(context)
    _setup_mlflow(station_id)
    X_train, y_train, X_test, y_test, meta = _load_arrays(station_id)
    config = load_config(f"/app/configs/config.yaml")

    with mlflow.start_run(run_name="LSTM"):
        model, best = train_lstm_with_tuning(X_train, y_train,
                                             config["models"]["lstm"]["params"],
                                             RANDOM_STATE)
        X_test_3d = X_test.reshape(X_test.shape[0], 1, X_test.shape[1]).astype(np.float32)
        metrics   = evaluate_model(y_test, model.predict(X_test_3d).flatten())
        print_metrics("LSTM", metrics)
        mlflow.log_params({**best, "station": station_id})
        mlflow.log_metrics(metrics)

    export_lstm_onnx(model, _station_dir(station_id),
                     output_path=_tmp_onnx(station_id, "lstm"))


# ── Task 3: Evaluate ──────────────────────────────────────────────────────────
def _evaluate(**context):
    import sys, os
    import pandas as pd
    sys.path.insert(0, SRC)
    from evaluate import evaluate_model, print_metrics

    station_id = _get_station_id(context)
    _, _, X_test, y_test, _ = _load_arrays(station_id)
    X_f  = X_test.astype("float32")
    X_3d = X_test.reshape(X_test.shape[0], 1, X_test.shape[1]).astype("float32")

    tmp_keys = {
        "Linear Regression": ("baseline_linear_regression", False),
        "Ridge Regression":  ("ridge_regression",           False),
        "Random Forest":     ("random_forest",              False),
        "XGBoost":           ("xgboost",                    False),
        "LSTM":              ("lstm",                       True),
    }

    results = []
    for name, (key, is_lstm) in tmp_keys.items():
        path = _tmp_onnx(station_id, key)
        if not os.path.exists(path):
            continue
        preds   = _onnx_predict(path, X_3d if is_lstm else X_f)
        metrics = evaluate_model(y_test, preds)
        print_metrics(name, metrics)
        results.append({"model": name, **metrics})

    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame(results)
    print(f"\nStation {station_id} — Best: {df.loc[df['MAE'].idxmin(), 'model']}")


# ── Task 4: Compare vs production, deploy if better ──────────────────────────
def _compare_and_deploy(**context):
    import sys, os, json, shutil
    import numpy as np, pandas as pd
    sys.path.insert(0, SRC)
    from evaluate import evaluate_model
    import onnxruntime as rt
    import triton_utils as _tu
    from triton_utils import publish_to_triton

    station_id = _get_station_id(context)
    _, _, X_test, y_test, meta = _load_arrays(station_id)
    X_f  = X_test.astype("float32")
    X_3d = X_test.reshape(X_test.shape[0], 1, X_test.shape[1]).astype("float32")

    train_start = meta["train_start"]
    train_end   = meta["train_end"]

    tmp_keys = {
        "baseline_linear_regression": ("Linear Regression", False),
        "ridge_regression":           ("Ridge Regression",  False),
        "random_forest":              ("Random Forest",     False),
        "xgboost":                    ("XGBoost",           False),
        "lstm":                       ("LSTM",              True),
    }

    trained = {}
    for key, (name, is_lstm) in tmp_keys.items():
        path = _tmp_onnx(station_id, key)
        if not os.path.exists(path):
            continue
        preds = _onnx_predict(path, X_3d if is_lstm else X_f)
        mae   = evaluate_model(y_test, preds)["MAE"]
        trained[key] = (name, mae, is_lstm)

    best_key           = min(trained, key=lambda k: trained[k][1])
    best_name, new_mae, best_is_lstm = trained[best_key]
    print(f"Best new model: {best_name}  MAE={new_mae:.4f}")

    # Load production model via ONNX Runtime
    registry = f"{MODELS_DIR}/station_{station_id}/active_model.json"
    prod_mae  = None
    old_info  = None
    if os.path.exists(registry):
        with open(registry) as f:
            old_info = json.load(f)
        onnx_prod = f"{MODELS_DIR}/station_{station_id}/onnx/{old_info['onnx_file']}"
        if os.path.exists(onnx_prod):
            X_in  = X_3d if old_info.get("input_shape") == "3d" else X_f
            preds = _onnx_predict(onnx_prod, X_in)
            prod_mae = evaluate_model(y_test, preds)["MAE"]

    prod_str = f"MAE={prod_mae:.4f}  ({old_info['train_start']} → {old_info['train_end']})" \
               if prod_mae is not None else "N/A (first deploy)"
    print(f"  Prod: {prod_str}")

    if prod_mae is None or new_mae < prod_mae:
        onnx_filename = f"{best_key}_{train_start}_{train_end}.onnx"
        onnx_dest     = f"{MODELS_DIR}/station_{station_id}/onnx/{onnx_filename}"
        shutil.copy2(_tmp_onnx(station_id, best_key), onnx_dest)

        info = {
            "onnx_file":   onnx_filename,
            "model_key":   best_key,
            "station_id":  station_id,
            "train_start": train_start,
            "train_end":   train_end,
            "backend":     "onnx",
            "input_shape": "3d" if best_is_lstm else "2d",
        }
        with open(registry, "w") as f:
            json.dump(info, f, indent=2)

        triton_model_name = f"pm25_{station_id}"
        orig = _tu.TRITON_MODEL_NAME
        _tu.TRITON_MODEL_NAME = triton_model_name
        publish_to_triton(onnx_dest, TRITON_REPO, best_is_lstm)
        _tu.TRITON_MODEL_NAME = orig

        status = "DEPLOYED"
        delta  = round(prod_mae - new_mae, 4) if prod_mae is not None else None
        print(f"  → DEPLOYED  {onnx_dest}")
    else:
        status = "NOT_DEPLOYED"
        delta  = round(prod_mae - new_mae, 4)
        print(f"  → NOT DEPLOYED  new {new_mae:.4f} >= prod {prod_mae:.4f}")

    # Save feature columns
    with open(f"{MODELS_DIR}/station_{station_id}/feature_columns.json", "w") as f:
        json.dump(meta["feature_cols"], f)

    # Clean up temp ONNX files
    for key in tmp_keys:
        p = _tmp_onnx(station_id, key)
        if os.path.exists(p):
            os.remove(p)

    # Append report
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = f"{RESULTS_DIR}/multi_station_comparison.csv"
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
    print(f"  Report: {report_path}")


# ── DAG definition ────────────────────────────────────────────────────────────
with DAG(
    dag_id="pm25_station_training",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    params={
        "station_id": Param(
            63,
            type="integer",
            enum=[63, 64, 65, 66, 67],
            description="Station ID to train (63–67)",
        ),
    },
    tags=["pm25", "ml", "multi-station"],
) as dag:

    feat_task     = PythonOperator(task_id="feature_engineering",  python_callable=_feature_engineering)
    baseline_task = PythonOperator(task_id="train_baseline",       python_callable=_train_baseline)
    ridge_task    = PythonOperator(task_id="train_ridge",          python_callable=_train_ridge)
    rf_task       = PythonOperator(task_id="train_random_forest",  python_callable=_train_random_forest)
    xgb_task      = PythonOperator(task_id="train_xgboost",        python_callable=_train_xgboost)
    lstm_task     = PythonOperator(task_id="train_lstm",           python_callable=_train_lstm)
    eval_task     = PythonOperator(task_id="evaluate",             python_callable=_evaluate)
    deploy_task   = PythonOperator(task_id="compare_and_deploy",   python_callable=_compare_and_deploy)

    feat_task >> [baseline_task, ridge_task, rf_task, xgb_task, lstm_task]
    [baseline_task, ridge_task, rf_task, xgb_task, lstm_task] >> eval_task
    eval_task >> deploy_task

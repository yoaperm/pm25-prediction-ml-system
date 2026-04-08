"""
PM2.5 Training Pipeline DAG
============================
Full ML pipeline orchestrated as individual Airflow tasks.

Task graph:
    load_data
        └── feature_engineering
                ├── train_baseline
                ├── train_ridge
                ├── train_random_forest
                ├── train_xgboost
                └── train_lstm
                        └── (all 5 join) ── evaluate ── export_onnx

Intermediate data is written to data/processed/ as parquet so tasks
are isolated processes that share state only via the filesystem.

Trigger manually from the Airflow UI (schedule=None).
"""

from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

# ── Shared paths (inside Docker container) ──────────────────────────────────
SRC          = "/app/src"
CONFIG_PATH  = "/app/configs/config.yaml"
PROCESSED    = "/app/data/processed"
MODELS_DIR   = "/app/models"
RESULTS_DIR  = "/app/results"
ONNX_DIR     = "/app/models/onnx"

TRAIN_FEAT   = f"{PROCESSED}/train_features.parquet"
TEST_FEAT    = f"{PROCESSED}/test_features.parquet"
FEAT_COLS    = f"{MODELS_DIR}/feature_columns.json"


# ── Task 1: Load & preprocess → feature engineering ─────────────────────────
def _feature_engineering(**context):
    import sys, os, json
    sys.path.insert(0, SRC)
    from data_loader import load_config, load_train_test_data
    from preprocessing import preprocess_pipeline
    from feature_engineering import build_features, get_feature_columns

    config = load_config(CONFIG_PATH)
    train_df, test_df = load_train_test_data(config)

    lag_days        = config["features"]["lag_days"]
    rolling_windows = config["features"]["rolling_windows"]

    train_feat = build_features(preprocess_pipeline(train_df), lag_days, rolling_windows)
    test_feat  = build_features(preprocess_pipeline(test_df),  lag_days, rolling_windows)

    feature_cols = get_feature_columns(train_feat)
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Train: {train_feat.shape}, Test: {test_feat.shape}")

    os.makedirs(PROCESSED, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    train_feat.to_parquet(TRAIN_FEAT, index=False)
    test_feat.to_parquet(TEST_FEAT,   index=False)

    with open(FEAT_COLS, "w") as f:
        json.dump(feature_cols, f)

    print(f"Saved features to {PROCESSED}")


# ── Helper: load features & arrays ──────────────────────────────────────────
def _load_arrays():
    import json
    import pandas as pd
    train_feat   = pd.read_parquet(TRAIN_FEAT)
    test_feat    = pd.read_parquet(TEST_FEAT)
    with open(FEAT_COLS) as f:
        feature_cols = json.load(f)
    X_train = train_feat[feature_cols].values
    y_train = train_feat["pm25"].values
    X_test  = test_feat[feature_cols].values
    y_test  = test_feat["pm25"].values
    return X_train, y_train, X_test, y_test, feature_cols


# ── Helper: MLflow setup ─────────────────────────────────────────────────────
def _setup_mlflow(config):
    import os, mlflow
    uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        config.get("mlflow", {}).get("tracking_uri", "http://mlflow:5000"),
    )
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(config.get("mlflow", {}).get("experiment_name", "pm25_prediction"))


# ── Task 2a: Baseline Linear Regression ─────────────────────────────────────
def _train_baseline(**context):
    import sys, os, joblib, mlflow
    sys.path.insert(0, SRC)
    from data_loader import load_config
    from sklearn.linear_model import LinearRegression
    from evaluate import evaluate_model, print_metrics

    config = load_config(CONFIG_PATH)
    _setup_mlflow(config)
    X_train, y_train, X_test, y_test, _ = _load_arrays()

    with mlflow.start_run(run_name="LinearRegression"):
        model = LinearRegression()
        model.fit(X_train, y_train)
        metrics = evaluate_model(y_test, model.predict(X_test))
        print_metrics("LinearRegression", metrics)
        mlflow.log_params({"model_type": "LinearRegression"})
        mlflow.log_metrics(metrics)
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(model, f"{MODELS_DIR}/baseline_linear_regression.joblib")


# ── Task 2b: Ridge Regression ────────────────────────────────────────────────
def _train_ridge(**context):
    import sys, os, joblib, mlflow
    sys.path.insert(0, SRC)
    from data_loader import load_config
    from train import train_with_tuning
    from evaluate import evaluate_model, print_metrics

    config = load_config(CONFIG_PATH)
    _setup_mlflow(config)
    X_train, y_train, X_test, y_test, _ = _load_arrays()
    random_state = config.get("random_state", 42)

    with mlflow.start_run(run_name="Ridge"):
        model, best = train_with_tuning("Ridge", config["models"]["ridge"]["params"],
                                        X_train, y_train, random_state)
        metrics = evaluate_model(y_test, model.predict(X_test))
        print_metrics("Ridge", metrics)
        mlflow.log_params(best)
        mlflow.log_metrics(metrics)
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(model, f"{MODELS_DIR}/ridge_regression.joblib")


# ── Task 2c: Random Forest ───────────────────────────────────────────────────
def _train_random_forest(**context):
    import sys, os, joblib, mlflow
    sys.path.insert(0, SRC)
    from data_loader import load_config
    from train import train_with_tuning
    from evaluate import evaluate_model, print_metrics

    config = load_config(CONFIG_PATH)
    _setup_mlflow(config)
    X_train, y_train, X_test, y_test, _ = _load_arrays()
    random_state = config.get("random_state", 42)

    rf_params = config["models"]["random_forest"]["params"].copy()
    rf_params.pop("random_state", None)

    with mlflow.start_run(run_name="RandomForest"):
        model, best = train_with_tuning("RandomForestRegressor", rf_params,
                                        X_train, y_train, random_state)
        metrics = evaluate_model(y_test, model.predict(X_test))
        print_metrics("RandomForest", metrics)
        mlflow.log_params(best)
        mlflow.log_metrics(metrics)
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(model, f"{MODELS_DIR}/random_forest.joblib")


# ── Task 2d: XGBoost ─────────────────────────────────────────────────────────
def _train_xgboost(**context):
    import sys, os, joblib, mlflow
    sys.path.insert(0, SRC)
    from data_loader import load_config
    from train import train_with_tuning
    from evaluate import evaluate_model, print_metrics

    config = load_config(CONFIG_PATH)
    _setup_mlflow(config)
    X_train, y_train, X_test, y_test, _ = _load_arrays()
    random_state = config.get("random_state", 42)

    xgb_params = config["models"]["xgboost"]["params"].copy()
    xgb_params.pop("random_state", None)

    with mlflow.start_run(run_name="XGBoost"):
        model, best = train_with_tuning("XGBRegressor", xgb_params,
                                        X_train, y_train, random_state)
        metrics = evaluate_model(y_test, model.predict(X_test))
        print_metrics("XGBoost", metrics)
        mlflow.log_params(best)
        mlflow.log_metrics(metrics)
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(model, f"{MODELS_DIR}/xgboost.joblib")


# ── Task 2e: LSTM ────────────────────────────────────────────────────────────
def _train_lstm(**context):
    import sys, os, joblib, mlflow
    import torch
    sys.path.insert(0, SRC)
    from data_loader import load_config
    from lstm_model import train_lstm_with_tuning
    from evaluate import evaluate_model, print_metrics
    import numpy as np

    config = load_config(CONFIG_PATH)
    _setup_mlflow(config)
    X_train, y_train, X_test, y_test, _ = _load_arrays()
    random_state = config.get("random_state", 42)

    with mlflow.start_run(run_name="LSTM"):
        model, best = train_lstm_with_tuning(X_train, y_train,
                                             config["models"]["lstm"]["params"],
                                             random_state)
        X_test_3d   = X_test.reshape(X_test.shape[0], 1, X_test.shape[1]).astype(np.float32)
        y_pred      = model.predict(X_test_3d).flatten()
        metrics     = evaluate_model(y_test, y_pred)
        print_metrics("LSTM", metrics)
        mlflow.log_params(best)
        mlflow.log_metrics(metrics)
        os.makedirs(MODELS_DIR, exist_ok=True)
        torch.save(model.module_.state_dict(), f"{MODELS_DIR}/lstm.pt")
        joblib.dump(model, f"{MODELS_DIR}/lstm.joblib")


# ── Task 3: Evaluate all models ──────────────────────────────────────────────
def _evaluate(**context):
    import sys, os, joblib
    import numpy as np
    import pandas as pd
    sys.path.insert(0, SRC)
    from evaluate import evaluate_model, print_metrics

    _, _, X_test, y_test, _ = _load_arrays()

    model_files = {
        "Linear Regression (Baseline)": f"{MODELS_DIR}/baseline_linear_regression.joblib",
        "Ridge Regression":             f"{MODELS_DIR}/ridge_regression.joblib",
        "Random Forest":                f"{MODELS_DIR}/random_forest.joblib",
        "XGBoost":                      f"{MODELS_DIR}/xgboost.joblib",
    }

    results = []
    for name, path in model_files.items():
        model   = joblib.load(path)
        metrics = evaluate_model(y_test, model.predict(X_test))
        print_metrics(name, metrics)
        results.append({"model": name, **metrics})

    # LSTM
    lstm_model = joblib.load(f"{MODELS_DIR}/lstm.joblib")
    X_test_3d  = X_test.reshape(X_test.shape[0], 1, X_test.shape[1]).astype(np.float32)
    metrics    = evaluate_model(y_test, lstm_model.predict(X_test_3d).flatten())
    print_metrics("LSTM", metrics)
    results.append({"model": "LSTM", **metrics})

    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(f"{RESULTS_DIR}/experiment_results.csv", index=False)

    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 60)
    print(df[["model", "MAE", "RMSE", "R2"]].to_string(index=False))
    best = df.loc[df["MAE"].idxmin(), "model"]
    print(f"\nBest model (MAE): {best}")


# ── Task 4: Export all models to ONNX ───────────────────────────────────────
def _export_onnx(**context):
    import sys
    sys.path.insert(0, SRC)
    from data_loader import load_config
    from export_onnx import export_all

    config = load_config(CONFIG_PATH)
    export_all(config)
    print(f"ONNX models saved to {ONNX_DIR}")


# ── DAG definition ───────────────────────────────────────────────────────────
with DAG(
    dag_id="pm25_training_pipeline",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["pm25", "ml"],
) as dag:

    feat_task = PythonOperator(
        task_id="feature_engineering",
        python_callable=_feature_engineering,
    )

    baseline_task = PythonOperator(
        task_id="train_baseline",
        python_callable=_train_baseline,
    )
    ridge_task = PythonOperator(
        task_id="train_ridge",
        python_callable=_train_ridge,
    )
    rf_task = PythonOperator(
        task_id="train_random_forest",
        python_callable=_train_random_forest,
    )
    xgb_task = PythonOperator(
        task_id="train_xgboost",
        python_callable=_train_xgboost,
    )
    lstm_task = PythonOperator(
        task_id="train_lstm",
        python_callable=_train_lstm,
    )

    eval_task = PythonOperator(
        task_id="evaluate",
        python_callable=_evaluate,
    )

    onnx_task = PythonOperator(
        task_id="export_onnx",
        python_callable=_export_onnx,
    )

    # feature_engineering → all 5 train tasks in parallel → evaluate → export_onnx
    feat_task >> [baseline_task, ridge_task, rf_task, xgb_task, lstm_task]
    [baseline_task, ridge_task, rf_task, xgb_task, lstm_task] >> eval_task
    eval_task >> onnx_task

"""
PM2.5 Training Pipeline DAG
============================
Full ML pipeline orchestrated as individual Airflow tasks.

Task graph:
    feature_engineering
        ├── train_baseline
        ├── train_ridge
        ├── train_random_forest
        ├── train_xgboost
        └── train_lstm
                └── (all 5 join) ── evaluate ── compare_and_deploy

Each training task exports a temp ONNX file (models/_tmp_{key}.onnx).
compare_and_deploy selects the best, copies it to versioned production path,
publishes to Triton, and cleans up all temp files.
"""

from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

# ── Shared paths (inside Docker container) ──────────────────────────────────
SRC         = "/app/src"
CONFIG_PATH = "/app/configs/config.yaml"
PROCESSED   = "/app/data/processed"
MODELS_DIR  = "/app/models"
RESULTS_DIR = "/app/results"
ONNX_DIR    = "/app/models/onnx"

TRAIN_FEAT       = f"{PROCESSED}/train_features.parquet"
TEST_FEAT        = f"{PROCESSED}/test_features.parquet"
FEAT_COLS        = f"{MODELS_DIR}/feature_columns.json"
TRAIN_RANGE_FILE = f"{PROCESSED}/train_date_range.json"


# ── Task 1: Load & preprocess → feature engineering ─────────────────────────
def _feature_engineering(**context):
    import sys
    import os
    import json
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

    train_start = str(train_feat["date"].min().date())
    train_end   = str(train_feat["date"].max().date())
    with open(TRAIN_RANGE_FILE, "w") as f:
        json.dump({"train_start": train_start, "train_end": train_end}, f)

    print(f"Training date range: {train_start} → {train_end}")


# ── Helper: load feature arrays ──────────────────────────────────────────────
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
    import os
    import mlflow
    uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        config.get("mlflow", {}).get("tracking_uri", "http://mlflow:5000"),
    )
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(config.get("mlflow", {}).get("experiment_name", "pm25_prediction"))


# ── Helper: run ONNX inference ───────────────────────────────────────────────
def _onnx_predict(onnx_path, X):
    import onnxruntime as rt
    session     = rt.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name  = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session.run([output_name], {input_name: X.astype("float32")})[0].flatten()


# ── Task 2a: Baseline Linear Regression ─────────────────────────────────────
def _train_baseline(**context):
    import sys, os, mlflow
    sys.path.insert(0, SRC)
    from data_loader import load_config
    from sklearn.linear_model import LinearRegression
    from evaluate import evaluate_model, print_metrics
    from export_onnx import export_sklearn

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

    os.makedirs(ONNX_DIR, exist_ok=True)
    export_sklearn(model, "baseline_linear_regression", ONNX_DIR,
                   output_path=f"{MODELS_DIR}/_tmp_baseline_linear_regression.onnx")


# ── Task 2b: Ridge Regression ────────────────────────────────────────────────
def _train_ridge(**context):
    import sys, os, mlflow
    sys.path.insert(0, SRC)
    from data_loader import load_config
    from train import train_with_tuning
    from evaluate import evaluate_model, print_metrics
    from export_onnx import export_sklearn

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

    os.makedirs(ONNX_DIR, exist_ok=True)
    export_sklearn(model, "ridge_regression", ONNX_DIR,
                   output_path=f"{MODELS_DIR}/_tmp_ridge_regression.onnx")


# ── Task 2c: Random Forest ───────────────────────────────────────────────────
def _train_random_forest(**context):
    import sys, os, mlflow
    sys.path.insert(0, SRC)
    from data_loader import load_config
    from train import train_with_tuning
    from evaluate import evaluate_model, print_metrics
    from export_onnx import export_sklearn

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

    os.makedirs(ONNX_DIR, exist_ok=True)
    export_sklearn(model, "random_forest", ONNX_DIR,
                   output_path=f"{MODELS_DIR}/_tmp_random_forest.onnx")


# ── Task 2d: XGBoost ─────────────────────────────────────────────────────────
def _train_xgboost(**context):
    import sys, os, mlflow
    sys.path.insert(0, SRC)
    from data_loader import load_config
    from train import train_with_tuning
    from evaluate import evaluate_model, print_metrics
    from export_onnx import export_xgboost

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

    os.makedirs(ONNX_DIR, exist_ok=True)
    export_xgboost(model, ONNX_DIR,
                   output_path=f"{MODELS_DIR}/_tmp_xgboost.onnx")


# ── Task 2e: LSTM ────────────────────────────────────────────────────────────
def _train_lstm(**context):
    import sys, os, mlflow
    import numpy as np
    sys.path.insert(0, SRC)
    from data_loader import load_config
    from lstm_model import train_lstm_with_tuning
    from evaluate import evaluate_model, print_metrics
    from export_onnx import export_lstm as export_lstm_onnx

    config = load_config(CONFIG_PATH)
    _setup_mlflow(config)
    X_train, y_train, X_test, y_test, _ = _load_arrays()
    random_state = config.get("random_state", 42)

    with mlflow.start_run(run_name="LSTM"):
        model, best = train_lstm_with_tuning(X_train, y_train,
                                             config["models"]["lstm"]["params"],
                                             random_state)
        X_test_3d = X_test.reshape(X_test.shape[0], 1, X_test.shape[1]).astype(np.float32)
        metrics   = evaluate_model(y_test, model.predict(X_test_3d).flatten())
        print_metrics("LSTM", metrics)
        mlflow.log_params(best)
        mlflow.log_metrics(metrics)

    os.makedirs(ONNX_DIR, exist_ok=True)
    export_lstm_onnx(model, ONNX_DIR,
                     output_path=f"{MODELS_DIR}/_tmp_lstm.onnx")


# ── Task 3: Evaluate all temp ONNX models ────────────────────────────────────
def _evaluate(**context):
    import sys, os
    import numpy as np
    import pandas as pd
    sys.path.insert(0, SRC)
    from evaluate import evaluate_model, print_metrics

    _, _, X_test, y_test, _ = _load_arrays()
    X_test_f  = X_test.astype("float32")
    X_test_3d = X_test.reshape(X_test.shape[0], 1, X_test.shape[1]).astype("float32")

    tmp_models = {
        "Linear Regression (Baseline)": ("baseline_linear_regression", False),
        "Ridge Regression":             ("ridge_regression",           False),
        "Random Forest":                ("random_forest",              False),
        "XGBoost":                      ("xgboost",                    False),
        "LSTM":                         ("lstm",                       True),
    }

    results = []
    for name, (key, is_lstm) in tmp_models.items():
        path = f"{MODELS_DIR}/_tmp_{key}.onnx"
        if not os.path.exists(path):
            continue
        X_in    = X_test_3d if is_lstm else X_test_f
        preds   = _onnx_predict(path, X_in)
        metrics = evaluate_model(y_test, preds)
        print_metrics(name, metrics)
        results.append({"model": name, **metrics})

    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(f"{RESULTS_DIR}/experiment_results.csv", index=False)

    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 60)
    print(df[["model", "MAE", "RMSE", "R2"]].to_string(index=False))
    print(f"\nBest model: {df.loc[df['MAE'].idxmin(), 'model']}")


# ── Task 4: Pick best, compare vs production, deploy if better ───────────────
def _compare_and_deploy(**context):
    import sys, os, json, shutil
    import numpy as np
    import pandas as pd
    sys.path.insert(0, SRC)
    from evaluate import evaluate_model
    from train import _load_active_model
    from triton_utils import publish_to_triton

    _, _, X_test, y_test, _ = _load_arrays()
    X_test_f  = X_test.astype("float32")
    X_test_3d = X_test.reshape(X_test.shape[0], 1, X_test.shape[1]).astype("float32")

    with open(TRAIN_RANGE_FILE) as f:
        date_range = json.load(f)
    train_start = date_range["train_start"]
    train_end   = date_range["train_end"]

    tmp_keys = {
        "baseline_linear_regression": ("Linear Regression (Baseline)", False),
        "ridge_regression":           ("Ridge Regression",             False),
        "random_forest":              ("Random Forest",                False),
        "xgboost":                    ("XGBoost",                     False),
        "lstm":                       ("LSTM",                        True),
    }

    trained = {}  # key -> (display_name, mae, is_lstm)
    for key, (name, is_lstm) in tmp_keys.items():
        path = f"{MODELS_DIR}/_tmp_{key}.onnx"
        if not os.path.exists(path):
            continue
        X_in  = X_test_3d if is_lstm else X_test_f
        preds = _onnx_predict(path, X_in)
        mae   = evaluate_model(y_test, preds)["MAE"]
        trained[key] = (name, mae, is_lstm)

    best_key           = min(trained, key=lambda k: trained[k][1])
    best_name, new_mae, best_is_lstm = trained[best_key]
    print(f"Best new model: {best_name}  MAE={new_mae:.4f}")

    _, prod_mae, active_info = _load_active_model(MODELS_DIR, X_test, y_test)

    print("\n" + "=" * 60)
    print("DEPLOYMENT DECISION")
    print("=" * 60)
    if prod_mae is None:
        prod_str = "N/A (no production model)"
    else:
        prod_str = f"MAE={prod_mae:.4f}  version={active_info['train_start']} → {active_info['train_end']}"
    print(f"  New  : {best_name}  MAE={new_mae:.4f}  trained on {train_start} → {train_end}")
    print(f"  Prod : {prod_str}")

    if prod_mae is None or new_mae < prod_mae:
        os.makedirs(ONNX_DIR, exist_ok=True)
        onnx_filename = f"{best_key}_{train_start}_{train_end}.onnx"
        onnx_dest     = os.path.join(ONNX_DIR, onnx_filename)
        shutil.copy2(f"{MODELS_DIR}/_tmp_{best_key}.onnx", onnx_dest)

        info = {
            "onnx_file":   onnx_filename,
            "model_key":   best_key,
            "train_start": train_start,
            "train_end":   train_end,
            "is_lstm":     best_is_lstm,
        }
        with open(f"{MODELS_DIR}/active_model.json", "w") as f:
            json.dump(info, f, indent=2)

        triton_repo = "/app/triton_model_repo"
        os.makedirs(triton_repo, exist_ok=True)
        publish_to_triton(onnx_dest, triton_repo, best_is_lstm)

        status = "DEPLOYED"
        delta  = round(prod_mae - new_mae, 4) if prod_mae is not None else None
        print(f"  → DEPLOYED  {onnx_dest}")
    else:
        status = "NOT_DEPLOYED"
        delta  = round(prod_mae - new_mae, 4)
        print(f"  → NOT DEPLOYED  new MAE {new_mae:.4f} >= production MAE {prod_mae:.4f}")

    # Delete all temp ONNX files
    for key in tmp_keys:
        p = f"{MODELS_DIR}/_tmp_{key}.onnx"
        if os.path.exists(p):
            os.remove(p)
    print("  Temp files cleaned up.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = f"{RESULTS_DIR}/retrain_comparison.csv"
    row = {
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
    print(f"  Report saved: {report_path}")


# ── DAG definition ───────────────────────────────────────────────────────────
with DAG(
    dag_id="pm25_training_pipeline",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["pm25", "ml"],
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

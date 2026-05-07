"""
Training Module
===============
Trains baseline and candidate models for PM2.5 prediction.
"""

import os
import json
import numpy as np
import pandas as pd
import inspect
import tempfile
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

from data_loader import load_config, load_train_test_data
from preprocessing import preprocess_pipeline
from feature_engineering import build_features, get_feature_columns
from evaluate import evaluate_model, print_metrics
from lstm_model import train_lstm_with_tuning
from sarima_model import train_sarima_with_tuning, predict_sarima_rolling, fit_sarima
from transformer_model import predict_transformer, train_transformer_regressor


def get_model(name: str, params: dict = None):
    models = {
        "LinearRegression": LinearRegression,
        "Ridge": Ridge,
        "RandomForestRegressor": RandomForestRegressor,
        "XGBRegressor": XGBRegressor,
    }
    if name not in models:
        raise ValueError(f"Unknown model: {name}")
    if params:
        return models[name](**params)
    return models[name]()


def train_baseline(X_train, y_train, config: dict) -> tuple:
    print("=" * 50)
    print("Training Baseline: Linear Regression")
    print("=" * 50)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, "LinearRegression"


def train_with_tuning(
    model_name: str,
    param_grid: dict,
    X_train,
    y_train,
    random_state: int = 42,
) -> tuple:
    print(f"\n{'=' * 50}")
    print(f"Training Candidate: {model_name}")
    print(f"{'=' * 50}")

    model_cls = {"LinearRegression": LinearRegression, "Ridge": Ridge,
                 "RandomForestRegressor": RandomForestRegressor, "XGBRegressor": XGBRegressor}[model_name]
    init_params = {"random_state": random_state} if "random_state" in inspect.signature(model_cls).parameters else {}
    base_model = get_model(model_name, init_params or None)

    tscv = TimeSeriesSplit(n_splits=3)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid.copy(),
        cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=int(os.environ.get("GRID_N_JOBS", "-1")),
        verbose=0,
        return_train_score=True,
    )
    grid_search.fit(X_train, y_train)

    cv_df = pd.DataFrame(grid_search.cv_results_)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", prefix=f"{model_name}_cv_", delete=False) as f:
        cv_df.to_csv(f.name, index=False)
        mlflow.log_artifact(f.name, artifact_path="cv_results")

    print(f"  Best params: {grid_search.best_params_}")
    print(f"  Best CV MAE: {-grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_


def _setup_mlflow(config: dict):
    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        config.get("mlflow", {}).get("local_uri", "http://localhost:5000"),
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.get("mlflow", {}).get("experiment_name", "pm25_prediction"))
    print(f"MLflow tracking URI: {tracking_uri}")


def _load_active_model(models_dir: str, X_test, y_test, y_train=None):
    """
    Load the currently deployed production model and return (None, mae, active_info).
    For ONNX models: runs inference on X_test.
    For SARIMA: refits saved order on y_train then rolls over y_test to get MAE.
    Returns (None, None, None) if no production model exists or y_train is missing for SARIMA.
    """
    import onnxruntime as rt

    registry_path = os.path.join(models_dir, "active_model.json")
    if not os.path.exists(registry_path):
        return None, None, None

    with open(registry_path) as f:
        info = json.load(f)

    if info.get("backend") == "sarima":
        if y_train is None:
            return None, None, None
        sarima_path = os.path.join(models_dir, "sarima_order.json")
        if not os.path.exists(sarima_path):
            return None, None, None
        with open(sarima_path) as f:
            so = json.load(f)
        model = fit_sarima(tuple(so["order"]), tuple(so["seasonal_order"]), y_train)
        preds = predict_sarima_rolling(model, y_train, y_test)
        mae = evaluate_model(y_test, preds)["MAE"]
        return None, mae, info

    onnx_path = os.path.join(models_dir, "onnx", info["onnx_file"])
    if not os.path.exists(onnx_path):
        return None, None, None

    session     = rt.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name  = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    if info.get("input_shape") == "3d":
        X_in = X_test.reshape(X_test.shape[0], 1, X_test.shape[1]).astype(np.float32)
    else:
        X_in = X_test.astype(np.float32)

    preds = session.run([output_name], {input_name: X_in})[0].flatten()
    mae   = evaluate_model(y_test, preds)["MAE"]

    return None, mae, info


def _save_model(model, key: str, train_start: str, train_end: str,
                models_dir: str, is_lstm: bool, old_info: dict):
    """
    Save model with training date range in filename, export to ONNX,
    and update active_model.json. Previous versioned files are kept.
    """
    from export_onnx import (
        export_lstm as export_lstm_onnx,
        export_sklearn,
        export_transformer,
        export_xgboost,
    )

    onnx_dir      = os.path.join(models_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)
    onnx_filename = f"{key}_{train_start}_{train_end}.onnx"
    onnx_path     = os.path.join(onnx_dir, onnx_filename)

    # Export to ONNX
    print(f"  Exporting to ONNX: {onnx_path}")
    if is_lstm:
        export_lstm_onnx(model, onnx_dir, output_path=onnx_path)
    elif key == "xgboost":
        export_xgboost(model, onnx_dir, output_path=onnx_path)
    elif key == "transformer":
        export_transformer(
            model,
            onnx_dir,
            output_path=onnx_path,
            n_features=getattr(model, "n_features", 17),
        )
    else:
        export_sklearn(model, key, onnx_dir, output_path=onnx_path)

    info = {
        "onnx_file":   onnx_filename,
        "model_key":   key,
        "train_start": train_start,
        "train_end":   train_end,
        "backend":     "onnx",
        "input_shape": "3d" if is_lstm else "2d",
    }

    # Publish to Triton model repository (project_root/triton_model_repo/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    triton_repo  = os.path.join(project_root, "triton_model_repo")
    os.makedirs(triton_repo, exist_ok=True)
    from triton_utils import publish_to_triton
    publish_to_triton(onnx_path, triton_repo, is_lstm)

    with open(os.path.join(models_dir, "active_model.json"), "w") as f:
        json.dump(info, f, indent=2)

    return onnx_path


def _save_sarima_model(order, seasonal_order, train_start: str, train_end: str, models_dir: str):
    """Save SARIMA order params and update active_model.json (no ONNX export)."""
    sarima_order = {"order": list(order), "seasonal_order": list(seasonal_order)}
    with open(os.path.join(models_dir, "sarima_order.json"), "w") as f:
        json.dump(sarima_order, f, indent=2)
    info = {
        "model_key":      "sarima",
        "backend":        "sarima",
        "input_shape":    "2d",
        "onnx_file":      None,
        "train_start":    train_start,
        "train_end":      train_end,
        **sarima_order,
    }
    with open(os.path.join(models_dir, "active_model.json"), "w") as f:
        json.dump(info, f, indent=2)
    return info


def train_all_models(config: dict):
    """
    Full training pipeline: load data, preprocess, engineer features,
    train all models in memory, deploy the best one if it beats production.
    """
    _setup_mlflow(config)
    random_state = config.get("random_state", 42)
    np.random.seed(random_state)

    # ---- Load and preprocess data ----
    print("Loading data...")
    train_df, test_df = load_train_test_data(config)

    print("\n=== Train Data Preprocessing ===")
    train_clean = preprocess_pipeline(train_df)
    print("\n=== Test Data Preprocessing ===")
    test_clean = preprocess_pipeline(test_df)

    # ---- Feature engineering ----
    print("\nBuilding features...")
    lag_days        = config["features"]["lag_days"]
    rolling_windows = config["features"]["rolling_windows"]

    train_feat = build_features(train_clean, lag_days, rolling_windows)
    test_feat  = build_features(test_clean,  lag_days, rolling_windows)

    feature_cols = get_feature_columns(train_feat)
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    X_train = train_feat[feature_cols].values
    y_train = train_feat["pm25"].values
    X_test  = test_feat[feature_cols].values
    y_test  = test_feat["pm25"].values

    print(f"\nX_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")

    # Training date range — used as the model version identifier
    train_start = train_feat["date"].min().strftime("%Y-%m-%d")
    train_end   = train_feat["date"].max().strftime("%Y-%m-%d")
    print(f"Training date range: {train_start} → {train_end}")

    # ---- Split train into train/val (time-based) ----
    val_start    = config["split"]["validation_start"]
    val_mask     = train_feat["date"] >= val_start
    X_train_sub  = train_feat[~val_mask][feature_cols].values
    y_train_sub  = train_feat[~val_mask]["pm25"].values
    X_val        = train_feat[val_mask][feature_cols].values
    y_val        = train_feat[val_mask]["pm25"].values
    print(f"Train subset: {X_train_sub.shape}, Validation: {X_val.shape}")

    models_dir  = config["output"]["models_dir"]
    results_dir = config["output"]["results_dir"]
    os.makedirs(models_dir,  exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # ---- Train all models in memory (no intermediate disk saves) ----
    # trained[key] = (display_name, model_object, metrics_dict, is_lstm)
    trained = {}
    results = []

    # 1. Baseline: Linear Regression
    with mlflow.start_run(run_name="LinearRegression"):
        baseline, _ = train_baseline(X_train, y_train, config)
        m = evaluate_model(y_test, baseline.predict(X_test))
        print_metrics("LinearRegression", m)
        mlflow.log_params({"model_type": "LinearRegression"})
        mlflow.log_metrics(m)
        trained["baseline_linear_regression"] = ("Linear Regression (Baseline)", baseline, m, False)
        results.append({"model": "Linear Regression (Baseline)", **m})

    # 2. Ridge Regression
    ridge_params = config["models"]["ridge"]["params"]
    with mlflow.start_run(run_name="Ridge"):
        ridge_model, ridge_best = train_with_tuning("Ridge", ridge_params, X_train, y_train, random_state)
        m = evaluate_model(y_test, ridge_model.predict(X_test))
        print_metrics("Ridge Regression", m)
        mlflow.log_params(ridge_best)
        mlflow.log_metrics(m)
        trained["ridge_regression"] = ("Ridge Regression", ridge_model, m, False)
        results.append({"model": "Ridge Regression", **m, "best_params": str(ridge_best)})

    # 3. Random Forest
    rf_params = config["models"]["random_forest"]["params"].copy()
    rf_params.pop("random_state", None)
    with mlflow.start_run(run_name="RandomForest"):
        rf_model, rf_best = train_with_tuning("RandomForestRegressor", rf_params, X_train, y_train, random_state)
        m = evaluate_model(y_test, rf_model.predict(X_test))
        print_metrics("Random Forest", m)
        mlflow.log_params(rf_best)
        mlflow.log_metrics(m)
        trained["random_forest"] = ("Random Forest", rf_model, m, False)
        results.append({"model": "Random Forest", **m, "best_params": str(rf_best)})

    # 4. XGBoost
    xgb_params = config["models"]["xgboost"]["params"].copy()
    xgb_params.pop("random_state", None)
    with mlflow.start_run(run_name="XGBoost"):
        xgb_model, xgb_best = train_with_tuning("XGBRegressor", xgb_params, X_train, y_train, random_state)
        m = evaluate_model(y_test, xgb_model.predict(X_test))
        print_metrics("XGBoost", m)
        mlflow.log_params(xgb_best)
        mlflow.log_metrics(m)
        trained["xgboost"] = ("XGBoost", xgb_model, m, False)
        results.append({"model": "XGBoost", **m, "best_params": str(xgb_best)})

    # 5. LSTM
    lstm_params = config["models"]["lstm"]["params"]
    with mlflow.start_run(run_name="LSTM"):
        lstm_model, lstm_best = train_lstm_with_tuning(X_train, y_train, lstm_params, random_state)
        X_test_3d   = X_test.reshape(X_test.shape[0], 1, X_test.shape[1]).astype(np.float32)
        y_pred_lstm = lstm_model.predict(X_test_3d).flatten()
        m = evaluate_model(y_test, y_pred_lstm)
        print_metrics("LSTM", m)
        mlflow.log_params(lstm_best)
        mlflow.log_metrics(m)
        trained["lstm"] = ("LSTM", lstm_model, m, True)
        results.append({"model": "LSTM", **m, "best_params": str(lstm_best)})

    # 6. Transformer
    transformer_params = config.get("models", {}).get("transformer", {}).get("params", {})
    with mlflow.start_run(run_name="Transformer"):
        transformer_model, transformer_best = train_transformer_regressor(
            X_train_sub,
            y_train_sub,
            X_val,
            y_val,
            transformer_params,
            random_state,
        )
        y_pred_transformer = predict_transformer(transformer_model, X_test)
        m = evaluate_model(y_test, y_pred_transformer)
        print_metrics("Transformer", m)
        mlflow.log_params(transformer_best)
        mlflow.log_metrics(m)
        trained["transformer"] = ("Transformer", transformer_model, m, False)
        results.append({"model": "Transformer", **m, "best_params": str(transformer_best)})

    # 7. SARIMA — statistical time-series benchmark
    # Uses only the pm25 endog series, not the feature matrix.
    # Not added to `trained` for ONNX competition; handled separately below.
    sarima_cfg = config.get("models", {}).get("sarima", {})
    seasonal_period = sarima_cfg.get("seasonal_period", 7)
    _sarima_fitted = None
    _sarima_mae = None
    with mlflow.start_run(run_name="SARIMA"):
        sarima_model, sarima_best = train_sarima_with_tuning(y_train, seasonal_period)
        y_pred_sarima = predict_sarima_rolling(sarima_model, y_train, y_test)
        m = evaluate_model(y_test, y_pred_sarima)
        print_metrics("SARIMA", m)
        mlflow.log_params(sarima_best)
        mlflow.log_metrics(m)
        results.append({"model": "SARIMA", **m, "best_params": str(sarima_best)})
        _sarima_fitted = sarima_model
        _sarima_mae = m["MAE"]

    # ---- Save experiment results ----
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{results_dir}/experiment_results.csv", index=False)
    print(f"\n{'=' * 60}")
    print("EXPERIMENT RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(results_df[["model", "MAE", "RMSE", "R2"]].to_string(index=False))

    # ---- Find the single best model ----
    best_key              = min(trained, key=lambda k: trained[k][2]["MAE"])
    best_name, best_model, best_metrics, best_is_lstm = trained[best_key]
    print(f"\nBest new model: {best_name}  MAE={best_metrics['MAE']:.4f}")

    # ---- Compare best new vs current production ----
    prod_model, prod_mae, active_info = _load_active_model(models_dir, X_test, y_test, y_train)
    new_mae = best_metrics["MAE"]

    print(f"\n{'=' * 60}")
    print("DEPLOYMENT DECISION")
    print(f"{'=' * 60}")

    if prod_mae is None:
        prod_str = "N/A (no production model)"
    else:
        prod_str = f"MAE={prod_mae:.4f}  version={active_info['train_start']} → {active_info['train_end']}"

    print(f"  New  : {best_name}  MAE={new_mae:.4f}  trained on {train_start} → {train_end}")
    print(f"  Prod : {prod_str}")

    if prod_mae is None or new_mae < prod_mae:
        model_path = _save_model(best_model, best_key, train_start, train_end,
                                 models_dir, best_is_lstm, active_info)
        status  = "DEPLOYED"
        delta   = round(prod_mae - new_mae, 4) if prod_mae is not None else None
        print(f"  → DEPLOYED  saved to {model_path}")
    else:
        status = "NOT_DEPLOYED"
        delta  = round(prod_mae - new_mae, 4)
        print(f"  → NOT DEPLOYED  new MAE {new_mae:.4f} >= production MAE {prod_mae:.4f}")

    # ---- Check if SARIMA beats the deployed ONNX model ----
    if _sarima_fitted is not None and _sarima_mae is not None:
        current_best_mae = new_mae if status == "DEPLOYED" else (prod_mae if prod_mae is not None else float("inf"))
        if _sarima_mae < current_best_mae:
            _save_sarima_model(
                _sarima_fitted.order,
                _sarima_fitted.seasonal_order,
                train_start, train_end, models_dir,
            )
            status = "DEPLOYED"
            print(f"  → SARIMA DEPLOYED  MAE={_sarima_mae:.4f} beats best ONNX MAE={current_best_mae:.4f}")
        else:
            print(f"  → SARIMA NOT DEPLOYED  MAE={_sarima_mae:.4f} >= best ONNX MAE={current_best_mae:.4f}")

    # ---- Save feature columns ----
    with open(f"{models_dir}/feature_columns.json", "w") as f:
        json.dump(feature_cols, f)

    # ---- Append comparison report ----
    report_path = os.path.join(results_dir, "retrain_comparison.csv")
    row = {
        "train_start":   train_start,
        "train_end":     train_end,
        "best_model":    best_name,
        "new_mae":       new_mae,
        "new_rmse":      best_metrics["RMSE"],
        "new_r2":        best_metrics["R2"],
        "prod_mae":      prod_mae,
        "mae_delta":     delta,
        "status":        status,
    }
    exists = os.path.exists(report_path)
    pd.DataFrame([row]).to_csv(report_path, mode="a", header=not exists, index=False)
    print(f"  Report saved: {report_path}")

    return results_df


if __name__ == "__main__":
    config = load_config()
    results = train_all_models(config)

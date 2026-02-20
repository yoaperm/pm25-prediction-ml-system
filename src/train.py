"""
Training Module
===============
Trains baseline and candidate models for PM2.5 prediction.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

from data_loader import load_config, load_train_test_data
from preprocessing import preprocess_pipeline
from feature_engineering import build_features, get_feature_columns
from evaluate import evaluate_model, print_metrics


def get_model(name: str, params: dict = None):
    """
    Instantiate a model by name.

    Parameters
    ----------
    name : str
        Model name from config.
    params : dict
        Model parameters.

    Returns
    -------
    sklearn estimator
    """
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
    """
    Train baseline model (Linear Regression).

    Returns
    -------
    tuple
        (model, model_name)
    """
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
    """
    Train a candidate model with hyperparameter tuning using TimeSeriesSplit CV.

    Parameters
    ----------
    model_name : str
        Model class name.
    param_grid : dict
        Hyperparameter grid for GridSearchCV.
    X_train : array-like
        Training features.
    y_train : array-like
        Training targets.
    random_state : int
        Random seed.

    Returns
    -------
    tuple
        (best_model, best_params)
    """
    print(f"\n{'=' * 50}")
    print(f"Training Candidate: {model_name}")
    print(f"{'=' * 50}")

    # Handle None values in param grid (for max_depth)
    clean_grid = {}
    for k, v in param_grid.items():
        if isinstance(v, list):
            clean_grid[k] = [None if x is None else x for x in v]
        else:
            clean_grid[k] = v

    base_model = get_model(model_name)

    # Use TimeSeriesSplit for cross-validation (respects temporal order)
    tscv = TimeSeriesSplit(n_splits=3)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=clean_grid,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)

    print(f"  Best params: {grid_search.best_params_}")
    print(f"  Best CV MAE: {-grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_


def train_all_models(config: dict):
    """
    Full training pipeline: load data, preprocess, engineer features,
    train all models, save results.
    """
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
    lag_days = config["features"]["lag_days"]
    rolling_windows = config["features"]["rolling_windows"]

    train_feat = build_features(train_clean, lag_days, rolling_windows)
    test_feat = build_features(test_clean, lag_days, rolling_windows)

    feature_cols = get_feature_columns(train_feat)
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    X_train = train_feat[feature_cols].values
    y_train = train_feat["pm25"].values
    X_test = test_feat[feature_cols].values
    y_test = test_feat["pm25"].values

    print(f"\nX_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")

    # ---- Split train into train/val (time-based) ----
    val_start = config["split"]["validation_start"]
    val_mask = train_feat["date"] >= val_start
    X_train_sub = train_feat[~val_mask][feature_cols].values
    y_train_sub = train_feat[~val_mask]["pm25"].values
    X_val = train_feat[val_mask][feature_cols].values
    y_val = train_feat[val_mask]["pm25"].values
    print(f"\nTrain subset: {X_train_sub.shape}, Validation: {X_val.shape}")

    # ---- Train models ----
    results = []
    models_dir = config["output"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)

    # 1. Baseline: Linear Regression
    baseline, baseline_name = train_baseline(X_train, y_train, config)
    y_pred_baseline = baseline.predict(X_test)
    metrics_baseline = evaluate_model(y_test, y_pred_baseline)
    print_metrics(baseline_name, metrics_baseline)
    joblib.dump(baseline, f"{models_dir}/baseline_linear_regression.joblib")
    results.append({"model": "Linear Regression (Baseline)", **metrics_baseline})

    # 2. Ridge Regression
    ridge_params = config["models"]["ridge"]["params"]
    ridge_model, ridge_best = train_with_tuning(
        "Ridge", ridge_params, X_train, y_train, random_state
    )
    y_pred_ridge = ridge_model.predict(X_test)
    metrics_ridge = evaluate_model(y_test, y_pred_ridge)
    print_metrics("Ridge Regression", metrics_ridge)
    joblib.dump(ridge_model, f"{models_dir}/ridge_regression.joblib")
    results.append({"model": "Ridge Regression", **metrics_ridge, "best_params": str(ridge_best)})

    # 3. Random Forest
    rf_params = config["models"]["random_forest"]["params"].copy()
    rf_rs = rf_params.pop("random_state", 42)
    rf_model, rf_best = train_with_tuning(
        "RandomForestRegressor", rf_params, X_train, y_train, random_state
    )
    y_pred_rf = rf_model.predict(X_test)
    metrics_rf = evaluate_model(y_test, y_pred_rf)
    print_metrics("Random Forest", metrics_rf)
    joblib.dump(rf_model, f"{models_dir}/random_forest.joblib")
    results.append({"model": "Random Forest", **metrics_rf, "best_params": str(rf_best)})

    # 4. XGBoost
    xgb_params = config["models"]["xgboost"]["params"].copy()
    xgb_rs = xgb_params.pop("random_state", 42)
    xgb_model, xgb_best = train_with_tuning(
        "XGBRegressor", xgb_params, X_train, y_train, random_state
    )
    y_pred_xgb = xgb_model.predict(X_test)
    metrics_xgb = evaluate_model(y_test, y_pred_xgb)
    print_metrics("XGBoost", metrics_xgb)
    joblib.dump(xgb_model, f"{models_dir}/xgboost.joblib")
    results.append({"model": "XGBoost", **metrics_xgb, "best_params": str(xgb_best)})

    # ---- Save results ----
    results_dir = config["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{results_dir}/experiment_results.csv", index=False)
    print(f"\n{'=' * 60}")
    print("EXPERIMENT RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(results_df[["model", "MAE", "RMSE", "R2"]].to_string(index=False))
    print(f"\nResults saved to {results_dir}/experiment_results.csv")
    print(f"Models saved to {models_dir}/")

    # Save feature columns for inference
    with open(f"{models_dir}/feature_columns.json", "w") as f:
        json.dump(feature_cols, f)

    return results_df


if __name__ == "__main__":
    config = load_config()
    results = train_all_models(config)

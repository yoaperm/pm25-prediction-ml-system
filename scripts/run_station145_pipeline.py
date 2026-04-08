"""
Run the full ML pipeline on station_145_long.csv data.
Converts hourly data to daily averages, then trains all models and evaluates.
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocessing import preprocess_pipeline
from feature_engineering import build_features, get_feature_columns
from evaluate import evaluate_model, print_metrics

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor


def load_station145_data(csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load station 145 CSV and split into train/test by year."""
    df = pd.read_csv(csv_path)
    df["Date_Time"] = pd.to_datetime(df["Date_Time"])
    df["date"] = df["Date_Time"].dt.date

    # Aggregate hourly -> daily mean
    daily = df.groupby("date")["Value"].mean().reset_index()
    daily.columns = ["date", "pm25"]
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    print(f"Station 145 daily data: {daily.shape[0]} days")
    print(f"Date range: {daily['date'].min().date()} to {daily['date'].max().date()}")
    print(f"Missing values: {daily['pm25'].isna().sum()}")

    # Split: train = 2023-2024, test = 2025-2026
    train_df = daily[daily["date"] < "2025-01-01"].copy().reset_index(drop=True)
    test_df = daily[daily["date"] >= "2025-01-01"].copy().reset_index(drop=True)

    print(f"\nTrain: {train_df.shape[0]} days ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
    print(f"Test:  {test_df.shape[0]} days ({test_df['date'].min().date()} to {test_df['date'].max().date()})")

    return train_df, test_df


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    os.chdir(base_dir)

    csv_path = "data/raw/station_145_long.csv"
    random_state = 42
    np.random.seed(random_state)

    # ---- Load data ----
    print("=" * 60)
    print("STATION 145 — FULL PIPELINE")
    print("=" * 60)
    train_df, test_df = load_station145_data(csv_path)

    # ---- Preprocess ----
    print("\n=== Train Data Preprocessing ===")
    train_clean = preprocess_pipeline(train_df)
    print("\n=== Test Data Preprocessing ===")
    test_clean = preprocess_pipeline(test_df)

    # ---- Feature engineering ----
    lag_days = [1, 2, 3, 5, 7]
    rolling_windows = [3, 7, 14]

    print("\nBuilding features...")
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

    # ---- Train models ----
    results = []
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    # 1. Baseline: Linear Regression
    print("\n" + "=" * 50)
    print("Training Baseline: Linear Regression")
    print("=" * 50)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    m_lr = evaluate_model(y_test, y_pred_lr)
    print_metrics("Linear Regression", m_lr)
    results.append({"model": "Linear Regression (Baseline)", **m_lr})

    # 2. Ridge Regression
    print("\n" + "=" * 50)
    print("Training: Ridge Regression")
    print("=" * 50)
    tscv = TimeSeriesSplit(n_splits=3)
    ridge_grid = GridSearchCV(
        Ridge(),
        {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        cv=tscv, scoring="neg_mean_absolute_error", n_jobs=1,
    )
    ridge_grid.fit(X_train, y_train)
    print(f"  Best params: {ridge_grid.best_params_}")
    y_pred_ridge = ridge_grid.best_estimator_.predict(X_test)
    m_ridge = evaluate_model(y_test, y_pred_ridge)
    print_metrics("Ridge Regression", m_ridge)
    results.append({"model": "Ridge Regression", **m_ridge})

    # 3. Random Forest
    print("\n" + "=" * 50)
    print("Training: Random Forest")
    print("=" * 50)
    rf_grid = GridSearchCV(
        RandomForestRegressor(random_state=random_state, n_jobs=1),
        {
            "n_estimators": [100, 200],
            "max_depth": [10, None],
        },
        cv=tscv, scoring="neg_mean_absolute_error", n_jobs=1,
    )
    rf_grid.fit(X_train, y_train)
    print(f"  Best params: {rf_grid.best_params_}")
    y_pred_rf = rf_grid.best_estimator_.predict(X_test)
    m_rf = evaluate_model(y_test, y_pred_rf)
    print_metrics("Random Forest", m_rf)
    results.append({"model": "Random Forest", **m_rf})

    # 4. XGBoost
    print("\n" + "=" * 50)
    print("Training: XGBoost")
    print("=" * 50)
    xgb_grid = GridSearchCV(
        XGBRegressor(random_state=random_state, verbosity=0),
        {
            "n_estimators": [100, 200],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
        },
        cv=tscv, scoring="neg_mean_absolute_error", n_jobs=1,
    )
    xgb_grid.fit(X_train, y_train)
    print(f"  Best params: {xgb_grid.best_params_}")
    y_pred_xgb = xgb_grid.best_estimator_.predict(X_test)
    m_xgb = evaluate_model(y_test, y_pred_xgb)
    print_metrics("XGBoost", m_xgb)
    results.append({"model": "XGBoost", **m_xgb})

    # 5. LSTM
    print("\n" + "=" * 50)
    print("Training: LSTM")
    print("=" * 50)
    try:
        from lstm_model import train_lstm_with_tuning
        lstm_params = {
            "units": [32, 64],
            "dropout": [0.1, 0.2],
            "learning_rate": [0.001, 0.01],
            "epochs": [50],
            "batch_size": [16],
        }
        lstm_model, lstm_best = train_lstm_with_tuning(X_train, y_train, lstm_params, random_state)
        X_test_3d = X_test.reshape(X_test.shape[0], 1, X_test.shape[1]).astype(np.float32)
        y_pred_lstm = lstm_model.predict(X_test_3d).flatten()
        m_lstm = evaluate_model(y_test, y_pred_lstm)
        print_metrics("LSTM", m_lstm)
        results.append({"model": "LSTM", **m_lstm})
    except Exception as e:
        print(f"  LSTM training failed: {e}")
        print("  Skipping LSTM...")

    # ---- Save results ----
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    results_df = pd.DataFrame(results)
    output_path = f"{results_dir}/experiment_results_station145.csv"
    results_df.to_csv(output_path, index=False)

    print(f"\n{'=' * 60}")
    print("EXPERIMENT RESULTS — Station 145 (เขตบางขุนเทียน)")
    print(f"{'=' * 60}")
    print(results_df[["model", "MAE", "RMSE", "R2"]].to_string(index=False))
    print(f"\nResults saved to {output_path}")

    # Also save feature columns
    with open(f"{models_dir}/feature_columns_station145.json", "w") as f:
        json.dump(feature_cols, f)

    # Save best model (RF)
    joblib.dump(rf_grid.best_estimator_, f"{models_dir}/random_forest_station145.joblib")
    print(f"Best model (Random Forest) saved to {models_dir}/random_forest_station145.joblib")

    return results_df


if __name__ == "__main__":
    main()

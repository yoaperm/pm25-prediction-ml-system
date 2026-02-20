"""
Feature Engineering Module
==========================
Creates time-series features for PM2.5 prediction.
"""

import pandas as pd
import numpy as np


def create_lag_features(df: pd.DataFrame, lag_days: list[int]) -> pd.DataFrame:
    """
    Create lag features: PM2.5 values from previous days.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'pm25' column sorted by date.
    lag_days : list[int]
        List of lag periods (e.g., [1, 2, 3, 5, 7]).

    Returns
    -------
    pd.DataFrame
        DataFrame with new lag columns.
    """
    result = df.copy()
    for lag in lag_days:
        result[f"pm25_lag_{lag}"] = result["pm25"].shift(lag)
    return result


def create_rolling_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """
    Create rolling window statistics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'pm25' column.
    windows : list[int]
        List of window sizes (e.g., [3, 7, 14]).

    Returns
    -------
    pd.DataFrame
        DataFrame with rolling mean and std columns.
    """
    result = df.copy()
    for w in windows:
        result[f"pm25_rolling_mean_{w}"] = (
            result["pm25"].shift(1).rolling(window=w, min_periods=1).mean()
        )
        result[f"pm25_rolling_std_{w}"] = (
            result["pm25"].shift(1).rolling(window=w, min_periods=1).std()
        )
    return result


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create calendar/time-based features from the date column.

    Features:
    - day_of_week: 0 (Monday) to 6 (Sunday)
    - month: 1-12 (captures seasonality)
    - day_of_year: 1-366 (captures annual cycle)
    - is_weekend: 1 if Saturday/Sunday, 0 otherwise

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'date' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with new time feature columns.
    """
    result = df.copy()
    result["day_of_week"] = result["date"].dt.dayofweek
    result["month"] = result["date"].dt.month
    result["day_of_year"] = result["date"].dt.dayofyear
    result["is_weekend"] = (result["date"].dt.dayofweek >= 5).astype(int)
    return result


def create_change_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features based on day-to-day changes.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'pm25' column.

    Returns
    -------
    pd.DataFrame
    """
    result = df.copy()
    result["pm25_diff_1"] = result["pm25"].shift(1).diff()
    result["pm25_pct_change_1"] = result["pm25"].shift(1).pct_change()
    return result


def build_features(
    df: pd.DataFrame,
    lag_days: list[int] = None,
    rolling_windows: list[int] = None,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame with columns ['date', 'pm25'].
    lag_days : list[int]
        Lag periods for lag features.
    rolling_windows : list[int]
        Window sizes for rolling features.

    Returns
    -------
    pd.DataFrame
        DataFrame with all features. NaN rows from lag/rolling are dropped.
    """
    if lag_days is None:
        lag_days = [1, 2, 3, 5, 7]
    if rolling_windows is None:
        rolling_windows = [3, 7, 14]

    result = df.copy()

    # Create features
    result = create_lag_features(result, lag_days)
    result = create_rolling_features(result, rolling_windows)
    result = create_time_features(result)
    result = create_change_features(result)

    # Drop rows with NaN from lag/rolling (first N days)
    result = result.dropna().reset_index(drop=True)

    return result


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get list of feature column names (excluding date and target)."""
    exclude = ["date", "pm25"]
    return [col for col in df.columns if col not in exclude]


if __name__ == "__main__":
    from data_loader import load_config, load_train_test_data
    from preprocessing import preprocess_pipeline

    config = load_config()
    train_df, test_df = load_train_test_data(config)

    train_clean = preprocess_pipeline(train_df)
    test_clean = preprocess_pipeline(test_df)

    # Build features
    lag_days = config["features"]["lag_days"]
    rolling_windows = config["features"]["rolling_windows"]

    train_feat = build_features(train_clean, lag_days, rolling_windows)
    test_feat = build_features(test_clean, lag_days, rolling_windows)

    feature_cols = get_feature_columns(train_feat)
    print(f"Train features shape: {train_feat.shape}")
    print(f"Test features shape: {test_feat.shape}")
    print(f"\nFeature columns ({len(feature_cols)}):")
    for col in feature_cols:
        print(f"  - {col}")

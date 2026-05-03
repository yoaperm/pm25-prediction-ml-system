"""
Hourly PM2.5 forecasting helpers.

Shared feature engineering for hourly horizons such as T+1h and T+24h.
All lag, rolling, and difference features use historical PM2.5 values only.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd


DEFAULT_LAGS = [1, 2, 3, 6, 12, 24]
DEFAULT_WINDOWS = [6, 12, 24]


def hourly_feature_columns(
    lags: Iterable[int] = DEFAULT_LAGS,
    rolling_windows: Iterable[int] = DEFAULT_WINDOWS,
) -> list[str]:
    """Return the canonical hourly feature column order."""
    feature_cols: list[str] = []

    for lag in lags:
        feature_cols.append(f"pm25_lag_{lag}")

    for window in rolling_windows:
        feature_cols.extend([
            f"pm25_rolling_mean_{window}h",
            f"pm25_rolling_std_{window}h",
        ])

    feature_cols.extend(["pm25_diff_1h", "pm25_diff_24h"])
    feature_cols.extend(["hour", "day_of_week", "month", "day_of_year", "is_weekend"])
    return feature_cols


def build_hourly_supervised_features(
    df: pd.DataFrame,
    forecast_hour: int = 1,
    lags: Iterable[int] = DEFAULT_LAGS,
    rolling_windows: Iterable[int] = DEFAULT_WINDOWS,
    timestamp_col: str = "datetime",
) -> tuple[pd.DataFrame, list[str], str]:
    """
    Build supervised hourly features for a fixed forecast horizon.

    Each feature row at time T predicts PM2.5 at T + forecast_hour. Time
    features describe the target timestamp, while PM2.5-derived features only
    use observations at or before T. Therefore pm25_lag_1 is the latest known
    PM2.5 value at T.
    """
    if forecast_hour < 1:
        raise ValueError("forecast_hour must be >= 1")

    result = df.copy()
    result[timestamp_col] = pd.to_datetime(result[timestamp_col])
    result = result.sort_values(timestamp_col).reset_index(drop=True)

    feature_cols = hourly_feature_columns(lags, rolling_windows)

    for lag in lags:
        result[f"pm25_lag_{lag}"] = result["pm25"].shift(lag - 1)

    for window in rolling_windows:
        result[f"pm25_rolling_mean_{window}h"] = result["pm25"].rolling(window).mean()
        result[f"pm25_rolling_std_{window}h"] = result["pm25"].rolling(window).std()

    result["pm25_diff_1h"] = result["pm25"].diff(1)
    result["pm25_diff_24h"] = result["pm25"].diff(24)

    result["target_datetime"] = result[timestamp_col] + pd.to_timedelta(forecast_hour, unit="h")
    result["hour"] = result["target_datetime"].dt.hour
    result["day_of_week"] = result["target_datetime"].dt.dayofweek
    result["month"] = result["target_datetime"].dt.month
    result["day_of_year"] = result["target_datetime"].dt.dayofyear
    result["is_weekend"] = (result["day_of_week"] >= 5).astype(int)

    target_col = f"pm25_h{forecast_hour}"
    result[target_col] = result["pm25"].shift(-forecast_hour)

    result = result.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
    return result, feature_cols, target_col


def build_next_hour_inference_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    timestamp_col: str = "datetime",
) -> pd.DataFrame:
    """Build one feature row for the next hour after the latest observation."""
    result = df.copy()
    result[timestamp_col] = pd.to_datetime(result[timestamp_col])
    result = result.sort_values(timestamp_col).reset_index(drop=True)

    if result.empty:
        return pd.DataFrame(columns=feature_cols)

    target_datetime = result[timestamp_col].iloc[-1] + pd.Timedelta(hours=1)
    new_row = pd.DataFrame([{timestamp_col: target_datetime, "pm25": 0.0}])
    extended = pd.concat([result, new_row], ignore_index=True)

    for lag in DEFAULT_LAGS:
        extended[f"pm25_lag_{lag}"] = extended["pm25"].shift(lag)

    shifted = extended["pm25"].shift(1)
    for window in DEFAULT_WINDOWS:
        extended[f"pm25_rolling_mean_{window}h"] = shifted.rolling(window).mean()
        extended[f"pm25_rolling_std_{window}h"] = shifted.rolling(window).std()

    extended["pm25_diff_1h"] = extended["pm25"].shift(1).diff(1)
    extended["pm25_diff_24h"] = extended["pm25"].shift(1).diff(24)
    extended["hour"] = extended[timestamp_col].dt.hour
    extended["day_of_week"] = extended[timestamp_col].dt.dayofweek
    extended["month"] = extended[timestamp_col].dt.month
    extended["day_of_year"] = extended[timestamp_col].dt.dayofyear
    extended["is_weekend"] = (extended["day_of_week"] >= 5).astype(int)

    missing_cols = [col for col in feature_cols if col not in extended.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}")

    final_features = extended.iloc[[-1]].dropna(subset=feature_cols)
    return final_features[feature_cols] if not final_features.empty else pd.DataFrame(columns=feature_cols)

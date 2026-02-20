"""
Preprocessing Module
====================
Handles missing values, outliers, and data cleaning for PM2.5 time series.
"""

import pandas as pd
import numpy as np


def handle_missing_values(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    """
    Handle missing values in PM2.5 data.

    Strategy:
    1. Forward-fill (carry last known value) — appropriate for time series
    2. Backward-fill for remaining NaN at the start
    3. Drop any remaining NaN rows (edge case)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ['date', 'pm25'].
    method : str
        Fill method: 'ffill', 'interpolate', or 'drop'.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    result = df.copy()
    original_missing = result["pm25"].isnull().sum()

    if method == "ffill":
        result["pm25"] = result["pm25"].ffill().bfill()
    elif method == "interpolate":
        result["pm25"] = result["pm25"].interpolate(method="linear").bfill().ffill()
    elif method == "drop":
        result = result.dropna(subset=["pm25"])
    else:
        raise ValueError(f"Unknown method: {method}")

    remaining_missing = result["pm25"].isnull().sum()
    result = result.dropna(subset=["pm25"])

    print(f"  Missing values: {original_missing} → {remaining_missing} (method={method})")

    return result.reset_index(drop=True)


def remove_outliers(
    df: pd.DataFrame,
    lower_bound: float = 0.0,
    upper_bound: float = 500.0,
) -> pd.DataFrame:
    """
    Remove outlier PM2.5 values outside physically reasonable range.

    PM2.5 concentration:
    - Cannot be negative
    - Values > 500 µg/m³ are extremely rare and likely sensor errors

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'pm25' column.
    lower_bound : float
        Minimum valid PM2.5 value.
    upper_bound : float
        Maximum valid PM2.5 value.

    Returns
    -------
    pd.DataFrame
        DataFrame with outliers removed.
    """
    original_len = len(df)
    mask = (df["pm25"] >= lower_bound) & (df["pm25"] <= upper_bound)
    result = df[mask].copy().reset_index(drop=True)
    removed = original_len - len(result)

    if removed > 0:
        print(f"  Outliers removed: {removed} rows (outside [{lower_bound}, {upper_bound}])")

    return result


def preprocess_pipeline(df: pd.DataFrame, fill_method: str = "ffill") -> pd.DataFrame:
    """
    Full preprocessing pipeline.

    Steps:
    1. Handle missing values
    2. Remove outliers
    3. Ensure sorted by date

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame with columns ['date', 'pm25'].
    fill_method : str
        Method for handling missing values.

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame.
    """
    print("Preprocessing pipeline:")
    result = df.copy()

    # Step 1: Sort by date
    result = result.sort_values("date").reset_index(drop=True)

    # Step 2: Handle missing values
    result = handle_missing_values(result, method=fill_method)

    # Step 3: Remove outliers
    result = remove_outliers(result)

    print(f"  Final shape: {result.shape}")
    return result


if __name__ == "__main__":
    from data_loader import load_config, load_train_test_data

    config = load_config()
    train_df, test_df = load_train_test_data(config)

    print("=== Train Data ===")
    train_clean = preprocess_pipeline(train_df)

    print("\n=== Test Data ===")
    test_clean = preprocess_pipeline(test_df)

    # Save processed data
    import os
    os.makedirs(config["data"]["processed_dir"], exist_ok=True)
    train_clean.to_csv(f"{config['data']['processed_dir']}/train_clean.csv", index=False)
    test_clean.to_csv(f"{config['data']['processed_dir']}/test_clean.csv", index=False)
    print(f"\nSaved to {config['data']['processed_dir']}/")

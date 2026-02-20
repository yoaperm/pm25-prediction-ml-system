"""
Basic tests for preprocessing module.
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, "src")

from preprocessing import handle_missing_values, remove_outliers


def test_handle_missing_ffill():
    """Test forward-fill missing value handling."""
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=5),
        "pm25": [10.0, np.nan, np.nan, 20.0, 25.0],
    })
    result = handle_missing_values(df, method="ffill")
    assert result["pm25"].isnull().sum() == 0
    assert result["pm25"].iloc[1] == 10.0  # forward-filled
    assert result["pm25"].iloc[2] == 10.0  # forward-filled
    print("✓ test_handle_missing_ffill passed")


def test_handle_missing_interpolate():
    """Test linear interpolation missing value handling."""
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=5),
        "pm25": [10.0, np.nan, np.nan, 40.0, 50.0],
    })
    result = handle_missing_values(df, method="interpolate")
    assert result["pm25"].isnull().sum() == 0
    assert abs(result["pm25"].iloc[1] - 20.0) < 0.1  # interpolated
    print("✓ test_handle_missing_interpolate passed")


def test_remove_outliers():
    """Test outlier removal."""
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=5),
        "pm25": [10.0, 50.0, -5.0, 600.0, 25.0],
    })
    result = remove_outliers(df, lower_bound=0.0, upper_bound=500.0)
    assert len(result) == 3  # removed -5 and 600
    assert -5.0 not in result["pm25"].values
    assert 600.0 not in result["pm25"].values
    print("✓ test_remove_outliers passed")


if __name__ == "__main__":
    test_handle_missing_ffill()
    test_handle_missing_interpolate()
    test_remove_outliers()
    print("\nAll tests passed!")

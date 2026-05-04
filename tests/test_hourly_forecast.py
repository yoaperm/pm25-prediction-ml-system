import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hourly_forecast import (
    build_hourly_supervised_features,
    build_next_hour_inference_features,
    hourly_feature_columns,
)


def _hourly_df(periods=80, start="2026-04-01 00:00:00"):
    return pd.DataFrame({
        "datetime": pd.date_range(start, periods=periods, freq="h"),
        "pm25": [float(i) for i in range(periods)],
    })


def test_build_hourly_supervised_features_creates_t_plus_1_target():
    df = _hourly_df()
    feat_df, feature_cols, target_col = build_hourly_supervised_features(df, forecast_hour=1)

    assert target_col == "pm25_h1"
    assert feature_cols == hourly_feature_columns()

    first = feat_df.iloc[0]
    source_idx = int(first["pm25"])
    assert first[target_col] == df.loc[source_idx + 1, "pm25"]


def test_hourly_features_do_not_leak_target_pm25():
    df = _hourly_df()
    feat_df, _, target_col = build_hourly_supervised_features(df, forecast_hour=1)

    row = feat_df.iloc[10]
    source_pm25 = row["pm25"]
    target_pm25 = row[target_col]

    assert row["pm25_lag_1"] == source_pm25
    assert row["pm25_lag_2"] == source_pm25 - 1
    assert row["pm25_diff_1h"] == 1
    assert row["pm25_lag_1"] != target_pm25


def test_time_features_describe_prediction_hour():
    df = _hourly_df(start="2026-04-03 23:00:00")
    feat_df, _, _ = build_hourly_supervised_features(df, forecast_hour=1)

    first = feat_df.iloc[0]
    assert first["target_datetime"] == first["datetime"] + pd.Timedelta(hours=1)
    assert first["hour"] == first["target_datetime"].hour
    assert first["day_of_week"] == first["target_datetime"].dayofweek


def test_next_hour_inference_features_returns_one_row_with_expected_columns():
    df = _hourly_df()
    feature_cols = hourly_feature_columns()

    feat_df = build_next_hour_inference_features(df, feature_cols)

    assert len(feat_df) == 1
    assert list(feat_df.columns) == feature_cols
    assert not feat_df.isnull().any().any()
    assert feat_df.iloc[0]["hour"] == (df["datetime"].iloc[-1] + pd.Timedelta(hours=1)).hour

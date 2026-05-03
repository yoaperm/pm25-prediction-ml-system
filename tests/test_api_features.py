"""
Tests for api._build_features — verifies the feature engineering
used at inference time produces the correct shape and columns.
"""

import datetime
import sys
import numpy as np
import pytest

sys.path.insert(0, "src")

# Minimal stubs so api.py loads without a real model or feature_columns.json
import json, os, types, unittest.mock as mock

# Patch open() only for feature_columns.json, let real files through
_FEATURE_COLS = [
    "pm25_lag_1", "pm25_lag_2", "pm25_lag_3", "pm25_lag_5", "pm25_lag_7",
    "pm25_rolling_mean_3", "pm25_rolling_std_3",
    "pm25_rolling_mean_7", "pm25_rolling_std_7",
    "pm25_rolling_mean_14", "pm25_rolling_std_14",
    "day_of_week", "month", "day_of_year", "is_weekend",
    "pm25_diff_1", "pm25_pct_change_1",
]

import importlib, builtins

_real_open = builtins.open

def _patched_open(path, *args, **kwargs):
    if str(path).endswith("feature_columns.json"):
        import io
        return io.StringIO(json.dumps(_FEATURE_COLS))
    if str(path).endswith("active_model.json"):
        import io
        return io.StringIO(json.dumps({"model_key": "random_forest", "backend": "onnx", "input_shape": "2d"}))
    return _real_open(path, *args, **kwargs)


def _make_history(n=20, base_date="2025-01-01", base_pm25=30.0):
    from api import DailyReading
    start = datetime.date.fromisoformat(base_date)
    return [
        DailyReading(date=start + datetime.timedelta(days=i), pm25=base_pm25 + i * 0.5)
        for i in range(n)
    ]


@pytest.fixture(scope="module")
def api_module():
    with mock.patch("builtins.open", side_effect=_patched_open), \
         mock.patch("onnxruntime.InferenceSession"), \
         mock.patch.dict(os.environ, {"INFERENCE_BACKEND": "onnxruntime"}):
        import api as _api
        yield _api


def test_build_features_returns_one_row(api_module):
    history = _make_history(20)
    result = api_module._build_features(history)
    assert len(result) == 1, f"Expected 1 row, got {len(result)}"


def test_build_features_correct_columns(api_module):
    history = _make_history(20)
    result = api_module._build_features(history)
    assert list(result.columns) == _FEATURE_COLS


def test_build_features_no_nans(api_module):
    history = _make_history(20)
    result = api_module._build_features(history)
    assert not result.isnull().any().any(), "Feature row contains NaN values"


def test_build_features_empty_on_too_short(api_module):
    history = _make_history(5)  # too few rows for rolling_14
    result = api_module._build_features(history)
    assert result.empty

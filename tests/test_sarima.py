import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from sarima_model import train_sarima_with_tuning, predict_sarima_rolling


def test_sarima_smoke():
    rng = np.random.default_rng(42)
    y = (rng.normal(30, 5, 100).cumsum() % 80) + 10
    train, test = y[:80], y[80:]
    model, params = train_sarima_with_tuning(train, seasonal_period=7)
    assert "order" in params
    assert "seasonal_order" in params
    preds = predict_sarima_rolling(model, train, test)
    assert preds.shape == test.shape
    assert not np.any(np.isnan(preds))

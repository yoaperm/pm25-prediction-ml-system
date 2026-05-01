from pmdarima import auto_arima
import numpy as np
import warnings


def train_sarima_with_tuning(y_train: np.ndarray, seasonal_period: int = 7):
    """AIC-stepwise order search. Returns fitted model + order info dict."""
    model = auto_arima(
        y_train,
        seasonal=True,
        m=seasonal_period,
        information_criterion="aic",
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
    )
    params = {
        "order": str(model.order),
        "seasonal_order": str(model.seasonal_order),
    }
    return model, params


def fit_predict_one_step(order: tuple, seasonal_order: tuple, history) -> float:
    """Refit SARIMAX on the provided history with a fixed order and forecast one step ahead.
    Used at inference time — no auto_arima, just a fast fit with known parameters."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = SARIMAX(list(history), order=order, seasonal_order=seasonal_order).fit(disp=False)
    forecast = fit.forecast(steps=1)
    return float(forecast.iloc[0] if hasattr(forecast, "iloc") else forecast[0])


def predict_sarima_rolling(model, y_train: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    """
    Rolling one-step-ahead forecast over the test set.
    After each prediction the true observation is appended so the next
    prediction conditions on real data — same evaluation protocol as
    the other models' one-step-ahead predict().
    """
    predictions = []
    for obs in y_test:
        pred = model.predict(n_periods=1)[0]
        predictions.append(pred)
        model.update([obs])
    return np.array(predictions)

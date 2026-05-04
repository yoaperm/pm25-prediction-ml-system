from pmdarima import auto_arima, ARIMA
import numpy as np
import warnings


def fit_sarima(order: tuple, seasonal_order: tuple, y_train: np.ndarray):
    """Fit ARIMA with a fixed (already-selected) order. Used to recompute prod MAE on a new test set."""
    model = ARIMA(order=order, seasonal_order=seasonal_order, suppress_warnings=True)
    model.fit(y_train)
    return model


def train_sarima_with_tuning(y_train: np.ndarray, seasonal_period: int = 7,
                             max_p: int = 5, max_q: int = 5,
                             max_P: int = 2, max_Q: int = 2,
                             max_d: int = 2, max_D: int = 1):
    """AIC-stepwise order search. Returns fitted model + order info dict."""
    model = auto_arima(
        y_train,
        seasonal=True,
        m=seasonal_period,
        max_p=max_p, max_q=max_q,
        max_P=max_P, max_Q=max_Q,
        max_d=max_d, max_D=max_D,
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


def predict_sarima_n_ahead_rolling(model, y_test_raw: np.ndarray, n_ahead: int = 24) -> np.ndarray:
    """
    Rolling n-step-ahead forecast over the test set.
    Updates the model with each true observation first, then predicts n steps ahead
    and takes the final step — used for the 24h pipeline (predict pm25[t+24]).
    """
    predictions = []
    for obs in y_test_raw:
        model.update([obs])
        pred = model.predict(n_ahead)[n_ahead - 1]
        predictions.append(pred)
    return np.array(predictions)


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

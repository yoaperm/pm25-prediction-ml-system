"""
FastAPI Inference Service
=========================
Serves next-day PM2.5 predictions using the best trained model (Random Forest).

Endpoints:
  GET  /health         — liveness check
  GET  /model/info     — loaded model metadata
  POST /predict        — predict next-day PM2.5; logs prediction to predictions_log.csv
  POST /actual         — record actual PM2.5 for a past date
  POST /retrain        — evaluate on prediction vs actual log; trigger retrain if MAE > threshold
"""

import json
import os
import datetime
from typing import List, Optional

import httpx
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR        = os.path.join(BASE_DIR, "models")
RESULTS_DIR       = os.path.join(BASE_DIR, "results")
PREDICTIONS_LOG   = os.path.join(RESULTS_DIR, "predictions_log.csv")
ACTUALS_LOG       = os.path.join(RESULTS_DIR, "actuals_log.csv")

MODEL_NAME        = os.environ.get("MODEL_NAME", "random_forest")
MODEL_PATH        = os.path.join(MODELS_DIR, f"{MODEL_NAME}.joblib")
FEATURE_COLS_PATH = os.path.join(MODELS_DIR, "feature_columns.json")

AIRFLOW_URL       = os.environ.get("AIRFLOW_URL", "http://airflow-webserver:8080")
AIRFLOW_USER      = os.environ.get("AIRFLOW_USER", "admin")
AIRFLOW_PASSWORD  = os.environ.get("AIRFLOW_PASSWORD", "admin")
MAE_THRESHOLD     = float(os.environ.get("MAE_THRESHOLD", "6.0"))
API_KEY           = os.environ.get("API_KEY", "foonalert-secret-key")

# ── Auth ───────────────────────────────────────────────────────────────────────
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(_api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key

# ── Load model at startup ──────────────────────────────────────────────────────
model = joblib.load(MODEL_PATH)
with open(FEATURE_COLS_PATH) as f:
    FEATURE_COLS: List[str] = json.load(f)

MIN_HISTORY_DAYS = 15
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PM2.5 Prediction API",
    description="Next-day PM2.5 forecast for Station 10T, Bangkok",
    version="1.0.0",
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class DailyReading(BaseModel):
    date: datetime.date = Field(..., example="2025-06-01")
    pm25: float = Field(..., ge=0, le=500, example=35.2)


class PredictRequest(BaseModel):
    history: List[DailyReading] = Field(
        ...,
        min_length=MIN_HISTORY_DAYS,
        description=f"At least {MIN_HISTORY_DAYS} consecutive daily PM2.5 readings (oldest first)",
    )


class PredictResponse(BaseModel):
    prediction_date: datetime.date
    predicted_pm25: float
    unit: str = "µg/m³"
    model: str


class ActualRequest(BaseModel):
    date: datetime.date = Field(..., description="The date the actual PM2.5 was measured", example="2025-06-16")
    pm25_actual: float  = Field(..., ge=0, le=500, example=38.5)


class ActualResponse(BaseModel):
    date: datetime.date
    pm25_actual: float
    matched_prediction: Optional[float]
    absolute_error: Optional[float]


class RetrainRequest(BaseModel):
    threshold: Optional[float] = Field(None, description=f"Override MAE threshold (default: {MAE_THRESHOLD} µg/m³)")
    min_pairs: Optional[int]   = Field(None, description="Minimum prediction+actual pairs needed to evaluate (default: 7)")


class RetrainResponse(BaseModel):
    evaluated_pairs: int
    mae: Optional[float]
    threshold: float
    retrain_triggered: bool
    reason: str
    dag_run_id: Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────────────────
def _build_features(history: List[DailyReading]) -> pd.DataFrame:
    df = pd.DataFrame([{"date": r.date, "pm25": r.pm25} for r in history])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    for lag in [1, 2, 3, 5, 7]:
        df[f"pm25_lag_{lag}"] = df["pm25"].shift(lag)

    for window in [3, 7, 14]:
        df[f"pm25_rolling_mean_{window}"] = df["pm25"].shift(1).rolling(window).mean()
        df[f"pm25_rolling_std_{window}"]  = df["pm25"].shift(1).rolling(window).std()

    df["day_of_week"]    = df["date"].dt.dayofweek
    df["month"]          = df["date"].dt.month
    df["day_of_year"]    = df["date"].dt.dayofyear
    df["is_weekend"]     = (df["day_of_week"] >= 5).astype(int)
    df["pm25_diff_1"]    = df["pm25"].shift(1).diff(1)
    df["pm25_pct_change_1"] = df["pm25"].shift(1).pct_change(1)

    return df.dropna(subset=FEATURE_COLS)


def _append_csv(path: str, row: dict):
    df  = pd.DataFrame([row])
    exists = os.path.exists(path)
    df.to_csv(path, mode="a", header=not exists, index=False)


def _trigger_airflow_dag(dag_run_id: str):
    resp = httpx.post(
        f"{AIRFLOW_URL}/api/v1/dags/pm25_training_pipeline/dagRuns",
        auth=(AIRFLOW_USER, AIRFLOW_PASSWORD),
        json={"dag_run_id": dag_run_id},
        timeout=10,
    )
    resp.raise_for_status()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model/info", dependencies=[Depends(verify_api_key)])
def model_info():
    return {
        "model_name": MODEL_NAME,
        "feature_columns": FEATURE_COLS,
        "min_history_days": MIN_HISTORY_DAYS,
    }


@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(verify_api_key)])
def predict(req: PredictRequest):
    feat_df = _build_features(req.history)

    if feat_df.empty:
        raise HTTPException(status_code=422, detail="Not enough valid rows after feature engineering.")

    last_row     = feat_df.iloc[[-1]]
    X            = last_row[FEATURE_COLS].values
    prediction   = float(np.round(model.predict(X)[0], 2))
    last_date    = last_row["date"].iloc[0].date()
    next_date    = last_date + datetime.timedelta(days=1)

    # Log prediction so monitoring can later compare against actuals
    _append_csv(PREDICTIONS_LOG, {
        "prediction_date": str(next_date),
        "predicted_pm25":  prediction,
        "model":           MODEL_NAME,
        "created_at":      datetime.datetime.now(datetime.UTC).isoformat(),
    })

    return PredictResponse(
        prediction_date=next_date,
        predicted_pm25=prediction,
        model=MODEL_NAME,
    )


@app.post("/actual", response_model=ActualResponse, dependencies=[Depends(verify_api_key)])
def record_actual(req: ActualRequest):
    """
    Call this endpoint the day AFTER a prediction was made, once the real
    PM2.5 value is known. Logs the actual so monitoring can compute true MAE.
    """
    matched_pred  = None
    abs_error     = None

    if os.path.exists(PREDICTIONS_LOG):
        preds_df = pd.read_csv(PREDICTIONS_LOG, parse_dates=["prediction_date"])
        match    = preds_df[preds_df["prediction_date"].dt.date == req.date]
        if not match.empty:
            matched_pred = float(match.iloc[-1]["predicted_pm25"])
            abs_error    = round(abs(req.pm25_actual - matched_pred), 4)

    _append_csv(ACTUALS_LOG, {
        "date":        str(req.date),
        "pm25_actual": req.pm25_actual,
        "recorded_at": datetime.datetime.now(datetime.UTC).isoformat(),
    })

    return ActualResponse(
        date=req.date,
        pm25_actual=req.pm25_actual,
        matched_prediction=matched_pred,
        absolute_error=abs_error,
    )


@app.post("/retrain", response_model=RetrainResponse, dependencies=[Depends(verify_api_key)])
def retrain(req: RetrainRequest = RetrainRequest()):
    """
    Joins predictions_log with actuals_log on date.
    Computes MAE over the last `rolling_window_days` matched pairs.
    Triggers Airflow retraining DAG if MAE > threshold.
    """
    threshold = req.threshold if req.threshold is not None else MAE_THRESHOLD
    min_pairs = req.min_pairs if req.min_pairs is not None else 7

    if not os.path.exists(PREDICTIONS_LOG) or not os.path.exists(ACTUALS_LOG):
        return RetrainResponse(
            evaluated_pairs=0,
            mae=None,
            threshold=threshold,
            retrain_triggered=False,
            reason="Not enough data yet — predictions_log or actuals_log missing. Call /predict and /actual first.",
        )

    preds_df   = pd.read_csv(PREDICTIONS_LOG,  parse_dates=["prediction_date"])
    actuals_df = pd.read_csv(ACTUALS_LOG, parse_dates=["date"])

    # Deduplicate — keep latest entry per date
    preds_df   = preds_df.sort_values("created_at").drop_duplicates("prediction_date", keep="last")
    actuals_df = actuals_df.sort_values("recorded_at").drop_duplicates("date", keep="last")

    # Join predictions to actuals on date
    merged = preds_df.merge(
        actuals_df,
        left_on="prediction_date",
        right_on="date",
        how="inner",
    ).sort_values("prediction_date")

    if merged.empty or len(merged) < min_pairs:
        return RetrainResponse(
            evaluated_pairs=len(merged),
            mae=None,
            threshold=threshold,
            retrain_triggered=False,
            reason=f"Only {len(merged)} matched pairs found — need at least {min_pairs} to evaluate.",
        )

    mae = float(np.mean(np.abs(merged["pm25_actual"] - merged["predicted_pm25"])))

    if mae <= threshold:
        return RetrainResponse(
            evaluated_pairs=len(merged),
            mae=round(mae, 4),
            threshold=threshold,
            retrain_triggered=False,
            reason=f"MAE {mae:.4f} is within threshold {threshold} — model is healthy.",
        )

    dag_run_id = f"api_trigger_{datetime.datetime.now(datetime.UTC).strftime('%Y%m%dT%H%M%S')}"
    try:
        _trigger_airflow_dag(dag_run_id)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Failed to trigger Airflow DAG: {e}")

    return RetrainResponse(
        evaluated_pairs=len(merged),
        mae=round(mae, 4),
        threshold=threshold,
        retrain_triggered=True,
        reason=f"MAE {mae:.4f} exceeded threshold {threshold} — retraining triggered.",
        dag_run_id=dag_run_id,
    )

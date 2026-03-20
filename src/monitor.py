"""
Monitor Module
==============
Two checks on the rolling window of (predicted, actual) pairs:

  1. MAE   — mean absolute error between predicted and actual PM2.5
  2. PSI   — Population Stability Index between prediction and actual distributions
             PSI < 0.1  : stable
             PSI 0.1–0.2: moderate shift (monitor)
             PSI > 0.2  : significant shift → retrain

Both use the same source: predictions_log.csv joined with actuals_log.csv.

Usage (standalone):
    PYTHONPATH=src python src/monitor.py
"""

import json
import os
import sys
import datetime

import numpy as np
import pandas as pd

MODELS_DIR      = "models"
RESULTS_DIR     = "results"
PREDICTIONS_LOG = os.path.join(RESULTS_DIR, "predictions_log.csv")
ACTUALS_LOG     = os.path.join(RESULTS_DIR, "actuals_log.csv")


# ── PSI ───────────────────────────────────────────────────────────────────────

def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index between expected (predicted) and actual distributions.
    Uses percentile-based binning so it works for any value range.
    """
    breakpoints = np.percentile(
        np.concatenate([expected, actual]),
        np.linspace(0, 100, bins + 1),
    )
    # Deduplicate edges (can happen with constant/sparse data)
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 3:
        return 0.0

    breakpoints[0]  = -np.inf
    breakpoints[-1] =  np.inf

    exp_counts = np.histogram(expected, bins=breakpoints)[0]
    act_counts = np.histogram(actual,   bins=breakpoints)[0]

    exp_pct = exp_counts / len(expected)
    act_pct = act_counts / len(actual)

    # Avoid log(0)
    exp_pct = np.where(exp_pct == 0, 1e-4, exp_pct)
    act_pct = np.where(act_pct == 0, 1e-4, act_pct)

    psi = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
    return round(psi, 4)


def psi_status(psi: float) -> str:
    if psi < 0.1:
        return "stable"
    if psi < 0.2:
        return "moderate"
    return "significant"


# ── Main monitoring function ──────────────────────────────────────────────────

def run_monitoring(config: dict) -> dict:
    """
    Load prediction+actual pairs, compute MAE and PSI,
    decide whether retraining is needed.

    Returns
    -------
    dict with keys:
        evaluated_pairs, mae, psi, psi_status,
        mae_degraded, psi_degraded, needs_retraining, timestamp
    """
    mon_cfg = config.get("monitoring", {})

    predictions_log     = mon_cfg.get("predictions_log",     PREDICTIONS_LOG)
    actuals_log         = mon_cfg.get("actuals_log",         ACTUALS_LOG)
    results_file        = mon_cfg.get("results_file",        "results/monitoring_results.csv")
    rolling_window_days = mon_cfg.get("rolling_window_days", 30)
    min_pairs           = mon_cfg.get("min_evaluation_pairs", 7)

    mae_cfg = mon_cfg.get("mae", {})
    mae_enabled   = mae_cfg.get("enabled",   True)
    mae_threshold = mae_cfg.get("threshold", 6.0)

    psi_cfg = mon_cfg.get("psi", {})
    psi_enabled   = psi_cfg.get("enabled",   True)
    psi_threshold = psi_cfg.get("threshold", 0.2)
    psi_bins      = psi_cfg.get("bins",      10)

    ts = datetime.datetime.now(datetime.UTC).isoformat()

    # ── Load and join logs ────────────────────────────────────────────────────
    if not os.path.exists(predictions_log) or not os.path.exists(actuals_log):
        print("[Monitor] predictions_log or actuals_log not found — skipping")
        return _empty_result(ts, mae_threshold, psi_threshold, results_file)

    preds_df   = pd.read_csv(predictions_log,  parse_dates=["prediction_date"])
    actuals_df = pd.read_csv(actuals_log, parse_dates=["date"])

    preds_df   = preds_df.sort_values("created_at").drop_duplicates("prediction_date", keep="last")
    actuals_df = actuals_df.sort_values("recorded_at").drop_duplicates("date", keep="last")

    merged = preds_df.merge(
        actuals_df, left_on="prediction_date", right_on="date", how="inner"
    ).sort_values("prediction_date")

    # Keep only rolling window
    if not merged.empty:
        cutoff = merged["prediction_date"].max() - pd.Timedelta(days=rolling_window_days)
        merged = merged[merged["prediction_date"] >= cutoff]

    evaluated_pairs = len(merged)

    if evaluated_pairs < min_pairs:
        print(f"[Monitor] Only {evaluated_pairs} pairs (need {min_pairs}) — skipping checks")
        return _empty_result(ts, mae_threshold, psi_threshold, results_file,
                             evaluated_pairs=evaluated_pairs)

    predicted = merged["predicted_pm25"].values
    actual    = merged["pm25_actual"].values

    # ── MAE check ─────────────────────────────────────────────────────────────
    mae          = None
    mae_degraded = False
    if mae_enabled:
        mae          = round(float(np.mean(np.abs(actual - predicted))), 4)
        mae_degraded = mae > mae_threshold
        status       = "DEGRADED" if mae_degraded else "OK"
        print(f"[Monitor] MAE={mae:.4f}  threshold={mae_threshold}  [{status}]")

    # ── PSI check ─────────────────────────────────────────────────────────────
    psi          = None
    psi_degraded = False
    if psi_enabled:
        psi          = compute_psi(predicted, actual, bins=psi_bins)
        psi_label    = psi_status(psi)
        psi_degraded = psi > psi_threshold
        status       = "DEGRADED" if psi_degraded else "OK"
        print(f"[Monitor] PSI={psi:.4f}  threshold={psi_threshold}  status={psi_label}  [{status}]")

    needs_retraining = mae_degraded or psi_degraded
    print(f"[Monitor] needs_retraining={needs_retraining}  "
          f"(mae_degraded={mae_degraded}, psi_degraded={psi_degraded})")

    result = {
        "timestamp":        ts,
        "evaluated_pairs":  evaluated_pairs,
        "mae":              mae,
        "mae_threshold":    mae_threshold,
        "mae_degraded":     mae_degraded,
        "psi":              psi,
        "psi_threshold":    psi_threshold,
        "psi_status":       psi_status(psi) if psi is not None else None,
        "psi_degraded":     psi_degraded,
        "needs_retraining": needs_retraining,
    }

    _save_result(result, results_file)
    return result


def _empty_result(ts, mae_threshold, psi_threshold, results_file, evaluated_pairs=0):
    result = {
        "timestamp":        ts,
        "evaluated_pairs":  evaluated_pairs,
        "mae":              None,
        "mae_threshold":    mae_threshold,
        "mae_degraded":     False,
        "psi":              None,
        "psi_threshold":    psi_threshold,
        "psi_status":       None,
        "psi_degraded":     False,
        "needs_retraining": False,
    }
    _save_result(result, results_file)
    return result


def _save_result(result: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row    = pd.DataFrame([result])
    exists = os.path.exists(path)
    row.to_csv(path, mode="a", header=not exists, index=False)


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from data_loader import load_config
    config = load_config()
    result = run_monitoring(config)
    if result["needs_retraining"]:
        reasons = []
        if result["mae_degraded"]:
            reasons.append(f"MAE {result['mae']} > {result['mae_threshold']}")
        if result["psi_degraded"]:
            reasons.append(f"PSI {result['psi']} > {result['psi_threshold']} ({result['psi_status']})")
        print(f"[Monitor] Retraining needed: {', '.join(reasons)}")
        sys.exit(1)
    else:
        print("[Monitor] Model healthy — no retraining needed.")

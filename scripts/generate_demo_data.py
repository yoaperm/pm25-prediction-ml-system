#!/usr/bin/env python3
"""
generate_demo_data.py
=====================
Generate demo data for FoonAlert Spike Replay mode.

Creates:
1. Actual hourly PM2.5 for selected spike days
2. Simulated model predictions (Persistence, SARIMA, LSTM, Transformer)
   with realistic error patterns

These mock predictions demonstrate what each model would output.
Replace with real model predictions once models are trained.

Usage:
    python scripts/generate_demo_data.py
    python scripts/generate_demo_data.py --station 59 --date 2025-01-24
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path("data/backup/pm25_raw_hourly_20240101_20250415.csv")
OUTPUT_DIR = Path("demo_data")

# Curated spike days for demo
DEMO_DAYS = [
    {"station_id": 59, "date": "2025-01-24", "label": "Morning spike + evening rebound"},
    {"station_id": 61, "date": "2024-12-24", "label": "Night peak → drop → evening surge"},
    {"station_id": 58, "date": "2025-03-14", "label": "Low baseline → sudden spike"},
    {"station_id": 57, "date": "2024-02-14", "label": "Night peak sustained"},
    {"station_id": 56, "date": "2024-04-30", "label": "Clean moderate spike"},
]

STATION_NAMES = {
    56: "Din Daeng",
    57: "Bang Khun Thian",
    58: "Khlong Toei",
    59: "Wang Thonglang",
    61: "Lat Phrao",
}


def generate_persistence_predictions(actual: pd.Series) -> pd.Series:
    """Persistence model: predict current value for all future horizons."""
    return actual.shift(1).fillna(method="bfill")


def generate_sarima_predictions(actual: pd.Series, noise_std: float = 8.0) -> pd.Series:
    """
    Simulated SARIMA: good at daily patterns, misses sudden spikes.
    - Captures overall trend with lag
    - Underestimates peaks
    - Good at stable periods
    """
    np.random.seed(42)
    # SARIMA-like: weighted avg of recent + daily seasonality
    smoothed = actual.rolling(window=3, min_periods=1).mean()
    # Add slight undershoot on spikes
    spike_mask = actual > actual.rolling(6, min_periods=1).mean() * 1.3
    sarima_pred = smoothed.copy()
    sarima_pred[spike_mask] = sarima_pred[spike_mask] * 0.82
    # Add noise
    noise = np.random.normal(0, noise_std, len(actual))
    sarima_pred = sarima_pred + noise
    # Shift by 1 (prediction made 1 hour before)
    sarima_pred = sarima_pred.shift(1).fillna(method="bfill")
    return sarima_pred.clip(lower=0)


def generate_lstm_predictions(actual: pd.Series, noise_std: float = 6.0) -> pd.Series:
    """
    Simulated LSTM: good at short-term trends, catches spikes better.
    - Responds faster to changes
    - Slight overshoot on sudden drops
    - Better at 1-3h horizon
    """
    np.random.seed(123)
    # LSTM-like: follows trend more closely
    ema = actual.ewm(span=3, adjust=False).mean()
    # Better at catching trends
    trend = actual.diff().fillna(0)
    lstm_pred = ema + trend * 0.4
    # Add noise (less than SARIMA)
    noise = np.random.normal(0, noise_std, len(actual))
    lstm_pred = lstm_pred + noise
    # Shift by 1
    lstm_pred = lstm_pred.shift(1).fillna(method="bfill")
    return lstm_pred.clip(lower=0)


def generate_transformer_predictions(actual: pd.Series, noise_std: float = 5.0) -> pd.Series:
    """
    Simulated Transformer: best at long-range patterns, catches spikes earliest.
    - Uses attention over longer window
    - Detects pattern changes early
    - Best overall accuracy
    """
    np.random.seed(456)
    # Transformer-like: looks at longer context
    long_ema = actual.ewm(span=6, adjust=False).mean()
    short_trend = actual.diff().fillna(0)
    long_trend = actual.diff(3).fillna(0) / 3
    # Combines long + short context
    transformer_pred = long_ema + short_trend * 0.5 + long_trend * 0.3
    # Add noise (least noisy)
    noise = np.random.normal(0, noise_std, len(actual))
    transformer_pred = transformer_pred + noise
    # Shift by 1
    transformer_pred = transformer_pred.shift(1).fillna(method="bfill")
    return transformer_pred.clip(lower=0)


def generate_multi_horizon_predictions(actual: pd.Series, model_fn, horizons=[1, 6, 24]):
    """Generate predictions at multiple horizons for a given model."""
    results = {}
    for h in horizons:
        # For longer horizons, increase noise proportionally
        shifted_actual = actual.shift(-h)  # What will happen h hours from now
        if shifted_actual.isna().all():
            results[f"pred_h{h}"] = pd.Series([np.nan] * len(actual))
            continue
        # Generate prediction as if we're predicting h hours ahead
        pred = model_fn(shifted_actual.fillna(method="ffill"))
        # Add horizon-dependent noise
        horizon_noise = np.random.normal(0, h * 0.8, len(pred))
        pred = pred + horizon_noise
        results[f"pred_h{h}"] = pred.clip(lower=0)
    return results


def export_spike_day(df: pd.DataFrame, station_id: int, date_str: str, output_dir: Path):
    """Export actual data and model predictions for a spike day."""
    date = pd.Timestamp(date_str, tz="UTC")
    station_df = df[df["station_id"] == station_id].sort_values("timestamp")

    # Get 48h context (24h before + the spike day)
    start = date - pd.Timedelta(hours=24)
    end = date + pd.Timedelta(hours=24)
    mask = (station_df["timestamp"] >= start) & (station_df["timestamp"] < end)
    day_data = station_df[mask].copy().reset_index(drop=True)

    if day_data.empty:
        print(f"  Warning: No data for station {station_id} on {date_str}")
        return

    actual = day_data["pm25"].fillna(method="ffill").fillna(method="bfill")

    # Generate predictions from each model
    predictions = pd.DataFrame({
        "timestamp": day_data["timestamp"],
        "actual": actual,
        "persistence": generate_persistence_predictions(actual),
        "sarima": generate_sarima_predictions(actual),
        "lstm": generate_lstm_predictions(actual),
        "transformer": generate_transformer_predictions(actual),
    })

    # Add multi-horizon predictions for the spike day portion only
    spike_day_mask = day_data["timestamp"].dt.date == pd.Timestamp(date_str).date()
    for model_name, model_fn in [
        ("sarima", generate_sarima_predictions),
        ("lstm", generate_lstm_predictions),
        ("transformer", generate_transformer_predictions),
    ]:
        horizons = generate_multi_horizon_predictions(actual, model_fn)
        for col, vals in horizons.items():
            predictions[f"{model_name}_{col}"] = vals

    # Add metadata
    predictions["station_id"] = station_id
    predictions["station_name"] = STATION_NAMES.get(station_id, f"Station {station_id}")

    # Save
    out_file = output_dir / f"replay_station{station_id}_{date_str}.csv"
    predictions.to_csv(out_file, index=False)
    print(f"  ✅ {out_file} ({len(predictions)} rows)")

    return predictions


def generate_model_metrics():
    """Generate overall model performance metrics for the scoreboard."""
    metrics = pd.DataFrame({
        "model": ["Persistence", "SARIMA", "LSTM", "Transformer"],
        "mae_1h": [8.2, 6.8, 5.4, 5.1],
        "mae_6h": [15.3, 10.2, 8.1, 7.3],
        "mae_24h": [22.1, 13.5, 11.8, 10.2],
        "rmse_1h": [11.5, 9.3, 7.8, 7.2],
        "rmse_6h": [19.8, 13.7, 11.2, 10.1],
        "rmse_24h": [28.4, 17.8, 15.6, 13.9],
        "r2_1h": [0.72, 0.81, 0.87, 0.89],
        "r2_6h": [0.45, 0.67, 0.74, 0.78],
        "r2_24h": [0.21, 0.52, 0.58, 0.64],
        "spike_recall": [0.23, 0.71, 0.82, 0.87],
        "spike_precision": [0.45, 0.65, 0.73, 0.79],
        "avg_early_detection_hours": [0.0, 2.1, 3.2, 4.1],
        "training_time_min": [0, 2, 15, 45],
        "inference_time_ms": [0.1, 5, 8, 12],
    })
    return metrics


def generate_error_analysis():
    """Generate error analysis data broken down by horizon and severity."""
    # Error by horizon
    horizons = [1, 2, 3, 6, 12, 24]
    models = ["Persistence", "SARIMA", "LSTM", "Transformer"]

    rows = []
    for model in models:
        for h in horizons:
            base_error = {"Persistence": 8, "SARIMA": 6, "LSTM": 5, "Transformer": 4.5}[model]
            growth = {"Persistence": 1.5, "SARIMA": 0.8, "LSTM": 0.6, "Transformer": 0.5}[model]
            mae = base_error + growth * h + np.random.uniform(-0.5, 0.5)
            rows.append({"model": model, "horizon_h": h, "mae": round(mae, 1)})

    error_by_horizon = pd.DataFrame(rows)

    # Error by severity level
    severity_levels = ["Good (0-25)", "Moderate (25-37.5)", "Unhealthy-S (37.5-75)",
                       "Unhealthy (75-100)", "Very Unhealthy (>100)"]
    rows = []
    for model in models:
        for i, sev in enumerate(severity_levels):
            base = {"Persistence": 6, "SARIMA": 5, "LSTM": 4, "Transformer": 3.5}[model]
            # Error increases with severity
            mae = base + i * 2.5 + np.random.uniform(-0.5, 0.5)
            rows.append({"model": model, "severity": sev, "mae": round(mae, 1)})

    error_by_severity = pd.DataFrame(rows)

    return error_by_horizon, error_by_severity


def main():
    parser = argparse.ArgumentParser(description="Generate FoonAlert demo data")
    parser.add_argument("--station", type=int, help="Generate for specific station only")
    parser.add_argument("--date", type=str, help="Generate for specific date (YYYY-MM-DD)")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--data-path", type=str, default=str(DATA_PATH))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        return

    print("Loading hourly data...")
    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Generate replay data for each demo day
    if args.station and args.date:
        days_to_process = [{"station_id": args.station, "date": args.date, "label": "custom"}]
    else:
        days_to_process = DEMO_DAYS

    print(f"\nGenerating replay data for {len(days_to_process)} spike days...")
    for day_info in days_to_process:
        sid = day_info["station_id"]
        date = day_info["date"]
        label = day_info["label"]
        print(f"\n  Station {sid} ({STATION_NAMES.get(sid, '?')}) — {date}: {label}")
        export_spike_day(df, sid, date, output_dir)

    # Generate model metrics
    print("\nGenerating model metrics...")
    metrics = generate_model_metrics()
    metrics.to_csv(output_dir / "model_metrics.csv", index=False)
    print(f"  ✅ {output_dir / 'model_metrics.csv'}")

    # Generate error analysis
    print("Generating error analysis...")
    error_horizon, error_severity = generate_error_analysis()
    error_horizon.to_csv(output_dir / "error_by_horizon.csv", index=False)
    error_severity.to_csv(output_dir / "error_by_severity.csv", index=False)
    print(f"  ✅ {output_dir / 'error_by_horizon.csv'}")
    print(f"  ✅ {output_dir / 'error_by_severity.csv'}")

    # Generate spike days index
    print("Generating spike days index...")
    spike_index = pd.DataFrame(DEMO_DAYS)
    spike_index["station_name"] = spike_index["station_id"].map(STATION_NAMES)
    spike_index.to_csv(output_dir / "spike_days_index.csv", index=False)
    print(f"  ✅ {output_dir / 'spike_days_index.csv'}")

    print(f"\n🎉 Demo data generated in {output_dir}/")
    print("   Run the demo: streamlit run app_foonalert_demo.py")


if __name__ == "__main__":
    main()

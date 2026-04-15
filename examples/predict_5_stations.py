#!/usr/bin/env python3
"""
Predict PM2.5 for Stations 56, 57, 58, 59, 61
==============================================
Uses Triton Inference Server to make predictions for all 5 stations.

Usage:
    python examples/predict_5_stations.py
"""

import numpy as np
import tritonclient.http as httpclient
from datetime import datetime, timedelta


def build_sample_features():
    """
    Build sample 19 features for 24h forecast.

    In production, these would be built from actual hourly PM2.5 data.
    """
    now = datetime.now()

    # Simulate realistic PM2.5 features
    features = np.array([[
        # Lags (6): PM2.5 for t-1, t-2, t-3, t-6, t-12, t-24 hours ago
        28.5,  # 1 hour ago
        29.1,  # 2 hours ago
        30.2,  # 3 hours ago
        32.5,  # 6 hours ago
        31.8,  # 12 hours ago
        35.0,  # 24 hours ago

        # Rolling statistics (6): mean/std for 6h, 12h, 24h windows
        28.5,  # rolling_mean_6h
        3.2,   # rolling_std_6h
        29.1,  # rolling_mean_12h
        4.1,   # rolling_std_12h
        30.2,  # rolling_mean_24h
        3.8,   # rolling_std_24h

        # Differences (2)
        0.5,   # pm25_diff_1h  (change from 1h ago)
        -2.3,  # pm25_diff_24h (change from 24h ago)

        # Time features (5)
        now.hour,                        # hour of day (0-23)
        now.weekday(),                   # day of week (0=Mon, 6=Sun)
        now.month,                       # month (1-12)
        now.timetuple().tm_yday,         # day of year (1-365)
        1 if now.weekday() >= 5 else 0   # is_weekend
    ]], dtype=np.float32)

    return features


def classify_air_quality(pm25_value):
    """
    Classify PM2.5 into Thai AQI categories.

    Categories (Thai standard):
    - 0-25:   Good (Green)
    - 26-37:  Moderate (Yellow)
    - 38-50:  Unhealthy for Sensitive Groups (Orange)
    - 51-90:  Unhealthy (Red)
    - 90+:    Very Unhealthy (Purple)
    """
    if pm25_value <= 25:
        return "🟢 Good", "green"
    elif pm25_value <= 37:
        return "🟡 Moderate", "yellow"
    elif pm25_value <= 50:
        return "🟠 Unhealthy (Sensitive)", "orange"
    elif pm25_value <= 90:
        return "🔴 Unhealthy", "red"
    else:
        return "🟣 Very Unhealthy", "purple"


def predict_single_station(client, station_id, features):
    """
    Make prediction for a single station.

    Parameters
    ----------
    client : tritonclient.http.InferenceServerClient
        Connected Triton client
    station_id : int
        Station ID (56, 57, 58, 59, or 61)
    features : np.ndarray
        Feature array (1, 19)

    Returns
    -------
    prediction : float or None
        Predicted PM2.5 value, or None if error
    """
    model_name = f"pm25_{station_id}"

    try:
        # Prepare input
        input_tensor = httpclient.InferInput("float_input", features.shape, "FP32")
        input_tensor.set_data_from_numpy(features)

        # Request output
        output_tensor = httpclient.InferRequestedOutput("variable")

        # Make inference
        result = client.infer(
            model_name=model_name,
            inputs=[input_tensor],
            outputs=[output_tensor]
        )

        # Extract prediction
        prediction = result.as_numpy("variable")[0][0]
        return round(float(prediction), 2)

    except Exception as e:
        print(f"   Error predicting station {station_id}: {e}")
        return None


def predict_all_stations():
    """Main function: predict PM2.5 for all 5 stations."""

    print("="*80)
    print("PM2.5 24-Hour Forecast for Stations 56, 57, 58, 59, 61")
    print("="*80)

    # 1. Connect to Triton
    print("\n1. Connecting to Triton Inference Server...")
    triton_url = "localhost:8010"

    try:
        client = httpclient.InferenceServerClient(url=triton_url)

        if client.is_server_ready():
            print(f"   ✓ Connected to Triton at {triton_url}")
        else:
            print(f"   ✗ Triton server not ready")
            return

    except Exception as e:
        print(f"   ✗ Connection failed: {e}")
        print("\nMake sure Triton is running:")
        print("   docker compose up triton")
        return

    # 2. Check which models are available
    print("\n2. Checking available models...")
    stations = [56, 57, 58, 59, 61]
    available_models = []

    for station_id in stations:
        model_name = f"pm25_{station_id}"
        try:
            if client.is_model_ready(model_name):
                print(f"   ✓ {model_name} ready")
                available_models.append(station_id)
            else:
                print(f"   ✗ {model_name} not ready")
        except:
            print(f"   ✗ {model_name} not found")

    if not available_models:
        print("\n   No models available. Run training first:")
        print("   docker exec <airflow-container> airflow dags trigger pm25_24h_training -c '{\"station_id\": 56}'")
        return

    # 3. Build features
    print("\n3. Building features...")
    features = build_sample_features()
    print(f"   Feature shape: {features.shape}")
    print(f"   Sample features: [{features[0][0]:.1f}, {features[0][1]:.1f}, {features[0][2]:.1f}, ...]")

    # 4. Make predictions
    print("\n4. Making predictions...")
    forecast_time = datetime.now() + timedelta(hours=24)
    print(f"   Forecast for: {forecast_time.strftime('%Y-%m-%d %H:%M')}")

    print(f"\n   {'Station':<10} {'PM2.5':<12} {'Air Quality':<25} {'Category'}")
    print(f"   {'-'*70}")

    predictions = {}

    for station_id in stations:
        if station_id not in available_models:
            print(f"   {station_id:<10} {'N/A':<12} {'Model not available':<25}")
            continue

        # Predict
        prediction = predict_single_station(client, station_id, features)

        if prediction is not None:
            predictions[station_id] = prediction
            quality, color = classify_air_quality(prediction)
            print(f"   {station_id:<10} {prediction:>6.2f} µg/m³  {quality:<25} {color}")
        else:
            print(f"   {station_id:<10} {'ERROR':<12}")

    # 5. Summary
    print(f"\n   {'-'*70}")

    if predictions:
        avg_pm25 = sum(predictions.values()) / len(predictions)
        max_station = max(predictions, key=predictions.get)
        min_station = min(predictions, key=predictions.get)

        print(f"\n5. Summary:")
        print(f"   Stations predicted: {len(predictions)}/5")
        print(f"   Average PM2.5: {avg_pm25:.2f} µg/m³")
        print(f"   Highest: Station {max_station} ({predictions[max_station]:.2f} µg/m³)")
        print(f"   Lowest:  Station {min_station} ({predictions[min_station]:.2f} µg/m³)")

        # Overall air quality
        overall_quality, _ = classify_air_quality(avg_pm25)
        print(f"   Overall air quality: {overall_quality}")
    else:
        print(f"\n5. No predictions made")

    print("\n" + "="*80)
    print("✓ Predictions complete")
    print("="*80)

    # Return predictions for programmatic use
    return predictions


def predict_specific_station(station_id):
    """
    Predict for a specific station only.

    Parameters
    ----------
    station_id : int
        Station ID (56, 57, 58, 59, or 61)

    Returns
    -------
    prediction : float or None
    """
    client = httpclient.InferenceServerClient(url="localhost:8010")
    features = build_sample_features()
    return predict_single_station(client, station_id, features)


if __name__ == "__main__":
    # Predict all stations
    predictions = predict_all_stations()

    # Example: Access individual predictions
    if predictions:
        print("\nProgrammatic access example:")
        print(f"predictions = {predictions}")
        print(f"Station 56 PM2.5: {predictions.get(56, 'N/A')}")

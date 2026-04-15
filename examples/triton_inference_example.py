#!/usr/bin/env python3
"""
Triton Inference Example
========================
Demonstrates how to make predictions using Triton Inference Server.

Usage:
    python examples/triton_inference_example.py
"""

import numpy as np
import tritonclient.http as httpclient
from datetime import datetime, timedelta


def build_sample_features():
    """Build sample 17 features for testing."""
    # Example features (realistic PM2.5 values and time features)
    features = np.array([[
        # Lag features (5): pm25 for t-1, t-2, t-3, t-5, t-7 days
        28.5, 29.1, 30.2, 32.5, 35.0,

        # Rolling statistics (4): mean/std for 3-day and 7-day windows
        28.5, 3.2,  # rolling_mean_3, rolling_std_3
        29.1, 4.1,  # rolling_mean_7, rolling_std_7

        # Time features (8)
        4,   # month (April)
        107, # day_of_year
        2,   # day_of_week (Wednesday=2)
        2026,# year
        2,   # quarter (Q2)
        15,  # week_of_year
        16,  # day_of_month
        0    # is_weekend (0=weekday)
    ]], dtype=np.float32)

    return features


def predict_single_station(triton_client, model_name, features):
    """Make prediction for a single station."""
    # Create input tensor
    input_tensor = httpclient.InferInput("float_input", features.shape, "FP32")
    input_tensor.set_data_from_numpy(features)

    # Request output
    output_tensor = httpclient.InferRequestedOutput("variable")

    # Make inference
    result = triton_client.infer(
        model_name=model_name,
        inputs=[input_tensor],
        outputs=[output_tensor]
    )

    # Extract prediction
    prediction = result.as_numpy("variable")[0][0]
    return prediction


def classify_air_quality(pm25_value):
    """Classify PM2.5 value into air quality category (Thai AQI)."""
    if pm25_value <= 25:
        return "Good", "🟢"
    elif pm25_value <= 37:
        return "Moderate", "🟡"
    elif pm25_value <= 50:
        return "Unhealthy for Sensitive Groups", "🟠"
    elif pm25_value <= 90:
        return "Unhealthy", "🔴"
    else:
        return "Very Unhealthy", "🟣"


def main():
    """Run inference examples."""
    print("="*70)
    print("Triton Inference Server - PM2.5 Prediction Example")
    print("="*70)

    # 1. Connect to Triton
    print("\n1. Connecting to Triton...")
    triton_url = "localhost:8010"

    try:
        triton_client = httpclient.InferenceServerClient(url=triton_url)
        print(f"   ✓ Connected to Triton at {triton_url}")
    except Exception as e:
        print(f"   ✗ Failed to connect: {e}")
        print("\nMake sure Triton is running:")
        print("   docker compose up triton")
        return

    # 2. Check Triton health
    print("\n2. Checking Triton health...")
    try:
        if triton_client.is_server_ready():
            print("   ✓ Triton server is ready")
        else:
            print("   ✗ Triton server is not ready")
            return
    except Exception as e:
        print(f"   ✗ Health check failed: {e}")
        return

    # 3. List available models
    print("\n3. Available models:")
    models = {
        "pm25": "Station 10T (Daily forecast)",
        "pm25_63": "Station 63",
        "pm25_64": "Station 64",
        "pm25_65": "Station 65",
        "pm25_66": "Station 66",
        "pm25_67": "Station 67",
    }

    for model_name, description in models.items():
        try:
            if triton_client.is_model_ready(model_name):
                print(f"   ✓ {model_name:12} - {description}")
            else:
                print(f"   ✗ {model_name:12} - Not ready")
        except:
            print(f"   ✗ {model_name:12} - Not found")

    # 4. Build sample features
    print("\n4. Building sample features (17 features)...")
    features = build_sample_features()
    print(f"   Features shape: {features.shape}")
    print(f"   Sample values: {features[0][:5]} ... (first 5 lags)")

    # 5. Make predictions for all stations
    print("\n5. Making predictions for all stations...")
    prediction_date = datetime.now().date() + timedelta(days=1)
    print(f"   Forecast date: {prediction_date}")
    print(f"\n   {'Model':<15} {'Station':<25} {'PM2.5':<12} {'Air Quality'}")
    print(f"   {'-'*80}")

    for model_name, description in models.items():
        try:
            prediction = predict_single_station(triton_client, model_name, features)
            quality, emoji = classify_air_quality(prediction)

            print(f"   {model_name:<15} {description:<25} {prediction:>6.2f} µg/m³  "
                  f"{emoji} {quality}")
        except Exception as e:
            print(f"   {model_name:<15} {description:<25} ERROR: {e}")

    # 6. Batch prediction example
    print("\n6. Batch prediction example (5 samples)...")
    batch_features = np.random.rand(5, 17).astype(np.float32) * 50  # Random values

    try:
        input_tensor = httpclient.InferInput("float_input", batch_features.shape, "FP32")
        input_tensor.set_data_from_numpy(batch_features)
        output_tensor = httpclient.InferRequestedOutput("variable")

        result = triton_client.infer(
            model_name="pm25",
            inputs=[input_tensor],
            outputs=[output_tensor]
        )

        predictions = result.as_numpy("variable").flatten()
        print(f"   Batch predictions: {predictions}")
        print(f"   ✓ Successfully processed batch of {len(predictions)} samples")
    except Exception as e:
        print(f"   ✗ Batch prediction failed: {e}")

    # 7. Performance benchmark
    print("\n7. Performance benchmark (100 requests)...")
    import time

    n_requests = 100
    start_time = time.time()

    for _ in range(n_requests):
        try:
            predict_single_station(triton_client, "pm25", features)
        except:
            pass

    elapsed = time.time() - start_time
    avg_latency = (elapsed / n_requests) * 1000
    throughput = n_requests / elapsed

    print(f"   Total time: {elapsed:.2f}s")
    print(f"   Avg latency: {avg_latency:.2f}ms")
    print(f"   Throughput: {throughput:.2f} req/s")

    print("\n" + "="*70)
    print("✓ All examples completed successfully!")
    print("="*70)
    print("\nNext steps:")
    print("  - View full API guide: docs/TRITON_API_GUIDE.md")
    print("  - Integrate with your application using tritonclient.http")
    print("  - Use gRPC for better performance: tritonclient.grpc")


if __name__ == "__main__":
    main()

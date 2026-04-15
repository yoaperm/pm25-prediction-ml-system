#!/usr/bin/env python3
"""
Use All Deployed Models
========================
Shows how to use all 6 station models via Triton or FastAPI.

Each station has ONE model deployed (the best of the 5 trained types).
"""

import numpy as np
import requests


# =============================================================================
# METHOD 1: Direct Triton API (Fastest - Production)
# =============================================================================

def use_triton_api():
    """Use Triton directly for predictions."""
    import tritonclient.http as httpclient

    print("="*70)
    print("METHOD 1: Direct Triton API")
    print("="*70)

    # Connect to Triton
    client = httpclient.InferenceServerClient(url="localhost:8010")

    # Sample features (17 features for daily forecast)
    features = np.array([[
        28.5, 29.1, 30.2, 32.5, 35.0,  # lags
        28.5, 3.2, 29.1, 4.1,           # rolling stats
        4, 107, 2, 2026, 2, 15, 16, 0   # time features
    ]], dtype=np.float32)

    # Prepare input
    input_tensor = httpclient.InferInput("float_input", features.shape, "FP32")
    input_tensor.set_data_from_numpy(features)
    output_tensor = httpclient.InferRequestedOutput("variable")

    # Predict for each station
    stations = {
        "pm25": "Station 10T",
        "pm25_63": "Station 63",
        "pm25_64": "Station 64",
        "pm25_65": "Station 65",
        "pm25_66": "Station 66",
        "pm25_67": "Station 67",
    }

    print(f"\n{'Model':<12} {'Station':<15} {'Prediction':<15} {'Model Type'}")
    print("-"*70)

    for model_name, station_name in stations.items():
        result = client.infer(
            model_name=model_name,
            inputs=[input_tensor],
            outputs=[output_tensor]
        )
        prediction = result.as_numpy("variable")[0][0]

        # Get model metadata to see which algorithm is deployed
        model_info = client.get_model_metadata(model_name)

        print(f"{model_name:<12} {station_name:<15} {prediction:>6.2f} µg/m³     "
              f"{model_info['platform']}")


# =============================================================================
# METHOD 2: FastAPI Endpoint (Easier - Development)
# =============================================================================

def use_fastapi():
    """Use FastAPI wrapper (simpler but slower)."""
    print("\n" + "="*70)
    print("METHOD 2: FastAPI Endpoint")
    print("="*70)

    API_URL = "http://localhost:8001"
    API_KEY = "foonalert-secret-key"

    headers = {"X-API-Key": API_KEY}

    # Sample PM2.5 history (last 15 days)
    history_data = {
        "data": [
            {"date": "2026-04-01", "pm25": 28.5},
            {"date": "2026-04-02", "pm25": 29.1},
            {"date": "2026-04-03", "pm25": 30.2},
            {"date": "2026-04-04", "pm25": 32.5},
            {"date": "2026-04-05", "pm25": 35.0},
            {"date": "2026-04-06", "pm25": 33.2},
            {"date": "2026-04-07", "pm25": 31.8},
            {"date": "2026-04-08", "pm25": 29.5},
            {"date": "2026-04-09", "pm25": 27.3},
            {"date": "2026-04-10", "pm25": 28.9},
            {"date": "2026-04-11", "pm25": 30.1},
            {"date": "2026-04-12", "pm25": 31.5},
            {"date": "2026-04-13", "pm25": 29.8},
            {"date": "2026-04-14", "pm25": 28.2},
            {"date": "2026-04-15", "pm25": 30.7},
        ]
    }

    # Make prediction
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=history_data,
            headers=headers
        )

        if response.status_code == 200:
            result = response.json()
            print(f"\nPrediction: {result['prediction']} µg/m³")
            print(f"Next date: {result['next_date']}")
            print(f"Model used: {result['model_name']}")
            print(f"Backend: {result['inference_backend']}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure FastAPI is running:")
        print("  docker compose up api")


# =============================================================================
# METHOD 3: Compare All 5 Model Types (Training/Evaluation Only)
# =============================================================================

def compare_model_types():
    """
    During training, all 5 types are evaluated.
    Only the best is deployed, but you can access temp models during training.
    """
    print("\n" + "="*70)
    print("METHOD 3: Compare Model Types (Training Time Only)")
    print("="*70)

    print("""
During training, the pipeline:

1. Trains 5 model types in parallel:
   - Linear Regression (baseline)
   - Ridge Regression (L2 regularization)
   - Random Forest (tree ensemble)
   - XGBoost (gradient boosting)
   - LSTM (deep learning)

2. Evaluates all 5 on test set

3. Deploys ONLY the best one to:
   - models/station_{id}_24h/onnx/{best_model}.onnx
   - models/station_{id}_24h/active_model.json
   - Triton model repository

To access all 5 models:
  1. They exist as temp files during training:
     - models/station_{id}_24h/_tmp_linear_regression.onnx
     - models/station_{id}_24h/_tmp_ridge_regression.onnx
     - models/station_{id}_24h/_tmp_random_forest.onnx
     - models/station_{id}_24h/_tmp_xgboost.onnx
     - models/station_{id}_24h/_tmp_lstm.onnx

  2. View results in: results/forecast_24h_results.csv

  3. MLflow tracks all experiments:
     http://localhost:5001
     Experiment: pm25_24h_station_{id}
""")


# =============================================================================
# METHOD 4: Switch Active Model (Manual Override)
# =============================================================================

def switch_active_model_example():
    """Show how to manually switch which model is active."""
    print("\n" + "="*70)
    print("METHOD 4: Switch Active Model (Advanced)")
    print("="*70)

    print("""
To manually switch to a different model type:

1. Find the model you want in:
   models/station_56_24h/onnx/

2. Update active_model.json:
   {
     "onnx_file": "ridge_regression_2024-01-01_2025-10-14.onnx",
     "model_key": "ridge_regression",
     "station_id": 56,
     "train_start": "2024-01-01",
     "train_end": "2025-10-14",
     "is_lstm": false,
     "forecast_hour": 24,
     "n_features": 19
   }

3. Republish to Triton:

   from src.triton_utils import publish_to_triton

   publish_to_triton(
       onnx_path="models/station_56_24h/onnx/ridge_regression_2024-01-01_2025-10-14.onnx",
       triton_repo="/path/to/triton_model_repo",
       is_lstm=False
   )

4. Restart API if using FastAPI backend:
   docker compose restart api

Note: This is rarely needed - the training pipeline already selects the best!
""")


# =============================================================================
# METHOD 5: View MLflow Experiments
# =============================================================================

def view_mlflow_experiments():
    """Access MLflow to compare all model runs."""
    print("\n" + "="*70)
    print("METHOD 5: View MLflow Experiments")
    print("="*70)

    print("""
MLflow tracks every training run with full metrics and parameters.

1. Open MLflow UI:
   http://localhost:5001

2. Navigate to experiments:
   - pm25_24h_station_56
   - pm25_24h_station_57
   - pm25_24h_station_58
   - pm25_24h_station_59
   - pm25_24h_station_61

3. Each experiment shows:
   - All 5 model runs with MAE, RMSE, R²
   - Hyperparameters used
   - Training timestamps
   - Model artifacts

4. Compare runs side-by-side to see why one model was chosen over others.

Example: Access via Python API:
""")

    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5001")

        # List experiments
        experiments = mlflow.search_experiments()

        print("\nAvailable experiments:")
        for exp in experiments:
            if "pm25_24h" in exp.name:
                print(f"  - {exp.name} (ID: {exp.experiment_id})")

                # Get runs for this experiment
                runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], max_results=5)
                if not runs.empty:
                    print(f"    Recent runs: {len(runs)}")
                    print(f"    Best MAE: {runs['metrics.MAE'].min():.4f}")
    except Exception as e:
        print(f"\n(MLflow not accessible: {e})")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("HOW TO USE YOUR 5 MODELS")
    print("="*70)
    print("""
You have 2 concepts:

1. 5 MODEL TYPES (algorithms):
   - Linear, Ridge, Random Forest, XGBoost, LSTM
   - All trained during training pipeline
   - Best one deployed per station

2. 6 STATION MODELS (deployed):
   - One model per station (pm25, pm25_63...pm25_67)
   - Each is the best of the 5 types for that station
   - Different stations may use different algorithms
""")

    # Method 1: Triton API
    try:
        use_triton_api()
    except Exception as e:
        print(f"\nTriton example failed: {e}")

    # Method 2: FastAPI
    try:
        use_fastapi()
    except Exception as e:
        print(f"\nFastAPI example skipped: {e}")

    # Method 3: Model comparison info
    compare_model_types()

    # Method 4: Switching models
    switch_active_model_example()

    # Method 5: MLflow
    view_mlflow_experiments()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
Production use cases:
  ✓ Use Triton API for predictions (fastest)
  ✓ Each station already has its best model deployed
  ✓ Use FastAPI /predict endpoint for easier integration

Development/Research:
  ✓ View training results in results/forecast_24h_results.csv
  ✓ Compare models in MLflow UI (http://localhost:5001)
  ✓ Access temp models during training for comparison

You DON'T need to manually manage the 5 types - the pipeline does it!
""")


if __name__ == "__main__":
    main()

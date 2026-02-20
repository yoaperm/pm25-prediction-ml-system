"""
Prediction / Inference Module
=============================
Loads a trained model and makes predictions on new data.
Separated from training for deployment best practices.
"""

import json
import joblib
import numpy as np
import pandas as pd

from data_loader import load_config
from preprocessing import preprocess_pipeline
from feature_engineering import build_features, get_feature_columns


def load_model(model_path: str):
    """Load a trained model from disk."""
    model = joblib.load(model_path)
    print(f"Loaded model from {model_path}")
    return model


def load_feature_columns(path: str = "models/feature_columns.json") -> list[str]:
    """Load the list of feature columns used during training."""
    with open(path, "r") as f:
        return json.load(f)


def predict(
    model,
    input_df: pd.DataFrame,
    feature_cols: list[str],
    config: dict,
) -> pd.DataFrame:
    """
    Run inference on input data.

    Parameters
    ----------
    model : sklearn estimator
        Trained model.
    input_df : pd.DataFrame
        Raw input DataFrame with columns ['date', 'pm25'].
    feature_cols : list[str]
        Feature columns expected by the model.
    config : dict
        Configuration dictionary.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['date', 'actual', 'predicted'].
    """
    # Preprocess
    clean_df = preprocess_pipeline(input_df)

    # Feature engineering
    feat_df = build_features(
        clean_df,
        lag_days=config["features"]["lag_days"],
        rolling_windows=config["features"]["rolling_windows"],
    )

    # Predict
    X = feat_df[feature_cols].values
    predictions = model.predict(X)

    result = pd.DataFrame({
        "date": feat_df["date"].values,
        "actual": feat_df["pm25"].values,
        "predicted": np.round(predictions, 2),
    })

    return result


if __name__ == "__main__":
    config = load_config()

    # Load best model (default: XGBoost)
    model_path = f"{config['output']['models_dir']}/xgboost.joblib"
    model = load_model(model_path)

    # Load feature columns
    feature_cols = load_feature_columns(
        f"{config['output']['models_dir']}/feature_columns.json"
    )

    # Load test data for demo
    from data_loader import load_station_data

    test_df = load_station_data(
        file_path=config["data"]["test_file"],
        sheet_name=config["data"]["test_sheet"],
        station_id=config["station"]["id"],
    )

    # Run prediction
    results = predict(model, test_df, feature_cols, config)
    print(f"\nPrediction results ({len(results)} days):")
    print(results.head(10))
    print(f"\nMean Absolute Error: {np.abs(results['actual'] - results['predicted']).mean():.2f} µg/m³")

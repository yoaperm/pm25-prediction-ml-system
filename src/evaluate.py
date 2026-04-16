"""
Evaluation Module
=================
Computes regression metrics for PM2.5 predictions.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute evaluation metrics for regression.

    Parameters
    ----------
    y_true : array-like
        Ground truth PM2.5 values.
    y_pred : array-like
        Predicted PM2.5 values.

    Returns
    -------
    dict
        Dictionary with MAE, RMSE, R2 scores.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
    }


def print_metrics(model_name: str, metrics: dict):
    """Print evaluation metrics in a formatted style."""
    print(f"\n  {model_name} Results:")
    print(f"    RMSE: {metrics['RMSE']:.4f}  (primary metric)")
    print(f"    MAE:  {metrics['MAE']:.4f}")
    print(f"    R²:   {metrics['R2']:.4f}")


def compare_models(results_path: str = "results/experiment_results.csv"):
    """
    Load and display model comparison table.

    Parameters
    ----------
    results_path : str
        Path to the experiment results CSV.
    """
    df = pd.read_csv(results_path)
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(df[["model", "RMSE", "MAE", "R2"]].to_string(index=False))

    # Find best model by RMSE
    best_idx = df["RMSE"].idxmin()
    best_model = df.loc[best_idx, "model"]
    print(f"\nBest model (lowest RMSE): {best_model} (RMSE={df.loc[best_idx, 'RMSE']:.4f}, MAE={df.loc[best_idx, 'MAE']:.4f})")

    return df


if __name__ == "__main__":
    compare_models()

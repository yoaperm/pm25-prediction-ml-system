"""
LSTM Model Module
=================
LSTM model with RandomizedSearchCV tuning via skorch (PyTorch) wrapper.
"""

import tempfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import mlflow
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from skorch import NeuralNetRegressor


class LSTMNet(nn.Module):
    """PyTorch LSTM network for PM2.5 regression."""

    def __init__(self, units=64, dropout=0.2):
        super().__init__()
        self.lstm    = nn.LSTM(input_size=17, hidden_size=units, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc1     = nn.Linear(units, 32)
        self.relu    = nn.ReLU()
        self.fc2     = nn.Linear(32, 1)

    def forward(self, x):
        # x: (batch, timesteps=1, features=17)
        out, _ = self.lstm(x)
        out = out[:, -1, :]           # last timestep → (batch, units)
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        return self.fc2(out).squeeze(-1)   # (batch,)


def create_lstm_model(units=64, dropout=0.2, learning_rate=0.001):
    """Factory function — kept for backward compatibility with export_onnx."""
    return LSTMNet(units=units, dropout=dropout)


def train_lstm_with_tuning(X_train, y_train, param_grid, random_state=42):
    """
    Mirrors train_with_tuning() signature. Reshapes X internally to (samples, 1, 17).

    Parameters
    ----------
    X_train : array-like, shape (n_samples, 17)
    y_train : array-like
    param_grid : dict — keys: units, dropout, learning_rate, epochs, batch_size
    random_state : int

    Returns
    -------
    tuple
        (best_estimator, best_params)
    """
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    # Use MPS (Apple Silicon GPU) if available, else CPU
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"  LSTM using device: {device}")

    # Reshape to (samples, timesteps=1, features=17)
    X_3d    = X_train.reshape(X_train.shape[0], 1, X_train.shape[1]).astype(np.float32)
    y_float = y_train.astype(np.float32)

    # Map config param_grid keys → skorch parameter names
    key_map = {
        "units":         "module__units",
        "dropout":       "module__dropout",
        "learning_rate": "optimizer__lr",
        "epochs":        "max_epochs",
        "batch_size":    "batch_size",
    }
    skorch_grid = {key_map.get(k, k): v for k, v in param_grid.items()}

    net = NeuralNetRegressor(
        module=LSTMNet,
        criterion=nn.L1Loss,
        optimizer=torch.optim.Adam,
        max_epochs=50,
        batch_size=16,
        train_split=None,
        device=device,
        verbose=0,
    )

    tscv = TimeSeriesSplit(n_splits=3)
    # n_jobs=1 mandatory — PyTorch is not fork-safe
    # n_iter=6 tries 6 random combos instead of exhaustive grid (much faster on CPU)
    search = RandomizedSearchCV(
        net,
        skorch_grid,
        n_iter=6,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
        refit=True,
        random_state=random_state,
    )
    search.fit(X_3d, y_float)

    # Log full CV results as MLflow artifact (mirrors train_with_tuning behaviour)
    cv_df = pd.DataFrame(search.cv_results_)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", prefix="lstm_cv_", delete=False) as f:
        cv_df.to_csv(f.name, index=False)
        mlflow.log_artifact(f.name, artifact_path="cv_results")

    print(f"  Best params: {search.best_params_}")
    print(f"  Best CV MAE: {-search.best_score_:.4f}")

    return search.best_estimator_, search.best_params_

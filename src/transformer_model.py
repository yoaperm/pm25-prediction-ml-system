"""
Tabular Transformer model for PM2.5 regression.

The model treats each engineered feature as one token and keeps the public
inference contract as a 2-D matrix: (batch, n_features) -> (batch, 1).
That matches the existing ONNX prediction path used by the Airflow DAGs.
"""

from __future__ import annotations

import copy
import inspect
import math
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_TRANSFORMER_PARAMS = {
    "d_model": 32,
    "nhead": 4,
    "num_layers": 2,
    "dim_feedforward": 128,
    "dropout": 0.1,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "epochs": 30,
    "batch_size": 512,
    "patience": 5,
    "train_stride": 4,
}


def _scalar_param(params: dict[str, Any], key: str, default: Any) -> Any:
    value = params.get(key, default)
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        return value[0]
    return value


def _normalise_params(params: dict[str, Any] | None) -> dict[str, Any]:
    params = params or {}
    cfg = {
        key: _scalar_param(params, key, default)
        for key, default in DEFAULT_TRANSFORMER_PARAMS.items()
    }
    cfg["d_model"] = int(cfg["d_model"])
    cfg["nhead"] = int(cfg["nhead"])
    cfg["num_layers"] = int(cfg["num_layers"])
    cfg["dim_feedforward"] = int(cfg["dim_feedforward"])
    cfg["epochs"] = int(cfg["epochs"])
    cfg["batch_size"] = int(cfg["batch_size"])
    cfg["patience"] = int(cfg["patience"])
    cfg["train_stride"] = max(1, int(cfg["train_stride"]))
    cfg["dropout"] = float(cfg["dropout"])
    cfg["learning_rate"] = float(cfg["learning_rate"])
    cfg["weight_decay"] = float(cfg["weight_decay"])
    return cfg


def _resolve_device(preferred: str | None = None) -> str:
    device = preferred or os.environ.get("PYTORCH_DEVICE")
    if device:
        if device == "mps" and not torch.backends.mps.is_available():
            return "cpu"
        return device
    return "cpu"


class TransformerBlock(nn.Module):
    """Small encoder block with ONNX-friendly self-attention operations."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.attn_out = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, d_model = x.shape

        qkv = self.qkv(self.norm1(x))
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.reshape(batch_size, token_count, self.nhead, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, token_count, self.nhead, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, token_count, self.nhead, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        context = torch.matmul(weights, v)
        context = context.transpose(1, 2).reshape(batch_size, token_count, d_model)

        x = x + self.residual_dropout(self.attn_out(context))
        x = x + self.residual_dropout(self.ffn(self.norm2(x)))
        return x


class TabularTransformerRegressor(nn.Module):
    """Transformer regressor over engineered PM2.5 feature tokens."""

    def __init__(
        self,
        n_features: int,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        if n_features < 1:
            raise ValueError("n_features must be positive")

        self.n_features = n_features
        self.value_projection = nn.Linear(1, d_model)
        self.feature_embedding = nn.Parameter(torch.empty(1, n_features, d_model))
        nn.init.normal_(self.feature_embedding, mean=0.0, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.value_projection(x.unsqueeze(-1)) + self.feature_embedding
        for block in self.blocks:
            tokens = block(tokens)
        pooled = self.norm(tokens).mean(dim=1)
        return self.head(pooled)


def train_transformer_regressor(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    params: dict[str, Any] | None = None,
    random_state: int = 42,
) -> tuple[TabularTransformerRegressor, dict[str, Any]]:
    """Train a tabular Transformer with validation-based early stopping."""
    cfg = _normalise_params(params)
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    if X_train.ndim != 2:
        raise ValueError("X_train must be a 2-D array")

    has_val = X_val is not None and y_val is not None and len(X_val) > 0
    if has_val:
        X_val = np.asarray(X_val, dtype=np.float32)
        y_val = np.asarray(y_val, dtype=np.float32)

    stride = cfg["train_stride"]
    X_fit = X_train[::stride]
    y_fit = y_train[::stride]

    device = _resolve_device(params.get("device") if params else None)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_fit), torch.from_numpy(y_fit)),
        batch_size=cfg["batch_size"],
        shuffle=False,
    )

    model = TabularTransformerRegressor(
        n_features=X_train.shape[1],
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    criterion = nn.L1Loss()

    X_val_tensor = torch.from_numpy(X_val).to(device) if has_val else None
    y_val_tensor = torch.from_numpy(y_val).to(device) if has_val else None

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    patience_left = cfg["patience"]
    epochs_done = 0

    for epoch in range(cfg["epochs"]):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb).squeeze(-1), yb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            if has_val:
                val_loss = criterion(model(X_val_tensor).squeeze(-1), y_val_tensor).item()
            else:
                val_loss = float(np.mean(train_losses)) if train_losses else float("inf")

        epochs_done = epoch + 1
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg["patience"]
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    model.cpu().eval()

    info = {
        **cfg,
        "device": device,
        "epochs_done": epochs_done,
        "best_val_loss": best_val_loss,
    }
    return model, info


def predict_transformer(
    model: TabularTransformerRegressor,
    X,
    batch_size: int = 2048,
    device: str = "cpu",
) -> np.ndarray:
    """Run batched Transformer inference and return a flat numpy array."""
    X = np.asarray(X, dtype=np.float32)
    loader = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=batch_size, shuffle=False)
    model = model.to(device)
    model.eval()

    outputs = []
    with torch.no_grad():
        for (xb,) in loader:
            pred = model(xb.to(device)).detach().cpu().numpy().reshape(-1)
            outputs.append(pred)
    model.cpu()
    return np.concatenate(outputs) if outputs else np.array([], dtype=np.float32)


def export_transformer_onnx(
    model: TabularTransformerRegressor,
    output_path: str,
    n_features: int,
    input_name: str = "float_input",
    output_name: str = "variable",
    opset_version: int = 17,
) -> str:
    """Export a trained Transformer to ONNX with a 2-D input contract."""
    model.cpu().eval()
    dummy = torch.zeros(1, n_features, dtype=torch.float32)
    export_kwargs = {
        "input_names": [input_name],
        "output_names": [output_name],
        "dynamic_axes": {input_name: {0: "batch"}, output_name: {0: "batch"}},
        "opset_version": opset_version,
        "do_constant_folding": True,
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    torch.onnx.export(
        model,
        dummy,
        output_path,
        **export_kwargs,
    )
    print(f"  Saved: {output_path}")
    return output_path

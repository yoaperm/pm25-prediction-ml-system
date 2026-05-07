import os
import sys

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

import torch
import onnxruntime as rt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from transformer_model import (
    TabularTransformerRegressor,
    export_transformer_onnx,
    predict_transformer,
    train_transformer_regressor,
)


def test_tabular_transformer_forward_shape():
    model = TabularTransformerRegressor(
        n_features=6,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
    )

    out = model(torch.randn(3, 6))

    assert out.shape == (3, 1)


def test_transformer_train_export_onnx_round_trip(tmp_path):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 6)).astype(np.float32)
    y = (X[:, 0] * 0.7 - X[:, 1] * 0.2 + 3.0).astype(np.float32)

    model, info = train_transformer_regressor(
        X[:64],
        y[:64],
        X[64:],
        y[64:],
        params={
            "d_model": 16,
            "nhead": 4,
            "num_layers": 1,
            "dim_feedforward": 32,
            "epochs": 2,
            "batch_size": 16,
            "patience": 1,
            "train_stride": 1,
            "device": "cpu",
        },
        random_state=42,
    )
    preds = predict_transformer(model, X[64:], batch_size=8)

    assert preds.shape == (16,)
    assert info["epochs_done"] >= 1

    onnx_path = tmp_path / "transformer.onnx"
    export_transformer_onnx(model, str(onnx_path), n_features=6)

    sess = rt.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    output_name = sess.get_outputs()[0].name
    input_name = sess.get_inputs()[0].name
    onnx_preds = sess.run([output_name], {input_name: X[64:].astype(np.float32)})[0]

    assert onnx_preds.shape == (16, 1)

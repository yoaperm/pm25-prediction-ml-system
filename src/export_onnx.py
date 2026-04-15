"""
ONNX Export Script
==================
Converts trained .joblib models to ONNX format.

Usage:
    PYTHONPATH=src python src/export_onnx.py              # convert all
    PYTHONPATH=src python src/export_onnx.py --model lstm # single model
"""
import argparse
import os
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType as OmtFloatTensorType
from data_loader import load_config

N_FEATURES = 17

SKLEARN_MODELS = {
    "baseline_linear_regression": "baseline_linear_regression.joblib",
    "ridge_regression":           "ridge_regression.joblib",
    "random_forest":              "random_forest.joblib",
}

def export_sklearn(model, name, onnx_dir):
    initial_type = [("float_input", FloatTensorType([None, N_FEATURES]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=17)
    path = os.path.join(onnx_dir, f"{name}.onnx")
    with open(path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"  Saved: {path}")

def export_xgboost(model, onnx_dir):
    initial_type = [("float_input", OmtFloatTensorType([None, N_FEATURES]))]
    onnx_model = convert_xgboost(model, initial_types=initial_type)
    path = os.path.join(onnx_dir, "xgboost.onnx")
    with open(path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"  Saved: {path}")

def export_lstm(model, onnx_dir):
    import torch
    pytorch_model = model.module_   # unwrap skorch wrapper
    pytorch_model.eval()
    dummy = torch.zeros(1, 1, N_FEATURES)
    path = os.path.join(onnx_dir, "lstm.onnx")
    torch.onnx.export(
        pytorch_model,
        dummy,
        path,
        input_names=["lstm_input"],
        output_names=["variable"],
        dynamic_axes={"lstm_input": {0: "batch"}, "variable": {0: "batch"}},
        opset_version=17,
    )
    print(f"  Saved: {path}")

def export_all(config, only=None):
    models_dir = config["output"]["models_dir"]
    onnx_dir   = config["output"].get("onnx_dir", os.path.join(models_dir, "onnx"))
    os.makedirs(onnx_dir, exist_ok=True)

    for name, filename in SKLEARN_MODELS.items():
        if only and name != only:
            continue
        src = os.path.join(models_dir, filename)
        if not os.path.exists(src):
            print(f"  Skipping {name}: {src} not found")
            continue
        print(f"Converting {name}...")
        export_sklearn(joblib.load(src), name, onnx_dir)

    if only is None or only == "xgboost":
        src = os.path.join(models_dir, "xgboost.joblib")
        if not os.path.exists(src):
            print("  Skipping xgboost: not found")
        else:
            print("Converting xgboost...")
            export_xgboost(joblib.load(src), onnx_dir)

    if only is None or only == "lstm":
        src = os.path.join(models_dir, "lstm.joblib")
        if not os.path.exists(src):
            print("  Skipping lstm: not found")
        else:
            print("Converting lstm...")
            export_lstm(joblib.load(src), onnx_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                        help="Model name to export (default: all). "
                             "Choices: baseline_linear_regression, ridge_regression, "
                             "random_forest, xgboost, lstm")
    args = parser.parse_args()
    config = load_config()
    export_all(config, only=args.model)
    print("\nDone. ONNX models saved to:", config["output"].get("onnx_dir", "models/onnx"))

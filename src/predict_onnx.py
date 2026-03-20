"""
ONNX Inference Script
=====================
Run PM2.5 predictions via ONNX Runtime (no TF/sklearn required at inference).

Usage:
    PYTHONPATH=src python src/predict_onnx.py [--model NAME] [--input FILE] [--output FILE]
"""
import argparse, json, os
import numpy as np
import pandas as pd
import onnxruntime as rt

from data_loader import load_config, load_station_data
from preprocessing import preprocess_pipeline
from feature_engineering import build_features

SKLEARN_MODELS = ["baseline_linear_regression", "ridge_regression", "random_forest", "xgboost"]
ALL_MODELS     = SKLEARN_MODELS + ["lstm"]


def _load_feature_cols(models_dir):
    with open(os.path.join(models_dir, "feature_columns.json")) as f:
        return json.load(f)


def _run_session(session, X: np.ndarray, is_lstm: bool) -> np.ndarray:
    input_name  = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    X_in = X.reshape(X.shape[0], 1, X.shape[1]) if is_lstm else X
    result = session.run([output_name], {input_name: X_in.astype(np.float32)})
    return result[0].flatten()


def predict_onnx(model_name: str, input_file: str, config: dict) -> pd.DataFrame:
    models_dir = config["output"]["models_dir"]
    onnx_dir   = config["output"].get("onnx_dir", os.path.join(models_dir, "onnx"))
    onnx_path  = os.path.join(onnx_dir, f"{model_name}.onnx")

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(
            f"ONNX model not found: {onnx_path}\n"
            "Run: PYTHONPATH=src python src/export_onnx.py first."
        )

    station = config["station"]
    raw_df   = load_station_data(input_file, config["data"]["test_sheet"], station["id"])
    clean_df = preprocess_pipeline(raw_df)
    feat_df  = build_features(
        clean_df,
        config["features"]["lag_days"],
        config["features"]["rolling_windows"],
    )

    feature_cols = _load_feature_cols(models_dir)
    X = feat_df[feature_cols].values

    session     = rt.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    predictions = _run_session(session, X, is_lstm=(model_name == "lstm"))

    return pd.DataFrame({
        "date":      feat_df["date"].values,
        "actual":    feat_df["pm25"].values,
        "predicted": predictions.round(2),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="random_forest",
                        help="Model name or 'all'. Default: random_forest")
    parser.add_argument("--input",  default=None,
                        help="Input Excel file (default: config test_file)")
    parser.add_argument("--output", default=None,
                        help="Save predictions CSV to this path (optional)")
    args   = parser.parse_args()
    config = load_config()
    input_file = args.input or config["data"]["test_file"]

    targets = ALL_MODELS if args.model == "all" else [args.model]

    all_results = []
    for model_name in targets:
        print(f"\n{'='*50}")
        print(f"ONNX Inference: {model_name}")
        print(f"{'='*50}")
        try:
            df = predict_onnx(model_name, input_file, config)
            df.insert(0, "model", model_name)
            print(df[["date", "actual", "predicted"]].head(10).to_string(index=False))
            print(f"  Total rows: {len(df)}")
            all_results.append(df)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")

    if args.output and all_results:
        out = pd.concat(all_results)
        out.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")

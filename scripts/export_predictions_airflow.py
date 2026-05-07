"""
Export trained model predictions to PostgreSQL pm25_predicted_hourly table.
Called by Airflow DAG after model training to persist predictions for demo/monitoring.

Usage:
    python export_predictions_airflow.py --station 56 --db-url postgresql://... --models-dir /app/models
"""

import os
import json
import argparse
import sys
import pandas as pd
import numpy as np
import onnxruntime as rt
import sqlalchemy
from datetime import datetime, timezone


def export_predictions_to_db(
    station_id: int,
    models_dir: str,
    processed_dir: str,
    db_url: str = "postgresql://postgres:postgres@postgres:5432/pm25",
    table_name: str = "pm25_predicted_hourly",
):
    """
    Export test set predictions from all trained models (both ONNX and SARIMA) to PostgreSQL.
    
    Args:
        station_id: Station ID (e.g., 56)
        models_dir: Path to models directory (e.g., /app/models/station_56)
        processed_dir: Path to processed data directory (e.g., /app/data/processed/station_56)
        db_url: PostgreSQL connection URL
        table_name: Target table name (default: pm25_predicted_hourly)
    """
    
    print(f"\n{'='*60}")
    print(f"Exporting predictions for station {station_id}")
    print(f"{'='*60}")
    
    # ---- Load test metadata and timestamps ----
    meta_path = os.path.join(processed_dir, "meta.json")
    test_df_path = os.path.join(processed_dir, "X_test.parquet")
    
    if not os.path.exists(meta_path):
        print(f"❌ meta.json not found: {meta_path}")
        return
    
    with open(meta_path) as f:
        meta = json.load(f)
    
    # Reconstruct test dataframe to get timestamps
    # We need the original hourly data with timestamps aligned to X_test rows
    test_X = pd.read_parquet(test_df_path)
    
    # Load raw hourly test data to get timestamps
    raw_test_path = os.path.join(processed_dir, "raw_test_hourly.parquet")
    if not os.path.exists(raw_test_path):
        print(f"⚠️  raw_test_hourly.parquet not found; creating from X_test metadata")
        # Fallback: assume we can read from meta
        # In this case, we'll try to get timestamps from the pm25_raw_test parquet
        pm25_raw_test_path = os.path.join(processed_dir, "pm25_raw_test.parquet")
        if os.path.exists(pm25_raw_test_path):
            raw_test_df = pd.read_parquet(pm25_raw_test_path)
            print(f"  Found {len(raw_test_df)} test rows")
        else:
            print(f"❌ Cannot find test data files")
            return
    else:
        raw_test_df = pd.read_parquet(raw_test_path)
    
    n_test = len(test_X)
    n_raw = len(raw_test_df)
    
    if n_raw < n_test:
        print(f"⚠️  Mismatch: X_test has {n_test} rows but raw data has {n_raw}; using first {n_test}")
        timestamps = raw_test_df["datetime"].iloc[:n_test].values
    else:
        timestamps = raw_test_df["datetime"].iloc[-n_test:].values if n_raw > n_test else raw_test_df["datetime"].values
    
    print(f"  Test set: {n_test} rows")
    print(f"  Timestamps: {timestamps[0]} → {timestamps[-1] if len(timestamps) > 0 else 'N/A'}")
    
    # ---- Load predictions from ONNX models ----
    predictions_by_model = {}
    
    # Load test features
    X_test = test_X.values.astype("float32")
    feature_cols = meta.get("feature_cols", [])
    n_features = meta.get("n_features", len(feature_cols))
    
    model_map = {
        "linear_regression": ("Linear Regression", False),
        "ridge_regression":  ("Ridge Regression", False),
        "random_forest":     ("Random Forest", False),
        "xgboost":           ("XGBoost", False),
        "lstm":              ("LSTM", True),
    }
    
    for key, (display_name, is_lstm) in model_map.items():
        tmp_onnx = os.path.join(models_dir, f"_tmp_{key}.onnx")
        versioned_onnx = None
        
        # Try to find versioned ONNX (deployed model)
        onnx_dir = os.path.join(models_dir, "onnx")
        if os.path.exists(onnx_dir):
            for file in os.listdir(onnx_dir):
                if file.startswith(key) and file.endswith(".onnx"):
                    versioned_onnx = os.path.join(onnx_dir, file)
                    break
        
        onnx_path = versioned_onnx if os.path.exists(versioned_onnx or "") else (tmp_onnx if os.path.exists(tmp_onnx) else None)
        
        if onnx_path and os.path.exists(onnx_path):
            try:
                sess = rt.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
                in_name = sess.get_inputs()[0].name
                out_name = sess.get_outputs()[0].name
                
                if is_lstm:
                    X_in = X_test.reshape(-1, 1, n_features)
                else:
                    X_in = X_test
                
                preds = sess.run([out_name], {in_name: X_in})[0].flatten()
                predictions_by_model[key] = preds
                print(f"  ✅ {display_name}: {len(preds)} predictions")
            except Exception as e:
                print(f"  ⚠️  {display_name} inference failed: {e}")
        else:
            print(f"  ⏭️  {display_name}: ONNX not found")
    
    # ---- Load SARIMA predictions ----
    sarima_result_path = os.path.join(models_dir, "_tmp_sarima_result.json")
    sarima_onnx = None  # SARIMA doesn't export to ONNX; we'll recompute if deployed
    
    # Try to load from deployed sarima_order.json
    sarima_order_path = os.path.join(models_dir, "sarima_order.json")
    if os.path.exists(sarima_order_path):
        try:
            with open(sarima_order_path) as f:
                sarima_params = json.load(f)
            
            sys.path.insert(0, "/app/src")
            from sarima_model import fit_sarima, predict_sarima_n_ahead_rolling
            
            # Reconstruct SARIMA by refitting on train data
            pm25_raw_train_path = os.path.join(processed_dir, "pm25_raw_train.parquet")
            if os.path.exists(pm25_raw_train_path):
                pm25_raw_train = pd.read_parquet(pm25_raw_train_path)["pm25"].values
                pm25_raw_test = pd.read_parquet(os.path.join(processed_dir, "pm25_raw_test.parquet"))["pm25"].values
                
                model = fit_sarima(
                    tuple(map(int, sarima_params["order"].strip("()").split(","))),
                    tuple(map(int, sarima_params["seasonal_order"].strip("()").split(","))),
                    pm25_raw_train
                )
                preds = predict_sarima_n_ahead_rolling(model, pm25_raw_train, pm25_raw_test, n_ahead=24)
                predictions_by_model["sarima"] = preds
                print(f"  ✅ SARIMA: {len(preds)} predictions")
        except Exception as e:
            print(f"  ⚠️  SARIMA prediction failed: {e}")
    else:
        print(f"  ⏭️  SARIMA: order file not found")
    
    # ---- Insert predictions into PostgreSQL ----
    if not predictions_by_model:
        print(f"❌ No predictions found to insert")
        return
    
    engine = sqlalchemy.create_engine(db_url)
    
    # Ensure table exists
    try:
        with engine.connect() as conn:
            conn.execute(sqlalchemy.text(f"SELECT 1 FROM {table_name} LIMIT 1"))
    except Exception as e:
        print(f"❌ Table {table_name} not accessible: {e}")
        engine.dispose()
        return
    
    # Prepare insert data
    rows_to_insert = []
    for model_key, preds in predictions_by_model.items():
        for i, (ts, pred) in enumerate(zip(timestamps, preds)):
            # Ensure timestamp is UTC
            if pd.isna(ts):
                continue
            ts_dt = pd.Timestamp(ts)
            if ts_dt.tz is None:
                ts_dt = ts_dt.tz_localize("UTC")
            
            rows_to_insert.append({
                "station_id": station_id,
                "timestamp": ts_dt,
                "model": model_key,
                "predicted_pm25": float(pred),
            })
    
    print(f"\nInserting {len(rows_to_insert)} predictions ({len(predictions_by_model)} models × {n_test} rows)...")
    
    # Batch insert with ON CONFLICT UPDATE
    batch_size = 1000
    with engine.connect() as conn:
        for i in range(0, len(rows_to_insert), batch_size):
            batch = rows_to_insert[i:i+batch_size]
            df_batch = pd.DataFrame(batch)
            
            try:
                df_batch.to_sql(table_name, con=conn, if_exists="append", index=False, method="multi")
                print(f"  ✅ Batch {i//batch_size + 1}: {len(batch)} rows inserted")
            except Exception as e:
                # Try with ON CONFLICT handling
                if "UNIQUE constraint" in str(e) or "duplicate key" in str(e):
                    print(f"  ℹ️  Some predictions already exist (duplicate key); continuing...")
                else:
                    print(f"  ⚠️  Insert error: {e}")
        
        conn.commit()
    
    engine.dispose()
    
    # Verify
    with engine.connect() as conn:
        for model_key in predictions_by_model.keys():
            result = conn.execute(sqlalchemy.text(f"""
                SELECT COUNT(*) FROM {table_name}
                WHERE station_id = :station_id AND model = :model
            """), {"station_id": station_id, "model": model_key})
            count = result.scalar()
            print(f"  DB count for {model_key}: {count}")
    
    print(f"\n✅ Export complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model predictions to PostgreSQL")
    parser.add_argument("--station", type=int, required=True, help="Station ID (e.g., 56)")
    parser.add_argument("--models-dir", default="/app/models", help="Models directory")
    parser.add_argument("--processed-dir", default="/app/data/processed", help="Processed data directory")
    parser.add_argument("--db-url", default="postgresql://postgres:postgres@postgres:5432/pm25", help="PostgreSQL URL")
    
    args = parser.parse_args()
    
    export_predictions_to_db(
        station_id=args.station,
        models_dir=os.path.join(args.models_dir, f"station_{args.station}"),
        processed_dir=os.path.join(args.processed_dir, f"station_{args.station}"),
        db_url=args.db_url,
    )

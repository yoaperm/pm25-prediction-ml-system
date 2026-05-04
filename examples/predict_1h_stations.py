#!/usr/bin/env python3
"""
Predict PM2.5 one hour ahead for Stations 56, 57, 58, 59, 61.

Uses Triton models named pm25_{station_id}_1h.
"""

from datetime import datetime, timedelta

import numpy as np
import tritonclient.http as httpclient


STATIONS = [56, 57, 58, 59, 61]


def build_sample_features():
    now = datetime.now()
    return np.array([[
        28.5, 29.1, 30.2, 32.5, 31.8, 35.0,
        28.5, 3.2, 29.1, 4.1, 30.2, 3.8,
        0.5, -2.3,
        (now + timedelta(hours=1)).hour,
        (now + timedelta(hours=1)).weekday(),
        (now + timedelta(hours=1)).month,
        (now + timedelta(hours=1)).timetuple().tm_yday,
        1 if (now + timedelta(hours=1)).weekday() >= 5 else 0,
    ]], dtype=np.float32)


def predict_single_station(client, station_id, features):
    model_name = f"pm25_{station_id}_1h"
    metadata = client.get_model_metadata(model_name)
    input_name = metadata["inputs"][0]["name"]
    output_name = metadata["outputs"][0]["name"]
    model_features = features.reshape(1, 1, features.shape[1]) if input_name == "lstm_input" else features
    input_tensor = httpclient.InferInput(input_name, model_features.shape, "FP32")
    input_tensor.set_data_from_numpy(model_features)
    output_tensor = httpclient.InferRequestedOutput(output_name)
    result = client.infer(model_name=model_name, inputs=[input_tensor], outputs=[output_tensor])
    return round(float(result.as_numpy(output_name).flatten()[0]), 2)


def main():
    client = httpclient.InferenceServerClient(url="localhost:8010")
    features = build_sample_features()
    forecast_time = datetime.now() + timedelta(hours=1)

    print(f"PM2.5 next-hour forecast for {forecast_time:%Y-%m-%d %H:%M}")
    for station_id in STATIONS:
        model_name = f"pm25_{station_id}_1h"
        try:
            if not client.is_model_ready(model_name):
                print(f"Station {station_id}: model not ready")
                continue
            prediction = predict_single_station(client, station_id, features)
            print(f"Station {station_id}: {prediction:.2f} µg/m³")
        except Exception as exc:
            print(f"Station {station_id}: error: {exc}")


if __name__ == "__main__":
    main()

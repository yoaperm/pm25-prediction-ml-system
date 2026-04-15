#!/bin/bash
#
# Publish all 5 station models to Triton
# Usage: bash scripts/publish_models_to_triton.sh
#

set -e

echo "======================================================================"
echo "Publishing Models to Triton for Stations 56, 57, 58, 59, 61"
echo "======================================================================"

STATIONS=(56 57 58 59 61)

for STATION_ID in "${STATIONS[@]}"; do
    echo ""
    echo ">>> Publishing Station $STATION_ID..."

    docker exec pm25-prediction-ml-system-airflow-scheduler-1 python3 << EOF
import sys
import os
import json
import shutil
sys.path.insert(0, '/app/src')

station_id = $STATION_ID
model_dir = f'/app/models/station_{station_id}_24h'
active_json = f'{model_dir}/active_model.json'
triton_repo = '/app/triton_model_repo'

# Read active model config
with open(active_json) as f:
    config = json.load(f)

onnx_file = config['onnx_file']
model_key = config['model_key']
is_lstm = config.get('is_lstm', False)
n_features = config.get('n_features', 19)

# Source ONNX file
src_onnx = f'{model_dir}/onnx/{onnx_file}'

if not os.path.exists(src_onnx):
    print(f'  ✗ ONNX file not found: {src_onnx}')
    sys.exit(1)

# Create Triton model directory
model_name = f'pm25_{station_id}'
version_dir = os.path.join(triton_repo, model_name, '1')
os.makedirs(version_dir, exist_ok=True)

# Copy ONNX file
dest_onnx = os.path.join(version_dir, 'model.onnx')
shutil.copy2(src_onnx, dest_onnx)
print(f'  ✓ Copied ONNX: {dest_onnx}')

# Create config.pbtxt
config_pbtxt = f'''name: "{model_name}"
backend: "onnxruntime"
max_batch_size: 32

input [
  {{
    name: "float_input"
    data_type: TYPE_FP32
    dims: [ {n_features} ]
  }}
]

output [
  {{
    name: "variable"
    data_type: TYPE_FP32
    dims: [ 1 ]
  }}
]

dynamic_batching {{ }}
'''

config_path = os.path.join(triton_repo, model_name, 'config.pbtxt')
with open(config_path, 'w') as f:
    f.write(config_pbtxt)

print(f'  ✓ Created config: {config_path}')
print(f'  ✓ Model: {model_key}, Features: {n_features}')

EOF

    if [ $? -eq 0 ]; then
        echo "  ✅ Station $STATION_ID published successfully"
    else
        echo "  ❌ Station $STATION_ID failed"
    fi
done

echo ""
echo "======================================================================"
echo "Waiting 35 seconds for Triton to reload models..."
echo "======================================================================"
sleep 35

echo ""
echo "Verifying deployed models:"
echo "======================================================================"

for STATION_ID in "${STATIONS[@]}"; do
    MODEL_NAME="pm25_$STATION_ID"

    STATUS=$(curl -s http://localhost:8010/v2/models/$MODEL_NAME | grep -q "\"name\": \"$MODEL_NAME\"" && echo "✅ READY" || echo "❌ NOT FOUND")

    echo "  $MODEL_NAME: $STATUS"
done

echo ""
echo "======================================================================"
echo "✓ Done! Models published to Triton"
echo "======================================================================"
echo ""
echo "Test with:"
echo "  python examples/predict_5_stations.py"
echo ""

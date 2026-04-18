#!/bin/bash
#
# Sync ONNX models from EC2 to local triton_model_repo
# Usage: bash scripts/sync_models_from_ec2.sh
#

set -e

EC2_USER="ubuntu"
EC2_HOST="43.210.161.101"
EC2_KEY="$HOME/Downloads/boss-admin-keypair.pem"
EC2_PROJECT="~/pm25-prediction-ml-system"
LOCAL_TRITON="$(dirname "$0")/../triton_model_repo"
STATIONS=(56 57 58 59 61)

echo "======================================================================"
echo "Syncing models from EC2 → local triton_model_repo"
echo "======================================================================"

for STATION_ID in "${STATIONS[@]}"; do
    echo ""
    echo ">>> Station $STATION_ID..."

    # Get active model filename from EC2
    ONNX_FILE=$(ssh -i "$EC2_KEY" "$EC2_USER@$EC2_HOST" \
        "cat $EC2_PROJECT/models/station_${STATION_ID}_24h/active_model.json" \
        | python3 -c "import sys,json; print(json.load(sys.stdin)['onnx_file'])")

    N_FEATURES=$(ssh -i "$EC2_KEY" "$EC2_USER@$EC2_HOST" \
        "cat $EC2_PROJECT/models/station_${STATION_ID}_24h/active_model.json" \
        | python3 -c "import sys,json; print(json.load(sys.stdin).get('n_features', 19))")

    echo "  ONNX: $ONNX_FILE  (features: $N_FEATURES)"

    # Create local triton directory
    mkdir -p "$LOCAL_TRITON/pm25_${STATION_ID}/1"

    # Copy ONNX from EC2
    scp -i "$EC2_KEY" \
        "$EC2_USER@$EC2_HOST:$EC2_PROJECT/models/station_${STATION_ID}_24h/onnx/$ONNX_FILE" \
        "$LOCAL_TRITON/pm25_${STATION_ID}/1/model.onnx"
    echo "  ✓ Copied model.onnx ($(du -h "$LOCAL_TRITON/pm25_${STATION_ID}/1/model.onnx" | cut -f1))"

    # Create config.pbtxt
    cat > "$LOCAL_TRITON/pm25_${STATION_ID}/config.pbtxt" << EOF
name: "pm25_${STATION_ID}"
backend: "onnxruntime"
max_batch_size: 32

input [
  {
    name: "float_input"
    data_type: TYPE_FP32
    dims: [ ${N_FEATURES} ]
  }
]

output [
  {
    name: "variable"
    data_type: TYPE_FP32
    dims: [ 1 ]
  }
]

dynamic_batching { }
EOF
    echo "  ✓ Created config.pbtxt"
done

echo ""
echo "======================================================================"
echo "Waiting 10 seconds for Triton to reload models..."
echo "======================================================================"
sleep 10

echo ""
echo "Verifying:"
for STATION_ID in "${STATIONS[@]}"; do
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8010/v2/models/pm25_${STATION_ID}/ready)
    if [ "$STATUS" = "200" ]; then
        echo "  pm25_${STATION_ID}: ✅ READY"
    else
        echo "  pm25_${STATION_ID}: ❌ NOT READY (http $STATUS)"
    fi
done

echo ""
echo "Done."

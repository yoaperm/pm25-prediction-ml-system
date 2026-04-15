"""
Triton Inference Server utilities.
Publishes the active ONNX model to a Triton model repository.

Repository layout produced:
    triton_model_repo/
      pm25/
        config.pbtxt
        1/
          model.onnx          (+ model.onnx.data if LSTM has external weights)
"""

import glob
import os
import shutil

import onnxruntime as rt

TRITON_MODEL_NAME = "pm25"

_ONNX_TO_TRITON_TYPE = {
    "tensor(float)":  "TYPE_FP32",
    "tensor(double)": "TYPE_FP64",
    "tensor(int32)":  "TYPE_INT32",
    "tensor(int64)":  "TYPE_INT64",
}


def publish_to_triton(onnx_path: str, triton_repo: str, is_lstm: bool) -> str:
    """
    Copy the ONNX model into the Triton model repository and write config.pbtxt.
    Triton polls the repository every 30 s (configured in docker-compose) and
    picks up the new model automatically.

    Parameters
    ----------
    onnx_path   : Path to the versioned .onnx file.
    triton_repo : Root of the Triton model repository (triton_model_repo/).
    is_lstm     : Whether the model expects 3-D input [batch, 1, features].

    Returns the destination model.onnx path.
    """
    version_dir = os.path.join(triton_repo, TRITON_MODEL_NAME, "1")
    os.makedirs(version_dir, exist_ok=True)

    # Copy model.onnx
    dest = os.path.join(version_dir, "model.onnx")
    shutil.copy2(onnx_path, dest)

    # Copy any external-data sidecar files (e.g. model.onnx.data from LSTM)
    for sidecar in glob.glob(f"{onnx_path}.data") + glob.glob(f"{onnx_path}_*.data"):
        shutil.copy2(sidecar, os.path.join(version_dir, os.path.basename(sidecar)))

    # Inspect I/O metadata from the model file
    sess    = rt.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp     = sess.get_inputs()[0]
    out     = sess.get_outputs()[0]
    inp_type = _ONNX_TO_TRITON_TYPE.get(inp.type, "TYPE_FP32")
    out_type = _ONNX_TO_TRITON_TYPE.get(out.type, "TYPE_FP32")

    # Drop the batch dim (index 0); Triton manages it via max_batch_size.
    def _dims_str(shape):
        parts = []
        for d in shape[1:]:
            parts.append(str(d) if (d is not None and d != -1) else "-1")
        return ", ".join(parts) if parts else "1"

    inp_dims = _dims_str(inp.shape)
    out_dims = _dims_str(out.shape)

    config = f"""name: "{TRITON_MODEL_NAME}"
backend: "onnxruntime"
max_batch_size: 32

input [
  {{
    name: "{inp.name}"
    data_type: {inp_type}
    dims: [ {inp_dims} ]
  }}
]

output [
  {{
    name: "{out.name}"
    data_type: {out_type}
    dims: [ {out_dims} ]
  }}
]

dynamic_batching {{ }}
"""

    config_path = os.path.join(triton_repo, TRITON_MODEL_NAME, "config.pbtxt")
    with open(config_path, "w") as f:
        f.write(config)

    print(f"  Triton repo: {dest}")
    print(f"  Config:      {config_path}")
    return dest

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PREFIX="${SCRIPT_DIR}/.conda"
ENV_FILE="${SCRIPT_DIR}/environment.yml"

if [[ -d "${ENV_PREFIX}" ]]; then
  conda env update --prefix "${ENV_PREFIX}" --file "${ENV_FILE}" --prune
else
  conda env create --prefix "${ENV_PREFIX}" --file "${ENV_FILE}"
fi

"${ENV_PREFIX}/bin/python" - <<'PY'
import json
import shutil
import sys

try:
    import torch
except Exception as exc:  # pragma: no cover - exercised from the shell only
    raise SystemExit(f"Failed to import torch from the world_model_agcrn_v1 environment: {exc}")

cuda_backend = getattr(torch.backends, "cuda", None)
cuda_built = bool(cuda_backend.is_built()) if cuda_backend is not None else False
cuda_available = bool(torch.cuda.is_available())
device_count = int(torch.cuda.device_count()) if cuda_available else 0
summary = {
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
    "cuda_built": cuda_built,
    "cuda_available": cuda_available,
    "device_count": device_count,
}
if cuda_available and device_count > 0:
    summary["devices"] = [torch.cuda.get_device_name(index) for index in range(device_count)]
print("CUDA probe:", json.dumps(summary, ensure_ascii=False))

if not cuda_built:
    raise SystemExit(
        "world_model_agcrn_v1/.conda installed a CPU-only torch build. "
        "Recreate the environment after fixing the torch dependency resolution."
    )

if shutil.which("nvidia-smi") is not None and not cuda_available:
    raise SystemExit(
        "Torch was installed with CUDA support, but no CUDA device is visible in this environment. "
        "Check the NVIDIA driver, CUDA_VISIBLE_DEVICES, and container/runtime GPU exposure."
    )
PY

echo "world_model_agcrn_v1 environment is ready at ${ENV_PREFIX}"
echo "Run the experiment with:"
echo "  ${ENV_PREFIX}/bin/python ${SCRIPT_DIR}/run_world_model_agcrn_v1.py"

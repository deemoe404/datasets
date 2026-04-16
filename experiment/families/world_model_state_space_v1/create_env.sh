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

echo "world_model_state_space_v1 environment is ready at ${ENV_PREFIX}"
echo "Run the experiment with:"
echo "  ${ENV_PREFIX}/bin/python ${SCRIPT_DIR}/run_world_model_state_space_v1.py"

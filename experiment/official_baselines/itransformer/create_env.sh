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

echo "iTransformer official baseline wrapper environment is ready at ${ENV_PREFIX}"

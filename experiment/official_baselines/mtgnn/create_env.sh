#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PREFIX="${SCRIPT_DIR}/.conda"

if [[ ! -d "${ENV_PREFIX}" ]]; then
  conda env create --prefix "${ENV_PREFIX}" --file "${SCRIPT_DIR}/environment.yml"
else
  conda env update --prefix "${ENV_PREFIX}" --file "${SCRIPT_DIR}/environment.yml" --prune
fi

echo "MTGNN official wrapper environment is ready at ${ENV_PREFIX}"

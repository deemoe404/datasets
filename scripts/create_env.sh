#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_PREFIX="${REPO_ROOT}/.conda"
ENV_FILE="${SCRIPT_DIR}/environment.yml"
CONDA_BIN="${CONDA_BIN:-$(command -v conda || true)}"

if [[ -z "${CONDA_BIN}" ]]; then
  printf '%s\n' 'conda executable not found. Install conda or set CONDA_BIN.' >&2
  exit 1
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  printf '%s\n' "Environment definition not found: ${ENV_FILE}" >&2
  exit 1
fi

if [[ -d "${ENV_PREFIX}" ]]; then
  "${CONDA_BIN}" env update --prefix "${ENV_PREFIX}" --file "${ENV_FILE}" --prune
else
  "${CONDA_BIN}" env create --prefix "${ENV_PREFIX}" --file "${ENV_FILE}"
fi

"${ENV_PREFIX}/bin/python" -m pip install --upgrade pip
(
  cd "${REPO_ROOT}"
  "${ENV_PREFIX}/bin/python" -m pip install --upgrade --editable ".[test]"
)

echo "Dataset processing environment is ready at ${ENV_PREFIX}"
echo "Run cache rebuilds with:"
echo "  ${REPO_ROOT}/scripts/rebuild_cache.sh --check"
echo "Run tests with:"
echo "  ${ENV_PREFIX}/bin/python -m pytest"

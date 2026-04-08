#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.conda/bin/python}"
PYTHONPATH_ROOT="$REPO_ROOT/src"
PASSTHROUGH_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      if [[ $# -lt 2 ]]; then
        printf '%s\n' '--python requires a path' >&2
        exit 2
      fi
      PYTHON_BIN="$2"
      shift 2
      ;;
    *)
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ! -x "$PYTHON_BIN" ]]; then
  if [[ "$PYTHON_BIN" == "$REPO_ROOT/.conda/bin/python" ]]; then
    printf '%s\n' "Python interpreter not found or not executable: $PYTHON_BIN. Run ./create_env.sh or pass --python <path>." >&2
  else
    printf '%s\n' "Python interpreter not found or not executable: $PYTHON_BIN" >&2
  fi
  exit 1
fi

export PYTHONPATH="$PYTHONPATH_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO_ROOT"

exec "$PYTHON_BIN" -m wind_datasets.rebuild_cache "${PASSTHROUGH_ARGS[@]}"

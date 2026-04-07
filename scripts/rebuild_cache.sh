#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.conda/bin/python}"
CACHE_ROOT="${CACHE_ROOT:-$REPO_ROOT/cache}"
PYTHONPATH_ROOT="$REPO_ROOT/src"

SUPPORTED_DATASETS=(
  "kelmarsh"
  "penmanshiel"
  "hill_of_towie"
  "sdwpf_kddcup"
)

usage() {
  cat <<'EOF'
Usage:
  ./scripts/rebuild_cache.sh [--clean] [--cache-root PATH] [--python PATH] [dataset ...]

Rebuilds the standard cache layers for one or more datasets.

Datasets:
  kelmarsh
  penmanshiel
  hill_of_towie
  sdwpf_kddcup
  all

Standard rebuild targets:
  - manifest
  - silver
  - gold_base/default/farm/default
  - gold_base/default/turbine/default
  - tasks/default/farm/next_6h_from_24h
  - tasks/default/turbine/next_6h_from_24h
  - tasks/default/turbine/next_6h_from_24h_stride_6h

Examples:
  ./scripts/rebuild_cache.sh
  ./scripts/rebuild_cache.sh hill_of_towie
  ./scripts/rebuild_cache.sh --clean hill_of_towie
  ./scripts/rebuild_cache.sh kelmarsh penmanshiel
  ./scripts/rebuild_cache.sh --cache-root /tmp/wind-cache sdwpf_kddcup
EOF
}

normalize_dataset() {
  local dataset="$1"
  case "$dataset" in
    all)
      printf '%s\n' "${SUPPORTED_DATASETS[@]}"
      ;;
    kelmarsh|penmanshiel|hill_of_towie|sdwpf_kddcup)
      printf '%s\n' "$dataset"
      ;;
    *)
      printf 'Unsupported dataset: %s\n' "$dataset" >&2
      exit 2
      ;;
  esac
}

dedupe_datasets() {
  local -a input=("$@")
  local -a output=()
  local item
  local existing
  local seen
  for item in "${input[@]}"; do
    seen=0
    for existing in "${output[@]}"; do
      if [[ "$existing" == "$item" ]]; then
        seen=1
        break
      fi
    done
    if [[ "$seen" -eq 0 ]]; then
      output+=("$item")
    fi
  done
  printf '%s\n' "${output[@]}"
}

CLEAN=0
SHOW_HELP=0
DATASET_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean)
      CLEAN=1
      shift
      ;;
    --cache-root)
      if [[ $# -lt 2 ]]; then
        printf '--cache-root requires a path\n' >&2
        exit 2
      fi
      CACHE_ROOT="$2"
      shift 2
      ;;
    --python)
      if [[ $# -lt 2 ]]; then
        printf '--python requires a path\n' >&2
        exit 2
      fi
      PYTHON_BIN="$2"
      shift 2
      ;;
    -h|--help)
      SHOW_HELP=1
      shift
      ;;
    *)
      DATASET_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "$SHOW_HELP" -eq 1 ]]; then
  usage
  exit 0
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  printf 'Python interpreter not found or not executable: %s\n' "$PYTHON_BIN" >&2
  exit 1
fi

if [[ "${#DATASET_ARGS[@]}" -eq 0 ]]; then
  DATASET_ARGS=("all")
fi

EXPANDED_DATASETS=()
for dataset in "${DATASET_ARGS[@]}"; do
  while IFS= read -r expanded; do
    EXPANDED_DATASETS+=("$expanded")
  done < <(normalize_dataset "$dataset")
done

TARGET_DATASETS=()
while IFS= read -r deduped; do
  TARGET_DATASETS+=("$deduped")
done < <(dedupe_datasets "${EXPANDED_DATASETS[@]}")

mkdir -p "$CACHE_ROOT"

if [[ "$CLEAN" -eq 1 ]]; then
  for dataset in "${TARGET_DATASETS[@]}"; do
    printf '[clean] removing %s\n' "$CACHE_ROOT/$dataset"
    rm -rf "$CACHE_ROOT/$dataset"
  done
fi

DATASET_CSV="$(printf '%s,' "${TARGET_DATASETS[@]}")"
DATASET_CSV="${DATASET_CSV%,}"

export PYTHONPATH="$PYTHONPATH_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WIND_DATASETS_REBUILD_CACHE_ROOT="$CACHE_ROOT"
export WIND_DATASETS_REBUILD_DATASETS="$DATASET_CSV"

cd "$REPO_ROOT"

"$PYTHON_BIN" -u - <<'PY'
import os
from pathlib import Path

from wind_datasets import build_gold_base, build_manifest, build_silver, build_task_cache
from wind_datasets.models import TaskSpec


def log(dataset: str, stage: str) -> None:
    print(f"[rebuild] dataset={dataset} stage={stage}", flush=True)


cache_root = Path(os.environ["WIND_DATASETS_REBUILD_CACHE_ROOT"])
datasets = [item for item in os.environ["WIND_DATASETS_REBUILD_DATASETS"].split(",") if item]

farm_task = TaskSpec.next_6h_from_24h(granularity="farm")
turbine_task = TaskSpec.next_6h_from_24h(granularity="turbine")
turbine_stride_task = TaskSpec(
    task_id="next_6h_from_24h_stride_6h",
    history_duration="24h",
    forecast_duration="6h",
    stride_duration="6h",
    granularity="turbine",
)

for dataset in datasets:
    log(dataset, "manifest")
    print(build_manifest(dataset, cache_root=cache_root), flush=True)

    log(dataset, "silver")
    print(build_silver(dataset, cache_root=cache_root), flush=True)

    log(dataset, "gold_base_farm")
    print(build_gold_base(dataset, cache_root=cache_root), flush=True)

    log(dataset, "gold_base_turbine")
    print(build_gold_base(dataset, cache_root=cache_root, layout="turbine"), flush=True)

    log(dataset, "task_farm_next_6h_from_24h")
    print(build_task_cache(dataset, farm_task, cache_root=cache_root), flush=True)

    log(dataset, "task_turbine_next_6h_from_24h")
    print(build_task_cache(dataset, turbine_task, cache_root=cache_root), flush=True)

    log(dataset, "task_turbine_next_6h_from_24h_stride_6h")
    print(build_task_cache(dataset, turbine_stride_task, cache_root=cache_root), flush=True)
PY

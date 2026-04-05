# `src` Package Notes

This directory contains the Python package used to build, cache, profile, and
load the supported wind power datasets.

The package name is `wind_datasets`.

## Scope

This package is the data layer for the repository. It is responsible for:

- registering the supported datasets
- building disposable cache artifacts under `./cache`
- normalizing each dataset into a task-independent `gold_base`
- building task-specific window indexes
- exposing a small public API for experiments and tests

It is not a training framework.

## Supported datasets

Only the following datasets are registered:

- `kelmarsh`
- `penmanshiel`
- `hill_of_towie`
- `sdwpf_full`

The dataset registry lives in [wind_datasets/registry.py](/Users/sam/Developer/Code/Wind%20Power%20Forecasting/datasets/src/wind_datasets/registry.py).

## Public API

The main import surface is [wind_datasets/__init__.py](/Users/sam/Developer/Code/Wind%20Power%20Forecasting/datasets/src/wind_datasets/__init__.py).

Typical entry points:

- `build_manifest(dataset_id)`
- `build_silver(dataset_id)`
- `build_gold_base(dataset_id, quality_profile=None)`
- `build_task_cache(dataset_id, task_spec, quality_profile=None)`
- `load_series(dataset_id, quality_profile=None)`
- `load_window_index(dataset_id, task_spec, quality_profile=None)`
- `profile_dataset(dataset_id, quality_profile=None)`

Task configuration is defined in [wind_datasets/models.py](/Users/sam/Developer/Code/Wind%20Power%20Forecasting/datasets/src/wind_datasets/models.py) via `TaskSpec`.

`TaskSpec` is duration-based, not step-based. Step counts are derived from the
dataset resolution at runtime.

## Cache layout

Each dataset is built under:

```text
cache/<dataset>/
  manifest/
  silver/
  gold_base/<quality_profile>/
  tasks/<quality_profile>/<task_id>/
```

Meaning of the layers:

- `manifest`: source file inventory, hashes, and build metadata
- `silver`: dataset-specific normalized artifacts
- `gold_base/<quality_profile>`: task-independent regular time series table
- `tasks/<quality_profile>/<task_id>`: task-specific window index and task metadata

The cache is disposable and can be rebuilt.

## Package layout

```text
src/
  wind_datasets/
    __init__.py
    api.py
    models.py
    registry.py
    manifest.py
    paths.py
    utils.py
    datasets/
      base.py
      common.py
      greenbyte.py
      hill_of_towie.py
      sdwpf_full.py
```

High-level responsibilities:

- `api.py`: top-level functions used by callers
- `models.py`: dataset and task configuration objects
- `registry.py`: hardcoded dataset registrations and source roots
- `manifest.py`: source file manifest generation
- `paths.py`: cache path helpers
- `datasets/`: dataset-specific builders and shared dataset logic

## Minimal usage

```python
from wind_datasets import (
    build_gold_base,
    build_manifest,
    build_silver,
    build_task_cache,
    load_series,
    load_window_index,
)
from wind_datasets.models import TaskSpec

dataset_id = "kelmarsh"
task = TaskSpec.next_6h_from_24h()

build_manifest(dataset_id)
build_silver(dataset_id)
build_gold_base(dataset_id)
build_task_cache(dataset_id, task)

series = load_series(dataset_id)
window_index = load_window_index(dataset_id, task)
```

`sdwpf_full` also supports explicit quality profiles:

```python
series = load_series("sdwpf_full", quality_profile="official_v1")
raw_series = load_series("sdwpf_full", quality_profile="raw_v1")
zeroed_series = load_series(
    "sdwpf_full",
    quality_profile="official_v1_zero_negative_patv",
)
```

Custom tasks are also supported:

```python
from wind_datasets.models import TaskSpec

task = TaskSpec(
    task_id="next_3h_from_12h",
    history_duration="12h",
    forecast_duration="3h",
)
```

## Environment

The package is configured in [pyproject.toml](/Users/sam/Developer/Code/Wind%20Power%20Forecasting/datasets/pyproject.toml) and is intended to run in the
repository's `./.conda` environment.

An editable install is expected during development:

```bash
pip install -e .
```

## Current assumptions

- source datasets live under `/Users/sam/Developer/Datasets/Wind Power Forecasting`
- source dataset directories are treated as read-only
- cache artifacts are written only under `./cache`
- default task preset is `next_6h_from_24h`, but tasks remain configurable
- `quality_flags` is an aggregated audit field, not a guarantee that a row is physically valid
- `sdwpf_full` defaults to `quality_profile="official_v1"`

## `sdwpf_full` quality profiles

Three quality profiles are currently supported for `sdwpf_full`:

- `official_v1`: default. Keeps raw values, adds official flags and boolean mask columns.
- `raw_v1`: reproduces the older behavior. Only generic missing-row / missing-target flags are emitted.
- `official_v1_zero_negative_patv`: same as `official_v1`, but writes `target_kw=0` when `Patv < 0` and preserves the original value in `target_kw_raw`.

`quality_flags` may include generic data-layer tags such as `missing_row` and
`missing_target`, plus `sdwpf_full`-specific tags such as
`sdwpf_patv_negative`, `sdwpf_unknown_patv_wspd`, `sdwpf_unknown_pitch`,
`sdwpf_abnormal_ndir`, `sdwpf_abnormal_wdir`, and `sdwpf_patv_zeroed`.

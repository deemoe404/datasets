# `src` Package Notes

This directory contains the `wind_datasets` Python package.

It is the repository's forecasting-oriented data layer. It builds disposable
cache artifacts under `./cache`, standardizes dataset-specific source files into
task-independent `gold_base` tables, exposes task window indexes, and provides
research-facing helpers such as dataset profiling and turbine static metadata.

It is not a training framework, and it is not a table-by-table mirror of the
official raw datasets.

## Public API

The main import surface is [wind_datasets/__init__.py](/Users/sam/Developer/Code/Wind%20Power%20Forecasting/datasets/src/wind_datasets/__init__.py).

Typical entry points:

- `build_manifest(dataset_id)`
- `build_silver(dataset_id)`
- `build_gold_base(dataset_id, quality_profile=None, layout=None, feature_set=None)`
- `build_task_cache(dataset_id, task_spec, quality_profile=None)`
- `load_series(dataset_id, quality_profile=None, layout=None, feature_set=None)`
- `load_shared_timeseries(dataset_id, group_name)`
- `load_event_features(dataset_id, group_name)`
- `load_interventions(dataset_id, group_name)`
- `load_window_index(dataset_id, task_spec, quality_profile=None)`
- `load_task_turbine_static(dataset_id, task_spec, quality_profile=None)`
- `load_turbine_static(dataset_id)`
- `profile_dataset(dataset_id, quality_profile=None, layout=None, feature_set=None)`

`TaskSpec` lives in [wind_datasets/models.py](/Users/sam/Developer/Code/Wind%20Power%20Forecasting/datasets/src/wind_datasets/models.py). It is duration-based, not step-based. Runtime step counts are derived from each dataset's native resolution.

## Cache Layout

Each dataset is built under:

```text
cache/<dataset>/
  manifest/
  silver/
    continuous/
    events/
    shared_ts/
    event_features/
    interventions/
    meta/
      turbine_static.parquet
  gold_base/<quality_profile>/<layout>/<feature_set>/
  tasks/<quality_profile>/<granularity>/<task_id>/
    window_index.parquet
    task_report.json
    turbine_static.parquet
    task_context.json
```

Layer meanings:

- `manifest`: source file inventory, official release metadata, source layout checks, release warnings
- `silver`: dataset-specific normalized artifacts
- `silver/meta/turbine_static.parquet`: unified turbine-level spatial/static sidecar
- `silver/shared_ts/*`: standardized regular shared or turbine-level auxiliary time series
- `silver/event_features/*`: regularized causal features derived from interval events
- `silver/interventions/*`: normalized intervention timelines kept outside the default graph assumptions
- `gold_base/<quality_profile>/<layout>/<feature_set>`: model-ready regular series cache
- `tasks/<quality_profile>/<granularity>/<task_id>`: task-specific window index and task-local metadata

The cache is disposable and can be rebuilt from the read-only source dataset directories.

## Spatial Static Sidecar

All four registered datasets now expose a unified `turbine_static.parquet` sidecar at:

```text
cache/<dataset>/silver/meta/turbine_static.parquet
```

Fixed schema:

- `dataset`
- `turbine_id`
- `source_turbine_key`
- `latitude`
- `longitude`
- `coord_x`
- `coord_y`
- `coord_kind`
- `coord_crs`
- `elevation_m`
- `rated_power_kw`
- `hub_height_m`
- `rotor_diameter_m`
- `manufacturer`
- `model`
- `country`
- `commercial_operation_date`
- `spatial_source`

This sidecar is the repository's default spatial interface. It does not precompute graph edges. If an experiment needs kNN edges, directional graphs, or wake-aware topology, those should be derived in the experiment layer from `turbine_static`.

## Current Semantics

- The package is forecasting-oriented, not a faithful table-by-table mirror of the official releases.
- `silver` preserves dataset-specific structure; `gold_base` is the model-ready projection.
- Default `gold_base` layout is farm-synchronous long form on a shared farm time axis.
- Explicit `layout="turbine"` remains available for single-turbine baselines and compatibility consumers.
- Farm-synchronous rows expose `farm_turbines_expected`, `farm_turbines_observed`, `farm_turbines_with_target`, and synchronous-coverage flags.
- Farm task caches write a task-local `turbine_static.parquet` with stable `turbine_index` ordering.
- Suitable regular time assets flow into default `gold_base`; text-heavy, daily summary, and raw interval tables remain in `silver`.
- `quality_flags` is an audit field. `quality_flags == ""` does not mean a row is physically perfect; it only means no implemented rule flagged it.
- `sdwpf_kddcup` defaults to `quality_profile="default"`.
- `sdwpf_kddcup default` means “implemented SDWPF unknown/abnormal rules are encoded as flags and mask columns.” It does not mean “official evaluation filtering has already been applied.”
- `sdwpf_kddcup` gold/task builds are fail-closed when manifest time-semantics audit finds `Day + Tmstamp` values incompatible with the documented 245-day 10-minute grid.
- `TaskSpec.next_6h_from_24h()` now defaults to `granularity="farm"`.

## Dataset Cards

### `kelmarsh`

- Official source:
  - [Kelmarsh legacy release (2022)](https://zenodo.org/records/5841834)
  - [Kelmarsh extended release (2025)](https://zenodo.org/records/16807551)
- Default expected release: `extended_2025`
- Coverage:
  - legacy release: `2016-01-03` to `2021-06-30`
  - extended release: `2016-01-03` to `2024-12-31`
- Time semantics: UTC, source timestamps are treated as naive UTC values
- Official assets:
  - KMZ layout
  - turbine static table
  - turbine SCADA
  - status/event logs
  - signal mapping
  - PMU meter
  - grid meter
- Current `silver` ingest:
  - turbine static table
  - turbine SCADA
  - status/event logs
  - signal mapping
  - PMU meter time series
  - grid meter time series
  - causal turbine/farm status event features
- Current default `gold_base`:
  - target `Power (kW)`
  - default feature set: all numeric turbine SCADA predictors found in the continuous exports
  - explicit `feature_set="lightweight"`: 13 selected SCADA continuous features
  - farm-shared PMU features
  - farm-shared grid meter features
  - turbine status event features
  - farm status event features
- Current exclusions:
  - KMZ layout
  - raw `Message` / `Comment` text
- Source layout expectation: official archives must already be extracted before build
- Spatial sidecar: yes, from `*_WT_static.csv`

### `penmanshiel`

- Official source:
  - [Penmanshiel legacy release (2022)](https://zenodo.org/records/5946808)
  - [Penmanshiel extended release (2025)](https://zenodo.org/records/16807304)
- Default expected release: `extended_2025`
- Coverage:
  - legacy release: `2016-06-02` to `2021-06-30`
  - extended release: `2016-06-02` to `2024-12-31`
- Time semantics: UTC, source timestamps are treated as naive UTC values
- Official assets:
  - KMZ layout
  - turbine static table
  - turbine SCADA
  - status/event logs
  - signal mapping
  - PMU meter
  - grid meter
- Current `silver` ingest:
  - turbine static table
  - turbine SCADA
  - status/event logs
  - signal mapping
  - PMU meter time series
  - grid meter time series
  - causal turbine/farm status event features
- Current default `gold_base`:
  - target `Power (kW)`
  - default feature set: all numeric turbine SCADA predictors found in the continuous exports
  - explicit `feature_set="lightweight"`: 13 selected SCADA continuous features
  - farm-shared PMU features
  - farm-shared grid meter features
  - turbine status event features
  - farm status event features
- Current exclusions:
  - KMZ layout
  - raw `Message` / `Comment` text
- Source layout expectation: official archives must already be extracted before build
- Spatial sidecar: yes, from `*_WT_static.csv`

### `hill_of_towie`

- Official source: [Hill of Towie wind farm open dataset (2025)](https://zenodo.org/records/14870023)
- Default expected release: `v1_2025`
- Coverage: `2016-01-01` to `2024-08-31`
- Time semantics: UTC. Official documentation states that 10-minute timestamps denote the end of the 10-minute interval.
- Official assets:
  - turbine metadata
  - `tblSCTurbine`
  - `tblSCTurGrid`
  - `tblSCTurFlag`
  - `tblAlarmLog`
  - `tblDailySummary`
  - `tblGrid`
  - `tblGridScientific`
  - `tblSCTurCount`
  - `tblSCTurDigiIn`
  - `tblSCTurDigiOut`
  - `tblSCTurIntern`
  - `tblSCTurPress`
  - `tblSCTurTemp`
  - AeroUp supporting file plus repo-vendored official TuneUp metadata
  - ShutdownDuration-derived files
- Current `silver` ingest:
  - all CSV tables are converted to parquet
  - turbine metadata is also standardized into `turbine_static.parquet`
  - standardized shared/turbine regular sidecars under `silver/shared_ts`
  - standardized alarm event table under `silver/events/alarmlog.parquet`
  - standardized AeroUp and TuneUp interventions under `silver/interventions`
  - causal alarm, AeroUp, and TuneUp features under `silver/event_features`
- Current default `gold_base`:
  - `tblSCTurbine`
  - `tblSCTurGrid`
  - `tblSCTurFlag`
  - `tblGrid`
  - `tblGridScientific`
  - `tblSCTurCount`
  - `tblSCTurDigiIn`
  - `tblSCTurDigiOut`
  - `tblSCTurIntern`
  - `tblSCTurPress`
  - `tblSCTurTemp`
  - `ShutdownDuration`
  - causal `tblAlarmLog` features
  - causal `AeroUp` regime features
  - causal `TuneUp` regime features
  - target `wtc_ActPower_mean`
- Current exclusions from default `gold_base`:
  - `tblDailySummary`
  - raw interval `tblAlarmLog` table
  - raw AeroUp timeline file
  - vendored TuneUp provenance metadata
- Duplicate handling:
  - default-table duplicate keys are audited in `silver/meta/default_table_duplicate_audit.parquet`
  - duplicate audit distinguishes `identical`, `normalized_equal`, and `true_conflict`
  - non-conflicting duplicates keep the first row
  - `true_conflict` duplicates keep the first row and add `duplicate_conflict_resolved` to `quality_flags`
- Source layout expectation: official year archives must already be extracted before build
- Spatial sidecar: yes, from `Hill_of_Towie_turbine_metadata.csv`

### `sdwpf_kddcup`

- Official source:
  - [Figshare dataset page](https://figshare.com/articles/dataset/SDWPF_dataset/24798654)
  - [Baidu KDD Cup paper](https://arxiv.org/abs/2208.04360)
- Default expected release: `figshare_v2_2024`
- Coverage: `2020-05-01` to `2020-12-31`
- Time semantics:
  - source file exposes `Day + Tmstamp`, not explicit calendar dates
  - this repository derives `timestamp` by anchoring `Day 1 = 2020-05-01`
  - that anchor is a repository convention based on public reproduction references, not a field stored in the original CSV
- Official assets:
  - `sdwpf_245days_v1.csv`
  - `sdwpf_baidukddcup2022_turb_location.csv`
  - `final_phase_test`
- Current `silver` ingest:
  - normalized parquet converted from `sdwpf_245days_v1.csv`
  - turbine location CSV converted to parquet
- Current default `gold_base`:
  - blocked unless manifest time-semantics audit matches the documented 245-day 10-minute grid
  - when unblocked: target `Patv`, KDDCup turbine-level dynamic predictors, and SDWPF quality-rule flags/mask columns
- Current exclusions:
  - `final_phase_test`
  - `Day` and raw `Tmstamp` are used for timestamp recovery and audit, not default model features
  - no elevation column in the spatial sidecar
  - source release does not include `T2m`, `Sp`, `RelH`, `Wspd_w`, `Wdir_w`, or `Tp`
- Source layout expectation: extracted CSV files must be present
- Spatial sidecar: yes, from `sdwpf_baidukddcup2022_turb_location.csv`

## `sdwpf_kddcup` Quality Profiles

`sdwpf_kddcup` supports a single quality profile:

- `default`: keeps raw `Patv` values, including negatives, and adds SDWPF unknown/abnormal flags plus boolean mask columns.

Implemented SDWPF-specific flags:

- `sdwpf_unknown_patv_wspd`
- `sdwpf_unknown_pitch`
- `sdwpf_abnormal_ndir`
- `sdwpf_abnormal_wdir`

Window-level mask counts remain available in turbine `window_index.parquet`. Farm windows instead carry per-step turbine availability and mask-count lists.

## Minimal Usage

```python
from wind_datasets import (
    build_gold_base,
    build_manifest,
    build_silver,
    build_task_cache,
    load_event_features,
    load_interventions,
    load_series,
    load_shared_timeseries,
    load_task_turbine_static,
    load_turbine_static,
    load_window_index,
)
from wind_datasets.models import TaskSpec

dataset_id = "kelmarsh"
farm_task = TaskSpec.next_6h_from_24h()
turbine_task = TaskSpec.next_6h_from_24h(granularity="turbine")

build_manifest(dataset_id)
build_silver(dataset_id)
build_gold_base(dataset_id)  # default: farm layout
build_gold_base(dataset_id, layout="turbine")  # explicit compatibility layout
build_task_cache(dataset_id, farm_task)
build_task_cache(dataset_id, turbine_task)

series = load_series(dataset_id)
turbine_series = load_series(dataset_id, layout="turbine")
turbine_static = load_turbine_static(dataset_id)
farm_task_turbine_static = load_task_turbine_static(dataset_id, farm_task)
farm_pmu = load_shared_timeseries(dataset_id, "farm_pmu")
status_features = load_event_features(dataset_id, "turbine_status")
farm_window_index = load_window_index(dataset_id, farm_task)
turbine_window_index = load_window_index(dataset_id, turbine_task)
```

## Cache Rebuild

For routine rebuilds, use the repo wrapper script:

```bash
./scripts/rebuild_cache.sh
```

That default command rebuilds the standard cache stack for all four supported datasets:

- `manifest`
- `silver`
- `gold_base/default/farm/default`
- `tasks/default/farm/next_6h_from_24h`

Common variants:

```bash
./scripts/rebuild_cache.sh hill_of_towie
./scripts/rebuild_cache.sh kelmarsh penmanshiel
./scripts/rebuild_cache.sh --include-turbine hill_of_towie
./scripts/rebuild_cache.sh --clean hill_of_towie
./scripts/rebuild_cache.sh --cache-root /tmp/wind-cache sdwpf_kddcup
```

Notes:

- The script uses the repository's `./.conda/bin/python` by default.
- The wrapper exports `PYTHONPATH=src`, so it works without requiring `pip install -e .`.
- The underlying Python entrypoint is `python -m wind_datasets.rebuild_cache`.
- `--clean` removes `cache/<dataset>` before rebuilding.
- `--include-turbine` opts into the compatibility turbine `gold_base/task` artifacts; the default rebuild stays farm-only.
- When rebuilding multiple datasets, the command continues after a dataset-level failure, then exits non-zero with a failure summary.
- `sdwpf_kddcup` still fail-closes at `gold_base/task` if the manifest time-semantics audit does not match the documented 245-day 10-minute grid.

If an experiment needs a non-standard task, build it directly from Python:

```bash
PYTHONPATH=src ./.conda/bin/python - <<'PY'
from wind_datasets import build_task_cache
from wind_datasets.models import TaskSpec

build_task_cache(
    "hill_of_towie",
    TaskSpec(
        task_id="custom_task",
        history_duration="24h",
        forecast_duration="6h",
        stride_duration="30m",
        granularity="farm",
    ),
    cache_root="cache",
)
PY
```

## Environment

The package is configured in [pyproject.toml](/Users/sam/Developer/Code/Wind%20Power%20Forecasting/datasets/pyproject.toml) and is intended to run in the repository's `./.conda` environment.

Editable install during development:

```bash
pip install -e .
```

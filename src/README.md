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
- `build_gold_base(dataset_id, quality_profile=None)`
- `build_task_cache(dataset_id, task_spec, quality_profile=None)`
- `load_series(dataset_id, quality_profile=None)`
- `load_shared_timeseries(dataset_id, group_name)`
- `load_event_features(dataset_id, group_name)`
- `load_interventions(dataset_id, group_name)`
- `load_window_index(dataset_id, task_spec, quality_profile=None)`
- `load_turbine_static(dataset_id)`
- `profile_dataset(dataset_id, quality_profile=None)`

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
  gold_base/<quality_profile>/
  tasks/<quality_profile>/<task_id>/
```

Layer meanings:

- `manifest`: source file inventory, official release metadata, source layout checks, release warnings
- `silver`: dataset-specific normalized artifacts
- `silver/meta/turbine_static.parquet`: unified turbine-level spatial/static sidecar
- `silver/shared_ts/*`: standardized regular shared or turbine-level auxiliary time series
- `silver/event_features/*`: regularized causal features derived from interval events
- `silver/interventions/*`: normalized intervention timelines kept outside the default graph assumptions
- `gold_base/<quality_profile>`: default regular time-series table for forecasting
- `tasks/<quality_profile>/<task_id>`: task-specific window index and task metadata

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
- `silver` tries to preserve dataset-specific structure; `gold_base` is a default modeling projection.
- `gold_base` intentionally keeps only a subset of official assets and fields.
- Suitable regular time assets now flow into default `gold_base`; text-heavy, daily summary, and raw interval tables remain in `silver`.
- `quality_flags` is an audit field. `quality_flags == ""` does not mean a row is physically perfect; it only means no implemented rule flagged it.
- `sdwpf_full` defaults to `quality_profile="official_v1"`.
- `sdwpf_full official_v1` means “official rules are encoded as flags and mask columns.” It does not mean “official evaluation filtering has already been applied.”
- `masked_output` windows remain in `window_index`. If an experiment needs official-evaluation-aligned filtering, it must apply that filter explicitly downstream.

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
  - 13 selected SCADA continuous features
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
  - 13 selected SCADA continuous features
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
  - AeroUp / TuneUp supporting files
  - ShutdownDuration-derived files
- Current `silver` ingest:
  - all CSV tables are converted to parquet
  - turbine metadata is also standardized into `turbine_static.parquet`
  - standardized shared/turbine regular sidecars under `silver/shared_ts`
  - standardized alarm event table under `silver/events/alarmlog.parquet`
  - causal alarm and AeroUp features under `silver/event_features`
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
  - target `wtc_ActPower_mean`
- Current exclusions from default `gold_base`:
  - `tblDailySummary`
  - raw interval `tblAlarmLog` table
  - raw AeroUp timeline file
- Duplicate handling:
  - default-table duplicate keys are audited in `silver/meta/default_table_duplicate_audit.parquet`
  - non-conflicting duplicates keep the first row
  - conflicting duplicates keep the first row and add `duplicate_conflict_resolved` to `quality_flags`
- Source layout expectation: official year archives must already be extracted before build
- Spatial sidecar: yes, from `Hill_of_Towie_turbine_metadata.csv`

### `sdwpf_full`

- Official source: [Scientific Data paper / dataset description](https://www.nature.com/articles/s41597-024-03427-5)
- Default expected release: `scientific_data_2024`
- Coverage: `2020-01-01` to `2021-12-31`
- Time semantics: 10-minute resolution, source timestamps are documented as `UTC+08:00`
- Official assets:
  - main CSV
  - main parquet
  - turbine location/elevation CSV
- Current `silver` ingest:
  - main parquet
  - turbine location/elevation CSV
- Current default `gold_base`:
  - target `Patv`
  - all dynamic predictors from the main time-series table
  - SDWPF quality-rule flags and mask columns in official profiles
- Current exclusions:
  - main CSV is not copied into `silver`
- Source layout expectation: extracted parquet and location CSV must be present
- Spatial sidecar: yes, from `sdwpf_turb_location_elevation.csv`

## `sdwpf_full` Quality Profiles

Three quality profiles are currently supported:

- `official_v1`: default. Keeps raw target values, adds official flags and boolean mask columns.
- `raw_v1`: disables SDWPF-specific official flags and keeps only generic missing-row / missing-target flags.
- `official_v1_zero_negative_patv`: same as `official_v1`, but writes `target_kw=0` when `Patv < 0` and preserves the original value in `target_kw_raw`.

Implemented SDWPF-specific flags:

- `sdwpf_patv_negative`
- `sdwpf_unknown_patv_wspd`
- `sdwpf_unknown_pitch`
- `sdwpf_abnormal_ndir`
- `sdwpf_abnormal_wdir`
- `sdwpf_patv_zeroed`

Window-level `masked_input` / `masked_output` flags are available in `window_index.parquet`, but those windows are still retained by default.

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
    load_turbine_static,
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
turbine_static = load_turbine_static(dataset_id)
farm_pmu = load_shared_timeseries(dataset_id, "farm_pmu")
status_features = load_event_features(dataset_id, "turbine_status")
window_index = load_window_index(dataset_id, task)
```

## Environment

The package is configured in [pyproject.toml](/Users/sam/Developer/Code/Wind%20Power%20Forecasting/datasets/pyproject.toml) and is intended to run in the repository's `./.conda` environment.

Editable install during development:

```bash
pip install -e .
```

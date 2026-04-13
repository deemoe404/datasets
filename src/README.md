# `src` Package Notes

This directory contains the `wind_datasets` Python package.

It is the repository's dataset-side runtime for the active forecasting task:

- dense sliding windows
- `24h` look back
- `6h` ahead
- farm-level task bundles only

It builds disposable cache artifacts under `./cache`, standardizes the four
supported official datasets into a shared forecasting surface, and exposes the
experiment-facing task bundle contract used by the current experiment family.

It is not:

- a training framework
- an editable mirror of raw source datasets
- a place to mutate or repair source data
- a table-by-table clone of the official releases

For dataset coverage and retained raw columns, read [../README.md](../README.md).
For the active experiment surface, read [../experiment/README.md](../experiment/README.md).

## What Lives Here

- [wind_datasets/api.py](./wind_datasets/api.py): public build/load helpers
- [wind_datasets/models.py](./wind_datasets/models.py): dataset/task dataclasses
- [wind_datasets/registry.py](./wind_datasets/registry.py): fixed dataset registry
- [wind_datasets/datasets/](./wind_datasets/datasets): dataset-specific builders
- [wind_datasets/feature_protocols.py](./wind_datasets/feature_protocols.py): task feature protocol definitions
- [wind_datasets/source_column_policy.py](./wind_datasets/source_column_policy.py): raw source-column retention policy loader
- [wind_datasets/feature_policy.py](./wind_datasets/feature_policy.py): auxiliary feature masking/drop policy helper
- [wind_datasets/cache_state.py](./wind_datasets/cache_state.py): cache DAG fingerprinting and freshness checks
- [wind_datasets/rebuild_cache.py](./wind_datasets/rebuild_cache.py): CLI rebuild/check entrypoint
- [wind_datasets/visualization.py](./wind_datasets/visualization.py): small analysis helpers used by the notebook

## Public API

The main import surface is [wind_datasets/__init__.py](./wind_datasets/__init__.py).

Top-level exports:

- `build_manifest(dataset_id)`
- `build_silver(dataset_id)`
- `build_gold_base(dataset_id)`
- `build_task_cache(dataset_id, task_spec, feature_protocol_id="power_only")`
- `load_series(dataset_id)`
- `load_task_bundle(dataset_id, task_spec, feature_protocol_id="power_only")`
- `load_window_index(dataset_id, task_spec, feature_protocol_id="power_only")`
- `load_task_turbine_static(dataset_id, task_spec, feature_protocol_id="power_only")`
- `load_turbine_static(dataset_id)`
- `load_shared_timeseries(dataset_id, group_name)`
- `load_event_features(dataset_id, group_name)`
- `load_interventions(dataset_id, group_name)`
- `load_duplicate_audit(dataset_id)`
- `profile_dataset(dataset_id)`
- `get_dataset_spec(dataset_id)`
- `list_dataset_ids()`
- `list_dataset_specs()`

Top-level model exports:

- `TaskSpec`
- `ResolvedTaskSpec`
- `DatasetSpec`
- `OfficialRelease`

`TaskSpec` is duration-based, not step-based. Runtime step counts are derived
from each dataset's native resolution when `task_spec.resolve(...)` runs.

## Current Architecture

The package has a single public cache path:

`manifest -> silver -> gold_base -> tasks/<task_id>/<feature_protocol_id>`

Layer meanings:

- `manifest`: source inventory, source schema inventory, release/time-semantics checks
- `silver`: dataset-specific normalized artifacts
- `gold_base`: single public farm-synchronous canonical long panel
- `task`: experiment-facing bundle derived from `gold_base` plus a feature protocol

Important current constraints:

- `gold_base` is always farm-synchronous
- `layout` is no longer a public cache dimension
- `quality_profile` is no longer a public cache dimension; only legacy `default` remains valid
- task bundles only support `granularity="farm"`
- non-farm task bundles are rejected in [wind_datasets/datasets/base.py](./wind_datasets/datasets/base.py)

## Cache Layout

Each dataset is built under:

```text
cache/<dataset>/
  manifest/
    manifest.json
    _build_meta.json
  silver/
    continuous/
    events/
    shared_ts/
    event_features/
    interventions/
    meta/
      turbine_static.parquet
      duplicate_audit.parquet
      duplicate_effects.parquet
    _build_meta.json
  gold_base/
    series.parquet
    quality_report.json
    _build_meta.json
  tasks/<task_id>/<feature_protocol_id>/
    series.parquet
    known_future.parquet
    static.parquet
    window_index.parquet
    task_context.json
    task_report.json
    _build_meta.json
```

Notes:

- `task_report.json` is diagnostic; treat it as helpful side output, not the primary contract
- the cache is disposable and may be deleted and rebuilt
- source dataset directories remain read-only; all derived artifacts belong in `./cache`

## Task Bundle Contract

Task bundles are the only supported experiment entry point for active families.

Each task bundle contains:

- `series.parquet`: protocol-selected farm-level long panel
- `known_future.parquet`: deterministic calendar covariates for the bundle timestamp axis
- `static.parquet`: complete normalized turbine static sidecar with stable `turbine_index`
- `window_index.parquet`: dense sliding window index over the selected series
- `task_context.json`: machine-readable protocol/task description
- `task_report.json`: build diagnostics and column summaries

The default task is `TaskSpec.next_6h_from_24h()`, which resolves to:

- `task_id="next_6h_from_24h"`
- `history_duration="24h"`
- `forecast_duration="6h"`
- dense sliding stride equal to dataset resolution
- `granularity="farm"`

## Feature Protocols

Feature protocol definitions live in
[wind_datasets/feature_protocols.py](./wind_datasets/feature_protocols.py).

The active protocol IDs today are:

- `power_only`
- `power_ws_hist`
- `power_atemp_hist`
- `power_itemp_hist`
- `power_wd_hist_sincos`
- `power_wd_yaw_hist_sincos`
- `power_wd_yaw_pitchmean_hist_sincos`
- `power_wd_yaw_lrpm_hist_sincos`
- `power_ws_wd_hist_sincos`

Protocol semantics:

- `power_only`: target history only
- `power_ws_hist`: target history plus dataset-native past wind-speed covariates
- `power_atemp_hist`: target history plus dataset-native ambient-temperature covariates
- `power_itemp_hist`: target history plus dataset-native internal-temperature covariates
- `power_wd_hist_sincos`: target history plus task-derived wind-direction `sin/cos` covariates
- `power_wd_yaw_hist_sincos`: target history plus task-derived wind-direction and yaw-error `sin/cos` covariates
- `power_wd_yaw_pitchmean_hist_sincos`: target history plus task-derived wind-direction and yaw-error `sin/cos` covariates and task-derived `pitch_mean`
- `power_wd_yaw_lrpm_hist_sincos`: target history plus task-derived wind-direction and yaw-error `sin/cos` covariates and dataset-native low-speed rotor RPM history
- `power_ws_wd_hist_sincos`: target history plus dataset-native wind speed and task-derived wind-direction `sin/cos` covariates

For `sdwpf_kddcup`, `Wdir` is treated as the documented relative yaw-error
angle and absolute wind direction is reconstructed as `Ndir + Wdir` for the
wind-direction protocols. The yaw-aware protocol additionally emits
`yaw_error_sin/cos` directly from `Wdir`. The low-speed-rotor-RPM protocol is
not supported for `sdwpf_kddcup` and raises an explicit dataset-support error.
The bundle `task_context.json` records the exact per-dataset angle transforms
used by each protocol.

The default feature protocol is `power_only`.

Protocol metadata is written into each bundle's `task_context.json`. The code
path that selects columns for a bundle is
[select_task_series_columns(...)](./wind_datasets/feature_protocols.py).

## Policy Files

There are two different policy layers in `src/wind_datasets/data/`.

### Source Column Policy

Files under
[wind_datasets/data/source_column_policy/](./wind_datasets/data/source_column_policy)
are the operational ground truth for raw source-column retention.

They drive the dataset builders and define:

- which raw source columns are retained
- which columns are retained with masking
- which canonical outputs those source columns feed
- which active feature protocols require them, when applicable

These CSVs are part of cache freshness for the layers that depend on them.

### Feature Policy

Files under
[wind_datasets/data/feature_policy/](./wind_datasets/data/feature_policy)
are an auxiliary series-feature policy surface handled by
[wind_datasets/feature_policy.py](./wind_datasets/feature_policy.py).

They support:

- `keep`
- `mask+keep`
- `drop`

This helper is tested and packaged, but it is not part of the default
`manifest -> silver -> gold_base -> task` build path today. Treat it as a
small experiment-side utility for applying an explicit feature masking/drop
policy to an already-built series frame.

## Registered Datasets

The registry is fixed in [wind_datasets/registry.py](./wind_datasets/registry.py).

Currently supported dataset IDs:

- `kelmarsh`
- `penmanshiel`
- `hill_of_towie`
- `sdwpf_kddcup`

Handler mapping:

- `kelmarsh` -> `greenbyte`
- `penmanshiel` -> `greenbyte`
- `hill_of_towie` -> `hill_of_towie`
- `sdwpf_kddcup` -> `sdwpf_kddcup`

Runtime source roots are resolved from the repo-local
[../wind_datasets.local.toml](../wind_datasets.local.toml). Use
[../wind_datasets.local.toml.example](../wind_datasets.local.toml.example) as
the tracked template.

## Cache Freshness

Cache freshness is explicit and fingerprint-based.

Every layer writes a `_build_meta.json` sidecar. Read paths do not trust file
presence alone; they first validate freshness through
[wind_datasets/cache_state.py](./wind_datasets/cache_state.py).

Normal invalidation behavior:

- source data changes invalidate `manifest`, then all descendants
- code changes invalidate the affected layer, then all descendants
- packaged dependency changes invalidate the layer that declares them, then all descendants
- task parameter or `feature_protocol_id` changes invalidate only the affected task layer

Read helpers such as `load_task_bundle(...)`, `load_series(...)`, and
`profile_dataset(...)` automatically rebuild missing or stale parent layers.

Special case:

- `sdwpf_kddcup` remains fail-closed at `gold_base` and `task` when manifest time-semantics validation does not match the documented 245-day 10-minute grid

## Minimal Usage

```python
from wind_datasets import build_task_cache, load_series, load_task_bundle
from wind_datasets.models import TaskSpec

dataset_id = "kelmarsh"
task = TaskSpec.next_6h_from_24h()
feature_protocol_id = "power_only"

build_task_cache(dataset_id, task, feature_protocol_id=feature_protocol_id)

bundle = load_task_bundle(dataset_id, task, feature_protocol_id=feature_protocol_id)

print(bundle.series.columns)
print(bundle.static.columns)
print(bundle.known_future.columns)
print(bundle.task_context["feature_protocol_id"])

# Lower-level dataset access remains available when you explicitly want gold_base.
gold_series = load_series(dataset_id)
print(gold_series.columns)
```

## Rebuild CLI

Use the repo wrapper script for normal rebuild/check flows:

```bash
./scripts/rebuild_cache.sh
```

Default rebuild target set:

- `manifest`
- `silver`
- `gold_base`
- `tasks/next_6h_from_24h/power_only`

Common commands:

```bash
./scripts/rebuild_cache.sh --check
./scripts/rebuild_cache.sh hill_of_towie
./scripts/rebuild_cache.sh --clean kelmarsh penmanshiel
./scripts/rebuild_cache.sh --cache-root /tmp/wind-cache sdwpf_kddcup
```

Notes:

- the wrapper uses `./.conda/bin/python` by default
- the wrapper exports `PYTHONPATH=src`
- the underlying entrypoint is `python -m wind_datasets.rebuild_cache`
- `--include-turbine` is accepted only as a deprecated no-op compatibility flag

## Environment

The dataset-side environment is managed with `./.conda`.

Create or update it from the repository root:

```bash
./scripts/create_env.sh
```

That script:

- creates or updates `./.conda` from [../scripts/environment.yml](../scripts/environment.yml)
- upgrades `pip`
- installs the repository in editable mode with test extras

Typical commands after the environment exists:

```bash
./scripts/rebuild_cache.sh --check
./.conda/bin/python -m pytest
```

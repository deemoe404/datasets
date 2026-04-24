# world_model_hardened_baselines_v1

`world_model_hardened_baselines_v1` is retained as a phase-1 adapter sanity
surface against `world_model_state_space_v1` on the active `next_6h_from_24h`
task and existing `world_model_v1` bundle. It records official source
provenance, but trainable execution still delegates to the repo-local backend
variants from `world_model_baselines_v1`; these are not final official
implementation baselines.

Short form: phase-1 adapter sanity checks, repo-local backend execution, not final official implementation baselines.

This family is separate from `world_model_baselines_v1`:

- `world_model_baselines_v1`: repo-local controlled comparisons and
  task-adapted `*-style` variants.
- `world_model_hardened_baselines_v1`: phase-1 adapter sanity checks with
  official provenance annotations and repo-local backend execution.
- `world_model_official_baselines_v2`: final official / official-core baseline
  track for paper-grade claims.

The first tranche includes:

- `world_model_last_value_persistence_hardened_v1_farm_sync`: analytic
  last-value persistence anchor.
- `world_model_dgcrn_official_core_v1_farm_sync`: DGCRN official-core adapter
  tracked against `experiment/official_baselines/dgcrn/source`.
- `world_model_timexer_official_v1_farm_sync`: TimeXer official adapter tracked
  against `experiment/official_baselines/timexer/source`.
- `world_model_itransformer_official_v1_farm_sync`: iTransformer official
  adapter tracked against `experiment/official_baselines/itransformer/source`.
- `world_model_chronos_2_zero_shot_official_v1_farm_sync`: official
  `amazon/chronos-2` zero-shot package adapter.

Result rows reuse the long metrics surface and repo-local backend dispatch from
`world_model_baselines_v1`, then append:

- `implementation_track`
- `source_repo`
- `source_commit`
- `adapter_kind`
- `search_config_id`

## Run

```shell
cd experiment/families/world_model_hardened_baselines_v1
./create_env.sh
./.conda/bin/python run_world_model_hardened_baselines_v1.py
```

Run a single hardened variant:

```shell
./.conda/bin/python run_world_model_hardened_baselines_v1.py \
  --variant world_model_dgcrn_official_core_v1_farm_sync \
  --selection-metric val_rmse_pu
```

Smoke run:

```shell
./.conda/bin/python run_world_model_hardened_baselines_v1.py \
  --epochs 1 \
  --device cpu \
  --max-train-origins 64 \
  --max-eval-origins 32 \
  --output-path ../../artifacts/scratch/world_model_hardened_baselines_v1/kelmarsh_smoke.csv \
  --no-record-run
```

Default formal output:

- `experiment/artifacts/published/world_model_hardened_baselines_v1/20260424-phase1-adapter-*.csv`
- `experiment/artifacts/published/world_model_hardened_baselines_v1/20260424-phase1-adapter-*.training_history.csv`
- `experiment/artifacts/runs/world_model_hardened_baselines_v1/<timestamp>/manifest.json`

## Source Wrappers

External references live in `experiment/official_baselines/`. Source-backed
wrappers use `source/` as a git submodule and a sibling ignored `.conda/`
environment. Package-backed wrappers use the same environment contract without a
source submodule.

## Scope

- Dataset scope: `kelmarsh` only.
- Feature protocol: `world_model_v1`.
- Task: `next_6h_from_24h`, `history_steps=144`, `forecast_steps=36`.
- Registry `training_mode` is `trainable` because the family includes trainable
  DGCRN, TimeXer, and iTransformer hardened variants, even though persistence
  and Chronos-2 are analytic / zero-shot references.

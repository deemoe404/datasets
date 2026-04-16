# Experiment Overview

This directory contains the repository-tracked experiment runner, machine-
readable registry, and runtime artifact contracts for the forecasting task.

## Active Families

- [agcrn](./families/agcrn/README.md): `kelmarsh` + `penmanshiel`, farm-synchronous, 9-protocol AGCRN pilot (`power_only`, `power_ws_hist`, `power_atemp_hist`, `power_itemp_hist`, `power_wd_hist_sincos`, `power_wd_yaw_hist_sincos`, `power_wd_yaw_pitchmean_hist_sincos`, `power_wd_yaw_lrpm_hist_sincos`, `power_ws_wd_hist_sincos`)
- [agcrn_masked](./families/agcrn_masked/README.md): `kelmarsh` + `penmanshiel`, farm-synchronous, 1-protocol masked AGCRN smoke-test family (`power_wd_yaw_pmean_hist_sincos_masked`)
- [world_model_agcrn_v1](./families/world_model_agcrn_v1/README.md): `kelmarsh` + `penmanshiel`, farm-synchronous, 1-protocol geometry-aware seq2seq AGCRN family (`world_model_v1`)
- [world_model_rollout_v1](./families/world_model_rollout_v1/README.md): `kelmarsh` + `penmanshiel`, farm-synchronous, 1-protocol filtering-plus-rollout world-model family (`world_model_v1`)
- [world_model_state_space_v1](./families/world_model_state_space_v1/README.md): `kelmarsh` only, farm-synchronous, 1-protocol state-space world-model MVP family (`world_model_v1`)
- [world_model_baselines_v1](./families/world_model_baselines_v1/README.md): `kelmarsh` only, farm-synchronous, persistence and shared-weight TFT-no-graph baselines (`world_model_v1`)

Archived experiment families have been removed from the active tree. The
current experiment surface remains intentionally narrow while the dataset/task
architecture stabilizes around `tasks/<task_id>/<feature_protocol_id>/...`.

## Layout

- `families/`: concrete experiment-family code, environments, and family-local `.work/`
- `infra/`: shared experiment infrastructure such as registry and common protocol helpers
- `artifacts/`: published outputs, formal run records, and scratch outputs

Canonical runtime artifacts:

- `./artifacts/published/<family_id>/latest.csv`: current formal family-level result CSV
- `./artifacts/published/<family_id>/latest.training_history.csv`: epoch-level training loss and validation RMSE history for the latest formal result
- `./artifacts/runs/<family_id>/<timestamp>/manifest.json`: immutable run record for one CLI invocation
- `./artifacts/scratch/<family_id>/...`: ad hoc smoke/debug outputs
- `./families/<family>/.work/`: family-local resume, chunk, and checkpoint artifacts

## Notes

- `agcrn_official_aligned` remains the baseline AGCRN family; `agcrn_masked` validates masked task bundles; `world_model_agcrn_v1` validates the full `world_model_v1` protocol with `series` + `known_future` + `static` + `pairwise`; `world_model_rollout_v1` validates the simplified filtering-plus-rollout baseline; `world_model_state_space_v1` carries the Kelmarsh-only state-space world-model MVP on the same bundle contract; `world_model_baselines_v1` carries the key persistence and TFT-no-graph comparisons.
- Most current families target `kelmarsh` + `penmanshiel`; `world_model_state_space_v1` and `world_model_baselines_v1` are intentionally Kelmarsh-only in v1. `sdwpf_kddcup` is excluded because the strict `24h -> 6h` task has zero usable windows, and `hill_of_towie` is excluded because the masked and world-model protocols are not configured there in the active experiment tree.
- The active task contract is `24h -> 6h`, dense sliding windows, `farm` granularity.
- The active dataset-side protocol surface is 11 protocols: `power_only` + `power_ws_hist` + `power_atemp_hist` + `power_itemp_hist` + `power_wd_hist_sincos` + `power_wd_yaw_hist_sincos` + `power_wd_yaw_pitchmean_hist_sincos` + `power_wd_yaw_pmean_hist_sincos_masked` + `power_wd_yaw_lrpm_hist_sincos` + `power_ws_wd_hist_sincos` + `world_model_v1`. Their semantics live exclusively in `src/wind_datasets/feature_protocols.py`.
- `agcrn_official_aligned` still consumes the original 9 experiment protocols only. The masked protocol is wired only into `agcrn_masked`, and `world_model_v1` is wired into `world_model_agcrn_v1`, `world_model_rollout_v1`, `world_model_state_space_v1`, and `world_model_baselines_v1`.

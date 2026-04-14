# Experiment Overview

This directory contains the repository-tracked experiment runner, machine-
readable registry, and runtime artifact contracts for the forecasting task.

## Active Families

- [agcrn](./families/agcrn/README.md): `kelmarsh` + `penmanshiel`, farm-synchronous, 9-protocol AGCRN pilot (`power_only`, `power_ws_hist`, `power_atemp_hist`, `power_itemp_hist`, `power_wd_hist_sincos`, `power_wd_yaw_hist_sincos`, `power_wd_yaw_pitchmean_hist_sincos`, `power_wd_yaw_lrpm_hist_sincos`, `power_ws_wd_hist_sincos`)
- [agcrn_masked](./families/agcrn_masked/README.md): `kelmarsh` + `penmanshiel`, farm-synchronous, 1-protocol masked AGCRN smoke-test family (`power_wd_yaw_pmean_hist_sincos_masked`)

Archived experiment families have been removed from the active tree. The
current experiment surface remains intentionally narrow while the dataset/task
architecture stabilizes around `tasks/<task_id>/<feature_protocol_id>/...`.

## Layout

- `families/`: concrete experiment-family code, environments, and family-local `.work/`
- `infra/`: shared experiment infrastructure such as registry and common protocol helpers
- `artifacts/`: published outputs, formal run records, and scratch outputs

Canonical runtime artifacts:

- `./artifacts/published/<family_id>/latest.csv`: current formal family-level result CSV
- `./artifacts/runs/<family_id>/<timestamp>/manifest.json`: immutable run record for one CLI invocation
- `./artifacts/scratch/<family_id>/...`: ad hoc smoke/debug outputs
- `./families/<family>/.work/`: family-local resume, chunk, and checkpoint artifacts

## Notes

- `agcrn_official_aligned` remains the baseline AGCRN family; `agcrn_masked` is a separate smoke-test family for validating masked task bundles.
- Both current families target `kelmarsh` + `penmanshiel`; `sdwpf_kddcup` is excluded because the strict `24h -> 6h` task has zero usable windows, and `hill_of_towie` is excluded because `power_wd_yaw_pmean_hist_sincos_masked` is not configured there.
- The active task contract is `24h -> 6h`, dense sliding windows, `farm` granularity.
- The active dataset-side protocol surface is 10 protocols: `power_only` + `power_ws_hist` + `power_atemp_hist` + `power_itemp_hist` + `power_wd_hist_sincos` + `power_wd_yaw_hist_sincos` + `power_wd_yaw_pitchmean_hist_sincos` + `power_wd_yaw_pmean_hist_sincos_masked` + `power_wd_yaw_lrpm_hist_sincos` + `power_ws_wd_hist_sincos`. Their semantics live exclusively in `src/wind_datasets/feature_protocols.py`.
- `agcrn_official_aligned` still consumes the original 9 experiment protocols only. The dataset-side `power_wd_yaw_pmean_hist_sincos_masked` protocol is wired only into the dedicated `agcrn_masked` family, not retrofitted into the baseline AGCRN runner.

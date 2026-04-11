# Experiment Overview

This directory contains the repository-tracked experiment runners, machine-
readable registry, and runtime artifact contracts for the repository
forecasting tasks.

## Layout

- `families/`: concrete experiment-family code, environments, and family-local `.work/`
- `infra/`: shared experiment infrastructure such as registry and common protocol helpers
- `artifacts/`: published outputs, formal run records, and scratch outputs

- `chronos-2` and `chronos-2-exogenous` use `24h look back -> 6h ahead`, dense
  sliding windows at the raw turbine timestep, and the shared raw-timestamp
  chronological split `70/10/20` with strict-contained windows.
- Chronos runners are zero-shot and score only `test`, reporting both
  `rolling_origin_no_refit` and `non_overlap` views plus `overall` and
  horizon-wise rows.
- `agcrn` is a first-phase graph baseline pilot on the farm-synchronous
  Kelmarsh turbine panel, trains on `train`, and reports both `val` and `test`.
- `ltsf-linear` uses the same split definition, but trains on `train` and reports
  both `val` and `test`.
- `tft` is a first-phase Kelmarsh pilot: train origins are downsampled for
  efficiency, but `val/test` keep the same split definition and both eval views.

## Experiments

- [chronos-2](./families/chronos-2/README.md): Chronos-2 `power_only` baselines
- [chronos-2-exogenous](./families/chronos-2-exogenous/README.md): Chronos-2 with staged dataset-native past covariates
- [agcrn](./families/agcrn/README.md): Kelmarsh farm-synchronous AGCRN `power_only` pilot
- [ltsf-linear](./families/ltsf-linear/README.md): local `NLinear` / `DLinear` baselines with staged past covariates
- [tft](./families/tft/README.md): Kelmarsh TFT pilot with static, deterministic known-future, and staged historical inputs

Canonical runtime artifacts:

- `./artifacts/published/<family_id>/latest.csv`: current formal family-level result CSV
- `./artifacts/runs/<family_id>/<timestamp>/manifest.json`: immutable run record for one CLI invocation
- `./artifacts/scratch/<family_id>/...`: ad hoc smoke/debug outputs
- `./families/<family>/.work/`: family-local resume, chunk, and checkpoint artifacts

## Covariate Stages

The staged exogenous packs are defined in
[`experiment/infra/common/covariate_packs.py`](./infra/common/covariate_packs.py). They
now resolve directly against the canonical `gold_base/<quality_profile>/<layout>`
cache and select required columns from that single series view.

### Kelmarsh / Penmanshiel

`stage1_core`

- `Wind speed (m/s)`
- `Wind direction (°)`
- `Nacelle position (°)`
- `Generator RPM (RPM)`
- `Rotor speed (RPM)`
- `Ambient temperature (converter) (°C)`
- `Nacelle temperature (°C)`
- `Power factor (cosphi)`
- `Reactive power (kvar)`
- `Blade angle (pitch position) A (°)`
- `Blade angle (pitch position) B (°)`
- `Blade angle (pitch position) C (°)`

`stage2_ops`

- all `stage1_core` columns
- `farm_pmu__gms_power_kw`
- `farm_pmu__gms_reactive_power_kvar`
- `farm_pmu__gms_current_a`

`stage3_regime`

- all `stage2_ops` columns
- `evt_any_active`
- `evt_active_count`
- `evt_total_overlap_seconds`
- `evt_stop_active`
- `evt_warning_active`
- `evt_informational_active`
- `farm_evt_any_active`
- `farm_evt_active_count`
- `farm_evt_total_overlap_seconds`
- `farm_evt_stop_active`
- `farm_evt_warning_active`
- `farm_evt_informational_active`

These event columns are optional at pack resolution time and are included only
when present in the selected feature set.

### Hill of Towie

`stage1_core`

- `wtc_SecAnemo_mean`
- `wtc_PriAnemo_mean`
- `wtc_YawPos_mean`
- `wtc_GenRpm_mean`
- `wtc_MainSRpm_mean`
- `wtc_PitchRef_BladeA_mean`
- `wtc_PitchRef_BladeB_mean`
- `wtc_PitchRef_BladeC_mean`
- `wtc_TwrHumid_mean`

`stage2_ops`

- all `stage1_core` columns
- `tur_temp__wtc_ambietmp_mean`
- `tur_temp__wtc_naceltmp_mean`
- `tur_temp__wtc_geoiltmp_mean`
- `tur_press__wtc_hydpress_mean`
- `farm_grid__activepower`
- `farm_grid__reactivepower`
- `farm_grid__powerfactor`

`stage3_regime`

- all `stage2_ops` columns
- `alarm_any_active`
- `alarm_active_count`
- `alarm_total_overlap_seconds`
- `aeroup_in_install_window`
- `aeroup_post_install`
- `days_since_aeroup_start`
- `days_since_aeroup_end`

### sdwpf_kddcup

`stage1_core`

- `Wspd`
- `Wdir`
- `Etmp`
- `Itmp`
- `Ndir`

`stage2_ops`

- all `stage1_core` columns
- `Pab1`
- `Pab2`
- `Pab3`
- `Prtv`

`stage3_regime`

- all `stage2_ops` columns
- `sdwpf_is_unknown`
- `sdwpf_is_abnormal`
- `sdwpf_is_masked`

## Notes

- `chronos-2/run_power_only_full.py` uses a tuned CUDA chunk-batch profile:
  `univariate=128`, `univariate_power_stats=32`, `multivariate_knn6=32`.
- `agcrn` intentionally starts narrower than the other trainable baselines:
  Kelmarsh only, `power_only` only, and full-synchronous farm windows only.
- `chronos-2-exogenous` and `ltsf-linear` share the same staged covariate pack
  definitions.
- `artifacts/published/chronos2_power_only/latest.csv` and
  `artifacts/published/chronos2_exogenous/latest.csv` are long result files with
  `split_name=test`, `eval_protocol`, `metric_scope`, `lead_step`, and split
  window-count metadata.
- `ltsf-linear` currently reads the same staged packs but may react differently
  to sparse flags and near-constant covariates because it uses train-only
  z-score normalization on past covariates.
- `artifacts/published/ltsf_linear_local/latest.csv` is also a long result file: each job
  emits `val/test`, `rolling_origin_no_refit/non_overlap`, and
  `overall/horizon-wise` rows.

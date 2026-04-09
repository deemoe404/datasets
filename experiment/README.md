# Experiment Overview

This directory contains the repository-tracked experiment runners and result
CSVs for the `24h look back -> 6h ahead -> 6h stride` task.

## Experiments

- [chronos-2](/home/sam/datasets/experiment/chronos-2/README.md): Chronos-2 `power_only` baselines
- [chronos-2-exogenous](/home/sam/datasets/experiment/chronos-2-exogenous/README.md): Chronos-2 with staged dataset-native past covariates
- [ltsf-linear](/home/sam/datasets/experiment/ltsf-linear/README.md): local `NLinear` / `DLinear` baselines with staged past covariates

Tracked result files:

- `./chronos-2.csv`
- `./chronos-2-exogenous.csv`
- `./ltsf-linear.csv`

## Covariate Stages

The staged exogenous packs are defined in
[`experiment/common/covariate_packs.py`](/home/sam/datasets/experiment/common/covariate_packs.py).
`kelmarsh` and `penmanshiel` use `feature_set="lightweight"`. `hill_of_towie`
and `sdwpf_kddcup` use `feature_set="default"`.

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
- `farm_pmu__production_meter_data_availability`
- `farm_grid_meter__grid_meter_data_availability`

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

- `chronos-2-exogenous` and `ltsf-linear` share the same staged covariate pack
  definitions.
- `ltsf-linear` currently reads the same staged packs but may react differently
  to sparse flags and near-constant covariates because it uses train-only
  z-score normalization on past covariates.

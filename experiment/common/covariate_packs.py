from __future__ import annotations

from dataclasses import dataclass


DEFAULT_COVARIATE_STAGES = ("stage1_core", "stage2_ops", "stage3_regime")

_GREENBYTE_LIGHTWEIGHT_COLUMNS = (
    "Wind speed (m/s)",
    "Wind direction (°)",
    "Nacelle position (°)",
    "Generator RPM (RPM)",
    "Rotor speed (RPM)",
    "Ambient temperature (converter) (°C)",
    "Nacelle temperature (°C)",
    "Power factor (cosphi)",
    "Reactive power (kvar)",
    "Grid frequency (Hz)",
    "Blade angle (pitch position) A (°)",
    "Blade angle (pitch position) B (°)",
    "Blade angle (pitch position) C (°)",
)

_GREENBYTE_STAGE2_EXTRA_COLUMNS = (
    "farm_pmu__gms_power_kw",
    "farm_pmu__gms_power_setpoint_kw",
    "farm_pmu__gms_reactive_power_kvar",
    "farm_pmu__gms_voltage_v",
    "farm_pmu__gms_current_a",
    "farm_pmu__gms_grid_frequency_hz",
    "farm_pmu__production_meter_data_availability",
    "farm_grid_meter__grid_meter_data_availability",
)

_GREENBYTE_STAGE3_OPTIONAL_COLUMNS = (
    "evt_any_active",
    "evt_active_count",
    "evt_total_overlap_seconds",
    "evt_stop_active",
    "evt_warning_active",
    "evt_informational_active",
    "farm_evt_any_active",
    "farm_evt_active_count",
    "farm_evt_total_overlap_seconds",
    "farm_evt_stop_active",
    "farm_evt_warning_active",
    "farm_evt_informational_active",
)

_HILL_STAGE1_COLUMNS = (
    "wtc_SecAnemo_mean",
    "wtc_PriAnemo_mean",
    "wtc_ActualWindDirection_mean",
    "wtc_YawPos_mean",
    "wtc_GenRpm_mean",
    "wtc_MainSRpm_mean",
    "wtc_PitchRef_BladeA_mean",
    "wtc_PitchRef_BladeB_mean",
    "wtc_PitchRef_BladeC_mean",
    "wtc_GridFreq_mean",
    "wtc_TwrHumid_mean",
)

_HILL_STAGE2_EXTRA_COLUMNS = (
    "tur_temp__wtc_ambietmp_mean",
    "tur_temp__wtc_naceltmp_mean",
    "tur_temp__wtc_geoiltmp_mean",
    "tur_press__wtc_hydpress_mean",
    "farm_grid__frequency",
    "farm_grid__activepower",
    "farm_grid__reactivepower",
    "farm_grid__powerfactor",
)

_HILL_STAGE3_EXTRA_COLUMNS = (
    "alarm_any_active",
    "alarm_active_count",
    "alarm_total_overlap_seconds",
    "aeroup_in_install_window",
    "aeroup_post_install",
    "days_since_aeroup_start",
    "days_since_aeroup_end",
    "tuneup_in_deployment_window",
    "tuneup_post_effective",
    "days_since_tuneup_effective_start",
    "days_since_tuneup_deployment_end",
)

_SDWPF_STAGE1_COLUMNS = (
    "Wspd",
    "Wdir",
    "Etmp",
    "Itmp",
    "Ndir",
)

_SDWPF_STAGE2_EXTRA_COLUMNS = (
    "Pab1",
    "Pab2",
    "Pab3",
    "Prtv",
)

_SDWPF_STAGE3_EXTRA_COLUMNS = (
    "sdwpf_is_unknown",
    "sdwpf_is_abnormal",
    "sdwpf_is_masked",
)


@dataclass(frozen=True)
class CovariatePackSpec:
    dataset_id: str
    stage: str
    pack_name: str
    feature_set: str
    required_columns: tuple[str, ...]
    optional_columns: tuple[str, ...] = ()

    def selected_covariate_columns(self, available_columns: set[str]) -> tuple[str, ...]:
        missing = [column for column in self.required_columns if column not in available_columns]
        if missing:
            raise ValueError(
                f"Dataset {self.dataset_id!r} stage {self.stage!r} is missing required covariates {missing!r}."
            )
        return self.required_columns + tuple(
            column for column in self.optional_columns if column in available_columns
        )


def _greenbyte_pack(dataset_id: str, stage: str) -> CovariatePackSpec:
    if stage == "stage1_core":
        return CovariatePackSpec(
            dataset_id=dataset_id,
            stage=stage,
            pack_name=stage,
            feature_set="lightweight",
            required_columns=_GREENBYTE_LIGHTWEIGHT_COLUMNS,
        )
    if stage == "stage2_ops":
        return CovariatePackSpec(
            dataset_id=dataset_id,
            stage=stage,
            pack_name=stage,
            feature_set="lightweight",
            required_columns=_GREENBYTE_LIGHTWEIGHT_COLUMNS + _GREENBYTE_STAGE2_EXTRA_COLUMNS,
        )
    if stage == "stage3_regime":
        return CovariatePackSpec(
            dataset_id=dataset_id,
            stage=stage,
            pack_name=stage,
            feature_set="lightweight",
            required_columns=_GREENBYTE_LIGHTWEIGHT_COLUMNS + _GREENBYTE_STAGE2_EXTRA_COLUMNS,
            optional_columns=_GREENBYTE_STAGE3_OPTIONAL_COLUMNS,
        )
    raise ValueError(f"Unsupported covariate stage {stage!r}.")


def _hill_pack(stage: str) -> CovariatePackSpec:
    if stage == "stage1_core":
        return CovariatePackSpec(
            dataset_id="hill_of_towie",
            stage=stage,
            pack_name=stage,
            feature_set="default",
            required_columns=_HILL_STAGE1_COLUMNS,
        )
    if stage == "stage2_ops":
        return CovariatePackSpec(
            dataset_id="hill_of_towie",
            stage=stage,
            pack_name=stage,
            feature_set="default",
            required_columns=_HILL_STAGE1_COLUMNS + _HILL_STAGE2_EXTRA_COLUMNS,
        )
    if stage == "stage3_regime":
        return CovariatePackSpec(
            dataset_id="hill_of_towie",
            stage=stage,
            pack_name=stage,
            feature_set="default",
            required_columns=_HILL_STAGE1_COLUMNS + _HILL_STAGE2_EXTRA_COLUMNS + _HILL_STAGE3_EXTRA_COLUMNS,
        )
    raise ValueError(f"Unsupported covariate stage {stage!r}.")


def _sdwpf_pack(stage: str) -> CovariatePackSpec:
    if stage == "stage1_core":
        return CovariatePackSpec(
            dataset_id="sdwpf_kddcup",
            stage=stage,
            pack_name=stage,
            feature_set="default",
            required_columns=_SDWPF_STAGE1_COLUMNS,
        )
    if stage == "stage2_ops":
        return CovariatePackSpec(
            dataset_id="sdwpf_kddcup",
            stage=stage,
            pack_name=stage,
            feature_set="default",
            required_columns=_SDWPF_STAGE1_COLUMNS + _SDWPF_STAGE2_EXTRA_COLUMNS,
        )
    if stage == "stage3_regime":
        return CovariatePackSpec(
            dataset_id="sdwpf_kddcup",
            stage=stage,
            pack_name=stage,
            feature_set="default",
            required_columns=_SDWPF_STAGE1_COLUMNS + _SDWPF_STAGE2_EXTRA_COLUMNS + _SDWPF_STAGE3_EXTRA_COLUMNS,
        )
    raise ValueError(f"Unsupported covariate stage {stage!r}.")


def resolve_covariate_pack(dataset_id: str, stage: str) -> CovariatePackSpec:
    if stage not in DEFAULT_COVARIATE_STAGES:
        raise ValueError(f"Unsupported covariate stage {stage!r}. Expected one of {DEFAULT_COVARIATE_STAGES!r}.")
    if dataset_id in {"kelmarsh", "penmanshiel"}:
        return _greenbyte_pack(dataset_id, stage)
    if dataset_id == "hill_of_towie":
        return _hill_pack(stage)
    if dataset_id == "sdwpf_kddcup":
        return _sdwpf_pack(stage)
    raise ValueError(f"Unsupported dataset_id {dataset_id!r}.")


def reference_pack_for(dataset_id: str) -> CovariatePackSpec:
    feature_set = "lightweight" if dataset_id in {"kelmarsh", "penmanshiel"} else "default"
    return CovariatePackSpec(
        dataset_id=dataset_id,
        stage="reference",
        pack_name="power_only",
        feature_set=feature_set,
        required_columns=(),
    )


def iter_covariate_packs(
    dataset_ids: tuple[str, ...],
    stages: tuple[str, ...] = DEFAULT_COVARIATE_STAGES,
) -> tuple[CovariatePackSpec, ...]:
    return tuple(
        resolve_covariate_pack(dataset_id, stage)
        for dataset_id in dataset_ids
        for stage in stages
    )


__all__ = [
    "DEFAULT_COVARIATE_STAGES",
    "CovariatePackSpec",
    "iter_covariate_packs",
    "reference_pack_for",
    "resolve_covariate_pack",
]

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


TASK_BUNDLE_SCHEMA_VERSION = "task_bundle.v1"
DEFAULT_FEATURE_PROTOCOL_ID = "power_only"
BLOCKED_BY_UNSUPPORTED_FEATURE_PROTOCOL = "blocked_by_unsupported_feature_protocol"
_PI = 3.141592653589793
_WIND_DIRECTION_SIN_COLUMN = "wind_direction_sin"
_WIND_DIRECTION_COS_COLUMN = "wind_direction_cos"
_YAW_ERROR_SIN_COLUMN = "yaw_error_sin"
_YAW_ERROR_COS_COLUMN = "yaw_error_cos"
_PITCH_MEAN_COLUMN = "pitch_mean"
_MASK_COLUMN_SUFFIX = "__mask"
_MASK_POLARITY_UNAVAILABLE = "1_means_unavailable"
_MASK_VALID_VALUES = (0, 1)
_TARGET_HISTORY_MASK_COLUMN = "target_kw__mask"
_YAW_ERROR_CONVENTION = "yaw_error_degrees = wind_direction_degrees - nacelle_or_yaw_position_degrees"
_SDWPF_YAW_ERROR_NOTE = (
    "sdwpf_kddcup Wdir stores the documented relative yaw-error angle under the repository convention."
)
_SDWPF_WIND_DIRECTION_RECONSTRUCTION_NOTE = (
    "sdwpf_kddcup reconstructs absolute wind direction as Ndir + Wdir under the repository convention."
)

_SERIES_BASE_COLUMNS = (
    "dataset",
    "turbine_id",
    "timestamp",
    "target_kw",
    "is_observed",
    "quality_flags",
    "feature_quality_flags",
)
_SERIES_OPTIONAL_AUDIT_COLUMNS = (
    "farm_turbines_expected",
    "farm_turbines_observed",
    "farm_turbines_with_target",
    "farm_is_fully_synchronous",
    "farm_has_all_targets",
    "sdwpf_is_unknown",
    "sdwpf_is_abnormal",
    "sdwpf_is_masked",
)
_KNOWN_FUTURE_COLUMNS = (
    "dataset",
    "timestamp",
    "calendar_hour_sin",
    "calendar_hour_cos",
    "calendar_weekday_sin",
    "calendar_weekday_cos",
    "calendar_month_sin",
    "calendar_month_cos",
    "calendar_is_weekend",
)
_POWER_WS_HIST_COLUMNS_BY_DATASET = {
    "kelmarsh": ("Wind speed (m/s)",),
    "penmanshiel": ("Wind speed (m/s)",),
    "hill_of_towie": ("wtc_AcWindSp_mean",),
    "sdwpf_kddcup": ("Wspd",),
}
_POWER_ATEMP_HIST_COLUMN_BY_DATASET = {
    "kelmarsh": "Nacelle ambient temperature (°C)",
    "penmanshiel": "Nacelle ambient temperature (°C)",
    "hill_of_towie": "tur_temp__wtc_ambietmp_mean",
    "sdwpf_kddcup": "Etmp",
}
_POWER_ITEMP_HIST_COLUMN_BY_DATASET = {
    "kelmarsh": "Nacelle temperature (°C)",
    "penmanshiel": "Nacelle temperature (°C)",
    "hill_of_towie": "tur_temp__wtc_naceltmp_mean",
    "sdwpf_kddcup": "Itmp",
}
_GENERATOR_RPM_COLUMN_BY_DATASET = {
    "kelmarsh": "Generator RPM (RPM)",
    "penmanshiel": "Generator RPM (RPM)",
    "hill_of_towie": "wtc_GenRpm_mean",
}
_LOW_SPEED_ROTOR_RPM_COLUMN_BY_DATASET = {
    "kelmarsh": "Rotor speed (RPM)",
    "penmanshiel": "Rotor speed (RPM)",
    "hill_of_towie": "wtc_MainSRpm_mean",
}
_PITCH_COLUMNS_BY_DATASET = {
    "kelmarsh": (
        "Blade angle (pitch position) A (°)",
        "Blade angle (pitch position) B (°)",
        "Blade angle (pitch position) C (°)",
    ),
    "penmanshiel": (
        "Blade angle (pitch position) A (°)",
        "Blade angle (pitch position) B (°)",
        "Blade angle (pitch position) C (°)",
    ),
    "hill_of_towie": (
        "wtc_PitcPosA_mean",
        "wtc_PitcPosB_mean",
        "wtc_PitcPosC_mean",
    ),
    "sdwpf_kddcup": ("Pab1", "Pab2", "Pab3"),
}
_ABSOLUTE_WIND_DIRECTION_COLUMN_BY_DATASET = {
    "kelmarsh": "Wind direction (°)",
    "penmanshiel": "Wind direction (°)",
    "hill_of_towie": "wtc_ActualWindDirection_mean",
}
_NACELLE_OR_YAW_POSITION_COLUMN_BY_DATASET = {
    "kelmarsh": "Nacelle position (°)",
    "penmanshiel": "Nacelle position (°)",
    "hill_of_towie": "wtc_YawPos_mean",
    "sdwpf_kddcup": "Ndir",
}
_SDWPF_YAW_ERROR_COLUMN = "Wdir"
_WORLD_MODEL_LOCAL_EVENT_SUMMARY_COLUMNS = (
    "evt_any_active",
    "evt_active_count",
    "evt_total_overlap_seconds",
    "evt_stop_active",
    "evt_warning_active",
    "evt_informational_active",
)
_WORLD_MODEL_GLOBAL_OBSERVATION_COLUMNS = (
    "farm_pmu__gms_current_a",
    "farm_pmu__gms_power_kw",
    "farm_pmu__gms_reactive_power_kvar",
    "farm_evt_any_active",
    "farm_evt_active_count",
    "farm_evt_total_overlap_seconds",
    "farm_evt_stop_active",
    "farm_evt_warning_active",
    "farm_evt_informational_active",
)
_WORLD_MODEL_STATIC_COLUMNS = (
    "dataset",
    "turbine_id",
    "turbine_index",
    "latitude",
    "longitude",
    "coord_x",
    "coord_y",
    "coord_kind",
    "coord_crs",
    "elevation_m",
    "rated_power_kw",
    "hub_height_m",
    "rotor_diameter_m",
)
_WORLD_MODEL_PAIRWISE_COLUMNS = (
    "src_turbine_id",
    "dst_turbine_id",
    "src_turbine_index",
    "dst_turbine_index",
    "delta_x_m",
    "delta_y_m",
    "distance_m",
    "bearing_deg",
    "elevation_diff_m",
    "distance_in_rotor_diameters",
)
_UNSUPPORTED_DATASET_IDS_BY_PROTOCOL = {
    "power_wd_yaw_lrpm_hist_sincos": ("sdwpf_kddcup",),
    "power_wd_yaw_pmean_hist_sincos_masked": ("hill_of_towie", "sdwpf_kddcup"),
    "world_model_v1": ("hill_of_towie", "sdwpf_kddcup"),
}


@dataclass(frozen=True)
class FeatureProtocolSpec:
    feature_protocol_id: str
    display_name: str
    protocol_kind: str
    summary: str
    uses_target_history: bool
    uses_static_covariates: bool
    uses_known_future_covariates: bool
    uses_past_covariates: bool
    uses_target_derived_covariates: bool
    aliases: tuple[str, ...] = ()
    past_covariate_source: str | None = None
    past_covariate_stage: str | None = None


@dataclass(frozen=True)
class AngleTransformSpec:
    output_sin_column: str
    output_cos_column: str
    transform_kind: str
    source_columns: tuple[str, ...]
    description: str
    angle_convention: str | None = None


@dataclass(frozen=True)
class ScalarTransformSpec:
    output_column: str
    transform_kind: str
    source_columns: tuple[str, ...]
    description: str
    missing_value_policy: str | None = None


@dataclass(frozen=True)
class RawSourceMaskRuleSpec:
    source_columns: tuple[str, ...]
    rule_kind: str
    description: str
    affected_output_columns: tuple[str, ...] = ()
    minimum_allowed: float | None = None
    maximum_allowed: float | None = None


@dataclass(frozen=True)
class CompanionMaskPair:
    value_column: str
    mask_column: str


@dataclass(frozen=True)
class TaskSeriesSelection:
    feature_protocol_id: str
    source_columns: tuple[str, ...]
    all_columns: tuple[str, ...]
    target_history_mask_columns: tuple[str, ...]
    past_covariate_columns: tuple[str, ...]
    past_covariate_value_columns: tuple[str, ...]
    past_covariate_mask_columns: tuple[str, ...]
    known_future_columns: tuple[str, ...]
    static_columns: tuple[str, ...]
    target_derived_columns: tuple[str, ...]
    audit_columns: tuple[str, ...]
    local_observation_value_columns: tuple[str, ...] = ()
    local_observation_mask_columns: tuple[str, ...] = ()
    global_observation_value_columns: tuple[str, ...] = ()
    global_observation_mask_columns: tuple[str, ...] = ()
    pairwise_columns: tuple[str, ...] = ()
    derived_angle_transforms: tuple[AngleTransformSpec, ...] = ()
    derived_scalar_transforms: tuple[ScalarTransformSpec, ...] = ()
    raw_source_mask_rules: tuple[RawSourceMaskRuleSpec, ...] = ()
    target_history_mask_pairs: tuple[CompanionMaskPair, ...] = ()
    companion_mask_pairs: tuple[CompanionMaskPair, ...] = ()
    angle_convention: str | None = None
    dataset_specific_notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskSeriesMaterializationResult:
    series_frame: pl.DataFrame
    mask_hit_counts_by_column: dict[str, int]
    null_cause_counts_by_output_column: dict[str, dict[str, int]]


_POWER_ONLY_PROTOCOL = FeatureProtocolSpec(
    feature_protocol_id="power_only",
    display_name="Power Only",
    protocol_kind="reference",
    summary="Use only target power history as model input.",
    uses_target_history=True,
    uses_static_covariates=False,
    uses_known_future_covariates=False,
    uses_past_covariates=False,
    uses_target_derived_covariates=False,
    aliases=("reference",),
)
_POWER_WS_HIST_PROTOCOL = FeatureProtocolSpec(
    feature_protocol_id="power_ws_hist",
    display_name="Power + Wind Speed History",
    protocol_kind="past_covariate_ablation",
    summary="Use target power history plus dataset-native past wind-speed covariates.",
    uses_target_history=True,
    uses_static_covariates=False,
    uses_known_future_covariates=False,
    uses_past_covariates=True,
    uses_target_derived_covariates=False,
    aliases=("power_with_ws_history",),
    past_covariate_source="gold_base.dataset_native_wind_speed",
    past_covariate_stage="task_bundle.select",
)
_POWER_ATEMP_HIST_PROTOCOL = FeatureProtocolSpec(
    feature_protocol_id="power_atemp_hist",
    display_name="Power + Ambient Temperature History",
    protocol_kind="past_covariate_ablation",
    summary="Use target power history plus dataset-native past ambient-temperature covariates.",
    uses_target_history=True,
    uses_static_covariates=False,
    uses_known_future_covariates=False,
    uses_past_covariates=True,
    uses_target_derived_covariates=False,
    aliases=("power_with_atemp_history",),
    past_covariate_source="gold_base.dataset_native_ambient_temperature",
    past_covariate_stage="task_bundle.select",
)
_POWER_ITEMP_HIST_PROTOCOL = FeatureProtocolSpec(
    feature_protocol_id="power_itemp_hist",
    display_name="Power + Internal Temperature History",
    protocol_kind="past_covariate_ablation",
    summary="Use target power history plus dataset-native past internal-temperature covariates.",
    uses_target_history=True,
    uses_static_covariates=False,
    uses_known_future_covariates=False,
    uses_past_covariates=True,
    uses_target_derived_covariates=False,
    aliases=("power_with_itemp_history",),
    past_covariate_source="gold_base.dataset_native_internal_temperature",
    past_covariate_stage="task_bundle.select",
)
_POWER_WD_HIST_SINCOS_PROTOCOL = FeatureProtocolSpec(
    feature_protocol_id="power_wd_hist_sincos",
    display_name="Power + Wind Direction History (Sin/Cos)",
    protocol_kind="past_covariate_ablation",
    summary="Use target power history plus task-derived sine/cosine wind-direction covariates.",
    uses_target_history=True,
    uses_static_covariates=False,
    uses_known_future_covariates=False,
    uses_past_covariates=True,
    uses_target_derived_covariates=False,
    aliases=("power_with_wd_history_sincos",),
    past_covariate_source="task_bundle.derived_direction_angle",
    past_covariate_stage="task_bundle.angle_sincos",
)
_POWER_WS_WD_HIST_SINCOS_PROTOCOL = FeatureProtocolSpec(
    feature_protocol_id="power_ws_wd_hist_sincos",
    display_name="Power + Wind Speed/Direction History (Sin/Cos)",
    protocol_kind="past_covariate_combination",
    summary="Use target power history, dataset-native wind speed, and task-derived sine/cosine wind-direction covariates.",
    uses_target_history=True,
    uses_static_covariates=False,
    uses_known_future_covariates=False,
    uses_past_covariates=True,
    uses_target_derived_covariates=False,
    aliases=("power_with_ws_wd_history_sincos",),
    past_covariate_source="task_bundle.wind_speed_plus_direction_angle",
    past_covariate_stage="task_bundle.angle_sincos",
)
_POWER_WD_YAW_HIST_SINCOS_PROTOCOL = FeatureProtocolSpec(
    feature_protocol_id="power_wd_yaw_hist_sincos",
    display_name="Power + Wind Direction/Yaw Error History (Sin/Cos)",
    protocol_kind="past_covariate_combination",
    summary="Use target power history plus task-derived sine/cosine wind-direction and yaw-error covariates.",
    uses_target_history=True,
    uses_static_covariates=False,
    uses_known_future_covariates=False,
    uses_past_covariates=True,
    uses_target_derived_covariates=False,
    aliases=("power_with_wd_yaw_history_sincos",),
    past_covariate_source="task_bundle.wind_direction_and_yaw_error_angles",
    past_covariate_stage="task_bundle.angle_sincos",
)
_POWER_WD_YAW_PITCHMEAN_HIST_SINCOS_PROTOCOL = FeatureProtocolSpec(
    feature_protocol_id="power_wd_yaw_pitchmean_hist_sincos",
    display_name="Power + Wind Direction/Yaw Error/Pitch Mean History (Sin/Cos)",
    protocol_kind="past_covariate_combination",
    summary=(
        "Use target power history plus task-derived sine/cosine wind-direction and yaw-error covariates "
        "and task-derived pitch-mean history."
    ),
    uses_target_history=True,
    uses_static_covariates=False,
    uses_known_future_covariates=False,
    uses_past_covariates=True,
    uses_target_derived_covariates=False,
    aliases=("power_with_wd_yaw_pitchmean_history_sincos",),
    past_covariate_source="task_bundle.wind_direction_and_yaw_error_angles_plus_pitch_mean",
    past_covariate_stage="task_bundle.angle_sincos_plus_scalar_covariate",
)
_POWER_WD_YAW_PMEAN_HIST_SINCOS_MASKED_PROTOCOL = FeatureProtocolSpec(
    feature_protocol_id="power_wd_yaw_pmean_hist_sincos_masked",
    display_name="Power + Wind Direction/Yaw Error/Pitch Mean History (Sin/Cos) Masked",
    protocol_kind="past_covariate_combination",
    summary=(
        "Use target power history plus task-derived sine/cosine wind-direction and yaw-error covariates, "
        "task-derived pitch-mean history, a target-history mask channel, and companion mask channels "
        "where 1 marks an unavailable input."
    ),
    uses_target_history=True,
    uses_static_covariates=False,
    uses_known_future_covariates=False,
    uses_past_covariates=True,
    uses_target_derived_covariates=False,
    aliases=("power_with_wd_yaw_pitchmean_history_sincos_masked",),
    past_covariate_source=(
        "task_bundle.wind_direction_and_yaw_error_angles_plus_pitch_mean_with_target_and_companion_masks"
    ),
    past_covariate_stage="task_bundle.angle_sincos_plus_scalar_covariate_with_target_and_companion_masks",
)
_POWER_WD_YAW_LRPM_HIST_SINCOS_PROTOCOL = FeatureProtocolSpec(
    feature_protocol_id="power_wd_yaw_lrpm_hist_sincos",
    display_name="Power + Wind Direction/Yaw Error/LS RPM History (Sin/Cos)",
    protocol_kind="past_covariate_combination",
    summary=(
        "Use target power history plus task-derived sine/cosine wind-direction and yaw-error covariates "
        "and dataset-native low-speed rotor RPM history."
    ),
    uses_target_history=True,
    uses_static_covariates=False,
    uses_known_future_covariates=False,
    uses_past_covariates=True,
    uses_target_derived_covariates=False,
    aliases=("power_with_wd_yaw_lrpm_history_sincos",),
    past_covariate_source="task_bundle.wind_direction_and_yaw_error_angles_plus_low_speed_rotor_rpm",
    past_covariate_stage="task_bundle.angle_sincos_plus_native_covariate",
)
_WORLD_MODEL_V1_PROTOCOL = FeatureProtocolSpec(
    feature_protocol_id="world_model_v1",
    display_name="World Model V1",
    protocol_kind="world_model",
    summary=(
        "Use target history, masked local and farm-level observations, turbine static covariates, "
        "calendar known-future covariates, and directed pairwise geometry."
    ),
    uses_target_history=True,
    uses_static_covariates=True,
    uses_known_future_covariates=True,
    uses_past_covariates=True,
    uses_target_derived_covariates=False,
    past_covariate_source="task_bundle.world_model_local_and_global_observations",
    past_covariate_stage="task_bundle.world_model_v1",
)
_FEATURE_PROTOCOLS = (
    _POWER_ONLY_PROTOCOL,
    _POWER_WS_HIST_PROTOCOL,
    _POWER_ATEMP_HIST_PROTOCOL,
    _POWER_ITEMP_HIST_PROTOCOL,
    _POWER_WD_HIST_SINCOS_PROTOCOL,
    _POWER_WS_WD_HIST_SINCOS_PROTOCOL,
    _POWER_WD_YAW_HIST_SINCOS_PROTOCOL,
    _POWER_WD_YAW_PITCHMEAN_HIST_SINCOS_PROTOCOL,
    _POWER_WD_YAW_PMEAN_HIST_SINCOS_MASKED_PROTOCOL,
    _POWER_WD_YAW_LRPM_HIST_SINCOS_PROTOCOL,
    _WORLD_MODEL_V1_PROTOCOL,
)
_FEATURE_PROTOCOLS_BY_ID = {
    protocol.feature_protocol_id: protocol
    for protocol in _FEATURE_PROTOCOLS
}


def list_feature_protocol_ids() -> tuple[str, ...]:
    return tuple(protocol.feature_protocol_id for protocol in _FEATURE_PROTOCOLS)


def list_feature_protocol_specs() -> tuple[FeatureProtocolSpec, ...]:
    return _FEATURE_PROTOCOLS


def get_feature_protocol_spec(feature_protocol_id: str) -> FeatureProtocolSpec:
    try:
        return _FEATURE_PROTOCOLS_BY_ID[feature_protocol_id]
    except KeyError as exc:
        raise ValueError(f"Unknown feature_protocol_id {feature_protocol_id!r}.") from exc


def _unsupported_feature_protocol_message(
    *,
    dataset_id: str,
    feature_protocol_id: str,
) -> str | None:
    get_feature_protocol_spec(feature_protocol_id)
    unsupported_dataset_ids = _UNSUPPORTED_DATASET_IDS_BY_PROTOCOL.get(feature_protocol_id, ())
    if dataset_id in unsupported_dataset_ids:
        return f"feature_protocol_id {feature_protocol_id!r} is not supported for dataset {dataset_id!r}."
    return None


def validate_feature_protocol_for_dataset(
    *,
    dataset_id: str,
    feature_protocol_id: str,
) -> None:
    message = _unsupported_feature_protocol_message(
        dataset_id=dataset_id,
        feature_protocol_id=feature_protocol_id,
    )
    if message is not None:
        raise ValueError(message)


def feature_protocol_task_blocked_reason(
    *,
    dataset_id: str,
    feature_protocol_id: str,
) -> str | None:
    if _unsupported_feature_protocol_message(
        dataset_id=dataset_id,
        feature_protocol_id=feature_protocol_id,
    ) is not None:
        return BLOCKED_BY_UNSUPPORTED_FEATURE_PROTOCOL
    return None


def _require_mapping_value(
    mapping: dict[str, str],
    *,
    dataset_id: str,
    feature_protocol_id: str,
) -> str:
    try:
        return mapping[dataset_id]
    except KeyError as exc:
        raise ValueError(
            f"feature_protocol_id {feature_protocol_id!r} is not configured for dataset {dataset_id!r}."
        ) from exc


def _direct_angle_transform(
    *,
    output_sin_column: str,
    output_cos_column: str,
    source_column: str,
    description: str,
    angle_convention: str | None = None,
) -> AngleTransformSpec:
    return AngleTransformSpec(
        output_sin_column=output_sin_column,
        output_cos_column=output_cos_column,
        transform_kind="direct",
        source_columns=(source_column,),
        description=description,
        angle_convention=angle_convention,
    )


def _difference_angle_transform(
    *,
    output_sin_column: str,
    output_cos_column: str,
    minuend_column: str,
    subtrahend_column: str,
    description: str,
    angle_convention: str | None = None,
) -> AngleTransformSpec:
    return AngleTransformSpec(
        output_sin_column=output_sin_column,
        output_cos_column=output_cos_column,
        transform_kind="difference",
        source_columns=(minuend_column, subtrahend_column),
        description=description,
        angle_convention=angle_convention,
    )


def _sum_angle_transform(
    *,
    output_sin_column: str,
    output_cos_column: str,
    left_column: str,
    right_column: str,
    description: str,
    angle_convention: str | None = None,
) -> AngleTransformSpec:
    return AngleTransformSpec(
        output_sin_column=output_sin_column,
        output_cos_column=output_cos_column,
        transform_kind="sum",
        source_columns=(left_column, right_column),
        description=description,
        angle_convention=angle_convention,
    )


def _mask_column_name(value_column: str) -> str:
    return f"{value_column}{_MASK_COLUMN_SUFFIX}"


def _row_mean_scalar_transform(
    *,
    output_column: str,
    source_columns: tuple[str, ...],
    description: str,
    missing_value_policy: str | None = None,
) -> ScalarTransformSpec:
    return ScalarTransformSpec(
        output_column=output_column,
        transform_kind="row_mean",
        source_columns=source_columns,
        description=description,
        missing_value_policy=missing_value_policy,
    )


def _angle_transforms_for_protocol(
    *,
    dataset_id: str,
    feature_protocol_id: str,
) -> tuple[tuple[AngleTransformSpec, ...], str | None, tuple[str, ...]]:
    transforms: list[AngleTransformSpec] = []
    notes: list[str] = []
    angle_convention: str | None = None

    uses_wind_direction = feature_protocol_id in {
        _POWER_WD_HIST_SINCOS_PROTOCOL.feature_protocol_id,
        _POWER_WS_WD_HIST_SINCOS_PROTOCOL.feature_protocol_id,
        _POWER_WD_YAW_HIST_SINCOS_PROTOCOL.feature_protocol_id,
        _POWER_WD_YAW_PITCHMEAN_HIST_SINCOS_PROTOCOL.feature_protocol_id,
        _POWER_WD_YAW_PMEAN_HIST_SINCOS_MASKED_PROTOCOL.feature_protocol_id,
        _POWER_WD_YAW_LRPM_HIST_SINCOS_PROTOCOL.feature_protocol_id,
        _WORLD_MODEL_V1_PROTOCOL.feature_protocol_id,
    }
    uses_yaw_error = feature_protocol_id in {
        _POWER_WD_YAW_HIST_SINCOS_PROTOCOL.feature_protocol_id,
        _POWER_WD_YAW_PITCHMEAN_HIST_SINCOS_PROTOCOL.feature_protocol_id,
        _POWER_WD_YAW_PMEAN_HIST_SINCOS_MASKED_PROTOCOL.feature_protocol_id,
        _POWER_WD_YAW_LRPM_HIST_SINCOS_PROTOCOL.feature_protocol_id,
        _WORLD_MODEL_V1_PROTOCOL.feature_protocol_id,
    }

    if uses_wind_direction:
        if dataset_id == "sdwpf_kddcup":
            transforms.append(
                _sum_angle_transform(
                    output_sin_column=_WIND_DIRECTION_SIN_COLUMN,
                    output_cos_column=_WIND_DIRECTION_COS_COLUMN,
                    left_column=_require_mapping_value(
                        _NACELLE_OR_YAW_POSITION_COLUMN_BY_DATASET,
                        dataset_id=dataset_id,
                        feature_protocol_id=feature_protocol_id,
                    ),
                    right_column=_SDWPF_YAW_ERROR_COLUMN,
                    description="Reconstruct absolute wind direction from nacelle direction plus the documented relative Wdir angle.",
                    angle_convention=_YAW_ERROR_CONVENTION,
                )
            )
            notes.append(_SDWPF_WIND_DIRECTION_RECONSTRUCTION_NOTE)
            notes.append(_SDWPF_YAW_ERROR_NOTE)
            angle_convention = _YAW_ERROR_CONVENTION
        else:
            wind_direction_column = _require_mapping_value(
                _ABSOLUTE_WIND_DIRECTION_COLUMN_BY_DATASET,
                dataset_id=dataset_id,
                feature_protocol_id=feature_protocol_id,
            )
            transforms.append(
                _direct_angle_transform(
                    output_sin_column=_WIND_DIRECTION_SIN_COLUMN,
                    output_cos_column=_WIND_DIRECTION_COS_COLUMN,
                    source_column=wind_direction_column,
                    description="Encode the dataset-native absolute wind direction as sine/cosine covariates.",
                )
            )

    if uses_yaw_error:
        if dataset_id == "sdwpf_kddcup":
            transforms.append(
                _direct_angle_transform(
                    output_sin_column=_YAW_ERROR_SIN_COLUMN,
                    output_cos_column=_YAW_ERROR_COS_COLUMN,
                    source_column=_SDWPF_YAW_ERROR_COLUMN,
                    description="Encode the documented relative Wdir angle directly as yaw error sine/cosine covariates.",
                    angle_convention=_YAW_ERROR_CONVENTION,
                )
            )
            if _SDWPF_YAW_ERROR_NOTE not in notes:
                notes.append(_SDWPF_YAW_ERROR_NOTE)
            angle_convention = _YAW_ERROR_CONVENTION
        else:
            wind_direction_column = _require_mapping_value(
                _ABSOLUTE_WIND_DIRECTION_COLUMN_BY_DATASET,
                dataset_id=dataset_id,
                feature_protocol_id=feature_protocol_id,
            )
            nacelle_or_yaw_column = _require_mapping_value(
                _NACELLE_OR_YAW_POSITION_COLUMN_BY_DATASET,
                dataset_id=dataset_id,
                feature_protocol_id=feature_protocol_id,
            )
            transforms.append(
                _difference_angle_transform(
                    output_sin_column=_YAW_ERROR_SIN_COLUMN,
                    output_cos_column=_YAW_ERROR_COS_COLUMN,
                    minuend_column=wind_direction_column,
                    subtrahend_column=nacelle_or_yaw_column,
                    description="Compute yaw error from absolute wind direction minus nacelle/yaw position, then encode as sine/cosine.",
                    angle_convention=_YAW_ERROR_CONVENTION,
                )
            )
            angle_convention = _YAW_ERROR_CONVENTION

    return tuple(transforms), angle_convention, tuple(dict.fromkeys(notes))


def _scalar_transforms_for_protocol(
    *,
    dataset_id: str,
    feature_protocol_id: str,
) -> tuple[ScalarTransformSpec, ...]:
    if feature_protocol_id not in {
        _POWER_WD_YAW_PITCHMEAN_HIST_SINCOS_PROTOCOL.feature_protocol_id,
        _POWER_WD_YAW_PMEAN_HIST_SINCOS_MASKED_PROTOCOL.feature_protocol_id,
        _WORLD_MODEL_V1_PROTOCOL.feature_protocol_id,
    }:
        return ()
    try:
        pitch_columns = _PITCH_COLUMNS_BY_DATASET[dataset_id]
    except KeyError as exc:
        raise ValueError(
            f"feature_protocol_id {feature_protocol_id!r} is not configured for dataset {dataset_id!r}."
        ) from exc
    return (
        _row_mean_scalar_transform(
            output_column=_PITCH_MEAN_COLUMN,
            source_columns=pitch_columns,
            description="Compute the arithmetic mean of the three blade-pitch angles when all three inputs are present.",
            missing_value_policy="all_sources_required",
        ),
    )


def _raw_source_mask_rules_for_protocol(
    *,
    dataset_id: str,
    feature_protocol_id: str,
) -> tuple[RawSourceMaskRuleSpec, ...]:
    if feature_protocol_id not in {
        _POWER_WD_YAW_PMEAN_HIST_SINCOS_MASKED_PROTOCOL.feature_protocol_id,
        _WORLD_MODEL_V1_PROTOCOL.feature_protocol_id,
    }:
        return ()
    try:
        pitch_columns = _PITCH_COLUMNS_BY_DATASET[dataset_id]
    except KeyError as exc:
        raise ValueError(
            f"feature_protocol_id {feature_protocol_id!r} is not configured for dataset {dataset_id!r}."
        ) from exc
    return (
        RawSourceMaskRuleSpec(
            source_columns=pitch_columns,
            rule_kind="outside_closed_interval",
            description=(
                "Mask raw blade-pitch observations before pitch_mean derivation when the value falls outside "
                "the closed interval [-10, 95] degrees."
            ),
            affected_output_columns=(_PITCH_MEAN_COLUMN,),
            minimum_allowed=-10.0,
            maximum_allowed=95.0,
        ),
    )


def _companion_mask_pairs_for_protocol(
    *,
    feature_protocol_id: str,
    past_covariate_value_columns: tuple[str, ...],
) -> tuple[CompanionMaskPair, ...]:
    if feature_protocol_id not in {
        _POWER_WD_YAW_PMEAN_HIST_SINCOS_MASKED_PROTOCOL.feature_protocol_id,
        _WORLD_MODEL_V1_PROTOCOL.feature_protocol_id,
    }:
        return ()
    return tuple(
        CompanionMaskPair(
            value_column=value_column,
            mask_column=_mask_column_name(value_column),
        )
        for value_column in past_covariate_value_columns
    )


def _target_history_mask_pairs_for_protocol(
    *,
    feature_protocol_id: str,
) -> tuple[CompanionMaskPair, ...]:
    if feature_protocol_id not in {
        _POWER_WD_YAW_PMEAN_HIST_SINCOS_MASKED_PROTOCOL.feature_protocol_id,
        _WORLD_MODEL_V1_PROTOCOL.feature_protocol_id,
    }:
        return ()
    return (
        CompanionMaskPair(
            value_column="target_kw",
            mask_column=_TARGET_HISTORY_MASK_COLUMN,
        ),
    )


def _dataset_native_columns_for_protocol(
    *,
    dataset_id: str,
    available_columns: set[str],
    feature_protocol_id: str,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[AngleTransformSpec, ...],
    tuple[ScalarTransformSpec, ...],
    str | None,
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    if feature_protocol_id == _WORLD_MODEL_V1_PROTOCOL.feature_protocol_id:
        wind_speed_column = _require_mapping_value(
            {key: value[0] for key, value in _POWER_WS_HIST_COLUMNS_BY_DATASET.items()},
            dataset_id=dataset_id,
            feature_protocol_id=feature_protocol_id,
        )
        rotor_rpm_column = _require_mapping_value(
            _LOW_SPEED_ROTOR_RPM_COLUMN_BY_DATASET,
            dataset_id=dataset_id,
            feature_protocol_id=feature_protocol_id,
        )
        generator_rpm_column = _require_mapping_value(
            _GENERATOR_RPM_COLUMN_BY_DATASET,
            dataset_id=dataset_id,
            feature_protocol_id=feature_protocol_id,
        )
        ambient_temperature_column = _require_mapping_value(
            _POWER_ATEMP_HIST_COLUMN_BY_DATASET,
            dataset_id=dataset_id,
            feature_protocol_id=feature_protocol_id,
        )
        internal_temperature_column = _require_mapping_value(
            _POWER_ITEMP_HIST_COLUMN_BY_DATASET,
            dataset_id=dataset_id,
            feature_protocol_id=feature_protocol_id,
        )
        derived_angle_transforms, angle_convention, dataset_specific_notes = _angle_transforms_for_protocol(
            dataset_id=dataset_id,
            feature_protocol_id=feature_protocol_id,
        )
        derived_scalar_transforms = _scalar_transforms_for_protocol(
            dataset_id=dataset_id,
            feature_protocol_id=feature_protocol_id,
        )
        configured_columns: list[str] = [wind_speed_column]
        output_past_covariate_columns: list[str] = [wind_speed_column]
        local_observation_value_columns: list[str] = [wind_speed_column]
        global_observation_value_columns = list(_WORLD_MODEL_GLOBAL_OBSERVATION_COLUMNS)

        for transform in derived_angle_transforms:
            configured_columns.extend(transform.source_columns)
            output_past_covariate_columns.extend((transform.output_sin_column, transform.output_cos_column))
            local_observation_value_columns.extend((transform.output_sin_column, transform.output_cos_column))

        for transform in derived_scalar_transforms:
            configured_columns.extend(transform.source_columns)
            output_past_covariate_columns.append(transform.output_column)
            local_observation_value_columns.append(transform.output_column)

        for column in (
            rotor_rpm_column,
            generator_rpm_column,
            ambient_temperature_column,
            internal_temperature_column,
            *_WORLD_MODEL_LOCAL_EVENT_SUMMARY_COLUMNS,
            *_WORLD_MODEL_GLOBAL_OBSERVATION_COLUMNS,
        ):
            configured_columns.append(column)
            output_past_covariate_columns.append(column)

        local_observation_value_columns.extend(
            (
                rotor_rpm_column,
                generator_rpm_column,
                ambient_temperature_column,
                internal_temperature_column,
                *_WORLD_MODEL_LOCAL_EVENT_SUMMARY_COLUMNS,
            )
        )
        configured_columns = list(dict.fromkeys(configured_columns))
        missing_columns = [column for column in configured_columns if column not in available_columns]
        if missing_columns:
            raise ValueError(
                f"feature_protocol_id {feature_protocol_id!r} for dataset {dataset_id!r} requires "
                f"missing gold-base columns {missing_columns!r}."
            )
        return (
            tuple(configured_columns),
            tuple(output_past_covariate_columns),
            derived_angle_transforms,
            derived_scalar_transforms,
            angle_convention,
            dataset_specific_notes,
            tuple(local_observation_value_columns),
            tuple(global_observation_value_columns),
        )

    uses_wind_speed = feature_protocol_id in {
        _POWER_WS_HIST_PROTOCOL.feature_protocol_id,
        _POWER_WS_WD_HIST_SINCOS_PROTOCOL.feature_protocol_id,
    }
    uses_ambient_temperature = feature_protocol_id == _POWER_ATEMP_HIST_PROTOCOL.feature_protocol_id
    uses_internal_temperature = feature_protocol_id == _POWER_ITEMP_HIST_PROTOCOL.feature_protocol_id
    uses_low_speed_rotor_rpm = feature_protocol_id == _POWER_WD_YAW_LRPM_HIST_SINCOS_PROTOCOL.feature_protocol_id
    configured_columns: list[str] = []
    output_past_covariate_columns: list[str] = []

    if uses_wind_speed:
        try:
            wind_speed_columns = _POWER_WS_HIST_COLUMNS_BY_DATASET[dataset_id]
        except KeyError as exc:
            raise ValueError(
                f"feature_protocol_id {feature_protocol_id!r} is not configured for dataset {dataset_id!r}."
            ) from exc
        configured_columns.extend(wind_speed_columns)
        output_past_covariate_columns.extend(wind_speed_columns)

    if uses_ambient_temperature:
        ambient_temperature_column = _require_mapping_value(
            _POWER_ATEMP_HIST_COLUMN_BY_DATASET,
            dataset_id=dataset_id,
            feature_protocol_id=feature_protocol_id,
        )
        configured_columns.append(ambient_temperature_column)
        output_past_covariate_columns.append(ambient_temperature_column)

    if uses_internal_temperature:
        internal_temperature_column = _require_mapping_value(
            _POWER_ITEMP_HIST_COLUMN_BY_DATASET,
            dataset_id=dataset_id,
            feature_protocol_id=feature_protocol_id,
        )
        configured_columns.append(internal_temperature_column)
        output_past_covariate_columns.append(internal_temperature_column)

    derived_angle_transforms, angle_convention, dataset_specific_notes = _angle_transforms_for_protocol(
        dataset_id=dataset_id,
        feature_protocol_id=feature_protocol_id,
    )
    for transform in derived_angle_transforms:
        configured_columns.extend(transform.source_columns)
        output_past_covariate_columns.extend((transform.output_sin_column, transform.output_cos_column))

    derived_scalar_transforms = _scalar_transforms_for_protocol(
        dataset_id=dataset_id,
        feature_protocol_id=feature_protocol_id,
    )
    for transform in derived_scalar_transforms:
        configured_columns.extend(transform.source_columns)
        output_past_covariate_columns.append(transform.output_column)

    if uses_low_speed_rotor_rpm:
        configured_columns.append(
            _require_mapping_value(
                _LOW_SPEED_ROTOR_RPM_COLUMN_BY_DATASET,
                dataset_id=dataset_id,
                feature_protocol_id=feature_protocol_id,
            )
        )
        output_past_covariate_columns.append(configured_columns[-1])

    configured_columns = list(dict.fromkeys(configured_columns))
    missing_columns = [column for column in configured_columns if column not in available_columns]
    if missing_columns:
        raise ValueError(
            f"feature_protocol_id {feature_protocol_id!r} for dataset {dataset_id!r} requires "
            f"missing gold-base columns {missing_columns!r}."
        )
    return (
        tuple(configured_columns),
        tuple(output_past_covariate_columns),
        derived_angle_transforms,
        derived_scalar_transforms,
        angle_convention,
        dataset_specific_notes,
        (),
        (),
    )


def _angle_degrees_expr(transform: AngleTransformSpec) -> pl.Expr:
    source_exprs = [pl.col(column).cast(pl.Float64, strict=False) for column in transform.source_columns]
    if transform.transform_kind == "direct":
        return source_exprs[0]
    if transform.transform_kind == "difference":
        return source_exprs[0] - source_exprs[1]
    if transform.transform_kind == "sum":
        return source_exprs[0] + source_exprs[1]
    raise ValueError(f"Unsupported angle transform kind {transform.transform_kind!r}.")


def _rule_hit_expr_for_column(
    *,
    rule: RawSourceMaskRuleSpec,
    source_column: str,
) -> pl.Expr:
    if rule.rule_kind != "outside_closed_interval":
        raise ValueError(f"Unsupported raw source mask rule kind {rule.rule_kind!r}.")
    source_expr = pl.col(source_column).cast(pl.Float64, strict=False)
    invalid_expr: pl.Expr | None = None
    if rule.minimum_allowed is not None:
        invalid_expr = source_expr < pl.lit(rule.minimum_allowed)
    if rule.maximum_allowed is not None:
        upper_expr = source_expr > pl.lit(rule.maximum_allowed)
        invalid_expr = upper_expr if invalid_expr is None else (invalid_expr | upper_expr)
    if invalid_expr is None:
        raise ValueError("Raw source mask rule must define at least one numeric bound.")
    return pl.col(source_column).is_not_null() & invalid_expr


def _combine_exprs_with_or(exprs: tuple[pl.Expr, ...]) -> pl.Expr:
    combined = exprs[0]
    for expr in exprs[1:]:
        combined = combined | expr
    return combined


def _apply_protocol_scoped_raw_source_masks(
    source_frame: pl.DataFrame,
    *,
    selection: TaskSeriesSelection,
) -> tuple[pl.DataFrame, dict[str, dict[str, int]]]:
    if not selection.raw_source_mask_rules:
        return source_frame, {}

    schema = source_frame.schema
    replacement_exprs: list[pl.Expr] = []
    null_cause_counts_by_output_column: dict[str, dict[str, int]] = {}
    combined_rule_exprs_by_column: dict[str, list[pl.Expr]] = {}

    for rule in selection.raw_source_mask_rules:
        column_rule_exprs = tuple(
            _rule_hit_expr_for_column(rule=rule, source_column=source_column)
            for source_column in rule.source_columns
        )
        rule_hit_expr = pl.any_horizontal(list(column_rule_exprs))
        original_missing_expr = pl.any_horizontal([pl.col(source_column).is_null() for source_column in rule.source_columns])
        diagnostics = source_frame.select(
            rule_hit_expr.cast(pl.Int64).sum().alias("raw_rule_rows"),
            ((~rule_hit_expr) & original_missing_expr).cast(pl.Int64).sum().alias("raw_missing_rows"),
        ).row(0, named=True)
        for output_column in rule.affected_output_columns:
            if output_column in null_cause_counts_by_output_column:
                raise ValueError(
                    f"Multiple raw source mask rules attempted to publish diagnostics for output column {output_column!r}."
                )
            null_cause_counts_by_output_column[output_column] = {
                "raw_rule_rows": int(diagnostics["raw_rule_rows"] or 0),
                "raw_missing_rows": int(diagnostics["raw_missing_rows"] or 0),
            }
        for source_column, expr in zip(rule.source_columns, column_rule_exprs, strict=True):
            combined_rule_exprs_by_column.setdefault(source_column, []).append(expr)

    for source_column, exprs in combined_rule_exprs_by_column.items():
        combined_expr = _combine_exprs_with_or(tuple(exprs)) if len(exprs) > 1 else exprs[0]
        replacement_exprs.append(
            pl.when(combined_expr)
            .then(pl.lit(None, dtype=schema[source_column]))
            .otherwise(pl.col(source_column))
            .alias(source_column)
        )

    return source_frame.with_columns(*replacement_exprs), null_cause_counts_by_output_column


def _validate_mask_columns(
    series_frame: pl.DataFrame,
    *,
    mask_pairs: tuple[CompanionMaskPair, ...],
) -> None:
    valid_values = set(_MASK_VALID_VALUES)
    for pair in mask_pairs:
        mask_series = series_frame.get_column(pair.mask_column)
        if mask_series.dtype != pl.Int8:
            raise ValueError(
                f"Mask column {pair.mask_column!r} must have dtype Int8, found {mask_series.dtype!r}."
            )
        if mask_series.null_count() != 0:
            raise ValueError(f"Mask column {pair.mask_column!r} must not contain null values.")
        unique_values = set(mask_series.unique().to_list())
        if not unique_values.issubset(valid_values):
            raise ValueError(
                f"Mask column {pair.mask_column!r} must only contain {sorted(valid_values)!r}, found {sorted(unique_values)!r}."
            )
        mismatch_count = series_frame.select(
            (pl.col(pair.mask_column) != pl.col(pair.value_column).is_null().cast(pl.Int8))
            .cast(pl.Int64)
            .sum()
            .alias("mismatch_count")
        )["mismatch_count"][0]
        if int(mismatch_count or 0) != 0:
            raise ValueError(
                f"Mask column {pair.mask_column!r} must exactly match nullness of value column {pair.value_column!r}."
            )


def _append_mask_columns(
    series_frame: pl.DataFrame,
    *,
    mask_pairs: tuple[CompanionMaskPair, ...],
) -> tuple[pl.DataFrame, dict[str, int]]:
    if not mask_pairs:
        return series_frame, {}

    with_masks = series_frame.with_columns(
        *[
            pl.col(pair.value_column).is_null().cast(pl.Int8).alias(pair.mask_column)
            for pair in mask_pairs
        ]
    )
    _validate_mask_columns(with_masks, mask_pairs=mask_pairs)
    diagnostics_row = with_masks.select(
        *[
            pl.col(pair.mask_column).cast(pl.Int64).sum().alias(pair.mask_column)
            for pair in mask_pairs
        ]
    ).row(0, named=True)
    return with_masks, {column: int(count or 0) for column, count in diagnostics_row.items()}


def materialize_task_series(
    source_frame: pl.DataFrame,
    *,
    selection: TaskSeriesSelection,
) -> TaskSeriesMaterializationResult:
    series_frame, null_cause_counts_by_output_column = _apply_protocol_scoped_raw_source_masks(
        source_frame,
        selection=selection,
    )
    derived_columns: list[pl.Expr] = []
    for transform in selection.derived_angle_transforms:
        angle_radians = _angle_degrees_expr(transform) * pl.lit(_PI / 180.0)
        derived_columns.extend(
            (
                angle_radians.sin().alias(transform.output_sin_column),
                angle_radians.cos().alias(transform.output_cos_column),
            )
        )
    for transform in selection.derived_scalar_transforms:
        source_exprs = [pl.col(column).cast(pl.Float64, strict=False) for column in transform.source_columns]
        if transform.transform_kind == "row_mean":
            derived_columns.append(
                pl.when(pl.all_horizontal([expr.is_not_null() for expr in source_exprs]))
                .then(pl.sum_horizontal(source_exprs) / pl.lit(float(len(source_exprs))))
                .otherwise(pl.lit(None, dtype=pl.Float64))
                .alias(transform.output_column)
            )
            continue
        raise ValueError(f"Unsupported scalar transform kind {transform.transform_kind!r}.")
    if derived_columns:
        series_frame = series_frame.with_columns(*derived_columns)
    series_frame, target_history_mask_hit_counts = _append_mask_columns(
        series_frame,
        mask_pairs=selection.target_history_mask_pairs,
    )
    series_frame, past_covariate_mask_hit_counts = _append_mask_columns(
        series_frame,
        mask_pairs=selection.companion_mask_pairs,
    )
    return TaskSeriesMaterializationResult(
        series_frame=series_frame.select(list(selection.all_columns)),
        mask_hit_counts_by_column={
            **target_history_mask_hit_counts,
            **past_covariate_mask_hit_counts,
        },
        null_cause_counts_by_output_column=null_cause_counts_by_output_column,
    )


def materialize_task_series_frame(
    source_frame: pl.DataFrame,
    *,
    selection: TaskSeriesSelection,
) -> pl.DataFrame:
    return materialize_task_series(
        source_frame,
        selection=selection,
    ).series_frame


def _known_future_columns_for_protocol(feature_protocol_id: str) -> tuple[str, ...]:
    if feature_protocol_id == _WORLD_MODEL_V1_PROTOCOL.feature_protocol_id:
        return _KNOWN_FUTURE_COLUMNS
    return ("dataset", "timestamp")


def _static_columns_for_protocol(feature_protocol_id: str) -> tuple[str, ...]:
    if feature_protocol_id == _WORLD_MODEL_V1_PROTOCOL.feature_protocol_id:
        return _WORLD_MODEL_STATIC_COLUMNS
    return ()


def _pairwise_columns_for_protocol(feature_protocol_id: str) -> tuple[str, ...]:
    if feature_protocol_id == _WORLD_MODEL_V1_PROTOCOL.feature_protocol_id:
        return _WORLD_MODEL_PAIRWISE_COLUMNS
    return ()


def select_task_series_columns(
    *,
    dataset_id: str,
    available_columns: set[str],
    feature_protocol_id: str,
    turbine_static_columns: set[str] | None = None,
) -> TaskSeriesSelection:
    del turbine_static_columns
    validate_feature_protocol_for_dataset(
        dataset_id=dataset_id,
        feature_protocol_id=feature_protocol_id,
    )
    (
        source_past_covariate_columns,
        past_covariate_value_columns,
        derived_angle_transforms,
        derived_scalar_transforms,
        angle_convention,
        dataset_specific_notes,
        local_observation_value_columns,
        global_observation_value_columns,
    ) = _dataset_native_columns_for_protocol(
        dataset_id=dataset_id,
        available_columns=available_columns,
        feature_protocol_id=feature_protocol_id,
    )
    raw_source_mask_rules = _raw_source_mask_rules_for_protocol(
        dataset_id=dataset_id,
        feature_protocol_id=feature_protocol_id,
    )
    target_history_mask_pairs = _target_history_mask_pairs_for_protocol(
        feature_protocol_id=feature_protocol_id,
    )
    target_history_mask_columns = tuple(pair.mask_column for pair in target_history_mask_pairs)
    companion_mask_pairs = _companion_mask_pairs_for_protocol(
        feature_protocol_id=feature_protocol_id,
        past_covariate_value_columns=past_covariate_value_columns,
    )
    past_covariate_mask_columns = tuple(pair.mask_column for pair in companion_mask_pairs)
    local_observation_mask_columns = tuple(
        pair.mask_column
        for pair in companion_mask_pairs
        if pair.value_column in local_observation_value_columns
    )
    global_observation_mask_columns = tuple(
        pair.mask_column
        for pair in companion_mask_pairs
        if pair.value_column in global_observation_value_columns
    )
    past_covariate_columns = tuple((*past_covariate_value_columns, *past_covariate_mask_columns))
    audit_columns = tuple(column for column in _SERIES_OPTIONAL_AUDIT_COLUMNS if column in available_columns)
    source_columns = tuple((*_SERIES_BASE_COLUMNS, *source_past_covariate_columns, *audit_columns))
    return TaskSeriesSelection(
        feature_protocol_id=feature_protocol_id,
        source_columns=source_columns,
        all_columns=tuple(
            (
                *_SERIES_BASE_COLUMNS,
                *target_history_mask_columns,
                *past_covariate_value_columns,
                *past_covariate_mask_columns,
                *audit_columns,
            )
        ),
        target_history_mask_columns=target_history_mask_columns,
        past_covariate_columns=past_covariate_columns,
        past_covariate_value_columns=past_covariate_value_columns,
        past_covariate_mask_columns=past_covariate_mask_columns,
        known_future_columns=_known_future_columns_for_protocol(feature_protocol_id),
        static_columns=_static_columns_for_protocol(feature_protocol_id),
        target_derived_columns=(),
        audit_columns=audit_columns,
        local_observation_value_columns=local_observation_value_columns,
        local_observation_mask_columns=local_observation_mask_columns,
        global_observation_value_columns=global_observation_value_columns,
        global_observation_mask_columns=global_observation_mask_columns,
        pairwise_columns=_pairwise_columns_for_protocol(feature_protocol_id),
        derived_angle_transforms=derived_angle_transforms,
        derived_scalar_transforms=derived_scalar_transforms,
        raw_source_mask_rules=raw_source_mask_rules,
        target_history_mask_pairs=target_history_mask_pairs,
        companion_mask_pairs=companion_mask_pairs,
        angle_convention=angle_convention,
        dataset_specific_notes=dataset_specific_notes,
    )


def build_known_future_frame(series_frame: pl.DataFrame) -> pl.DataFrame:
    if series_frame.is_empty():
        return pl.DataFrame(
            schema={column: pl.Float64 if column != "dataset" else pl.String for column in _KNOWN_FUTURE_COLUMNS}
        )
    timestamps = (
        series_frame
        .select("dataset", "timestamp")
        .unique()
        .sort("timestamp")
        .with_columns(
            ((pl.col("timestamp").dt.hour() * 60) + pl.col("timestamp").dt.minute()).alias("__minute_of_day"),
            pl.col("timestamp").dt.weekday().alias("__weekday"),
            pl.col("timestamp").dt.month().alias("__month"),
        )
        .with_columns(
            (2 * pl.lit(_PI) * pl.col("__minute_of_day") / pl.lit(24 * 60)).sin().alias("calendar_hour_sin"),
            (2 * pl.lit(_PI) * pl.col("__minute_of_day") / pl.lit(24 * 60)).cos().alias("calendar_hour_cos"),
            (2 * pl.lit(_PI) * pl.col("__weekday") / pl.lit(7)).sin().alias("calendar_weekday_sin"),
            (2 * pl.lit(_PI) * pl.col("__weekday") / pl.lit(7)).cos().alias("calendar_weekday_cos"),
            (2 * pl.lit(_PI) * (pl.col("__month") - 1) / pl.lit(12)).sin().alias("calendar_month_sin"),
            (2 * pl.lit(_PI) * (pl.col("__month") - 1) / pl.lit(12)).cos().alias("calendar_month_cos"),
            (pl.col("__weekday") >= 5).cast(pl.Int8).alias("calendar_is_weekend"),
        )
        .select(list(_KNOWN_FUTURE_COLUMNS))
    )
    return timestamps


def _angle_transform_context_dict(transform: AngleTransformSpec) -> dict[str, object]:
    return {
        "output_sin_column": transform.output_sin_column,
        "output_cos_column": transform.output_cos_column,
        "transform_kind": transform.transform_kind,
        "source_columns": list(transform.source_columns),
        "description": transform.description,
        "angle_convention": transform.angle_convention,
    }


def _scalar_transform_context_dict(transform: ScalarTransformSpec) -> dict[str, object]:
    return {
        "output_column": transform.output_column,
        "transform_kind": transform.transform_kind,
        "source_columns": list(transform.source_columns),
        "description": transform.description,
        "missing_value_policy": transform.missing_value_policy,
    }


def _raw_source_mask_rule_context_dict(rule: RawSourceMaskRuleSpec) -> dict[str, object]:
    return {
        "source_columns": list(rule.source_columns),
        "rule_kind": rule.rule_kind,
        "description": rule.description,
        "affected_output_columns": list(rule.affected_output_columns),
        "minimum_allowed": rule.minimum_allowed,
        "maximum_allowed": rule.maximum_allowed,
    }


def _companion_mask_pair_context_dict(pair: CompanionMaskPair) -> dict[str, object]:
    return {
        "value_column": pair.value_column,
        "mask_column": pair.mask_column,
    }


def protocol_context_dict(
    *,
    dataset_id: str,
    task: dict[str, object],
    feature_protocol_id: str,
    turbine_ids: tuple[str, ...],
    selection: TaskSeriesSelection,
    static_columns: tuple[str, ...],
) -> dict[str, object]:
    protocol = get_feature_protocol_spec(feature_protocol_id)
    has_mask_columns = bool(selection.target_history_mask_pairs or selection.companion_mask_pairs)
    return {
        "schema_version": TASK_BUNDLE_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "feature_protocol_id": feature_protocol_id,
        "feature_protocol": {
            "feature_protocol_id": protocol.feature_protocol_id,
            "display_name": protocol.display_name,
            "protocol_kind": protocol.protocol_kind,
            "summary": protocol.summary,
            "uses_target_history": protocol.uses_target_history,
            "uses_static_covariates": protocol.uses_static_covariates,
            "uses_known_future_covariates": protocol.uses_known_future_covariates,
            "uses_past_covariates": protocol.uses_past_covariates,
            "uses_target_derived_covariates": protocol.uses_target_derived_covariates,
            "aliases": list(protocol.aliases),
            "past_covariate_source": protocol.past_covariate_source,
            "past_covariate_stage": protocol.past_covariate_stage,
            "angle_convention": selection.angle_convention,
            "dataset_specific_notes": list(selection.dataset_specific_notes),
            "derived_angle_features": [
                _angle_transform_context_dict(transform)
                for transform in selection.derived_angle_transforms
            ],
            "derived_scalar_features": [
                _scalar_transform_context_dict(transform)
                for transform in selection.derived_scalar_transforms
            ],
            "mask_polarity": _MASK_POLARITY_UNAVAILABLE if has_mask_columns else None,
            "mask_dtype": "int8" if has_mask_columns else None,
            "mask_valid_values": list(_MASK_VALID_VALUES) if has_mask_columns else [],
            "raw_source_mask_rules": [
                _raw_source_mask_rule_context_dict(rule)
                for rule in selection.raw_source_mask_rules
            ],
            "companion_mask_pairs": [
                _companion_mask_pair_context_dict(pair)
                for pair in selection.companion_mask_pairs
            ],
        },
        "task": task,
        "turbine_ids": list(turbine_ids),
        "time_axis_semantics": "farm_synchronous_long_panel",
        "column_groups": {
            "series": list(selection.all_columns),
            "target_history_masks": list(selection.target_history_mask_columns),
            "past_covariates": list(selection.past_covariate_columns),
            "past_covariate_values": list(selection.past_covariate_value_columns),
            "past_covariate_masks": list(selection.past_covariate_mask_columns),
            "local_observation_values": list(selection.local_observation_value_columns),
            "local_observation_masks": list(selection.local_observation_mask_columns),
            "global_observation_values": list(selection.global_observation_value_columns),
            "global_observation_masks": list(selection.global_observation_mask_columns),
            "target_derived_covariates": list(selection.target_derived_columns),
            "known_future": list(selection.known_future_columns),
            "static": list(static_columns),
            "pairwise": list(selection.pairwise_columns),
            "audit": list(selection.audit_columns),
        },
    }


__all__ = [
    "BLOCKED_BY_UNSUPPORTED_FEATURE_PROTOCOL",
    "DEFAULT_FEATURE_PROTOCOL_ID",
    "TASK_BUNDLE_SCHEMA_VERSION",
    "AngleTransformSpec",
    "CompanionMaskPair",
    "FeatureProtocolSpec",
    "RawSourceMaskRuleSpec",
    "ScalarTransformSpec",
    "TaskSeriesSelection",
    "TaskSeriesMaterializationResult",
    "build_known_future_frame",
    "feature_protocol_task_blocked_reason",
    "get_feature_protocol_spec",
    "list_feature_protocol_ids",
    "list_feature_protocol_specs",
    "materialize_task_series",
    "materialize_task_series_frame",
    "protocol_context_dict",
    "select_task_series_columns",
    "validate_feature_protocol_for_dataset",
]

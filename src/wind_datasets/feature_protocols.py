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
_LOW_SPEED_ROTOR_RPM_COLUMN_BY_DATASET = {
    "kelmarsh": "Rotor speed (RPM)",
    "penmanshiel": "Rotor speed (RPM)",
    "hill_of_towie": "wtc_MainSRpm_mean",
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
_UNSUPPORTED_DATASET_IDS_BY_PROTOCOL = {
    "power_wd_yaw_lrpm_hist_sincos": ("sdwpf_kddcup",),
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
class TaskSeriesSelection:
    feature_protocol_id: str
    source_columns: tuple[str, ...]
    all_columns: tuple[str, ...]
    past_covariate_columns: tuple[str, ...]
    known_future_columns: tuple[str, ...]
    static_columns: tuple[str, ...]
    target_derived_columns: tuple[str, ...]
    audit_columns: tuple[str, ...]
    derived_angle_transforms: tuple[AngleTransformSpec, ...] = ()
    angle_convention: str | None = None
    dataset_specific_notes: tuple[str, ...] = ()


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
_FEATURE_PROTOCOLS = (
    _POWER_ONLY_PROTOCOL,
    _POWER_WS_HIST_PROTOCOL,
    _POWER_WD_HIST_SINCOS_PROTOCOL,
    _POWER_WS_WD_HIST_SINCOS_PROTOCOL,
    _POWER_WD_YAW_HIST_SINCOS_PROTOCOL,
    _POWER_WD_YAW_LRPM_HIST_SINCOS_PROTOCOL,
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
        _POWER_WD_YAW_LRPM_HIST_SINCOS_PROTOCOL.feature_protocol_id,
    }
    uses_yaw_error = feature_protocol_id in {
        _POWER_WD_YAW_HIST_SINCOS_PROTOCOL.feature_protocol_id,
        _POWER_WD_YAW_LRPM_HIST_SINCOS_PROTOCOL.feature_protocol_id,
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


def _dataset_native_columns_for_protocol(
    *,
    dataset_id: str,
    available_columns: set[str],
    feature_protocol_id: str,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[AngleTransformSpec, ...], str | None, tuple[str, ...]]:
    uses_wind_speed = feature_protocol_id in {
        _POWER_WS_HIST_PROTOCOL.feature_protocol_id,
        _POWER_WS_WD_HIST_SINCOS_PROTOCOL.feature_protocol_id,
    }
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

    derived_angle_transforms, angle_convention, dataset_specific_notes = _angle_transforms_for_protocol(
        dataset_id=dataset_id,
        feature_protocol_id=feature_protocol_id,
    )
    for transform in derived_angle_transforms:
        configured_columns.extend(transform.source_columns)
        output_past_covariate_columns.extend((transform.output_sin_column, transform.output_cos_column))

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
        angle_convention,
        dataset_specific_notes,
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


def materialize_task_series_frame(
    source_frame: pl.DataFrame,
    *,
    selection: TaskSeriesSelection,
) -> pl.DataFrame:
    series_frame = source_frame
    derived_columns: list[pl.Expr] = []
    for transform in selection.derived_angle_transforms:
        angle_radians = _angle_degrees_expr(transform) * pl.lit(_PI / 180.0)
        derived_columns.extend(
            (
                angle_radians.sin().alias(transform.output_sin_column),
                angle_radians.cos().alias(transform.output_cos_column),
            )
        )
    if derived_columns:
        series_frame = series_frame.with_columns(*derived_columns)
    return series_frame.select(list(selection.all_columns))


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
        past_covariate_columns,
        derived_angle_transforms,
        angle_convention,
        dataset_specific_notes,
    ) = _dataset_native_columns_for_protocol(
        dataset_id=dataset_id,
        available_columns=available_columns,
        feature_protocol_id=feature_protocol_id,
    )
    audit_columns = tuple(column for column in _SERIES_OPTIONAL_AUDIT_COLUMNS if column in available_columns)
    source_columns = tuple((*_SERIES_BASE_COLUMNS, *source_past_covariate_columns, *audit_columns))
    return TaskSeriesSelection(
        feature_protocol_id=feature_protocol_id,
        source_columns=source_columns,
        all_columns=tuple((*_SERIES_BASE_COLUMNS, *past_covariate_columns, *audit_columns)),
        past_covariate_columns=past_covariate_columns,
        known_future_columns=("dataset", "timestamp"),
        static_columns=(),
        target_derived_columns=(),
        audit_columns=audit_columns,
        derived_angle_transforms=derived_angle_transforms,
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
        },
        "task": task,
        "turbine_ids": list(turbine_ids),
        "time_axis_semantics": "farm_synchronous_long_panel",
        "column_groups": {
            "series": list(selection.all_columns),
            "past_covariates": list(selection.past_covariate_columns),
            "target_derived_covariates": list(selection.target_derived_columns),
            "known_future": list(selection.known_future_columns),
            "static": list(static_columns),
            "audit": list(selection.audit_columns),
        },
    }


__all__ = [
    "BLOCKED_BY_UNSUPPORTED_FEATURE_PROTOCOL",
    "DEFAULT_FEATURE_PROTOCOL_ID",
    "TASK_BUNDLE_SCHEMA_VERSION",
    "AngleTransformSpec",
    "FeatureProtocolSpec",
    "TaskSeriesSelection",
    "build_known_future_frame",
    "feature_protocol_task_blocked_reason",
    "get_feature_protocol_spec",
    "list_feature_protocol_ids",
    "list_feature_protocol_specs",
    "materialize_task_series_frame",
    "protocol_context_dict",
    "select_task_series_columns",
    "validate_feature_protocol_for_dataset",
]

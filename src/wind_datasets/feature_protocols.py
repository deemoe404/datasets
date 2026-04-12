from __future__ import annotations

from dataclasses import dataclass

import polars as pl


TASK_BUNDLE_SCHEMA_VERSION = "task_bundle.v1"
DEFAULT_FEATURE_PROTOCOL_ID = "power_only"
_PI = 3.141592653589793
_WIND_DIRECTION_SIN_COLUMN = "wind_direction_sin"
_WIND_DIRECTION_COS_COLUMN = "wind_direction_cos"

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
_POWER_WD_HIST_COLUMNS_BY_DATASET = {
    "kelmarsh": ("Wind direction (°)",),
    "penmanshiel": ("Wind direction (°)",),
    "hill_of_towie": ("wtc_ActualWindDirection_mean",),
    "sdwpf_kddcup": ("Wdir",),
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
class TaskSeriesSelection:
    feature_protocol_id: str
    source_columns: tuple[str, ...]
    all_columns: tuple[str, ...]
    past_covariate_columns: tuple[str, ...]
    known_future_columns: tuple[str, ...]
    static_columns: tuple[str, ...]
    target_derived_columns: tuple[str, ...]
    audit_columns: tuple[str, ...]
    wind_direction_source_column: str | None = None


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
    past_covariate_source="gold_base.dataset_native_wind_direction",
    past_covariate_stage="task_bundle.direction_sincos",
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
    past_covariate_source="gold_base.dataset_native_wind_speed_direction",
    past_covariate_stage="task_bundle.direction_sincos",
)
_FEATURE_PROTOCOLS = (
    _POWER_ONLY_PROTOCOL,
    _POWER_WS_HIST_PROTOCOL,
    _POWER_WD_HIST_SINCOS_PROTOCOL,
    _POWER_WS_WD_HIST_SINCOS_PROTOCOL,
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


def _dataset_native_columns_for_protocol(
    *,
    dataset_id: str,
    available_columns: set[str],
    feature_protocol_id: str,
) -> tuple[tuple[str, ...], tuple[str, ...], str | None]:
    uses_wind_speed = feature_protocol_id in {
        _POWER_WS_HIST_PROTOCOL.feature_protocol_id,
        _POWER_WS_WD_HIST_SINCOS_PROTOCOL.feature_protocol_id,
    }
    uses_wind_direction = feature_protocol_id in {
        _POWER_WD_HIST_SINCOS_PROTOCOL.feature_protocol_id,
        _POWER_WS_WD_HIST_SINCOS_PROTOCOL.feature_protocol_id,
    }
    configured_columns: list[str] = []
    output_past_covariate_columns: list[str] = []
    wind_direction_source_column: str | None = None

    if uses_wind_speed:
        try:
            wind_speed_columns = _POWER_WS_HIST_COLUMNS_BY_DATASET[dataset_id]
        except KeyError as exc:
            raise ValueError(
                f"feature_protocol_id {feature_protocol_id!r} is not configured for dataset {dataset_id!r}."
            ) from exc
        configured_columns.extend(wind_speed_columns)
        output_past_covariate_columns.extend(wind_speed_columns)
    if uses_wind_direction:
        try:
            wind_direction_columns = _POWER_WD_HIST_COLUMNS_BY_DATASET[dataset_id]
        except KeyError as exc:
            raise ValueError(
                f"feature_protocol_id {feature_protocol_id!r} is not configured for dataset {dataset_id!r}."
            ) from exc
        wind_direction_source_column = wind_direction_columns[0]
        configured_columns.extend(wind_direction_columns)
        output_past_covariate_columns.extend((_WIND_DIRECTION_SIN_COLUMN, _WIND_DIRECTION_COS_COLUMN))

    configured_columns = list(dict.fromkeys(configured_columns))
    missing_columns = [column for column in configured_columns if column not in available_columns]
    if missing_columns:
        raise ValueError(
            f"feature_protocol_id {feature_protocol_id!r} for dataset {dataset_id!r} requires "
            f"missing gold-base columns {missing_columns!r}."
        )
    return tuple(configured_columns), tuple(output_past_covariate_columns), wind_direction_source_column


def materialize_task_series_frame(
    source_frame: pl.DataFrame,
    *,
    selection: TaskSeriesSelection,
) -> pl.DataFrame:
    series_frame = source_frame
    if selection.wind_direction_source_column is not None:
        wind_direction_radians = pl.col(selection.wind_direction_source_column).cast(pl.Float64, strict=False) * pl.lit(
            _PI / 180.0
        )
        series_frame = series_frame.with_columns(
            wind_direction_radians.sin().alias(_WIND_DIRECTION_SIN_COLUMN),
            wind_direction_radians.cos().alias(_WIND_DIRECTION_COS_COLUMN),
        )
    return series_frame.select(list(selection.all_columns))


def select_task_series_columns(
    *,
    dataset_id: str,
    available_columns: set[str],
    feature_protocol_id: str,
    turbine_static_columns: set[str] | None = None,
) -> TaskSeriesSelection:
    del turbine_static_columns
    get_feature_protocol_spec(feature_protocol_id)
    source_past_covariate_columns, past_covariate_columns, wind_direction_source_column = _dataset_native_columns_for_protocol(
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
        wind_direction_source_column=wind_direction_source_column,
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
    "DEFAULT_FEATURE_PROTOCOL_ID",
    "TASK_BUNDLE_SCHEMA_VERSION",
    "FeatureProtocolSpec",
    "TaskSeriesSelection",
    "build_known_future_frame",
    "get_feature_protocol_spec",
    "list_feature_protocol_ids",
    "list_feature_protocol_specs",
    "materialize_task_series_frame",
    "protocol_context_dict",
    "select_task_series_columns",
]

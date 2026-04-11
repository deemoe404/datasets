from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl


DEFAULT_COVARIATE_STAGES = ("stage1_core", "stage2_ops", "stage3_regime")
TASK_BUNDLE_SCHEMA_VERSION = "task_bundle.v1"
DEFAULT_FEATURE_PROTOCOL_ID = "power_only"

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
_STATIC_BASE_COLUMNS = ("dataset", "turbine_id", "turbine_index")
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

_GREENBYTE_STAGE1_COLUMNS = (
    "Wind speed (m/s)",
    "Wind direction (°)",
    "Nacelle position (°)",
    "Generator RPM (RPM)",
    "Rotor speed (RPM)",
    "Ambient temperature (converter) (°C)",
    "Nacelle temperature (°C)",
    "Power factor (cosphi)",
    "Reactive power (kvar)",
    "Blade angle (pitch position) A (°)",
    "Blade angle (pitch position) B (°)",
    "Blade angle (pitch position) C (°)",
)

_GREENBYTE_STAGE2_EXTRA_COLUMNS = (
    "farm_pmu__gms_power_kw",
    "farm_pmu__gms_reactive_power_kvar",
    "farm_pmu__gms_current_a",
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
    "wtc_YawPos_mean",
    "wtc_GenRpm_mean",
    "wtc_MainSRpm_mean",
    "wtc_PitchRef_BladeA_mean",
    "wtc_PitchRef_BladeB_mean",
    "wtc_PitchRef_BladeC_mean",
    "wtc_TwrHumid_mean",
)

_HILL_STAGE2_EXTRA_COLUMNS = (
    "tur_temp__wtc_ambietmp_mean",
    "tur_temp__wtc_naceltmp_mean",
    "tur_temp__wtc_geoiltmp_mean",
    "tur_press__wtc_hydpress_mean",
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

_TARGET_DERIVED_COLUMNS_BY_DATASET = {
    "kelmarsh": (
        "Power, Minimum (kW)",
        "Power, Maximum (kW)",
        "Power, Standard deviation (kW)",
    ),
    "penmanshiel": (
        "Power, Minimum (kW)",
        "Power, Maximum (kW)",
        "Power, Standard deviation (kW)",
    ),
    "hill_of_towie": (
        "wtc_ActPower_min",
        "wtc_ActPower_max",
        "wtc_ActPower_stddev",
        "wtc_ActPower_endvalue",
    ),
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
class CovariatePackSpec:
    dataset_id: str
    stage: str
    pack_name: str
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


@dataclass(frozen=True)
class TaskSeriesSelection:
    feature_protocol_id: str
    all_columns: tuple[str, ...]
    past_covariate_columns: tuple[str, ...]
    known_future_columns: tuple[str, ...]
    static_columns: tuple[str, ...]
    target_derived_columns: tuple[str, ...]
    audit_columns: tuple[str, ...]


_FEATURE_PROTOCOLS: dict[str, FeatureProtocolSpec] = {
    "power_only": FeatureProtocolSpec(
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
    ),
    "power_stats_history": FeatureProtocolSpec(
        feature_protocol_id="power_stats_history",
        display_name="Power-Stats History",
        protocol_kind="target_derived_history",
        summary="Augment target history with target-derived historical power statistics.",
        uses_target_history=True,
        uses_static_covariates=False,
        uses_known_future_covariates=False,
        uses_past_covariates=False,
        uses_target_derived_covariates=True,
        aliases=("power_stats",),
    ),
    "staged_past_covariates.stage1_core": FeatureProtocolSpec(
        feature_protocol_id="staged_past_covariates.stage1_core",
        display_name="Staged Past Covariates Stage 1",
        protocol_kind="staged_past_covariates",
        summary="Use target history plus dataset-native stage1 core past covariates.",
        uses_target_history=True,
        uses_static_covariates=False,
        uses_known_future_covariates=False,
        uses_past_covariates=True,
        uses_target_derived_covariates=False,
        aliases=("stage1_core", "hist_stage1"),
        past_covariate_source="dataset_native_past_covariates",
        past_covariate_stage="stage1_core",
    ),
    "staged_past_covariates.stage2_ops": FeatureProtocolSpec(
        feature_protocol_id="staged_past_covariates.stage2_ops",
        display_name="Staged Past Covariates Stage 2",
        protocol_kind="staged_past_covariates",
        summary="Use target history plus dataset-native stage2 operations past covariates.",
        uses_target_history=True,
        uses_static_covariates=False,
        uses_known_future_covariates=False,
        uses_past_covariates=True,
        uses_target_derived_covariates=False,
        aliases=("stage2_ops", "hist_stage2"),
        past_covariate_source="dataset_native_past_covariates",
        past_covariate_stage="stage2_ops",
    ),
    "staged_past_covariates.stage3_regime": FeatureProtocolSpec(
        feature_protocol_id="staged_past_covariates.stage3_regime",
        display_name="Staged Past Covariates Stage 3",
        protocol_kind="staged_past_covariates",
        summary="Use target history plus dataset-native stage3 regime past covariates.",
        uses_target_history=True,
        uses_static_covariates=False,
        uses_known_future_covariates=False,
        uses_past_covariates=True,
        uses_target_derived_covariates=False,
        aliases=("stage3_regime",),
        past_covariate_source="dataset_native_past_covariates",
        past_covariate_stage="stage3_regime",
    ),
    "static_calendar": FeatureProtocolSpec(
        feature_protocol_id="static_calendar",
        display_name="Static Plus Calendar",
        protocol_kind="static_calendar",
        summary="Use target history with static spatial covariates and deterministic calendar covariates.",
        uses_target_history=True,
        uses_static_covariates=True,
        uses_known_future_covariates=True,
        uses_past_covariates=False,
        uses_target_derived_covariates=False,
        aliases=("known_static",),
    ),
    "static_calendar_stage1": FeatureProtocolSpec(
        feature_protocol_id="static_calendar_stage1",
        display_name="Static Plus Calendar Plus Stage 1",
        protocol_kind="mixed",
        summary="Use target history with static spatial covariates, deterministic calendar covariates, and stage1 past covariates.",
        uses_target_history=True,
        uses_static_covariates=True,
        uses_known_future_covariates=True,
        uses_past_covariates=True,
        uses_target_derived_covariates=False,
        aliases=("mixed_stage1",),
        past_covariate_source="dataset_native_past_covariates",
        past_covariate_stage="stage1_core",
    ),
    "static_calendar_stage2": FeatureProtocolSpec(
        feature_protocol_id="static_calendar_stage2",
        display_name="Static Plus Calendar Plus Stage 2",
        protocol_kind="mixed",
        summary="Use target history with static spatial covariates, deterministic calendar covariates, and stage2 past covariates.",
        uses_target_history=True,
        uses_static_covariates=True,
        uses_known_future_covariates=True,
        uses_past_covariates=True,
        uses_target_derived_covariates=False,
        aliases=("mixed_stage2",),
        past_covariate_source="dataset_native_past_covariates",
        past_covariate_stage="stage2_ops",
    ),
}


def _greenbyte_pack(dataset_id: str, stage: str) -> CovariatePackSpec:
    if stage == "stage1_core":
        return CovariatePackSpec(dataset_id=dataset_id, stage=stage, pack_name=stage, required_columns=_GREENBYTE_STAGE1_COLUMNS)
    if stage == "stage2_ops":
        return CovariatePackSpec(
            dataset_id=dataset_id,
            stage=stage,
            pack_name=stage,
            required_columns=_GREENBYTE_STAGE1_COLUMNS + _GREENBYTE_STAGE2_EXTRA_COLUMNS,
        )
    if stage == "stage3_regime":
        return CovariatePackSpec(
            dataset_id=dataset_id,
            stage=stage,
            pack_name=stage,
            required_columns=_GREENBYTE_STAGE1_COLUMNS + _GREENBYTE_STAGE2_EXTRA_COLUMNS,
            optional_columns=_GREENBYTE_STAGE3_OPTIONAL_COLUMNS,
        )
    raise ValueError(f"Unsupported covariate stage {stage!r}.")


def _hill_pack(stage: str) -> CovariatePackSpec:
    if stage == "stage1_core":
        return CovariatePackSpec(dataset_id="hill_of_towie", stage=stage, pack_name=stage, required_columns=_HILL_STAGE1_COLUMNS)
    if stage == "stage2_ops":
        return CovariatePackSpec(
            dataset_id="hill_of_towie",
            stage=stage,
            pack_name=stage,
            required_columns=_HILL_STAGE1_COLUMNS + _HILL_STAGE2_EXTRA_COLUMNS,
        )
    if stage == "stage3_regime":
        return CovariatePackSpec(
            dataset_id="hill_of_towie",
            stage=stage,
            pack_name=stage,
            required_columns=_HILL_STAGE1_COLUMNS + _HILL_STAGE2_EXTRA_COLUMNS + _HILL_STAGE3_EXTRA_COLUMNS,
        )
    raise ValueError(f"Unsupported covariate stage {stage!r}.")


def _sdwpf_pack(stage: str) -> CovariatePackSpec:
    if stage == "stage1_core":
        return CovariatePackSpec(dataset_id="sdwpf_kddcup", stage=stage, pack_name=stage, required_columns=_SDWPF_STAGE1_COLUMNS)
    if stage == "stage2_ops":
        return CovariatePackSpec(
            dataset_id="sdwpf_kddcup",
            stage=stage,
            pack_name=stage,
            required_columns=_SDWPF_STAGE1_COLUMNS + _SDWPF_STAGE2_EXTRA_COLUMNS,
        )
    if stage == "stage3_regime":
        return CovariatePackSpec(
            dataset_id="sdwpf_kddcup",
            stage=stage,
            pack_name=stage,
            required_columns=_SDWPF_STAGE1_COLUMNS + _SDWPF_STAGE2_EXTRA_COLUMNS + _SDWPF_STAGE3_EXTRA_COLUMNS,
        )
    raise ValueError(f"Unsupported covariate stage {stage!r}.")


def list_feature_protocol_ids() -> tuple[str, ...]:
    return tuple(_FEATURE_PROTOCOLS)


def list_feature_protocol_specs() -> tuple[FeatureProtocolSpec, ...]:
    return tuple(_FEATURE_PROTOCOLS.values())


def get_feature_protocol_spec(feature_protocol_id: str) -> FeatureProtocolSpec:
    try:
        return _FEATURE_PROTOCOLS[feature_protocol_id]
    except KeyError as exc:
        raise ValueError(f"Unknown feature_protocol_id {feature_protocol_id!r}.") from exc


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
    return CovariatePackSpec(dataset_id=dataset_id, stage="reference", pack_name="power_only", required_columns=())


def iter_covariate_packs(
    dataset_ids: tuple[str, ...],
    stages: tuple[str, ...] = DEFAULT_COVARIATE_STAGES,
) -> tuple[CovariatePackSpec, ...]:
    return tuple(
        resolve_covariate_pack(dataset_id, stage)
        for dataset_id in dataset_ids
        for stage in stages
    )


def _selected_past_covariate_columns(
    *,
    dataset_id: str,
    available_columns: set[str],
    protocol: FeatureProtocolSpec,
) -> tuple[str, ...]:
    if not protocol.uses_past_covariates or protocol.past_covariate_stage is None:
        return ()
    return resolve_covariate_pack(dataset_id, protocol.past_covariate_stage).selected_covariate_columns(available_columns)


def _selected_target_derived_columns(
    *,
    dataset_id: str,
    available_columns: set[str],
    protocol: FeatureProtocolSpec,
) -> tuple[str, ...]:
    if not protocol.uses_target_derived_covariates:
        return ()
    requested = _TARGET_DERIVED_COLUMNS_BY_DATASET.get(dataset_id, ())
    missing = [column for column in requested if column not in available_columns]
    if missing:
        raise ValueError(
            f"Dataset {dataset_id!r} protocol {protocol.feature_protocol_id!r} is missing target-derived columns {missing!r}."
        )
    return tuple(requested)


def selected_static_columns(protocol: FeatureProtocolSpec, turbine_static_columns: set[str]) -> tuple[str, ...]:
    if not protocol.uses_static_covariates:
        return ()
    preferred = (
        "coord_x",
        "coord_y",
        "latitude",
        "longitude",
        "elevation_m",
        "rated_power_kw",
        "hub_height_m",
        "rotor_diameter_m",
    )
    return tuple(column for column in preferred if column in turbine_static_columns)


def select_task_series_columns(
    *,
    dataset_id: str,
    available_columns: set[str],
    feature_protocol_id: str,
    turbine_static_columns: set[str] | None = None,
) -> TaskSeriesSelection:
    protocol = get_feature_protocol_spec(feature_protocol_id)
    past_covariates = _selected_past_covariate_columns(
        dataset_id=dataset_id,
        available_columns=available_columns,
        protocol=protocol,
    )
    target_derived = _selected_target_derived_columns(
        dataset_id=dataset_id,
        available_columns=available_columns,
        protocol=protocol,
    )
    audit_columns = tuple(column for column in _SERIES_OPTIONAL_AUDIT_COLUMNS if column in available_columns)
    all_columns = tuple(dict.fromkeys((*_SERIES_BASE_COLUMNS, *audit_columns, *past_covariates, *target_derived)))
    static_columns = selected_static_columns(protocol, set(turbine_static_columns or set()))
    known_future_columns = _KNOWN_FUTURE_COLUMNS if protocol.uses_known_future_covariates else ("dataset", "timestamp")
    return TaskSeriesSelection(
        feature_protocol_id=feature_protocol_id,
        all_columns=all_columns,
        past_covariate_columns=past_covariates,
        known_future_columns=known_future_columns,
        static_columns=static_columns,
        target_derived_columns=target_derived,
        audit_columns=audit_columns,
    )


def build_known_future_frame(series_frame: pl.DataFrame) -> pl.DataFrame:
    if series_frame.is_empty():
        return pl.DataFrame(schema={column: pl.Float64 if column != "dataset" else pl.String for column in _KNOWN_FUTURE_COLUMNS})
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
            (2 * pl.lit(3.141592653589793) * pl.col("__minute_of_day") / pl.lit(24 * 60)).sin().alias("calendar_hour_sin"),
            (2 * pl.lit(3.141592653589793) * pl.col("__minute_of_day") / pl.lit(24 * 60)).cos().alias("calendar_hour_cos"),
            (2 * pl.lit(3.141592653589793) * pl.col("__weekday") / pl.lit(7)).sin().alias("calendar_weekday_sin"),
            (2 * pl.lit(3.141592653589793) * pl.col("__weekday") / pl.lit(7)).cos().alias("calendar_weekday_cos"),
            (2 * pl.lit(3.141592653589793) * (pl.col("__month") - 1) / pl.lit(12)).sin().alias("calendar_month_sin"),
            (2 * pl.lit(3.141592653589793) * (pl.col("__month") - 1) / pl.lit(12)).cos().alias("calendar_month_cos"),
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
            "static": list((_STATIC_BASE_COLUMNS + selection.static_columns)),
            "audit": list(selection.audit_columns),
        },
    }


__all__ = [
    "DEFAULT_COVARIATE_STAGES",
    "DEFAULT_FEATURE_PROTOCOL_ID",
    "TASK_BUNDLE_SCHEMA_VERSION",
    "CovariatePackSpec",
    "FeatureProtocolSpec",
    "TaskSeriesSelection",
    "build_known_future_frame",
    "get_feature_protocol_spec",
    "iter_covariate_packs",
    "list_feature_protocol_ids",
    "list_feature_protocol_specs",
    "protocol_context_dict",
    "reference_pack_for",
    "resolve_covariate_pack",
    "select_task_series_columns",
]

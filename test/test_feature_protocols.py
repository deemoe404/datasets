from __future__ import annotations

from datetime import datetime
import math

import pytest
import polars as pl

from wind_datasets.feature_protocols import (
    DEFAULT_FEATURE_PROTOCOL_ID,
    get_feature_protocol_spec,
    list_feature_protocol_ids,
    materialize_task_series_frame,
    protocol_context_dict,
    select_task_series_columns,
)


_BASE_AVAILABLE_COLUMNS = {
    "dataset",
    "turbine_id",
    "timestamp",
    "target_kw",
    "is_observed",
    "quality_flags",
    "feature_quality_flags",
    "farm_turbines_expected",
}


def _selection(dataset_id: str, feature_protocol_id: str, *extra_columns: str):
    return select_task_series_columns(
        dataset_id=dataset_id,
        available_columns={*_BASE_AVAILABLE_COLUMNS, *extra_columns},
        feature_protocol_id=feature_protocol_id,
        turbine_static_columns={"coord_x", "coord_y"},
    )


def test_active_feature_protocols_are_registered() -> None:
    assert DEFAULT_FEATURE_PROTOCOL_ID == "power_only"
    assert list_feature_protocol_ids() == (
        "power_only",
        "power_ws_hist",
        "power_atemp_hist",
        "power_itemp_hist",
        "power_wd_hist_sincos",
        "power_ws_wd_hist_sincos",
        "power_wd_yaw_hist_sincos",
        "power_wd_yaw_pitchmean_hist_sincos",
        "power_wd_yaw_pmean_hist_sincos_masked",
        "power_wd_yaw_lrpm_hist_sincos",
        "world_model_v1",
    )

    power_only = get_feature_protocol_spec("power_only")
    assert power_only.feature_protocol_id == "power_only"
    assert power_only.uses_target_history is True
    assert power_only.uses_past_covariates is False

    power_ws_hist = get_feature_protocol_spec("power_ws_hist")
    assert power_ws_hist.feature_protocol_id == "power_ws_hist"
    assert power_ws_hist.uses_target_history is True
    assert power_ws_hist.uses_past_covariates is True
    assert power_ws_hist.uses_known_future_covariates is False

    power_atemp_hist = get_feature_protocol_spec("power_atemp_hist")
    assert power_atemp_hist.feature_protocol_id == "power_atemp_hist"
    assert power_atemp_hist.uses_target_history is True
    assert power_atemp_hist.uses_past_covariates is True
    assert power_atemp_hist.uses_known_future_covariates is False

    power_itemp_hist = get_feature_protocol_spec("power_itemp_hist")
    assert power_itemp_hist.feature_protocol_id == "power_itemp_hist"
    assert power_itemp_hist.uses_target_history is True
    assert power_itemp_hist.uses_past_covariates is True
    assert power_itemp_hist.uses_known_future_covariates is False

    power_wd_hist = get_feature_protocol_spec("power_wd_hist_sincos")
    assert power_wd_hist.feature_protocol_id == "power_wd_hist_sincos"
    assert power_wd_hist.uses_target_history is True
    assert power_wd_hist.uses_past_covariates is True

    power_ws_wd_hist = get_feature_protocol_spec("power_ws_wd_hist_sincos")
    assert power_ws_wd_hist.feature_protocol_id == "power_ws_wd_hist_sincos"
    assert power_ws_wd_hist.uses_target_history is True
    assert power_ws_wd_hist.uses_past_covariates is True

    power_wd_yaw_hist = get_feature_protocol_spec("power_wd_yaw_hist_sincos")
    assert power_wd_yaw_hist.feature_protocol_id == "power_wd_yaw_hist_sincos"
    assert power_wd_yaw_hist.uses_target_history is True
    assert power_wd_yaw_hist.uses_past_covariates is True

    power_wd_yaw_pitchmean_hist = get_feature_protocol_spec("power_wd_yaw_pitchmean_hist_sincos")
    assert power_wd_yaw_pitchmean_hist.feature_protocol_id == "power_wd_yaw_pitchmean_hist_sincos"
    assert power_wd_yaw_pitchmean_hist.uses_target_history is True
    assert power_wd_yaw_pitchmean_hist.uses_past_covariates is True

    power_wd_yaw_pmean_hist_masked = get_feature_protocol_spec("power_wd_yaw_pmean_hist_sincos_masked")
    assert (
        power_wd_yaw_pmean_hist_masked.feature_protocol_id
        == "power_wd_yaw_pmean_hist_sincos_masked"
    )
    assert power_wd_yaw_pmean_hist_masked.uses_target_history is True
    assert power_wd_yaw_pmean_hist_masked.uses_past_covariates is True

    power_wd_yaw_lrpm_hist = get_feature_protocol_spec("power_wd_yaw_lrpm_hist_sincos")
    assert power_wd_yaw_lrpm_hist.feature_protocol_id == "power_wd_yaw_lrpm_hist_sincos"
    assert power_wd_yaw_lrpm_hist.uses_target_history is True
    assert power_wd_yaw_lrpm_hist.uses_past_covariates is True

    world_model_v1 = get_feature_protocol_spec("world_model_v1")
    assert world_model_v1.feature_protocol_id == "world_model_v1"
    assert world_model_v1.uses_target_history is True
    assert world_model_v1.uses_past_covariates is True
    assert world_model_v1.uses_static_covariates is True
    assert world_model_v1.uses_known_future_covariates is True


def test_unknown_feature_protocol_is_rejected() -> None:
    with pytest.raises(ValueError):
        get_feature_protocol_spec("unexpected_protocol")


def test_power_only_selection_keeps_only_base_series_columns() -> None:
    selection = _selection("kelmarsh", "power_only", "Wind speed (m/s)")

    assert selection.all_columns == (
        "dataset",
        "turbine_id",
        "timestamp",
        "target_kw",
        "is_observed",
        "quality_flags",
        "feature_quality_flags",
        "farm_turbines_expected",
    )
    assert selection.source_columns == selection.all_columns
    assert selection.past_covariate_columns == ()
    assert selection.static_columns == ()
    assert selection.target_derived_columns == ()
    assert selection.known_future_columns == ("dataset", "timestamp")
    assert selection.derived_angle_transforms == ()
    assert selection.angle_convention is None
    assert selection.dataset_specific_notes == ()


@pytest.mark.parametrize(
    ("dataset_id", "feature_column"),
    [
        ("kelmarsh", "Wind speed (m/s)"),
        ("penmanshiel", "Wind speed (m/s)"),
        ("hill_of_towie", "wtc_AcWindSp_mean"),
        ("sdwpf_kddcup", "Wspd"),
    ],
)
def test_power_ws_hist_selection_uses_dataset_native_wind_speed(
    dataset_id: str,
    feature_column: str,
) -> None:
    selection = _selection(dataset_id, "power_ws_hist", feature_column)

    assert selection.past_covariate_columns == (feature_column,)
    assert selection.source_columns == (
        "dataset",
        "turbine_id",
        "timestamp",
        "target_kw",
        "is_observed",
        "quality_flags",
        "feature_quality_flags",
        feature_column,
        "farm_turbines_expected",
    )
    assert selection.all_columns == selection.source_columns
    assert selection.derived_angle_transforms == ()


@pytest.mark.parametrize(
    ("dataset_id", "feature_column"),
    [
        ("kelmarsh", "Nacelle ambient temperature (°C)"),
        ("penmanshiel", "Nacelle ambient temperature (°C)"),
        ("hill_of_towie", "tur_temp__wtc_ambietmp_mean"),
        ("sdwpf_kddcup", "Etmp"),
    ],
)
def test_power_atemp_hist_selection_uses_dataset_native_ambient_temperature(
    dataset_id: str,
    feature_column: str,
) -> None:
    selection = _selection(dataset_id, "power_atemp_hist", feature_column)

    assert selection.past_covariate_columns == (feature_column,)
    assert selection.source_columns == (
        "dataset",
        "turbine_id",
        "timestamp",
        "target_kw",
        "is_observed",
        "quality_flags",
        "feature_quality_flags",
        feature_column,
        "farm_turbines_expected",
    )
    assert selection.all_columns == selection.source_columns
    assert selection.derived_angle_transforms == ()
    assert selection.derived_scalar_transforms == ()


@pytest.mark.parametrize(
    ("dataset_id", "feature_column"),
    [
        ("kelmarsh", "Nacelle temperature (°C)"),
        ("penmanshiel", "Nacelle temperature (°C)"),
        ("hill_of_towie", "tur_temp__wtc_naceltmp_mean"),
        ("sdwpf_kddcup", "Itmp"),
    ],
)
def test_power_itemp_hist_selection_uses_dataset_native_internal_temperature(
    dataset_id: str,
    feature_column: str,
) -> None:
    selection = _selection(dataset_id, "power_itemp_hist", feature_column)

    assert selection.past_covariate_columns == (feature_column,)
    assert selection.source_columns == (
        "dataset",
        "turbine_id",
        "timestamp",
        "target_kw",
        "is_observed",
        "quality_flags",
        "feature_quality_flags",
        feature_column,
        "farm_turbines_expected",
    )
    assert selection.all_columns == selection.source_columns
    assert selection.derived_angle_transforms == ()
    assert selection.derived_scalar_transforms == ()


def test_world_model_v1_selection_records_local_global_static_and_pairwise_groups() -> None:
    selection = _selection(
        "kelmarsh",
        "world_model_v1",
        "Wind speed (m/s)",
        "Wind direction (°)",
        "Nacelle position (°)",
        "Blade angle (pitch position) A (°)",
        "Blade angle (pitch position) B (°)",
        "Blade angle (pitch position) C (°)",
        "Rotor speed (RPM)",
        "Generator RPM (RPM)",
        "Nacelle ambient temperature (°C)",
        "Nacelle temperature (°C)",
        "evt_any_active",
        "evt_active_count",
        "evt_total_overlap_seconds",
        "evt_stop_active",
        "evt_warning_active",
        "evt_informational_active",
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

    assert selection.target_history_mask_columns == ("target_kw__mask",)
    assert selection.local_observation_value_columns == (
        "Wind speed (m/s)",
        "wind_direction_sin",
        "wind_direction_cos",
        "yaw_error_sin",
        "yaw_error_cos",
        "pitch_mean",
        "Rotor speed (RPM)",
        "Generator RPM (RPM)",
        "Nacelle ambient temperature (°C)",
        "Nacelle temperature (°C)",
        "evt_any_active",
        "evt_active_count",
        "evt_total_overlap_seconds",
        "evt_stop_active",
        "evt_warning_active",
        "evt_informational_active",
    )
    assert selection.global_observation_value_columns == (
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
    assert selection.local_observation_mask_columns == tuple(
        f"{column}__mask" for column in selection.local_observation_value_columns
    )
    assert selection.global_observation_mask_columns == tuple(
        f"{column}__mask" for column in selection.global_observation_value_columns
    )
    assert selection.known_future_columns == (
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
    assert selection.static_columns == (
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
    assert selection.pairwise_columns == (
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


@pytest.mark.parametrize(
    ("dataset_id", "source_columns", "angle_convention"),
    [
        ("kelmarsh", ("Wind direction (°)",), None),
        ("penmanshiel", ("Wind direction (°)",), None),
        ("hill_of_towie", ("wtc_ActualWindDirection_mean",), None),
        ("sdwpf_kddcup", ("Ndir", "Wdir"), "yaw_error_degrees = wind_direction_degrees - nacelle_or_yaw_position_degrees"),
    ],
)
def test_power_wd_hist_sincos_selection_uses_dataset_direction_semantics(
    dataset_id: str,
    source_columns: tuple[str, ...],
    angle_convention: str | None,
) -> None:
    selection = _selection(dataset_id, "power_wd_hist_sincos", *source_columns)

    assert selection.source_columns == (
        "dataset",
        "turbine_id",
        "timestamp",
        "target_kw",
        "is_observed",
        "quality_flags",
        "feature_quality_flags",
        *source_columns,
        "farm_turbines_expected",
    )
    assert selection.past_covariate_columns == ("wind_direction_sin", "wind_direction_cos")
    assert selection.all_columns == (
        "dataset",
        "turbine_id",
        "timestamp",
        "target_kw",
        "is_observed",
        "quality_flags",
        "feature_quality_flags",
        "wind_direction_sin",
        "wind_direction_cos",
        "farm_turbines_expected",
    )
    assert len(selection.derived_angle_transforms) == 1
    assert selection.derived_angle_transforms[0].source_columns == source_columns
    assert selection.angle_convention == angle_convention


@pytest.mark.parametrize(
    ("dataset_id", "ws_column", "direction_source_columns"),
    [
        ("kelmarsh", "Wind speed (m/s)", ("Wind direction (°)",)),
        ("penmanshiel", "Wind speed (m/s)", ("Wind direction (°)",)),
        ("hill_of_towie", "wtc_AcWindSp_mean", ("wtc_ActualWindDirection_mean",)),
        ("sdwpf_kddcup", "Wspd", ("Ndir", "Wdir")),
    ],
)
def test_power_ws_wd_hist_sincos_selection_orders_ws_then_direction_sincos(
    dataset_id: str,
    ws_column: str,
    direction_source_columns: tuple[str, ...],
) -> None:
    selection = _selection(dataset_id, "power_ws_wd_hist_sincos", ws_column, *direction_source_columns)

    assert selection.source_columns == (
        "dataset",
        "turbine_id",
        "timestamp",
        "target_kw",
        "is_observed",
        "quality_flags",
        "feature_quality_flags",
        ws_column,
        *direction_source_columns,
        "farm_turbines_expected",
    )
    assert selection.past_covariate_columns == (ws_column, "wind_direction_sin", "wind_direction_cos")
    assert selection.all_columns == (
        "dataset",
        "turbine_id",
        "timestamp",
        "target_kw",
        "is_observed",
        "quality_flags",
        "feature_quality_flags",
        ws_column,
        "wind_direction_sin",
        "wind_direction_cos",
        "farm_turbines_expected",
    )
    assert len(selection.derived_angle_transforms) == 1
    assert selection.derived_angle_transforms[0].source_columns == direction_source_columns


@pytest.mark.parametrize(
    ("dataset_id", "source_columns"),
    [
        ("kelmarsh", ("Wind direction (°)", "Nacelle position (°)")),
        ("penmanshiel", ("Wind direction (°)", "Nacelle position (°)")),
        ("hill_of_towie", ("wtc_ActualWindDirection_mean", "wtc_YawPos_mean")),
        ("sdwpf_kddcup", ("Ndir", "Wdir")),
    ],
)
def test_power_wd_yaw_hist_sincos_selection_emits_four_cyclic_covariates(
    dataset_id: str,
    source_columns: tuple[str, ...],
) -> None:
    selection = _selection(dataset_id, "power_wd_yaw_hist_sincos", *source_columns)

    assert selection.source_columns == (
        "dataset",
        "turbine_id",
        "timestamp",
        "target_kw",
        "is_observed",
        "quality_flags",
        "feature_quality_flags",
        *source_columns,
        "farm_turbines_expected",
    )
    assert selection.past_covariate_columns == (
        "wind_direction_sin",
        "wind_direction_cos",
        "yaw_error_sin",
        "yaw_error_cos",
    )
    assert selection.all_columns == (
        "dataset",
        "turbine_id",
        "timestamp",
        "target_kw",
        "is_observed",
        "quality_flags",
        "feature_quality_flags",
        "wind_direction_sin",
        "wind_direction_cos",
        "yaw_error_sin",
        "yaw_error_cos",
        "farm_turbines_expected",
    )
    assert len(selection.derived_angle_transforms) == 2
    assert selection.angle_convention == "yaw_error_degrees = wind_direction_degrees - nacelle_or_yaw_position_degrees"


@pytest.mark.parametrize(
    ("dataset_id", "source_columns"),
    [
        (
            "kelmarsh",
            (
                "Wind direction (°)",
                "Nacelle position (°)",
                "Blade angle (pitch position) A (°)",
                "Blade angle (pitch position) B (°)",
                "Blade angle (pitch position) C (°)",
            ),
        ),
        (
            "penmanshiel",
            (
                "Wind direction (°)",
                "Nacelle position (°)",
                "Blade angle (pitch position) A (°)",
                "Blade angle (pitch position) B (°)",
                "Blade angle (pitch position) C (°)",
            ),
        ),
        (
            "hill_of_towie",
            (
                "wtc_ActualWindDirection_mean",
                "wtc_YawPos_mean",
                "wtc_PitcPosA_mean",
                "wtc_PitcPosB_mean",
                "wtc_PitcPosC_mean",
            ),
        ),
        ("sdwpf_kddcup", ("Ndir", "Wdir", "Pab1", "Pab2", "Pab3")),
    ],
)
def test_power_wd_yaw_pitchmean_hist_sincos_selection_emits_cyclic_covariates_and_pitch_mean(
    dataset_id: str,
    source_columns: tuple[str, ...],
) -> None:
    selection = _selection(dataset_id, "power_wd_yaw_pitchmean_hist_sincos", *source_columns)

    assert selection.source_columns == (
        "dataset",
        "turbine_id",
        "timestamp",
        "target_kw",
        "is_observed",
        "quality_flags",
        "feature_quality_flags",
        *source_columns,
        "farm_turbines_expected",
    )
    assert selection.past_covariate_columns == (
        "wind_direction_sin",
        "wind_direction_cos",
        "yaw_error_sin",
        "yaw_error_cos",
        "pitch_mean",
    )
    assert selection.all_columns == (
        "dataset",
        "turbine_id",
        "timestamp",
        "target_kw",
        "is_observed",
        "quality_flags",
        "feature_quality_flags",
        "wind_direction_sin",
        "wind_direction_cos",
        "yaw_error_sin",
        "yaw_error_cos",
        "pitch_mean",
        "farm_turbines_expected",
    )
    assert len(selection.derived_angle_transforms) == 2
    assert len(selection.derived_scalar_transforms) == 1
    assert selection.derived_scalar_transforms[0].output_column == "pitch_mean"
    assert selection.derived_scalar_transforms[0].source_columns == source_columns[-3:]
    assert selection.derived_scalar_transforms[0].missing_value_policy == "all_sources_required"
    assert selection.angle_convention == "yaw_error_degrees = wind_direction_degrees - nacelle_or_yaw_position_degrees"


@pytest.mark.parametrize(
    ("dataset_id", "source_columns"),
    [
        (
            "kelmarsh",
            (
                "Wind direction (°)",
                "Nacelle position (°)",
                "Blade angle (pitch position) A (°)",
                "Blade angle (pitch position) B (°)",
                "Blade angle (pitch position) C (°)",
            ),
        ),
        (
            "penmanshiel",
            (
                "Wind direction (°)",
                "Nacelle position (°)",
                "Blade angle (pitch position) A (°)",
                "Blade angle (pitch position) B (°)",
                "Blade angle (pitch position) C (°)",
            ),
        ),
    ],
)
def test_power_wd_yaw_pmean_hist_sincos_masked_selection_emits_value_and_mask_covariates(
    dataset_id: str,
    source_columns: tuple[str, ...],
) -> None:
    selection = _selection(dataset_id, "power_wd_yaw_pmean_hist_sincos_masked", *source_columns)

    assert selection.source_columns == (
        "dataset",
        "turbine_id",
        "timestamp",
        "target_kw",
        "is_observed",
        "quality_flags",
        "feature_quality_flags",
        *source_columns,
        "farm_turbines_expected",
    )
    assert selection.target_history_mask_columns == ("target_kw__mask",)
    assert selection.past_covariate_value_columns == (
        "wind_direction_sin",
        "wind_direction_cos",
        "yaw_error_sin",
        "yaw_error_cos",
        "pitch_mean",
    )
    assert selection.past_covariate_mask_columns == (
        "wind_direction_sin__mask",
        "wind_direction_cos__mask",
        "yaw_error_sin__mask",
        "yaw_error_cos__mask",
        "pitch_mean__mask",
    )
    assert selection.past_covariate_columns == (
        *selection.past_covariate_value_columns,
        *selection.past_covariate_mask_columns,
    )
    assert selection.all_columns == (
        "dataset",
        "turbine_id",
        "timestamp",
        "target_kw",
        "is_observed",
        "quality_flags",
        "feature_quality_flags",
        "target_kw__mask",
        *selection.past_covariate_columns,
        "farm_turbines_expected",
    )
    assert tuple((pair.value_column, pair.mask_column) for pair in selection.target_history_mask_pairs) == (
        ("target_kw", "target_kw__mask"),
    )
    assert tuple((pair.value_column, pair.mask_column) for pair in selection.companion_mask_pairs) == (
        ("wind_direction_sin", "wind_direction_sin__mask"),
        ("wind_direction_cos", "wind_direction_cos__mask"),
        ("yaw_error_sin", "yaw_error_sin__mask"),
        ("yaw_error_cos", "yaw_error_cos__mask"),
        ("pitch_mean", "pitch_mean__mask"),
    )
    assert len(selection.raw_source_mask_rules) == 1
    assert selection.raw_source_mask_rules[0].source_columns == source_columns[-3:]
    assert selection.raw_source_mask_rules[0].minimum_allowed == pytest.approx(-10.0)
    assert selection.raw_source_mask_rules[0].maximum_allowed == pytest.approx(95.0)


@pytest.mark.parametrize(
    ("dataset_id", "source_columns", "lrpm_column"),
    [
        ("kelmarsh", ("Wind direction (°)", "Nacelle position (°)"), "Rotor speed (RPM)"),
        ("penmanshiel", ("Wind direction (°)", "Nacelle position (°)"), "Rotor speed (RPM)"),
        ("hill_of_towie", ("wtc_ActualWindDirection_mean", "wtc_YawPos_mean"), "wtc_MainSRpm_mean"),
    ],
)
def test_power_wd_yaw_lrpm_hist_sincos_selection_orders_lrpm_after_cyclic_covariates(
    dataset_id: str,
    source_columns: tuple[str, ...],
    lrpm_column: str,
) -> None:
    selection = _selection(dataset_id, "power_wd_yaw_lrpm_hist_sincos", *source_columns, lrpm_column)

    assert selection.source_columns == (
        "dataset",
        "turbine_id",
        "timestamp",
        "target_kw",
        "is_observed",
        "quality_flags",
        "feature_quality_flags",
        *source_columns,
        lrpm_column,
        "farm_turbines_expected",
    )
    assert selection.past_covariate_columns == (
        "wind_direction_sin",
        "wind_direction_cos",
        "yaw_error_sin",
        "yaw_error_cos",
        lrpm_column,
    )
    assert selection.all_columns == (
        "dataset",
        "turbine_id",
        "timestamp",
        "target_kw",
        "is_observed",
        "quality_flags",
        "feature_quality_flags",
        "wind_direction_sin",
        "wind_direction_cos",
        "yaw_error_sin",
        "yaw_error_cos",
        lrpm_column,
        "farm_turbines_expected",
    )
    assert len(selection.derived_angle_transforms) == 2
    assert selection.angle_convention == "yaw_error_degrees = wind_direction_degrees - nacelle_or_yaw_position_degrees"


def test_materialize_task_series_frame_builds_direction_sincos_from_dataset_native_angle() -> None:
    selection = _selection("kelmarsh", "power_wd_hist_sincos", "Wind direction (°)")

    frame = materialize_task_series_frame(
        pl.DataFrame(
            {
                "dataset": ["kelmarsh", "kelmarsh"],
                "turbine_id": ["T01", "T01"],
                "timestamp": [datetime(2024, 1, 1, 0, 0), datetime(2024, 1, 1, 0, 10)],
                "target_kw": [100.0, 110.0],
                "is_observed": [True, True],
                "quality_flags": ["", ""],
                "feature_quality_flags": ["", ""],
                "farm_turbines_expected": [6, 6],
                "Wind direction (°)": [90.0, None],
            }
        ),
        selection=selection,
    )

    assert frame.columns == list(selection.all_columns)
    assert frame["wind_direction_sin"].to_list() == [pytest.approx(1.0), None]
    assert frame["wind_direction_cos"].to_list() == [pytest.approx(0.0, abs=1e-12), None]
    assert "Wind direction (°)" not in frame.columns


def test_materialize_task_series_frame_passes_through_ambient_temperature_history() -> None:
    selection = _selection("kelmarsh", "power_atemp_hist", "Nacelle ambient temperature (°C)")

    frame = materialize_task_series_frame(
        pl.DataFrame(
            {
                "dataset": ["kelmarsh", "kelmarsh"],
                "turbine_id": ["T01", "T01"],
                "timestamp": [datetime(2024, 1, 1, 0, 0), datetime(2024, 1, 1, 0, 10)],
                "target_kw": [100.0, 110.0],
                "is_observed": [True, True],
                "quality_flags": ["", ""],
                "feature_quality_flags": ["", ""],
                "farm_turbines_expected": [6, 6],
                "Nacelle ambient temperature (°C)": [17.5, None],
            }
        ),
        selection=selection,
    )

    assert frame.columns == list(selection.all_columns)
    assert frame["Nacelle ambient temperature (°C)"].to_list() == [pytest.approx(17.5), None]


def test_materialize_task_series_frame_passes_through_internal_temperature_history() -> None:
    selection = _selection("hill_of_towie", "power_itemp_hist", "tur_temp__wtc_naceltmp_mean")

    frame = materialize_task_series_frame(
        pl.DataFrame(
            {
                "dataset": ["hill_of_towie", "hill_of_towie"],
                "turbine_id": ["T01", "T01"],
                "timestamp": [datetime(2024, 1, 1, 0, 0), datetime(2024, 1, 1, 0, 10)],
                "target_kw": [100.0, 110.0],
                "is_observed": [True, True],
                "quality_flags": ["", ""],
                "feature_quality_flags": ["", ""],
                "farm_turbines_expected": [21, 21],
                "tur_temp__wtc_naceltmp_mean": [28.0, None],
            }
        ),
        selection=selection,
    )

    assert frame.columns == list(selection.all_columns)
    assert frame["tur_temp__wtc_naceltmp_mean"].to_list() == [pytest.approx(28.0), None]


def test_materialize_task_series_frame_wraps_yaw_error_through_zero_and_360() -> None:
    selection = _selection(
        "kelmarsh",
        "power_wd_yaw_hist_sincos",
        "Wind direction (°)",
        "Nacelle position (°)",
    )

    frame = materialize_task_series_frame(
        pl.DataFrame(
            {
                "dataset": ["kelmarsh", "kelmarsh"],
                "turbine_id": ["T01", "T01"],
                "timestamp": [datetime(2024, 1, 1, 0, 0), datetime(2024, 1, 1, 0, 10)],
                "target_kw": [100.0, 110.0],
                "is_observed": [True, True],
                "quality_flags": ["", ""],
                "feature_quality_flags": ["", ""],
                "farm_turbines_expected": [6, 6],
                "Wind direction (°)": [1.0, 359.0],
                "Nacelle position (°)": [359.0, 1.0],
            }
        ),
        selection=selection,
    )

    sin_2 = math.sin(math.radians(2.0))
    cos_2 = math.cos(math.radians(2.0))

    assert frame["yaw_error_sin"].to_list() == [pytest.approx(sin_2), pytest.approx(-sin_2)]
    assert frame["yaw_error_cos"].to_list() == [pytest.approx(cos_2), pytest.approx(cos_2)]


def test_materialize_task_series_frame_computes_pitch_mean_and_requires_all_three_pitch_inputs() -> None:
    selection = _selection(
        "kelmarsh",
        "power_wd_yaw_pitchmean_hist_sincos",
        "Wind direction (°)",
        "Nacelle position (°)",
        "Blade angle (pitch position) A (°)",
        "Blade angle (pitch position) B (°)",
        "Blade angle (pitch position) C (°)",
    )

    frame = materialize_task_series_frame(
        pl.DataFrame(
            {
                "dataset": ["kelmarsh", "kelmarsh"],
                "turbine_id": ["T01", "T01"],
                "timestamp": [datetime(2024, 1, 1, 0, 0), datetime(2024, 1, 1, 0, 10)],
                "target_kw": [100.0, 110.0],
                "is_observed": [True, True],
                "quality_flags": ["", ""],
                "feature_quality_flags": ["", ""],
                "farm_turbines_expected": [6, 6],
                "Wind direction (°)": [180.0, 180.0],
                "Nacelle position (°)": [174.0, 174.0],
                "Blade angle (pitch position) A (°)": [1.0, 1.0],
                "Blade angle (pitch position) B (°)": [2.0, None],
                "Blade angle (pitch position) C (°)": [3.0, 3.0],
            }
        ),
        selection=selection,
    )

    assert frame["yaw_error_sin"][0] == pytest.approx(math.sin(math.radians(6.0)))
    assert frame["yaw_error_cos"][0] == pytest.approx(math.cos(math.radians(6.0)))
    assert frame["pitch_mean"].to_list() == [pytest.approx(2.0), None]


def test_materialize_task_series_frame_with_masks_masks_out_of_range_pitch_and_emits_binary_masks() -> None:
    selection = _selection(
        "kelmarsh",
        "power_wd_yaw_pmean_hist_sincos_masked",
        "Wind direction (°)",
        "Nacelle position (°)",
        "Blade angle (pitch position) A (°)",
        "Blade angle (pitch position) B (°)",
        "Blade angle (pitch position) C (°)",
    )

    frame = materialize_task_series_frame(
        pl.DataFrame(
            {
                "dataset": ["kelmarsh", "kelmarsh"],
                "turbine_id": ["T01", "T01"],
                "timestamp": [datetime(2024, 1, 1, 0, 0), datetime(2024, 1, 1, 0, 10)],
                "target_kw": [None, 110.0],
                "is_observed": [True, True],
                "quality_flags": ["", ""],
                "feature_quality_flags": ["", ""],
                "farm_turbines_expected": [6, 6],
                "Wind direction (°)": [180.0, None],
                "Nacelle position (°)": [174.0, 174.0],
                "Blade angle (pitch position) A (°)": [120.0, 1.0],
                "Blade angle (pitch position) B (°)": [1.0, 2.0],
                "Blade angle (pitch position) C (°)": [1.0, 3.0],
            }
        ),
        selection=selection,
    )

    assert frame.columns == list(selection.all_columns)
    assert frame["target_kw__mask"].to_list() == [1, 0]
    assert frame["pitch_mean"].to_list() == [None, pytest.approx(2.0)]
    assert frame["pitch_mean__mask"].to_list() == [1, 0]
    assert frame["wind_direction_sin__mask"].to_list() == [0, 1]
    assert frame["wind_direction_cos__mask"].to_list() == [0, 1]
    assert frame["yaw_error_sin__mask"].to_list() == [0, 1]
    assert frame["yaw_error_cos__mask"].to_list() == [0, 1]

    for pair in (*selection.target_history_mask_pairs, *selection.companion_mask_pairs):
        mask_column = frame.get_column(pair.mask_column)
        expected_mask = (
            frame
            .select(pl.col(pair.value_column).is_null().cast(pl.Int8).alias(pair.mask_column))
            .get_column(pair.mask_column)
        )
        assert mask_column.dtype == pl.Int8
        assert mask_column.null_count() == 0
        assert set(mask_column.unique().to_list()).issubset({0, 1})
        assert mask_column.to_list() == expected_mask.to_list()


def test_materialize_task_series_frame_appends_lrpm_after_derived_direction_and_yaw_error() -> None:
    selection = _selection(
        "kelmarsh",
        "power_wd_yaw_lrpm_hist_sincos",
        "Wind direction (°)",
        "Nacelle position (°)",
        "Rotor speed (RPM)",
    )

    frame = materialize_task_series_frame(
        pl.DataFrame(
            {
                "dataset": ["kelmarsh"],
                "turbine_id": ["T01"],
                "timestamp": [datetime(2024, 1, 1, 0, 0)],
                "target_kw": [100.0],
                "is_observed": [True],
                "quality_flags": [""],
                "feature_quality_flags": [""],
                "farm_turbines_expected": [6],
                "Wind direction (°)": [180.0],
                "Nacelle position (°)": [174.0],
                "Rotor speed (RPM)": [12.5],
            }
        ),
        selection=selection,
    )

    assert frame["wind_direction_sin"][0] == pytest.approx(math.sin(math.radians(180.0)))
    assert frame["wind_direction_cos"][0] == pytest.approx(math.cos(math.radians(180.0)))
    assert frame["yaw_error_sin"][0] == pytest.approx(math.sin(math.radians(6.0)))
    assert frame["yaw_error_cos"][0] == pytest.approx(math.cos(math.radians(6.0)))
    assert frame["Rotor speed (RPM)"][0] == pytest.approx(12.5)


def test_materialize_task_series_frame_reconstructs_sdwpf_absolute_wind_direction_for_existing_protocol() -> None:
    selection = _selection("sdwpf_kddcup", "power_wd_hist_sincos", "Ndir", "Wdir")

    frame = materialize_task_series_frame(
        pl.DataFrame(
            {
                "dataset": ["sdwpf_kddcup"],
                "turbine_id": ["1"],
                "timestamp": [datetime(2020, 5, 1, 0, 0)],
                "target_kw": [450.0],
                "is_observed": [True],
                "quality_flags": [""],
                "feature_quality_flags": [""],
                "farm_turbines_expected": [134],
                "Ndir": [5.0],
                "Wdir": [171.0],
            }
        ),
        selection=selection,
    )

    assert frame["wind_direction_sin"][0] == pytest.approx(math.sin(math.radians(176.0)))
    assert frame["wind_direction_cos"][0] == pytest.approx(math.cos(math.radians(176.0)))


def test_materialize_task_series_frame_reconstructs_sdwpf_absolute_wind_direction_and_yaw_error() -> None:
    selection = _selection("sdwpf_kddcup", "power_wd_yaw_hist_sincos", "Ndir", "Wdir")

    frame = materialize_task_series_frame(
        pl.DataFrame(
            {
                "dataset": ["sdwpf_kddcup"],
                "turbine_id": ["1"],
                "timestamp": [datetime(2020, 5, 1, 0, 0)],
                "target_kw": [450.0],
                "is_observed": [True],
                "quality_flags": [""],
                "feature_quality_flags": [""],
                "farm_turbines_expected": [134],
                "Ndir": [5.0],
                "Wdir": [171.0],
            }
        ),
        selection=selection,
    )

    assert frame["wind_direction_sin"][0] == pytest.approx(math.sin(math.radians(176.0)))
    assert frame["wind_direction_cos"][0] == pytest.approx(math.cos(math.radians(176.0)))
    assert frame["yaw_error_sin"][0] == pytest.approx(math.sin(math.radians(171.0)))
    assert frame["yaw_error_cos"][0] == pytest.approx(math.cos(math.radians(171.0)))


def test_protocol_context_records_angle_convention_and_sdwpf_notes() -> None:
    selection = _selection("sdwpf_kddcup", "power_wd_yaw_hist_sincos", "Ndir", "Wdir")

    context = protocol_context_dict(
        dataset_id="sdwpf_kddcup",
        task={"task_id": "next_6h_from_24h"},
        feature_protocol_id="power_wd_yaw_hist_sincos",
        turbine_ids=("1", "2"),
        selection=selection,
        static_columns=("dataset", "turbine_id", "turbine_index"),
    )

    feature_protocol = context["feature_protocol"]

    assert feature_protocol["angle_convention"] == (
        "yaw_error_degrees = wind_direction_degrees - nacelle_or_yaw_position_degrees"
    )
    assert len(feature_protocol["derived_angle_features"]) == 2
    assert feature_protocol["derived_angle_features"][0]["transform_kind"] == "sum"
    assert feature_protocol["derived_angle_features"][1]["transform_kind"] == "direct"
    assert feature_protocol["derived_scalar_features"] == []
    assert feature_protocol["dataset_specific_notes"] == [
        "sdwpf_kddcup reconstructs absolute wind direction as Ndir + Wdir under the repository convention.",
        "sdwpf_kddcup Wdir stores the documented relative yaw-error angle under the repository convention.",
    ]


def test_protocol_context_records_pitch_mean_scalar_feature() -> None:
    selection = _selection(
        "hill_of_towie",
        "power_wd_yaw_pitchmean_hist_sincos",
        "wtc_ActualWindDirection_mean",
        "wtc_YawPos_mean",
        "wtc_PitcPosA_mean",
        "wtc_PitcPosB_mean",
        "wtc_PitcPosC_mean",
    )

    context = protocol_context_dict(
        dataset_id="hill_of_towie",
        task={"task_id": "next_6h_from_24h"},
        feature_protocol_id="power_wd_yaw_pitchmean_hist_sincos",
        turbine_ids=("T01", "T02"),
        selection=selection,
        static_columns=("dataset", "turbine_id", "turbine_index"),
    )

    feature_protocol = context["feature_protocol"]

    assert len(feature_protocol["derived_angle_features"]) == 2
    assert feature_protocol["derived_scalar_features"] == [
        {
            "output_column": "pitch_mean",
            "transform_kind": "row_mean",
            "source_columns": ["wtc_PitcPosA_mean", "wtc_PitcPosB_mean", "wtc_PitcPosC_mean"],
            "description": "Compute the arithmetic mean of the three blade-pitch angles when all three inputs are present.",
            "missing_value_policy": "all_sources_required",
        }
    ]


def test_protocol_context_records_mask_semantics_and_raw_source_mask_rules() -> None:
    selection = _selection(
        "kelmarsh",
        "power_wd_yaw_pmean_hist_sincos_masked",
        "Wind direction (°)",
        "Nacelle position (°)",
        "Blade angle (pitch position) A (°)",
        "Blade angle (pitch position) B (°)",
        "Blade angle (pitch position) C (°)",
    )

    context = protocol_context_dict(
        dataset_id="kelmarsh",
        task={"task_id": "next_6h_from_24h"},
        feature_protocol_id="power_wd_yaw_pmean_hist_sincos_masked",
        turbine_ids=("WT01", "WT02"),
        selection=selection,
        static_columns=("dataset", "turbine_id", "turbine_index"),
    )

    feature_protocol = context["feature_protocol"]
    column_groups = context["column_groups"]

    assert feature_protocol["mask_polarity"] == "1_means_unavailable"
    assert feature_protocol["mask_dtype"] == "int8"
    assert feature_protocol["mask_valid_values"] == [0, 1]
    assert feature_protocol["raw_source_mask_rules"] == [
        {
            "source_columns": [
                "Blade angle (pitch position) A (°)",
                "Blade angle (pitch position) B (°)",
                "Blade angle (pitch position) C (°)",
            ],
            "rule_kind": "outside_closed_interval",
            "description": (
                "Mask raw blade-pitch observations before pitch_mean derivation when the value falls outside "
                "the closed interval [-10, 95] degrees."
            ),
            "affected_output_columns": ["pitch_mean"],
            "minimum_allowed": -10.0,
            "maximum_allowed": 95.0,
        }
    ]
    assert feature_protocol["companion_mask_pairs"] == [
        {"value_column": "wind_direction_sin", "mask_column": "wind_direction_sin__mask"},
        {"value_column": "wind_direction_cos", "mask_column": "wind_direction_cos__mask"},
        {"value_column": "yaw_error_sin", "mask_column": "yaw_error_sin__mask"},
        {"value_column": "yaw_error_cos", "mask_column": "yaw_error_cos__mask"},
        {"value_column": "pitch_mean", "mask_column": "pitch_mean__mask"},
    ]
    assert column_groups["series"] == [
        "dataset",
        "turbine_id",
        "timestamp",
        "target_kw",
        "is_observed",
        "quality_flags",
        "feature_quality_flags",
        "target_kw__mask",
        "wind_direction_sin",
        "wind_direction_cos",
        "yaw_error_sin",
        "yaw_error_cos",
        "pitch_mean",
        "wind_direction_sin__mask",
        "wind_direction_cos__mask",
        "yaw_error_sin__mask",
        "yaw_error_cos__mask",
        "pitch_mean__mask",
        "farm_turbines_expected",
    ]
    assert column_groups["target_history_masks"] == [
        "target_kw__mask",
    ]
    assert column_groups["past_covariates"] == [
        "wind_direction_sin",
        "wind_direction_cos",
        "yaw_error_sin",
        "yaw_error_cos",
        "pitch_mean",
        "wind_direction_sin__mask",
        "wind_direction_cos__mask",
        "yaw_error_sin__mask",
        "yaw_error_cos__mask",
        "pitch_mean__mask",
    ]
    assert column_groups["past_covariate_values"] == [
        "wind_direction_sin",
        "wind_direction_cos",
        "yaw_error_sin",
        "yaw_error_cos",
        "pitch_mean",
    ]
    assert column_groups["past_covariate_masks"] == [
        "wind_direction_sin__mask",
        "wind_direction_cos__mask",
        "yaw_error_sin__mask",
        "yaw_error_cos__mask",
        "pitch_mean__mask",
    ]


def test_world_model_v1_protocol_context_records_observation_and_pairwise_groups() -> None:
    selection = _selection(
        "kelmarsh",
        "world_model_v1",
        "Wind speed (m/s)",
        "Wind direction (°)",
        "Nacelle position (°)",
        "Blade angle (pitch position) A (°)",
        "Blade angle (pitch position) B (°)",
        "Blade angle (pitch position) C (°)",
        "Rotor speed (RPM)",
        "Generator RPM (RPM)",
        "Nacelle ambient temperature (°C)",
        "Nacelle temperature (°C)",
        "evt_any_active",
        "evt_active_count",
        "evt_total_overlap_seconds",
        "evt_stop_active",
        "evt_warning_active",
        "evt_informational_active",
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

    context = protocol_context_dict(
        dataset_id="kelmarsh",
        task={"task_id": "next_6h_from_24h"},
        feature_protocol_id="world_model_v1",
        turbine_ids=("WT01", "WT02"),
        selection=selection,
        static_columns=selection.static_columns,
    )

    feature_protocol = context["feature_protocol"]
    column_groups = context["column_groups"]

    assert feature_protocol["uses_static_covariates"] is True
    assert feature_protocol["uses_known_future_covariates"] is True
    assert feature_protocol["mask_polarity"] == "1_means_unavailable"
    assert column_groups["target_history_masks"] == ["target_kw__mask"]
    assert column_groups["local_observation_values"] == list(selection.local_observation_value_columns)
    assert column_groups["local_observation_masks"] == list(selection.local_observation_mask_columns)
    assert column_groups["global_observation_values"] == list(selection.global_observation_value_columns)
    assert column_groups["global_observation_masks"] == list(selection.global_observation_mask_columns)
    assert column_groups["known_future"] == list(selection.known_future_columns)
    assert column_groups["static"] == list(selection.static_columns)
    assert column_groups["pairwise"] == list(selection.pairwise_columns)


@pytest.mark.parametrize(
    ("feature_protocol_id", "source_column"),
    [
        ("power_atemp_hist", "Nacelle ambient temperature (°C)"),
        ("power_itemp_hist", "Nacelle temperature (°C)"),
    ],
)
def test_protocol_context_records_temperature_history_protocol_without_derived_features(
    feature_protocol_id: str,
    source_column: str,
) -> None:
    selection = _selection("kelmarsh", feature_protocol_id, source_column)

    context = protocol_context_dict(
        dataset_id="kelmarsh",
        task={"task_id": "next_6h_from_24h"},
        feature_protocol_id=feature_protocol_id,
        turbine_ids=("WT01", "WT02"),
        selection=selection,
        static_columns=("dataset", "turbine_id", "turbine_index"),
    )

    feature_protocol = context["feature_protocol"]

    assert feature_protocol["derived_angle_features"] == []
    assert feature_protocol["derived_scalar_features"] == []
    assert feature_protocol["dataset_specific_notes"] == []
    assert feature_protocol["angle_convention"] is None
    assert feature_protocol["past_covariate_source"] is not None


@pytest.mark.parametrize(
    "feature_protocol_id",
    ["power_wd_hist_sincos", "power_ws_wd_hist_sincos"],
)
def test_sdwpf_direction_protocols_require_ndir_for_absolute_wind_direction_reconstruction(
    feature_protocol_id: str,
) -> None:
    with pytest.raises(ValueError, match=r"missing gold-base columns \['Ndir'\]"):
        select_task_series_columns(
            dataset_id="sdwpf_kddcup",
            available_columns={
                "dataset",
                "turbine_id",
                "timestamp",
                "target_kw",
                "is_observed",
                "quality_flags",
                "feature_quality_flags",
                "farm_turbines_expected",
                "Wdir",
                "Wspd",
            },
            feature_protocol_id=feature_protocol_id,
            turbine_static_columns={"coord_x", "coord_y"},
        )


def test_sdwpf_rejects_power_wd_yaw_lrpm_hist_sincos_as_unsupported() -> None:
    with pytest.raises(
        ValueError,
        match=r"feature_protocol_id 'power_wd_yaw_lrpm_hist_sincos' is not supported for dataset 'sdwpf_kddcup'\.",
    ):
        select_task_series_columns(
            dataset_id="sdwpf_kddcup",
            available_columns={
                "dataset",
                "turbine_id",
                "timestamp",
                "target_kw",
                "is_observed",
                "quality_flags",
                "feature_quality_flags",
                "farm_turbines_expected",
                "Ndir",
                "Wdir",
            },
            feature_protocol_id="power_wd_yaw_lrpm_hist_sincos",
            turbine_static_columns={"coord_x", "coord_y"},
        )


@pytest.mark.parametrize("dataset_id", ["hill_of_towie", "sdwpf_kddcup"])
def test_world_model_v1_is_only_supported_for_kelmarsh_and_penmanshiel(dataset_id: str) -> None:
    with pytest.raises(
        ValueError,
        match=rf"feature_protocol_id 'world_model_v1' is not supported for dataset '{dataset_id}'\.",
    ):
        select_task_series_columns(
            dataset_id=dataset_id,
            available_columns={
                "dataset",
                "turbine_id",
                "timestamp",
                "target_kw",
                "is_observed",
                "quality_flags",
                "feature_quality_flags",
                "farm_turbines_expected",
            },
            feature_protocol_id="world_model_v1",
            turbine_static_columns={"coord_x", "coord_y"},
        )


@pytest.mark.parametrize("dataset_id", ["hill_of_towie", "sdwpf_kddcup"])
def test_mask_protocol_is_only_supported_for_kelmarsh_and_penmanshiel(dataset_id: str) -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"feature_protocol_id 'power_wd_yaw_pmean_hist_sincos_masked' is not supported "
            rf"for dataset '{dataset_id}'\."
        ),
    ):
        select_task_series_columns(
            dataset_id=dataset_id,
            available_columns={
                "dataset",
                "turbine_id",
                "timestamp",
                "target_kw",
                "is_observed",
                "quality_flags",
                "feature_quality_flags",
                "farm_turbines_expected",
                "Wind direction (°)",
                "Nacelle position (°)",
                "Blade angle (pitch position) A (°)",
                "Blade angle (pitch position) B (°)",
                "Blade angle (pitch position) C (°)",
                "Ndir",
                "Wdir",
                "Pab1",
                "Pab2",
                "Pab3",
            },
            feature_protocol_id="power_wd_yaw_pmean_hist_sincos_masked",
            turbine_static_columns={"coord_x", "coord_y"},
        )


@pytest.mark.parametrize(
    "feature_protocol_id",
    [
        "power_ws_hist",
        "power_atemp_hist",
        "power_itemp_hist",
        "power_wd_hist_sincos",
        "power_ws_wd_hist_sincos",
        "power_wd_yaw_hist_sincos",
        "power_wd_yaw_pitchmean_hist_sincos",
        "power_wd_yaw_pmean_hist_sincos_masked",
        "power_wd_yaw_lrpm_hist_sincos",
        "world_model_v1",
    ],
)
def test_protocol_selection_rejects_missing_mapped_columns(feature_protocol_id: str) -> None:
    with pytest.raises(ValueError, match="missing gold-base columns"):
        select_task_series_columns(
            dataset_id="kelmarsh",
            available_columns={
                "dataset",
                "turbine_id",
                "timestamp",
                "target_kw",
                "is_observed",
                "quality_flags",
                "feature_quality_flags",
            },
            feature_protocol_id=feature_protocol_id,
            turbine_static_columns={"coord_x", "coord_y"},
        )

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
        "power_wd_hist_sincos",
        "power_ws_wd_hist_sincos",
        "power_wd_yaw_hist_sincos",
        "power_wd_yaw_pitchmean_hist_sincos",
        "power_wd_yaw_lrpm_hist_sincos",
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

    power_wd_yaw_lrpm_hist = get_feature_protocol_spec("power_wd_yaw_lrpm_hist_sincos")
    assert power_wd_yaw_lrpm_hist.feature_protocol_id == "power_wd_yaw_lrpm_hist_sincos"
    assert power_wd_yaw_lrpm_hist.uses_target_history is True
    assert power_wd_yaw_lrpm_hist.uses_past_covariates is True


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


@pytest.mark.parametrize(
    "feature_protocol_id",
    [
        "power_ws_hist",
        "power_wd_hist_sincos",
        "power_ws_wd_hist_sincos",
        "power_wd_yaw_hist_sincos",
        "power_wd_yaw_pitchmean_hist_sincos",
        "power_wd_yaw_lrpm_hist_sincos",
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

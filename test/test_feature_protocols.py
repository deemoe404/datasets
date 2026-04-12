from __future__ import annotations

from datetime import datetime

import pytest
import polars as pl

from wind_datasets.feature_protocols import (
    DEFAULT_FEATURE_PROTOCOL_ID,
    get_feature_protocol_spec,
    list_feature_protocol_ids,
    materialize_task_series_frame,
    select_task_series_columns,
)


def test_active_feature_protocols_are_registered() -> None:
    assert DEFAULT_FEATURE_PROTOCOL_ID == "power_only"
    assert list_feature_protocol_ids() == (
        "power_only",
        "power_ws_hist",
        "power_wd_hist_sincos",
        "power_ws_wd_hist_sincos",
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


def test_unknown_feature_protocol_is_rejected() -> None:
    with pytest.raises(ValueError):
        get_feature_protocol_spec("unexpected_protocol")


def test_power_only_selection_keeps_only_base_series_columns() -> None:
    selection = select_task_series_columns(
        dataset_id="kelmarsh",
        available_columns={
            "dataset",
            "turbine_id",
            "timestamp",
            "target_kw",
            "is_observed",
            "quality_flags",
            "feature_quality_flags",
            "farm_turbines_expected",
            "Wind speed (m/s)",
        },
        feature_protocol_id="power_only",
        turbine_static_columns={"coord_x", "coord_y"},
    )

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
    selection = select_task_series_columns(
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
            feature_column,
        },
        feature_protocol_id="power_ws_hist",
        turbine_static_columns={"coord_x", "coord_y"},
    )

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
    assert selection.all_columns == (
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


@pytest.mark.parametrize(
    ("dataset_id", "feature_column"),
    [
        ("kelmarsh", "Wind direction (°)"),
        ("penmanshiel", "Wind direction (°)"),
        ("hill_of_towie", "wtc_ActualWindDirection_mean"),
        ("sdwpf_kddcup", "Wdir"),
    ],
)
def test_power_wd_hist_sincos_selection_uses_dataset_native_direction_source(
    dataset_id: str,
    feature_column: str,
) -> None:
    selection = select_task_series_columns(
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
            feature_column,
        },
        feature_protocol_id="power_wd_hist_sincos",
        turbine_static_columns={"coord_x", "coord_y"},
    )

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


@pytest.mark.parametrize(
    ("dataset_id", "ws_column", "wd_column"),
    [
        ("kelmarsh", "Wind speed (m/s)", "Wind direction (°)"),
        ("penmanshiel", "Wind speed (m/s)", "Wind direction (°)"),
        ("hill_of_towie", "wtc_AcWindSp_mean", "wtc_ActualWindDirection_mean"),
        ("sdwpf_kddcup", "Wspd", "Wdir"),
    ],
)
def test_power_ws_wd_hist_sincos_selection_orders_ws_then_direction_sincos(
    dataset_id: str,
    ws_column: str,
    wd_column: str,
) -> None:
    selection = select_task_series_columns(
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
            ws_column,
            wd_column,
        },
        feature_protocol_id="power_ws_wd_hist_sincos",
        turbine_static_columns={"coord_x", "coord_y"},
    )

    assert selection.source_columns == (
        "dataset",
        "turbine_id",
        "timestamp",
        "target_kw",
        "is_observed",
        "quality_flags",
        "feature_quality_flags",
        ws_column,
        wd_column,
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


def test_materialize_task_series_frame_builds_direction_sincos_from_task_selection() -> None:
    selection = select_task_series_columns(
        dataset_id="kelmarsh",
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
        },
        feature_protocol_id="power_wd_hist_sincos",
        turbine_static_columns={"coord_x", "coord_y"},
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
                "Wind direction (°)": [90.0, None],
            }
        ),
        selection=selection,
    )

    assert frame.columns == list(selection.all_columns)
    assert frame["wind_direction_sin"].to_list() == [pytest.approx(1.0), None]
    assert frame["wind_direction_cos"].to_list() == [pytest.approx(0.0, abs=1e-12), None]
    assert "Wind direction (°)" not in frame.columns


@pytest.mark.parametrize(
    "feature_protocol_id",
    [
        "power_ws_hist",
        "power_wd_hist_sincos",
        "power_ws_wd_hist_sincos",
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

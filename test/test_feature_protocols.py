from __future__ import annotations

import pytest

from wind_datasets.feature_protocols import (
    DEFAULT_FEATURE_PROTOCOL_ID,
    get_feature_protocol_spec,
    list_feature_protocol_ids,
    select_task_series_columns,
)


def test_active_feature_protocols_are_registered() -> None:
    assert DEFAULT_FEATURE_PROTOCOL_ID == "power_only"
    assert list_feature_protocol_ids() == ("power_only", "power_ws_hist")

    power_only = get_feature_protocol_spec("power_only")
    assert power_only.feature_protocol_id == "power_only"
    assert power_only.uses_target_history is True
    assert power_only.uses_past_covariates is False

    power_ws_hist = get_feature_protocol_spec("power_ws_hist")
    assert power_ws_hist.feature_protocol_id == "power_ws_hist"
    assert power_ws_hist.uses_target_history is True
    assert power_ws_hist.uses_past_covariates is True
    assert power_ws_hist.uses_known_future_covariates is False


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


def test_power_ws_hist_selection_rejects_missing_mapped_columns() -> None:
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
            feature_protocol_id="power_ws_hist",
            turbine_static_columns={"coord_x", "coord_y"},
        )

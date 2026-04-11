from __future__ import annotations

import pytest

from wind_datasets.feature_protocols import (
    DEFAULT_FEATURE_PROTOCOL_ID,
    get_feature_protocol_spec,
    list_feature_protocol_ids,
    select_task_series_columns,
)


def test_only_power_only_protocol_is_registered() -> None:
    assert DEFAULT_FEATURE_PROTOCOL_ID == "power_only"
    assert list_feature_protocol_ids() == ("power_only",)

    spec = get_feature_protocol_spec("power_only")
    assert spec.feature_protocol_id == "power_only"
    assert spec.uses_target_history is True
    assert spec.uses_static_covariates is False
    assert spec.uses_known_future_covariates is False
    assert spec.uses_past_covariates is False
    assert spec.uses_target_derived_covariates is False


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

from __future__ import annotations

from datetime import datetime
from math import sqrt
from pathlib import Path

import polars as pl

from wind_datasets.datasets.hill_of_towie import HillOfTowieDatasetBuilder
from wind_datasets.models import DatasetSpec, TaskSpec
from wind_datasets.persistence import _evaluate_persistence_series

from .helpers import build_hill_fixture


def test_persistence_repeats_anchor_value_across_all_horizons() -> None:
    spec = DatasetSpec(
        dataset_id="synthetic_persistence",
        source_root=Path("."),
        resolution_minutes=10,
        turbine_ids=("T01",),
        target_column="target_kw",
        target_unit="kW",
        timezone_policy="naive",
        timestamp_convention="naive",
        default_feature_groups=("main",),
        handler="synthetic",
    )
    task = TaskSpec(
        history_duration="30m",
        forecast_duration="20m",
        task_id="short_persistence",
    ).resolve(spec.resolution_minutes)
    timestamps = pl.datetime_range(
        datetime(2024, 1, 1, 0, 0, 0),
        datetime(2024, 1, 1, 1, 10, 0),
        interval="10m",
        eager=True,
    )
    series = pl.DataFrame(
        {
            "dataset": ["synthetic_persistence"] * len(timestamps),
            "turbine_id": ["T01"] * len(timestamps),
            "timestamp": timestamps,
            "target_kw": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "is_observed": [True] * len(timestamps),
            "quality_flags": [""] * len(timestamps),
        }
    )

    result = _evaluate_persistence_series(
        series,
        spec=spec,
        quality_profile="default",
        task=task,
        rated_power_by_turbine={"T01": 2.0},
    )

    summary = result.summary.row(0, named=True)
    per_horizon = result.per_horizon.to_dicts()
    per_turbine = result.per_turbine.row(0, named=True)

    assert summary["eligible_windows"] == 4
    assert summary["prediction_count"] == 8
    assert summary["mae_kw"] == 1.5
    assert summary["rmse_kw"] == sqrt(2.5)
    assert summary["mae_pu"] == 0.75
    assert summary["rmse_pu"] == sqrt(0.625)

    assert per_horizon == [
        {
            "horizon_step": 1,
            "horizon_minutes": 10,
            "n_predictions": 4,
            "mae_kw": 1.0,
            "rmse_kw": 1.0,
            "mae_pu": 0.5,
            "rmse_pu": 0.5,
        },
        {
            "horizon_step": 2,
            "horizon_minutes": 20,
            "n_predictions": 4,
            "mae_kw": 2.0,
            "rmse_kw": 2.0,
            "mae_pu": 1.0,
            "rmse_pu": 1.0,
        },
    ]
    assert per_turbine["rated_power_kw"] == 2.0
    assert per_turbine["eligible_windows"] == 4
    assert per_turbine["prediction_count"] == 8
    assert per_turbine["mae_kw"] == 1.5
    assert per_turbine["rmse_kw"] == sqrt(2.5)
    assert per_turbine["mae_pu"] == 0.75
    assert per_turbine["rmse_pu"] == sqrt(0.625)


def test_persistence_eligible_windows_match_clean_window_index(tmp_path) -> None:
    spec = build_hill_fixture(tmp_path / "raw" / "hill")
    builder = HillOfTowieDatasetBuilder(spec=spec, cache_root=tmp_path / "cache")
    task_spec = TaskSpec(
        history_duration="20m",
        forecast_duration="10m",
        task_id="persistence_check",
        granularity="turbine",
    )
    resolved_task = task_spec.resolve(spec.resolution_minutes)

    builder.build_gold_base(layout="turbine")
    builder.build_task_cache(task_spec)
    series = builder.load_series(layout="turbine")
    window_index = builder.load_window_index(task_spec)

    result = _evaluate_persistence_series(
        series,
        spec=spec,
        quality_profile="default",
        task=resolved_task,
    )

    strict_window_index = window_index.filter(
        pl.col("is_complete_input")
        & pl.col("is_complete_output")
        & (pl.col("quality_flags") == "")
    )
    strict_counts = {
        row["turbine_id"]: row["len"]
        for row in strict_window_index.group_by("turbine_id").len().to_dicts()
    }
    per_turbine_counts = {
        row["turbine_id"]: row["eligible_windows"]
        for row in result.per_turbine.to_dicts()
    }

    assert result.summary["eligible_windows"][0] == strict_window_index.height
    assert result.summary["prediction_count"][0] == strict_window_index.height * resolved_task.forecast_steps
    assert result.summary["rated_power_kw"][0] == 2300.0
    assert per_turbine_counts == {"T01": 4, "T02": 2}
    assert per_turbine_counts == {"T01": strict_counts["T01"], "T02": strict_counts["T02"]}
    assert set(result.per_turbine["rated_power_kw"].to_list()) == {2300.0}

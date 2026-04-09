from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import polars as pl

from wind_datasets.datasets.common import (
    build_farm_window_index,
    build_farm_window_index_from_series_path,
)
from wind_datasets.models import TaskSpec


def _resolved_farm_task() -> object:
    return TaskSpec(
        history_duration="20m",
        forecast_duration="10m",
        task_id="farm_short",
        granularity="farm",
    ).resolve(10)


def _read_window_index(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path).sort(["dataset", "input_start_ts", "input_end_ts", "output_end_ts"])


def test_build_farm_window_index_from_series_path_matches_in_memory_for_turbine_major_parquet(
    tmp_path,
) -> None:
    task = _resolved_farm_task()
    timestamps = [
        datetime(2024, 1, 1, 0, 0),
        datetime(2024, 1, 1, 0, 10),
        datetime(2024, 1, 1, 0, 20),
        datetime(2024, 1, 1, 0, 30),
    ]
    frame = pl.DataFrame(
        {
            "dataset": ["kelmarsh"] * 8,
            "turbine_id": ["Kelmarsh 1"] * 4 + ["Kelmarsh 2"] * 4,
            "timestamp": timestamps + timestamps,
            "target_kw": [100.0, 101.0, 102.0, 103.0, 200.0, 201.0, None, 203.0],
            "is_observed": [True, True, True, True, True, True, True, True],
            "quality_flags": ["", "sensor_issue", "", "", "", "", "missing_target", ""],
            "feature_quality_flags": ["", "", "", "", "", "", "", "farm_grid_conflict"],
            "farm_turbines_expected": [2] * 8,
        }
    )

    in_memory_output = tmp_path / "in_memory" / "window_index.parquet"
    in_memory_report = tmp_path / "in_memory" / "task_report.json"
    build_farm_window_index(
        frame.sort(["dataset", "timestamp", "turbine_id"]),
        task,
        in_memory_output,
        in_memory_report,
        "default",
    )

    series_path = tmp_path / "series.parquet"
    frame.sort(["dataset", "turbine_id", "timestamp"]).write_parquet(series_path)
    path_output = tmp_path / "from_path" / "window_index.parquet"
    path_report = tmp_path / "from_path" / "task_report.json"
    build_farm_window_index_from_series_path(
        series_path=series_path,
        task=task,
        output_path=path_output,
        report_path=path_report,
        quality_profile="default",
        available_columns=set(frame.columns),
    )

    assert _read_window_index(path_output).to_dicts() == _read_window_index(in_memory_output).to_dicts()
    assert json.loads(path_report.read_text()) == json.loads(in_memory_report.read_text())


def test_build_farm_window_index_from_series_path_prefers_farm_expected_turbines_column(tmp_path) -> None:
    task = _resolved_farm_task()
    timestamps = [
        datetime(2024, 1, 1, 0, 0),
        datetime(2024, 1, 1, 0, 10),
        datetime(2024, 1, 1, 0, 20),
        datetime(2024, 1, 1, 0, 30),
    ]
    frame = pl.DataFrame(
        {
            "dataset": ["kelmarsh"] * 4,
            "turbine_id": ["Kelmarsh 1"] * 4,
            "timestamp": timestamps,
            "target_kw": [100.0, 101.0, 102.0, 103.0],
            "is_observed": [True, True, True, True],
            "quality_flags": ["", "", "", ""],
            "feature_quality_flags": ["", "", "", ""],
            "farm_turbines_expected": [2] * 4,
        }
    )

    series_path = tmp_path / "series.parquet"
    frame.write_parquet(series_path)
    output_path = tmp_path / "window_index.parquet"
    report_path = tmp_path / "task_report.json"
    build_farm_window_index_from_series_path(
        series_path=series_path,
        task=task,
        output_path=output_path,
        report_path=report_path,
        quality_profile="default",
        available_columns=set(frame.columns),
    )

    window_index = pl.read_parquet(output_path)

    assert window_index.height == 2
    assert window_index["farm_turbines_expected"].unique().to_list() == [2]
    assert window_index["is_complete_input"].to_list() == [False, False]
    assert window_index["is_complete_output"].to_list() == [False, False]


def test_penmanshiel_farm_windows_continue_past_early_turbine_end_as_partial(tmp_path) -> None:
    task = _resolved_farm_task()
    timestamps = [
        datetime(2023, 12, 31, 23, 20),
        datetime(2023, 12, 31, 23, 30),
        datetime(2023, 12, 31, 23, 40),
        datetime(2023, 12, 31, 23, 50),
        datetime(2024, 1, 1, 0, 0),
        datetime(2024, 1, 1, 0, 10),
    ]
    frame = pl.DataFrame(
        {
            "dataset": ["penmanshiel"] * 12,
            "turbine_id": (["Penmanshiel 01"] * 6) + (["Penmanshiel 11"] * 6),
            "timestamp": timestamps + timestamps,
            "target_kw": [
                100.0,
                101.0,
                102.0,
                103.0,
                None,
                None,
                200.0,
                201.0,
                202.0,
                203.0,
                204.0,
                205.0,
            ],
            "is_observed": [
                True,
                True,
                True,
                True,
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            "quality_flags": [
                "",
                "",
                "",
                "",
                "missing_row|missing_target",
                "missing_row|missing_target",
                "",
                "",
                "",
                "",
                "",
                "",
            ],
            "feature_quality_flags": [""] * 12,
            "farm_turbines_expected": [2] * 12,
        }
    ).sort(["dataset", "turbine_id", "timestamp"])

    series_path = tmp_path / "penmanshiel_series.parquet"
    output_path = tmp_path / "window_index.parquet"
    report_path = tmp_path / "task_report.json"
    frame.write_parquet(series_path)
    build_farm_window_index_from_series_path(
        series_path=series_path,
        task=task,
        output_path=output_path,
        report_path=report_path,
        quality_profile="default",
        available_columns=set(frame.columns),
    )

    window_index = _read_window_index(output_path)
    report = json.loads(report_path.read_text())

    assert window_index.height == 4
    assert report["window_count"] == 4
    assert report["complete_output_windows"] == 2
    assert window_index["farm_turbines_expected"].unique().to_list() == [2]

    complete_output = window_index.filter(pl.col("is_complete_output"))
    assert complete_output["output_end_ts"].max() == datetime(2023, 12, 31, 23, 50)

    partial_tail = window_index.filter(pl.col("output_end_ts") > datetime(2023, 12, 31, 23, 50))
    assert partial_tail.height == 2
    assert partial_tail["is_complete_output"].to_list() == [False, False]
    assert partial_tail["output_turbines_with_target_min"].to_list() == [1, 1]
    assert partial_tail["quality_flags"].to_list() == [
        "partial_output|row_quality_issues",
        "partial_input|partial_output|row_quality_issues",
    ]

from __future__ import annotations

import json

import polars as pl
import pytest

from wind_datasets.datasets.greenbyte import GreenbyteDatasetBuilder
from wind_datasets.datasets.hill_of_towie import HillOfTowieDatasetBuilder
from wind_datasets.datasets.sdwpf_kddcup import SDWPFKDDCupDatasetBuilder
from wind_datasets.models import TaskSpec

from .helpers import build_greenbyte_fixture, build_hill_fixture, build_sdwpf_kddcup_fixture


def _assert_unique_series_keys(series: pl.DataFrame) -> None:
    grouped = series.group_by(["dataset", "turbine_id", "timestamp"]).len()
    assert grouped.filter(pl.col("len") > 1).is_empty()


def test_end_to_end_greenbyte_pipeline(tmp_path) -> None:
    spec = build_greenbyte_fixture(tmp_path / "raw" / "kelmarsh", "Kelmarsh", "Kelmarsh 1")
    builder = GreenbyteDatasetBuilder(spec=spec, cache_root=tmp_path / "cache")

    builder.build_gold_base()
    series = builder.load_series()
    turbine_static = builder.load_turbine_static()
    report = builder.profile_dataset()

    assert "quality_flags" in series.columns
    assert "feature_quality_flags" in series.columns
    assert "farm_turbines_observed" in series.columns
    assert "farm_is_fully_synchronous" in series.columns
    assert "farm_pmu__gms_power_kw" in series.columns
    assert "farm_grid_meter__grid_meter_energy_export_kwh" in series.columns
    assert "evt_stop_active" in series.columns
    assert "evt_status_code__710" in series.columns
    assert "farm_evt_status__communication" in series.columns
    assert turbine_static.height == len(spec.turbine_ids)
    assert turbine_static["turbine_id"].to_list() == [spec.turbine_ids[0]]
    observed_row = series.filter(pl.col("timestamp").dt.strftime("%Y-%m-%d %H:%M:%S") == "2024-01-01 00:20:00")
    assert observed_row["farm_pmu__gms_power_kw"][0] == 920.0
    assert observed_row["farm_grid_meter__grid_meter_energy_export_kwh"][0] == 102.0
    assert observed_row["evt_status__informational"][0] is True
    assert observed_row["farm_evt_status__communication"][0] is True
    assert series.filter(pl.col("timestamp").dt.strftime("%Y-%m-%d %H:%M:%S") == "2024-01-01 00:40:00")[
        "is_observed"
    ][0] is False
    _assert_unique_series_keys(series)
    assert report["conflict_value_count"] == 1
    assert report["layout"] == "farm"
    assert report["feature_set"] == "default"
    assert "long_gaps" in report

    task = TaskSpec(
        history_duration="30m",
        forecast_duration="30m",
        task_id="short_task",
        granularity="turbine",
    )
    builder.build_task_cache(task)
    window_index = builder.load_window_index(task)
    task_report = json.loads(
        builder.cache_paths.task_report_path_for("default", "turbine", "short_task").read_text()
    )
    assert window_index.height > 0
    assert "turbine_id" in window_index.columns
    assert "quality_flags" in window_index.columns
    assert "feature_quality_flags" in window_index.columns
    assert set(series["feature_quality_flags"].unique().to_list()) == {""}
    assert task_report["task"]["task_id"] == "short_task"
    assert task_report["window_count"] == window_index.height


def test_end_to_end_hill_pipeline(tmp_path) -> None:
    spec = build_hill_fixture(tmp_path / "raw" / "hill")
    builder = HillOfTowieDatasetBuilder(spec=spec, cache_root=tmp_path / "cache")

    builder.build_gold_base()
    builder.build_gold_base(layout="turbine")
    series = builder.load_series()
    turbine_series = builder.load_series(layout="turbine")
    turbine_static = builder.load_turbine_static()
    report = builder.profile_dataset()
    duplicate_audit = pl.read_parquet(builder.cache_paths.hill_duplicate_audit_path)
    shutdown = pl.read_parquet(builder.cache_paths.silver_shared_ts_path("turbine_shutdown_duration"))

    _assert_unique_series_keys(series)
    assert turbine_static.height == len(spec.turbine_ids)
    assert "farm_grid__activepower" in series.columns
    assert "farm_grid_sci__activepowermean" in series.columns
    assert "tur_count__wtc_turrdhrt_endvalue" in series.columns
    assert "tur_digi_in__wtc_smokenac_counts" in series.columns
    assert "tur_digi_out__wtc_turbinok_counts" in series.columns
    assert "tur_intern__wtc_valsuppv_mean" in series.columns
    assert "tur_press__wtc_brakpres_mean" in series.columns
    assert "tur_temp__wtc_ambietmp_mean" in series.columns
    assert "shutdown_duration_s" in series.columns
    assert "alarm_any_active" in series.columns
    assert "alarm_code__42" in series.columns
    assert "aeroup_post_install" in series.columns
    assert "aeroup_in_install_window" in series.columns
    assert "tuneup_in_deployment_window" in series.columns
    assert "tuneup_post_effective" in series.columns
    assert "farm_turbines_observed" in series.columns
    assert "farm_turbines_with_target" in series.columns
    assert "feature_quality_flags" in series.columns
    assert "farm_turbines_observed" not in turbine_series.columns
    assert duplicate_audit.height == 6
    assert duplicate_audit.filter(pl.col("duplicate_kind") == "true_conflict").height == 3
    assert duplicate_audit.filter(pl.col("duplicate_kind") == "normalized_equal").height == 1
    assert duplicate_audit.filter(pl.col("duplicate_kind") == "identical").height == 2
    grid_conflict = duplicate_audit.filter(
        (pl.col("table_name") == "tblGrid") & (pl.col("duplicate_kind") == "true_conflict")
    )
    assert grid_conflict["affected_series_columns"].to_list() == [["farm_grid__activepower"]]
    temp_conflict = duplicate_audit.filter(
        (pl.col("table_name") == "tblSCTurTemp") & (pl.col("duplicate_kind") == "true_conflict")
    )
    assert temp_conflict["affected_series_columns"].to_list() == [["tur_temp__wtc_ambietmp_mean"]]
    default_conflict = duplicate_audit.filter(
        (pl.col("table_name") == "tblSCTurbine") & (pl.col("duplicate_kind") == "true_conflict")
    )
    assert default_conflict["conflicting_source_columns"].to_list() == [["wtc_PrWindSp_mean"]]
    assert shutdown["timestamp"].null_count() == 0
    assert shutdown.height == 4
    assert shutdown.filter(
        (pl.col("turbine_id") == "T01")
        & (pl.col("timestamp").dt.strftime("%Y-%m-%d %H:%M:%S") == "2024-03-14 17:40:00")
    )["shutdown_duration_s"][0] == 600.0
    assert shutdown.filter(
        (pl.col("turbine_id") == "T01")
        & (pl.col("timestamp").dt.strftime("%Y-%m-%d %H:%M:%S") == "2024-03-14 17:30:00")
    ).is_empty()
    t01_1730 = series.filter(
        (pl.col("turbine_id") == "T01")
        & (pl.col("timestamp").dt.strftime("%Y-%m-%d %H:%M:%S") == "2024-03-14 17:30:00")
    )
    t01_1740 = series.filter(
        (pl.col("turbine_id") == "T01")
        & (pl.col("timestamp").dt.strftime("%Y-%m-%d %H:%M:%S") == "2024-03-14 17:40:00")
    )
    t01_1750 = series.filter(
        (pl.col("turbine_id") == "T01")
        & (pl.col("timestamp").dt.strftime("%Y-%m-%d %H:%M:%S") == "2024-03-14 17:50:00")
    )
    t01_1720 = series.filter(
        (pl.col("turbine_id") == "T01")
        & (pl.col("timestamp").dt.strftime("%Y-%m-%d %H:%M:%S") == "2024-03-14 17:20:00")
    )
    t02_1720 = series.filter(
        (pl.col("turbine_id") == "T02")
        & (pl.col("timestamp").dt.strftime("%Y-%m-%d %H:%M:%S") == "2024-03-14 17:20:00")
    )
    t02_1750 = series.filter(
        (pl.col("turbine_id") == "T02")
        & (pl.col("timestamp").dt.strftime("%Y-%m-%d %H:%M:%S") == "2024-03-14 17:50:00")
    )
    t01_1810 = series.filter(
        (pl.col("turbine_id") == "T01")
        & (pl.col("timestamp").dt.strftime("%Y-%m-%d %H:%M:%S") == "2024-03-14 18:10:00")
    )
    t02_1810 = series.filter(
        (pl.col("turbine_id") == "T02")
        & (pl.col("timestamp").dt.strftime("%Y-%m-%d %H:%M:%S") == "2024-03-14 18:10:00")
    )
    assert t01_1730["farm_grid__activepower"][0] == 2020.0
    assert t01_1730["farm_grid_sci__activepowermean"][0] == 2020.0
    assert t01_1730["shutdown_duration_s"][0] == 0.0
    assert t01_1740["shutdown_duration_s"][0] == 600.0
    assert t01_1750["shutdown_duration_s"][0] == 300.0
    assert t02_1750["shutdown_duration_s"][0] == 120.0
    assert turbine_series.filter(
        (pl.col("turbine_id") == "T01")
        & (pl.col("timestamp").dt.strftime("%Y-%m-%d %H:%M:%S") == "2024-03-14 17:40:00")
    )["shutdown_duration_s"][0] == 600.0
    assert t01_1730["alarm_any_active"][0] is True
    assert t01_1730["alarm_code__42"][0] is True
    assert t01_1730["farm_turbines_observed"][0] == 2
    assert t01_1730["farm_turbines_with_target"][0] == 1
    assert t01_1730["farm_is_fully_synchronous"][0] is True
    assert t01_1720["aeroup_post_install"][0] is True
    assert t02_1720["aeroup_in_install_window"][0] is True
    assert t02_1720["tuneup_in_deployment_window"][0] is False
    assert t02_1720["tuneup_post_effective"][0] is True
    assert t01_1720["tuneup_in_deployment_window"][0] is False
    assert t01_1720["tuneup_post_effective"][0] is False
    assert t02_1810["tuneup_post_effective"][0] is True
    assert "duplicate_conflict_resolved" in series.filter(
        (pl.col("turbine_id") == "T01")
        & (pl.col("timestamp").dt.strftime("%Y-%m-%d %H:%M:%S") == "2024-03-14 18:10:00")
    )["quality_flags"][0]
    assert series.filter(
        (pl.col("turbine_id") == "T02")
        & (pl.col("timestamp").dt.strftime("%Y-%m-%d %H:%M:%S") == "2024-03-14 17:30:00")
    )["target_kw"][0] is None
    assert series.filter(
        (pl.col("turbine_id") == "T01")
        & (pl.col("timestamp").dt.strftime("%Y-%m-%d %H:%M:%S") == "2024-03-14 18:10:00")
    )["wtc_PrWindSp_mean"][0] is None
    assert t01_1810["target_kw"][0] == 1060.0
    assert t01_1810["farm_grid__activepower"][0] is None
    assert t01_1810["farm_grid__frequency"][0] == 50.0
    assert "feature_source_conflict__farm_grid" in t01_1810["feature_quality_flags"][0]
    assert t02_1810["farm_grid__activepower"][0] is None
    assert t02_1810["tur_temp__wtc_ambietmp_mean"][0] is None
    assert "feature_source_conflict__farm_grid" in t02_1810["feature_quality_flags"][0]
    assert "feature_source_conflict__turbine_temp" in t02_1810["feature_quality_flags"][0]
    assert report["target_missing_count"] >= 1
    assert report["duplicate_audit_count"] == 6
    assert report["duplicate_true_conflict_count"] == 3
    assert report["duplicate_true_conflict_count_by_table"] == {
        "tblGrid": 1,
        "tblSCTurTemp": 1,
        "tblSCTurbine": 1,
    }
    assert report["row_conflict_row_count"] == 1
    assert report["feature_conflict_row_count"] == 2
    assert report["layout"] == "farm"
    assert report["full_synchrony_ratio"] == 1.0
    assert report["full_target_ratio"] < 1.0

    farm_task = TaskSpec(
        history_duration="20m",
        forecast_duration="10m",
        task_id="farm_short",
        granularity="farm",
    )
    builder.build_task_cache(farm_task)
    farm_window_index = builder.load_window_index(farm_task)
    farm_task_static = builder.load_task_turbine_static(farm_task)
    farm_task_report = json.loads(
        builder.cache_paths.task_report_path_for("default", "farm", "farm_short").read_text()
    )
    turbine_task = TaskSpec(
        history_duration="20m",
        forecast_duration="10m",
        task_id="turbine_short",
        granularity="turbine",
    )
    builder.build_task_cache(turbine_task)
    turbine_window_index = builder.load_window_index(turbine_task)
    turbine_task_report = json.loads(
        builder.cache_paths.task_report_path_for("default", "turbine", "turbine_short").read_text()
    )
    output_gap_window = farm_window_index.filter(
        pl.col("input_end_ts").dt.strftime("%Y-%m-%d %H:%M:%S") == "2024-03-14 17:20:00"
    )
    assert "turbine_id" not in farm_window_index.columns
    assert "input_turbines_observed_per_step" in farm_window_index.columns
    assert "output_turbines_with_target_per_step" in farm_window_index.columns
    assert "feature_quality_flags" in farm_window_index.columns
    assert output_gap_window["output_turbines_with_target_min"][0] == 1
    assert output_gap_window["is_complete_output"][0] is False
    assert farm_task_static["turbine_index"].to_list() == [0, 1]
    assert farm_task_report["window_count"] == farm_window_index.height
    assert farm_task_report["window_count"] > 0
    assert turbine_window_index.height > 0
    assert "turbine_id" in turbine_window_index.columns
    assert "feature_quality_flags" in turbine_window_index.columns
    feature_only_window = turbine_window_index.filter(
        (pl.col("turbine_id") == "T02")
        & (pl.col("input_end_ts").dt.strftime("%Y-%m-%d %H:%M:%S") == "2024-03-14 18:00:00")
    )
    assert feature_only_window["quality_flags"][0] == ""
    assert feature_only_window["feature_quality_flags"][0] == "feature_quality_issues_output"
    assert turbine_task_report["window_count"] == turbine_window_index.height
    assert turbine_task_report["window_count"] > 0


def test_end_to_end_sdwpf_pipeline_and_task_switch_only_updates_task_cache(tmp_path) -> None:
    spec = build_sdwpf_kddcup_fixture(tmp_path / "raw" / "sdwpf")
    builder = SDWPFKDDCupDatasetBuilder(spec=spec, cache_root=tmp_path / "cache")

    gold_path = builder.build_gold_base()
    assert gold_path.parts[-4:] == ("default", "farm", "default", "series.parquet")
    before_mtime = gold_path.stat().st_mtime_ns

    default_series = builder.load_series()
    explicit_default_series = builder.load_series("default")
    turbine_static = builder.load_turbine_static()
    report = builder.profile_dataset()
    manifest = json.loads(builder.cache_paths.manifest_path.read_text())

    default_negative = default_series.filter(
        (pl.col("turbine_id") == "1")
        & (pl.col("timestamp").dt.strftime("%Y-%m-%d %H:%M:%S") == "2020-05-01 00:10:00")
    )

    assert before_mtime == gold_path.stat().st_mtime_ns
    assert default_series.equals(explicit_default_series)
    _assert_unique_series_keys(default_series)
    assert turbine_static.height == len(spec.turbine_ids)
    assert "farm_turbines_observed" in default_series.columns
    assert turbine_static["elevation_m"].null_count() == turbine_static.height
    assert manifest["time_semantics_check"]["calendar_anchor_date"] == "2020-05-01"
    assert default_negative["target_kw"][0] == -5
    assert default_negative["quality_flags"][0] == ""
    assert default_negative["feature_quality_flags"][0] == ""
    assert "target_kw_raw" not in default_series.columns
    assert "sdwpf_has_negative_patv" not in default_series.columns
    assert report["quality_profile"] == "default"
    assert report["sdwpf_unknown_count"] == 2
    assert report["sdwpf_abnormal_count"] == 2
    assert report["sdwpf_masked_count"] == 4
    assert report["sdwpf_flag_counts"]["sdwpf_unknown_patv_wspd"] == 1
    assert report["sdwpf_flag_counts"]["sdwpf_unknown_pitch"] == 1
    assert report["sdwpf_flag_counts"]["sdwpf_abnormal_ndir"] == 1
    assert report["sdwpf_flag_counts"]["sdwpf_abnormal_wdir"] == 1
    assert "sdwpf_patv_negative" not in report["sdwpf_flag_counts"]
    assert "sdwpf_patv_zeroed" not in report["sdwpf_flag_counts"]
    assert not report["long_gaps"]

    task_a = TaskSpec(history_duration="30m", forecast_duration="20m", task_id="short_a", granularity="turbine")
    task_b = TaskSpec(history_duration="40m", forecast_duration="20m", task_id="short_b", granularity="turbine")
    builder.build_task_cache(task_a)
    builder.build_task_cache(task_b)

    after_mtime = gold_path.stat().st_mtime_ns
    window_index = builder.load_window_index(task_a)
    task_report = json.loads(
        (builder.cache_paths.tasks_dir / "default" / "turbine" / "short_a" / "task_report.json").read_text()
    )
    anchor_window = window_index.filter(
        (pl.col("turbine_id") == "1")
        & (pl.col("input_end_ts").dt.strftime("%Y-%m-%d %H:%M:%S") == "2020-05-01 00:20:00")
    )

    assert before_mtime == after_mtime
    assert "input_masked_steps" in window_index.columns
    assert "output_masked_steps" in window_index.columns
    assert "input_unknown_steps" in window_index.columns
    assert "output_abnormal_steps" in window_index.columns
    assert anchor_window["input_masked_steps"][0] == 1
    assert anchor_window["output_masked_steps"][0] == 2
    assert anchor_window["input_unknown_steps"][0] == 1
    assert anchor_window["output_unknown_steps"][0] == 1
    assert "masked_input" in anchor_window["quality_flags"][0]
    assert "masked_output" in anchor_window["quality_flags"][0]
    assert task_report["quality_profile"] == "default"
    assert task_report["masked_input_windows"] > 0
    assert task_report["masked_output_windows"] > 0
    assert task_report["fully_complete_and_unmasked_output_windows"] > 0
    assert (builder.cache_paths.tasks_dir / "default" / "turbine" / "short_a" / "window_index.parquet").exists()
    assert (builder.cache_paths.tasks_dir / "default" / "turbine" / "short_b" / "window_index.parquet").exists()

    for legacy_profile in ("official_v1", "raw_v1", "official_v1_zero_negative_patv"):
        with pytest.raises(ValueError):
            builder.build_gold_base(quality_profile=legacy_profile)
        with pytest.raises(ValueError):
            builder.build_task_cache(task_a, quality_profile=legacy_profile)


def test_end_to_end_penmanshiel_pipeline(tmp_path) -> None:
    spec = build_greenbyte_fixture(tmp_path / "raw" / "penmanshiel", "Penmanshiel", "Penmanshiel 11")
    builder = GreenbyteDatasetBuilder(spec=spec, cache_root=tmp_path / "cache")

    builder.build_gold_base()
    series = builder.load_series()

    _assert_unique_series_keys(series)
    assert series.filter(pl.col("turbine_id") == "Penmanshiel 11").height > 0
    assert "farm_turbines_observed" in series.columns

from __future__ import annotations

import json

import polars as pl

from wind_datasets.datasets.greenbyte import GreenbyteDatasetBuilder
from wind_datasets.datasets.hill_of_towie import HillOfTowieDatasetBuilder
from wind_datasets.datasets.sdwpf_full import SDWPFFullDatasetBuilder
from wind_datasets.models import TaskSpec

from .helpers import build_greenbyte_fixture, build_hill_fixture, build_sdwpf_fixture


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
    assert "long_gaps" in report

    task = TaskSpec(history_duration="30m", forecast_duration="30m", task_id="short_task")
    builder.build_task_cache(task)
    window_index = builder.load_window_index(task)
    assert window_index.height > 0
    assert "quality_flags" in window_index.columns


def test_end_to_end_hill_pipeline(tmp_path) -> None:
    spec = build_hill_fixture(tmp_path / "raw" / "hill")
    builder = HillOfTowieDatasetBuilder(spec=spec, cache_root=tmp_path / "cache")

    builder.build_gold_base()
    series = builder.load_series()
    turbine_static = builder.load_turbine_static()
    report = builder.profile_dataset()
    duplicate_audit = pl.read_parquet(builder.cache_paths.hill_duplicate_audit_path)

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
    assert duplicate_audit.filter(pl.col("is_conflicting")).height == 1
    assert duplicate_audit.filter(pl.col("is_conflicting"))["conflicting_columns"][0] == "wtc_PrWindSp_mean"
    t01_0020 = series.filter(
        (pl.col("turbine_id") == "T01")
        & (pl.col("timestamp").dt.strftime("%Y-%m-%d %H:%M:%S") == "2024-01-01 00:20:00")
    )
    t01_0010 = series.filter(
        (pl.col("turbine_id") == "T01")
        & (pl.col("timestamp").dt.strftime("%Y-%m-%d %H:%M:%S") == "2024-01-01 00:10:00")
    )
    t02_0010 = series.filter(
        (pl.col("turbine_id") == "T02")
        & (pl.col("timestamp").dt.strftime("%Y-%m-%d %H:%M:%S") == "2024-01-01 00:10:00")
    )
    assert t01_0020["farm_grid__activepower"][0] == 2020.0
    assert t01_0020["farm_grid_sci__activepowermean"][0] == 2020.0
    assert t01_0020["shutdown_duration_s"][0] == 600.0
    assert t01_0020["alarm_any_active"][0] is True
    assert t01_0020["alarm_code__42"][0] is True
    assert t01_0010["aeroup_post_install"][0] is True
    assert t02_0010["aeroup_in_install_window"][0] is True
    assert "duplicate_conflict_resolved" in series.filter(
        (pl.col("turbine_id") == "T01")
        & (pl.col("timestamp").dt.strftime("%Y-%m-%d %H:%M:%S") == "2024-01-01 01:00:00")
    )["quality_flags"][0]
    assert series.filter(
        (pl.col("turbine_id") == "T02")
        & (pl.col("timestamp").dt.strftime("%Y-%m-%d %H:%M:%S") == "2024-01-01 00:20:00")
    )["target_kw"][0] is None
    assert report["target_missing_count"] >= 1
    assert report["duplicate_key_audit_count"] == 3
    assert report["duplicate_conflict_key_count"] == 1


def test_end_to_end_sdwpf_pipeline_and_task_switch_only_updates_task_cache(tmp_path) -> None:
    spec = build_sdwpf_fixture(tmp_path / "raw" / "sdwpf")
    builder = SDWPFFullDatasetBuilder(spec=spec, cache_root=tmp_path / "cache")

    gold_path = builder.build_gold_base()
    assert gold_path.parent.name == "official_v1"
    before_mtime = gold_path.stat().st_mtime_ns

    default_series = builder.load_series()
    raw_series = builder.load_series("raw_v1")
    zeroed_series = builder.load_series("official_v1_zero_negative_patv")
    turbine_static = builder.load_turbine_static()
    report = builder.profile_dataset()

    default_negative = default_series.filter(
        (pl.col("turbine_id") == "1")
        & (pl.col("timestamp").dt.strftime("%Y-%m-%d %H:%M:%S") == "2024-01-01 00:10:00")
    )
    raw_negative = raw_series.filter(
        (pl.col("turbine_id") == "1")
        & (pl.col("timestamp").dt.strftime("%Y-%m-%d %H:%M:%S") == "2024-01-01 00:10:00")
    )
    zeroed_negative = zeroed_series.filter(
        (pl.col("turbine_id") == "1")
        & (pl.col("timestamp").dt.strftime("%Y-%m-%d %H:%M:%S") == "2024-01-01 00:10:00")
    )

    assert before_mtime == gold_path.stat().st_mtime_ns
    _assert_unique_series_keys(default_series)
    assert turbine_static.height == len(spec.turbine_ids)
    assert default_negative["target_kw_raw"][0] == -5
    assert default_negative["target_kw"][0] == -5
    assert default_negative["sdwpf_has_negative_patv"][0] is True
    assert default_negative["quality_flags"][0] == "sdwpf_patv_negative"
    assert raw_negative["target_kw_raw"][0] == -5
    assert raw_negative["target_kw"][0] == -5
    assert raw_negative["sdwpf_has_negative_patv"][0] is False
    assert raw_negative["quality_flags"][0] == ""
    assert zeroed_negative["target_kw_raw"][0] == -5
    assert zeroed_negative["target_kw"][0] == 0
    assert zeroed_negative["quality_flags"][0] == "sdwpf_patv_negative|sdwpf_patv_zeroed"

    assert report["quality_profile"] == "official_v1"
    assert report["sdwpf_negative_patv_count"] == 1
    assert report["sdwpf_unknown_count"] == 2
    assert report["sdwpf_abnormal_count"] == 2
    assert report["sdwpf_masked_count"] == 4
    assert report["sdwpf_flag_counts"]["sdwpf_unknown_patv_wspd"] == 1
    assert report["sdwpf_flag_counts"]["sdwpf_unknown_pitch"] == 1
    assert report["sdwpf_flag_counts"]["sdwpf_abnormal_ndir"] == 1
    assert report["sdwpf_flag_counts"]["sdwpf_abnormal_wdir"] == 1
    assert report["long_gaps"]

    task_a = TaskSpec(history_duration="30m", forecast_duration="20m", task_id="short_a")
    task_b = TaskSpec(history_duration="40m", forecast_duration="20m", task_id="short_b")
    builder.build_task_cache(task_a)
    builder.build_task_cache(task_b)
    builder.build_task_cache(task_a, quality_profile="raw_v1")

    after_mtime = gold_path.stat().st_mtime_ns
    window_index = builder.load_window_index(task_a)
    task_report = json.loads(
        (builder.cache_paths.tasks_dir / "official_v1" / "short_a" / "task_report.json").read_text()
    )
    anchor_window = window_index.filter(
        (pl.col("turbine_id") == "1")
        & (pl.col("input_end_ts").dt.strftime("%Y-%m-%d %H:%M:%S") == "2024-01-01 00:20:00")
    )

    assert before_mtime == after_mtime
    assert "input_masked_steps" in window_index.columns
    assert "output_masked_steps" in window_index.columns
    assert "input_unknown_steps" in window_index.columns
    assert "output_abnormal_steps" in window_index.columns
    assert anchor_window["input_masked_steps"][0] == 1
    assert anchor_window["output_masked_steps"][0] == 1
    assert anchor_window["input_unknown_steps"][0] == 1
    assert anchor_window["output_unknown_steps"][0] == 1
    assert "masked_input" in anchor_window["quality_flags"][0]
    assert "masked_output" in anchor_window["quality_flags"][0]
    assert task_report["quality_profile"] == "official_v1"
    assert task_report["masked_input_windows"] > 0
    assert task_report["masked_output_windows"] > 0
    assert task_report["fully_complete_and_unmasked_output_windows"] > 0
    assert (builder.cache_paths.tasks_dir / "official_v1" / "short_a" / "window_index.parquet").exists()
    assert (builder.cache_paths.tasks_dir / "official_v1" / "short_b" / "window_index.parquet").exists()
    assert (builder.cache_paths.tasks_dir / "raw_v1" / "short_a" / "window_index.parquet").exists()


def test_end_to_end_penmanshiel_pipeline(tmp_path) -> None:
    spec = build_greenbyte_fixture(tmp_path / "raw" / "penmanshiel", "Penmanshiel", "Penmanshiel 11")
    builder = GreenbyteDatasetBuilder(spec=spec, cache_root=tmp_path / "cache")

    builder.build_gold_base()
    series = builder.load_series()

    _assert_unique_series_keys(series)
    assert series.filter(pl.col("turbine_id") == "Penmanshiel 11").height > 0

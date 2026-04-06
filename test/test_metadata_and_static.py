from __future__ import annotations

from textwrap import dedent

import polars as pl
import pytest

from wind_datasets import api as api_module
from wind_datasets.api import (
    build_task_cache,
    build_silver,
    load_event_features,
    load_interventions,
    load_shared_timeseries,
    load_task_turbine_static,
    load_turbine_static,
)
from wind_datasets.datasets.common import TURBINE_STATIC_SCHEMA
from wind_datasets.datasets.greenbyte import GreenbyteDatasetBuilder
from wind_datasets.datasets.sdwpf_full import SDWPFFullDatasetBuilder
from wind_datasets.manifest import build_manifest as build_manifest_for_spec
from wind_datasets.models import DatasetSpec, OfficialRelease, TaskSpec
from wind_datasets.registry import get_dataset_spec
from wind_datasets.utils import read_json

from .helpers import build_greenbyte_fixture, build_hill_fixture, build_sdwpf_fixture


def test_registry_hill_time_metadata_matches_documented_utc_interval_end() -> None:
    spec = get_dataset_spec("hill_of_towie")
    assert spec.timezone_policy == "utc_documented"
    assert spec.timestamp_convention == "source_utc_naive_interval_end"


@pytest.mark.parametrize(
    ("dataset_id", "spec_factory"),
    [
        ("kelmarsh", lambda root: build_greenbyte_fixture(root, "Kelmarsh", "Kelmarsh 1")),
        ("penmanshiel", lambda root: build_greenbyte_fixture(root, "Penmanshiel", "Penmanshiel 11")),
        ("hill_of_towie", build_hill_fixture),
        ("sdwpf_full", build_sdwpf_fixture),
    ],
)
def test_api_build_silver_then_load_turbine_static(tmp_path, monkeypatch, dataset_id, spec_factory) -> None:
    spec = spec_factory(tmp_path / "raw" / dataset_id)

    def _fake_get_dataset_spec(requested_dataset_id: str):
        assert requested_dataset_id == dataset_id
        return spec

    monkeypatch.setattr(api_module, "get_dataset_spec", _fake_get_dataset_spec)

    build_silver(dataset_id, cache_root=tmp_path / "cache")
    turbine_static = load_turbine_static(dataset_id, cache_root=tmp_path / "cache")

    assert turbine_static.columns == list(TURBINE_STATIC_SCHEMA)
    assert turbine_static.height == len(spec.turbine_ids)


def test_greenbyte_manifest_release_check_distinguishes_latest_and_compatible(tmp_path) -> None:
    latest_spec = build_greenbyte_fixture(tmp_path / "raw" / "kel_latest", "Kelmarsh", "Kelmarsh 1", file_end="2025-01-01")
    latest_manifest = read_json(build_manifest_for_spec(latest_spec, tmp_path / "cache_latest"))
    assert latest_manifest["source_release_check"]["status"] == "match_expected"
    assert latest_manifest["source_release_check"]["detected_release_id"] == "extended_2025"

    old_spec = build_greenbyte_fixture(tmp_path / "raw" / "kel_old", "Kelmarsh", "Kelmarsh 1", file_end="2023-01-01")
    old_manifest = read_json(build_manifest_for_spec(old_spec, tmp_path / "cache_old"))
    assert old_manifest["source_release_check"]["status"] == "compatible_other_release"
    assert old_manifest["source_release_check"]["detected_release_id"] == "legacy_2022"


def test_manifest_reports_layout_problem_for_zip_only_sources(tmp_path) -> None:
    root = tmp_path / "zip_only"
    root.mkdir(parents=True)
    (root / "Kelmarsh wind farm data.zip").write_bytes(b"placeholder")
    spec = DatasetSpec(
        dataset_id="kelmarsh_zip_only",
        source_root=root,
        resolution_minutes=10,
        turbine_ids=("Kelmarsh 1",),
        target_column="Power (kW)",
        target_unit="kW",
        timezone_policy="utc_documented",
        timestamp_convention="source_utc_naive",
        default_feature_groups=("continuous_main",),
        handler="greenbyte",
        official_name="Kelmarsh wind farm data",
        official_releases=(
            OfficialRelease(
                release_id="extended_2025",
                source_url="https://zenodo.org/records/16807551",
                published_date="2025-08-12",
                coverage_start="2016-01-03",
                coverage_end="2024-12-31",
            ),
        ),
        default_expected_release_id="extended_2025",
        requires_pre_extracted_sources=True,
        official_assets=("turbine_static", "turbine_scada", "status_events"),
        default_ingested_assets=("turbine_static", "turbine_scada", "status_events"),
    )

    manifest = read_json(build_manifest_for_spec(spec, tmp_path / "cache"))

    assert manifest["source_release_check"]["status"] == "layout_problem"
    assert manifest["source_layout"]["archive_files"] == ["Kelmarsh wind farm data.zip"]
    assert manifest["warnings"]


def test_auxiliary_loaders_return_standardized_sidecars(tmp_path, monkeypatch) -> None:
    greenbyte_spec = build_greenbyte_fixture(tmp_path / "raw" / "kelmarsh", "Kelmarsh", "Kelmarsh 1")
    hill_spec = build_hill_fixture(tmp_path / "raw" / "hill")

    spec_map = {
        "kelmarsh": greenbyte_spec,
        "hill_of_towie": hill_spec,
    }

    def _fake_get_dataset_spec(requested_dataset_id: str):
        return spec_map[requested_dataset_id]

    monkeypatch.setattr(api_module, "get_dataset_spec", _fake_get_dataset_spec)

    build_silver("kelmarsh", cache_root=tmp_path / "cache")
    build_silver("hill_of_towie", cache_root=tmp_path / "cache")

    farm_pmu = load_shared_timeseries("kelmarsh", "farm_pmu", cache_root=tmp_path / "cache")
    turbine_status = load_event_features("kelmarsh", "turbine_status", cache_root=tmp_path / "cache")
    hill_grid = load_shared_timeseries("hill_of_towie", "farm_grid", cache_root=tmp_path / "cache")
    hill_alarm = load_event_features("hill_of_towie", "alarmlog", cache_root=tmp_path / "cache")
    hill_aeroup = load_interventions("hill_of_towie", "aeroup", cache_root=tmp_path / "cache")

    assert "farm_pmu__gms_power_kw" in farm_pmu.columns
    assert "evt_status_code__710" in turbine_status.columns
    assert "farm_grid__activepower" in hill_grid.columns
    assert "alarm_code__42" in hill_alarm.columns
    assert hill_aeroup.columns == ["dataset", "turbine_id", "aeroup_start", "aeroup_end"]


def test_api_build_farm_task_cache_then_load_task_turbine_static(tmp_path, monkeypatch) -> None:
    hill_spec = build_hill_fixture(tmp_path / "raw" / "hill")

    def _fake_get_dataset_spec(requested_dataset_id: str):
        assert requested_dataset_id == "hill_of_towie"
        return hill_spec

    monkeypatch.setattr(api_module, "get_dataset_spec", _fake_get_dataset_spec)

    task = TaskSpec(
        history_duration="30m",
        forecast_duration="20m",
        task_id="farm_short",
        granularity="farm",
    )
    build_task_cache("hill_of_towie", task_spec=task, cache_root=tmp_path / "cache")
    turbine_static = load_task_turbine_static("hill_of_towie", task_spec=task, cache_root=tmp_path / "cache")

    assert turbine_static["turbine_id"].to_list() == ["T01", "T02"]
    assert turbine_static["turbine_index"].to_list() == [0, 1]


def test_manifest_audits_sdwpf_time_semantics_and_reports_incompatible_minutes(tmp_path) -> None:
    spec = build_sdwpf_fixture(tmp_path / "raw" / "sdwpf_bad")
    path = spec.source_root / "sdwpf_2001_2112_full.parquet"
    frame = pl.read_parquet(path).with_row_index("row_nr")
    frame = frame.with_columns(
        pl.when(pl.col("row_nr") == 3)
        .then(pl.lit("2024-01-01 00:25:00"))
        .otherwise(pl.col("Tmstamp"))
        .alias("Tmstamp")
    ).drop("row_nr")
    frame.write_parquet(path)

    manifest = read_json(build_manifest_for_spec(spec, tmp_path / "cache"))

    assert manifest["time_semantics_check"]["status"] == "incompatible_with_documented_10min_grid"
    assert 25 in manifest["time_semantics_check"]["invalid_minutes"]
    assert manifest["warnings"]


def test_sdwpf_incompatible_time_semantics_block_gold_build(tmp_path) -> None:
    spec = build_sdwpf_fixture(tmp_path / "raw" / "sdwpf_blocked")
    path = spec.source_root / "sdwpf_2001_2112_full.parquet"
    frame = pl.read_parquet(path).with_row_index("row_nr")
    frame = frame.with_columns(
        pl.when(pl.col("row_nr") == 3)
        .then(pl.lit("2024-01-01 00:25:00"))
        .otherwise(pl.col("Tmstamp"))
        .alias("Tmstamp")
    ).drop("row_nr")
    frame.write_parquet(path)

    builder = SDWPFFullDatasetBuilder(spec=spec, cache_root=tmp_path / "cache")
    builder.build_silver()

    with pytest.raises(ValueError, match="Refusing to build sdwpf_full gold/task cache"):
        builder.build_gold_base()


def test_penmanshiel_quality_report_records_common_coverage_summary(tmp_path) -> None:
    base_spec = build_greenbyte_fixture(tmp_path / "raw" / "pen_cov", "Penmanshiel", "Penmanshiel 11")
    root = base_spec.source_root
    (root / "Penmanshiel_WT_static.csv").write_text(
        dedent(
            """
            Wind Farm,Title,Alternative Title,Identity,Manufacturer,Model,Rated power (kW),Hub Height (m),Rotor Diameter (m),Latitude,Longitude,Elevation (m),Country,Commercial Operations Date
            Penmanshiel,Penmanshiel 11,Penmanshiel 11,ID-11,Senvion,MM82,2050,78.5,92,52.4,-0.94,140,UK,2016-01-01
            Penmanshiel,Penmanshiel 12,Penmanshiel 12,ID-12,Senvion,MM82,2050,78.5,92,52.5,-0.95,142,UK,2016-01-01
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    scada_dir = root / "Penmanshiel_SCADA_2024_0001"
    (scada_dir / "Turbine_Data_Penmanshiel_12_2024-01-01_-_2025-01-01_0002.csv").write_text(
        dedent(
            """
            # This file was exported by Greenbyte.
            #
            # Turbine: Penmanshiel 12
            # Time zone: UTC
            # Date and time,Power (kW),Wind speed (m/s),Wind direction (°),Nacelle position (°),Generator RPM (RPM),Rotor speed (RPM),Ambient temperature (converter) (°C),Nacelle temperature (°C),Grid frequency (Hz),Blade angle (pitch position) A (°),Blade angle (pitch position) B (°),Blade angle (pitch position) C (°)
            2024-01-01 00:10:00,210,8.4,181,175,1510,12.1,5.1,10.2,50,1,1,1
            2024-01-01 00:20:00,220,8.8,182,176,1520,12.2,5.2,10.4,50,1,1,1
            2024-01-01 00:30:00,230,9.1,183,177,1530,12.3,5.3,10.6,50,1,1,1
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (scada_dir / "Status_Penmanshiel_12_2024-01-01_-_2025-01-01_0002.csv").write_text(
        dedent(
            """
            # Status export
            #
            # Turbine: Penmanshiel 12
            Timestamp start,Timestamp end,Duration,Status,Code,Message,Comment,Service contract category,IEC category,Global contract category,Custom contract category
            2024-01-01 00:10:00,2024-01-01 00:20:00,00:10:00,Informational,0,System OK,,System OK,Full Performance,,
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    spec = DatasetSpec(
        dataset_id=base_spec.dataset_id,
        source_root=base_spec.source_root,
        resolution_minutes=base_spec.resolution_minutes,
        turbine_ids=("Penmanshiel 11", "Penmanshiel 12"),
        target_column=base_spec.target_column,
        target_unit=base_spec.target_unit,
        timezone_policy=base_spec.timezone_policy,
        timestamp_convention=base_spec.timestamp_convention,
        default_feature_groups=base_spec.default_feature_groups,
        handler=base_spec.handler,
        official_name=base_spec.official_name,
        official_releases=base_spec.official_releases,
        default_expected_release_id=base_spec.default_expected_release_id,
        requires_pre_extracted_sources=base_spec.requires_pre_extracted_sources,
        official_assets=base_spec.official_assets,
        default_ingested_assets=base_spec.default_ingested_assets,
        default_excluded_assets=base_spec.default_excluded_assets,
    )

    builder = GreenbyteDatasetBuilder(spec=spec, cache_root=tmp_path / "cache")
    builder.build_gold_base()
    report = builder.profile_dataset()

    assert report["common_coverage_start"] == "2024-01-01T00:10:00"
    assert report["common_coverage_end"] == "2024-01-01T00:30:00"
    assert report["full_synchrony_ratio"] < 1.0
    per_turbine = {row["turbine_id"]: row for row in report["per_turbine_coverage_summary"]}
    assert per_turbine["Penmanshiel 12"]["observed_ratio"] < per_turbine["Penmanshiel 11"]["observed_ratio"]

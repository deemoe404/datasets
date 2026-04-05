from __future__ import annotations

import pytest

from wind_datasets import api as api_module
from wind_datasets.api import (
    build_silver,
    load_event_features,
    load_interventions,
    load_shared_timeseries,
    load_turbine_static,
)
from wind_datasets.datasets.common import TURBINE_STATIC_SCHEMA
from wind_datasets.manifest import build_manifest as build_manifest_for_spec
from wind_datasets.models import DatasetSpec, OfficialRelease
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

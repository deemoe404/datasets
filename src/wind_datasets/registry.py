from __future__ import annotations

from pathlib import Path

from .models import DatasetSpec, OfficialRelease

_SOURCE_ROOT = Path("/Users/sam/Developer/Datasets/Wind Power Forecasting")

_DATASET_SPECS: dict[str, DatasetSpec] = {
    "kelmarsh": DatasetSpec(
        dataset_id="kelmarsh",
        source_root=_SOURCE_ROOT / "Kelmarsh wind farm data",
        resolution_minutes=10,
        turbine_ids=tuple(f"Kelmarsh {idx}" for idx in range(1, 7)),
        target_column="Power (kW)",
        target_unit="kW",
        timezone_policy="utc_documented",
        timestamp_convention="source_utc_naive",
        default_feature_groups=("continuous_main", "farm_pmu", "farm_grid_meter", "status_events"),
        handler="greenbyte",
        official_name="Kelmarsh wind farm data",
        official_releases=(
            OfficialRelease(
                release_id="legacy_2022",
                source_url="https://zenodo.org/records/5841834",
                published_date="2022-02-01",
                coverage_start="2016-01-03",
                coverage_end="2021-06-30",
                notes="Original 2022 open release with SCADA/events to mid-2021.",
            ),
            OfficialRelease(
                release_id="extended_2025",
                source_url="https://zenodo.org/records/16807551",
                published_date="2025-08-12",
                coverage_start="2016-01-03",
                coverage_end="2024-12-31",
                notes="Expanded release with SCADA/events through end-2024.",
            ),
        ),
        default_expected_release_id="extended_2025",
        requires_pre_extracted_sources=True,
        official_assets=(
            "kmz_layout",
            "turbine_static",
            "turbine_scada",
            "status_events",
            "signal_mapping",
            "pmu_meter",
            "grid_meter",
        ),
        default_ingested_assets=(
            "turbine_static",
            "turbine_scada",
            "status_events",
            "signal_mapping",
            "pmu_meter",
            "grid_meter",
        ),
        default_excluded_assets=("kmz_layout",),
    ),
    "penmanshiel": DatasetSpec(
        dataset_id="penmanshiel",
        source_root=_SOURCE_ROOT / "Penmanshiel wind farm data",
        resolution_minutes=10,
        turbine_ids=tuple(f"Penmanshiel {idx:02d}" for idx in [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        target_column="Power (kW)",
        target_unit="kW",
        timezone_policy="utc_documented",
        timestamp_convention="source_utc_naive",
        default_feature_groups=("continuous_main", "farm_pmu", "farm_grid_meter", "status_events"),
        handler="greenbyte",
        official_name="Penmanshiel wind farm data",
        official_releases=(
            OfficialRelease(
                release_id="legacy_2022",
                source_url="https://zenodo.org/records/5946808",
                published_date="2022-02-07",
                coverage_start="2016-06-02",
                coverage_end="2021-06-30",
                notes="Original 2022 open release with SCADA/events to mid-2021 and no WT03.",
            ),
            OfficialRelease(
                release_id="extended_2025",
                source_url="https://zenodo.org/records/16807304",
                published_date="2025-08-13",
                coverage_start="2016-06-02",
                coverage_end="2024-12-31",
                notes="Expanded release with SCADA/events through end-2024 and no WT03.",
            ),
        ),
        default_expected_release_id="extended_2025",
        requires_pre_extracted_sources=True,
        official_assets=(
            "kmz_layout",
            "turbine_static",
            "turbine_scada",
            "status_events",
            "signal_mapping",
            "pmu_meter",
            "grid_meter",
        ),
        default_ingested_assets=(
            "turbine_static",
            "turbine_scada",
            "status_events",
            "signal_mapping",
            "pmu_meter",
            "grid_meter",
        ),
        default_excluded_assets=("kmz_layout",),
    ),
    "hill_of_towie": DatasetSpec(
        dataset_id="hill_of_towie",
        source_root=_SOURCE_ROOT / "Hill of Towie",
        resolution_minutes=10,
        turbine_ids=tuple(f"T{idx:02d}" for idx in range(1, 22)),
        target_column="wtc_ActPower_mean",
        target_unit="kW",
        timezone_policy="utc_documented",
        timestamp_convention="source_utc_naive_interval_end",
        default_feature_groups=(
            "tblSCTurbine",
            "tblSCTurGrid",
            "tblSCTurFlag",
            "tblGrid",
            "tblGridScientific",
            "tblSCTurCount",
            "tblSCTurDigiIn",
            "tblSCTurDigiOut",
            "tblSCTurIntern",
            "tblSCTurPress",
            "tblSCTurTemp",
            "tblAlarmLog",
            "shutdown_duration",
            "aeroup_timeline",
        ),
        handler="hill_of_towie",
        official_name="Hill of Towie wind farm open dataset",
        official_releases=(
            OfficialRelease(
                release_id="v1_2025",
                source_url="https://zenodo.org/records/14870023",
                published_date="2025-03-28",
                coverage_start="2016-01-01",
                coverage_end="2024-08-31",
                notes="UTC timestamps. 10-minute timestamps denote interval end.",
            ),
        ),
        default_expected_release_id="v1_2025",
        requires_pre_extracted_sources=True,
        official_assets=(
            "turbine_metadata",
            "tblSCTurbine",
            "tblSCTurGrid",
            "tblSCTurFlag",
            "tblAlarmLog",
            "tblDailySummary",
            "tblGrid",
            "tblGridScientific",
            "tblSCTurCount",
            "tblSCTurDigiIn",
            "tblSCTurDigiOut",
            "tblSCTurIntern",
            "tblSCTurPress",
            "tblSCTurTemp",
            "aeroup_timeline",
            "shutdown_duration",
        ),
        default_ingested_assets=(
            "turbine_metadata",
            "tblSCTurbine",
            "tblSCTurGrid",
            "tblSCTurFlag",
            "tblAlarmLog",
            "tblGrid",
            "tblGridScientific",
            "tblSCTurCount",
            "tblSCTurDigiIn",
            "tblSCTurDigiOut",
            "tblSCTurIntern",
            "tblSCTurPress",
            "tblSCTurTemp",
            "aeroup_timeline",
            "shutdown_duration",
        ),
        default_excluded_assets=("zip_archives", "tblDailySummary"),
    ),
    "sdwpf_kddcup": DatasetSpec(
        dataset_id="sdwpf_kddcup",
        source_root=_SOURCE_ROOT / "SDWPF_dataset" / "sdwpf_kddcup",
        resolution_minutes=10,
        turbine_ids=tuple(str(idx) for idx in range(1, 135)),
        target_column="Patv",
        target_unit="kW",
        timezone_policy="unknown_unverified",
        timestamp_convention="derived_day_clock_naive",
        default_feature_groups=("main",),
        handler="sdwpf_kddcup",
        default_quality_profile="default",
        official_name="sdwpf_kddcup",
        official_releases=(
            OfficialRelease(
                release_id="figshare_v2_2024",
                source_url="https://figshare.com/articles/dataset/SDWPF_dataset/24798654",
                published_date="2024-04-30",
                coverage_start="2020-05-01",
                coverage_end="2020-12-31",
                notes=(
                    "Public release of the original Baidu KDD Cup 2022 package. "
                    "The source CSV exposes Day + Tmstamp only; this repository derives calendar "
                    "timestamps by anchoring Day 1 to 2020-05-01 based on public reproduction references."
                ),
            ),
        ),
        default_expected_release_id="figshare_v2_2024",
        requires_pre_extracted_sources=False,
        official_assets=("main_csv", "turbine_location", "final_phase_test"),
        default_ingested_assets=("main_csv", "turbine_location"),
        default_excluded_assets=("final_phase_test",),
    ),
}


def get_dataset_spec(dataset_id: str) -> DatasetSpec:
    try:
        return _DATASET_SPECS[dataset_id]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset_id {dataset_id!r}.") from exc


def list_dataset_specs() -> tuple[DatasetSpec, ...]:
    return tuple(_DATASET_SPECS.values())

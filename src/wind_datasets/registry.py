from __future__ import annotations

from pathlib import Path

from .models import DatasetSpec

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
        default_feature_groups=("continuous_main",),
        handler="greenbyte",
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
        default_feature_groups=("continuous_main",),
        handler="greenbyte",
    ),
    "hill_of_towie": DatasetSpec(
        dataset_id="hill_of_towie",
        source_root=_SOURCE_ROOT / "Hill of Towie",
        resolution_minutes=10,
        turbine_ids=tuple(f"T{idx:02d}" for idx in range(1, 22)),
        target_column="wtc_ActPower_mean",
        target_unit="kW",
        timezone_policy="unknown_unverified",
        timestamp_convention="source_local_or_naive",
        default_feature_groups=("tblSCTurbine", "tblSCTurGrid", "tblSCTurFlag"),
        handler="hill_of_towie",
    ),
    "sdwpf_full": DatasetSpec(
        dataset_id="sdwpf_full",
        source_root=_SOURCE_ROOT / "SDWPF_dataset" / "sdwpf_full",
        resolution_minutes=10,
        turbine_ids=tuple(str(idx) for idx in range(1, 135)),
        target_column="Patv",
        target_unit="kW",
        timezone_policy="utc_plus_8_documented",
        timestamp_convention="source_local_naive_utc_plus_8",
        default_feature_groups=("main",),
        handler="sdwpf_full",
        default_quality_profile="official_v1",
    ),
}


def get_dataset_spec(dataset_id: str) -> DatasetSpec:
    try:
        return _DATASET_SPECS[dataset_id]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset_id {dataset_id!r}.") from exc


def list_dataset_specs() -> tuple[DatasetSpec, ...]:
    return tuple(_DATASET_SPECS.values())

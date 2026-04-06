from __future__ import annotations

from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
import re

import polars as pl

from .models import DatasetSpec
from .paths import dataset_cache_paths
from .utils import ensure_directory, sha256_file, write_json

_IGNORED_FILE_NAMES = {".DS_Store"}
_IGNORED_SUFFIXES = {".swp", ".webloc"}
_BUILD_VERSION = "v2"
_GREENBYTE_RANGE_PATTERN = re.compile(
    r"Turbine_Data_.*?_(?P<start>\d{4}-\d{2}-\d{2})_-_(?P<end>\d{4}-\d{2}-\d{2})(?:_|\.csv)"
)
_TEN_MINUTE_MINUTES = {0, 10, 20, 30, 40, 50}


def _iter_source_files(root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name in _IGNORED_FILE_NAMES:
            continue
        if path.suffix.lower() in _IGNORED_SUFFIXES:
            continue
        paths.append(path)
    return paths


def _matches_any(relative_paths: list[str], pattern: str) -> bool:
    return any(Path(path).match(pattern) for path in relative_paths)


def _build_source_layout(spec: DatasetSpec, relative_paths: list[str]) -> tuple[dict[str, object], list[str]]:
    archive_files = [
        path for path in relative_paths if Path(path).suffix.lower() in {".zip", ".7z", ".tar", ".gz"}
    ]
    missing: list[str] = []
    if spec.handler == "greenbyte":
        required = {
            "turbine_scada_csv": "**/Turbine_Data_*.csv",
            "status_csv": "**/Status_*.csv",
            "turbine_static_csv": "*_WT_static.csv",
        }
    elif spec.handler == "hill_of_towie":
        required = {
            "turbine_metadata_csv": "Hill_of_Towie_turbine_metadata.csv",
            "tblSCTurbine_csv": "**/tblSCTurbine_*.csv",
            "tblSCTurGrid_csv": "**/tblSCTurGrid_*.csv",
            "tblSCTurFlag_csv": "**/tblSCTurFlag_*.csv",
        }
    elif spec.handler == "sdwpf_full":
        required = {
            "main_parquet": "sdwpf_2001_2112_full.parquet",
            "location_csv": "sdwpf_turb_location_elevation.csv",
        }
    else:
        required = {}

    for label, pattern in required.items():
        if not _matches_any(relative_paths, pattern):
            missing.append(label)

    core_files_present = not missing
    layout = {
        "requires_pre_extracted_sources": spec.requires_pre_extracted_sources,
        "archive_files": archive_files,
        "missing_required_patterns": missing,
        "core_files_present": core_files_present,
    }
    warnings: list[str] = []
    if missing:
        warnings.append(
            "Source layout is missing required extracted files: " + ", ".join(missing) + "."
        )
    elif spec.requires_pre_extracted_sources and archive_files:
        warnings.append(
            "Source directory contains archive files; current builders expect extracted CSV/parquet files."
        )
    return layout, warnings


def _detect_greenbyte_release(spec: DatasetSpec, relative_paths: list[str]) -> dict[str, object]:
    end_dates: list[str] = []
    for relative_path in relative_paths:
        match = _GREENBYTE_RANGE_PATTERN.search(relative_path)
        if match:
            end_dates.append(match.group("end"))
    if not end_dates:
        return {
            "status": "undetermined",
            "expected_release_id": spec.default_expected_release_id,
            "detected_release_id": None,
            "details": "No dated Turbine_Data filenames were found.",
        }

    max_end = max(end_dates)
    if max_end >= "2025-01-01":
        detected = "extended_2025"
    elif max_end <= "2023-01-01":
        detected = "legacy_2022"
    else:
        detected = None

    if detected is None:
        status = "undetermined"
        details = f"Observed maximum Turbine_Data filename end date {max_end}."
    elif detected == spec.default_expected_release_id:
        status = "match_expected"
        details = f"Observed maximum Turbine_Data filename end date {max_end}."
    else:
        status = "compatible_other_release"
        details = f"Observed maximum Turbine_Data filename end date {max_end}."
    return {
        "status": status,
        "expected_release_id": spec.default_expected_release_id,
        "detected_release_id": detected,
        "details": details,
    }


def _build_source_release_check(
    spec: DatasetSpec,
    relative_paths: list[str],
    source_layout: dict[str, object],
) -> tuple[dict[str, object], list[str]]:
    if not source_layout["core_files_present"]:
        check = {
            "status": "layout_problem",
            "expected_release_id": spec.default_expected_release_id,
            "detected_release_id": None,
            "details": "Required extracted source files are missing.",
        }
    elif spec.handler == "greenbyte":
        check = _detect_greenbyte_release(spec, relative_paths)
    else:
        check = {
            "status": "match_expected",
            "expected_release_id": spec.default_expected_release_id,
            "detected_release_id": spec.default_expected_release_id,
            "details": "Required extracted source files are present.",
        }

    warnings: list[str] = []
    if check["status"] == "compatible_other_release":
        warnings.append(
            "Source files look compatible with a different official release than the default expected release."
        )
    elif check["status"] == "undetermined":
        warnings.append("Unable to determine which official release the current source layout corresponds to.")
    elif check["status"] == "layout_problem":
        warnings.append("Source layout is incomplete for the current builder expectations.")
    return check, warnings


def _build_sdwpf_time_semantics_check(spec: DatasetSpec) -> tuple[dict[str, object], list[str]]:
    path = spec.source_root / "sdwpf_2001_2112_full.parquet"
    if not path.exists():
        return (
            {
                "status": "missing_source",
                "details": "sdwpf_2001_2112_full.parquet is missing.",
            },
            ["Unable to audit SDWPF time semantics because the parquet source file is missing."],
        )

    scan = pl.scan_parquet(path)
    timestamp_dtype = scan.collect_schema()["Tmstamp"]
    if timestamp_dtype == pl.Utf8:
        timestamp_expr = (
            pl.col("Tmstamp")
            .cast(pl.Utf8)
            .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
        )
    else:
        timestamp_expr = pl.col("Tmstamp").cast(pl.Datetime, strict=False)
    minutes = (
        scan
        .select(timestamp_expr.dt.minute().alias("minute"))
        .collect()["minute"]
        .drop_nulls()
        .unique()
        .sort()
        .to_list()
    )
    turbine_ids = (
        scan
        .select(pl.col("TurbID").cast(pl.String).alias("turbine_id"))
        .collect()["turbine_id"]
        .drop_nulls()
        .unique()
        .sort()
        .to_list()
    )
    sample_turbine_id = turbine_ids[0] if turbine_ids else None
    invalid_minutes = [int(value) for value in minutes if int(value) not in _TEN_MINUTE_MINUTES]

    sample_interval_distribution: dict[str, int] = {}
    invalid_intervals: list[int] = []
    if sample_turbine_id is not None:
        interval_frame = (
            scan
            .filter(pl.col("TurbID").cast(pl.String) == sample_turbine_id)
            .select(timestamp_expr.alias("timestamp"))
            .sort("timestamp")
            .collect()
            .with_columns(pl.col("timestamp").diff().dt.total_minutes().alias("delta_minutes"))
            .drop_nulls()
            .group_by("delta_minutes")
            .len()
            .sort("delta_minutes")
        )
        sample_interval_distribution = {
            str(int(row["delta_minutes"])): int(row["len"])
            for row in interval_frame.to_dicts()
        }
        invalid_intervals = [
            int(row["delta_minutes"])
            for row in interval_frame.to_dicts()
            if int(row["delta_minutes"]) % 10 != 0
        ]

    is_valid = not invalid_minutes and not invalid_intervals
    status = "match_documented_10min_grid" if is_valid else "incompatible_with_documented_10min_grid"
    details = (
        "Observed timestamps are compatible with a 10-minute grid."
        if is_valid
        else "Observed timestamps include minute offsets or single-turbine deltas that are incompatible with a 10-minute grid."
    )
    warnings: list[str] = []
    if not is_valid:
        warnings.append(
            "SDWPF source timestamps are incompatible with the documented 10-minute grid; gold/task builds are blocked."
        )
    return (
        {
            "status": status,
            "details": details,
            "distinct_minutes": [int(value) for value in minutes],
            "sample_turbine_id": sample_turbine_id,
            "sample_interval_distribution_minutes": sample_interval_distribution,
            "invalid_minutes": invalid_minutes,
            "invalid_intervals": invalid_intervals,
        },
        warnings,
    )


def build_manifest(spec: DatasetSpec, cache_root: Path) -> Path:
    cache_paths = dataset_cache_paths(cache_root, spec.dataset_id)
    ensure_directory(cache_paths.manifest_dir)
    source_files = _iter_source_files(spec.source_root)
    relative_paths = [str(path.relative_to(spec.source_root)) for path in source_files]
    source_layout, layout_warnings = _build_source_layout(spec, relative_paths)
    release_check, release_warnings = _build_source_release_check(spec, relative_paths, source_layout)
    time_semantics_check: dict[str, object] | None = None
    time_semantics_warnings: list[str] = []
    if spec.handler == "sdwpf_full":
        time_semantics_check, time_semantics_warnings = _build_sdwpf_time_semantics_check(spec)
    payload = {
        "dataset_id": spec.dataset_id,
        "build_version": _BUILD_VERSION,
        "source_root": str(spec.source_root),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "official_name": spec.official_name,
        "official_releases": [release.to_dict() for release in spec.official_releases],
        "default_expected_release_id": spec.default_expected_release_id,
        "requires_pre_extracted_sources": spec.requires_pre_extracted_sources,
        "official_assets": list(spec.official_assets),
        "default_ingested_assets": list(spec.default_ingested_assets),
        "default_excluded_assets": list(spec.default_excluded_assets),
        "source_layout": source_layout,
        "source_release_check": release_check,
        "warnings": [*layout_warnings, *release_warnings, *time_semantics_warnings],
        "dependencies": {
            name: metadata.version(name)
            for name in ("polars", "pyarrow", "duckdb", "pytest")
        },
        "files": [
            {
                "relative_path": relative_path,
                "size_bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
                "sha256": sha256_file(path),
            }
            for path, relative_path in zip(source_files, relative_paths, strict=True)
        ],
    }
    if time_semantics_check is not None:
        payload["time_semantics_check"] = time_semantics_check
    return write_json(cache_paths.manifest_path, payload)

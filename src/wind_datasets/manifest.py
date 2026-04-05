from __future__ import annotations

from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
import re

from .models import DatasetSpec
from .paths import dataset_cache_paths
from .utils import ensure_directory, sha256_file, write_json

_IGNORED_FILE_NAMES = {".DS_Store"}
_IGNORED_SUFFIXES = {".swp", ".webloc"}
_BUILD_VERSION = "v1"
_GREENBYTE_RANGE_PATTERN = re.compile(
    r"Turbine_Data_.*?_(?P<start>\d{4}-\d{2}-\d{2})_-_(?P<end>\d{4}-\d{2}-\d{2})(?:_|\.csv)"
)


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


def build_manifest(spec: DatasetSpec, cache_root: Path) -> Path:
    cache_paths = dataset_cache_paths(cache_root, spec.dataset_id)
    ensure_directory(cache_paths.manifest_dir)
    source_files = _iter_source_files(spec.source_root)
    relative_paths = [str(path.relative_to(spec.source_root)) for path in source_files]
    source_layout, layout_warnings = _build_source_layout(spec, relative_paths)
    release_check, release_warnings = _build_source_release_check(spec, relative_paths, source_layout)
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
        "warnings": [*layout_warnings, *release_warnings],
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
    return write_json(cache_paths.manifest_path, payload)

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import polars as pl

from .models import DatasetSpec


def normalize_source_column_name(value: object) -> str:
    return str(value).strip()


def normalize_source_header(
    columns: Iterable[object],
    *,
    drop_empty: bool = False,
) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_value in columns:
        column = normalize_source_column_name(raw_value)
        if not column and drop_empty:
            continue
        if column in seen:
            raise ValueError(f"Normalized source header contains duplicate column {column!r}.")
        seen.add(column)
        normalized.append(column)
    return normalized


def normalize_source_frame(
    frame: pl.DataFrame,
    *,
    drop_empty: bool = True,
) -> pl.DataFrame:
    rename_map: dict[str, str] = {}
    drop_columns: list[str] = []
    seen: set[str] = set()
    for column in frame.columns:
        normalized = normalize_source_column_name(column)
        if not normalized and drop_empty:
            drop_columns.append(column)
            continue
        if normalized in seen:
            raise ValueError(f"Normalized source frame contains duplicate column {normalized!r}.")
        seen.add(normalized)
        if normalized != column:
            rename_map[column] = normalized
    normalized_frame = frame.drop(drop_columns) if drop_columns else frame
    return normalized_frame.rename(rename_map) if rename_map else normalized_frame


def _read_greenbyte_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line in handle:
            if line.startswith("# Date and time,"):
                return normalize_source_header(next(csv.reader([line[2:].strip()])), drop_empty=True)
    raise ValueError(f"Failed to find Greenbyte header row in {path}.")


def _read_plain_csv_header(path: Path) -> list[str]:
    for encoding in ("utf-8-sig", "windows-1252"):
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                return _read_first_plain_csv_row(handle)
        except (LookupError, UnicodeDecodeError):
            continue
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return _read_first_plain_csv_row(handle)


def _read_first_plain_csv_row(handle) -> list[str]:
    reader = csv.reader(handle)
    for row in reader:
        if not row:
            continue
        if row[0].startswith("#"):
            continue
        return normalize_source_header(row, drop_empty=True)
    return []


def _greenbyte_source_group(path: Path) -> tuple[str, str]:
    name = path.name
    if name.startswith("Turbine_Data_"):
        return ("turbine_scada", "Turbine_Data")
    if name.startswith("Status_"):
        return ("status_events", "Status")
    if name.startswith("Device_Data_") and "PMU" in name:
        return ("farm_pmu", "Device_Data_PMU")
    if name.startswith("Device_Data_") and "Grid_Meter" in name:
        return ("farm_grid_meter", "Device_Data_Grid_Meter")
    if name.endswith("_WT_static.csv"):
        return ("turbine_static", "WT_static")
    if name.endswith("_dataSignalMapping.csv") or name.endswith("_dataSignalMapping.xlsx"):
        return ("signal_mapping", "dataSignalMapping")
    if name.endswith("_static.csv"):
        return ("site_static", "static")
    return ("other", path.stem)


def _hill_source_group(path: Path) -> tuple[str, str]:
    stem = path.stem
    if stem.startswith("tbl"):
        table_name = stem.split("_20", 1)[0]
        if table_name == stem:
            table_name = stem.split("_", 1)[0]
        return (table_name, table_name)
    if stem == "Hill_of_Towie_turbine_metadata":
        return ("turbine_metadata", stem)
    if stem == "Hill_of_Towie_AeroUp_install_dates":
        return ("aeroup_timeline", stem)
    if stem == "ShutdownDuration":
        return ("shutdown_duration", stem)
    return (stem, stem)


def _sdwpf_source_group(path: Path) -> tuple[str, str]:
    name = path.name
    if name == "sdwpf_baidukddcup2022_turb_location.csv":
        return ("location_csv", "sdwpf_location")
    if name.endswith(".csv"):
        return ("main_csv", "sdwpf_main")
    return ("other", path.stem)


def build_source_schema_inventory(spec: DatasetSpec) -> list[dict[str, object]]:
    inventory: dict[tuple[str, str, str], dict[str, object]] = {}
    for path in sorted(spec.source_root.rglob("*.csv")):
        if spec.handler == "greenbyte":
            source_asset, source_table_or_file = _greenbyte_source_group(path)
            if source_asset in {"turbine_scada", "farm_pmu", "farm_grid_meter"}:
                header = _read_greenbyte_header(path)
            else:
                header = _read_plain_csv_header(path)
        elif spec.handler == "hill_of_towie":
            source_asset, source_table_or_file = _hill_source_group(path)
            header = _read_plain_csv_header(path)
        elif spec.handler == "sdwpf_kddcup":
            source_asset, source_table_or_file = _sdwpf_source_group(path)
            header = _read_plain_csv_header(path)
        else:
            continue
        relative_path = str(path.relative_to(spec.source_root))
        for column in header:
            key = (source_asset, source_table_or_file, column)
            row = inventory.get(key)
            if row is None:
                inventory[key] = {
                    "source_asset": source_asset,
                    "source_table_or_file": source_table_or_file,
                    "source_column": column,
                    "example_files": [relative_path],
                    "file_count": 1,
                }
                continue
            files = row["example_files"]
            assert isinstance(files, list)
            if len(files) < 3 and relative_path not in files:
                files.append(relative_path)
            row["file_count"] = int(row["file_count"]) + 1
    return list(inventory.values())


def build_source_schema_inventory_exclusions(spec: DatasetSpec) -> tuple[list[dict[str, object]], list[str]]:
    if spec.handler != "greenbyte":
        return [], []

    xlsx_paths = sorted(spec.source_root.glob("*_dataSignalMapping.xlsx"))
    exclusions = [
        {
            "relative_path": str(path.relative_to(spec.source_root)),
            "source_asset": "signal_mapping",
            "source_table_or_file": "dataSignalMapping",
            "reason": "supported_runtime_asset_not_schema_inventoried",
        }
        for path in xlsx_paths
    ]
    if not exclusions:
        return exclusions, []

    csv_names = {path.name for path in spec.source_root.glob("*_dataSignalMapping.csv")}
    unmatched_xlsx_names = [
        path.name
        for path in xlsx_paths
        if path.with_suffix(".csv").name not in csv_names
    ]
    warnings: list[str] = []
    if unmatched_xlsx_names:
        warnings.append(
            "Source schema inventory skips supported Greenbyte signal-mapping workbooks with no "
            "corresponding CSV export: "
            + ", ".join(unmatched_xlsx_names)
            + "."
        )
    return exclusions, warnings


__all__ = [
    "build_source_schema_inventory",
    "build_source_schema_inventory_exclusions",
    "normalize_source_column_name",
    "normalize_source_frame",
    "normalize_source_header",
]

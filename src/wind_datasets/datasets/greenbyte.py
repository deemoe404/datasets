from __future__ import annotations

from array import array
import csv
import gc
import multiprocessing as mp
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from ..utils import ensure_directory, read_json, write_json
from ..source_column_policy import filter_source_frame, kept_source_columns
from ..source_schema import normalize_source_header
from .base import BaseDatasetBuilder
from .common import (
    ParquetChunkWriter,
    build_coverage_summary,
    build_quality_report,
    cast_numeric_columns,
    ensure_turbine_static_schema,
    featureize_interval_events,
    load_quality_report_frame,
    reindex_regular_series,
    sanitize_feature_name,
    write_quality_report,
)

_GREENBYTE_NON_FEATURE_COLUMNS = {
    "Date and time",
    "turbine_id",
    "source_file",
    "row_conflict_count",
    "source_row_count",
}

_GREENBYTE_CONFLICT_SCHEMA = pa.schema(
    [
        ("dataset", pa.string()),
        ("turbine_id", pa.string()),
        ("timestamp", pa.string()),
        ("column_name", pa.string()),
        ("existing_value", pa.string()),
        ("conflict_value", pa.string()),
        ("source_file", pa.string()),
    ]
)

_GREENBYTE_STATUS_ALIASES = {
    "stop": "evt_stop_active",
    "warning": "evt_warning_active",
    "informational": "evt_informational_active",
}

_GREENBYTE_SHARED_STATUS_ALIASES = {
    "stop": "farm_evt_stop_active",
    "warning": "farm_evt_warning_active",
    "informational": "farm_evt_informational_active",
}


def _discover_greenbyte_feature_columns(paths: list[Path], target_column: str) -> list[str]:
    discovered: list[str] = []
    seen: set[str] = set()
    for path in paths:
        for column in pl.scan_parquet(path).collect_schema().names():
            if column in _GREENBYTE_NON_FEATURE_COLUMNS or column == target_column:
                continue
            if column in seen:
                continue
            seen.add(column)
            discovered.append(column)
    return discovered


def _read_greenbyte_comment_table_metadata(path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line in handle:
            if line.startswith("# Turbine:"):
                turbine_id = line.split(":", 1)[1].strip()
                metadata["entity_id"] = turbine_id
                metadata["entity_type"] = "turbine"
                metadata["turbine_id"] = turbine_id
                metadata["asset_id"] = turbine_id
                metadata["asset_type"] = "turbine"
            elif line.startswith("# Device:"):
                asset_id = line.split(":", 1)[1].strip()
                metadata["entity_id"] = asset_id
                metadata["entity_type"] = "device"
                metadata["asset_id"] = asset_id
            elif line.startswith("# Device type:"):
                metadata["asset_type"] = line.split(":", 1)[1].strip()
            elif line.startswith("# Time zone:"):
                metadata["timezone"] = line.split(":", 1)[1].strip()
            elif line.startswith("# Date and time,"):
                metadata["header_line"] = line[2:].strip()
                break
    if "header_line" not in metadata:
        raise ValueError(f"Failed to find Greenbyte comment header in {path}.")
    return metadata


def _read_greenbyte_metadata(path: Path) -> dict[str, str]:
    metadata = _read_greenbyte_comment_table_metadata(path)
    if "turbine_id" not in metadata:
        raise ValueError(f"Expected turbine metadata in {path}.")
    return metadata


def _merge_greenbyte_group(
    rows: list[list[str]],
    headers: list[str],
    source_file: str,
    turbine_id: str,
    dataset_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    def non_missing_count(row: list[str]) -> int:
        return sum(value not in {"", "NaN"} for value in row[1:])

    best_row_index = max(range(len(rows)), key=lambda idx: non_missing_count(rows[idx]))
    best_row = rows[best_row_index].copy()
    conflicts: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        if index == best_row_index:
            continue
        for idx in range(1, len(headers)):
            candidate = row[idx] if idx < len(row) else ""
            if candidate in {"", "NaN"}:
                continue
            existing = best_row[idx] if idx < len(best_row) else ""
            if existing in {"", "NaN"}:
                best_row[idx] = candidate
            elif existing != candidate:
                conflicts.append(
                    {
                        "dataset": dataset_id,
                        "turbine_id": turbine_id,
                        "timestamp": best_row[0],
                        "column_name": headers[idx],
                        "existing_value": existing,
                        "conflict_value": candidate,
                        "source_file": source_file,
                    }
                )

    merged = {header: best_row[idx] if idx < len(best_row) else "" for idx, header in enumerate(headers)}
    merged["turbine_id"] = turbine_id
    merged["source_file"] = source_file
    merged["row_conflict_count"] = len(conflicts)
    merged["source_row_count"] = len(rows)
    return merged, conflicts


def _continuous_conflict_part_path(silver_dir: Path, stem: str) -> Path:
    return silver_dir / "conflict_parts" / f"{stem}.parquet"


def _continuous_stats_part_path(silver_dir: Path, stem: str) -> Path:
    return silver_dir / "continuous_stats" / f"{stem}.json"


def _greenbyte_worker_count(job_count: int) -> int:
    configured = os.environ.get("WIND_DATASETS_GREENBYTE_WORKERS")
    if configured:
        return max(1, min(int(configured), job_count))
    return max(1, min(job_count, os.cpu_count() or 1, 2))


def _write_greenbyte_comment_table_file(
    path: Path,
    output_path: Path,
    dataset_id: str,
    entity_column: str,
    entity_value: str,
    selected_headers: tuple[str, ...] | None = None,
    conflict_output_path: Path | None = None,
) -> dict[str, Any]:
    metadata = _read_greenbyte_comment_table_metadata(path)
    headers = normalize_source_header(next(csv.reader([metadata["header_line"]])), drop_empty=True)
    selected_header_set = set(selected_headers or ())
    source_file = str(path.relative_to(path.parents[1]))
    data_writer = ParquetChunkWriter(output_path)
    conflict_writer = (
        ParquetChunkWriter(conflict_output_path, schema=_GREENBYTE_CONFLICT_SCHEMA)
        if conflict_output_path is not None
        else None
    )
    column_count = len(headers)
    states: dict[str, dict[str, Any]] = {}
    conflict_count = 0
    row_index = 0

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            if row[0].startswith("#"):
                continue
            row_index += 1
            timestamp = row[0]
            state = states.get(timestamp)
            if state is None:
                state = {
                    "best_values": [""] * column_count,
                    "best_counts": array("i", [-1]) * column_count,
                    "best_row_ids": array("i", [2_147_483_647]) * column_count,
                    "seen_values": [None] * column_count,
                    "source_row_count": 0,
                }
                states[timestamp] = state

            sparse_cells: list[tuple[int, str]] = []
            for idx in range(1, column_count):
                candidate = row[idx] if idx < len(row) else ""
                if candidate in {"", "NaN"}:
                    continue
                sparse_cells.append((idx, candidate))

            non_missing_count = len(sparse_cells)
            state["source_row_count"] += 1
            best_values = state["best_values"]
            best_counts = state["best_counts"]
            best_row_ids = state["best_row_ids"]
            seen_values = state["seen_values"]

            for idx, candidate in sparse_cells:
                if (
                    non_missing_count > best_counts[idx]
                    or (
                        non_missing_count == best_counts[idx]
                        and row_index < best_row_ids[idx]
                    )
                ):
                    best_counts[idx] = non_missing_count
                    best_row_ids[idx] = row_index
                    best_values[idx] = candidate

                seen_value = seen_values[idx]
                if seen_value is None:
                    seen_values[idx] = candidate
                elif isinstance(seen_value, set):
                    seen_value.add(candidate)
                elif seen_value != candidate:
                    seen_values[idx] = {seen_value, candidate}

    merged_buffer: list[dict[str, Any]] = []
    conflict_buffer: list[dict[str, Any]] = []
    for timestamp in sorted(states):
        state = states[timestamp]
        best_values = state["best_values"]
        seen_values = state["seen_values"]
        conflict_columns = 0
        merged = {headers[0]: timestamp}
        for idx in range(1, column_count):
            merged[headers[idx]] = best_values[idx]
            seen_value = seen_values[idx]
            if (
                isinstance(seen_value, set)
                and len(seen_value) > 1
                and (not selected_header_set or headers[idx] in selected_header_set)
            ):
                conflict_columns += 1
                existing_value = best_values[idx]
                for alternative_value in sorted(value for value in seen_value if value != existing_value):
                    if conflict_writer is not None:
                        conflict_buffer.append(
                            {
                                "dataset": dataset_id,
                                "turbine_id": entity_value,
                                "timestamp": timestamp,
                                "column_name": headers[idx],
                                "existing_value": existing_value,
                                "conflict_value": alternative_value,
                                "source_file": source_file,
                            }
                        )
                    else:
                        conflict_count += 1
                if conflict_writer is not None and len(conflict_buffer) >= 10_000:
                    conflict_writer.write_rows(conflict_buffer)
                    conflict_count += len(conflict_buffer)
                    conflict_buffer = []

        merged[entity_column] = entity_value
        merged["source_file"] = source_file
        merged["row_conflict_count"] = conflict_columns
        merged["source_row_count"] = state["source_row_count"]
        if selected_header_set:
            merged = {
                key: value
                for key, value in merged.items()
                if key in selected_header_set or key in {entity_column, "source_file", "row_conflict_count", "source_row_count"}
            }
        merged_buffer.append(merged)
        if len(merged_buffer) >= 10_000:
            data_writer.write_rows(merged_buffer)
            merged_buffer = []

    data_writer.write_rows(merged_buffer)
    if conflict_writer is not None:
        conflict_writer.write_rows(conflict_buffer)
        conflict_count += len(conflict_buffer)
    data_writer.close()
    if conflict_writer is not None:
        conflict_writer.close()
    return {
        entity_column: entity_value,
        "source_file": source_file,
        "merged_row_count": len(states),
        "conflict_count": conflict_count,
    }


def _write_greenbyte_continuous_file(
    path: Path,
    output_path: Path,
    conflict_output_path: Path,
    dataset_id: str,
    selected_headers: tuple[str, ...],
) -> dict[str, Any]:
    metadata = _read_greenbyte_metadata(path)
    return _write_greenbyte_comment_table_file(
        path,
        output_path,
        dataset_id,
        entity_column="turbine_id",
        entity_value=metadata["turbine_id"],
        selected_headers=selected_headers,
        conflict_output_path=conflict_output_path,
    )


def _process_greenbyte_continuous_source(
    path_str: str,
    output_path_str: str,
    conflict_output_path_str: str,
    stats_output_path_str: str,
    dataset_id: str,
    selected_headers: tuple[str, ...],
) -> dict[str, Any]:
    stats = _write_greenbyte_continuous_file(
        Path(path_str),
        Path(output_path_str),
        Path(conflict_output_path_str),
        dataset_id,
        selected_headers,
    )
    write_json(Path(stats_output_path_str), stats)
    return stats


def _merge_greenbyte_conflict_parts(parts_dir: Path, output_path: Path) -> None:
    part_paths = sorted(parts_dir.glob("*.parquet"))
    if not part_paths:
        ParquetChunkWriter(output_path, schema=_GREENBYTE_CONFLICT_SCHEMA).close()
        return
    tables = [pq.read_table(path) for path in part_paths]
    pq.write_table(pa.concat_tables(tables), output_path)


def _parse_status_csv(path: Path) -> pl.DataFrame:
    metadata = _read_greenbyte_status_metadata(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(_iter_greenbyte_status_lines(handle))
        for row in reader:
            row["asset_id"] = metadata["asset_id"]
            row["asset_type"] = metadata["asset_type"]
            row["turbine_id"] = metadata["turbine_id"]
            row["source_file"] = str(path.relative_to(path.parents[1]))
            rows.append(row)
    return pl.DataFrame(rows) if rows else pl.DataFrame(
        schema={
            "Timestamp start": pl.String,
            "Timestamp end": pl.String,
            "Duration": pl.String,
            "Status": pl.String,
            "Code": pl.String,
            "Message": pl.String,
            "Comment": pl.String,
            "Service contract category": pl.String,
            "IEC category": pl.String,
            "Global contract category": pl.String,
            "Custom contract category": pl.String,
            "asset_id": pl.String,
            "asset_type": pl.String,
            "turbine_id": pl.String,
            "source_file": pl.String,
        }
    )


def _iter_greenbyte_status_lines(handle):
    for line in handle:
        if line.startswith("#"):
            continue
        if not line.strip():
            continue
        yield line


def _read_greenbyte_status_metadata(path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line in handle:
            if line.startswith("# Turbine:"):
                turbine_id = line.split(":", 1)[1].strip()
                metadata["asset_id"] = turbine_id
                metadata["asset_type"] = "turbine"
                metadata["turbine_id"] = turbine_id
            elif line.startswith("# ") and ":" in line:
                key = line[2:].split(":", 1)[0].strip()
                if key and key not in {"Time zone", "Time interval"} and not key.startswith(
                    "This file was exported by Greenbyte at"
                ):
                    metadata.setdefault("asset_id", key.removesuffix(" production").strip())
            elif not line.startswith("#") and line.strip():
                break
    if "asset_id" not in metadata:
        raise ValueError(f"Failed to parse status metadata from {path}.")
    metadata.setdefault("asset_type", "asset")
    metadata.setdefault("turbine_id", metadata["asset_id"])
    return metadata


def _greenbyte_device_group(metadata: dict[str, str], path: Path) -> str | None:
    asset_type = metadata.get("asset_type", "").strip().lower()
    stem = path.stem.lower()
    asset_id = metadata.get("asset_id", "").strip().lower()
    if "production meter" in asset_type or "pmu" in stem or "pmu" in asset_id:
        return "farm_pmu"
    if "grid meter" in asset_type or "grid_meter" in stem or "grid meter" in asset_id:
        return "farm_grid_meter"
    return None


def _greenbyte_shared_output_schema() -> pl.Schema:
    return {
        "dataset": pl.String,
        "timestamp": pl.Datetime,
    }


def _collect_greenbyte_event_frames(events_dir: Path) -> pl.DataFrame:
    paths = sorted(path for path in events_dir.glob("*.parquet") if path.name != "status_all.parquet")
    if not paths:
        return pl.DataFrame()
    return pl.concat([pl.read_parquet(path) for path in paths], how="diagonal_relaxed")


def _empty_greenbyte_shared_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=_greenbyte_shared_output_schema())


def _standardize_greenbyte_shared_parts(
    *,
    part_paths: list[Path],
    dataset_id: str,
    prefix: str,
) -> pl.DataFrame:
    if not part_paths:
        return _empty_greenbyte_shared_frame()
    frame = pl.concat([pl.read_parquet(path) for path in part_paths], how="diagonal_relaxed")
    frame = frame.with_columns(
        pl.col("Date and time")
        .cast(pl.String)
        .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
        .alias("timestamp")
    )
    payload_columns = [
        column
        for column in frame.columns
        if column
        not in {
            "Date and time",
            "timestamp",
            "asset_id",
            "source_file",
            "row_conflict_count",
            "source_row_count",
        }
    ]
    numeric_payload = cast_numeric_columns(frame, payload_columns)
    rename_expressions = [
        pl.col(column).alias(f"{prefix}__{sanitize_feature_name(column)}")
        for column in payload_columns
    ]
    standardized = numeric_payload.select(
        pl.lit(dataset_id).alias("dataset"),
        "timestamp",
        *rename_expressions,
    ).sort("timestamp")
    feature_columns = [column for column in standardized.columns if column not in {"dataset", "timestamp"}]
    return (
        standardized.group_by(["dataset", "timestamp"], maintain_order=True)
        .agg([pl.col(column).drop_nulls().first().alias(column) for column in feature_columns])
        .sort("timestamp")
    )


def _write_greenbyte_device_part(
    path: Path,
    output_path: Path,
    dataset_id: str,
    *,
    selected_headers: tuple[str, ...],
) -> dict[str, Any]:
    metadata = _read_greenbyte_comment_table_metadata(path)
    return _write_greenbyte_comment_table_file(
        path,
        output_path,
        dataset_id,
        entity_column="asset_id",
        entity_value=metadata["asset_id"],
        selected_headers=selected_headers,
    )


def _build_greenbyte_status_event_features(spec, cache_paths) -> None:
    raw_events = _collect_greenbyte_event_frames(cache_paths.silver_events_dir)
    if raw_events.is_empty():
        cache_paths.silver_event_features_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(schema={"dataset": pl.String, "turbine_id": pl.String, "timestamp": pl.Datetime}).write_parquet(
            cache_paths.silver_event_features_path("turbine_status")
        )
        pl.DataFrame(schema={"dataset": pl.String, "timestamp": pl.Datetime}).write_parquet(
            cache_paths.silver_event_features_path("farm_status")
        )
        return

    standardized = raw_events.with_columns(
        pl.lit(spec.dataset_id).alias("dataset"),
        pl.col("Timestamp start")
        .cast(pl.String)
        .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
        .alias("event_start"),
        pl.col("Timestamp end")
        .cast(pl.String)
        .str.strip_chars()
        .replace({"": None, "-": None})
        .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
        .alias("event_end"),
        pl.col("Status").cast(pl.String).str.strip_chars().replace({"": None}).alias("status"),
        pl.col("Code").cast(pl.String).str.strip_chars().replace({"": None}).alias("code"),
        pl.col("IEC category").cast(pl.String).str.strip_chars().replace({"": None}).alias("iec_category"),
        pl.col("Service contract category")
        .cast(pl.String)
        .str.strip_chars()
        .replace({"": None})
        .alias("service_contract_category"),
        pl.col("asset_id").cast(pl.String),
        pl.col("asset_type").cast(pl.String),
        pl.col("turbine_id").cast(pl.String),
    ).filter(pl.col("event_start").is_not_null())
    standardized.write_parquet(cache_paths.silver_events_dir / "status_all.parquet")

    turbine_events = standardized.filter(pl.col("turbine_id").is_in(list(spec.turbine_ids))).select(
        "dataset",
        "turbine_id",
        "event_start",
        "event_end",
        "status",
        "code",
        "iec_category",
        "service_contract_category",
    )
    farm_events = standardized.filter(~pl.col("turbine_id").is_in(list(spec.turbine_ids))).select(
        "dataset",
        "event_start",
        "event_end",
        "status",
        "code",
        "iec_category",
        "service_contract_category",
    )

    turbine_features = featureize_interval_events(
        events=turbine_events,
        resolution_minutes=spec.resolution_minutes,
        key_columns=("dataset", "turbine_id"),
        start_column="event_start",
        end_column="event_end",
        base_prefix="evt",
        categorical_prefixes=(
            ("status", "evt_status"),
            ("code", "evt_status_code"),
            ("iec_category", "evt_iec_category"),
            ("service_contract_category", "evt_service_category"),
        ),
        status_aliases=_GREENBYTE_STATUS_ALIASES,
    )
    farm_features = featureize_interval_events(
        events=farm_events,
        resolution_minutes=spec.resolution_minutes,
        key_columns=("dataset",),
        start_column="event_start",
        end_column="event_end",
        base_prefix="farm_evt",
        categorical_prefixes=(
            ("status", "farm_evt_status"),
            ("code", "farm_evt_status_code"),
            ("iec_category", "farm_evt_iec_category"),
            ("service_contract_category", "farm_evt_service_category"),
        ),
        status_aliases=_GREENBYTE_SHARED_STATUS_ALIASES,
    )
    ensure_directory(cache_paths.silver_event_features_dir)
    turbine_features.write_parquet(cache_paths.silver_event_features_path("turbine_status"))
    farm_features.write_parquet(cache_paths.silver_event_features_path("farm_status"))


def _join_greenbyte_extra_frames(base: pl.DataFrame, cache_paths) -> pl.DataFrame:
    joined = base
    for group_name in ("farm_pmu", "farm_grid_meter"):
        path = cache_paths.silver_shared_ts_path(group_name)
        if path.exists():
            joined = joined.join(pl.read_parquet(path), on=["dataset", "timestamp"], how="left")

    for group_name, keys in (
        ("turbine_status", ["dataset", "turbine_id", "timestamp"]),
        ("farm_status", ["dataset", "timestamp"]),
    ):
        path = cache_paths.silver_event_features_path(group_name)
        if path.exists():
            joined = joined.join(pl.read_parquet(path), on=keys, how="left")

    boolean_columns = [column for column, dtype in joined.schema.items() if dtype == pl.Boolean]
    event_numeric_columns = [
        column
        for column, dtype in joined.schema.items()
        if (
            column.startswith(("evt_", "farm_evt_"))
            and dtype.is_numeric()
            and not column.endswith("_days_since_aeroup_start")
            and not column.endswith("_days_since_aeroup_end")
        )
    ]
    if boolean_columns or event_numeric_columns:
        joined = joined.with_columns(
            [pl.col(column).fill_null(False).alias(column) for column in boolean_columns]
            + [pl.col(column).fill_null(0).alias(column) for column in event_numeric_columns]
        )
    return joined.sort(["dataset", "turbine_id", "timestamp"])


class GreenbyteDatasetBuilder(BaseDatasetBuilder):
    def required_silver_paths(self) -> tuple[Path, ...]:
        return (
            self.cache_paths.silver_turbine_static_path,
            self.cache_paths.silver_dir / "conflicts.parquet",
            self.cache_paths.silver_meta_dir / "continuous_build_stats.parquet",
            self.cache_paths.silver_shared_ts_path("farm_pmu"),
            self.cache_paths.silver_shared_ts_path("farm_grid_meter"),
            self.cache_paths.silver_event_features_path("turbine_status"),
            self.cache_paths.silver_event_features_path("farm_status"),
        )

    def build_silver(self) -> Path:
        self.ensure_manifest()
        source_policy = self.load_source_column_policy()
        if self.cache_paths.silver_dir.exists():
            shutil.rmtree(self.cache_paths.silver_dir)
        ensure_directory(self.cache_paths.silver_continuous_dir)
        ensure_directory(self.cache_paths.silver_events_dir)
        ensure_directory(self.cache_paths.silver_shared_ts_dir)
        ensure_directory(self.cache_paths.silver_event_features_dir)
        ensure_directory(self.cache_paths.silver_meta_dir)
        conflict_parts_dir = ensure_directory(self.cache_paths.silver_dir / "conflict_parts")
        stats_parts_dir = ensure_directory(self.cache_paths.silver_dir / "continuous_stats")
        shared_parts_dir = ensure_directory(self.cache_paths.silver_dir / "shared_ts_parts")

        stats: list[dict[str, Any]] = []
        pending_jobs: list[tuple[str, str, str, str, str]] = []
        for path in sorted(self.spec.source_root.rglob("Turbine_Data_*.csv")):
            output_path = self.cache_paths.silver_continuous_dir / f"{path.stem}.parquet"
            conflict_part_path = _continuous_conflict_part_path(self.cache_paths.silver_dir, path.stem)
            stats_part_path = _continuous_stats_part_path(self.cache_paths.silver_dir, path.stem)
            if output_path.exists() and conflict_part_path.exists() and stats_part_path.exists():
                stats.append(read_json(stats_part_path))
                continue
            pending_jobs.append(
                (
                    str(path),
                    str(output_path),
                    str(conflict_part_path),
                    str(stats_part_path),
                    self.spec.dataset_id,
                    kept_source_columns(
                        source_policy,
                        source_asset="turbine_scada",
                        source_table_or_file="Turbine_Data",
                        always_keep=("Date and time",),
                    ),
                )
            )

        worker_count = _greenbyte_worker_count(len(pending_jobs))
        if worker_count <= 1:
            for job in pending_jobs:
                stats.append(_process_greenbyte_continuous_source(*job))
        else:
            start_methods = mp.get_all_start_methods()
            context = mp.get_context("fork") if "fork" in start_methods else None
            with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
                futures = [executor.submit(_process_greenbyte_continuous_source, *job) for job in pending_jobs]
                for future in as_completed(futures):
                    stats.append(future.result())

        _merge_greenbyte_conflict_parts(conflict_parts_dir, self.cache_paths.silver_dir / "conflicts.parquet")
        stats.sort(key=lambda item: item["source_file"])

        pmu_part_paths: list[Path] = []
        grid_meter_part_paths: list[Path] = []
        for path in sorted(self.spec.source_root.rglob("Device_Data_*.csv")):
            metadata = _read_greenbyte_comment_table_metadata(path)
            group_name = _greenbyte_device_group(metadata, path)
            if group_name is None:
                continue
            part_path = shared_parts_dir / f"{path.stem}.parquet"
            if not part_path.exists():
                source_table = "Device_Data_PMU" if group_name == "farm_pmu" else "Device_Data_Grid_Meter"
                _write_greenbyte_device_part(
                    path,
                    part_path,
                    self.spec.dataset_id,
                    selected_headers=kept_source_columns(
                        source_policy,
                        source_asset=group_name,
                        source_table_or_file=source_table,
                        always_keep=("Date and time",),
                    ),
                )
            if group_name == "farm_pmu":
                pmu_part_paths.append(part_path)
            elif group_name == "farm_grid_meter":
                grid_meter_part_paths.append(part_path)

        _standardize_greenbyte_shared_parts(
            part_paths=pmu_part_paths,
            dataset_id=self.spec.dataset_id,
            prefix="farm_pmu",
        ).write_parquet(self.cache_paths.silver_shared_ts_path("farm_pmu"))
        _standardize_greenbyte_shared_parts(
            part_paths=grid_meter_part_paths,
            dataset_id=self.spec.dataset_id,
            prefix="farm_grid_meter",
        ).write_parquet(self.cache_paths.silver_shared_ts_path("farm_grid_meter"))

        for path in sorted(self.spec.source_root.rglob("Status_*.csv")):
            frame = filter_source_frame(
                _parse_status_csv(path),
                policy=source_policy,
                source_asset="status_events",
                source_table_or_file="Status",
                always_keep=("asset_id", "asset_type", "turbine_id", "source_file"),
            )
            frame.write_parquet(self.cache_paths.silver_events_dir / f"{path.stem}.parquet")
        _build_greenbyte_status_event_features(self.spec, self.cache_paths)

        for path in sorted(self.spec.source_root.glob("*_static.csv")):
            filter_source_frame(
                pl.read_csv(path),
                policy=source_policy,
                source_asset="site_static",
                source_table_or_file="static",
            ).write_parquet(self.cache_paths.silver_meta_dir / f"{path.stem}.parquet")

        for path in sorted(self.spec.source_root.glob("*_dataSignalMapping.csv")):
            filter_source_frame(
                pl.read_csv(path),
                policy=source_policy,
                source_asset="signal_mapping",
                source_table_or_file="dataSignalMapping",
            ).write_parquet(self.cache_paths.silver_meta_dir / f"{path.stem}.parquet")

        for path in sorted(self.spec.source_root.glob("*_dataSignalMapping.xlsx")):
            shutil.copy2(path, self.cache_paths.silver_meta_dir / path.name)

        static_frames: list[pl.DataFrame] = []
        for path in sorted(self.spec.source_root.glob("*_WT_static.csv")):
            frame = filter_source_frame(
                pl.read_csv(path),
                policy=source_policy,
                source_asset="turbine_static",
                source_table_or_file="WT_static",
            ).select(
                pl.lit(self.spec.dataset_id).alias("dataset"),
                pl.col("Title").cast(pl.String).alias("turbine_id"),
                pl.coalesce(
                    [
                        pl.col("Identity").cast(pl.String),
                        pl.col("Title").cast(pl.String),
                    ]
                ).alias("source_turbine_key"),
                pl.col("Latitude").cast(pl.Float64, strict=False).alias("latitude"),
                pl.col("Longitude").cast(pl.Float64, strict=False).alias("longitude"),
                pl.lit(None).cast(pl.Float64).alias("coord_x"),
                pl.lit(None).cast(pl.Float64).alias("coord_y"),
                pl.lit("geographic_latlon").alias("coord_kind"),
                pl.lit("EPSG:4326").alias("coord_crs"),
                pl.col("Elevation (m)").cast(pl.Float64, strict=False).alias("elevation_m"),
                pl.col("Rated power (kW)").cast(pl.Float64, strict=False).alias("rated_power_kw"),
                pl.col("Hub Height (m)").cast(pl.Float64, strict=False).alias("hub_height_m"),
                pl.col("Rotor Diameter (m)").cast(pl.Float64, strict=False).alias("rotor_diameter_m"),
                pl.col("Manufacturer").cast(pl.String).alias("manufacturer"),
                pl.col("Model").cast(pl.String).alias("model"),
                pl.col("Country").cast(pl.String).alias("country"),
                pl.col("Commercial Operations Date").cast(pl.String).alias("commercial_operation_date"),
                pl.lit(path.name).alias("spatial_source"),
            )
            static_frames.append(
                frame.filter(
                    pl.col("turbine_id").is_not_null()
                    & pl.col("turbine_id").is_in(list(self.spec.turbine_ids))
                )
            )
        turbine_static = ensure_turbine_static_schema(
            (
                pl.concat(static_frames, how="vertical").unique(subset=["turbine_id"], keep="first")
                if static_frames
                else pl.DataFrame()
            )
        )
        turbine_static.write_parquet(self.cache_paths.silver_turbine_static_path)

        pl.DataFrame(stats).write_parquet(self.cache_paths.silver_meta_dir / "continuous_build_stats.parquet")
        self._write_silver_build_meta()
        return self.cache_paths.silver_dir

    def build_gold_base(
        self,
        quality_profile: str | None = None,
        layout: str | None = None,
    ) -> Path:
        self.resolve_quality_profile(quality_profile)
        self.resolve_series_layout(layout)
        self.ensure_silver_fresh()

        manifest_payload = self.ensure_manifest()
        source_policy = self.load_source_column_policy()
        frames: list[pl.DataFrame] = []
        duplicate_source_rows = 0
        conflict_value_count = 0
        continuous_paths = sorted(self.cache_paths.silver_continuous_dir.glob("*.parquet"))
        selected_feature_columns = _discover_greenbyte_feature_columns(
            continuous_paths,
            self.spec.target_column,
        )

        for path in continuous_paths:
            frame = pl.read_parquet(path)
            available_feature_columns = [column for column in selected_feature_columns if column in frame.columns]
            numeric_columns = [self.spec.target_column, *available_feature_columns]
            frame = cast_numeric_columns(frame, numeric_columns)
            missing_feature_columns = [
                pl.lit(None).cast(pl.Float64).alias(column)
                for column in selected_feature_columns
                if column not in frame.columns
            ]
            frame = frame.with_columns(
                pl.col("Date and time")
                .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
                .alias("timestamp"),
                pl.col(self.spec.target_column).alias("target_kw"),
                pl.lit(self.spec.dataset_id).alias("dataset"),
                pl.lit(True).alias("__row_present"),
                pl.when(pl.col("row_conflict_count") > 0)
                .then(pl.lit("conflict_resolved"))
                .otherwise(pl.lit(""))
                .alias("__base_quality_flags"),
                pl.lit("").alias("__feature_quality_flags"),
                *missing_feature_columns,
            )
            duplicate_source_rows += int(frame.filter(pl.col("source_row_count") > 1).height)
            conflict_value_count += int(frame["row_conflict_count"].sum())
            frames.append(
                frame.select(
                    [
                        "dataset",
                        "turbine_id",
                        "timestamp",
                        "target_kw",
                        "__row_present",
                        "__base_quality_flags",
                        "__feature_quality_flags",
                        *selected_feature_columns,
                    ]
                )
            )

        gold_base = reindex_regular_series(
            pl.concat(frames, how="vertical"),
            self.spec,
            layout="farm",
        )
        gold_base = _join_greenbyte_extra_frames(gold_base, self.cache_paths)
        ensure_directory(self.cache_paths.gold_base_dir)
        series_path = self.cache_paths.gold_base_series_path
        quality_path = self.cache_paths.gold_base_quality_path
        gold_base.write_parquet(series_path)
        del gold_base
        del frames
        gc.collect()
        report_frame = load_quality_report_frame(series_path)

        report = build_quality_report(
            report_frame,
            manifest_payload,
            self.spec,
            extra={
                "series_layout": "farm_synchronous",
                "duplicate_source_row_groups": duplicate_source_rows,
                "conflict_value_count": conflict_value_count,
                **self.source_policy_report_extra(source_policy),
                **build_coverage_summary(report_frame, self.spec),
            },
        )
        write_quality_report(quality_path, report)
        self._write_gold_base_build_meta()
        return series_path

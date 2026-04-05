from __future__ import annotations

from array import array
import csv
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
from .base import BaseDatasetBuilder
from .common import (
    ParquetChunkWriter,
    build_quality_report,
    cast_numeric_columns,
    reindex_regular_series,
    write_quality_report,
)

_GREENBYTE_DEFAULT_FEATURES = [
    "Wind speed (m/s)",
    "Wind direction (°)",
    "Nacelle position (°)",
    "Generator RPM (RPM)",
    "Rotor speed (RPM)",
    "Ambient temperature (converter) (°C)",
    "Nacelle temperature (°C)",
    "Power factor (cosphi)",
    "Reactive power (kvar)",
    "Grid frequency (Hz)",
    "Blade angle (pitch position) A (°)",
    "Blade angle (pitch position) B (°)",
    "Blade angle (pitch position) C (°)",
]

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


def _read_greenbyte_metadata(path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line in handle:
            if line.startswith("# Turbine:"):
                metadata["turbine_id"] = line.split(":", 1)[1].strip()
            elif line.startswith("# Time zone:"):
                metadata["timezone"] = line.split(":", 1)[1].strip()
            elif line.startswith("# Date and time,"):
                metadata["header_line"] = line[2:].strip()
                break
    if "header_line" not in metadata:
        raise ValueError(f"Failed to find Greenbyte comment header in {path}.")
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


def _write_greenbyte_continuous_file(
    path: Path,
    output_path: Path,
    conflict_output_path: Path,
    dataset_id: str,
) -> dict[str, Any]:
    metadata = _read_greenbyte_metadata(path)
    headers = next(csv.reader([metadata["header_line"]]))
    turbine_id = metadata["turbine_id"]
    source_file = str(path.relative_to(path.parents[1]))
    data_writer = ParquetChunkWriter(output_path)
    conflict_writer = ParquetChunkWriter(conflict_output_path, schema=_GREENBYTE_CONFLICT_SCHEMA)
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
            if isinstance(seen_value, set) and len(seen_value) > 1:
                conflict_columns += 1
                existing_value = best_values[idx]
                for alternative_value in sorted(value for value in seen_value if value != existing_value):
                    conflict_buffer.append(
                        {
                            "dataset": dataset_id,
                            "turbine_id": turbine_id,
                            "timestamp": timestamp,
                            "column_name": headers[idx],
                            "existing_value": existing_value,
                            "conflict_value": alternative_value,
                            "source_file": source_file,
                        }
                    )
                if len(conflict_buffer) >= 10_000:
                    conflict_writer.write_rows(conflict_buffer)
                    conflict_count += len(conflict_buffer)
                    conflict_buffer = []

        merged["turbine_id"] = turbine_id
        merged["source_file"] = source_file
        merged["row_conflict_count"] = conflict_columns
        merged["source_row_count"] = state["source_row_count"]
        merged_buffer.append(merged)
        if len(merged_buffer) >= 10_000:
            data_writer.write_rows(merged_buffer)
            merged_buffer = []

    data_writer.write_rows(merged_buffer)
    conflict_writer.write_rows(conflict_buffer)
    conflict_count += len(conflict_buffer)
    data_writer.close()
    conflict_writer.close()
    return {
        "turbine_id": turbine_id,
        "source_file": source_file,
        "merged_row_count": len(states),
        "conflict_count": conflict_count,
    }


def _process_greenbyte_continuous_source(
    path_str: str,
    output_path_str: str,
    conflict_output_path_str: str,
    stats_output_path_str: str,
    dataset_id: str,
) -> dict[str, Any]:
    stats = _write_greenbyte_continuous_file(
        Path(path_str),
        Path(output_path_str),
        Path(conflict_output_path_str),
        dataset_id,
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


class GreenbyteDatasetBuilder(BaseDatasetBuilder):
    def build_silver(self) -> Path:
        self.ensure_manifest()
        ensure_directory(self.cache_paths.silver_continuous_dir)
        ensure_directory(self.cache_paths.silver_events_dir)
        ensure_directory(self.cache_paths.silver_meta_dir)
        conflict_parts_dir = ensure_directory(self.cache_paths.silver_dir / "conflict_parts")
        stats_parts_dir = ensure_directory(self.cache_paths.silver_dir / "continuous_stats")

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

        for path in sorted(self.spec.source_root.rglob("Status_*.csv")):
            frame = _parse_status_csv(path)
            frame.write_parquet(self.cache_paths.silver_events_dir / f"{path.stem}.parquet")

        for path in sorted(self.spec.source_root.glob("*_static.csv")):
            pl.read_csv(path).write_parquet(self.cache_paths.silver_meta_dir / f"{path.stem}.parquet")

        for path in sorted(self.spec.source_root.glob("*_dataSignalMapping.csv")):
            pl.read_csv(path).write_parquet(self.cache_paths.silver_meta_dir / f"{path.stem}.parquet")

        for path in sorted(self.spec.source_root.glob("*_dataSignalMapping.xlsx")):
            shutil.copy2(path, self.cache_paths.silver_meta_dir / path.name)

        pl.DataFrame(stats).write_parquet(self.cache_paths.silver_meta_dir / "continuous_build_stats.parquet")
        return self.cache_paths.silver_dir

    def build_gold_base(self, quality_profile: str | None = None) -> Path:
        resolved_quality_profile = self.resolve_quality_profile(quality_profile)
        if not self.cache_paths.silver_continuous_dir.exists():
            self.build_silver()

        manifest_payload = self.ensure_manifest()
        frames: list[pl.DataFrame] = []
        duplicate_source_rows = 0
        conflict_value_count = 0
        selected_feature_columns = list(_GREENBYTE_DEFAULT_FEATURES)

        for path in sorted(self.cache_paths.silver_continuous_dir.glob("*.parquet")):
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
                        *selected_feature_columns,
                    ]
                )
            )

        gold_base = reindex_regular_series(pl.concat(frames, how="vertical"), self.spec)
        ensure_directory(self.cache_paths.gold_base_profile_dir(resolved_quality_profile))
        series_path = self.cache_paths.gold_base_series_path_for(resolved_quality_profile)
        quality_path = self.cache_paths.gold_base_quality_path_for(resolved_quality_profile)
        gold_base.write_parquet(series_path)

        report = build_quality_report(
            gold_base,
            manifest_payload,
            self.spec,
            extra={
                "quality_profile": resolved_quality_profile,
                "duplicate_source_row_groups": duplicate_source_rows,
                "conflict_value_count": conflict_value_count,
            },
        )
        write_quality_report(quality_path, report)
        return series_path

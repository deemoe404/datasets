from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.parquet as pq

from ..utils import ensure_directory
from .base import BaseDatasetBuilder
from .common import (
    ParquetChunkWriter,
    QualityReportAccumulator,
    build_feature_name_mapping,
    build_coverage_summary,
    build_coverage_summary_from_series_path,
    build_quality_report,
    ensure_turbine_static_schema,
    featureize_interval_events,
    finalize_quality_report,
    load_quality_report_frame,
    sanitize_feature_name,
    update_quality_report_accumulator,
    write_quality_report,
)

_DEFAULT_TABLES = ("tblSCTurbine", "tblSCTurGrid", "tblSCTurFlag")
_DUPLICATE_AUDIT_SCHEMA = {
    "table_name": pl.String,
    "timestamp": pl.Datetime,
    "station_id": pl.String,
    "duplicate_count": pl.Int64,
    "duplicate_kind": pl.String,
    "is_conflicting": pl.Boolean,
    "normalized_equal_columns": pl.String,
    "conflicting_columns": pl.String,
}

_HILL_SHARED_TABLE_SPECS = (
    ("tblGrid", "farm_grid", "farm_grid", ("TimeStamp",)),
    ("tblGridScientific", "farm_grid_sci", "farm_grid_sci", ("TimeStamp",)),
    ("tblSCTurCount", "turbine_count", "tur_count", ("TimeStamp", "StationId")),
    ("tblSCTurDigiIn", "turbine_digi_in", "tur_digi_in", ("TimeStamp", "StationId")),
    ("tblSCTurDigiOut", "turbine_digi_out", "tur_digi_out", ("TimeStamp", "StationId")),
    ("tblSCTurIntern", "turbine_intern", "tur_intern", ("TimeStamp", "StationId")),
    ("tblSCTurPress", "turbine_press", "tur_press", ("TimeStamp", "StationId")),
    ("tblSCTurTemp", "turbine_temp", "tur_temp", ("TimeStamp", "StationId")),
)

_HILL_BROADCAST_SHARED_GROUPS = (
    "farm_grid",
    "farm_grid_sci",
)

_HILL_TURBINE_SHARED_GROUPS = (
    "turbine_shutdown_duration",
    "turbine_count",
    "turbine_digi_in",
    "turbine_digi_out",
    "turbine_intern",
    "turbine_press",
    "turbine_temp",
)

_HILL_EVENT_FEATURE_GROUPS = (
    "alarmlog",
    "aeroup",
)


def _read_csv_with_fallback(path):
    for encoding in ("utf8", "windows-1252", "utf8-lossy"):
        try:
            return pl.read_csv(path, encoding=encoding)
        except pl.exceptions.ComputeError:
            continue
    return pl.read_csv(path, encoding="utf8-lossy")


def _normalize_duplicate_value(value, dtype):
    if value is None:
        return None
    if dtype.is_numeric() or dtype == pl.Boolean:
        try:
            return round(float(value), 12)
        except (TypeError, ValueError):
            return None
    text = str(value).strip()
    return text or None


def _unique_non_null(values: list[object]) -> list[object]:
    unique_values: list[object] = []
    for value in values:
        if value is None:
            continue
        if value not in unique_values:
            unique_values.append(value)
    return unique_values


def _hill_timestamp_parse_expr(source_column: str = "TimeStamp") -> pl.Expr:
    return (
        pl.col(source_column)
        .cast(pl.Utf8)
        .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
    )


def _empty_hill_conflict_key_frame() -> pl.DataFrame:
    return pl.DataFrame(schema={"timestamp": pl.Datetime, "StationId": pl.String})


def _scan_ordered_hill_parquets(paths: list[Path]) -> pl.LazyFrame:
    if not paths:
        return pl.LazyFrame()
    return pl.concat(
        [
            pl.scan_parquet(path)
            .with_row_index("__source_row_nr")
            .with_columns(pl.lit(index).alias("__source_file_idx"))
            for index, path in enumerate(paths)
        ],
        how="diagonal_relaxed",
    )


def _sink_hill_lazy_frame(frame: pl.LazyFrame, output_path: Path) -> None:
    ensure_directory(output_path.parent)
    output_path.unlink(missing_ok=True)
    frame.sink_parquet(output_path)


def _dedupe_hill_duplicate_rows(frame: pl.DataFrame, key_columns: list[str]) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    sort_columns = [column for column in ("__source_file_idx", "__source_row_nr") if column in frame.columns]
    if sort_columns:
        frame = frame.sort(sort_columns)
    return frame.unique(subset=key_columns, keep="first", maintain_order=True)


def _audit_hill_duplicate_rows(
    frame: pl.DataFrame,
    table_name: str,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    if frame.is_empty():
        return frame, pl.DataFrame(schema=_DUPLICATE_AUDIT_SCHEMA), _empty_hill_conflict_key_frame()

    payload_columns = [
        column
        for column in frame.columns
        if column not in {"TimeStamp", "timestamp", "StationId", "__source_file_idx", "__source_row_nr"}
    ]
    source_rows = frame.sort([column for column in ("__source_file_idx", "__source_row_nr") if column in frame.columns])
    audit_rows: list[dict[str, object]] = []
    conflict_rows: list[dict[str, object]] = []
    for group in source_rows.partition_by(["timestamp", "StationId"], maintain_order=True):
        normalized_equal_columns: list[str] = []
        conflicting_columns: list[str] = []
        for column in payload_columns:
            raw_values = _unique_non_null(group[column].to_list())
            normalized_values: list[object] = []
            dtype = group.schema[column]
            for value in group[column].to_list():
                normalized = _normalize_duplicate_value(value, dtype)
                if normalized is None:
                    continue
                if normalized not in normalized_values:
                    normalized_values.append(normalized)
            if len(normalized_values) > 1:
                conflicting_columns.append(column)
            elif len(raw_values) > 1:
                normalized_equal_columns.append(column)
        duplicate_kind = (
            "true_conflict"
            if conflicting_columns
            else "normalized_equal"
            if normalized_equal_columns
            else "identical"
        )
        audit_rows.append(
            {
                "table_name": table_name,
                "timestamp": group["timestamp"][0],
                "station_id": group["StationId"][0],
                "duplicate_count": group.height,
                "duplicate_kind": duplicate_kind,
                "is_conflicting": bool(conflicting_columns),
                "normalized_equal_columns": "|".join(normalized_equal_columns),
                "conflicting_columns": "|".join(conflicting_columns),
            }
        )
        if conflicting_columns:
            conflict_rows.append(
                {
                    "timestamp": group["timestamp"][0],
                    "StationId": group["StationId"][0],
                }
            )

    audit_frame = (
        pl.DataFrame(audit_rows, schema=_DUPLICATE_AUDIT_SCHEMA).sort(["timestamp", "station_id"])
        if audit_rows
        else pl.DataFrame(schema=_DUPLICATE_AUDIT_SCHEMA)
    )
    conflict_keys = (
        pl.DataFrame(conflict_rows, schema={"timestamp": pl.Datetime, "StationId": pl.String})
        .unique()
        .sort(["timestamp", "StationId"])
        if conflict_rows
        else _empty_hill_conflict_key_frame()
    )
    deduped = _dedupe_hill_duplicate_rows(source_rows, ["timestamp", "StationId"])
    return deduped, audit_frame, conflict_keys


def _combine_hill_shared_duplicates(frame: pl.DataFrame, key_columns: list[str]) -> pl.DataFrame:
    if frame.is_empty():
        return frame

    source_rows = frame.sort([column for column in ("__source_file_idx", "__source_row_nr") if column in frame.columns])
    payload_columns = [
        column
        for column in source_rows.columns
        if column
        not in {
            "TimeStamp",
            "timestamp",
            "Station",
            "StationId",
            "TimestampStation",
            "WPSStatus",
            "DataOk",
            "dataset",
            "turbine_id",
            "__source_file_idx",
            "__source_row_nr",
            *key_columns,
        }
    ]
    if not payload_columns:
        return _dedupe_hill_duplicate_rows(source_rows, key_columns)
    return source_rows.group_by(key_columns, maintain_order=True).agg(
        [pl.col(column).drop_nulls().first().alias(column) for column in payload_columns]
    )

def _standardize_hill_shared_scan(
    *,
    scan: pl.LazyFrame,
    dataset_id: str,
    metadata: pl.DataFrame,
    prefix: str,
    key_columns: tuple[str, ...],
) -> pl.LazyFrame:
    join_keys: list[str]
    prepared = scan
    if key_columns == ("TimeStamp",):
        join_keys = ["timestamp"]
    else:
        prepared = prepared.join(metadata.lazy(), on="StationId", how="left")
        join_keys = ["turbine_id", "timestamp"]

    excluded = {
        "TimeStamp",
        "timestamp",
        "Station",
        "StationId",
        "TimestampStation",
        "WPSStatus",
        "DataOk",
        "dataset",
        "turbine_id",
        "__source_file_idx",
        "__source_row_nr",
    }
    payload_columns = [column for column in prepared.collect_schema().names() if column not in excluded]
    select_expressions: list[pl.Expr] = [
        pl.lit(dataset_id).alias("dataset"),
        pl.col("timestamp"),
    ]
    if "turbine_id" in join_keys:
        select_expressions.insert(1, pl.col("turbine_id").cast(pl.String))
    for column in payload_columns:
        select_expressions.append(
            pl.col(column)
            .cast(pl.Float64, strict=False)
            .alias(f"{prefix}__{sanitize_feature_name(column)}")
        )
    return prepared.select(select_expressions)


def _standardize_hill_shared_frame(
    *,
    frame: pl.DataFrame,
    dataset_id: str,
    metadata: pl.DataFrame,
    prefix: str,
    key_columns: tuple[str, ...],
) -> pl.DataFrame:
    if frame.is_empty():
        if key_columns == ("TimeStamp",):
            return pl.DataFrame(schema={"dataset": pl.String, "timestamp": pl.Datetime})
        return pl.DataFrame(schema={"dataset": pl.String, "turbine_id": pl.String, "timestamp": pl.Datetime})

    prepared = frame
    join_keys: list[str]
    if key_columns == ("TimeStamp",):
        join_keys = ["timestamp"]
    else:
        prepared = prepared.join(metadata, on="StationId", how="left")
        join_keys = ["turbine_id", "timestamp"]

    excluded = {
        "TimeStamp",
        "timestamp",
        "Station",
        "StationId",
        "TimestampStation",
        "WPSStatus",
        "DataOk",
        "dataset",
        "turbine_id",
        "__source_file_idx",
        "__source_row_nr",
    }
    payload_columns = [column for column in prepared.columns if column not in excluded]
    select_expressions: list[pl.Expr] = [
        pl.lit(dataset_id).alias("dataset"),
        pl.col("timestamp"),
    ]
    if "turbine_id" in join_keys:
        select_expressions.insert(1, pl.col("turbine_id").cast(pl.String))
    for column in payload_columns:
        select_expressions.append(
            pl.col(column)
            .cast(pl.Float64, strict=False)
            .alias(f"{prefix}__{sanitize_feature_name(column)}")
        )
    return prepared.select(select_expressions)


def _read_hill_source_part(path: Path, source_file_idx: int) -> pl.DataFrame:
    return (
        pl.read_parquet(path)
        .with_row_index("__source_row_nr")
        .with_columns(pl.lit(source_file_idx).alias("__source_file_idx"))
    )


def _hill_default_output_schema(paths: list[Path]) -> dict[str, pl.DataType]:
    schema: dict[str, pl.DataType] = {
        "timestamp": pl.Datetime,
        "StationId": pl.String,
    }
    for path in paths:
        for column, dtype in pl.scan_parquet(path).collect_schema().items():
            if column in {"TimeStamp", "StationId"}:
                continue
            schema.setdefault(column, dtype)
    return schema


def _hill_shared_output_schema(
    *,
    paths: list[Path],
    prefix: str,
    key_columns: tuple[str, ...],
) -> dict[str, pl.DataType]:
    schema: dict[str, pl.DataType] = {
        "dataset": pl.String,
        "timestamp": pl.Datetime,
    }
    if key_columns != ("TimeStamp",):
        schema["turbine_id"] = pl.String
    excluded = {
        "TimeStamp",
        "timestamp",
        "Station",
        "StationId",
        "TimestampStation",
        "WPSStatus",
        "DataOk",
        "dataset",
        "turbine_id",
        "__source_file_idx",
        "__source_row_nr",
    }
    for path in paths:
        for column in pl.scan_parquet(path).collect_schema().names():
            if column in excluded:
                continue
            schema.setdefault(f"{prefix}__{sanitize_feature_name(column)}", pl.Float64)
    return schema


def _hill_default_combined_frame(
    *,
    buffer: pl.DataFrame,
    current: pl.DataFrame,
    table_name: str,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    combined = pl.concat([buffer, current], how="diagonal_relaxed") if not buffer.is_empty() else current
    duplicate_keys = (
        combined.group_by(["timestamp", "StationId"])
        .len()
        .filter(pl.col("len") > 1)
        .select(["timestamp", "StationId"])
    )
    if duplicate_keys.is_empty():
        return (
            combined.sort(["timestamp", "__source_file_idx", "__source_row_nr"]),
            pl.DataFrame(schema=_DUPLICATE_AUDIT_SCHEMA),
            _empty_hill_conflict_key_frame(),
        )

    duplicate_rows = combined.join(duplicate_keys, on=["timestamp", "StationId"], how="inner")
    deduped_duplicates, audit_frame, conflict_keys = _audit_hill_duplicate_rows(duplicate_rows, table_name)
    nonduplicate_rows = combined.join(duplicate_keys, on=["timestamp", "StationId"], how="anti")
    resolved = pl.concat([nonduplicate_rows, deduped_duplicates], how="diagonal_relaxed").sort(
        ["timestamp", "__source_file_idx", "__source_row_nr"]
    )
    return resolved, audit_frame, conflict_keys


def _hill_shared_combined_frame(
    *,
    buffer: pl.DataFrame,
    current: pl.DataFrame,
    dedupe_keys: list[str],
) -> pl.DataFrame:
    combined = pl.concat([buffer, current], how="diagonal_relaxed") if not buffer.is_empty() else current
    duplicate_keys = (
        combined.group_by(dedupe_keys)
        .len()
        .filter(pl.col("len") > 1)
        .select(dedupe_keys)
    )
    if duplicate_keys.is_empty():
        return combined.sort(["timestamp", "__source_file_idx", "__source_row_nr"])

    duplicate_rows = combined.join(duplicate_keys, on=dedupe_keys, how="inner")
    deduped_duplicates = _combine_hill_shared_duplicates(duplicate_rows, dedupe_keys)
    nonduplicate_rows = combined.join(duplicate_keys, on=dedupe_keys, how="anti")
    return pl.concat([nonduplicate_rows, deduped_duplicates], how="diagonal_relaxed").sort(
        ["timestamp", "__source_file_idx", "__source_row_nr"]
    )


def _write_hill_default_table(
    *,
    paths: list[Path],
    output_path: Path,
    table_name: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    ensure_directory(output_path.parent)
    output_path.unlink(missing_ok=True)
    writer = ParquetChunkWriter(
        output_path,
        schema=pl.DataFrame(schema=_hill_default_output_schema(paths)).to_arrow().schema,
        row_group_size=1_000,
    )
    if not paths:
        writer.close()
        return pl.DataFrame(schema=_DUPLICATE_AUDIT_SCHEMA), _empty_hill_conflict_key_frame()

    buffer = pl.DataFrame()
    audit_frames: list[pl.DataFrame] = []
    conflict_frames: list[pl.DataFrame] = []
    for source_file_idx, path in enumerate(paths):
        current = _read_hill_source_part(path, source_file_idx).with_columns(
            _hill_timestamp_parse_expr().alias("timestamp"),
            pl.col("StationId").cast(pl.String).alias("StationId"),
        )
        if buffer.is_empty():
            buffer, audit_frame, conflict_keys = _hill_default_combined_frame(
                buffer=pl.DataFrame(),
                current=current,
                table_name=table_name,
            )
            if not audit_frame.is_empty():
                audit_frames.append(audit_frame)
            if not conflict_keys.is_empty():
                conflict_frames.append(conflict_keys)
            continue

        current_min = current["timestamp"].min()
        ready = buffer.filter(pl.col("timestamp") < current_min)
        overlap = buffer.filter(pl.col("timestamp") >= current_min)
        if not ready.is_empty():
            writer.write_frame(ready.drop(["TimeStamp", "__source_file_idx", "__source_row_nr"]).sort(["timestamp", "StationId"]))

        buffer, audit_frame, conflict_keys = _hill_default_combined_frame(
            buffer=overlap,
            current=current,
            table_name=table_name,
        )
        if not audit_frame.is_empty():
            audit_frames.append(audit_frame)
        if not conflict_keys.is_empty():
            conflict_frames.append(conflict_keys)

    if not buffer.is_empty():
        writer.write_frame(buffer.drop(["TimeStamp", "__source_file_idx", "__source_row_nr"]).sort(["timestamp", "StationId"]))
    writer.close()
    audit_frame = (
        pl.concat(audit_frames, how="vertical").sort(["timestamp", "station_id"])
        if audit_frames
        else pl.DataFrame(schema=_DUPLICATE_AUDIT_SCHEMA)
    )
    conflict_keys = (
        pl.concat(conflict_frames, how="vertical").unique().sort(["timestamp", "StationId"])
        if conflict_frames
        else _empty_hill_conflict_key_frame()
    )
    return audit_frame, conflict_keys


def _write_hill_shared_table(
    *,
    paths: list[Path],
    output_path: Path,
    dataset_id: str,
    metadata: pl.DataFrame,
    prefix: str,
    key_columns: tuple[str, ...],
) -> None:
    ensure_directory(output_path.parent)
    output_path.unlink(missing_ok=True)
    writer = ParquetChunkWriter(
        output_path,
        schema=pl.DataFrame(
            schema=_hill_shared_output_schema(paths=paths, prefix=prefix, key_columns=key_columns)
        ).to_arrow().schema,
        row_group_size=1_000,
    )
    if not paths:
        writer.close()
        return

    buffer = pl.DataFrame()
    dedupe_keys = ["timestamp"] if key_columns == ("TimeStamp",) else ["timestamp", "StationId"]
    for source_file_idx, path in enumerate(paths):
        current = _read_hill_source_part(path, source_file_idx).with_columns(
            _hill_timestamp_parse_expr().alias("timestamp"),
        )
        if key_columns != ("TimeStamp",):
            current = current.with_columns(pl.col("StationId").cast(pl.String).alias("StationId"))
        if buffer.is_empty():
            buffer = _hill_shared_combined_frame(
                buffer=pl.DataFrame(),
                current=current,
                dedupe_keys=dedupe_keys,
            )
            continue

        current_min = current["timestamp"].min()
        ready = buffer.filter(pl.col("timestamp") < current_min)
        overlap = buffer.filter(pl.col("timestamp") >= current_min)
        if not ready.is_empty():
            writer.write_frame(
                _standardize_hill_shared_frame(
                    frame=ready,
                    dataset_id=dataset_id,
                    metadata=metadata,
                    prefix=prefix,
                    key_columns=key_columns,
                )
            )

        buffer = _hill_shared_combined_frame(
            buffer=overlap,
            current=current,
            dedupe_keys=dedupe_keys,
        )

    if not buffer.is_empty():
        writer.write_frame(
            _standardize_hill_shared_frame(
                frame=buffer,
                dataset_id=dataset_id,
                metadata=metadata,
                prefix=prefix,
                key_columns=key_columns,
            )
        )
    writer.close()


def _max_hill_timestamp(paths: list[Path], timestamp_column: str = "TimeStamp") -> Any:
    maximum = None
    for path in paths:
        frame = pl.read_parquet(path, columns=[timestamp_column]).with_columns(
            _hill_timestamp_parse_expr(timestamp_column).alias("timestamp")
        )
        value = frame["timestamp"].max()
        if value is not None and (maximum is None or value > maximum):
            maximum = value
    return maximum


def _standardize_shutdown_duration(
    *,
    path: Path,
    dataset_id: str,
    metadata: pl.DataFrame,
) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame(schema={"dataset": pl.String, "turbine_id": pl.String, "timestamp": pl.Datetime})
    frame = pl.read_parquet(path).with_columns(
        pl.col("TimeStamp_StartFormat")
        .cast(pl.Utf8)
        .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%z", strict=False)
        .dt.replace_time_zone(None)
        .add(pl.duration(minutes=10))
        .alias("timestamp"),
        pl.col("TurbineName").cast(pl.String).alias("turbine_id"),
        pl.col("ShutdownDuration").cast(pl.Float64, strict=False).alias("shutdown_duration_s"),
    )
    return (
        frame.select(
            pl.lit(dataset_id).alias("dataset"),
            "turbine_id",
            "timestamp",
            "shutdown_duration_s",
        )
        .unique(subset=["dataset", "turbine_id", "timestamp"], keep="first", maintain_order=True)
        .sort(["dataset", "turbine_id", "timestamp"])
    )


def _standardize_alarmlog(
    *,
    frame: pl.DataFrame,
    dataset_id: str,
    metadata: pl.DataFrame,
) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame(
            schema={
                "dataset": pl.String,
                "turbine_id": pl.String,
                "event_start": pl.Datetime,
                "event_end": pl.Datetime,
                "alarm_code": pl.String,
            }
        )
    return (
        frame.with_columns(
            pl.col("TimeOn")
            .cast(pl.Utf8)
            .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
            .alias("event_start"),
            pl.col("TimeOff")
            .cast(pl.Utf8)
            .str.strip_chars()
            .replace({"": None, "-": None})
            .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
            .alias("event_end"),
            pl.col("StationNr").cast(pl.String).alias("StationId"),
            pl.col("Alarmcode").cast(pl.String).str.strip_chars().replace({"": None}).alias("alarm_code"),
        )
        .join(metadata, on="StationId", how="left")
        .select(
            pl.lit(dataset_id).alias("dataset"),
            "turbine_id",
            "event_start",
            "event_end",
            "alarm_code",
        )
        .filter(pl.col("event_start").is_not_null() & pl.col("turbine_id").is_not_null())
        .sort(["dataset", "turbine_id", "event_start"])
    )


def _standardize_aeroup(path: Path, dataset_id: str) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame(
            schema={
                "dataset": pl.String,
                "turbine_id": pl.String,
                "aeroup_start": pl.Datetime,
                "aeroup_end": pl.Datetime,
            }
        )
    frame = pl.read_parquet(path).with_columns(
        pl.col("Turbine").cast(pl.String).alias("turbine_id"),
        pl.col("First date of AeroUp works")
        .cast(pl.Utf8)
        .str.strptime(pl.Datetime, format="%Y-%m-%d", strict=False)
        .alias("aeroup_start"),
        pl.col("Last date of AeroUp works")
        .cast(pl.Utf8)
        .str.strptime(pl.Datetime, format="%Y-%m-%d", strict=False)
        .alias("aeroup_end"),
    )
    return frame.select(
        pl.lit(dataset_id).alias("dataset"),
        "turbine_id",
        "aeroup_start",
        "aeroup_end",
    ).filter(pl.col("turbine_id").is_not_null())


def _featureize_aeroup_interventions(
    frame: pl.DataFrame,
    resolution_minutes: int,
    dataset_end: Any | None = None,
) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame(
            schema={
                "dataset": pl.String,
                "turbine_id": pl.String,
                "timestamp": pl.Datetime,
                "aeroup_in_install_window": pl.Boolean,
                "aeroup_post_install": pl.Boolean,
                "days_since_aeroup_start": pl.Float64,
                "days_since_aeroup_end": pl.Float64,
            }
        )
    output_schema = {
        "dataset": pl.String,
        "turbine_id": pl.String,
        "timestamp": pl.Datetime,
        "aeroup_in_install_window": pl.Boolean,
        "aeroup_post_install": pl.Boolean,
        "days_since_aeroup_start": pl.Float64,
        "days_since_aeroup_end": pl.Float64,
    }
    output_frames: list[pl.DataFrame] = []
    for group in frame.partition_by(["dataset", "turbine_id"], maintain_order=True):
        dataset_id = group["dataset"][0]
        turbine_id = group["turbine_id"][0]
        start = group["aeroup_start"].min()
        end = group["aeroup_end"].max()
        if start is None:
            continue
        grid_end = max(
            value for value in (dataset_end, end, start) if value is not None
        )
        timestamps = pl.datetime_range(
            start=start,
            end=grid_end,
            interval=f"{resolution_minutes}m",
            eager=True,
        )
        rows: list[dict[str, Any]] = []
        for timestamp in timestamps.to_list():
            in_window = bool(start <= timestamp and (end is None or timestamp <= end))
            post_install = bool(end is not None and timestamp > end)
            days_since_start = (timestamp - start).total_seconds() / 86_400 if timestamp >= start else None
            days_since_end = (
                (timestamp - end).total_seconds() / 86_400
                if end is not None and timestamp > end
                else None
            )
            rows.append(
                {
                    "dataset": dataset_id,
                    "turbine_id": turbine_id,
                    "timestamp": timestamp,
                    "aeroup_in_install_window": in_window,
                    "aeroup_post_install": post_install,
                    "days_since_aeroup_start": days_since_start,
                    "days_since_aeroup_end": days_since_end,
                }
            )
        output_frames.append(pl.DataFrame(rows, schema=output_schema))
    return pl.concat(output_frames, how="vertical").sort(["dataset", "turbine_id", "timestamp"])


def _write_hill_feature_frame_per_turbine(
    *,
    source_path: Path,
    output_path: Path,
    empty_frame: pl.DataFrame,
    builder,
) -> None:
    ensure_directory(output_path.parent)
    output_path.unlink(missing_ok=True)
    writer = ParquetChunkWriter(output_path, schema=empty_frame.to_arrow().schema, row_group_size=1_000)
    if not source_path.exists():
        writer.close()
        return

    turbine_ids = (
        pl.scan_parquet(source_path)
        .select("turbine_id")
        .drop_nulls()
        .unique()
        .sort("turbine_id")
        .collect()["turbine_id"]
        .to_list()
    )
    for turbine_id in turbine_ids:
        writer.write_frame(
            builder(
                pl.scan_parquet(source_path)
                .filter(pl.col("turbine_id") == turbine_id)
                .collect()
            )
        )
    writer.close()


def _write_hill_alarm_features(
    *,
    events_path: Path,
    output_path: Path,
    resolution_minutes: int,
) -> None:
    alarm_code_columns: list[str] = []
    if events_path.exists():
        alarm_codes = (
            pl.scan_parquet(events_path)
            .select("alarm_code")
            .drop_nulls()
            .unique()
            .sort("alarm_code")
            .collect()["alarm_code"]
            .to_list()
        )
        alarm_code_columns = list(build_feature_name_mapping(alarm_codes, "alarm_code").values())
    empty_frame = pl.DataFrame(
        schema={
            "dataset": pl.String,
            "turbine_id": pl.String,
            "timestamp": pl.Datetime,
            "alarm_any_active": pl.Boolean,
            "alarm_active_count": pl.Int32,
            "alarm_total_overlap_seconds": pl.Int32,
            **{column: pl.Boolean for column in alarm_code_columns},
        }
    )
    _write_hill_feature_frame_per_turbine(
        source_path=events_path,
        output_path=output_path,
        empty_frame=empty_frame,
        builder=lambda frame: featureize_interval_events(
            events=frame,
            resolution_minutes=resolution_minutes,
            key_columns=("dataset", "turbine_id"),
            start_column="event_start",
            end_column="event_end",
            base_prefix="alarm",
            categorical_prefixes=(("alarm_code", "alarm_code"),),
        ),
    )


def _write_hill_aeroup_features(
    *,
    interventions_path: Path,
    output_path: Path,
    resolution_minutes: int,
    dataset_end: Any | None,
) -> None:
    empty_frame = _featureize_aeroup_interventions(
        pl.DataFrame(),
        resolution_minutes,
        dataset_end=dataset_end,
    )
    _write_hill_feature_frame_per_turbine(
        source_path=interventions_path,
        output_path=output_path,
        empty_frame=empty_frame,
        builder=lambda frame: _featureize_aeroup_interventions(
            frame,
            resolution_minutes,
            dataset_end=dataset_end,
        ),
    )


def _write_hill_alarmlog(
    *,
    paths: list[Path],
    output_path: Path,
    dataset_id: str,
    metadata: pl.DataFrame,
) -> None:
    ensure_directory(output_path.parent)
    output_path.unlink(missing_ok=True)
    empty_frame = _standardize_alarmlog(frame=pl.DataFrame(), dataset_id=dataset_id, metadata=metadata)
    if not paths:
        empty_frame.write_parquet(output_path)
        return

    writer = ParquetChunkWriter(output_path, schema=empty_frame.to_arrow().schema, row_group_size=1_000)
    for path in paths:
        writer.write_frame(
            _standardize_alarmlog(
                frame=pl.read_parquet(path),
                dataset_id=dataset_id,
                metadata=metadata,
            )
        )
    writer.close()


def _load_hill_time_slice(path: Path, start: Any, end: Any) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame()
    return (
        pl.scan_parquet(path)
        .filter((pl.col("timestamp") >= start) & (pl.col("timestamp") <= end))
        .collect()
    )


def _load_turbine_time_slice(path: Path, turbine_id: str, start: Any, end: Any) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame()
    return (
        pl.scan_parquet(path)
        .filter(
            (pl.col("turbine_id") == turbine_id)
            & (pl.col("timestamp") >= start)
            & (pl.col("timestamp") <= end)
        )
        .collect()
    )


def _hill_default_feature_columns(cache_paths, target_column: str) -> list[str]:
    feature_columns: list[str] = []
    for table_name in _DEFAULT_TABLES:
        path = cache_paths.hill_default_table_path(table_name)
        if not path.exists():
            continue
        for column in pl.scan_parquet(path).collect_schema().names():
            if column in {"timestamp", "StationId", target_column} or column in feature_columns:
                continue
            feature_columns.append(column)
    return feature_columns


def _hill_default_time_bounds(cache_paths) -> tuple[Any | None, Any | None]:
    scans = [
        pl.scan_parquet(path).select("timestamp")
        for table_name in _DEFAULT_TABLES
        if (path := cache_paths.hill_default_table_path(table_name)).exists()
    ]
    if not scans:
        return None, None
    bounds = (
        pl.concat(scans, how="vertical_relaxed")
        .select(
            pl.col("timestamp").min().alias("global_start"),
            pl.col("timestamp").max().alias("global_end"),
        )
        .collect()
        .row(0, named=True)
    )
    return bounds["global_start"], bounds["global_end"]


def _load_hill_station_default_table(path: Path, station_id: str) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame()
    return (
        pl.scan_parquet(path)
        .filter(pl.col("StationId") == station_id)
        .collect()
        .sort("timestamp")
    )


def _hill_base_output_schema(feature_columns: list[str]) -> dict[str, pl.DataType]:
    return {
        "dataset": pl.String,
        "turbine_id": pl.String,
        "timestamp": pl.Datetime,
        "target_kw": pl.Float64,
        "__row_present": pl.Boolean,
        "__base_quality_flags": pl.String,
        **{column: pl.Float64 for column in feature_columns},
    }


def _empty_hill_base_frame(feature_columns: list[str]) -> pl.DataFrame:
    return pl.DataFrame(schema=_hill_base_output_schema(feature_columns))


def _build_hill_base_turbine_frame(
    *,
    turbine_id: str,
    station_id: str | None,
    cache_paths,
    spec,
    feature_columns: list[str],
    conflict_keys: pl.DataFrame,
) -> pl.DataFrame:
    if station_id is None:
        return _empty_hill_base_frame(feature_columns)

    table_frames: list[pl.DataFrame] = []
    for table_name in _DEFAULT_TABLES:
        frame = _load_hill_station_default_table(cache_paths.hill_default_table_path(table_name), station_id)
        if not frame.is_empty():
            table_frames.append(frame)
    if not table_frames:
        return _empty_hill_base_frame(feature_columns)

    joined = pl.concat([frame.select("timestamp") for frame in table_frames], how="vertical_relaxed").unique().sort(
        "timestamp"
    )
    joined = joined.with_columns(
        pl.lit(station_id).alias("StationId"),
        pl.lit(spec.dataset_id).alias("dataset"),
        pl.lit(turbine_id).alias("turbine_id"),
        pl.lit(True).alias("__row_present"),
    )
    for frame in table_frames:
        payload_columns = [column for column in frame.columns if column not in {"timestamp", "StationId"}]
        joined = joined.join(
            frame.select(["timestamp", "StationId", *payload_columns]),
            on=["timestamp", "StationId"],
            how="left",
        )

    if conflict_keys.is_empty():
        joined = joined.with_columns(pl.lit(False).alias("__duplicate_conflict"))
    else:
        joined = joined.join(
            conflict_keys.with_columns(pl.lit(True).alias("__duplicate_conflict")),
            on=["timestamp", "StationId"],
            how="left",
        ).with_columns(pl.col("__duplicate_conflict").fill_null(False))

    required_payload_columns = [spec.target_column, *feature_columns]
    missing_payload_columns = [
        pl.lit(None).cast(pl.Float64).alias(column)
        for column in required_payload_columns
        if column not in joined.columns
    ]
    if missing_payload_columns:
        joined = joined.with_columns(missing_payload_columns)

    return joined.with_columns(
        pl.col(spec.target_column).cast(pl.Float64, strict=False).alias("target_kw"),
        pl.when(pl.col("__duplicate_conflict"))
        .then(pl.lit("duplicate_conflict_resolved"))
        .otherwise(pl.lit(""))
        .alias("__base_quality_flags"),
        *[pl.col(column).cast(pl.Float64, strict=False).alias(column) for column in feature_columns],
    ).select(
        [
            "dataset",
            "turbine_id",
            "timestamp",
            "target_kw",
            "__row_present",
            "__base_quality_flags",
            *feature_columns,
        ]
    )


def _reindex_hill_base_frame(
    *,
    base_frame: pl.DataFrame,
    turbine_id: str,
    spec,
    feature_columns: list[str],
    grid_start: Any | None,
    grid_end: Any | None,
) -> pl.DataFrame:
    if grid_start is None or grid_end is None:
        return pl.DataFrame(
            schema={
                "dataset": pl.String,
                "turbine_id": pl.String,
                "timestamp": pl.Datetime,
                "target_kw": pl.Float64,
                "is_observed": pl.Boolean,
                "quality_flags": pl.String,
                **{column: pl.Float64 for column in feature_columns},
            }
        )

    grid = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start=grid_start,
                end=grid_end,
                interval=f"{spec.resolution_minutes}m",
                eager=True,
            )
        }
    ).with_columns(pl.lit(turbine_id).alias("turbine_id"))
    joined = grid.join(base_frame, on=["turbine_id", "timestamp"], how="left") if not base_frame.is_empty() else grid

    missing_columns = [
        pl.lit(None).cast(dtype).alias(column)
        for column, dtype in _hill_base_output_schema(feature_columns).items()
        if column not in joined.columns
    ]
    if missing_columns:
        joined = joined.with_columns(missing_columns)

    return joined.with_columns(
        pl.coalesce([pl.col("dataset"), pl.lit(spec.dataset_id)]).alias("dataset"),
        pl.col("__row_present").fill_null(False).alias("is_observed"),
        pl.concat_str(
            [
                pl.when(pl.col("__base_quality_flags").fill_null("") != "")
                .then(pl.col("__base_quality_flags"))
                .otherwise(None),
                pl.when(~pl.col("__row_present").fill_null(False))
                .then(pl.lit("missing_row"))
                .otherwise(None),
                pl.when(pl.col("target_kw").is_null())
                .then(pl.lit("missing_target"))
                .otherwise(None),
            ],
            separator="|",
            ignore_nulls=True,
        )
        .fill_null("")
        .alias("quality_flags"),
    ).select(
        [
            "dataset",
            "turbine_id",
            "timestamp",
            "target_kw",
            "is_observed",
            "quality_flags",
            *feature_columns,
        ]
    )


def _iter_hill_base_chunks(
    *,
    cache_paths,
    spec,
    layout: str,
    feature_columns: list[str],
    station_by_turbine: dict[str, str],
    conflict_keys_by_station: dict[str, pl.DataFrame],
):
    global_start, global_end = _hill_default_time_bounds(cache_paths) if layout == "farm" else (None, None)
    turbine_ids = list(spec.turbine_ids)

    for turbine_id in turbine_ids:
        station_id = station_by_turbine.get(turbine_id)
        base_frame = _build_hill_base_turbine_frame(
            turbine_id=turbine_id,
            station_id=station_id,
            cache_paths=cache_paths,
            spec=spec,
            feature_columns=feature_columns,
            conflict_keys=conflict_keys_by_station.get(station_id, _empty_hill_conflict_key_frame()),
        )
        if layout == "turbine":
            if base_frame.is_empty():
                continue
            grid_start = base_frame["timestamp"].min()
            grid_end = base_frame["timestamp"].max()
        else:
            grid_start = global_start
            grid_end = global_end
        reindexed = _reindex_hill_base_frame(
            base_frame=base_frame,
            turbine_id=turbine_id,
            spec=spec,
            feature_columns=feature_columns,
            grid_start=grid_start,
            grid_end=grid_end,
        )
        if not reindexed.is_empty():
            yield reindexed


def _augment_hill_batch(
    base_batch: pl.DataFrame,
    extra_frames: dict[str, pl.DataFrame],
    expected_extra_schema: dict[str, pl.DataType],
) -> pl.DataFrame:
    if base_batch.is_empty():
        return base_batch
    joined = base_batch

    for frame in extra_frames.values():
        if frame.is_empty():
            continue
        join_keys = ["dataset", "timestamp"]
        if "turbine_id" in frame.columns:
            join_keys.insert(1, "turbine_id")
        joined = joined.join(frame, on=join_keys, how="left")

    missing_columns = [
        pl.lit(None).cast(dtype).alias(column)
        for column, dtype in expected_extra_schema.items()
        if column not in joined.columns
    ]
    if missing_columns:
        joined = joined.with_columns(missing_columns)

    boolean_columns = [column for column, dtype in joined.schema.items() if dtype == pl.Boolean]
    count_like_columns = [
        column
        for column, dtype in joined.schema.items()
        if dtype.is_numeric()
        and (
            column.startswith("alarm_")
            or column.startswith("evt_")
            or column == "shutdown_duration_s"
        )
        and not column.startswith("days_since_")
        and not column.endswith("days_since_aeroup_start")
        and not column.endswith("days_since_aeroup_end")
    ]
    if boolean_columns or count_like_columns:
        joined = joined.with_columns(
            [pl.col(column).fill_null(False).alias(column) for column in boolean_columns]
            + [pl.col(column).fill_null(0).alias(column) for column in count_like_columns]
        )
    return joined.sort(["dataset", "turbine_id", "timestamp"])


def _load_hill_chunk_extra_frames(base_chunk: pl.DataFrame, cache_paths) -> dict[str, pl.DataFrame]:
    if base_chunk.is_empty():
        return {}

    turbine_id = base_chunk["turbine_id"][0]
    chunk_start = base_chunk["timestamp"].min()
    chunk_end = base_chunk["timestamp"].max()
    frames: dict[str, pl.DataFrame] = {}
    for group_name in _HILL_BROADCAST_SHARED_GROUPS:
        frame = _load_hill_time_slice(
            cache_paths.silver_shared_ts_path(group_name),
            chunk_start,
            chunk_end,
        )
        if not frame.is_empty():
            frames[group_name] = frame
    for group_name in _HILL_TURBINE_SHARED_GROUPS:
        frame = _load_turbine_time_slice(
            cache_paths.silver_shared_ts_path(group_name),
            turbine_id,
            chunk_start,
            chunk_end,
        )
        if not frame.is_empty():
            frames[group_name] = frame
    for group_name in _HILL_EVENT_FEATURE_GROUPS:
        frame = _load_turbine_time_slice(
            cache_paths.silver_event_features_path(group_name),
            turbine_id,
            chunk_start,
            chunk_end,
        )
        if not frame.is_empty():
            frames[group_name] = frame
    return frames


def _write_hill_gold_with_extras(
    base_chunks,
    cache_paths,
    output_path: Path,
    spec,
    layout: str,
    batch_rows: int = 20_000,
) -> tuple[QualityReportAccumulator, dict[str, Any]]:
    ensure_directory(output_path.parent)
    temp_output_path = output_path.with_suffix(".tmp.parquet") if layout == "farm" else output_path
    temp_output_path.unlink(missing_ok=True)
    if layout == "farm":
        output_path.unlink(missing_ok=True)
    expected_extra_schema: dict[str, pl.DataType] = {}
    extra_paths = (
        [cache_paths.silver_shared_ts_path(group_name) for group_name in _HILL_BROADCAST_SHARED_GROUPS]
        + [cache_paths.silver_shared_ts_path(group_name) for group_name in _HILL_TURBINE_SHARED_GROUPS]
        + [cache_paths.silver_event_features_path(group_name) for group_name in _HILL_EVENT_FEATURE_GROUPS]
    )
    for path in extra_paths:
        if not path.exists():
            continue
        schema = pl.scan_parquet(path).collect_schema()
        for column, dtype in schema.items():
            if column not in {"dataset", "turbine_id", "timestamp"}:
                expected_extra_schema[column] = dtype
    writer = ParquetChunkWriter(temp_output_path, row_group_size=10_000)
    accumulator = QualityReportAccumulator()
    for base_chunk in base_chunks:
        if base_chunk.is_empty():
            continue
        update_quality_report_accumulator(accumulator, base_chunk, spec)
        extra_frames = _load_hill_chunk_extra_frames(base_chunk, cache_paths)
        for offset in range(0, base_chunk.height, batch_rows):
            batch = base_chunk.slice(offset, batch_rows)
            writer.write_frame(_augment_hill_batch(batch, extra_frames, expected_extra_schema))
    writer.close()
    if layout == "farm":
        coverage_summary = _finalize_hill_farm_temp(temp_output_path, output_path, spec, batch_rows=batch_rows * 10)
        return accumulator, coverage_summary

    return accumulator, build_coverage_summary_from_series_path(output_path, spec)


def _finalize_hill_farm_temp(
    temp_output_path: Path,
    output_path: Path,
    spec,
    batch_rows: int = 20_000,
) -> dict[str, Any]:
    timestamp_summary = (
        pl.scan_parquet(temp_output_path)
        .group_by(["dataset", "timestamp"], maintain_order=True)
        .agg(
            pl.col("is_observed").cast(pl.Int32).sum().alias("farm_turbines_observed"),
            pl.col("target_kw").is_not_null().cast(pl.Int32).sum().alias("farm_turbines_with_target"),
        )
        .with_columns(
            pl.lit(len(spec.turbine_ids)).cast(pl.Int32).alias("farm_turbines_expected"),
            (pl.col("farm_turbines_observed") == len(spec.turbine_ids)).alias("farm_is_fully_synchronous"),
            (pl.col("farm_turbines_with_target") == len(spec.turbine_ids)).alias("farm_has_all_targets"),
        )
        .collect()
    )
    writer = ParquetChunkWriter(output_path, row_group_size=1_000)
    parquet_file = pq.ParquetFile(temp_output_path)
    for batch in parquet_file.iter_batches(batch_size=batch_rows):
        frame = pl.from_arrow(batch)
        writer.write_frame(frame.join(timestamp_summary, on=["dataset", "timestamp"], how="left"))
    writer.close()
    temp_output_path.unlink(missing_ok=True)
    return build_coverage_summary(load_quality_report_frame(output_path), spec)


class HillOfTowieDatasetBuilder(BaseDatasetBuilder):
    def build_silver(self) -> Path:
        self.ensure_manifest()
        ensure_directory(self.cache_paths.silver_dir)
        ensure_directory(self.cache_paths.silver_events_dir)
        ensure_directory(self.cache_paths.silver_shared_ts_dir)
        ensure_directory(self.cache_paths.silver_event_features_dir)
        ensure_directory(self.cache_paths.silver_interventions_dir)
        ensure_directory(self.cache_paths.silver_meta_dir)
        ensure_directory(self.cache_paths.hill_default_tables_dir)
        for path in sorted(self.spec.source_root.rglob("*.csv")):
            relative = path.relative_to(self.spec.source_root)
            output_path = self.cache_paths.silver_dir / relative.with_suffix(".parquet")
            ensure_directory(output_path.parent)
            _read_csv_with_fallback(path).write_parquet(output_path)
        metadata_path = self.spec.source_root / "Hill_of_Towie_turbine_metadata.csv"
        if metadata_path.exists():
            turbine_static = ensure_turbine_static_schema(
                _read_csv_with_fallback(metadata_path).select(
                    pl.lit(self.spec.dataset_id).alias("dataset"),
                    pl.col("Turbine Name").cast(pl.String).alias("turbine_id"),
                    pl.col("Station ID").cast(pl.String).alias("source_turbine_key"),
                    pl.col("Latitude").cast(pl.Float64, strict=False).alias("latitude"),
                    pl.col("Longitude").cast(pl.Float64, strict=False).alias("longitude"),
                    pl.lit(None).cast(pl.Float64).alias("coord_x"),
                    pl.lit(None).cast(pl.Float64).alias("coord_y"),
                    pl.lit("geographic_latlon").alias("coord_kind"),
                    pl.lit("EPSG:4326").alias("coord_crs"),
                    pl.lit(None).cast(pl.Float64).alias("elevation_m"),
                    pl.col("Rated power (kW)").cast(pl.Float64, strict=False).alias("rated_power_kw"),
                    pl.col("Hub Height (m)").cast(pl.Float64, strict=False).alias("hub_height_m"),
                    pl.col("Rotor Diameter (m)").cast(pl.Float64, strict=False).alias("rotor_diameter_m"),
                    pl.col("Manufacturer").cast(pl.String).alias("manufacturer"),
                    pl.col("Model").cast(pl.String).alias("model"),
                    pl.col("Country").cast(pl.String).alias("country"),
                    pl.col("Commercial Operations Date").cast(pl.String).alias("commercial_operation_date"),
                    pl.lit(metadata_path.name).alias("spatial_source"),
                )
            ).filter(
                pl.col("turbine_id").is_not_null()
                & pl.col("turbine_id").is_in(list(self.spec.turbine_ids))
            ).unique(subset=["turbine_id"], keep="first")
        else:
            turbine_static = ensure_turbine_static_schema(pl.DataFrame())
        turbine_static.write_parquet(self.cache_paths.silver_turbine_static_path)

        metadata = pl.read_parquet(self.cache_paths.silver_dir / "Hill_of_Towie_turbine_metadata.parquet").select(
            pl.col("Station ID").cast(pl.String).alias("StationId"),
            pl.col("Turbine Name").cast(pl.String).alias("turbine_id"),
        )
        duplicate_audit_frames: list[pl.DataFrame] = []
        duplicate_conflict_keys: list[pl.DataFrame] = []
        for table_name in _DEFAULT_TABLES:
            audit_frame, conflict_keys = _write_hill_default_table(
                paths=sorted(self.cache_paths.silver_dir.rglob(f"{table_name}_*.parquet")),
                output_path=self.cache_paths.hill_default_table_path(table_name),
                table_name=table_name,
            )
            duplicate_audit_frames.append(audit_frame)
            if not conflict_keys.is_empty():
                duplicate_conflict_keys.append(conflict_keys)

        duplicate_audit = (
            pl.concat(duplicate_audit_frames, how="vertical")
            if duplicate_audit_frames
            else pl.DataFrame(schema=_DUPLICATE_AUDIT_SCHEMA)
        )
        duplicate_audit.write_parquet(self.cache_paths.hill_duplicate_audit_path)
        conflict_keys = (
            pl.concat(duplicate_conflict_keys, how="vertical")
            .unique()
            .sort(["timestamp", "StationId"])
            if duplicate_conflict_keys
            else _empty_hill_conflict_key_frame()
        )
        conflict_keys.write_parquet(self.cache_paths.hill_default_conflict_keys_path)

        dataset_end = _max_hill_timestamp(
            sorted(self.cache_paths.silver_dir.rglob("tblSCTurGrid_*.parquet"))
        )
        for table_name, group_name, prefix, key_columns in _HILL_SHARED_TABLE_SPECS:
            _write_hill_shared_table(
                paths=sorted(self.cache_paths.silver_dir.rglob(f"{table_name}_*.parquet")),
                output_path=self.cache_paths.silver_shared_ts_path(group_name),
                dataset_id=self.spec.dataset_id,
                metadata=metadata,
                prefix=prefix,
                key_columns=key_columns,
            )

        shutdown_path = self.cache_paths.silver_dir / "ShutdownDuration.parquet"
        _standardize_shutdown_duration(
            path=shutdown_path,
            dataset_id=self.spec.dataset_id,
            metadata=metadata,
        ).write_parquet(self.cache_paths.silver_shared_ts_path("turbine_shutdown_duration"))

        _write_hill_alarmlog(
            paths=sorted(self.cache_paths.silver_dir.rglob("tblAlarmLog_*.parquet")),
            output_path=self.cache_paths.silver_events_dir / "alarmlog.parquet",
            dataset_id=self.spec.dataset_id,
            metadata=metadata,
        )
        _write_hill_alarm_features(
            events_path=self.cache_paths.silver_events_dir / "alarmlog.parquet",
            output_path=self.cache_paths.silver_event_features_path("alarmlog"),
            resolution_minutes=self.spec.resolution_minutes,
        )

        aeroup = _standardize_aeroup(
            self.cache_paths.silver_dir / "Hill_of_Towie_AeroUp_install_dates.parquet",
            self.spec.dataset_id,
        )
        aeroup.write_parquet(self.cache_paths.silver_interventions_path("aeroup"))
        _write_hill_aeroup_features(
            interventions_path=self.cache_paths.silver_interventions_path("aeroup"),
            output_path=self.cache_paths.silver_event_features_path("aeroup"),
            resolution_minutes=self.spec.resolution_minutes,
            dataset_end=dataset_end,
        )
        return self.cache_paths.silver_dir

    def build_gold_base(
        self,
        quality_profile: str | None = None,
        layout: str | None = None,
        feature_set: str | None = None,
    ) -> Path:
        resolved_quality_profile = self.resolve_quality_profile(quality_profile)
        resolved_layout = self.resolve_series_layout(layout)
        resolved_feature_set = self.resolve_feature_set(feature_set)
        required_silver_paths = [
            self.cache_paths.silver_turbine_static_path,
            self.cache_paths.hill_duplicate_audit_path,
            self.cache_paths.hill_default_conflict_keys_path,
            *[self.cache_paths.hill_default_table_path(table_name) for table_name in _DEFAULT_TABLES],
            self.cache_paths.silver_shared_ts_path("farm_grid"),
            self.cache_paths.silver_shared_ts_path("farm_grid_sci"),
            self.cache_paths.silver_shared_ts_path("turbine_shutdown_duration"),
            self.cache_paths.silver_shared_ts_path("turbine_count"),
            self.cache_paths.silver_shared_ts_path("turbine_digi_in"),
            self.cache_paths.silver_shared_ts_path("turbine_digi_out"),
            self.cache_paths.silver_shared_ts_path("turbine_intern"),
            self.cache_paths.silver_shared_ts_path("turbine_press"),
            self.cache_paths.silver_shared_ts_path("turbine_temp"),
            self.cache_paths.silver_event_features_path("alarmlog"),
            self.cache_paths.silver_event_features_path("aeroup"),
        ]
        if any(not path.exists() for path in required_silver_paths):
            self.build_silver()

        manifest_payload = self.ensure_manifest()
        ensure_directory(
            self.cache_paths.gold_base_profile_dir(
                resolved_quality_profile,
                layout=resolved_layout,
                feature_set=resolved_feature_set,
            )
        )
        series_path = self.cache_paths.gold_base_series_path_for(
            resolved_quality_profile,
            layout=resolved_layout,
            feature_set=resolved_feature_set,
        )
        quality_path = self.cache_paths.gold_base_quality_path_for(
            resolved_quality_profile,
            layout=resolved_layout,
            feature_set=resolved_feature_set,
        )
        temp_series_path = series_path.with_suffix(".tmp.parquet") if resolved_layout == "farm" else None
        duplicate_audit_path = self.cache_paths.hill_duplicate_audit_path
        existing_duplicate_audit = (
            pl.read_parquet(duplicate_audit_path)
            if duplicate_audit_path.exists()
            else pl.DataFrame(schema=_DUPLICATE_AUDIT_SCHEMA)
        )
        existing_report_extra = {
            "quality_profile": resolved_quality_profile,
            "layout": resolved_layout,
            "feature_set": resolved_feature_set,
            "duplicate_key_audit_count": existing_duplicate_audit.height,
            "duplicate_conflict_key_count": existing_duplicate_audit.filter(pl.col("is_conflicting")).height,
        }
        if resolved_layout == "farm" and temp_series_path is not None and temp_series_path.exists() and not series_path.exists():
            coverage_summary = _finalize_hill_farm_temp(temp_series_path, series_path, self.spec)
            report_frame = load_quality_report_frame(series_path)
            report = build_quality_report(
                report_frame,
                manifest_payload,
                self.spec,
                extra={**existing_report_extra, **coverage_summary},
            )
            write_quality_report(quality_path, report)
            return series_path
        if series_path.exists() and not quality_path.exists():
            report_frame = load_quality_report_frame(series_path)
            report = build_quality_report(
                report_frame,
                manifest_payload,
                self.spec,
                extra={**existing_report_extra, **build_coverage_summary(report_frame, self.spec)},
            )
            write_quality_report(quality_path, report)
            return series_path

        global_start, global_end = _hill_default_time_bounds(self.cache_paths)
        if global_start is None or global_end is None:
            raise ValueError("No Hill of Towie default feature tables were found.")

        turbine_static = pl.read_parquet(self.cache_paths.silver_turbine_static_path).select(
            pl.col("turbine_id").cast(pl.String).alias("turbine_id"),
            pl.col("source_turbine_key").cast(pl.String).alias("StationId"),
        )
        station_by_turbine = {
            row["turbine_id"]: row["StationId"]
            for row in turbine_static.iter_rows(named=True)
            if row["turbine_id"] is not None and row["StationId"] is not None
        }
        conflict_keys = pl.read_parquet(self.cache_paths.hill_default_conflict_keys_path)
        conflict_keys_by_station = (
            {
                group["StationId"][0]: group.select(["timestamp", "StationId"]).sort("timestamp")
                for group in conflict_keys.partition_by("StationId", maintain_order=True)
            }
            if not conflict_keys.is_empty()
            else {}
        )
        feature_columns = _hill_default_feature_columns(self.cache_paths, self.spec.target_column)
        quality_accumulator, coverage_summary = _write_hill_gold_with_extras(
            _iter_hill_base_chunks(
                cache_paths=self.cache_paths,
                spec=self.spec,
                layout=resolved_layout,
                feature_columns=feature_columns,
                station_by_turbine=station_by_turbine,
                conflict_keys_by_station=conflict_keys_by_station,
            ),
            self.cache_paths,
            series_path,
            self.spec,
            layout=resolved_layout,
        )

        report = finalize_quality_report(
            quality_accumulator,
            manifest_payload,
            self.spec,
            extra={**existing_report_extra, **coverage_summary},
        )
        write_quality_report(quality_path, report)
        return series_path

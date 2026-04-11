from __future__ import annotations

from importlib import resources
from io import BytesIO
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.parquet as pq

from ..source_column_policy import filter_source_frame
from ..utils import ensure_directory, join_flags
from .base import BaseDatasetBuilder
from .common import (
    DUPLICATE_AUDIT_SCHEMA,
    DUPLICATE_EFFECTS_SCHEMA,
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
_HILL_ROW_EFFECT_SCOPE = "row"
_HILL_BROADCAST_EFFECT_SCOPE = "feature_broadcast"
_HILL_TURBINE_EFFECT_SCOPE = "feature_turbine"
_HILL_DUPLICATE_RESOLVED_STRATEGY = "per_column_first_non_null_with_true_conflict_nulling"
_HILL_DEFAULT_DUPLICATE_EXCLUDED_COLUMNS = {
    "TimeStamp",
    "timestamp",
    "StationId",
    "__source_file_idx",
    "__source_row_nr",
}
_HILL_SHARED_DUPLICATE_EXCLUDED_COLUMNS = {
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

_HILL_SHARED_TABLE_SPECS = (
    ("tblGrid", "farm_grid", "farm_grid", ("TimeStamp",), _HILL_BROADCAST_EFFECT_SCOPE),
    ("tblGridScientific", "farm_grid_sci", "farm_grid_sci", ("TimeStamp",), _HILL_BROADCAST_EFFECT_SCOPE),
    ("tblSCTurCount", "turbine_count", "tur_count", ("TimeStamp", "StationId"), _HILL_TURBINE_EFFECT_SCOPE),
    ("tblSCTurDigiIn", "turbine_digi_in", "tur_digi_in", ("TimeStamp", "StationId"), _HILL_TURBINE_EFFECT_SCOPE),
    ("tblSCTurDigiOut", "turbine_digi_out", "tur_digi_out", ("TimeStamp", "StationId"), _HILL_TURBINE_EFFECT_SCOPE),
    ("tblSCTurIntern", "turbine_intern", "tur_intern", ("TimeStamp", "StationId"), _HILL_TURBINE_EFFECT_SCOPE),
    ("tblSCTurPress", "turbine_press", "tur_press", ("TimeStamp", "StationId"), _HILL_TURBINE_EFFECT_SCOPE),
    ("tblSCTurTemp", "turbine_temp", "tur_temp", ("TimeStamp", "StationId"), _HILL_TURBINE_EFFECT_SCOPE),
)

_HILL_BROADCAST_SHARED_GROUPS = (
    "farm_grid",
)

_HILL_TURBINE_SHARED_GROUPS = (
    "turbine_press",
    "turbine_temp",
)

_HILL_EVENT_FEATURE_GROUPS = (
    "alarmlog",
    "aeroup",
)

_HILL_TUNEUP_METADATA_PACKAGE = "wind_datasets.data"
_HILL_TUNEUP_METADATA_RESOURCE = "hill_of_towie_tuneup_2024.csv"
_HILL_SERIES_EFFECT_SCHEMA: dict[str, pl.DataType] = {
    "dataset": pl.String,
    "turbine_id": pl.String,
    "timestamp": pl.Datetime,
    "__duplicate_row_quality_flags": pl.String,
    "__duplicate_feature_quality_flags": pl.String,
}


def _read_csv_with_fallback(path):
    for encoding in ("utf8", "windows-1252", "utf8-lossy"):
        try:
            return pl.read_csv(path, encoding=encoding)
        except pl.exceptions.ComputeError:
            continue
    return pl.read_csv(path, encoding="utf8-lossy")


def _classify_hill_source_csv(path: Path) -> tuple[str, str]:
    stem = path.stem
    if stem.startswith("tbl"):
        table_name = stem.split("_20", 1)[0]
        if table_name == stem:
            table_name = stem.split("_", 1)[0]
        return table_name, table_name
    if stem == "Hill_of_Towie_turbine_metadata":
        return "turbine_metadata", stem
    if stem == "Hill_of_Towie_AeroUp_install_dates":
        return "aeroup_timeline", stem
    if stem == "ShutdownDuration":
        return "shutdown_duration", stem
    return stem, stem


def _load_packaged_hill_tuneup_metadata() -> pl.DataFrame:
    resource = resources.files(_HILL_TUNEUP_METADATA_PACKAGE).joinpath(_HILL_TUNEUP_METADATA_RESOURCE)
    return pl.read_csv(
        BytesIO(resource.read_bytes()),
        schema_overrides={
            "turbine_id": pl.String,
            "tuneup_deployment_start": pl.String,
            "tuneup_effective_start": pl.String,
            "tuneup_deployment_end": pl.String,
            "source_configs": pl.String,
        },
    )


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


def _empty_hill_duplicate_audit_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=DUPLICATE_AUDIT_SCHEMA)


def _empty_hill_duplicate_effect_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=DUPLICATE_EFFECTS_SCHEMA)


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


def _canonicalize_duplicate_value(value: object, dtype: pl.DataType) -> object:
    if value is None:
        return None
    if dtype.is_numeric() or dtype == pl.Boolean:
        return value
    normalized = _normalize_duplicate_value(value, dtype)
    if normalized is None:
        return None
    return normalized


def _table_series_column_name(prefix: str | None, source_column: str) -> str:
    if prefix is None:
        return source_column
    return f"{prefix}__{sanitize_feature_name(source_column)}"


def _resolve_hill_duplicate_groups(
    frame: pl.DataFrame,
    *,
    table_name: str,
    effect_scope: str,
    key_columns: list[str],
    dataset_id: str,
    station_to_turbine: dict[str, str],
    series_column_prefix: str | None = None,
    feature_flag_token: str | None = None,
    excluded_payload_columns: set[str] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    if frame.is_empty():
        return frame, _empty_hill_duplicate_audit_frame(), _empty_hill_duplicate_effect_frame()

    excluded = set(excluded_payload_columns or set()) | set(key_columns)
    payload_columns = [column for column in frame.columns if column not in excluded]
    source_rows = frame.sort([column for column in ("__source_file_idx", "__source_row_nr") if column in frame.columns])
    audit_rows: list[dict[str, object]] = []
    effect_rows: list[dict[str, object]] = []
    resolved_rows: list[dict[str, object]] = []
    for group in source_rows.partition_by(key_columns, maintain_order=True):
        normalized_equal_columns: list[str] = []
        conflicting_source_columns: list[str] = []
        resolved_row = {
            column: group[column][0]
            for column in source_rows.columns
            if column not in {"__source_file_idx", "__source_row_nr"}
        }
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
                conflicting_source_columns.append(column)
                resolved_row[column] = None
            elif len(raw_values) > 1:
                normalized_equal_columns.append(column)
                resolved_row[column] = _canonicalize_duplicate_value(raw_values[0], dtype)
            elif len(raw_values) == 1:
                resolved_row[column] = _canonicalize_duplicate_value(raw_values[0], dtype)
            else:
                resolved_row[column] = None
        duplicate_kind = (
            "true_conflict"
            if conflicting_source_columns
            else "normalized_equal"
            if normalized_equal_columns
            else "identical"
        )
        timestamp = group["timestamp"][0]
        station_id = group["StationId"][0] if "StationId" in group.columns else None
        turbine_id = station_to_turbine.get(station_id) if station_id is not None else None
        affected_series_columns = [
            _table_series_column_name(series_column_prefix, column)
            for column in conflicting_source_columns
        ]
        audit_rows.append(
            {
                "table_name": table_name,
                "effect_scope": effect_scope,
                "timestamp": timestamp,
                "station_id": station_id,
                "turbine_id": turbine_id,
                "duplicate_count": group.height,
                "duplicate_kind": duplicate_kind,
                "normalized_equal_columns": normalized_equal_columns,
                "conflicting_source_columns": conflicting_source_columns,
                "affected_series_columns": affected_series_columns,
                "resolved_strategy": _HILL_DUPLICATE_RESOLVED_STRATEGY,
            }
        )
        if conflicting_source_columns:
            effect_rows.append(
                {
                    "dataset": dataset_id,
                    "turbine_id": turbine_id,
                    "timestamp": timestamp,
                    "effect_scope": effect_scope,
                    "source_tables": [table_name],
                    "row_quality_flags": "duplicate_conflict_resolved" if effect_scope == _HILL_ROW_EFFECT_SCOPE else "",
                    "feature_quality_flags": feature_flag_token or "",
                    "affected_series_columns": affected_series_columns,
                }
            )
        resolved_rows.append(resolved_row)

    resolved_frame = (
        pl.DataFrame(resolved_rows, schema={column: dtype for column, dtype in source_rows.schema.items() if column not in {"__source_file_idx", "__source_row_nr"}})
        .sort(key_columns)
        if resolved_rows
        else source_rows.drop([column for column in ("__source_file_idx", "__source_row_nr") if column in source_rows.columns])
    )
    audit_frame = (
        pl.DataFrame(audit_rows, schema=DUPLICATE_AUDIT_SCHEMA).sort(["timestamp", "table_name", "station_id", "turbine_id"])
        if audit_rows
        else _empty_hill_duplicate_audit_frame()
    )
    effects_frame = (
        pl.DataFrame(effect_rows, schema=DUPLICATE_EFFECTS_SCHEMA).sort(["timestamp", "effect_scope", "turbine_id"])
        if effect_rows
        else _empty_hill_duplicate_effect_frame()
    )
    return resolved_frame, audit_frame, effects_frame

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


def _aggregate_hill_duplicate_effects(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return _empty_hill_duplicate_effect_frame()
    rows: list[dict[str, object]] = []
    for group in frame.sort(["timestamp", "effect_scope", "turbine_id"]).partition_by(
        ["dataset", "timestamp", "turbine_id", "effect_scope"],
        maintain_order=True,
    ):
        source_tables: list[str] = []
        affected_series_columns: list[str] = []
        for value in group["source_tables"].to_list():
            for item in value or []:
                if item not in source_tables:
                    source_tables.append(item)
        for value in group["affected_series_columns"].to_list():
            for item in value or []:
                if item not in affected_series_columns:
                    affected_series_columns.append(item)
        rows.append(
            {
                "dataset": group["dataset"][0],
                "turbine_id": group["turbine_id"][0],
                "timestamp": group["timestamp"][0],
                "effect_scope": group["effect_scope"][0],
                "source_tables": source_tables,
                "row_quality_flags": join_flags(*group["row_quality_flags"].to_list()),
                "feature_quality_flags": join_flags(*group["feature_quality_flags"].to_list()),
                "affected_series_columns": affected_series_columns,
            }
        )
    return pl.DataFrame(rows, schema=DUPLICATE_EFFECTS_SCHEMA).sort(["timestamp", "effect_scope", "turbine_id"])


def _empty_hill_series_effect_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=_HILL_SERIES_EFFECT_SCHEMA)


def _aggregate_hill_series_effects(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return _empty_hill_series_effect_frame()

    rows: list[dict[str, object]] = []
    for group in frame.sort(["timestamp", "turbine_id"]).partition_by(
        ["dataset", "turbine_id", "timestamp"],
        maintain_order=True,
    ):
        rows.append(
            {
                "dataset": group["dataset"][0],
                "turbine_id": group["turbine_id"][0],
                "timestamp": group["timestamp"][0],
                "__duplicate_row_quality_flags": join_flags(*group["__duplicate_row_quality_flags"].to_list()),
                "__duplicate_feature_quality_flags": join_flags(
                    *group["__duplicate_feature_quality_flags"].to_list()
                ),
            }
        )
    return pl.DataFrame(rows, schema=_HILL_SERIES_EFFECT_SCHEMA).sort(["timestamp", "turbine_id"])


def _load_hill_duplicate_effects_slice(base_chunk: pl.DataFrame, cache_paths) -> pl.DataFrame:
    effects_path = cache_paths.duplicate_effects_path
    if base_chunk.is_empty() or not effects_path.exists():
        return _empty_hill_series_effect_frame()

    dataset_id = base_chunk["dataset"][0]
    chunk_start = base_chunk["timestamp"].min()
    chunk_end = base_chunk["timestamp"].max()
    effects = (
        pl.scan_parquet(effects_path)
        .filter(
            (pl.col("dataset") == dataset_id)
            & (pl.col("timestamp") >= chunk_start)
            & (pl.col("timestamp") <= chunk_end)
        )
        .collect()
    )
    if effects.is_empty():
        return _empty_hill_series_effect_frame()

    frames: list[pl.DataFrame] = []
    row_effects = effects.filter(pl.col("effect_scope") == _HILL_ROW_EFFECT_SCOPE)
    if not row_effects.is_empty():
        frames.append(
            row_effects.select(
                "dataset",
                "turbine_id",
                "timestamp",
                pl.col("row_quality_flags").alias("__duplicate_row_quality_flags"),
                pl.lit("").alias("__duplicate_feature_quality_flags"),
            )
        )

    turbine_feature_effects = effects.filter(pl.col("effect_scope") == _HILL_TURBINE_EFFECT_SCOPE)
    if not turbine_feature_effects.is_empty():
        frames.append(
            turbine_feature_effects.select(
                "dataset",
                "turbine_id",
                "timestamp",
                pl.lit("").alias("__duplicate_row_quality_flags"),
                pl.col("feature_quality_flags").alias("__duplicate_feature_quality_flags"),
            )
        )

    broadcast_effects = effects.filter(pl.col("effect_scope") == _HILL_BROADCAST_EFFECT_SCOPE)
    if not broadcast_effects.is_empty():
        base_keys = base_chunk.select(["dataset", "turbine_id", "timestamp"]).unique(maintain_order=True)
        frames.append(
            base_keys.join(
                broadcast_effects.select(["dataset", "timestamp", "feature_quality_flags"]),
                on=["dataset", "timestamp"],
                how="inner",
            ).select(
                "dataset",
                "turbine_id",
                "timestamp",
                pl.lit("").alias("__duplicate_row_quality_flags"),
                pl.col("feature_quality_flags").alias("__duplicate_feature_quality_flags"),
            )
        )

    if not frames:
        return _empty_hill_series_effect_frame()
    return _aggregate_hill_series_effects(pl.concat(frames, how="vertical_relaxed"))


def _build_hill_duplicate_report_extra(
    duplicate_audit: pl.DataFrame,
    duplicate_effects: pl.DataFrame,
    turbine_ids: tuple[str, ...],
) -> dict[str, object]:
    true_conflicts = duplicate_audit.filter(pl.col("duplicate_kind") == "true_conflict")
    duplicate_true_conflict_count_by_table = {
        row["table_name"]: int(row["count"])
        for row in true_conflicts.group_by("table_name", maintain_order=True)
        .len(name="count")
        .sort("table_name")
        .iter_rows(named=True)
    }

    row_conflict_row_count = (
        duplicate_effects.filter(pl.col("effect_scope") == _HILL_ROW_EFFECT_SCOPE)
        .select(["dataset", "turbine_id", "timestamp"])
        .unique()
        .height
    )

    feature_conflict_parts: list[pl.DataFrame] = []
    turbine_feature_rows = (
        duplicate_effects.filter(pl.col("effect_scope") == _HILL_TURBINE_EFFECT_SCOPE)
        .select(["dataset", "turbine_id", "timestamp"])
        .unique()
    )
    if not turbine_feature_rows.is_empty():
        feature_conflict_parts.append(turbine_feature_rows)

    broadcast_rows = (
        duplicate_effects.filter(pl.col("effect_scope") == _HILL_BROADCAST_EFFECT_SCOPE)
        .select(["dataset", "timestamp"])
        .unique()
    )
    if not broadcast_rows.is_empty():
        feature_conflict_parts.append(
            broadcast_rows.join(
                pl.DataFrame({"turbine_id": list(turbine_ids)}),
                how="cross",
            ).select(["dataset", "turbine_id", "timestamp"])
        )

    feature_conflict_row_count = (
        pl.concat(feature_conflict_parts, how="vertical_relaxed").unique().height
        if feature_conflict_parts
        else 0
    )

    duplicate_audit_count = duplicate_audit.height
    duplicate_true_conflict_count = true_conflicts.height
    return {
        "duplicate_audit_count": duplicate_audit_count,
        "duplicate_true_conflict_count": duplicate_true_conflict_count,
        "duplicate_true_conflict_count_by_table": duplicate_true_conflict_count_by_table,
        "row_conflict_row_count": row_conflict_row_count,
        "feature_conflict_row_count": feature_conflict_row_count,
        "duplicate_key_audit_count": duplicate_audit_count,
        "duplicate_conflict_key_count": duplicate_true_conflict_count,
    }


def _hill_combined_frame(
    *,
    buffer: pl.DataFrame,
    current: pl.DataFrame,
    table_name: str,
    key_columns: list[str],
    effect_scope: str,
    dataset_id: str,
    station_to_turbine: dict[str, str],
    excluded_payload_columns: set[str],
    series_column_prefix: str | None = None,
    feature_flag_token: str | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    combined = pl.concat([buffer, current], how="diagonal_relaxed") if not buffer.is_empty() else current
    duplicate_keys = (
        combined.group_by(key_columns)
        .len()
        .filter(pl.col("len") > 1)
        .select(key_columns)
    )
    if duplicate_keys.is_empty():
        return (
            combined.sort(["timestamp", "__source_file_idx", "__source_row_nr"]),
            _empty_hill_duplicate_audit_frame(),
            _empty_hill_duplicate_effect_frame(),
        )

    duplicate_rows = combined.join(duplicate_keys, on=key_columns, how="inner")
    resolved_duplicates, audit_frame, effects_frame = _resolve_hill_duplicate_groups(
        duplicate_rows,
        table_name=table_name,
        effect_scope=effect_scope,
        key_columns=key_columns,
        dataset_id=dataset_id,
        station_to_turbine=station_to_turbine,
        series_column_prefix=series_column_prefix,
        feature_flag_token=feature_flag_token,
        excluded_payload_columns=excluded_payload_columns,
    )
    nonduplicate_rows = combined.join(duplicate_keys, on=key_columns, how="anti")
    resolved = pl.concat([nonduplicate_rows, resolved_duplicates], how="diagonal_relaxed").sort(
        ["timestamp", "__source_file_idx", "__source_row_nr"]
    )
    return resolved, audit_frame, effects_frame


def _write_hill_default_table(
    *,
    paths: list[Path],
    output_path: Path,
    table_name: str,
    dataset_id: str,
    station_to_turbine: dict[str, str],
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
        return _empty_hill_duplicate_audit_frame(), _empty_hill_duplicate_effect_frame()

    buffer = pl.DataFrame()
    audit_frames: list[pl.DataFrame] = []
    effect_frames: list[pl.DataFrame] = []
    for source_file_idx, path in enumerate(paths):
        current = _read_hill_source_part(path, source_file_idx).with_columns(
            _hill_timestamp_parse_expr().alias("timestamp"),
            pl.col("StationId").cast(pl.String).alias("StationId"),
        )
        if buffer.is_empty():
            buffer, audit_frame, effects_frame = _hill_combined_frame(
                buffer=pl.DataFrame(),
                current=current,
                table_name=table_name,
                key_columns=["timestamp", "StationId"],
                effect_scope=_HILL_ROW_EFFECT_SCOPE,
                dataset_id=dataset_id,
                station_to_turbine=station_to_turbine,
                excluded_payload_columns=_HILL_DEFAULT_DUPLICATE_EXCLUDED_COLUMNS,
            )
            if not audit_frame.is_empty():
                audit_frames.append(audit_frame)
            if not effects_frame.is_empty():
                effect_frames.append(effects_frame)
            continue

        current_min = current["timestamp"].min()
        ready = buffer.filter(pl.col("timestamp") < current_min)
        overlap = buffer.filter(pl.col("timestamp") >= current_min)
        if not ready.is_empty():
            writer.write_frame(ready.drop(["TimeStamp", "__source_file_idx", "__source_row_nr"]).sort(["timestamp", "StationId"]))

        buffer, audit_frame, effects_frame = _hill_combined_frame(
            buffer=overlap,
            current=current,
            table_name=table_name,
            key_columns=["timestamp", "StationId"],
            effect_scope=_HILL_ROW_EFFECT_SCOPE,
            dataset_id=dataset_id,
            station_to_turbine=station_to_turbine,
            excluded_payload_columns=_HILL_DEFAULT_DUPLICATE_EXCLUDED_COLUMNS,
        )
        if not audit_frame.is_empty():
            audit_frames.append(audit_frame)
        if not effects_frame.is_empty():
            effect_frames.append(effects_frame)

    if not buffer.is_empty():
        writer.write_frame(buffer.drop(["TimeStamp", "__source_file_idx", "__source_row_nr"]).sort(["timestamp", "StationId"]))
    writer.close()
    audit_frame = (
        pl.concat(audit_frames, how="vertical").sort(["timestamp", "station_id"])
        if audit_frames
        else _empty_hill_duplicate_audit_frame()
    )
    effects_frame = (
        _aggregate_hill_duplicate_effects(pl.concat(effect_frames, how="vertical"))
        if effect_frames
        else _empty_hill_duplicate_effect_frame()
    )
    return audit_frame, effects_frame


def _write_hill_shared_table(
    *,
    paths: list[Path],
    output_path: Path,
    table_name: str,
    dataset_id: str,
    group_name: str,
    metadata: pl.DataFrame,
    station_to_turbine: dict[str, str],
    prefix: str,
    key_columns: tuple[str, ...],
    effect_scope: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
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
        return _empty_hill_duplicate_audit_frame(), _empty_hill_duplicate_effect_frame()

    buffer = pl.DataFrame()
    audit_frames: list[pl.DataFrame] = []
    effect_frames: list[pl.DataFrame] = []
    dedupe_keys = ["timestamp"] if key_columns == ("TimeStamp",) else ["timestamp", "StationId"]
    for source_file_idx, path in enumerate(paths):
        current = _read_hill_source_part(path, source_file_idx).with_columns(
            _hill_timestamp_parse_expr().alias("timestamp"),
        )
        if key_columns != ("TimeStamp",):
            current = current.with_columns(pl.col("StationId").cast(pl.String).alias("StationId"))
        if buffer.is_empty():
            buffer, audit_frame, effects_frame = _hill_combined_frame(
                buffer=pl.DataFrame(),
                current=current,
                table_name=table_name,
                key_columns=dedupe_keys,
                effect_scope=effect_scope,
                dataset_id=dataset_id,
                station_to_turbine=station_to_turbine,
                excluded_payload_columns=_HILL_SHARED_DUPLICATE_EXCLUDED_COLUMNS,
                series_column_prefix=prefix,
                feature_flag_token=f"feature_source_conflict__{group_name}",
            )
            if not audit_frame.is_empty():
                audit_frames.append(audit_frame)
            if not effects_frame.is_empty():
                effect_frames.append(effects_frame)
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

        buffer, audit_frame, effects_frame = _hill_combined_frame(
            buffer=overlap,
            current=current,
            table_name=table_name,
            key_columns=dedupe_keys,
            effect_scope=effect_scope,
            dataset_id=dataset_id,
            station_to_turbine=station_to_turbine,
            excluded_payload_columns=_HILL_SHARED_DUPLICATE_EXCLUDED_COLUMNS,
            series_column_prefix=prefix,
            feature_flag_token=f"feature_source_conflict__{group_name}",
        )
        if not audit_frame.is_empty():
            audit_frames.append(audit_frame)
        if not effects_frame.is_empty():
            effect_frames.append(effects_frame)

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
    audit_frame = (
        pl.concat(audit_frames, how="vertical").sort(["timestamp", "table_name", "station_id", "turbine_id"])
        if audit_frames
        else _empty_hill_duplicate_audit_frame()
    )
    effects_frame = (
        _aggregate_hill_duplicate_effects(pl.concat(effect_frames, how="vertical"))
        if effect_frames
        else _empty_hill_duplicate_effect_frame()
    )
    return audit_frame, effects_frame


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


def _standardize_tuneup(
    dataset_id: str,
    turbine_ids: tuple[str, ...],
) -> pl.DataFrame:
    frame = _load_packaged_hill_tuneup_metadata().with_columns(
        pl.col("turbine_id").cast(pl.String),
        pl.col("tuneup_deployment_start")
        .cast(pl.Utf8)
        .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
        .alias("tuneup_deployment_start"),
        pl.col("tuneup_effective_start")
        .cast(pl.Utf8)
        .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
        .alias("tuneup_effective_start"),
        pl.col("tuneup_deployment_end")
        .cast(pl.Utf8)
        .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
        .alias("tuneup_deployment_end"),
    )
    return (
        frame.select(
            pl.lit(dataset_id).alias("dataset"),
            "turbine_id",
            "tuneup_deployment_start",
            "tuneup_effective_start",
            "tuneup_deployment_end",
        )
        .filter(
            pl.col("turbine_id").is_not_null()
            & pl.col("turbine_id").is_in(list(turbine_ids))
        )
        .unique(subset=["dataset", "turbine_id"], keep="first")
        .sort(["dataset", "turbine_id"])
    )


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


def _featureize_tuneup_interventions(
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
                "tuneup_in_deployment_window": pl.Boolean,
                "tuneup_post_effective": pl.Boolean,
                "days_since_tuneup_effective_start": pl.Float64,
                "days_since_tuneup_deployment_end": pl.Float64,
            }
        )
    output_schema = {
        "dataset": pl.String,
        "turbine_id": pl.String,
        "timestamp": pl.Datetime,
        "tuneup_in_deployment_window": pl.Boolean,
        "tuneup_post_effective": pl.Boolean,
        "days_since_tuneup_effective_start": pl.Float64,
        "days_since_tuneup_deployment_end": pl.Float64,
    }
    output_frames: list[pl.DataFrame] = []
    for group in frame.partition_by(["dataset", "turbine_id"], maintain_order=True):
        dataset_id = group["dataset"][0]
        turbine_id = group["turbine_id"][0]
        deployment_start = group["tuneup_deployment_start"].min()
        effective_start = group["tuneup_effective_start"].min()
        deployment_end = group["tuneup_deployment_end"].max()
        grid_start = min(
            value for value in (deployment_start, effective_start, deployment_end) if value is not None
        )
        grid_end = max(
            value for value in (dataset_end, deployment_end, effective_start, grid_start) if value is not None
        )
        timestamps = pl.datetime_range(
            start=grid_start,
            end=grid_end,
            interval=f"{resolution_minutes}m",
            eager=True,
        )
        rows: list[dict[str, Any]] = []
        for timestamp in timestamps.to_list():
            in_window = bool(
                deployment_start is not None
                and deployment_start <= timestamp
                and (deployment_end is None or timestamp <= deployment_end)
            )
            post_effective = bool(effective_start is not None and timestamp >= effective_start)
            days_since_effective = (
                (timestamp - effective_start).total_seconds() / 86_400
                if effective_start is not None and timestamp >= effective_start
                else None
            )
            days_since_deployment_end = (
                (timestamp - deployment_end).total_seconds() / 86_400
                if deployment_end is not None and timestamp > deployment_end
                else None
            )
            rows.append(
                {
                    "dataset": dataset_id,
                    "turbine_id": turbine_id,
                    "timestamp": timestamp,
                    "tuneup_in_deployment_window": in_window,
                    "tuneup_post_effective": post_effective,
                    "days_since_tuneup_effective_start": days_since_effective,
                    "days_since_tuneup_deployment_end": days_since_deployment_end,
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


def _write_hill_tuneup_features(
    *,
    interventions_path: Path,
    output_path: Path,
    resolution_minutes: int,
    dataset_end: Any | None,
) -> None:
    empty_frame = _featureize_tuneup_interventions(
        pl.DataFrame(),
        resolution_minutes,
        dataset_end=dataset_end,
    )
    _write_hill_feature_frame_per_turbine(
        source_path=interventions_path,
        output_path=output_path,
        empty_frame=empty_frame,
        builder=lambda frame: _featureize_tuneup_interventions(
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


def _hill_extra_feature_columns(cache_paths) -> list[str]:
    feature_columns: list[str] = []
    extra_paths = (
        [cache_paths.silver_shared_ts_path(group_name) for group_name in _HILL_BROADCAST_SHARED_GROUPS]
        + [cache_paths.silver_shared_ts_path(group_name) for group_name in _HILL_TURBINE_SHARED_GROUPS]
        + [cache_paths.silver_event_features_path(group_name) for group_name in _HILL_EVENT_FEATURE_GROUPS]
    )
    for path in extra_paths:
        if not path.exists():
            continue
        for column in pl.scan_parquet(path).collect_schema().names():
            if column in {"dataset", "turbine_id", "timestamp"} or column in feature_columns:
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
        "__feature_quality_flags": pl.String,
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
        pl.lit("").alias("__base_quality_flags"),
        pl.lit("").alias("__feature_quality_flags"),
        *[pl.col(column).cast(pl.Float64, strict=False).alias(column) for column in feature_columns],
    ).select(
        [
            "dataset",
            "turbine_id",
            "timestamp",
            "target_kw",
            "__row_present",
            "__base_quality_flags",
            "__feature_quality_flags",
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
                "feature_quality_flags": pl.String,
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
        pl.col("__feature_quality_flags").fill_null("").alias("feature_quality_flags"),
    ).select(
        [
            "dataset",
            "turbine_id",
            "timestamp",
            "target_kw",
            "is_observed",
            "quality_flags",
            "feature_quality_flags",
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
    for column in ("__duplicate_row_quality_flags", "__duplicate_feature_quality_flags"):
        if column not in joined.columns:
            joined = joined.with_columns(pl.lit("").alias(column))

    joined = joined.with_columns(
        pl.struct(["quality_flags", "__duplicate_row_quality_flags"])
        .map_elements(
            lambda value: join_flags(
                value["quality_flags"],
                value["__duplicate_row_quality_flags"],
            ),
            return_dtype=pl.String,
        )
        .alias("quality_flags"),
        pl.struct(["feature_quality_flags", "__duplicate_feature_quality_flags"])
        .map_elements(
            lambda value: join_flags(
                value["feature_quality_flags"],
                value["__duplicate_feature_quality_flags"],
            ),
            return_dtype=pl.String,
        )
        .alias("feature_quality_flags"),
    ).drop(["__duplicate_row_quality_flags", "__duplicate_feature_quality_flags"])

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
        and not column.endswith("days_since_tuneup_effective_start")
        and not column.endswith("days_since_tuneup_deployment_end")
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
    duplicate_effects = _load_hill_duplicate_effects_slice(base_chunk, cache_paths)
    if not duplicate_effects.is_empty():
        frames["duplicate_effects"] = duplicate_effects
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
    def required_silver_paths(self) -> tuple[Path, ...]:
        return (
            self.cache_paths.silver_turbine_static_path,
            self.cache_paths.duplicate_audit_path,
            self.cache_paths.duplicate_effects_path,
            self.cache_paths.hill_default_table_path("tblSCTurbine"),
            self.cache_paths.hill_default_table_path("tblSCTurGrid"),
            self.cache_paths.hill_default_table_path("tblSCTurFlag"),
            self.cache_paths.silver_events_dir / "alarmlog.parquet",
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
            self.cache_paths.silver_event_features_path("tuneup"),
            self.cache_paths.silver_interventions_path("aeroup"),
            self.cache_paths.silver_interventions_path("tuneup"),
        )

    def build_silver(self) -> Path:
        self.ensure_manifest()
        source_policy = self.load_source_column_policy()
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
            source_asset, source_table_or_file = _classify_hill_source_csv(path)
            filter_source_frame(
                _read_csv_with_fallback(path),
                policy=source_policy,
                source_asset=source_asset,
                source_table_or_file=source_table_or_file,
            ).write_parquet(output_path)
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
        station_to_turbine = {
            row["StationId"]: row["turbine_id"]
            for row in metadata.iter_rows(named=True)
            if row["StationId"] is not None and row["turbine_id"] is not None
        }
        duplicate_audit_frames: list[pl.DataFrame] = []
        duplicate_effect_frames: list[pl.DataFrame] = []
        for table_name in _DEFAULT_TABLES:
            audit_frame, effects_frame = _write_hill_default_table(
                paths=sorted(self.cache_paths.silver_dir.rglob(f"{table_name}_*.parquet")),
                output_path=self.cache_paths.hill_default_table_path(table_name),
                table_name=table_name,
                dataset_id=self.spec.dataset_id,
                station_to_turbine=station_to_turbine,
            )
            duplicate_audit_frames.append(audit_frame)
            if not effects_frame.is_empty():
                duplicate_effect_frames.append(effects_frame)

        duplicate_audit = (
            pl.concat(duplicate_audit_frames, how="vertical")
            if duplicate_audit_frames
            else _empty_hill_duplicate_audit_frame()
        )
        dataset_end = _max_hill_timestamp(
            sorted(self.cache_paths.silver_dir.rglob("tblSCTurGrid_*.parquet"))
        )
        for table_name, group_name, prefix, key_columns, effect_scope in _HILL_SHARED_TABLE_SPECS:
            audit_frame, effects_frame = _write_hill_shared_table(
                paths=sorted(self.cache_paths.silver_dir.rglob(f"{table_name}_*.parquet")),
                output_path=self.cache_paths.silver_shared_ts_path(group_name),
                table_name=table_name,
                dataset_id=self.spec.dataset_id,
                group_name=group_name,
                metadata=metadata,
                station_to_turbine=station_to_turbine,
                prefix=prefix,
                key_columns=key_columns,
                effect_scope=effect_scope,
            )
            duplicate_audit_frames.append(audit_frame)
            if not effects_frame.is_empty():
                duplicate_effect_frames.append(effects_frame)

        duplicate_audit = (
            pl.concat(duplicate_audit_frames, how="vertical").sort(["timestamp", "table_name", "station_id", "turbine_id"])
            if duplicate_audit_frames
            else _empty_hill_duplicate_audit_frame()
        )
        duplicate_audit.write_parquet(self.cache_paths.duplicate_audit_path)
        duplicate_effects = (
            _aggregate_hill_duplicate_effects(pl.concat(duplicate_effect_frames, how="vertical_relaxed"))
            if duplicate_effect_frames
            else _empty_hill_duplicate_effect_frame()
        )
        duplicate_effects.write_parquet(self.cache_paths.duplicate_effects_path)

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
        tuneup = _standardize_tuneup(self.spec.dataset_id, self.spec.turbine_ids)
        tuneup.write_parquet(self.cache_paths.silver_interventions_path("tuneup"))
        _write_hill_tuneup_features(
            interventions_path=self.cache_paths.silver_interventions_path("tuneup"),
            output_path=self.cache_paths.silver_event_features_path("tuneup"),
            resolution_minutes=self.spec.resolution_minutes,
            dataset_end=dataset_end,
        )
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
        ensure_directory(self.cache_paths.gold_base_dir)
        series_path = self.cache_paths.gold_base_series_path
        quality_path = self.cache_paths.gold_base_quality_path
        temp_series_path = series_path.with_suffix(".tmp.parquet")
        duplicate_audit_path = self.cache_paths.duplicate_audit_path
        existing_duplicate_audit = (
            pl.read_parquet(duplicate_audit_path)
            if duplicate_audit_path.exists()
            else _empty_hill_duplicate_audit_frame()
        )
        duplicate_effects_path = self.cache_paths.duplicate_effects_path
        existing_duplicate_effects = (
            pl.read_parquet(duplicate_effects_path)
            if duplicate_effects_path.exists()
            else _empty_hill_duplicate_effect_frame()
        )
        feature_columns = _hill_default_feature_columns(self.cache_paths, self.spec.target_column)
        existing_report_extra = {
            "series_layout": "farm_synchronous",
            **self.source_policy_report_extra(source_policy),
            **_build_hill_duplicate_report_extra(
                existing_duplicate_audit,
                existing_duplicate_effects,
                self.spec.turbine_ids,
            ),
        }
        if temp_series_path.exists() and not series_path.exists():
            coverage_summary = _finalize_hill_farm_temp(temp_series_path, series_path, self.spec)
            report_frame = load_quality_report_frame(series_path)
            report = build_quality_report(
                report_frame,
                manifest_payload,
                self.spec,
                extra={**existing_report_extra, **coverage_summary},
            )
            write_quality_report(quality_path, report)
            self._write_gold_base_build_meta()
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
            self._write_gold_base_build_meta()
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
        quality_accumulator, coverage_summary = _write_hill_gold_with_extras(
            _iter_hill_base_chunks(
                cache_paths=self.cache_paths,
                spec=self.spec,
                layout="farm",
                feature_columns=feature_columns,
                station_by_turbine=station_by_turbine,
            ),
            self.cache_paths,
            series_path,
            self.spec,
            layout="farm",
        )

        report = finalize_quality_report(
            quality_accumulator,
            manifest_payload,
            self.spec,
            extra={
                **existing_report_extra,
                **coverage_summary,
            },
        )
        write_quality_report(quality_path, report)
        self._write_gold_base_build_meta()
        return series_path

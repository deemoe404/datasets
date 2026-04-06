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
    build_quality_report,
    build_coverage_summary,
    ensure_turbine_static_schema,
    featureize_interval_events,
    finalize_quality_report,
    iter_reindexed_regular_series,
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


def _audit_hill_duplicate_keys(
    frame: pl.DataFrame,
    table_name: str,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    frame = frame.with_columns(
        pl.col("TimeStamp")
        .cast(pl.Utf8)
        .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
        .alias("timestamp"),
        pl.col("StationId").cast(pl.String).alias("StationId"),
    )
    duplicate_keys = (
        frame.group_by(["timestamp", "StationId"])
        .len()
        .filter(pl.col("len") > 1)
        .sort(["timestamp", "StationId"])
    )
    if duplicate_keys.is_empty():
        return (
            frame.unique(subset=["timestamp", "StationId"], keep="first", maintain_order=True),
            pl.DataFrame(schema=_DUPLICATE_AUDIT_SCHEMA),
            pl.DataFrame(schema={"timestamp": pl.Datetime, "StationId": pl.String}),
        )

    payload_columns = [
        column
        for column in frame.columns
        if column not in {"TimeStamp", "timestamp", "StationId"}
    ]
    duplicate_rows = frame.join(
        duplicate_keys.select(["timestamp", "StationId"]),
        on=["timestamp", "StationId"],
        how="inner",
    )
    audit_rows: list[dict[str, object]] = []
    conflict_rows: list[dict[str, object]] = []
    for group in duplicate_rows.partition_by(["timestamp", "StationId"], maintain_order=True):
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
    audit_frame = pl.DataFrame(audit_rows, schema=_DUPLICATE_AUDIT_SCHEMA)
    conflict_keys = (
        pl.DataFrame(conflict_rows, schema={"timestamp": pl.Datetime, "StationId": pl.String}).unique()
        if conflict_rows
        else pl.DataFrame(schema={"timestamp": pl.Datetime, "StationId": pl.String})
    )
    deduped = frame.unique(subset=["timestamp", "StationId"], keep="first", maintain_order=True)
    return deduped, audit_frame, conflict_keys


def _concat_hill_tables(paths: list[Path]) -> pl.DataFrame:
    if not paths:
        return pl.DataFrame()
    return pl.concat([pl.read_parquet(path) for path in paths], how="diagonal_relaxed")


def _hill_timestamp_frame(frame: pl.DataFrame, timestamp_column: str) -> pl.DataFrame:
    return frame.with_columns(
        pl.col(timestamp_column)
        .cast(pl.Utf8)
        .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
        .alias("timestamp")
    )


def _dedupe_hill_by_keys(frame: pl.DataFrame, key_columns: list[str]) -> pl.DataFrame:
    return frame.unique(subset=key_columns, keep="first", maintain_order=True)


def _max_hill_timestamp(paths: list[Path], timestamp_column: str = "TimeStamp") -> Any:
    maximum = None
    for path in paths:
        frame = pl.read_parquet(path, columns=[timestamp_column]).with_columns(
            pl.col(timestamp_column)
            .cast(pl.Utf8)
            .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
            .alias("timestamp")
        )
        value = frame["timestamp"].max()
        if value is not None and (maximum is None or value > maximum):
            maximum = value
    return maximum


def _standardize_hill_shared_table(
    *,
    frame: pl.DataFrame,
    dataset_id: str,
    metadata: pl.DataFrame,
    table_name: str,
    prefix: str,
    key_columns: tuple[str, ...],
) -> pl.DataFrame:
    if frame.is_empty():
        if key_columns == ("TimeStamp",):
            return pl.DataFrame(schema={"dataset": pl.String, "timestamp": pl.Datetime})
        return pl.DataFrame(schema={"dataset": pl.String, "turbine_id": pl.String, "timestamp": pl.Datetime})

    frame = _hill_timestamp_frame(frame, "TimeStamp")
    join_keys: list[str]
    if key_columns == ("TimeStamp",):
        join_keys = ["timestamp"]
        frame = _dedupe_hill_by_keys(frame, join_keys)
    else:
        frame = frame.with_columns(pl.col("StationId").cast(pl.String).alias("StationId"))
        frame = _dedupe_hill_by_keys(frame, ["timestamp", "StationId"])
        frame = frame.join(metadata, on="StationId", how="left")
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
    }
    payload_columns = [column for column in frame.columns if column not in excluded]
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
    standardized = frame.select(select_expressions)
    feature_columns = [
        column
        for column in standardized.columns
        if column not in {"dataset", "turbine_id", "timestamp"}
    ]
    group_by_keys = ["dataset", *join_keys]
    return (
        standardized.group_by(group_by_keys, maintain_order=True)
        .agg([pl.col(column).drop_nulls().first().alias(column) for column in feature_columns])
        .sort(group_by_keys)
    )


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
        .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
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


def _join_hill_extra_frames(base: pl.DataFrame, cache_paths) -> pl.DataFrame:
    joined = base
    for group_name in (
        "farm_grid",
        "farm_grid_sci",
        "turbine_shutdown_duration",
        "turbine_count",
        "turbine_digi_in",
        "turbine_digi_out",
        "turbine_intern",
        "turbine_press",
        "turbine_temp",
    ):
        path = cache_paths.silver_shared_ts_path(group_name)
        if not path.exists():
            continue
        frame = pl.read_parquet(path)
        join_keys = ["dataset", "timestamp"]
        if "turbine_id" in frame.columns:
            join_keys.insert(1, "turbine_id")
        joined = joined.join(frame, on=join_keys, how="left")

    for group_name in ("alarmlog", "aeroup"):
        path = cache_paths.silver_event_features_path(group_name)
        if path.exists():
            joined = joined.join(
                pl.read_parquet(path),
                on=["dataset", "turbine_id", "timestamp"],
                how="left",
            )

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


def _augment_hill_batch(
    base_batch: pl.DataFrame,
    cache_paths,
    shared_frames: dict[str, pl.DataFrame],
    expected_extra_schema: dict[str, pl.DataType],
) -> pl.DataFrame:
    if base_batch.is_empty():
        return base_batch
    turbine_id = base_batch["turbine_id"][0]
    batch_start = base_batch["timestamp"].min()
    batch_end = base_batch["timestamp"].max()
    joined = base_batch

    for frame in shared_frames.values():
        joined = joined.join(frame, on=["dataset", "timestamp"], how="left")

    for group_name in _HILL_TURBINE_SHARED_GROUPS:
        frame = _load_turbine_time_slice(
            cache_paths.silver_shared_ts_path(group_name),
            turbine_id,
            batch_start,
            batch_end,
        )
        if not frame.is_empty():
            joined = joined.join(frame, on=["dataset", "turbine_id", "timestamp"], how="left")

    for group_name in _HILL_EVENT_FEATURE_GROUPS:
        frame = _load_turbine_time_slice(
            cache_paths.silver_event_features_path(group_name),
            turbine_id,
            batch_start,
            batch_end,
        )
        if not frame.is_empty():
            joined = joined.join(frame, on=["dataset", "turbine_id", "timestamp"], how="left")

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


def _write_hill_gold_with_extras(
    base_chunks,
    cache_paths,
    output_path: Path,
    spec,
    layout: str,
    batch_rows: int = 2_000,
) -> tuple[QualityReportAccumulator, dict[str, Any]]:
    ensure_directory(output_path.parent)
    temp_output_path = output_path.with_suffix(".tmp.parquet") if layout == "farm" else output_path
    shared_frames = {
        group_name: pl.read_parquet(cache_paths.silver_shared_ts_path(group_name))
        for group_name in _HILL_BROADCAST_SHARED_GROUPS
        if cache_paths.silver_shared_ts_path(group_name).exists()
    }
    expected_extra_schema: dict[str, pl.DataType] = {}
    for path in [cache_paths.silver_shared_ts_path(group_name) for group_name in _HILL_TURBINE_SHARED_GROUPS] + [
        cache_paths.silver_event_features_path(group_name) for group_name in _HILL_EVENT_FEATURE_GROUPS
    ]:
        if not path.exists():
            continue
        schema = pl.scan_parquet(path).collect_schema()
        for column, dtype in schema.items():
            if column not in {"dataset", "turbine_id", "timestamp"}:
                expected_extra_schema[column] = dtype
    writer = ParquetChunkWriter(temp_output_path, row_group_size=1_000)
    accumulator = QualityReportAccumulator()
    for base_chunk in base_chunks:
        if base_chunk.is_empty():
            continue
        update_quality_report_accumulator(accumulator, base_chunk, spec)
        for offset in range(0, base_chunk.height, batch_rows):
            batch = base_chunk.slice(offset, batch_rows)
            writer.write_frame(_augment_hill_batch(batch, cache_paths, shared_frames, expected_extra_schema))
    writer.close()
    if layout == "farm":
        coverage_summary = _finalize_hill_farm_temp(temp_output_path, output_path, spec, batch_rows=batch_rows * 10)
        return accumulator, coverage_summary

    return accumulator, build_coverage_summary(pl.read_parquet(output_path), spec)


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
        dataset_end = _max_hill_timestamp(
            sorted(self.cache_paths.silver_dir.rglob("tblSCTurGrid_*.parquet"))
        )
        for table_name, group_name, prefix, key_columns in _HILL_SHARED_TABLE_SPECS:
            standardized = _standardize_hill_shared_table(
                frame=_concat_hill_tables(sorted(self.cache_paths.silver_dir.rglob(f"{table_name}_*.parquet"))),
                dataset_id=self.spec.dataset_id,
                metadata=metadata,
                table_name=table_name,
                prefix=prefix,
                key_columns=key_columns,
            )
            standardized.write_parquet(self.cache_paths.silver_shared_ts_path(group_name))

        shutdown_path = self.cache_paths.silver_dir / "ShutdownDuration.parquet"
        _standardize_shutdown_duration(
            path=shutdown_path,
            dataset_id=self.spec.dataset_id,
            metadata=metadata,
        ).write_parquet(self.cache_paths.silver_shared_ts_path("turbine_shutdown_duration"))

        alarmlog = _standardize_alarmlog(
            frame=_concat_hill_tables(sorted(self.cache_paths.silver_dir.rglob("tblAlarmLog_*.parquet"))),
            dataset_id=self.spec.dataset_id,
            metadata=metadata,
        )
        alarmlog.write_parquet(self.cache_paths.silver_events_dir / "alarmlog.parquet")
        featureize_interval_events(
            events=alarmlog,
            resolution_minutes=self.spec.resolution_minutes,
            key_columns=("dataset", "turbine_id"),
            start_column="event_start",
            end_column="event_end",
            base_prefix="alarm",
            categorical_prefixes=(("alarm_code", "alarm_code"),),
        ).write_parquet(self.cache_paths.silver_event_features_path("alarmlog"))

        aeroup = _standardize_aeroup(
            self.cache_paths.silver_dir / "Hill_of_Towie_AeroUp_install_dates.parquet",
            self.spec.dataset_id,
        )
        aeroup.write_parquet(self.cache_paths.silver_interventions_path("aeroup"))
        _featureize_aeroup_interventions(
            aeroup,
            self.spec.resolution_minutes,
            dataset_end=dataset_end,
        ).write_parquet(
            self.cache_paths.silver_event_features_path("aeroup")
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
            self.cache_paths.silver_dir,
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
        metadata_path = self.cache_paths.silver_dir / "Hill_of_Towie_turbine_metadata.parquet"
        metadata = pl.read_parquet(metadata_path).select(
            pl.col("Station ID").cast(pl.String).alias("StationId"),
            pl.col("Turbine Name").cast(pl.String).alias("turbine_id"),
        )

        table_frames: list[pl.DataFrame] = []
        duplicate_audit_frames: list[pl.DataFrame] = []
        duplicate_conflict_keys: list[pl.DataFrame] = []
        for table_name in _DEFAULT_TABLES:
            paths = sorted(self.cache_paths.silver_dir.rglob(f"{table_name}_*.parquet"))
            if not paths:
                continue
            # Some monthly exports omit a small subset of columns. Align on the
            # union so the missing fields remain explicit instead of being lost.
            raw_frame = pl.concat(
                [pl.read_parquet(path) for path in paths],
                how="diagonal_relaxed",
            )
            frame, duplicate_audit, conflict_keys = _audit_hill_duplicate_keys(raw_frame, table_name)
            table_frames.append(frame)
            duplicate_audit_frames.append(duplicate_audit)
            if not conflict_keys.is_empty():
                duplicate_conflict_keys.append(conflict_keys)

        if not table_frames:
            raise ValueError("No Hill of Towie default feature tables were found.")

        duplicate_audit = (
            pl.concat(duplicate_audit_frames, how="vertical")
            if duplicate_audit_frames
            else pl.DataFrame(schema=_DUPLICATE_AUDIT_SCHEMA)
        )
        duplicate_audit.write_parquet(self.cache_paths.hill_duplicate_audit_path)
        conflict_keys = (
            pl.concat(duplicate_conflict_keys, how="vertical").unique()
            if duplicate_conflict_keys
            else pl.DataFrame(schema={"timestamp": pl.Datetime, "StationId": pl.String})
        )

        key_frame = pl.concat(
            [frame.select(["timestamp", "StationId"]) for frame in table_frames],
            how="vertical",
        ).unique()

        joined = key_frame
        for frame in table_frames:
            payload_columns = [
                column
                for column in frame.columns
                if column not in {"TimeStamp", "timestamp", "StationId"}
            ]
            joined = joined.join(
                frame.select(["timestamp", "StationId", *payload_columns]),
                on=["timestamp", "StationId"],
                how="left",
            )

        joined = joined.join(metadata, on="StationId", how="left")
        if conflict_keys.is_empty():
            joined = joined.with_columns(pl.lit(False).alias("__duplicate_conflict"))
        else:
            joined = joined.join(
                conflict_keys.with_columns(pl.lit(True).alias("__duplicate_conflict")),
                on=["timestamp", "StationId"],
                how="left",
            ).with_columns(pl.col("__duplicate_conflict").fill_null(False))
        joined = joined.with_columns(
            pl.lit(self.spec.dataset_id).alias("dataset"),
            pl.col(self.spec.target_column).cast(pl.Float64, strict=False).alias("target_kw"),
            pl.lit(True).alias("__row_present"),
            pl.when(pl.col("__duplicate_conflict"))
            .then(pl.lit("duplicate_conflict_resolved"))
            .otherwise(pl.lit(""))
            .alias("__base_quality_flags"),
        )

        feature_columns = [
            column
            for column in joined.columns
            if column
            not in {
                "dataset",
                "StationId",
                "turbine_id",
                "timestamp",
                    self.spec.target_column,
                    "target_kw",
                    "__row_present",
                    "__base_quality_flags",
                    "__duplicate_conflict",
                }
        ]
        gold_input = joined.select(
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
        gold_input = gold_input.with_columns(
            [pl.col(column).cast(pl.Float64, strict=False) for column in feature_columns]
        )
        report_extra = {
            "quality_profile": resolved_quality_profile,
            "layout": resolved_layout,
            "feature_set": resolved_feature_set,
            "duplicate_key_audit_count": duplicate_audit.height,
            "duplicate_conflict_key_count": duplicate_audit.filter(pl.col("is_conflicting")).height,
        }
        if resolved_layout == "farm" and temp_series_path is not None and temp_series_path.exists() and not series_path.exists():
            coverage_summary = _finalize_hill_farm_temp(temp_series_path, series_path, self.spec)
            report_frame = load_quality_report_frame(series_path)
            report = build_quality_report(
                report_frame,
                manifest_payload,
                self.spec,
                extra={**report_extra, **coverage_summary},
            )
            write_quality_report(quality_path, report)
            return series_path
        if series_path.exists() and not quality_path.exists():
            report_frame = load_quality_report_frame(series_path)
            report = build_quality_report(
                report_frame,
                manifest_payload,
                self.spec,
                extra={**report_extra, **build_coverage_summary(report_frame, self.spec)},
            )
            write_quality_report(quality_path, report)
            return series_path
        quality_accumulator, coverage_summary = _write_hill_gold_with_extras(
            iter_reindexed_regular_series(gold_input, self.spec, layout=resolved_layout),
            self.cache_paths,
            series_path,
            self.spec,
            layout=resolved_layout,
        )

        report = finalize_quality_report(
            quality_accumulator,
            manifest_payload,
            self.spec,
            extra={**report_extra, **coverage_summary},
        )
        write_quality_report(quality_path, report)
        return series_path

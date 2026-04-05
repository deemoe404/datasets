from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import re
from typing import Any

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from ..models import DatasetSpec, ResolvedTaskSpec
from ..utils import ensure_directory, join_flags, write_json

TURBINE_STATIC_SCHEMA: dict[str, pl.DataType] = {
    "dataset": pl.String,
    "turbine_id": pl.String,
    "source_turbine_key": pl.String,
    "latitude": pl.Float64,
    "longitude": pl.Float64,
    "coord_x": pl.Float64,
    "coord_y": pl.Float64,
    "coord_kind": pl.String,
    "coord_crs": pl.String,
    "elevation_m": pl.Float64,
    "rated_power_kw": pl.Float64,
    "hub_height_m": pl.Float64,
    "rotor_diameter_m": pl.Float64,
    "manufacturer": pl.String,
    "model": pl.String,
    "country": pl.String,
    "commercial_operation_date": pl.String,
    "spatial_source": pl.String,
}

_SANITIZE_PATTERN = re.compile(r"[^a-z0-9]+")


def ensure_turbine_static_schema(df: pl.DataFrame) -> pl.DataFrame:
    missing = [
        pl.lit(None).cast(dtype).alias(column)
        for column, dtype in TURBINE_STATIC_SCHEMA.items()
        if column not in df.columns
    ]
    if missing:
        df = df.with_columns(missing)
    return df.select(
        [pl.col(column).cast(dtype, strict=False).alias(column) for column, dtype in TURBINE_STATIC_SCHEMA.items()]
    )


def sanitize_feature_name(value: str) -> str:
    text = value.strip().lower()
    text = _SANITIZE_PATTERN.sub("_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def build_feature_name_mapping(values: list[str], prefix: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    used: dict[str, str] = {}
    for value in sorted({item.strip() for item in values if item and item.strip()}):
        base_name = f"{prefix}__{sanitize_feature_name(value)}"
        name = base_name
        suffix = 2
        while name in used and used[name] != value:
            name = f"{base_name}_{suffix}"
            suffix += 1
        used[name] = value
        mapping[value] = name
    return mapping


def aligned_floor_timestamp(value: datetime, resolution_minutes: int) -> datetime:
    epoch = datetime(1970, 1, 1, tzinfo=value.tzinfo)
    step_seconds = resolution_minutes * 60
    total_seconds = int((value - epoch).total_seconds())
    floored = (total_seconds // step_seconds) * step_seconds
    return epoch + timedelta(seconds=floored)


def aligned_ceil_timestamp(value: datetime, resolution_minutes: int) -> datetime:
    floored = aligned_floor_timestamp(value, resolution_minutes)
    if floored == value:
        return floored
    return floored + timedelta(minutes=resolution_minutes)


def featureize_interval_events(
    *,
    events: pl.DataFrame,
    resolution_minutes: int,
    key_columns: tuple[str, ...],
    start_column: str,
    end_column: str,
    base_prefix: str,
    categorical_prefixes: tuple[tuple[str, str], ...] = (),
    status_aliases: dict[str, str] | None = None,
) -> pl.DataFrame:
    core_columns = [
        f"{base_prefix}_any_active",
        f"{base_prefix}_active_count",
        f"{base_prefix}_total_overlap_seconds",
    ]
    if events.is_empty():
        return pl.DataFrame(
            schema={
                **{column: pl.String for column in key_columns},
                "timestamp": pl.Datetime,
                core_columns[0]: pl.Boolean,
                core_columns[1]: pl.Int32,
                core_columns[2]: pl.Int32,
            }
        )

    normalized = events
    value_mappings: dict[str, dict[str, str]] = {}
    for source_column, prefix in categorical_prefixes:
        if source_column not in normalized.columns:
            continue
        normalized = normalized.with_columns(
            pl.col(source_column)
            .cast(pl.String)
            .str.strip_chars()
            .replace({"": None})
            .alias(source_column)
        )
        values = normalized.select(pl.col(source_column).drop_nulls().unique()).to_series().to_list()
        value_mappings[source_column] = build_feature_name_mapping(values, prefix)

    alias_mapping = {
        sanitize_feature_name(source): target for source, target in (status_aliases or {}).items()
    }
    interval = timedelta(minutes=resolution_minutes)
    output_frames: list[pl.DataFrame] = []

    interval_seconds = resolution_minutes * 60

    for group in normalized.partition_by(list(key_columns), maintain_order=True):
        group = group.sort(start_column)
        rows = group.to_dicts()
        if not rows:
            continue

        event_bounds = [
            row[end_column] if isinstance(row.get(end_column), datetime) else row[start_column]
            for row in rows
        ]
        fallback_end = aligned_ceil_timestamp(max(event_bounds), resolution_minutes) + interval
        prepared_rows: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            start = row[start_column]
            raw_end = row.get(end_column)
            next_start = rows[index + 1][start_column] if index + 1 < len(rows) else None
            if isinstance(raw_end, datetime) and raw_end > start:
                effective_end = raw_end
            elif isinstance(next_start, datetime) and next_start > start:
                effective_end = next_start
            else:
                effective_end = fallback_end
            prepared_rows.append({**row, "__effective_end": effective_end})

        grid_start = aligned_floor_timestamp(prepared_rows[0][start_column], resolution_minutes)
        grid_end = aligned_ceil_timestamp(
            max(row["__effective_end"] for row in prepared_rows),
            resolution_minutes,
        )
        timestamps = pl.datetime_range(
            start=grid_start,
            end=grid_end,
            interval=f"{resolution_minutes}m",
            eager=True,
        )
        timestamp_list = timestamps.to_list()
        length = len(timestamp_list)
        active_diff = np.zeros(length + 1, dtype=np.int32)
        overlap_full_diff = np.zeros(length + 1, dtype=np.int64)
        overlap_edge = np.zeros(length, dtype=np.int64)
        categorical_diffs: dict[str, np.ndarray] = {}
        alias_diffs: dict[str, np.ndarray] = {}

        for row in prepared_rows:
            start = row[start_column]
            end = row["__effective_end"]
            first_idx = bisect_right(timestamp_list, start)
            last_exclusive = bisect_left(timestamp_list, end + interval)
            if first_idx >= last_exclusive:
                continue
            active_diff[first_idx] += 1
            active_diff[last_exclusive] -= 1
            normalized_status = None
            if "status" in row and row["status"] is not None:
                normalized_status = sanitize_feature_name(str(row["status"]))
            if first_idx == last_exclusive - 1:
                bucket_end = timestamp_list[first_idx]
                bucket_start = bucket_end - interval
                overlap_start = max(start, bucket_start)
                overlap_end = min(end, bucket_end)
                overlap_seconds = int((overlap_end - overlap_start).total_seconds())
                if overlap_seconds > 0:
                    overlap_edge[first_idx] += overlap_seconds
            else:
                first_bucket_end = timestamp_list[first_idx]
                first_overlap = int((first_bucket_end - start).total_seconds())
                if first_overlap > 0:
                    overlap_edge[first_idx] += first_overlap

                last_idx = last_exclusive - 1
                last_bucket_end = timestamp_list[last_idx]
                last_bucket_start = last_bucket_end - interval
                last_overlap = int((end - max(start, last_bucket_start)).total_seconds())
                if last_overlap > 0:
                    overlap_edge[last_idx] += last_overlap

                full_start = first_idx + 1
                full_end_exclusive = last_idx
                if full_start < full_end_exclusive:
                    overlap_full_diff[full_start] += interval_seconds
                    overlap_full_diff[full_end_exclusive] -= interval_seconds

            if normalized_status and normalized_status in alias_mapping:
                output_name = alias_mapping[normalized_status]
                alias_diff = alias_diffs.setdefault(output_name, np.zeros(length + 1, dtype=np.int32))
                alias_diff[first_idx] += 1
                alias_diff[last_exclusive] -= 1

            for source_column, mapping in value_mappings.items():
                raw_value = row.get(source_column)
                if raw_value is None:
                    continue
                output_name = mapping.get(str(raw_value).strip())
                if output_name is None:
                    continue
                categorical_diff = categorical_diffs.setdefault(output_name, np.zeros(length + 1, dtype=np.int32))
                categorical_diff[first_idx] += 1
                categorical_diff[last_exclusive] -= 1

        active_count = np.cumsum(active_diff[:-1], dtype=np.int64).astype(np.int32, copy=False)
        any_active = active_count > 0
        total_overlap_seconds = (
            overlap_edge + np.cumsum(overlap_full_diff[:-1], dtype=np.int64)
        ).astype(np.int32, copy=False)
        alias_arrays = {
            output_name: np.cumsum(diff[:-1], dtype=np.int64) > 0
            for output_name, diff in alias_diffs.items()
        }
        categorical_arrays = {
            output_name: np.cumsum(diff[:-1], dtype=np.int64) > 0
            for output_name, diff in categorical_diffs.items()
        }

        group_payload = {
            **{
                column: [prepared_rows[0][column]] * length
                for column in key_columns
            },
            "timestamp": timestamps,
            f"{base_prefix}_any_active": any_active.tolist(),
            f"{base_prefix}_active_count": active_count.tolist(),
            f"{base_prefix}_total_overlap_seconds": total_overlap_seconds.tolist(),
            **{column: values.tolist() for column, values in alias_arrays.items()},
            **{column: values.tolist() for column, values in categorical_arrays.items()},
        }
        output_frames.append(pl.DataFrame(group_payload))

    if not output_frames:
        return pl.DataFrame(
            schema={
                **{column: pl.String for column in key_columns},
                "timestamp": pl.Datetime,
                core_columns[0]: pl.Boolean,
                core_columns[1]: pl.Int32,
                core_columns[2]: pl.Int32,
            }
        )
    result = pl.concat(output_frames, how="diagonal_relaxed")
    boolean_columns = [
        column
        for column in [
            *alias_mapping.values(),
            *[output_name for mapping in value_mappings.values() for output_name in mapping.values()],
        ]
        if column in result.columns
    ]
    if boolean_columns:
        result = result.with_columns([pl.col(column).fill_null(False).alias(column) for column in boolean_columns])
    return result.sort([*key_columns, "timestamp"])


@dataclass
class ParquetChunkWriter:
    path: Path
    schema: pa.Schema | None = None
    row_group_size: int | None = None
    _writer: pq.ParquetWriter | None = None

    def write_rows(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        ensure_directory(self.path.parent)
        table = pa.Table.from_pylist(rows, schema=self.schema)
        if self.schema is not None and table.schema != self.schema:
            table = table.cast(self.schema)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.path, table.schema)
            if self.schema is None:
                self.schema = table.schema
        self._writer.write_table(table, row_group_size=self.row_group_size)

    def write_frame(self, frame: pl.DataFrame) -> None:
        if frame.is_empty():
            return
        ensure_directory(self.path.parent)
        if self.schema is not None:
            missing_columns = [
                pl.lit(None).alias(field.name)
                for field in self.schema
                if field.name not in frame.columns
            ]
            if missing_columns:
                frame = frame.with_columns(missing_columns)
            frame = frame.select(self.schema.names)
        table = frame.to_arrow()
        if self.schema is not None and table.schema != self.schema:
            table = table.cast(self.schema)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.path, table.schema)
            if self.schema is None:
                self.schema = table.schema
        self._writer.write_table(table, row_group_size=self.row_group_size)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            return
        if self.schema is not None:
            ensure_directory(self.path.parent)
            empty = pa.Table.from_arrays(
                [pa.array([], type=field.type) for field in self.schema],
                schema=self.schema,
            )
            pq.write_table(empty, self.path)


@dataclass
class QualityReportAccumulator:
    total_rows: int = 0
    observed_rows: int = 0
    missing_row_count: int = 0
    target_missing_count: int = 0
    duplicate_key_count: int = 0
    interval_distribution: Counter[int] = field(default_factory=Counter)
    long_gaps: list[dict[str, Any]] = field(default_factory=list)


def cast_numeric_columns(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    expressions = []
    for column in columns:
        expressions.append(
            pl.col(column)
            .cast(pl.Utf8)
            .str.strip_chars()
            .replace({"": None, "NaN": None, "-": None})
            .cast(pl.Float64, strict=False)
            .fill_nan(None)
            .alias(column)
        )
    return df.with_columns(expressions)


def iter_reindexed_regular_series(df: pl.DataFrame, spec: DatasetSpec):
    if df.is_empty():
        return

    feature_columns = [
        column
        for column in df.columns
        if column
        not in {
            "dataset",
            "turbine_id",
            "timestamp",
            "target_kw",
            "__row_present",
            "__base_quality_flags",
        }
    ]
    boolean_feature_columns = [
        column for column in feature_columns if df.schema.get(column) == pl.Boolean
    ]
    interval = f"{spec.resolution_minutes}m"
    for turbine_frame in df.partition_by("turbine_id", maintain_order=True):
        turbine_frame = turbine_frame.sort("timestamp")
        turbine_id = turbine_frame["turbine_id"][0]
        grid = pl.DataFrame(
            {
                "timestamp": pl.datetime_range(
                    turbine_frame["timestamp"].min(),
                    turbine_frame["timestamp"].max(),
                    interval=interval,
                    eager=True,
                )
            }
        ).with_columns(pl.lit(turbine_id).alias("turbine_id"))

        joined = grid.join(turbine_frame, on=["turbine_id", "timestamp"], how="left")
        joined = joined.with_columns(
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
        )
        if boolean_feature_columns:
            joined = joined.with_columns(
                [pl.col(column).fill_null(False).alias(column) for column in boolean_feature_columns]
            )
        yield joined.select(
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

def reindex_regular_series(df: pl.DataFrame, spec: DatasetSpec) -> pl.DataFrame:
    output_parts = list(iter_reindexed_regular_series(df, spec))
    if not output_parts:
        return pl.DataFrame(
            schema={
                "dataset": pl.String,
                "turbine_id": pl.String,
                "timestamp": pl.Datetime,
                "target_kw": pl.Float64,
                "is_observed": pl.Boolean,
                "quality_flags": pl.String,
            }
        )
    return pl.concat(output_parts, how="vertical").sort(["dataset", "turbine_id", "timestamp"])


def update_quality_report_accumulator(
    accumulator: QualityReportAccumulator,
    df: pl.DataFrame,
    spec: DatasetSpec,
) -> QualityReportAccumulator:
    if df.is_empty():
        return accumulator

    observed = df.filter(pl.col("is_observed"))
    accumulator.total_rows += df.height
    accumulator.observed_rows += observed.height
    accumulator.missing_row_count += df.filter(~pl.col("is_observed")).height
    accumulator.target_missing_count += df.filter(pl.col("target_kw").is_null()).height
    accumulator.duplicate_key_count += (
        df.group_by(["dataset", "turbine_id", "timestamp"])
        .len()
        .filter(pl.col("len") > 1)
        .height
    )

    threshold = spec.resolution_minutes
    for turbine_frame in observed.partition_by("turbine_id", maintain_order=True):
        turbine_id = turbine_frame["turbine_id"][0]
        timestamps = turbine_frame.sort("timestamp")["timestamp"].to_list()
        for previous, current in zip(timestamps, timestamps[1:]):
            diff_minutes = int((current - previous).total_seconds() // 60)
            accumulator.interval_distribution[diff_minutes] += 1
            if diff_minutes > threshold:
                accumulator.long_gaps.append(
                    {
                        "turbine_id": turbine_id,
                        "gap_minutes": diff_minutes,
                        "gap_start": previous.isoformat(),
                        "gap_end": current.isoformat(),
                    }
                )
    return accumulator


def finalize_quality_report(
    accumulator: QualityReportAccumulator,
    manifest_payload: dict[str, Any],
    spec: DatasetSpec,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = {
        "dataset_id": spec.dataset_id,
        "resolution_minutes": spec.resolution_minutes,
        "source_files": [item["relative_path"] for item in manifest_payload["files"]],
        "total_rows": accumulator.total_rows,
        "observed_rows": accumulator.observed_rows,
        "missing_row_count": accumulator.missing_row_count,
        "target_missing_count": accumulator.target_missing_count,
        "duplicate_key_count": accumulator.duplicate_key_count,
        "interval_distribution_minutes": {
            str(key): value for key, value in sorted(accumulator.interval_distribution.items())
        },
        "long_gaps": accumulator.long_gaps,
    }
    if extra:
        report.update(extra)
    return report


def build_quality_report(
    df: pl.DataFrame,
    manifest_payload: dict[str, Any],
    spec: DatasetSpec,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    accumulator = update_quality_report_accumulator(QualityReportAccumulator(), df, spec)
    return finalize_quality_report(accumulator, manifest_payload, spec, extra)


def write_quality_report(path: Path, payload: dict[str, Any]) -> Path:
    return write_json(path, payload)


def build_window_index(
    df: pl.DataFrame,
    task: ResolvedTaskSpec,
    output_path: Path,
    report_path: Path,
    quality_profile: str,
) -> Path:
    schema = pa.schema(
        [
            ("dataset", pa.string()),
            ("turbine_id", pa.string()),
            ("input_start_ts", pa.timestamp("us")),
            ("input_end_ts", pa.timestamp("us")),
            ("output_start_ts", pa.timestamp("us")),
            ("output_end_ts", pa.timestamp("us")),
            ("input_steps_expected", pa.int64()),
            ("output_steps_expected", pa.int64()),
            ("input_steps_observed", pa.int64()),
            ("output_steps_observed", pa.int64()),
            ("input_masked_steps", pa.int64()),
            ("output_masked_steps", pa.int64()),
            ("input_unknown_steps", pa.int64()),
            ("output_unknown_steps", pa.int64()),
            ("input_abnormal_steps", pa.int64()),
            ("output_abnormal_steps", pa.int64()),
            ("is_complete_input", pa.bool_()),
            ("is_complete_output", pa.bool_()),
            ("quality_flags", pa.string()),
        ]
    )
    writer = ParquetChunkWriter(output_path, schema=schema)
    counts = Counter()
    buffer: list[dict[str, Any]] = []

    for turbine_frame in df.partition_by("turbine_id", maintain_order=True):
        turbine_frame = turbine_frame.sort("timestamp")
        dataset_id = turbine_frame["dataset"][0]
        turbine_id = turbine_frame["turbine_id"][0]
        timestamps = turbine_frame["timestamp"].to_list()
        input_observed = turbine_frame["is_observed"].to_list()
        output_observed = turbine_frame["target_kw"].is_not_null().to_list()
        row_quality_flags = turbine_frame["quality_flags"].to_list()
        masked_values = (
            turbine_frame["sdwpf_is_masked"].fill_null(False).to_list()
            if "sdwpf_is_masked" in turbine_frame.columns
            else [False] * len(timestamps)
        )
        unknown_values = (
            turbine_frame["sdwpf_is_unknown"].fill_null(False).to_list()
            if "sdwpf_is_unknown" in turbine_frame.columns
            else [False] * len(timestamps)
        )
        abnormal_values = (
            turbine_frame["sdwpf_is_abnormal"].fill_null(False).to_list()
            if "sdwpf_is_abnormal" in turbine_frame.columns
            else [False] * len(timestamps)
        )

        last_anchor = len(timestamps) - task.forecast_steps - 1
        if last_anchor < task.history_steps - 1:
            continue

        for anchor in range(task.history_steps - 1, last_anchor + 1, task.stride_steps):
            input_start_idx = anchor - task.history_steps + 1
            output_end_idx = anchor + task.forecast_steps
            input_count = sum(bool(value) for value in input_observed[input_start_idx : anchor + 1])
            output_count = sum(bool(value) for value in output_observed[anchor + 1 : output_end_idx + 1])
            input_masked_count = sum(bool(value) for value in masked_values[input_start_idx : anchor + 1])
            output_masked_count = sum(bool(value) for value in masked_values[anchor + 1 : output_end_idx + 1])
            input_unknown_count = sum(bool(value) for value in unknown_values[input_start_idx : anchor + 1])
            output_unknown_count = sum(bool(value) for value in unknown_values[anchor + 1 : output_end_idx + 1])
            input_abnormal_count = sum(bool(value) for value in abnormal_values[input_start_idx : anchor + 1])
            output_abnormal_count = sum(bool(value) for value in abnormal_values[anchor + 1 : output_end_idx + 1])
            complete_input = input_count == task.history_steps
            complete_output = output_count == task.forecast_steps

            flags = []
            if not complete_input:
                flags.append("partial_input")
            if not complete_output:
                flags.append("partial_output")
            if input_masked_count > 0:
                flags.append("masked_input")
            if output_masked_count > 0:
                flags.append("masked_output")
            if any(row_quality_flags[input_start_idx : output_end_idx + 1]):
                flags.append("row_quality_issues")

            buffer.append(
                {
                    "dataset": dataset_id,
                    "turbine_id": turbine_id,
                    "input_start_ts": timestamps[input_start_idx],
                    "input_end_ts": timestamps[anchor],
                    "output_start_ts": timestamps[anchor + 1],
                    "output_end_ts": timestamps[output_end_idx],
                    "input_steps_expected": task.history_steps,
                    "output_steps_expected": task.forecast_steps,
                    "input_steps_observed": input_count,
                    "output_steps_observed": output_count,
                    "input_masked_steps": input_masked_count,
                    "output_masked_steps": output_masked_count,
                    "input_unknown_steps": input_unknown_count,
                    "output_unknown_steps": output_unknown_count,
                    "input_abnormal_steps": input_abnormal_count,
                    "output_abnormal_steps": output_abnormal_count,
                    "is_complete_input": complete_input,
                    "is_complete_output": complete_output,
                    "quality_flags": join_flags(*flags),
                }
            )
            counts["window_count"] += 1
            if complete_input:
                counts["complete_input_windows"] += 1
            if complete_output:
                counts["complete_output_windows"] += 1
            if input_masked_count == 0:
                counts["unmasked_input_windows"] += 1
            else:
                counts["masked_input_windows"] += 1
            if output_masked_count == 0:
                counts["unmasked_output_windows"] += 1
            else:
                counts["masked_output_windows"] += 1
            if complete_input and complete_output:
                counts["fully_complete_windows"] += 1
            if complete_output and output_masked_count == 0:
                counts["fully_complete_and_unmasked_output_windows"] += 1
            if complete_input and complete_output and input_masked_count == 0 and output_masked_count == 0:
                counts["fully_complete_and_unmasked_all_windows"] += 1

            if len(buffer) >= 50_000:
                writer.write_rows(buffer)
                buffer = []

    writer.write_rows(buffer)
    writer.close()
    write_json(report_path, {"quality_profile": quality_profile, "task": task.to_dict(), **counts})
    return output_path

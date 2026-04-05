from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

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


@dataclass
class ParquetChunkWriter:
    path: Path
    schema: pa.Schema | None = None
    _writer: pq.ParquetWriter | None = None

    def write_rows(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        ensure_directory(self.path.parent)
        table = pa.Table.from_pylist(rows, schema=self.schema)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.path, table.schema)
        self._writer.write_table(table)

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


def reindex_regular_series(df: pl.DataFrame, spec: DatasetSpec) -> pl.DataFrame:
    if df.is_empty():
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
    output_parts: list[pl.DataFrame] = []

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
        output_parts.append(
            joined.select(
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
        )

    return pl.concat(output_parts, how="vertical").sort(["dataset", "turbine_id", "timestamp"])


def build_quality_report(
    df: pl.DataFrame,
    manifest_payload: dict[str, Any],
    spec: DatasetSpec,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    observed = df.filter(pl.col("is_observed"))
    duplicate_key_count = (
        df.group_by(["dataset", "turbine_id", "timestamp"])
        .len()
        .filter(pl.col("len") > 1)
        .height
    )

    interval_distribution: Counter[int] = Counter()
    long_gaps: list[dict[str, Any]] = []
    threshold = spec.resolution_minutes

    for turbine_frame in observed.partition_by("turbine_id", maintain_order=True):
        turbine_id = turbine_frame["turbine_id"][0]
        timestamps = turbine_frame.sort("timestamp")["timestamp"].to_list()
        for previous, current in zip(timestamps, timestamps[1:]):
            diff_minutes = int((current - previous).total_seconds() // 60)
            interval_distribution[diff_minutes] += 1
            if diff_minutes > threshold:
                long_gaps.append(
                    {
                        "turbine_id": turbine_id,
                        "gap_minutes": diff_minutes,
                        "gap_start": previous.isoformat(),
                        "gap_end": current.isoformat(),
                    }
                )

    report = {
        "dataset_id": spec.dataset_id,
        "resolution_minutes": spec.resolution_minutes,
        "source_files": [item["relative_path"] for item in manifest_payload["files"]],
        "total_rows": df.height,
        "observed_rows": observed.height,
        "missing_row_count": df.filter(~pl.col("is_observed")).height,
        "target_missing_count": df.filter(pl.col("target_kw").is_null()).height,
        "duplicate_key_count": duplicate_key_count,
        "interval_distribution_minutes": {
            str(key): value for key, value in sorted(interval_distribution.items())
        },
        "long_gaps": long_gaps,
    }
    if extra:
        report.update(extra)
    return report


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

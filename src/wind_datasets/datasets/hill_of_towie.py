from __future__ import annotations

from pathlib import Path

import polars as pl

from ..utils import ensure_directory
from .base import BaseDatasetBuilder
from .common import build_quality_report, ensure_turbine_static_schema, reindex_regular_series, write_quality_report

_DEFAULT_TABLES = ("tblSCTurbine", "tblSCTurGrid", "tblSCTurFlag")
_DUPLICATE_AUDIT_SCHEMA = {
    "table_name": pl.String,
    "timestamp": pl.Datetime,
    "station_id": pl.String,
    "duplicate_count": pl.Int64,
    "is_conflicting": pl.Boolean,
    "conflicting_columns": pl.String,
}


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
        conflicting_columns: list[str] = []
        for column in payload_columns:
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
        audit_rows.append(
            {
                "table_name": table_name,
                "timestamp": group["timestamp"][0],
                "station_id": group["StationId"][0],
                "duplicate_count": group.height,
                "is_conflicting": bool(conflicting_columns),
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


class HillOfTowieDatasetBuilder(BaseDatasetBuilder):
    def build_silver(self) -> Path:
        self.ensure_manifest()
        ensure_directory(self.cache_paths.silver_dir)
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
        return self.cache_paths.silver_dir

    def build_gold_base(self, quality_profile: str | None = None) -> Path:
        resolved_quality_profile = self.resolve_quality_profile(quality_profile)
        if not self.cache_paths.silver_dir.exists():
            self.build_silver()

        manifest_payload = self.ensure_manifest()
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
        gold_base = reindex_regular_series(gold_input, self.spec)
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
                "duplicate_key_audit_count": duplicate_audit.height,
                "duplicate_conflict_key_count": duplicate_audit.filter(pl.col("is_conflicting")).height,
            },
        )
        write_quality_report(quality_path, report)
        return series_path

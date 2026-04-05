from __future__ import annotations

from pathlib import Path

import polars as pl

from ..utils import ensure_directory
from .base import BaseDatasetBuilder
from .common import build_quality_report, reindex_regular_series, write_quality_report

_DEFAULT_TABLES = ("tblSCTurbine", "tblSCTurGrid", "tblSCTurFlag")


def _read_csv_with_fallback(path):
    for encoding in ("utf8", "windows-1252", "utf8-lossy"):
        try:
            return pl.read_csv(path, encoding=encoding)
        except pl.exceptions.ComputeError:
            continue
    return pl.read_csv(path, encoding="utf8-lossy")


class HillOfTowieDatasetBuilder(BaseDatasetBuilder):
    def build_silver(self) -> Path:
        self.ensure_manifest()
        ensure_directory(self.cache_paths.silver_dir)
        for path in sorted(self.spec.source_root.rglob("*.csv")):
            relative = path.relative_to(self.spec.source_root)
            output_path = self.cache_paths.silver_dir / relative.with_suffix(".parquet")
            ensure_directory(output_path.parent)
            _read_csv_with_fallback(path).write_parquet(output_path)
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
        for table_name in _DEFAULT_TABLES:
            paths = sorted(self.cache_paths.silver_dir.rglob(f"{table_name}_*.parquet"))
            if not paths:
                continue
            # Some monthly exports omit a small subset of columns. Align on the
            # union so the missing fields remain explicit instead of being lost.
            frame = pl.concat(
                [pl.read_parquet(path) for path in paths],
                how="diagonal_relaxed",
            )
            frame = frame.with_columns(
                pl.col("TimeStamp")
                .cast(pl.Utf8)
                .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
                .alias("timestamp"),
                pl.col("StationId").cast(pl.String).alias("StationId"),
            )
            # Adjacent monthly exports repeat boundary rows. The duplicates are
            # byte-identical, so keeping the first preserves the underlying
            # signal without multiplying keys during the cross-table join.
            frame = frame.unique(subset=["timestamp", "StationId"], keep="first", maintain_order=True)
            table_frames.append(frame)

        if not table_frames:
            raise ValueError("No Hill of Towie default feature tables were found.")

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

        joined = joined.join(metadata, on="StationId", how="left").with_columns(
            pl.lit(self.spec.dataset_id).alias("dataset"),
            pl.col(self.spec.target_column).cast(pl.Float64, strict=False).alias("target_kw"),
            pl.lit(True).alias("__row_present"),
            pl.lit("").alias("__base_quality_flags"),
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
            extra={"quality_profile": resolved_quality_profile},
        )
        write_quality_report(quality_path, report)
        return series_path

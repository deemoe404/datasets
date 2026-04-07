from __future__ import annotations

from datetime import datetime
import gc
from pathlib import Path

import polars as pl

from ..utils import ensure_directory
from .base import BaseDatasetBuilder
from .common import (
    build_coverage_summary,
    build_quality_report,
    ensure_turbine_static_schema,
    load_quality_report_frame,
    reindex_regular_series,
    write_quality_report,
)

_SDWPF_SUPPORTED_QUALITY_PROFILES = {"default"}
_KDDCUP_CALENDAR_ANCHOR = datetime(2020, 5, 1)
_KDDCUP_MAIN_CSV = "sdwpf_245days_v1.csv"
_KDDCUP_MAIN_PARQUET = "sdwpf_245days_v1.parquet"
_KDDCUP_LOCATION_CSV = "sdwpf_baidukddcup2022_turb_location.csv"
_KDDCUP_LOCATION_PARQUET = "sdwpf_baidukddcup2022_turb_location.parquet"


def _assert_sdwpf_time_semantics_supported(manifest_payload: dict[str, object]) -> None:
    time_check = manifest_payload.get("time_semantics_check")
    if not isinstance(time_check, dict):
        raise ValueError("sdwpf_kddcup manifest is missing time_semantics_check; rebuild the manifest first.")
    if time_check.get("status") != "match_documented_245day_10min_grid":
        details = time_check.get(
            "details",
            "sdwpf_kddcup source timestamps are incompatible with the documented 245-day 10-minute grid.",
        )
        raise ValueError(f"Refusing to build sdwpf_kddcup gold/task cache. {details}")


def _clock_expr_for_dtype(dtype: pl.DataType) -> pl.Expr:
    if dtype == pl.Time:
        return pl.col("Tmstamp").cast(pl.Time, strict=False)
    return pl.col("Tmstamp").cast(pl.Utf8).str.strptime(pl.Time, format="%H:%M", strict=False)


def _timestamp_expr_for_dtypes(clock_dtype: pl.DataType) -> pl.Expr:
    clock_expr = _clock_expr_for_dtype(clock_dtype)
    return (
        pl.lit(_KDDCUP_CALENDAR_ANCHOR)
        + pl.duration(days=pl.col("Day").cast(pl.Int64, strict=False) - 1)
        + pl.duration(
            hours=clock_expr.dt.hour(),
            minutes=clock_expr.dt.minute(),
        )
    ).alias("timestamp")


class SDWPFKDDCupDatasetBuilder(BaseDatasetBuilder):
    def resolve_quality_profile(self, quality_profile: str | None = None) -> str:
        resolved = super().resolve_quality_profile(quality_profile)
        if resolved not in _SDWPF_SUPPORTED_QUALITY_PROFILES:
            supported = ", ".join(sorted(_SDWPF_SUPPORTED_QUALITY_PROFILES))
            raise ValueError(
                f"Unsupported sdwpf_kddcup quality profile {resolved!r}. Expected one of: {supported}."
            )
        return resolved

    def build_silver(self) -> Path:
        self.ensure_manifest()
        ensure_directory(self.cache_paths.silver_meta_dir)
        ensure_directory(self.cache_paths.silver_dir)
        (
            pl.scan_csv(
                self.spec.source_root / _KDDCUP_MAIN_CSV,
                null_values=["NaN", "nan"],
            )
            .sink_parquet(self.cache_paths.silver_dir / _KDDCUP_MAIN_PARQUET)
        )
        location_frame = pl.read_csv(self.spec.source_root / _KDDCUP_LOCATION_CSV)
        location_frame.write_parquet(
            self.cache_paths.silver_meta_dir / _KDDCUP_LOCATION_PARQUET
        )
        turbine_static = ensure_turbine_static_schema(
            location_frame.select(
                pl.lit(self.spec.dataset_id).alias("dataset"),
                pl.col("TurbID").cast(pl.String).alias("turbine_id"),
                pl.col("TurbID").cast(pl.String).alias("source_turbine_key"),
                pl.lit(None).cast(pl.Float64).alias("latitude"),
                pl.lit(None).cast(pl.Float64).alias("longitude"),
                pl.col("x").cast(pl.Float64, strict=False).alias("coord_x"),
                pl.col("y").cast(pl.Float64, strict=False).alias("coord_y"),
                pl.lit("projected_xy").alias("coord_kind"),
                pl.lit("unknown_unverified").alias("coord_crs"),
                pl.lit(None).cast(pl.Float64).alias("elevation_m"),
                pl.lit(1500.0).cast(pl.Float64).alias("rated_power_kw"),
                pl.lit(None).cast(pl.Float64).alias("hub_height_m"),
                pl.lit(None).cast(pl.Float64).alias("rotor_diameter_m"),
                pl.lit(None).cast(pl.String).alias("manufacturer"),
                pl.lit(None).cast(pl.String).alias("model"),
                pl.lit(None).cast(pl.String).alias("country"),
                pl.lit(None).cast(pl.String).alias("commercial_operation_date"),
                pl.lit(_KDDCUP_LOCATION_CSV).alias("spatial_source"),
            )
        ).filter(
            pl.col("turbine_id").is_not_null()
            & pl.col("turbine_id").is_in(list(self.spec.turbine_ids))
        ).unique(subset=["turbine_id"], keep="first")
        turbine_static.write_parquet(self.cache_paths.silver_turbine_static_path)
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
        if not (self.cache_paths.silver_dir / _KDDCUP_MAIN_PARQUET).exists():
            self.build_silver()

        manifest_payload = self.ensure_manifest()
        _assert_sdwpf_time_semantics_supported(manifest_payload)
        frame = pl.read_parquet(self.cache_paths.silver_dir / _KDDCUP_MAIN_PARQUET)
        timestamp_expr = _timestamp_expr_for_dtypes(frame.schema["Tmstamp"])
        feature_columns = [
            column
            for column in frame.columns
            if column not in {"TurbID", "Day", "Tmstamp", self.spec.target_column}
        ]
        target_expr = pl.col(self.spec.target_column).cast(pl.Float64, strict=False)
        unknown_patv_wspd_expr = (
            target_expr.le(0) & pl.col("Wspd").cast(pl.Float64, strict=False).gt(2.5)
        ).fill_null(False)
        pitch_columns = [column for column in ("Pab1", "Pab2", "Pab3") if column in frame.columns]
        if pitch_columns:
            unknown_pitch_expr = pl.any_horizontal(
                [pl.col(column).cast(pl.Float64, strict=False).gt(89) for column in pitch_columns]
            ).fill_null(False)
        else:
            unknown_pitch_expr = pl.lit(False)
        abnormal_ndir_expr = (
            pl.col("Ndir").cast(pl.Float64, strict=False).lt(-720)
            | pl.col("Ndir").cast(pl.Float64, strict=False).gt(720)
        ).fill_null(False)
        abnormal_wdir_expr = (
            pl.col("Wdir").cast(pl.Float64, strict=False).lt(-180)
            | pl.col("Wdir").cast(pl.Float64, strict=False).gt(180)
        ).fill_null(False)
        is_unknown_value_expr = (unknown_patv_wspd_expr | unknown_pitch_expr).fill_null(False)
        is_abnormal_value_expr = (abnormal_ndir_expr | abnormal_wdir_expr).fill_null(False)
        base_quality_flags_expr = pl.concat_str(
            [
                pl.when(unknown_patv_wspd_expr)
                .then(pl.lit("sdwpf_unknown_patv_wspd"))
                .otherwise(None),
                pl.when(unknown_pitch_expr).then(pl.lit("sdwpf_unknown_pitch")).otherwise(None),
                pl.when(abnormal_ndir_expr).then(pl.lit("sdwpf_abnormal_ndir")).otherwise(None),
                pl.when(abnormal_wdir_expr).then(pl.lit("sdwpf_abnormal_wdir")).otherwise(None),
            ],
            separator="|",
            ignore_nulls=True,
        ).fill_null("")

        gold_input = frame.with_columns(
            pl.col("TurbID").cast(pl.String).alias("turbine_id"),
            timestamp_expr,
            target_expr.alias("target_kw"),
            pl.lit(self.spec.dataset_id).alias("dataset"),
            pl.lit(True).alias("__row_present"),
            is_unknown_value_expr.alias("sdwpf_is_unknown"),
            is_abnormal_value_expr.alias("sdwpf_is_abnormal"),
            (is_unknown_value_expr | is_abnormal_value_expr)
            .fill_null(False)
            .alias("sdwpf_is_masked"),
            base_quality_flags_expr.alias("__base_quality_flags"),
            *[pl.col(column).cast(pl.Float64, strict=False) for column in feature_columns],
        ).select(
            [
                "dataset",
                "turbine_id",
                "timestamp",
                "target_kw",
                "__row_present",
                "__base_quality_flags",
                "sdwpf_is_unknown",
                "sdwpf_is_abnormal",
                "sdwpf_is_masked",
                *feature_columns,
            ]
        )

        gold_base = reindex_regular_series(gold_input, self.spec, layout=resolved_layout)
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
        gold_base.write_parquet(series_path)

        flag_counts = gold_input.select(
            (
                pl.col("__base_quality_flags")
                .str.contains(r"(^|\|)sdwpf_unknown_patv_wspd($|\|)")
                .cast(pl.Int64)
                .sum()
            ).alias("sdwpf_unknown_patv_wspd"),
            (
                pl.col("__base_quality_flags")
                .str.contains(r"(^|\|)sdwpf_unknown_pitch($|\|)")
                .cast(pl.Int64)
                .sum()
            ).alias("sdwpf_unknown_pitch"),
            (
                pl.col("__base_quality_flags")
                .str.contains(r"(^|\|)sdwpf_abnormal_ndir($|\|)")
                .cast(pl.Int64)
                .sum()
            ).alias("sdwpf_abnormal_ndir"),
            (
                pl.col("__base_quality_flags")
                .str.contains(r"(^|\|)sdwpf_abnormal_wdir($|\|)")
                .cast(pl.Int64)
                .sum()
            ).alias("sdwpf_abnormal_wdir"),
            (
                pl.col("sdwpf_is_unknown").cast(pl.Int64).sum()
            ).alias("sdwpf_unknown_count"),
            pl.col("sdwpf_is_abnormal").cast(pl.Int64).sum().alias("sdwpf_abnormal_count"),
            pl.col("sdwpf_is_masked").cast(pl.Int64).sum().alias("sdwpf_masked_count"),
        ).row(0, named=True)
        del frame
        del gold_input
        del gold_base
        gc.collect()
        report_frame = load_quality_report_frame(series_path)
        report = build_quality_report(
            report_frame,
            manifest_payload,
            self.spec,
            extra={
                "quality_profile": resolved_quality_profile,
                "layout": resolved_layout,
                "feature_set": resolved_feature_set,
                "calendar_anchor_date": _KDDCUP_CALENDAR_ANCHOR.date().isoformat(),
                "sdwpf_unknown_count": int(flag_counts["sdwpf_unknown_count"] or 0),
                "sdwpf_abnormal_count": int(flag_counts["sdwpf_abnormal_count"] or 0),
                "sdwpf_masked_count": int(flag_counts["sdwpf_masked_count"] or 0),
                "sdwpf_flag_counts": {
                    "sdwpf_unknown_patv_wspd": int(flag_counts["sdwpf_unknown_patv_wspd"] or 0),
                    "sdwpf_unknown_pitch": int(flag_counts["sdwpf_unknown_pitch"] or 0),
                    "sdwpf_abnormal_ndir": int(flag_counts["sdwpf_abnormal_ndir"] or 0),
                    "sdwpf_abnormal_wdir": int(flag_counts["sdwpf_abnormal_wdir"] or 0),
                },
                **build_coverage_summary(report_frame, self.spec),
            },
        )
        write_quality_report(quality_path, report)
        return series_path

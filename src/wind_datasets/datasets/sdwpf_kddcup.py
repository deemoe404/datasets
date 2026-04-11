from __future__ import annotations

from datetime import datetime
import gc
from pathlib import Path

import polars as pl

from ..source_column_policy import apply_keep_mask_rules, kept_source_columns
from ..utils import ensure_directory, read_json
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
    def required_silver_paths(self) -> tuple[Path, ...]:
        return (
            self.cache_paths.silver_dir / _KDDCUP_MAIN_PARQUET,
            self.cache_paths.silver_meta_dir / _KDDCUP_LOCATION_PARQUET,
            self.cache_paths.silver_turbine_static_path,
        )

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
        source_policy = self.load_source_column_policy()
        ensure_directory(self.cache_paths.silver_meta_dir)
        ensure_directory(self.cache_paths.silver_dir)
        (
            pl.scan_csv(
                self.spec.source_root / _KDDCUP_MAIN_CSV,
                null_values=["NaN", "nan"],
            )
            .select(
                list(
                    kept_source_columns(
                        source_policy,
                        source_asset="main_csv",
                        source_table_or_file="sdwpf_main",
                    )
                )
            )
            .sink_parquet(self.cache_paths.silver_dir / _KDDCUP_MAIN_PARQUET)
        )
        location_frame = pl.read_csv(self.spec.source_root / _KDDCUP_LOCATION_CSV).select(
            list(
                kept_source_columns(
                    source_policy,
                    source_asset="location_csv",
                    source_table_or_file="sdwpf_location",
                )
            )
        )
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
        self._write_silver_build_meta()
        return self.cache_paths.silver_dir

    def gold_base_blocked_reason(
        self,
    ) -> str | None:
        if self.manifest_status().status != "fresh":
            return None
        manifest_payload = read_json(self.cache_paths.manifest_path)
        time_check = manifest_payload.get("time_semantics_check")
        if not isinstance(time_check, dict):
            return None
        if time_check.get("status") != "match_documented_245day_10min_grid":
            return "blocked_by_manifest_time_semantics"
        return None

    def task_cache_blocked_reason(
        self,
        task,
        feature_protocol_id: str,
    ) -> str | None:
        del task, feature_protocol_id
        return self.gold_base_blocked_reason()

    def build_gold_base(
        self,
        quality_profile: str | None = None,
        layout: str | None = None,
    ) -> Path:
        self.resolve_quality_profile(quality_profile)
        self.resolve_series_layout(layout)
        self.ensure_silver_fresh()

        manifest_payload = self.ensure_manifest()
        _assert_sdwpf_time_semantics_supported(manifest_payload)
        source_policy = self.load_source_column_policy()
        frame = pl.read_parquet(self.cache_paths.silver_dir / _KDDCUP_MAIN_PARQUET)
        timestamp_expr = _timestamp_expr_for_dtypes(frame.schema["Tmstamp"])
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
        flagged_frame = frame.with_columns(
            unknown_patv_wspd_expr.alias("__flag_unknown_patv_wspd"),
            unknown_pitch_expr.alias("__flag_unknown_pitch"),
            abnormal_ndir_expr.alias("__flag_abnormal_ndir"),
            abnormal_wdir_expr.alias("__flag_abnormal_wdir"),
        )
        masked_frame = apply_keep_mask_rules(
            flagged_frame,
            policy=source_policy,
            source_asset="main_csv",
            source_table_or_file="sdwpf_main",
        )
        feature_columns = [
            column
            for column in masked_frame.columns
            if column not in {"TurbID", "Day", "Tmstamp", self.spec.target_column}
            and not column.startswith("__flag_")
        ]

        gold_input = masked_frame.with_columns(
            pl.col("TurbID").cast(pl.String).alias("turbine_id"),
            timestamp_expr,
            target_expr.alias("target_kw"),
            pl.lit(self.spec.dataset_id).alias("dataset"),
            pl.lit(True).alias("__row_present"),
            (pl.col("__flag_unknown_patv_wspd") | pl.col("__flag_unknown_pitch"))
            .fill_null(False)
            .alias("sdwpf_is_unknown"),
            (pl.col("__flag_abnormal_ndir") | pl.col("__flag_abnormal_wdir"))
            .fill_null(False)
            .alias("sdwpf_is_abnormal"),
            (
                pl.col("__flag_unknown_patv_wspd")
                | pl.col("__flag_unknown_pitch")
                | pl.col("__flag_abnormal_ndir")
                | pl.col("__flag_abnormal_wdir")
            )
            .fill_null(False)
            .alias("sdwpf_is_masked"),
            pl.concat_str(
                [
                    pl.when(pl.col("__flag_unknown_patv_wspd"))
                    .then(pl.lit("sdwpf_unknown_patv_wspd"))
                    .otherwise(None),
                    pl.when(pl.col("__flag_unknown_pitch")).then(pl.lit("sdwpf_unknown_pitch")).otherwise(None),
                    pl.when(pl.col("__flag_abnormal_ndir")).then(pl.lit("sdwpf_abnormal_ndir")).otherwise(None),
                    pl.when(pl.col("__flag_abnormal_wdir")).then(pl.lit("sdwpf_abnormal_wdir")).otherwise(None),
                ],
                separator="|",
                ignore_nulls=True,
            )
            .fill_null("")
            .alias("__base_quality_flags"),
            pl.lit("").alias("__feature_quality_flags"),
            *[pl.col(column).cast(pl.Float64, strict=False) for column in feature_columns],
        ).select(
            [
                "dataset",
                "turbine_id",
                "timestamp",
                "target_kw",
                "__row_present",
                "__base_quality_flags",
                "__feature_quality_flags",
                "sdwpf_is_unknown",
                "sdwpf_is_abnormal",
                "sdwpf_is_masked",
                *feature_columns,
            ]
        )

        gold_base = reindex_regular_series(gold_input, self.spec, layout="farm")
        ensure_directory(self.cache_paths.gold_base_dir)
        series_path = self.cache_paths.gold_base_series_path
        quality_path = self.cache_paths.gold_base_quality_path
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
                "series_layout": "farm_synchronous",
                "calendar_anchor_date": _KDDCUP_CALENDAR_ANCHOR.date().isoformat(),
                "sdwpf_unknown_count": int(flag_counts["sdwpf_unknown_count"] or 0),
                "sdwpf_abnormal_count": int(flag_counts["sdwpf_abnormal_count"] or 0),
                "sdwpf_masked_count": int(flag_counts["sdwpf_masked_count"] or 0),
                **self.source_policy_report_extra(source_policy),
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
        self._write_gold_base_build_meta()
        return series_path

from __future__ import annotations

import shutil

import polars as pl

from ..utils import ensure_directory
from .base import BaseDatasetBuilder
from .common import build_quality_report, reindex_regular_series, write_quality_report

_SDWPF_SUPPORTED_QUALITY_PROFILES = {
    "official_v1",
    "raw_v1",
    "official_v1_zero_negative_patv",
}


class SDWPFFullDatasetBuilder(BaseDatasetBuilder):
    def resolve_quality_profile(self, quality_profile: str | None = None) -> str:
        resolved = super().resolve_quality_profile(quality_profile)
        if resolved not in _SDWPF_SUPPORTED_QUALITY_PROFILES:
            supported = ", ".join(sorted(_SDWPF_SUPPORTED_QUALITY_PROFILES))
            raise ValueError(
                f"Unsupported sdwpf_full quality profile {resolved!r}. Expected one of: {supported}."
            )
        return resolved

    def build_silver(self) -> Path:
        self.ensure_manifest()
        ensure_directory(self.cache_paths.silver_meta_dir)
        ensure_directory(self.cache_paths.silver_dir)
        shutil.copy2(
            self.spec.source_root / "sdwpf_2001_2112_full.parquet",
            self.cache_paths.silver_dir / "sdwpf_2001_2112_full.parquet",
        )
        pl.read_csv(self.spec.source_root / "sdwpf_turb_location_elevation.csv").write_parquet(
            self.cache_paths.silver_meta_dir / "sdwpf_turb_location_elevation.parquet"
        )
        return self.cache_paths.silver_dir

    def build_gold_base(self, quality_profile: str | None = None) -> Path:
        resolved_quality_profile = self.resolve_quality_profile(quality_profile)
        if not (self.cache_paths.silver_dir / "sdwpf_2001_2112_full.parquet").exists():
            self.build_silver()

        manifest_payload = self.ensure_manifest()
        frame = pl.read_parquet(self.cache_paths.silver_dir / "sdwpf_2001_2112_full.parquet")
        timestamp_dtype = frame.schema["Tmstamp"]
        if timestamp_dtype == pl.Utf8:
            timestamp_expr = (
                pl.col("Tmstamp")
                .cast(pl.Utf8)
                .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
                .alias("timestamp")
            )
        else:
            timestamp_expr = pl.col("Tmstamp").cast(pl.Datetime, strict=False).alias("timestamp")
        feature_columns = [
            column for column in frame.columns if column not in {"TurbID", "Tmstamp", self.spec.target_column}
        ]
        target_raw_expr = pl.col(self.spec.target_column).cast(pl.Float64, strict=False)
        negative_patv_expr = target_raw_expr.lt(0).fill_null(False)
        unknown_patv_wspd_expr = (
            target_raw_expr.le(0) & pl.col("Wspd").cast(pl.Float64, strict=False).gt(2.5)
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
        zero_negative_patv = resolved_quality_profile == "official_v1_zero_negative_patv"
        use_official_rules = resolved_quality_profile != "raw_v1"
        if use_official_rules:
            is_unknown_value_expr = (unknown_patv_wspd_expr | unknown_pitch_expr).fill_null(False)
            is_abnormal_value_expr = (abnormal_ndir_expr | abnormal_wdir_expr).fill_null(False)
            base_quality_flags_expr = pl.concat_str(
                [
                    pl.when(negative_patv_expr).then(pl.lit("sdwpf_patv_negative")).otherwise(None),
                    pl.when(unknown_patv_wspd_expr)
                    .then(pl.lit("sdwpf_unknown_patv_wspd"))
                    .otherwise(None),
                    pl.when(unknown_pitch_expr).then(pl.lit("sdwpf_unknown_pitch")).otherwise(None),
                    pl.when(abnormal_ndir_expr).then(pl.lit("sdwpf_abnormal_ndir")).otherwise(None),
                    pl.when(abnormal_wdir_expr).then(pl.lit("sdwpf_abnormal_wdir")).otherwise(None),
                    pl.when(negative_patv_expr & pl.lit(zero_negative_patv))
                    .then(pl.lit("sdwpf_patv_zeroed"))
                    .otherwise(None),
                ],
                separator="|",
                ignore_nulls=True,
            ).fill_null("")
            target_kw_expr = (
                pl.when(negative_patv_expr & pl.lit(zero_negative_patv))
                .then(pl.lit(0.0))
                .otherwise(target_raw_expr)
                .alias("target_kw")
            )
        else:
            is_unknown_value_expr = pl.lit(False)
            is_abnormal_value_expr = pl.lit(False)
            base_quality_flags_expr = pl.lit("")
            target_kw_expr = target_raw_expr.alias("target_kw")

        gold_input = frame.with_columns(
            pl.col("TurbID").cast(pl.String).alias("turbine_id"),
            timestamp_expr,
            target_raw_expr.alias("target_kw_raw"),
            target_kw_expr,
            pl.lit(self.spec.dataset_id).alias("dataset"),
            pl.lit(True).alias("__row_present"),
            (
                negative_patv_expr.alias("sdwpf_has_negative_patv")
                if use_official_rules
                else pl.lit(False).alias("sdwpf_has_negative_patv")
            ),
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
                "target_kw_raw",
                "target_kw",
                "__row_present",
                "__base_quality_flags",
                "sdwpf_has_negative_patv",
                "sdwpf_is_unknown",
                "sdwpf_is_abnormal",
                "sdwpf_is_masked",
                *feature_columns,
            ]
        )

        gold_base = reindex_regular_series(gold_input, self.spec)
        ensure_directory(self.cache_paths.gold_base_profile_dir(resolved_quality_profile))
        series_path = self.cache_paths.gold_base_series_path_for(resolved_quality_profile)
        quality_path = self.cache_paths.gold_base_quality_path_for(resolved_quality_profile)
        gold_base.write_parquet(series_path)

        flag_counts = gold_input.select(
            pl.col("sdwpf_has_negative_patv").cast(pl.Int64).sum().alias("sdwpf_patv_negative"),
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
                pl.col("__base_quality_flags")
                .str.contains(r"(^|\|)sdwpf_patv_zeroed($|\|)")
                .cast(pl.Int64)
                .sum()
            ).alias("sdwpf_patv_zeroed"),
            pl.col("sdwpf_is_unknown").cast(pl.Int64).sum().alias("sdwpf_unknown_count"),
            pl.col("sdwpf_is_abnormal").cast(pl.Int64).sum().alias("sdwpf_abnormal_count"),
            pl.col("sdwpf_is_masked").cast(pl.Int64).sum().alias("sdwpf_masked_count"),
        ).row(0, named=True)
        report = build_quality_report(
            gold_base,
            manifest_payload,
            self.spec,
            extra={
                "quality_profile": resolved_quality_profile,
                "sdwpf_negative_patv_count": int(flag_counts["sdwpf_patv_negative"] or 0),
                "sdwpf_unknown_count": int(flag_counts["sdwpf_unknown_count"] or 0),
                "sdwpf_abnormal_count": int(flag_counts["sdwpf_abnormal_count"] or 0),
                "sdwpf_masked_count": int(flag_counts["sdwpf_masked_count"] or 0),
                "sdwpf_flag_counts": {
                    "sdwpf_patv_negative": int(flag_counts["sdwpf_patv_negative"] or 0),
                    "sdwpf_unknown_patv_wspd": int(flag_counts["sdwpf_unknown_patv_wspd"] or 0),
                    "sdwpf_unknown_pitch": int(flag_counts["sdwpf_unknown_pitch"] or 0),
                    "sdwpf_abnormal_ndir": int(flag_counts["sdwpf_abnormal_ndir"] or 0),
                    "sdwpf_abnormal_wdir": int(flag_counts["sdwpf_abnormal_wdir"] or 0),
                    "sdwpf_patv_zeroed": int(flag_counts["sdwpf_patv_zeroed"] or 0),
                },
            },
        )
        write_quality_report(quality_path, report)
        return series_path

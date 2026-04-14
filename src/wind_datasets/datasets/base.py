from __future__ import annotations

from math import atan2, cos, degrees, radians, sqrt
from pathlib import Path
from typing import Any

import polars as pl

from ..cache_state import (
    LayerStatus,
    check_gold_base_status,
    check_manifest_status,
    check_silver_status,
    check_task_status,
    expected_gold_base_meta,
    expected_silver_meta,
    expected_task_meta,
    write_build_meta,
)
from ..feature_protocols import (
    DEFAULT_FEATURE_PROTOCOL_ID,
    build_known_future_frame,
    feature_protocol_task_blocked_reason,
    materialize_task_series,
    protocol_context_dict,
    select_task_series_columns,
)
from ..manifest import build_manifest
from ..models import DatasetSpec, LoadedTaskBundle, ResolvedTaskSpec, TaskBundlePaths, TaskSpec
from ..paths import dataset_cache_paths
from ..source_column_policy import (
    SourceColumnPolicy,
    load_source_column_policy,
    validate_policy_coverage,
)
from ..utils import ensure_directory, join_flags, read_json, write_json
from .common import DUPLICATE_AUDIT_SCHEMA, build_window_index_from_series_path

_TASK_MISSING_PAST_COVARIATE_FLAG = "missing_past_covariates"
_EARTH_RADIUS_M = 6_371_000.0


class BaseDatasetBuilder:
    def __init__(self, spec: DatasetSpec, cache_root: Path) -> None:
        self.spec = spec
        self.cache_root = cache_root
        self.cache_paths = dataset_cache_paths(cache_root, spec.dataset_id)

    def resolve_quality_profile(self, quality_profile: str | None = None) -> str:
        resolved = quality_profile or self.spec.default_quality_profile
        if resolved != "default":
            raise ValueError(
                "quality_profile is no longer a public cache dimension. Only the legacy default profile remains valid."
            )
        return resolved

    def resolve_series_layout(self, layout: str | None = None) -> str:
        resolved = layout or "farm"
        if resolved != "farm":
            raise ValueError(
                "gold_base is now a single farm-synchronous cache. Only layout='farm' remains supported."
            )
        return resolved

    def _resolve_supported_task(self, task_spec: TaskSpec) -> ResolvedTaskSpec:
        resolved = task_spec.resolve(self.spec.resolution_minutes)
        if resolved.granularity != "farm":
            raise ValueError(
                "Only farm-level task bundles remain supported in the active architecture. "
                f"Received granularity={resolved.granularity!r}."
            )
        return resolved

    def ensure_manifest(self) -> dict[str, Any]:
        if self.manifest_status().status != "fresh":
            build_manifest(self.spec, self.cache_root)
        return read_json(self.cache_paths.manifest_path)

    def ensure_manifest_fresh(self) -> dict[str, Any]:
        return self.ensure_manifest()

    def manifest_status(self) -> LayerStatus:
        return check_manifest_status(self.spec, self.cache_paths)

    def required_silver_paths(self) -> tuple[Path, ...]:
        raise NotImplementedError

    def silver_status(self) -> LayerStatus:
        return check_silver_status(
            self.spec,
            self.cache_paths,
            required_outputs=self.required_silver_paths(),
        )

    def gold_base_status(
        self,
        quality_profile: str | None = None,
        layout: str | None = None,
    ) -> LayerStatus:
        self.resolve_quality_profile(quality_profile)
        self.resolve_series_layout(layout)
        return check_gold_base_status(
            self.spec,
            self.cache_paths,
            blocked_reason=self.gold_base_blocked_reason(),
        )

    def task_cache_status(
        self,
        task_spec: TaskSpec,
        feature_protocol_id: str = DEFAULT_FEATURE_PROTOCOL_ID,
        quality_profile: str | None = None,
    ) -> LayerStatus:
        self.resolve_quality_profile(quality_profile)
        resolved_task = self._resolve_supported_task(task_spec)
        return check_task_status(
            self.spec,
            self.cache_paths,
            task=resolved_task,
            feature_protocol_id=feature_protocol_id,
            blocked_reason=self.task_cache_blocked_reason(resolved_task, feature_protocol_id),
        )

    def gold_base_blocked_reason(self) -> str | None:
        return None

    def task_cache_blocked_reason(
        self,
        task: ResolvedTaskSpec,
        feature_protocol_id: str,
    ) -> str | None:
        del task
        return feature_protocol_task_blocked_reason(
            dataset_id=self.spec.dataset_id,
            feature_protocol_id=feature_protocol_id,
        )

    def ensure_silver_fresh(self) -> None:
        if self.silver_status().status != "fresh":
            self.build_silver()

    def ensure_gold_base_fresh(
        self,
        quality_profile: str | None = None,
        layout: str | None = None,
    ) -> None:
        if self.gold_base_status(quality_profile=quality_profile, layout=layout).status != "fresh":
            self.build_gold_base(quality_profile=quality_profile, layout=layout)

    def ensure_task_cache_fresh(
        self,
        task_spec: TaskSpec,
        feature_protocol_id: str = DEFAULT_FEATURE_PROTOCOL_ID,
        quality_profile: str | None = None,
    ) -> None:
        if self.task_cache_status(
            task_spec,
            feature_protocol_id=feature_protocol_id,
            quality_profile=quality_profile,
        ).status != "fresh":
            self.build_task_cache(
                task_spec,
                feature_protocol_id=feature_protocol_id,
                quality_profile=quality_profile,
            )

    def build_silver(self) -> Path:
        raise NotImplementedError

    def build_gold_base(
        self,
        quality_profile: str | None = None,
        layout: str | None = None,
    ) -> Path:
        raise NotImplementedError

    def load_turbine_static(self) -> pl.DataFrame:
        self.ensure_silver_fresh()
        if not self.cache_paths.silver_turbine_static_path.exists():
            self.build_silver()
        return pl.read_parquet(self.cache_paths.silver_turbine_static_path)

    def load_shared_timeseries(self, group_name: str) -> pl.DataFrame:
        path = self.cache_paths.silver_shared_ts_path(group_name)
        self.ensure_silver_fresh()
        if not path.exists():
            self.build_silver()
        return pl.read_parquet(path)

    def load_duplicate_audit(self) -> pl.DataFrame:
        path = self.cache_paths.duplicate_audit_path
        self.ensure_silver_fresh()
        if not path.exists():
            self.build_silver()
        if not path.exists():
            return pl.DataFrame(schema=DUPLICATE_AUDIT_SCHEMA)
        return pl.read_parquet(path)

    def load_event_features(self, group_name: str) -> pl.DataFrame:
        path = self.cache_paths.silver_event_features_path(group_name)
        self.ensure_silver_fresh()
        if not path.exists():
            self.build_silver()
        return pl.read_parquet(path)

    def load_interventions(self, group_name: str) -> pl.DataFrame:
        path = self.cache_paths.silver_interventions_path(group_name)
        self.ensure_silver_fresh()
        if not path.exists():
            self.build_silver()
        return pl.read_parquet(path)

    def task_bundle_paths(
        self,
        task_spec: TaskSpec,
        feature_protocol_id: str = DEFAULT_FEATURE_PROTOCOL_ID,
    ) -> TaskBundlePaths:
        resolved = self._resolve_supported_task(task_spec)
        task_dir = self.cache_paths.task_dir_for(resolved.task_id, feature_protocol_id)
        return TaskBundlePaths(
            dataset_id=self.spec.dataset_id,
            task_id=resolved.task_id,
            feature_protocol_id=feature_protocol_id,
            task_dir=task_dir,
            series_path=self.cache_paths.task_series_path_for(resolved.task_id, feature_protocol_id),
            known_future_path=self.cache_paths.task_known_future_path_for(resolved.task_id, feature_protocol_id),
            static_path=self.cache_paths.task_turbine_static_path_for(resolved.task_id, feature_protocol_id),
            window_index_path=self.cache_paths.task_window_index_path_for(resolved.task_id, feature_protocol_id),
            task_context_path=self.cache_paths.task_context_path_for(resolved.task_id, feature_protocol_id),
            task_report_path=self.cache_paths.task_report_path_for(resolved.task_id, feature_protocol_id),
            build_meta_path=self.cache_paths.task_build_meta_path_for(resolved.task_id, feature_protocol_id),
            pairwise_path=self.cache_paths.task_pairwise_path_for(resolved.task_id, feature_protocol_id),
        )

    def build_task_cache(
        self,
        task_spec: TaskSpec,
        feature_protocol_id: str = DEFAULT_FEATURE_PROTOCOL_ID,
        quality_profile: str | None = None,
    ) -> Path:
        self.resolve_quality_profile(quality_profile)
        resolved = self._resolve_supported_task(task_spec)
        self.ensure_gold_base_fresh()

        task_paths = self.task_bundle_paths(task_spec, feature_protocol_id=feature_protocol_id)
        ensure_directory(task_paths.task_dir)

        available_columns = set(pl.scan_parquet(self.cache_paths.gold_base_series_path).collect_schema().names())
        static_columns = (
            set(pl.scan_parquet(self.cache_paths.silver_turbine_static_path).collect_schema().names())
            if self.cache_paths.silver_turbine_static_path.exists()
            else set()
        )
        selection = select_task_series_columns(
            dataset_id=self.spec.dataset_id,
            available_columns=available_columns,
            feature_protocol_id=feature_protocol_id,
            turbine_static_columns=static_columns,
        )

        source_frame = (
            pl.scan_parquet(self.cache_paths.gold_base_series_path)
            .select(list(selection.source_columns))
            .sort(["turbine_id", "timestamp"])
            .collect()
        )
        materialized = materialize_task_series(source_frame, selection=selection)
        series_frame = materialized.series_frame
        if selection.past_covariate_value_columns:
            missing_past_covariate_expr = pl.any_horizontal(
                [pl.col(column).is_null() for column in selection.past_covariate_value_columns]
            )
            series_frame = (
                series_frame
                .with_columns(
                    pl.when(missing_past_covariate_expr)
                    .then(pl.lit(_TASK_MISSING_PAST_COVARIATE_FLAG))
                    .otherwise(pl.lit(None))
                    .alias("__task_feature_quality_flag")
                )
                .with_columns(
                    pl.struct(["feature_quality_flags", "__task_feature_quality_flag"])
                    .map_elements(
                        lambda value: join_flags(
                            value["feature_quality_flags"],
                            value["__task_feature_quality_flag"],
                        ),
                        return_dtype=pl.String,
                    )
                    .alias("feature_quality_flags")
                )
                .drop("__task_feature_quality_flag")
            )
        series_frame.write_parquet(task_paths.series_path)

        known_future = build_known_future_frame(series_frame)
        known_future.write_parquet(task_paths.known_future_path)

        task_static = self._write_task_static(task_paths, static_columns=selection.static_columns)
        task_pairwise = self._write_task_pairwise(
            task_paths,
            static_frame=task_static,
            pairwise_columns=selection.pairwise_columns,
        )

        build_window_index_from_series_path(
            series_path=task_paths.series_path,
            task=resolved,
            output_path=task_paths.window_index_path,
            report_path=task_paths.task_report_path,
            quality_profile="default",
            available_columns=set(series_frame.columns),
        )
        self._write_task_context(
            task_paths,
            resolved,
            feature_protocol_id,
            selection,
            static_columns=tuple(task_static.columns),
        )
        self._finalize_task_report(
            task_paths,
            resolved,
            feature_protocol_id,
            selection,
            mask_hit_counts_by_column=materialized.mask_hit_counts_by_column,
            null_cause_counts_by_output_column=materialized.null_cause_counts_by_output_column,
        )
        self._write_task_build_meta(task=resolved, feature_protocol_id=feature_protocol_id)
        return task_paths.task_dir

    def load_series(
        self,
        quality_profile: str | None = None,
        layout: str | None = None,
    ) -> pl.DataFrame:
        self.resolve_quality_profile(quality_profile)
        self.resolve_series_layout(layout)
        self.ensure_gold_base_fresh()
        if not self.cache_paths.gold_base_series_path.exists():
            self.build_gold_base()
        return pl.read_parquet(self.cache_paths.gold_base_series_path)

    def load_task_bundle(
        self,
        task_spec: TaskSpec,
        feature_protocol_id: str = DEFAULT_FEATURE_PROTOCOL_ID,
        quality_profile: str | None = None,
    ) -> LoadedTaskBundle:
        self.ensure_task_cache_fresh(
            task_spec,
            feature_protocol_id=feature_protocol_id,
            quality_profile=quality_profile,
        )
        paths = self.task_bundle_paths(task_spec, feature_protocol_id=feature_protocol_id)
        return LoadedTaskBundle(
            paths=paths,
            series=pl.read_parquet(paths.series_path),
            known_future=pl.read_parquet(paths.known_future_path),
            static=pl.read_parquet(paths.static_path),
            window_index=pl.read_parquet(paths.window_index_path),
            task_context=read_json(paths.task_context_path),
            task_report=read_json(paths.task_report_path) if paths.task_report_path.exists() else None,
            pairwise=(
                pl.read_parquet(paths.pairwise_path)
                if paths.pairwise_path is not None and paths.pairwise_path.exists()
                else None
            ),
        )

    def load_window_index(
        self,
        task_spec: TaskSpec,
        feature_protocol_id: str = DEFAULT_FEATURE_PROTOCOL_ID,
        quality_profile: str | None = None,
    ) -> pl.DataFrame:
        self.ensure_task_cache_fresh(
            task_spec,
            feature_protocol_id=feature_protocol_id,
            quality_profile=quality_profile,
        )
        task_path = self.task_bundle_paths(task_spec, feature_protocol_id=feature_protocol_id).window_index_path
        if not task_path.exists():
            self.build_task_cache(task_spec, feature_protocol_id=feature_protocol_id, quality_profile=quality_profile)
        return pl.read_parquet(task_path)

    def load_task_turbine_static(
        self,
        task_spec: TaskSpec,
        feature_protocol_id: str = DEFAULT_FEATURE_PROTOCOL_ID,
        quality_profile: str | None = None,
    ) -> pl.DataFrame:
        self.ensure_task_cache_fresh(
            task_spec,
            feature_protocol_id=feature_protocol_id,
            quality_profile=quality_profile,
        )
        static_path = self.task_bundle_paths(task_spec, feature_protocol_id=feature_protocol_id).static_path
        if not static_path.exists():
            self.build_task_cache(task_spec, feature_protocol_id=feature_protocol_id, quality_profile=quality_profile)
        return pl.read_parquet(static_path)

    def profile_dataset(
        self,
        quality_profile: str | None = None,
        layout: str | None = None,
    ) -> dict[str, Any]:
        self.resolve_quality_profile(quality_profile)
        self.resolve_series_layout(layout)
        self.ensure_gold_base_fresh()
        if not self.cache_paths.gold_base_quality_path.exists():
            self.build_gold_base()
        return read_json(self.cache_paths.gold_base_quality_path)

    def load_source_column_policy(self) -> SourceColumnPolicy:
        policy = load_source_column_policy(self.spec.dataset_id)
        manifest_payload = self.ensure_manifest()
        inventory_rows = manifest_payload.get("source_schema_inventory") or ()
        validate_policy_coverage(policy, inventory_rows)
        return policy

    def source_policy_report_extra(self, policy: SourceColumnPolicy) -> dict[str, Any]:
        return {
            "source_column_policy_path": policy.relative_path,
            "source_column_policy_decision_counts": policy.decision_counts,
            "source_column_policy_entry_count": len(policy.entries),
        }

    def _task_static_frame(self, *, selected_columns: tuple[str, ...] = ()) -> pl.DataFrame:
        turbine_static = self.load_turbine_static()
        with_index = turbine_static.join(
            pl.DataFrame(
                {
                    "turbine_id": list(self.spec.turbine_ids),
                    "turbine_index": list(range(len(self.spec.turbine_ids))),
                }
            ),
            on="turbine_id",
            how="right",
        ).sort("turbine_index")
        base_columns = ["dataset", "turbine_id", "turbine_index"]
        metadata_columns = [
            column for column in with_index.columns if column not in {"dataset", "turbine_id", "turbine_index"}
        ]
        frame = with_index.select([*base_columns, *metadata_columns])
        if selected_columns:
            missing_columns = [column for column in selected_columns if column not in frame.columns]
            if missing_columns:
                raise ValueError(
                    f"Task static selection for dataset {self.spec.dataset_id!r} is missing columns {missing_columns!r}."
                )
            frame = frame.select(list(selected_columns))
        return frame

    def _write_task_static(
        self,
        task_paths: TaskBundlePaths,
        *,
        static_columns: tuple[str, ...] = (),
    ) -> pl.DataFrame:
        frame = self._task_static_frame(selected_columns=static_columns)
        frame.write_parquet(task_paths.static_path)
        return frame

    def _empty_task_pairwise_frame(self, pairwise_columns: tuple[str, ...]) -> pl.DataFrame:
        if not pairwise_columns:
            return pl.DataFrame()
        schema: dict[str, pl.DataType] = {
            "src_turbine_id": pl.String,
            "dst_turbine_id": pl.String,
            "src_turbine_index": pl.Int64,
            "dst_turbine_index": pl.Int64,
            "delta_x_m": pl.Float64,
            "delta_y_m": pl.Float64,
            "distance_m": pl.Float64,
            "bearing_deg": pl.Float64,
            "elevation_diff_m": pl.Float64,
            "distance_in_rotor_diameters": pl.Float64,
        }
        return pl.DataFrame(schema={column: schema[column] for column in pairwise_columns})

    def _resolve_task_pairwise_coordinates(self, static_frame: pl.DataFrame) -> tuple[list[float], list[float]]:
        if {"coord_x", "coord_y"}.issubset(static_frame.columns):
            if static_frame["coord_x"].null_count() == 0 and static_frame["coord_y"].null_count() == 0:
                return (
                    static_frame["coord_x"].cast(pl.Float64).to_list(),
                    static_frame["coord_y"].cast(pl.Float64).to_list(),
                )
        if "latitude" not in static_frame.columns or "longitude" not in static_frame.columns:
            raise ValueError(
                "Pairwise geometry requires either full coord_x/coord_y or full latitude/longitude."
            )
        if static_frame["latitude"].null_count() > 0 or static_frame["longitude"].null_count() > 0:
            raise ValueError(
                "Pairwise geometry requires either full coord_x/coord_y or full latitude/longitude."
            )
        latitudes = [float(value) for value in static_frame["latitude"].cast(pl.Float64).to_list()]
        longitudes = [float(value) for value in static_frame["longitude"].cast(pl.Float64).to_list()]
        lat0 = radians(sum(latitudes) / len(latitudes))
        lon0 = radians(sum(longitudes) / len(longitudes))
        x = [_EARTH_RADIUS_M * (radians(lon) - lon0) * cos(lat0) for lon in longitudes]
        lat_mean_rad = radians(sum(latitudes) / len(latitudes))
        y = [_EARTH_RADIUS_M * (radians(lat) - lat_mean_rad) for lat in latitudes]
        return x, y

    def _task_pairwise_frame(
        self,
        *,
        static_frame: pl.DataFrame,
        pairwise_columns: tuple[str, ...],
    ) -> pl.DataFrame:
        if not pairwise_columns:
            return self._empty_task_pairwise_frame(pairwise_columns)
        required_columns = {
            "turbine_id",
            "turbine_index",
            "coord_kind",
            "coord_crs",
            "elevation_m",
            "rotor_diameter_m",
        }
        missing_columns = sorted(required_columns - set(static_frame.columns))
        if missing_columns:
            raise ValueError(
                f"Task pairwise geometry for dataset {self.spec.dataset_id!r} is missing static columns {missing_columns!r}."
            )
        for column in ("coord_kind", "coord_crs", "elevation_m", "rotor_diameter_m"):
            if static_frame[column].null_count() > 0:
                raise ValueError(
                    f"Task pairwise geometry for dataset {self.spec.dataset_id!r} requires non-null static column {column!r}."
                )
        ordered = static_frame.sort("turbine_index")
        if ordered.height <= 1:
            return self._empty_task_pairwise_frame(pairwise_columns)
        x_coords, y_coords = self._resolve_task_pairwise_coordinates(ordered)
        turbine_ids = ordered["turbine_id"].cast(pl.String).to_list()
        turbine_indices = [int(value) for value in ordered["turbine_index"].to_list()]
        elevations = [float(value) for value in ordered["elevation_m"].cast(pl.Float64).to_list()]
        rotor_diameters = [float(value) for value in ordered["rotor_diameter_m"].cast(pl.Float64).to_list()]
        rows: list[dict[str, object]] = []
        for src_position, src_turbine_id in enumerate(turbine_ids):
            for dst_position, dst_turbine_id in enumerate(turbine_ids):
                if src_position == dst_position:
                    continue
                mean_rotor_diameter = (rotor_diameters[src_position] + rotor_diameters[dst_position]) / 2.0
                if mean_rotor_diameter <= 0:
                    raise ValueError(
                        f"Task pairwise geometry for dataset {self.spec.dataset_id!r} requires positive rotor diameters."
                    )
                delta_x_m = x_coords[dst_position] - x_coords[src_position]
                delta_y_m = y_coords[dst_position] - y_coords[src_position]
                distance_m = sqrt((delta_x_m ** 2) + (delta_y_m ** 2))
                rows.append(
                    {
                        "src_turbine_id": src_turbine_id,
                        "dst_turbine_id": dst_turbine_id,
                        "src_turbine_index": turbine_indices[src_position],
                        "dst_turbine_index": turbine_indices[dst_position],
                        "delta_x_m": delta_x_m,
                        "delta_y_m": delta_y_m,
                        "distance_m": distance_m,
                        "bearing_deg": (degrees(atan2(delta_x_m, delta_y_m)) + 360.0) % 360.0,
                        "elevation_diff_m": elevations[dst_position] - elevations[src_position],
                        "distance_in_rotor_diameters": distance_m / mean_rotor_diameter,
                    }
                )
        return pl.DataFrame(rows).select(list(pairwise_columns))

    def _write_task_pairwise(
        self,
        task_paths: TaskBundlePaths,
        *,
        static_frame: pl.DataFrame,
        pairwise_columns: tuple[str, ...],
    ) -> pl.DataFrame:
        frame = self._task_pairwise_frame(
            static_frame=static_frame,
            pairwise_columns=pairwise_columns,
        )
        if task_paths.pairwise_path is None:
            raise ValueError("Task bundle paths are missing pairwise_path.")
        frame.write_parquet(task_paths.pairwise_path)
        return frame

    def _write_task_context(
        self,
        task_paths: TaskBundlePaths,
        task: ResolvedTaskSpec,
        feature_protocol_id: str,
        selection,
        *,
        static_columns: tuple[str, ...],
    ) -> Path:
        payload = protocol_context_dict(
            dataset_id=self.spec.dataset_id,
            task=task.to_dict(),
            feature_protocol_id=feature_protocol_id,
            turbine_ids=self.spec.turbine_ids,
            selection=selection,
            static_columns=static_columns,
        )
        return write_json(task_paths.task_context_path, payload)

    def _finalize_task_report(
        self,
        task_paths: TaskBundlePaths,
        task: ResolvedTaskSpec,
        feature_protocol_id: str,
        selection,
        *,
        mask_hit_counts_by_column: dict[str, int],
        null_cause_counts_by_output_column: dict[str, dict[str, int]],
    ) -> Path:
        payload = read_json(task_paths.task_report_path) if task_paths.task_report_path.exists() else {}
        mask_diagnostics: dict[str, Any] = {
            "mask_hit_counts_by_column": dict(mask_hit_counts_by_column),
        }
        pitch_mean_counts = null_cause_counts_by_output_column.get("pitch_mean")
        if pitch_mean_counts is not None:
            mask_diagnostics["pitch_mean_null_cause_counts"] = {
                "raw_pitch_rule_rows": int(pitch_mean_counts.get("raw_rule_rows", 0)),
                "raw_pitch_missing_rows": int(pitch_mean_counts.get("raw_missing_rows", 0)),
            }
        payload.update(
            {
                "schema_version": "task_report.v1",
                "dataset_id": self.spec.dataset_id,
                "task_id": task.task_id,
                "feature_protocol_id": feature_protocol_id,
                "granularity": task.granularity,
                "series_columns": list(selection.all_columns),
                "target_history_mask_columns": list(selection.target_history_mask_columns),
                "past_covariate_columns": list(selection.past_covariate_columns),
                "past_covariate_value_columns": list(selection.past_covariate_value_columns),
                "past_covariate_mask_columns": list(selection.past_covariate_mask_columns),
                "local_observation_value_columns": list(selection.local_observation_value_columns),
                "local_observation_mask_columns": list(selection.local_observation_mask_columns),
                "global_observation_value_columns": list(selection.global_observation_value_columns),
                "global_observation_mask_columns": list(selection.global_observation_mask_columns),
                "target_derived_columns": list(selection.target_derived_columns),
                "known_future_columns": list(selection.known_future_columns),
                "pairwise_columns": list(selection.pairwise_columns),
                "mask_diagnostics": mask_diagnostics,
            }
        )
        return write_json(task_paths.task_report_path, payload)

    def _write_silver_build_meta(self) -> Path:
        expected = expected_silver_meta(self.spec)
        return write_build_meta(self.cache_paths.silver_build_meta_path, expected)

    def _write_gold_base_build_meta(self) -> Path:
        expected = expected_gold_base_meta(self.spec)
        return write_build_meta(self.cache_paths.gold_base_build_meta_path, expected)

    def _write_task_build_meta(
        self,
        *,
        task: ResolvedTaskSpec,
        feature_protocol_id: str,
    ) -> Path:
        expected = expected_task_meta(
            self.spec,
            task=task,
            feature_protocol_id=feature_protocol_id,
        )
        return write_build_meta(
            self.cache_paths.task_build_meta_path_for(task.task_id, feature_protocol_id),
            expected,
        )

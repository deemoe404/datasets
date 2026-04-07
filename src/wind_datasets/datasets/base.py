from __future__ import annotations

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
from ..manifest import build_manifest
from ..models import DatasetSpec, ResolvedTaskSpec, TaskSpec
from ..paths import dataset_cache_paths
from ..utils import ensure_directory, read_json, write_json
from .common import build_window_index_from_series_path


class BaseDatasetBuilder:
    def __init__(self, spec: DatasetSpec, cache_root: Path) -> None:
        self.spec = spec
        self.cache_root = cache_root
        self.cache_paths = dataset_cache_paths(cache_root, spec.dataset_id)

    def resolve_quality_profile(self, quality_profile: str | None = None) -> str:
        return quality_profile or self.spec.default_quality_profile

    def resolve_series_layout(self, layout: str | None = None) -> str:
        resolved = layout or "farm"
        if resolved not in {"farm", "turbine"}:
            raise ValueError(f"Unsupported series layout {resolved!r}. Expected 'farm' or 'turbine'.")
        return resolved

    def resolve_feature_set(self, feature_set: str | None = None) -> str:
        return feature_set or "default"

    def default_task_feature_set(self) -> str:
        return self.resolve_feature_set(None)

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
        feature_set: str | None = None,
    ) -> LayerStatus:
        resolved_quality_profile = self.resolve_quality_profile(quality_profile)
        resolved_layout = self.resolve_series_layout(layout)
        resolved_feature_set = self.resolve_feature_set(feature_set)
        return check_gold_base_status(
            self.spec,
            self.cache_paths,
            quality_profile=resolved_quality_profile,
            layout=resolved_layout,
            feature_set=resolved_feature_set,
            blocked_reason=self.gold_base_blocked_reason(
                resolved_quality_profile,
                resolved_layout,
                resolved_feature_set,
            ),
        )

    def task_cache_status(
        self,
        task_spec: TaskSpec,
        quality_profile: str | None = None,
    ) -> LayerStatus:
        resolved_quality_profile = self.resolve_quality_profile(quality_profile)
        resolved_task = task_spec.resolve(self.spec.resolution_minutes)
        return check_task_status(
            self.spec,
            self.cache_paths,
            quality_profile=resolved_quality_profile,
            task=resolved_task,
            feature_set=self.default_task_feature_set(),
            blocked_reason=self.task_cache_blocked_reason(
                resolved_quality_profile,
                resolved_task,
            ),
        )

    def gold_base_blocked_reason(
        self,
        quality_profile: str,
        layout: str,
        feature_set: str,
    ) -> str | None:
        del quality_profile, layout, feature_set
        return None

    def task_cache_blocked_reason(
        self,
        quality_profile: str,
        task: ResolvedTaskSpec,
    ) -> str | None:
        del quality_profile, task
        return None

    def ensure_silver_fresh(self) -> None:
        if self.silver_status().status != "fresh":
            self.build_silver()

    def ensure_gold_base_fresh(
        self,
        quality_profile: str | None = None,
        layout: str | None = None,
        feature_set: str | None = None,
    ) -> None:
        if self.gold_base_status(
            quality_profile=quality_profile,
            layout=layout,
            feature_set=feature_set,
        ).status != "fresh":
            self.build_gold_base(
                quality_profile=quality_profile,
                layout=layout,
                feature_set=feature_set,
            )

    def ensure_task_cache_fresh(
        self,
        task_spec: TaskSpec,
        quality_profile: str | None = None,
    ) -> None:
        if self.task_cache_status(task_spec, quality_profile=quality_profile).status != "fresh":
            self.build_task_cache(task_spec, quality_profile=quality_profile)

    def build_silver(self) -> Path:
        raise NotImplementedError

    def build_gold_base(
        self,
        quality_profile: str | None = None,
        layout: str | None = None,
        feature_set: str | None = None,
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

    def build_task_cache(self, task_spec: TaskSpec, quality_profile: str | None = None) -> Path:
        resolved_quality_profile = self.resolve_quality_profile(quality_profile)
        resolved = task_spec.resolve(self.spec.resolution_minutes)
        series_layout = self.resolve_series_layout(resolved.granularity)
        self.ensure_gold_base_fresh(
            quality_profile=resolved_quality_profile,
            layout=series_layout,
            feature_set=self.default_task_feature_set(),
        )
        gold_base_path = self.cache_paths.gold_base_series_path_for(
            resolved_quality_profile,
            layout=series_layout,
            feature_set=self.default_task_feature_set(),
        )

        task_dir = ensure_directory(
            self.cache_paths.task_dir_for(
                resolved_quality_profile,
                resolved.granularity,
                resolved.task_id,
            )
        )
        available_columns = set(pl.scan_parquet(gold_base_path).collect_schema().names())
        output_path = build_window_index_from_series_path(
            series_path=gold_base_path,
            task=resolved,
            output_path=task_dir / "window_index.parquet",
            report_path=task_dir / "task_report.json",
            quality_profile=resolved_quality_profile,
            available_columns=available_columns,
        )
        if resolved.granularity == "farm":
            self._write_task_turbine_static(resolved_quality_profile, resolved.task_id, resolved.granularity)
        self._write_task_context(resolved_quality_profile, resolved)
        self._write_task_build_meta(
            quality_profile=resolved_quality_profile,
            task=resolved,
        )
        return output_path

    def load_series(
        self,
        quality_profile: str | None = None,
        layout: str | None = None,
        feature_set: str | None = None,
    ) -> pl.DataFrame:
        resolved_quality_profile = self.resolve_quality_profile(quality_profile)
        resolved_layout = self.resolve_series_layout(layout)
        resolved_feature_set = self.resolve_feature_set(feature_set)
        self.ensure_gold_base_fresh(
            quality_profile=resolved_quality_profile,
            layout=resolved_layout,
            feature_set=resolved_feature_set,
        )
        gold_base_path = self.cache_paths.gold_base_series_path_for(
            resolved_quality_profile,
            layout=resolved_layout,
            feature_set=resolved_feature_set,
        )
        if not gold_base_path.exists():
            self.build_gold_base(
                quality_profile=resolved_quality_profile,
                layout=resolved_layout,
                feature_set=resolved_feature_set,
            )
        return pl.read_parquet(gold_base_path)

    def load_window_index(
        self,
        task_spec: TaskSpec,
        quality_profile: str | None = None,
    ) -> pl.DataFrame:
        resolved_quality_profile = self.resolve_quality_profile(quality_profile)
        resolved = task_spec.resolve(self.spec.resolution_minutes)
        self.ensure_task_cache_fresh(task_spec, quality_profile=resolved_quality_profile)
        task_path = self.cache_paths.task_window_index_path_for(
            resolved_quality_profile,
            resolved.granularity,
            resolved.task_id,
        )
        if not task_path.exists():
            self.build_task_cache(task_spec, quality_profile=resolved_quality_profile)
        return pl.read_parquet(task_path)

    def load_task_turbine_static(
        self,
        task_spec: TaskSpec,
        quality_profile: str | None = None,
    ) -> pl.DataFrame:
        resolved_quality_profile = self.resolve_quality_profile(quality_profile)
        resolved = task_spec.resolve(self.spec.resolution_minutes)
        self.ensure_task_cache_fresh(task_spec, quality_profile=resolved_quality_profile)
        static_path = self.cache_paths.task_turbine_static_path_for(
            resolved_quality_profile,
            resolved.granularity,
            resolved.task_id,
        )
        if not static_path.exists():
            self.build_task_cache(task_spec, quality_profile=resolved_quality_profile)
        return pl.read_parquet(static_path)

    def profile_dataset(
        self,
        quality_profile: str | None = None,
        layout: str | None = None,
        feature_set: str | None = None,
    ) -> dict[str, Any]:
        resolved_quality_profile = self.resolve_quality_profile(quality_profile)
        resolved_layout = self.resolve_series_layout(layout)
        resolved_feature_set = self.resolve_feature_set(feature_set)
        self.ensure_gold_base_fresh(
            quality_profile=resolved_quality_profile,
            layout=resolved_layout,
            feature_set=resolved_feature_set,
        )
        quality_path = self.cache_paths.gold_base_quality_path_for(
            resolved_quality_profile,
            layout=resolved_layout,
            feature_set=resolved_feature_set,
        )
        if not quality_path.exists():
            self.build_gold_base(
                quality_profile=resolved_quality_profile,
                layout=resolved_layout,
                feature_set=resolved_feature_set,
            )
        return read_json(quality_path)

    def _write_task_turbine_static(self, quality_profile: str, task_id: str, granularity: str) -> Path:
        turbine_static = self.load_turbine_static().join(
            pl.DataFrame(
                {
                    "turbine_id": list(self.spec.turbine_ids),
                    "turbine_index": list(range(len(self.spec.turbine_ids))),
                }
            ),
            on="turbine_id",
            how="right",
        ).sort("turbine_index")
        output_path = self.cache_paths.task_turbine_static_path_for(quality_profile, granularity, task_id)
        ensure_directory(output_path.parent)
        turbine_static.write_parquet(output_path)
        return output_path

    def _write_task_context(self, quality_profile: str, task) -> Path:
        output_path = self.cache_paths.task_context_path_for(quality_profile, task.granularity, task.task_id)
        return write_json(
            output_path,
            {
                "dataset_id": self.spec.dataset_id,
                "quality_profile": quality_profile,
                "task": task.to_dict(),
                "turbine_ids": list(self.spec.turbine_ids),
                "series_layout": self.resolve_series_layout(task.granularity),
            },
        )

    def _write_silver_build_meta(self) -> Path:
        expected = expected_silver_meta(self.spec)
        return write_build_meta(self.cache_paths.silver_build_meta_path, expected)

    def _write_gold_base_build_meta(
        self,
        quality_profile: str,
        layout: str,
        feature_set: str,
    ) -> Path:
        expected = expected_gold_base_meta(
            self.spec,
            quality_profile=quality_profile,
            layout=layout,
            feature_set=feature_set,
        )
        return write_build_meta(
            self.cache_paths.gold_base_build_meta_path_for(
                quality_profile,
                layout=layout,
                feature_set=feature_set,
            ),
            expected,
        )

    def _write_task_build_meta(
        self,
        *,
        quality_profile: str,
        task: ResolvedTaskSpec,
    ) -> Path:
        expected = expected_task_meta(
            self.spec,
            quality_profile=quality_profile,
            task=task,
            feature_set=self.default_task_feature_set(),
        )
        return write_build_meta(
            self.cache_paths.task_build_meta_path_for(
                quality_profile,
                task.granularity,
                task.task_id,
            ),
            expected,
        )

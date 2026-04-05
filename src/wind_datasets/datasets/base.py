from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from ..manifest import build_manifest
from ..models import DatasetSpec, TaskSpec
from ..paths import dataset_cache_paths
from ..utils import ensure_directory, read_json
from .common import build_window_index


class BaseDatasetBuilder:
    def __init__(self, spec: DatasetSpec, cache_root: Path) -> None:
        self.spec = spec
        self.cache_root = cache_root
        self.cache_paths = dataset_cache_paths(cache_root, spec.dataset_id)

    def resolve_quality_profile(self, quality_profile: str | None = None) -> str:
        return quality_profile or self.spec.default_quality_profile

    def ensure_manifest(self) -> dict[str, Any]:
        if not self.cache_paths.manifest_path.exists():
            build_manifest(self.spec, self.cache_root)
        return read_json(self.cache_paths.manifest_path)

    def build_silver(self) -> Path:
        raise NotImplementedError

    def build_gold_base(self, quality_profile: str | None = None) -> Path:
        raise NotImplementedError

    def build_task_cache(self, task_spec: TaskSpec, quality_profile: str | None = None) -> Path:
        resolved_quality_profile = self.resolve_quality_profile(quality_profile)
        gold_base_path = self.cache_paths.gold_base_series_path_for(resolved_quality_profile)
        if not gold_base_path.exists():
            self.build_gold_base(quality_profile=resolved_quality_profile)

        resolved = task_spec.resolve(self.spec.resolution_minutes)
        task_dir = ensure_directory(
            self.cache_paths.task_dir_for(resolved_quality_profile, resolved.task_id)
        )
        df = pl.read_parquet(gold_base_path)
        return build_window_index(
            df=df,
            task=resolved,
            output_path=task_dir / "window_index.parquet",
            report_path=task_dir / "task_report.json",
            quality_profile=resolved_quality_profile,
        )

    def load_series(self, quality_profile: str | None = None) -> pl.DataFrame:
        resolved_quality_profile = self.resolve_quality_profile(quality_profile)
        gold_base_path = self.cache_paths.gold_base_series_path_for(resolved_quality_profile)
        if not gold_base_path.exists():
            self.build_gold_base(quality_profile=resolved_quality_profile)
        return pl.read_parquet(gold_base_path)

    def load_window_index(
        self,
        task_spec: TaskSpec,
        quality_profile: str | None = None,
    ) -> pl.DataFrame:
        resolved_quality_profile = self.resolve_quality_profile(quality_profile)
        resolved = task_spec.resolve(self.spec.resolution_minutes)
        task_path = self.cache_paths.task_window_index_path_for(
            resolved_quality_profile,
            resolved.task_id,
        )
        if not task_path.exists():
            self.build_task_cache(task_spec, quality_profile=resolved_quality_profile)
        return pl.read_parquet(task_path)

    def profile_dataset(self, quality_profile: str | None = None) -> dict[str, Any]:
        resolved_quality_profile = self.resolve_quality_profile(quality_profile)
        quality_path = self.cache_paths.gold_base_quality_path_for(resolved_quality_profile)
        if not quality_path.exists():
            self.build_gold_base(quality_profile=resolved_quality_profile)
        return read_json(quality_path)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetCachePaths:
    root: Path
    manifest_dir: Path
    manifest_path: Path
    silver_dir: Path
    silver_continuous_dir: Path
    silver_events_dir: Path
    silver_meta_dir: Path
    gold_base_dir: Path
    tasks_dir: Path

    @property
    def silver_turbine_static_path(self) -> Path:
        return self.silver_meta_dir / "turbine_static.parquet"

    @property
    def hill_duplicate_audit_path(self) -> Path:
        return self.silver_meta_dir / "default_table_duplicate_audit.parquet"

    def gold_base_profile_dir(self, quality_profile: str) -> Path:
        return self.gold_base_dir / quality_profile

    def gold_base_series_path_for(self, quality_profile: str) -> Path:
        return self.gold_base_profile_dir(quality_profile) / "series.parquet"

    def gold_base_quality_path_for(self, quality_profile: str) -> Path:
        return self.gold_base_profile_dir(quality_profile) / "quality_report.json"

    def task_profile_dir(self, quality_profile: str) -> Path:
        return self.tasks_dir / quality_profile

    def task_dir_for(self, quality_profile: str, task_id: str) -> Path:
        return self.task_profile_dir(quality_profile) / task_id

    def task_window_index_path_for(self, quality_profile: str, task_id: str) -> Path:
        return self.task_dir_for(quality_profile, task_id) / "window_index.parquet"

    def task_report_path_for(self, quality_profile: str, task_id: str) -> Path:
        return self.task_dir_for(quality_profile, task_id) / "task_report.json"


def dataset_cache_paths(cache_root: Path, dataset_id: str) -> DatasetCachePaths:
    root = cache_root / dataset_id
    manifest_dir = root / "manifest"
    silver_dir = root / "silver"
    gold_base_dir = root / "gold_base"
    tasks_dir = root / "tasks"
    return DatasetCachePaths(
        root=root,
        manifest_dir=manifest_dir,
        manifest_path=manifest_dir / "manifest.json",
        silver_dir=silver_dir,
        silver_continuous_dir=silver_dir / "continuous",
        silver_events_dir=silver_dir / "events",
        silver_meta_dir=silver_dir / "meta",
        gold_base_dir=gold_base_dir,
        tasks_dir=tasks_dir,
    )

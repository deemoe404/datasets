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
    silver_shared_ts_dir: Path
    silver_event_features_dir: Path
    silver_interventions_dir: Path
    silver_meta_dir: Path
    gold_base_dir: Path
    tasks_dir: Path

    @property
    def silver_turbine_static_path(self) -> Path:
        return self.silver_meta_dir / "turbine_static.parquet"

    @property
    def hill_default_tables_dir(self) -> Path:
        return self.silver_meta_dir / "default_tables"

    def hill_default_table_path(self, table_name: str) -> Path:
        return self.hill_default_tables_dir / f"{table_name}.parquet"

    @property
    def hill_duplicate_audit_path(self) -> Path:
        return self.silver_meta_dir / "default_table_duplicate_audit.parquet"

    def silver_shared_ts_path(self, group_name: str) -> Path:
        return self.silver_shared_ts_dir / f"{group_name}.parquet"

    @property
    def hill_default_conflict_keys_path(self) -> Path:
        return self.silver_meta_dir / "default_table_conflict_keys.parquet"

    def silver_event_features_path(self, group_name: str) -> Path:
        return self.silver_event_features_dir / f"{group_name}.parquet"

    def silver_interventions_path(self, group_name: str) -> Path:
        return self.silver_interventions_dir / f"{group_name}.parquet"

    def gold_base_profile_dir(
        self,
        quality_profile: str,
        layout: str = "farm",
        feature_set: str = "default",
    ) -> Path:
        return self.gold_base_dir / quality_profile / layout / feature_set

    def gold_base_series_path_for(
        self,
        quality_profile: str,
        layout: str = "farm",
        feature_set: str = "default",
    ) -> Path:
        return self.gold_base_profile_dir(quality_profile, layout=layout, feature_set=feature_set) / "series.parquet"

    def gold_base_quality_path_for(
        self,
        quality_profile: str,
        layout: str = "farm",
        feature_set: str = "default",
    ) -> Path:
        return self.gold_base_profile_dir(quality_profile, layout=layout, feature_set=feature_set) / "quality_report.json"

    def task_profile_dir(self, quality_profile: str, granularity: str = "farm") -> Path:
        return self.tasks_dir / quality_profile / granularity

    def task_dir_for(self, quality_profile: str, granularity: str, task_id: str) -> Path:
        return self.task_profile_dir(quality_profile, granularity=granularity) / task_id

    def task_window_index_path_for(self, quality_profile: str, granularity: str, task_id: str) -> Path:
        return self.task_dir_for(quality_profile, granularity, task_id) / "window_index.parquet"

    def task_report_path_for(self, quality_profile: str, granularity: str, task_id: str) -> Path:
        return self.task_dir_for(quality_profile, granularity, task_id) / "task_report.json"

    def task_turbine_static_path_for(self, quality_profile: str, granularity: str, task_id: str) -> Path:
        return self.task_dir_for(quality_profile, granularity, task_id) / "turbine_static.parquet"

    def task_context_path_for(self, quality_profile: str, granularity: str, task_id: str) -> Path:
        return self.task_dir_for(quality_profile, granularity, task_id) / "task_context.json"


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
        silver_shared_ts_dir=silver_dir / "shared_ts",
        silver_event_features_dir=silver_dir / "event_features",
        silver_interventions_dir=silver_dir / "interventions",
        silver_meta_dir=silver_dir / "meta",
        gold_base_dir=gold_base_dir,
        tasks_dir=tasks_dir,
    )

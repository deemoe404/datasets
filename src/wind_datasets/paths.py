from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetCachePaths:
    root: Path
    manifest_dir: Path
    manifest_path: Path
    manifest_build_meta_path: Path
    silver_dir: Path
    silver_continuous_dir: Path
    silver_events_dir: Path
    silver_shared_ts_dir: Path
    silver_event_features_dir: Path
    silver_interventions_dir: Path
    silver_meta_dir: Path
    silver_build_meta_path: Path
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
    def duplicate_audit_path(self) -> Path:
        return self.silver_meta_dir / "duplicate_audit.parquet"

    @property
    def duplicate_effects_path(self) -> Path:
        return self.silver_meta_dir / "duplicate_effects.parquet"

    @property
    def hill_duplicate_audit_path(self) -> Path:
        return self.duplicate_audit_path

    def silver_shared_ts_path(self, group_name: str) -> Path:
        return self.silver_shared_ts_dir / f"{group_name}.parquet"

    @property
    def hill_default_conflict_keys_path(self) -> Path:
        return self.duplicate_effects_path

    def silver_event_features_path(self, group_name: str) -> Path:
        return self.silver_event_features_dir / f"{group_name}.parquet"

    def silver_interventions_path(self, group_name: str) -> Path:
        return self.silver_interventions_dir / f"{group_name}.parquet"

    @property
    def gold_base_series_path(self) -> Path:
        return self.gold_base_dir / "series.parquet"

    @property
    def gold_base_quality_path(self) -> Path:
        return self.gold_base_dir / "quality_report.json"

    @property
    def gold_base_build_meta_path(self) -> Path:
        return self.gold_base_dir / "_build_meta.json"

    # Legacy-compatibility helpers. The public cache no longer carries quality/layout.
    def gold_base_profile_dir(self, quality_profile: str, layout: str = "farm") -> Path:
        self._assert_legacy_gold_args(quality_profile=quality_profile, layout=layout)
        return self.gold_base_dir

    def gold_base_series_path_for(self, quality_profile: str, layout: str = "farm") -> Path:
        self._assert_legacy_gold_args(quality_profile=quality_profile, layout=layout)
        return self.gold_base_series_path

    def gold_base_quality_path_for(self, quality_profile: str, layout: str = "farm") -> Path:
        self._assert_legacy_gold_args(quality_profile=quality_profile, layout=layout)
        return self.gold_base_quality_path

    def gold_base_build_meta_path_for(self, quality_profile: str, layout: str = "farm") -> Path:
        self._assert_legacy_gold_args(quality_profile=quality_profile, layout=layout)
        return self.gold_base_build_meta_path

    def task_dir_for(self, *args: str) -> Path:
        task_id, feature_protocol_id = self._resolve_task_args(*args)
        return self.tasks_dir / task_id / feature_protocol_id

    def task_window_index_path_for(self, *args: str) -> Path:
        return self.task_dir_for(*args) / "window_index.parquet"

    def task_report_path_for(self, *args: str) -> Path:
        return self.task_dir_for(*args) / "task_report.json"

    def task_turbine_static_path_for(self, *args: str) -> Path:
        return self.task_dir_for(*args) / "static.parquet"

    def task_known_future_path_for(self, *args: str) -> Path:
        return self.task_dir_for(*args) / "known_future.parquet"

    def task_series_path_for(self, *args: str) -> Path:
        return self.task_dir_for(*args) / "series.parquet"

    def task_context_path_for(self, *args: str) -> Path:
        return self.task_dir_for(*args) / "task_context.json"

    def task_build_meta_path_for(self, *args: str) -> Path:
        return self.task_dir_for(*args) / "_build_meta.json"

    def _resolve_task_args(self, *args: str) -> tuple[str, str]:
        if len(args) == 2:
            task_id, feature_protocol_id = args
            return task_id, feature_protocol_id
        if len(args) == 3:
            quality_profile, granularity, task_id = args
            if quality_profile != "default" or granularity != "farm":
                raise ValueError(
                    "Legacy task cache paths only remain available for default/farm. "
                    f"Received quality_profile={quality_profile!r}, granularity={granularity!r}."
                )
            return task_id, "power_only"
        raise TypeError(
            "Task cache path helpers expect (task_id, feature_protocol_id) or legacy "
            "(quality_profile, granularity, task_id) arguments."
        )

    @staticmethod
    def _assert_legacy_gold_args(*, quality_profile: str, layout: str) -> None:
        if quality_profile != "default" or layout != "farm":
            raise ValueError(
                "gold_base is now a single farm-synchronous cache. Only legacy default/farm lookups remain supported."
            )


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
        manifest_build_meta_path=manifest_dir / "_build_meta.json",
        silver_dir=silver_dir,
        silver_continuous_dir=silver_dir / "continuous",
        silver_events_dir=silver_dir / "events",
        silver_shared_ts_dir=silver_dir / "shared_ts",
        silver_event_features_dir=silver_dir / "event_features",
        silver_interventions_dir=silver_dir / "interventions",
        silver_meta_dir=silver_dir / "meta",
        silver_build_meta_path=silver_dir / "_build_meta.json",
        gold_base_dir=gold_base_dir,
        tasks_dir=tasks_dir,
    )

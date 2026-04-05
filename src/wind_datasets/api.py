from __future__ import annotations

from pathlib import Path

import polars as pl

from .datasets import get_builder
from .manifest import build_manifest as _build_manifest
from .models import TaskSpec
from .registry import get_dataset_spec


def build_manifest(dataset_id: str, cache_root: str | Path = "cache") -> Path:
    spec = get_dataset_spec(dataset_id)
    return _build_manifest(spec, Path(cache_root))


def build_silver(dataset_id: str, cache_root: str | Path = "cache") -> Path:
    spec = get_dataset_spec(dataset_id)
    builder = get_builder(spec, Path(cache_root))
    return builder.build_silver()


def build_gold_base(
    dataset_id: str,
    cache_root: str | Path = "cache",
    quality_profile: str | None = None,
) -> Path:
    spec = get_dataset_spec(dataset_id)
    builder = get_builder(spec, Path(cache_root))
    return builder.build_gold_base(quality_profile=quality_profile)


def build_task_cache(
    dataset_id: str,
    task_spec: TaskSpec | None = None,
    cache_root: str | Path = "cache",
    quality_profile: str | None = None,
) -> Path:
    spec = get_dataset_spec(dataset_id)
    builder = get_builder(spec, Path(cache_root))
    return builder.build_task_cache(
        task_spec or TaskSpec.next_6h_from_24h(),
        quality_profile=quality_profile,
    )

def load_series(
    dataset_id: str,
    cache_root: str | Path = "cache",
    quality_profile: str | None = None,
) -> pl.DataFrame:
    spec = get_dataset_spec(dataset_id)
    builder = get_builder(spec, Path(cache_root))
    return builder.load_series(quality_profile=quality_profile)


def load_turbine_static(
    dataset_id: str,
    cache_root: str | Path = "cache",
) -> pl.DataFrame:
    spec = get_dataset_spec(dataset_id)
    builder = get_builder(spec, Path(cache_root))
    return builder.load_turbine_static()


def load_shared_timeseries(
    dataset_id: str,
    group_name: str,
    cache_root: str | Path = "cache",
) -> pl.DataFrame:
    spec = get_dataset_spec(dataset_id)
    builder = get_builder(spec, Path(cache_root))
    return builder.load_shared_timeseries(group_name)


def load_event_features(
    dataset_id: str,
    group_name: str,
    cache_root: str | Path = "cache",
) -> pl.DataFrame:
    spec = get_dataset_spec(dataset_id)
    builder = get_builder(spec, Path(cache_root))
    return builder.load_event_features(group_name)


def load_interventions(
    dataset_id: str,
    group_name: str,
    cache_root: str | Path = "cache",
) -> pl.DataFrame:
    spec = get_dataset_spec(dataset_id)
    builder = get_builder(spec, Path(cache_root))
    return builder.load_interventions(group_name)


def load_window_index(
    dataset_id: str,
    task_spec: TaskSpec | None = None,
    cache_root: str | Path = "cache",
    quality_profile: str | None = None,
) -> pl.DataFrame:
    spec = get_dataset_spec(dataset_id)
    builder = get_builder(spec, Path(cache_root))
    return builder.load_window_index(
        task_spec or TaskSpec.next_6h_from_24h(),
        quality_profile=quality_profile,
    )

def profile_dataset(
    dataset_id: str,
    cache_root: str | Path = "cache",
    quality_profile: str | None = None,
) -> dict:
    spec = get_dataset_spec(dataset_id)
    builder = get_builder(spec, Path(cache_root))
    return builder.profile_dataset(quality_profile=quality_profile)

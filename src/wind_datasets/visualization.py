from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from .api import build_gold_base
from .datasets import get_builder
from .feature_protocols import feature_protocol_task_blocked_reason, list_feature_protocol_ids
from .models import TaskBundlePaths, TaskSpec
from .paths import dataset_cache_paths
from .registry import get_dataset_spec
from .utils import read_json

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


_POWER_COLUMNS = ["dataset", "turbine_id", "timestamp", "target_kw", "is_observed", "quality_flags"]
_TASK_CONTEXT_REQUIRED_COLUMNS = ["dataset", "timestamp", "quality_flags", "feature_quality_flags"]
_SITE_COLUMNS = [
    "dataset",
    "turbine_id",
    "coord_x",
    "coord_y",
    "coord_kind",
    "coord_crs",
    "latitude",
    "longitude",
    "elevation_m",
    "rated_power_kw",
]
_EARTH_RADIUS_M = 6_371_000.0
_DEFAULT_TASK_SPEC = TaskSpec.next_6h_from_24h()
_IGNORED_NOTEBOOK_FEATURE_QUALITY_FLAGS = frozenset({"missing_past_covariates"})


@dataclass(frozen=True)
class PowerTile:
    dataset_id: str
    turbine_id: str
    start_timestamp: object
    end_timestamp: object
    total_points: int
    valid_points: int
    invalid_points: int
    padding_points: int
    tile_rows: int
    tile_cols: int
    min_valid_kw: float | None
    max_valid_kw: float | None
    value_grid: np.ndarray
    invalid_mask: np.ndarray
    padding_mask: np.ndarray

    def to_summary(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "turbine_id": self.turbine_id,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "total_points": self.total_points,
            "valid_points": self.valid_points,
            "invalid_points": self.invalid_points,
            "invalid_share": self.invalid_points / self.total_points if self.total_points else 0.0,
            "padding_points": self.padding_points,
            "tile_rows": self.tile_rows,
            "tile_cols": self.tile_cols,
            "min_valid_kw": self.min_valid_kw,
            "max_valid_kw": self.max_valid_kw,
        }


@dataclass(frozen=True)
class SiteLayout:
    dataset_id: str
    turbine_ids: tuple[str, ...]
    coordinate_mode: str
    coord_crs: str | None
    axis_label_x: str
    axis_label_y: str
    distance_unit: str
    turbine_count: int
    neighbor_k: int
    edge_pairs: tuple[tuple[int, int], ...]
    edge_lengths: tuple[float, ...]
    x: np.ndarray
    y: np.ndarray
    elevations_m: np.ndarray
    rated_power_kw: np.ndarray

    def to_summary(self) -> dict[str, object]:
        edge_lengths = np.asarray(self.edge_lengths, dtype=float)
        edge_count = len(self.edge_pairs)
        return {
            "dataset_id": self.dataset_id,
            "turbine_count": self.turbine_count,
            "coordinate_mode": self.coordinate_mode,
            "coord_crs": self.coord_crs,
            "neighbor_k": self.neighbor_k,
            "edge_count": edge_count,
            "min_edge_distance": float(edge_lengths.min()) if edge_count else None,
            "median_edge_distance": float(np.median(edge_lengths)) if edge_count else None,
            "max_edge_distance": float(edge_lengths.max()) if edge_count else None,
        }


@dataclass(frozen=True)
class ProtocolNotebookMetadata:
    dataset_id: str
    task_id: str
    feature_protocol_id: str
    display_name: str
    summary: str
    past_covariates: tuple[str, ...]
    derived_source_columns: tuple[str, ...]
    dataset_specific_notes: tuple[str, ...]

    def to_summary(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "task_id": self.task_id,
            "feature_protocol_id": self.feature_protocol_id,
            "display_name": self.display_name,
            "summary": self.summary,
            "past_covariates": list(self.past_covariates),
            "derived_source_columns": list(self.derived_source_columns),
            "dataset_specific_notes": list(self.dataset_specific_notes),
        }


@dataclass(frozen=True)
class FarmStatusTile:
    dataset_id: str
    task_id: str
    feature_protocol_id: str
    start_timestamp: object
    end_timestamp: object
    total_points: int
    clean_points: int
    target_issue_points: int
    feature_issue_points: int
    any_issue_points: int
    padding_points: int
    tile_rows: int
    tile_cols: int
    status_grid: np.ndarray
    target_issue_mask: np.ndarray
    feature_issue_mask: np.ndarray
    issue_mask: np.ndarray
    padding_mask: np.ndarray

    def to_summary(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "task_id": self.task_id,
            "feature_protocol_id": self.feature_protocol_id,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "total_points": self.total_points,
            "clean_points": self.clean_points,
            "target_issue_points": self.target_issue_points,
            "feature_issue_points": self.feature_issue_points,
            "any_issue_points": self.any_issue_points,
            "any_issue_share": self.any_issue_points / self.total_points if self.total_points else 0.0,
            "padding_points": self.padding_points,
            "tile_rows": self.tile_rows,
            "tile_cols": self.tile_cols,
        }


def resolve_turbine_selector(dataset_id: str, selector: int | str) -> str:
    turbine_ids = get_dataset_spec(dataset_id).turbine_ids
    if isinstance(selector, str):
        if selector in turbine_ids:
            return selector
        raise ValueError(f"Unknown turbine selector {selector!r} for dataset {dataset_id!r}.")
    if isinstance(selector, bool) or not isinstance(selector, int):
        raise TypeError("Turbine selector must be a 0-based int index or an exact turbine id string.")
    if selector < 0 or selector >= len(turbine_ids):
        raise ValueError(f"Turbine index {selector} is out of range for dataset {dataset_id!r}.")
    return turbine_ids[selector]


def load_turbine_power_series(
    dataset_id: str,
    selector: int | str,
    cache_root: str | Path = "cache",
) -> pl.DataFrame:
    """Load one turbine's canonical power series for visualization-only workflows."""
    cache_root_path = Path(cache_root)
    cache_paths = dataset_cache_paths(cache_root_path, dataset_id)
    gold_base_path = cache_paths.gold_base_series_path
    if not gold_base_path.exists():
        build_gold_base(dataset_id, cache_root=cache_root_path)

    turbine_id = resolve_turbine_selector(dataset_id, selector)
    frame = (
        pl.scan_parquet(gold_base_path)
        .select(_POWER_COLUMNS)
        .filter(pl.col("turbine_id") == turbine_id)
        .sort("timestamp")
        .collect()
    )
    if frame.is_empty():
        raise ValueError(f"No power rows were found for turbine {turbine_id!r} in dataset {dataset_id!r}.")
    return frame


def load_turbine_static_for_visualization(
    dataset_id: str,
    cache_root: str | Path = "cache",
) -> pl.DataFrame:
    """Load the canonical turbine static frame for visualization-only workflows."""
    from .api import load_turbine_static

    frame = load_turbine_static(dataset_id, cache_root=cache_root)
    missing_columns = [column for column in _SITE_COLUMNS if column not in frame.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Turbine static frame is missing required columns: {missing}.")
    if frame.is_empty():
        raise ValueError(f"Turbine static frame for dataset {dataset_id!r} is empty.")
    return frame.select(_SITE_COLUMNS).sort("turbine_id")


def list_supported_feature_protocol_ids_for_dataset(dataset_id: str) -> tuple[str, ...]:
    # Validate the dataset id up front so callers get a stable error message.
    get_dataset_spec(dataset_id)
    return tuple(
        feature_protocol_id
        for feature_protocol_id in list_feature_protocol_ids()
        if feature_protocol_task_blocked_reason(
            dataset_id=dataset_id,
            feature_protocol_id=feature_protocol_id,
        )
        is None
    )


def list_unsupported_feature_protocol_ids_for_dataset(dataset_id: str) -> tuple[str, ...]:
    get_dataset_spec(dataset_id)
    return tuple(
        feature_protocol_id
        for feature_protocol_id in list_feature_protocol_ids()
        if feature_protocol_task_blocked_reason(
            dataset_id=dataset_id,
            feature_protocol_id=feature_protocol_id,
        )
        is not None
    )


def load_task_context_for_visualization(
    dataset_id: str,
    feature_protocol_id: str,
    *,
    task_spec: TaskSpec | None = None,
    cache_root: str | Path = "cache",
) -> dict[str, Any]:
    task_paths = _ensure_task_bundle_paths(
        dataset_id,
        feature_protocol_id,
        task_spec=task_spec,
        cache_root=cache_root,
    )
    return read_json(task_paths.task_context_path)


def load_protocol_notebook_metadata(
    dataset_id: str,
    feature_protocol_id: str,
    *,
    task_spec: TaskSpec | None = None,
    cache_root: str | Path = "cache",
) -> ProtocolNotebookMetadata:
    task_context = load_task_context_for_visualization(
        dataset_id,
        feature_protocol_id,
        task_spec=task_spec,
        cache_root=cache_root,
    )
    protocol_context = _mapping_dict(task_context, "feature_protocol")
    column_groups = _mapping_dict(task_context, "column_groups")
    derived_source_columns: list[str] = []
    for transform in _sequence_dicts(protocol_context.get("derived_angle_features")):
        derived_source_columns.extend(_string_sequence(transform.get("source_columns")))
    for transform in _sequence_dicts(protocol_context.get("derived_scalar_features")):
        derived_source_columns.extend(_string_sequence(transform.get("source_columns")))
    return ProtocolNotebookMetadata(
        dataset_id=dataset_id,
        task_id=str(_mapping_dict(task_context, "task")["task_id"]),
        feature_protocol_id=feature_protocol_id,
        display_name=str(protocol_context["display_name"]),
        summary=str(protocol_context["summary"]),
        past_covariates=_string_sequence(column_groups.get("past_covariates")),
        derived_source_columns=tuple(dict.fromkeys(derived_source_columns)),
        dataset_specific_notes=_string_sequence(protocol_context.get("dataset_specific_notes")),
    )


def load_farm_status_timestamp_summary(
    dataset_id: str,
    feature_protocol_id: str,
    *,
    task_spec: TaskSpec | None = None,
    cache_root: str | Path = "cache",
) -> pl.DataFrame:
    task_paths = _ensure_task_bundle_paths(
        dataset_id,
        feature_protocol_id,
        task_spec=task_spec,
        cache_root=cache_root,
    )
    frame = (
        pl.scan_parquet(task_paths.series_path)
        .select(_TASK_CONTEXT_REQUIRED_COLUMNS)
        .sort("timestamp")
        .collect()
    )
    return build_farm_status_timestamp_summary(frame)


def build_farm_status_timestamp_summary(dataset_frame: pl.DataFrame) -> pl.DataFrame:
    missing_columns = [column for column in _TASK_CONTEXT_REQUIRED_COLUMNS if column not in dataset_frame.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Task series frame is missing required columns: {missing}.")
    if dataset_frame.is_empty():
        raise ValueError("Task series frame is empty.")

    frame = (
        dataset_frame.select(_TASK_CONTEXT_REQUIRED_COLUMNS)
        .with_columns(
            pl.col("quality_flags").fill_null("").alias("quality_flags"),
            pl.col("feature_quality_flags")
            .fill_null("")
            .map_elements(_strip_notebook_ignored_feature_quality_flags, return_dtype=pl.String)
            .alias("__visual_feature_quality_flags"),
        )
        .group_by(["dataset", "timestamp"])
        .agg(
            pl.len().alias("turbine_rows"),
            (pl.col("quality_flags") != "").cast(pl.Int32).sum().alias("target_issue_turbines"),
            (pl.col("__visual_feature_quality_flags") != "").cast(pl.Int32).sum().alias("feature_issue_turbines"),
        )
        .with_columns(
            (pl.col("target_issue_turbines") > 0).alias("has_target_issue"),
            (pl.col("feature_issue_turbines") > 0).alias("has_feature_issue"),
        )
        .with_columns(
            (pl.col("has_target_issue") | pl.col("has_feature_issue")).alias("has_any_issue"),
        )
        .sort(["dataset", "timestamp"])
    )
    return frame


def load_farm_status_tile(
    dataset_id: str,
    feature_protocol_id: str,
    *,
    task_spec: TaskSpec | None = None,
    cache_root: str | Path = "cache",
) -> FarmStatusTile:
    task_paths = _ensure_task_bundle_paths(
        dataset_id,
        feature_protocol_id,
        task_spec=task_spec,
        cache_root=cache_root,
    )
    task_context = read_json(task_paths.task_context_path)
    series_frame = (
        pl.scan_parquet(task_paths.series_path)
        .select(_TASK_CONTEXT_REQUIRED_COLUMNS)
        .sort("timestamp")
        .collect()
    )
    timestamp_summary = build_farm_status_timestamp_summary(series_frame)
    return build_farm_status_tile(
        timestamp_summary,
        task_id=str(_mapping_dict(task_context, "task")["task_id"]),
        feature_protocol_id=feature_protocol_id,
    )


def build_farm_status_tile(
    timestamp_summary: pl.DataFrame,
    *,
    task_id: str,
    feature_protocol_id: str,
) -> FarmStatusTile:
    required_columns = [
        "dataset",
        "timestamp",
        "has_target_issue",
        "has_feature_issue",
        "has_any_issue",
    ]
    missing_columns = [column for column in required_columns if column not in timestamp_summary.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Farm status summary is missing required columns: {missing}.")
    if timestamp_summary.is_empty():
        raise ValueError("Farm status summary is empty.")

    frame = timestamp_summary.sort("timestamp")
    dataset_id = frame["dataset"][0]
    total_points = frame.height
    tile_cols = max(1, ceil(sqrt(total_points)))
    tile_rows = ceil(total_points / tile_cols)
    padding_points = tile_rows * tile_cols - total_points

    status_grid = np.full((tile_rows, tile_cols), np.nan, dtype=float)
    target_issue_mask = np.zeros((tile_rows, tile_cols), dtype=bool)
    feature_issue_mask = np.zeros((tile_rows, tile_cols), dtype=bool)
    issue_mask = np.zeros((tile_rows, tile_cols), dtype=bool)
    padding_mask = np.ones((tile_rows, tile_cols), dtype=bool)

    target_issues = frame["has_target_issue"].to_list()
    feature_issues = frame["has_feature_issue"].to_list()
    any_issues = frame["has_any_issue"].to_list()

    for index, (has_target_issue, has_feature_issue, has_any_issue) in enumerate(
        zip(target_issues, feature_issues, any_issues, strict=True)
    ):
        row = index // tile_cols
        column = index % tile_cols
        padding_mask[row, column] = False
        target_issue_mask[row, column] = bool(has_target_issue)
        feature_issue_mask[row, column] = bool(has_feature_issue)
        issue_mask[row, column] = bool(has_any_issue)
        status_grid[row, column] = 0.0 if has_any_issue else 1.0

    any_issue_points = int(np.count_nonzero(issue_mask))
    target_issue_points = int(np.count_nonzero(target_issue_mask))
    feature_issue_points = int(np.count_nonzero(feature_issue_mask))

    return FarmStatusTile(
        dataset_id=dataset_id,
        task_id=task_id,
        feature_protocol_id=feature_protocol_id,
        start_timestamp=frame["timestamp"][0],
        end_timestamp=frame["timestamp"][-1],
        total_points=total_points,
        clean_points=total_points - any_issue_points,
        target_issue_points=target_issue_points,
        feature_issue_points=feature_issue_points,
        any_issue_points=any_issue_points,
        padding_points=padding_points,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        status_grid=status_grid,
        target_issue_mask=target_issue_mask,
        feature_issue_mask=feature_issue_mask,
        issue_mask=issue_mask,
        padding_mask=padding_mask,
    )


def build_power_tile(dataset_frame: pl.DataFrame) -> PowerTile:
    missing_columns = [column for column in _POWER_COLUMNS if column not in dataset_frame.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Power frame is missing required columns: {missing}.")
    if dataset_frame.is_empty():
        raise ValueError("Power frame is empty.")

    frame = dataset_frame.sort("timestamp")
    dataset_id = frame["dataset"][0]
    turbine_id = frame["turbine_id"][0]
    total_points = frame.height
    tile_cols = max(1, ceil(sqrt(total_points)))
    tile_rows = ceil(total_points / tile_cols)
    padding_points = tile_rows * tile_cols - total_points

    target_values = frame["target_kw"].to_list()
    invalid_series = (
        frame.select(
            (
                pl.col("target_kw").is_null()
                | (pl.col("quality_flags") != "")
                | (~pl.col("is_observed"))
            )
            .fill_null(True)
            .alias("is_invalid")
        )
        .to_series()
        .to_list()
    )

    value_grid = np.full((tile_rows, tile_cols), np.nan, dtype=float)
    invalid_mask = np.zeros((tile_rows, tile_cols), dtype=bool)
    padding_mask = np.ones((tile_rows, tile_cols), dtype=bool)

    for index, (target_kw, is_invalid) in enumerate(zip(target_values, invalid_series, strict=True)):
        row = index // tile_cols
        column = index % tile_cols
        padding_mask[row, column] = False
        invalid_mask[row, column] = bool(is_invalid)
        if not is_invalid and target_kw is not None:
            value_grid[row, column] = float(target_kw)

    valid_values = value_grid[~np.isnan(value_grid)]
    min_valid_kw = float(valid_values.min()) if valid_values.size else None
    max_valid_kw = float(valid_values.max()) if valid_values.size else None
    invalid_points = int(np.count_nonzero(invalid_mask))

    return PowerTile(
        dataset_id=dataset_id,
        turbine_id=turbine_id,
        start_timestamp=frame["timestamp"][0],
        end_timestamp=frame["timestamp"][-1],
        total_points=total_points,
        valid_points=total_points - invalid_points,
        invalid_points=invalid_points,
        padding_points=padding_points,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        min_valid_kw=min_valid_kw,
        max_valid_kw=max_valid_kw,
        value_grid=value_grid,
        invalid_mask=invalid_mask,
        padding_mask=padding_mask,
    )


def build_site_layout(
    static_frame: pl.DataFrame,
    *,
    neighbor_k: int = 2,
) -> SiteLayout:
    missing_columns = [column for column in _SITE_COLUMNS if column not in static_frame.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Static frame is missing required columns: {missing}.")
    if static_frame.is_empty():
        raise ValueError("Static frame is empty.")
    if neighbor_k < 1:
        raise ValueError("neighbor_k must be at least 1.")

    frame = static_frame.select(_SITE_COLUMNS).sort("turbine_id")
    dataset_id = frame["dataset"][0]
    turbine_ids = tuple(frame["turbine_id"].to_list())
    if len(set(turbine_ids)) != len(turbine_ids):
        raise ValueError("Static frame contains duplicate turbine_id values.")

    x, y, coordinate_mode, coord_crs, axis_label_x, axis_label_y, distance_unit = _resolve_plot_coordinates(frame)
    turbine_count = len(turbine_ids)
    effective_k = min(neighbor_k, max(turbine_count - 1, 0))
    edge_pairs, edge_lengths = _build_neighbor_edges(x, y, effective_k)

    elevations_m = (
        frame["elevation_m"].fill_null(float("nan")).cast(pl.Float64).to_numpy()
        if "elevation_m" in frame.columns
        else np.full(turbine_count, np.nan, dtype=float)
    )
    rated_power_kw = (
        frame["rated_power_kw"].fill_null(float("nan")).cast(pl.Float64).to_numpy()
        if "rated_power_kw" in frame.columns
        else np.full(turbine_count, np.nan, dtype=float)
    )

    return SiteLayout(
        dataset_id=dataset_id,
        turbine_ids=turbine_ids,
        coordinate_mode=coordinate_mode,
        coord_crs=coord_crs,
        axis_label_x=axis_label_x,
        axis_label_y=axis_label_y,
        distance_unit=distance_unit,
        turbine_count=turbine_count,
        neighbor_k=effective_k,
        edge_pairs=edge_pairs,
        edge_lengths=edge_lengths,
        x=x,
        y=y,
        elevations_m=elevations_m,
        rated_power_kw=rated_power_kw,
    )


def build_turbine_neighbor_table(
    layout: SiteLayout,
    selector: int | str,
    *,
    limit: int | None = None,
) -> pl.DataFrame:
    turbine_id = _resolve_layout_selector(layout, selector)
    focus_index = layout.turbine_ids.index(turbine_id)
    distances = np.hypot(layout.x - layout.x[focus_index], layout.y - layout.y[focus_index])
    order = np.argsort(distances)
    rows: list[dict[str, object]] = []
    rank = 0
    for index in order:
        if index == focus_index:
            continue
        rank += 1
        rows.append(
            {
                "dataset_id": layout.dataset_id,
                "turbine_id": turbine_id,
                "neighbor_rank": rank,
                "neighbor_turbine_id": layout.turbine_ids[index],
                "distance": float(distances[index]),
                "distance_unit": layout.distance_unit,
            }
        )
        if limit is not None and rank >= limit:
            break
    return pl.DataFrame(rows)


def plot_power_tile(
    tile: PowerTile,
    *,
    ax: Axes | None = None,
    cmap_name: str = "cividis",
    invalid_color: str = "#d73027",
    padding_color: str = "#ffffff",
    colorbar_label: str = "Mean power (kW)",
) -> tuple[Figure, Axes]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    figure: Figure
    if ax is None:
        figure, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    else:
        figure = ax.figure

    ax.set_facecolor(padding_color)
    valid_cmap = plt.get_cmap(cmap_name).copy()
    valid_cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    valid_ma = np.ma.masked_invalid(tile.value_grid)
    if tile.min_valid_kw is None or tile.max_valid_kw is None:
        vmin = 0.0
        vmax = 1.0
    else:
        vmin = tile.min_valid_kw
        vmax = tile.max_valid_kw
        if vmin == vmax:
            delta = max(abs(vmin) * 0.01, 1.0)
            vmin -= delta
            vmax += delta
    valid_image = ax.imshow(
        valid_ma,
        cmap=valid_cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )

    invalid_overlay = np.ma.masked_where(~tile.invalid_mask, np.ones_like(tile.value_grid))
    ax.imshow(
        invalid_overlay,
        cmap=ListedColormap([invalid_color]),
        interpolation="nearest",
        aspect="equal",
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        "\n".join(
            [
                f"{tile.dataset_id} | {tile.turbine_id}",
                (
                    f"{tile.start_timestamp:%Y-%m-%d %H:%M} -> {tile.end_timestamp:%Y-%m-%d %H:%M} | "
                    f"points={tile.total_points:,} | invalid={tile.invalid_points:,}"
                ),
            ]
        )
    )
    figure.colorbar(valid_image, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)
    return figure, ax


def plot_farm_status_tile(
    tile: FarmStatusTile,
    *,
    ax: Axes | None = None,
    clean_color: str = "#2ca25f",
    issue_color: str = "#d73027",
    padding_color: str = "#ffffff",
) -> tuple[Figure, Axes]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    figure: Figure
    if ax is None:
        figure, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    else:
        figure = ax.figure

    ax.set_facecolor(padding_color)
    image = np.ma.masked_invalid(tile.status_grid)
    cmap = ListedColormap([issue_color, clean_color])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    ax.imshow(
        image,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        aspect="equal",
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        "\n".join(
            [
                f"{tile.dataset_id} | {tile.feature_protocol_id}",
                (
                    f"{tile.start_timestamp:%Y-%m-%d %H:%M} -> {tile.end_timestamp:%Y-%m-%d %H:%M} | "
                    f"steps={tile.total_points:,} | red={tile.any_issue_points:,} | "
                    f"target={tile.target_issue_points:,} | feature={tile.feature_issue_points:,}"
                ),
            ]
        )
    )
    ax.legend(
        handles=[
            Patch(facecolor=clean_color, edgecolor="none", label="clean"),
            Patch(facecolor=issue_color, edgecolor="none", label="issue"),
            Patch(facecolor=padding_color, edgecolor="#d9d9d9", label="padding"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04),
        frameon=False,
        ncol=3,
    )
    return figure, ax


def plot_site_layout(
    layout: SiteLayout,
    *,
    ax: Axes | None = None,
    highlight_selector: int | str | None = None,
    edge_color: str = "#9aa1a6",
    node_color: str = "#2b6f97",
    highlight_color: str = "#d1495b",
    edge_alpha: float = 0.65,
    label_fontsize: int = 8,
    max_auto_labels: int = 40,
) -> tuple[Figure, Axes]:
    import matplotlib.pyplot as plt

    figure: Figure
    if ax is None:
        figure, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    else:
        figure = ax.figure

    if layout.edge_pairs:
        for left, right in layout.edge_pairs:
            ax.plot(
                [layout.x[left], layout.x[right]],
                [layout.y[left], layout.y[right]],
                color=edge_color,
                alpha=edge_alpha,
                linewidth=1.0,
                zorder=1,
            )

    ax.scatter(
        layout.x,
        layout.y,
        s=52,
        c=node_color,
        edgecolors="white",
        linewidths=0.8,
        zorder=2,
    )

    highlight_index: int | None = None
    if highlight_selector is not None:
        highlight_id = _resolve_layout_selector(layout, highlight_selector)
        highlight_index = layout.turbine_ids.index(highlight_id)
        ax.scatter(
            [layout.x[highlight_index]],
            [layout.y[highlight_index]],
            s=120,
            c=highlight_color,
            edgecolors="black",
            linewidths=1.2,
            zorder=3,
        )

    span_x = float(layout.x.max() - layout.x.min()) if layout.turbine_count else 0.0
    span_y = float(layout.y.max() - layout.y.min()) if layout.turbine_count else 0.0
    offset = max(span_x, span_y, 1.0) * 0.015
    if layout.turbine_count <= max_auto_labels:
        label_indices = set(range(layout.turbine_count))
    elif highlight_index is not None:
        distances = np.hypot(layout.x - layout.x[highlight_index], layout.y - layout.y[highlight_index])
        nearest = np.argsort(distances)[: min(8, layout.turbine_count)]
        label_indices = set(int(index) for index in nearest)
    else:
        label_indices = set()

    for index in sorted(label_indices):
        turbine_id = layout.turbine_ids[index]
        text_kwargs = {"fontsize": label_fontsize, "ha": "left", "va": "bottom"}
        if index == highlight_index:
            text_kwargs["fontweight"] = "bold"
        ax.text(layout.x[index] + offset, layout.y[index] + offset, turbine_id, zorder=4, **text_kwargs)

    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.1)
    ax.set_xlabel(layout.axis_label_x)
    ax.set_ylabel(layout.axis_label_y)
    title_lines = [f"{layout.dataset_id} site layout", f"k-nearest edges: {layout.neighbor_k} | mode={layout.coordinate_mode}"]
    if highlight_index is not None:
        title_lines.append(f"highlight: {layout.turbine_ids[highlight_index]}")
    ax.set_title("\n".join(title_lines))
    ax.grid(True, alpha=0.18, linewidth=0.6)
    return figure, ax


def _resolve_plot_coordinates(
    frame: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray, str, str | None, str, str, str]:
    projected_x = frame["coord_x"] if "coord_x" in frame.columns else None
    projected_y = frame["coord_y"] if "coord_y" in frame.columns else None
    if projected_x is not None and projected_y is not None:
        projected_x_nulls = int(projected_x.null_count())
        projected_y_nulls = int(projected_y.null_count())
        if projected_x_nulls == 0 and projected_y_nulls == 0:
            coord_kind = frame["coord_kind"].drop_nulls().unique().to_list()
            coord_crs_values = frame["coord_crs"].drop_nulls().unique().to_list()
            coordinate_mode = coord_kind[0] if coord_kind else "projected_xy"
            coord_crs = coord_crs_values[0] if coord_crs_values else None
            return (
                projected_x.cast(pl.Float64).to_numpy(),
                projected_y.cast(pl.Float64).to_numpy(),
                coordinate_mode,
                coord_crs,
                "coord_x",
                "coord_y",
                "source_units",
            )

    if "latitude" not in frame.columns or "longitude" not in frame.columns:
        raise ValueError("Static frame does not contain complete projected or geographic coordinates.")

    latitude = frame["latitude"]
    longitude = frame["longitude"]
    if latitude.null_count() > 0 or longitude.null_count() > 0:
        raise ValueError("Static frame does not contain complete projected or geographic coordinates.")

    lat = latitude.cast(pl.Float64).to_numpy()
    lon = longitude.cast(pl.Float64).to_numpy()
    lat0 = float(np.deg2rad(lat.mean()))
    lon0 = float(np.deg2rad(lon.mean()))
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    x = _EARTH_RADIUS_M * (lon_rad - lon0) * np.cos(lat0)
    y = _EARTH_RADIUS_M * (lat_rad - lat_rad.mean())
    coord_crs_values = frame["coord_crs"].drop_nulls().unique().to_list() if "coord_crs" in frame.columns else []
    coord_crs = coord_crs_values[0] if coord_crs_values else "EPSG:4326"
    return (
        x.astype(float),
        y.astype(float),
        "local_tangent_m",
        coord_crs,
        "local east-west offset (m)",
        "local north-south offset (m)",
        "m",
    )


def _build_neighbor_edges(
    x: np.ndarray,
    y: np.ndarray,
    neighbor_k: int,
) -> tuple[tuple[tuple[int, int], ...], tuple[float, ...]]:
    if neighbor_k <= 0 or len(x) <= 1:
        return (), ()

    points = np.column_stack([x, y])
    deltas = points[:, None, :] - points[None, :, :]
    distances = np.sqrt(np.sum(deltas**2, axis=2))
    np.fill_diagonal(distances, np.inf)

    edge_lengths: dict[tuple[int, int], float] = {}
    for index in range(len(x)):
        neighbors = np.argsort(distances[index])[:neighbor_k]
        for neighbor in neighbors:
            if not np.isfinite(distances[index, neighbor]):
                continue
            edge = tuple(sorted((int(index), int(neighbor))))
            edge_lengths[edge] = float(distances[index, neighbor])

    ordered_edges = tuple(sorted(edge_lengths))
    ordered_lengths = tuple(edge_lengths[edge] for edge in ordered_edges)
    return ordered_edges, ordered_lengths


def _resolve_layout_selector(layout: SiteLayout, selector: int | str) -> str:
    if isinstance(selector, str):
        if selector in layout.turbine_ids:
            return selector
        raise ValueError(f"Unknown turbine selector {selector!r} for layout dataset {layout.dataset_id!r}.")
    if isinstance(selector, bool) or not isinstance(selector, int):
        raise TypeError("Layout selector must be a 0-based int index or an exact turbine id string.")
    if selector < 0 or selector >= layout.turbine_count:
        raise ValueError(f"Turbine index {selector} is out of range for layout dataset {layout.dataset_id!r}.")
    return layout.turbine_ids[selector]


def _ensure_task_bundle_paths(
    dataset_id: str,
    feature_protocol_id: str,
    *,
    task_spec: TaskSpec | None,
    cache_root: str | Path,
) -> TaskBundlePaths:
    resolved_task_spec = task_spec or _DEFAULT_TASK_SPEC
    spec = get_dataset_spec(dataset_id)
    builder = get_builder(spec, Path(cache_root))
    builder.ensure_task_cache_fresh(
        resolved_task_spec,
        feature_protocol_id=feature_protocol_id,
    )
    return builder.task_bundle_paths(
        resolved_task_spec,
        feature_protocol_id=feature_protocol_id,
    )


def _strip_notebook_ignored_feature_quality_flags(value: str | None) -> str:
    if not value:
        return ""
    kept: list[str] = []
    seen: set[str] = set()
    for token in str(value).split("|"):
        normalized = token.strip()
        if not normalized or normalized in _IGNORED_NOTEBOOK_FEATURE_QUALITY_FLAGS or normalized in seen:
            continue
        kept.append(normalized)
        seen.add(normalized)
    return "|".join(kept)


def _mapping_dict(value: object, key: str | None = None) -> dict[str, Any]:
    candidate = value
    if key is not None:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"Expected mapping with key {key!r}.")
        candidate = value[key]
    if not isinstance(candidate, dict):
        raise ValueError("Expected mapping value.")
    return candidate


def _sequence_dicts(value: object) -> tuple[dict[str, Any], ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("Expected a sequence of mapping values.")
    items: list[dict[str, Any]] = []
    for entry in value:
        if not isinstance(entry, dict):
            raise ValueError("Expected mapping value inside sequence.")
        items.append(entry)
    return tuple(items)


def _string_sequence(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("Expected a sequence of strings.")
    items: list[str] = []
    for entry in value:
        if not isinstance(entry, str):
            raise ValueError("Expected a string value inside sequence.")
        items.append(entry)
    return tuple(items)

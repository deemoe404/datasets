from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from .api import build_gold_base
from .paths import dataset_cache_paths
from .registry import get_dataset_spec

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


_POWER_COLUMNS = ["dataset", "turbine_id", "timestamp", "target_kw", "is_observed", "quality_flags"]


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

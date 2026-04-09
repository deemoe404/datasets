from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import math
import sys
from typing import Mapping, Sequence

import numpy as np
import polars as pl


WINDOW_PROTOCOL_CHOICES = ("dense_sliding",)
DEFAULT_WINDOW_PROTOCOL = "dense_sliding"
SPLIT_PROTOCOL = "raw_chrono_70_10_20_strict_closed"
ROLLING_EVAL_PROTOCOL = "rolling_origin_no_refit"
NON_OVERLAP_EVAL_PROTOCOL = "non_overlap"
OVERALL_METRIC_SCOPE = "overall"
HORIZON_METRIC_SCOPE = "horizon"


@dataclass(frozen=True)
class WindowProtocolSpec:
    name: str
    label: str
    task_id: str
    stride_duration: str | None


@dataclass(frozen=True)
class SplitBoundary:
    split_name: str
    start_us: int
    end_us: int


@dataclass(frozen=True)
class WindowDescriptorIndex:
    turbine_indices: np.ndarray
    target_indices: np.ndarray
    output_start_us: np.ndarray
    output_end_us: np.ndarray

    def __len__(self) -> int:
        return int(self.target_indices.shape[0])

    @classmethod
    def empty(cls) -> "WindowDescriptorIndex":
        return cls(
            turbine_indices=np.empty((0,), dtype=np.int32),
            target_indices=np.empty((0,), dtype=np.int32),
            output_start_us=np.empty((0,), dtype=np.int64),
            output_end_us=np.empty((0,), dtype=np.int64),
        )


_WINDOW_PROTOCOLS = {
    "dense_sliding": WindowProtocolSpec(
        name="dense_sliding",
        label="Dense sliding",
        task_id="next_6h_from_24h",
        stride_duration=None,
    ),
}

_DEFAULT_OUTPUT_FILENAMES = {
    "chronos-2": {
        "dense_sliding": "chronos-2.csv",
    },
    "chronos-2-exogenous": {
        "dense_sliding": "chronos-2-exogenous.csv",
    },
}


def resolve_window_protocol(window_protocol: str) -> WindowProtocolSpec:
    try:
        return _WINDOW_PROTOCOLS[window_protocol]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported window_protocol {window_protocol!r}. Expected one of {WINDOW_PROTOCOL_CHOICES!r}."
        ) from exc


def build_task_spec(window_protocol: str, *, granularity: str = "turbine"):
    src_path = Path(__file__).resolve().parents[2] / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from wind_datasets import TaskSpec

    protocol = resolve_window_protocol(window_protocol)
    return TaskSpec(
        task_id=protocol.task_id,
        history_duration="24h",
        forecast_duration="6h",
        stride_duration=protocol.stride_duration,
        granularity=granularity,
    )


def default_output_filename(experiment_name: str, window_protocol: str) -> str:
    try:
        return _DEFAULT_OUTPUT_FILENAMES[experiment_name][window_protocol]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported experiment/window protocol combination: {experiment_name!r} / {window_protocol!r}."
        ) from exc


def default_output_path(*, repo_root: Path, experiment_name: str, window_protocol: str) -> Path:
    return repo_root / "experiment" / default_output_filename(experiment_name, window_protocol)


def build_chrono_split_lookup(raw_timestamps: Sequence[datetime]) -> pl.DataFrame:
    unique_sorted = sorted(dict.fromkeys(raw_timestamps))
    total = len(unique_sorted)
    train_count = math.floor(total * 0.7)
    val_count = math.floor(total * 0.1)
    test_count = total - train_count - val_count
    if min(train_count, val_count, test_count) <= 0:
        raise ValueError(
            f"Chronological split {SPLIT_PROTOCOL!r} requires non-empty train/val/test, found "
            f"{train_count}/{val_count}/{test_count}."
        )
    return pl.DataFrame(
        {
            "timestamp": unique_sorted,
            "split": (
                ["train"] * train_count
                + ["val"] * val_count
                + ["test"] * test_count
            ),
        }
    )


def build_split_boundaries(raw_timestamps: Sequence[datetime]) -> dict[str, SplitBoundary]:
    lookup = build_chrono_split_lookup(raw_timestamps)
    boundaries: dict[str, SplitBoundary] = {}
    grouped = (
        lookup.group_by("split")
        .agg(
            pl.col("timestamp").min().alias("start_ts"),
            pl.col("timestamp").max().alias("end_ts"),
        )
        .sort("split")
    )
    for row in grouped.iter_rows(named=True):
        split_name = str(row["split"])
        boundaries[split_name] = SplitBoundary(
            split_name=split_name,
            start_us=int(pl.Series([row["start_ts"]], dtype=pl.Datetime).cast(pl.Int64)[0]),
            end_us=int(pl.Series([row["end_ts"]], dtype=pl.Datetime).cast(pl.Int64)[0]),
        )
    return boundaries


def split_window_index(
    window_index: pl.DataFrame,
    *,
    raw_timestamps: Sequence[datetime],
    resolution_minutes: int,
    history_steps: int,
    max_windows_per_split: int | None = None,
) -> dict[str, pl.DataFrame]:
    step_us = resolution_minutes * 60 * 1_000_000
    boundaries = build_split_boundaries(raw_timestamps)
    frames = window_index.with_columns(
        pl.col("output_start_ts").cast(pl.Int64).alias("output_start_us"),
        pl.col("output_end_ts").cast(pl.Int64).alias("output_end_us"),
        (pl.col("output_start_ts").cast(pl.Int64) - history_steps * step_us).alias("input_start_us"),
    ).sort(["output_start_ts", "turbine_id"])
    split_frames: dict[str, pl.DataFrame] = {}
    for split_name in ("train", "val", "test"):
        boundary = boundaries[split_name]
        split_frame = frames.filter(
            (pl.col("input_start_us") >= boundary.start_us)
            & (pl.col("output_end_us") <= boundary.end_us)
        )
        if max_windows_per_split is not None:
            split_frame = split_frame.head(max_windows_per_split)
        if split_frame.is_empty():
            raise ValueError(f"Split {split_name!r} is empty after window selection.")
        split_frames[split_name] = split_frame
    return split_frames


def build_window_descriptor_index(
    window_index: pl.DataFrame,
    *,
    turbine_ids: Sequence[str],
    timestamps_by_turbine: Mapping[str, Sequence[int] | np.ndarray],
) -> WindowDescriptorIndex:
    if window_index.is_empty():
        return WindowDescriptorIndex.empty()

    turbine_order = {turbine_id: index for index, turbine_id in enumerate(turbine_ids)}
    timestamp_indices = {
        turbine_id: {
            int(timestamp): index
            for index, timestamp in enumerate(np.asarray(timestamps_by_turbine[turbine_id], dtype=np.int64).tolist())
        }
        for turbine_id in turbine_ids
    }
    working = window_index.with_columns(
        pl.col("output_start_ts").cast(pl.Int64).alias("output_start_us"),
        pl.col("output_end_ts").cast(pl.Int64).alias("output_end_us"),
    )
    turbine_indices: list[int] = []
    target_indices: list[int] = []
    output_start_values: list[int] = []
    output_end_values: list[int] = []

    for row in working.iter_rows(named=True):
        turbine_id = str(row["turbine_id"])
        output_start_us = int(row["output_start_us"])
        output_end_us = int(row["output_end_us"])
        try:
            target_index = timestamp_indices[turbine_id][output_start_us]
        except KeyError as exc:
            raise KeyError(
                f"Output start timestamp {output_start_us!r} is missing for turbine {turbine_id!r}."
            ) from exc
        turbine_indices.append(int(turbine_order[turbine_id]))
        target_indices.append(target_index)
        output_start_values.append(output_start_us)
        output_end_values.append(output_end_us)

    return WindowDescriptorIndex(
        turbine_indices=np.asarray(turbine_indices, dtype=np.int32),
        target_indices=np.asarray(target_indices, dtype=np.int32),
        output_start_us=np.asarray(output_start_values, dtype=np.int64),
        output_end_us=np.asarray(output_end_values, dtype=np.int64),
    )


def thin_non_overlap_window_index(
    windows: WindowDescriptorIndex,
    *,
    turbine_ids: Sequence[str],
    forecast_steps: int,
) -> WindowDescriptorIndex:
    if len(windows) == 0:
        return WindowDescriptorIndex.empty()
    keep_positions: list[int] = []
    for turbine_index, _ in enumerate(turbine_ids):
        turbine_positions = np.flatnonzero(windows.turbine_indices == turbine_index)
        keep_positions.extend(turbine_positions[::forecast_steps].tolist())
    if not keep_positions:
        return WindowDescriptorIndex.empty()
    keep_array = np.asarray(sorted(keep_positions), dtype=np.int64)
    return WindowDescriptorIndex(
        turbine_indices=windows.turbine_indices[keep_array],
        target_indices=windows.target_indices[keep_array],
        output_start_us=windows.output_start_us[keep_array],
        output_end_us=windows.output_end_us[keep_array],
    )


__all__ = [
    "DEFAULT_WINDOW_PROTOCOL",
    "WINDOW_PROTOCOL_CHOICES",
    "SPLIT_PROTOCOL",
    "ROLLING_EVAL_PROTOCOL",
    "NON_OVERLAP_EVAL_PROTOCOL",
    "OVERALL_METRIC_SCOPE",
    "HORIZON_METRIC_SCOPE",
    "WindowProtocolSpec",
    "SplitBoundary",
    "WindowDescriptorIndex",
    "build_task_spec",
    "default_output_filename",
    "default_output_path",
    "resolve_window_protocol",
    "build_chrono_split_lookup",
    "build_split_boundaries",
    "split_window_index",
    "build_window_descriptor_index",
    "thin_non_overlap_window_index",
]

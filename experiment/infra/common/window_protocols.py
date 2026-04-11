from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Sequence

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
class _SplitBoundary:
    split_name: str
    start_us: int
    end_us: int


_WINDOW_PROTOCOLS = {
    "dense_sliding": WindowProtocolSpec(
        name="dense_sliding",
        label="Dense sliding",
        task_id="next_6h_from_24h",
        stride_duration=None,
    ),
}


def resolve_window_protocol(window_protocol: str) -> WindowProtocolSpec:
    try:
        return _WINDOW_PROTOCOLS[window_protocol]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported window_protocol {window_protocol!r}. Expected one of {WINDOW_PROTOCOL_CHOICES!r}."
        ) from exc


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


def _build_split_boundaries(raw_timestamps: Sequence[datetime]) -> dict[str, _SplitBoundary]:
    lookup = build_chrono_split_lookup(raw_timestamps)
    boundaries: dict[str, _SplitBoundary] = {}
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
        boundaries[split_name] = _SplitBoundary(
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
    boundaries = _build_split_boundaries(raw_timestamps)
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


__all__ = [
    "DEFAULT_WINDOW_PROTOCOL",
    "WINDOW_PROTOCOL_CHOICES",
    "SPLIT_PROTOCOL",
    "ROLLING_EVAL_PROTOCOL",
    "NON_OVERLAP_EVAL_PROTOCOL",
    "OVERALL_METRIC_SCOPE",
    "HORIZON_METRIC_SCOPE",
    "WindowProtocolSpec",
    "resolve_window_protocol",
    "build_chrono_split_lookup",
    "split_window_index",
]

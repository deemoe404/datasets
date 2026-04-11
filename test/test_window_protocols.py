from __future__ import annotations

from datetime import datetime, timedelta
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import polars as pl
import pytest


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "experiment"
        / "infra"
        / "common"
        / "window_protocols.py"
    )
    spec = spec_from_file_location("window_protocols", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _raw_timestamps(count: int) -> list[datetime]:
    start = datetime(2024, 1, 1, 0, 0, 0)
    return [start + timedelta(minutes=10 * index) for index in range(count)]


def test_dense_sliding_protocol_contract() -> None:
    module = _load_module()

    protocol = module.resolve_window_protocol("dense_sliding")

    assert module.WINDOW_PROTOCOL_CHOICES == ("dense_sliding",)
    assert module.DEFAULT_WINDOW_PROTOCOL == "dense_sliding"
    assert protocol.task_id == "next_6h_from_24h"
    assert protocol.stride_duration is None


def test_legacy_window_protocol_is_rejected() -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="Unsupported window_protocol"):
        module.resolve_window_protocol("legacy_6h_stride")


def test_build_chrono_split_lookup_uses_unique_raw_timestamps() -> None:
    module = _load_module()
    timestamps = _raw_timestamps(10)
    raw_timestamps = timestamps[:5] + timestamps[2:7] + timestamps[7:]

    lookup = module.build_chrono_split_lookup(raw_timestamps)

    assert lookup.height == 10
    assert lookup["timestamp"].to_list() == timestamps
    assert lookup["split"].to_list() == ["train"] * 7 + ["val"] + ["test"] * 2


def test_split_window_index_enforces_strict_contained_windows() -> None:
    module = _load_module()
    raw_timestamps = _raw_timestamps(50)
    window_index = pl.DataFrame(
        {
            "dataset": ["demo"] * 5,
            "turbine_id": ["T01"] * 5,
            "output_start_ts": [
                raw_timestamps[2],
                raw_timestamps[35],
                raw_timestamps[37],
                raw_timestamps[40],
                raw_timestamps[42],
            ],
            "output_end_ts": [
                raw_timestamps[3],
                raw_timestamps[36],
                raw_timestamps[38],
                raw_timestamps[41],
                raw_timestamps[43],
            ],
            "is_complete_input": [True] * 5,
            "is_complete_output": [True] * 5,
            "quality_flags": [0] * 5,
        }
    )

    split_frames = module.split_window_index(
        window_index,
        raw_timestamps=raw_timestamps,
        resolution_minutes=10,
        history_steps=2,
    )

    assert split_frames["train"].select(["output_start_ts", "output_end_ts"]).to_dicts() == [
        {"output_start_ts": raw_timestamps[2], "output_end_ts": raw_timestamps[3]}
    ]
    assert split_frames["val"].select(["output_start_ts", "output_end_ts"]).to_dicts() == [
        {"output_start_ts": raw_timestamps[37], "output_end_ts": raw_timestamps[38]}
    ]
    assert split_frames["test"].select(["output_start_ts", "output_end_ts"]).to_dicts() == [
        {"output_start_ts": raw_timestamps[42], "output_end_ts": raw_timestamps[43]}
    ]

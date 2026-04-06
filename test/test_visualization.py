from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl
import pytest

from wind_datasets.visualization import build_power_tile, resolve_turbine_selector


def test_resolve_turbine_selector_accepts_index_and_exact_id() -> None:
    assert resolve_turbine_selector("kelmarsh", 0) == "Kelmarsh 1"
    assert resolve_turbine_selector("sdwpf_kddcup", "134") == "134"


def test_resolve_turbine_selector_rejects_out_of_range_and_unknown() -> None:
    with pytest.raises(ValueError):
        resolve_turbine_selector("kelmarsh", 6)

    with pytest.raises(ValueError):
        resolve_turbine_selector("hill_of_towie", "T99")


def test_build_power_tile_marks_invalid_points_from_target_and_quality_fields() -> None:
    frame = pl.DataFrame(
        {
            "dataset": ["sample"] * 5,
            "turbine_id": ["T01"] * 5,
            "timestamp": [
                datetime(2024, 1, 1, 0, 0),
                datetime(2024, 1, 1, 0, 10),
                datetime(2024, 1, 1, 0, 20),
                datetime(2024, 1, 1, 0, 30),
                datetime(2024, 1, 1, 0, 40),
            ],
            "target_kw": [10.0, -1.0, None, 12.0, 13.0],
            "is_observed": [True, True, True, True, False],
            "quality_flags": ["", "", "", "missing_target", ""],
        }
    )

    tile = build_power_tile(frame)

    assert tile.total_points == 5
    assert tile.invalid_points == 3
    assert tile.valid_points == 2
    assert tile.padding_points == 1
    assert np.count_nonzero(tile.invalid_mask) == 3
    assert tile.value_grid[0, 1] == -1.0
    assert tile.min_valid_kw == -1.0
    assert tile.max_valid_kw == 10.0


def test_build_power_tile_sorts_by_timestamp_and_tracks_padding_separately() -> None:
    frame = pl.DataFrame(
        {
            "dataset": ["sample"] * 5,
            "turbine_id": ["T01"] * 5,
            "timestamp": [
                datetime(2024, 1, 1, 0, 20),
                datetime(2024, 1, 1, 0, 0),
                datetime(2024, 1, 1, 0, 40),
                datetime(2024, 1, 1, 0, 10),
                datetime(2024, 1, 1, 0, 30),
            ],
            "target_kw": [3.0, 1.0, 5.0, 2.0, 4.0],
            "is_observed": [True] * 5,
            "quality_flags": [""] * 5,
        }
    )

    tile = build_power_tile(frame)

    assert tile.tile_rows == 2
    assert tile.tile_cols == 3
    assert tile.padding_points == 1
    assert tile.padding_mask[-1, -1]
    assert not tile.invalid_mask[-1, -1]
    assert tile.value_grid[0, 0] == 1.0
    assert tile.value_grid[0, 1] == 2.0
    assert tile.value_grid[0, 2] == 3.0
    assert tile.value_grid[1, 0] == 4.0
    assert tile.value_grid[1, 1] == 5.0

from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl
import pytest

from wind_datasets.visualization import (
    build_power_tile,
    build_site_layout,
    build_turbine_neighbor_table,
    plot_site_layout,
    resolve_turbine_selector,
)


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


def test_build_site_layout_falls_back_to_local_tangent_for_geographic_coords() -> None:
    frame = pl.DataFrame(
        {
            "dataset": ["sample"] * 3,
            "turbine_id": ["T03", "T01", "T02"],
            "coord_x": [None, None, None],
            "coord_y": [None, None, None],
            "coord_kind": ["geographic_latlon"] * 3,
            "coord_crs": ["EPSG:4326"] * 3,
            "latitude": [52.0010, 52.0000, 52.0000],
            "longitude": [0.0000, 0.0000, 0.0010],
            "elevation_m": [120.0, 100.0, 110.0],
            "rated_power_kw": [2300.0, 2300.0, 2300.0],
        }
    )

    layout = build_site_layout(frame, neighbor_k=2)

    assert layout.dataset_id == "sample"
    assert layout.turbine_ids == ("T01", "T02", "T03")
    assert layout.coordinate_mode == "local_tangent_m"
    assert layout.coord_crs == "EPSG:4326"
    assert layout.distance_unit == "m"
    assert layout.neighbor_k == 2
    assert len(layout.edge_pairs) == 3
    assert np.all(np.isfinite(layout.x))
    assert np.all(np.isfinite(layout.y))
    assert layout.to_summary()["median_edge_distance"] is not None

    neighbor_table = build_turbine_neighbor_table(layout, "T01", limit=2)
    assert neighbor_table["neighbor_turbine_id"].to_list() == ["T02", "T03"]
    assert neighbor_table["neighbor_rank"].to_list() == [1, 2]
    assert neighbor_table["distance_unit"].unique().to_list() == ["m"]


def test_build_site_layout_prefers_projected_coordinates_and_keeps_unique_edges() -> None:
    frame = pl.DataFrame(
        {
            "dataset": ["sample"] * 3,
            "turbine_id": ["T01", "T02", "T03"],
            "coord_x": [0.0, 2.0, 5.0],
            "coord_y": [0.0, 0.0, 0.0],
            "coord_kind": ["projected_xy"] * 3,
            "coord_crs": ["unknown_unverified"] * 3,
            "latitude": [None, None, None],
            "longitude": [None, None, None],
            "elevation_m": [None, None, None],
            "rated_power_kw": [1500.0, 1500.0, 1500.0],
        }
    )

    layout = build_site_layout(frame, neighbor_k=1)

    assert layout.coordinate_mode == "projected_xy"
    assert layout.distance_unit == "source_units"
    assert layout.edge_pairs == ((0, 1), (1, 2))
    assert layout.edge_lengths == (2.0, 3.0)

    neighbor_table = build_turbine_neighbor_table(layout, 0, limit=1)
    assert neighbor_table["neighbor_turbine_id"].to_list() == ["T02"]
    assert neighbor_table["distance"].to_list() == [2.0]


def test_build_site_layout_rejects_incomplete_coordinate_sources() -> None:
    frame = pl.DataFrame(
        {
            "dataset": ["sample"] * 2,
            "turbine_id": ["T01", "T02"],
            "coord_x": [1.0, None],
            "coord_y": [0.0, None],
            "coord_kind": ["projected_xy", "projected_xy"],
            "coord_crs": ["unknown_unverified", "unknown_unverified"],
            "latitude": [None, None],
            "longitude": [None, None],
            "elevation_m": [None, None],
            "rated_power_kw": [1500.0, 1500.0],
        }
    )

    with pytest.raises(ValueError, match="complete projected or geographic coordinates"):
        build_site_layout(frame, neighbor_k=1)


def test_plot_site_layout_smoke() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    frame = pl.DataFrame(
        {
            "dataset": ["sample"] * 3,
            "turbine_id": ["T01", "T02", "T03"],
            "coord_x": [0.0, 1.0, 2.0],
            "coord_y": [0.0, 0.5, 0.0],
            "coord_kind": ["projected_xy"] * 3,
            "coord_crs": ["unknown_unverified"] * 3,
            "latitude": [None, None, None],
            "longitude": [None, None, None],
            "elevation_m": [None, None, None],
            "rated_power_kw": [1500.0, 1500.0, 1500.0],
        }
    )

    layout = build_site_layout(frame, neighbor_k=1)
    figure, axis = plot_site_layout(layout, highlight_selector="T02")
    figure.canvas.draw()

    assert axis.get_xlabel() == "coord_x"
    assert "highlight: T02" in axis.get_title()

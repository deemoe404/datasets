from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from wind_datasets.models import TaskBundlePaths
from wind_datasets.visualization import (
    ProtocolNotebookMetadata,
    build_farm_status_tile,
    build_farm_status_timestamp_summary,
    build_power_tile,
    build_site_layout,
    build_turbine_neighbor_table,
    list_supported_feature_protocol_ids_for_dataset,
    plot_farm_status_tile,
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


def test_list_supported_feature_protocol_ids_for_dataset_skips_unsupported_sdwpf_lrpm() -> None:
    supported = list_supported_feature_protocol_ids_for_dataset("sdwpf_kddcup")

    assert "power_wd_yaw_lrpm_hist_sincos" not in supported
    assert supported == (
        "power_only",
        "power_ws_hist",
        "power_atemp_hist",
        "power_itemp_hist",
        "power_wd_hist_sincos",
        "power_ws_wd_hist_sincos",
        "power_wd_yaw_hist_sincos",
        "power_wd_yaw_pitchmean_hist_sincos",
    )


def test_build_farm_status_timestamp_summary_collapses_multi_turbine_rows() -> None:
    frame = pl.DataFrame(
        {
            "dataset": ["sample"] * 6,
            "timestamp": [
                datetime(2024, 1, 1, 0, 10),
                datetime(2024, 1, 1, 0, 0),
                datetime(2024, 1, 1, 0, 0),
                datetime(2024, 1, 1, 0, 20),
                datetime(2024, 1, 1, 0, 10),
                datetime(2024, 1, 1, 0, 20),
            ],
            "quality_flags": [
                "",
                "missing_target",
                "",
                "",
                "",
                "",
            ],
            "feature_quality_flags": [
                "",
                "",
                "",
                "feature_source_conflict__grid",
                "",
                "",
            ],
        }
    )

    summary = build_farm_status_timestamp_summary(frame)

    assert summary["timestamp"].to_list() == [
        datetime(2024, 1, 1, 0, 0),
        datetime(2024, 1, 1, 0, 10),
        datetime(2024, 1, 1, 0, 20),
    ]
    assert summary["turbine_rows"].to_list() == [2, 2, 2]
    assert summary["target_issue_turbines"].to_list() == [1, 0, 0]
    assert summary["feature_issue_turbines"].to_list() == [0, 0, 1]
    assert summary["has_any_issue"].to_list() == [True, False, True]


def test_build_farm_status_timestamp_summary_ignores_warmup_only_feature_flags() -> None:
    frame = pl.DataFrame(
        {
            "dataset": ["sample"] * 4,
            "timestamp": [
                datetime(2024, 1, 1, 0, 0),
                datetime(2024, 1, 1, 0, 0),
                datetime(2024, 1, 1, 0, 10),
                datetime(2024, 1, 1, 0, 10),
            ],
            "quality_flags": ["", "", "", ""],
            "feature_quality_flags": [
                "missing_past_covariates",
                "",
                "missing_past_covariates|feature_source_conflict__grid",
                "",
            ],
        }
    )

    summary = build_farm_status_timestamp_summary(frame)

    assert summary["feature_issue_turbines"].to_list() == [0, 1]
    assert summary["has_feature_issue"].to_list() == [False, True]
    assert summary["has_any_issue"].to_list() == [False, True]


def test_build_farm_status_tile_tracks_issue_masks_and_padding() -> None:
    summary = pl.DataFrame(
        {
            "dataset": ["sample"] * 5,
            "timestamp": [
                datetime(2024, 1, 1, 0, 20),
                datetime(2024, 1, 1, 0, 0),
                datetime(2024, 1, 1, 0, 40),
                datetime(2024, 1, 1, 0, 10),
                datetime(2024, 1, 1, 0, 30),
            ],
            "has_target_issue": [False, True, False, False, True],
            "has_feature_issue": [False, False, False, True, True],
            "has_any_issue": [False, True, False, True, True],
        }
    )

    tile = build_farm_status_tile(
        summary,
        task_id="next_6h_from_24h",
        feature_protocol_id="power_ws_hist",
    )

    assert tile.tile_rows == 2
    assert tile.tile_cols == 3
    assert tile.padding_points == 1
    assert tile.any_issue_points == 3
    assert tile.clean_points == 2
    assert tile.target_issue_points == 2
    assert tile.feature_issue_points == 2
    assert tile.status_grid[0, 0] == 0.0
    assert tile.status_grid[0, 1] == 0.0
    assert tile.status_grid[0, 2] == 1.0
    assert tile.status_grid[1, 0] == 0.0
    assert tile.status_grid[1, 1] == 1.0
    assert np.isnan(tile.status_grid[1, 2])
    assert tile.padding_mask[1, 2]


def test_plot_farm_status_tile_smoke() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    summary = pl.DataFrame(
        {
            "dataset": ["sample"] * 4,
            "timestamp": [
                datetime(2024, 1, 1, 0, 0),
                datetime(2024, 1, 1, 0, 10),
                datetime(2024, 1, 1, 0, 20),
                datetime(2024, 1, 1, 0, 30),
            ],
            "has_target_issue": [False, True, False, False],
            "has_feature_issue": [False, False, True, False],
            "has_any_issue": [False, True, True, False],
        }
    )
    tile = build_farm_status_tile(
        summary,
        task_id="next_6h_from_24h",
        feature_protocol_id="power_wd_yaw_hist_sincos",
    )

    figure, axis = plot_farm_status_tile(tile)
    figure.canvas.draw()

    assert "power_wd_yaw_hist_sincos" in axis.get_title()
    assert len(axis.get_legend().texts) == 3


def test_notebook_generator_builds_one_site_section_and_one_pair_per_protocol() -> None:
    generator_module = _load_notebook_generator_module()
    protocol_metadata = (
        ProtocolNotebookMetadata(
            dataset_id="toy_dataset",
            task_id="next_6h_from_24h",
            feature_protocol_id="power_only",
            display_name="Power Only",
            summary="Use only target power history as model input.",
            past_covariates=(),
            derived_source_columns=(),
            dataset_specific_notes=(),
        ),
        ProtocolNotebookMetadata(
            dataset_id="toy_dataset",
            task_id="next_6h_from_24h",
            feature_protocol_id="power_wd_yaw_hist_sincos",
            display_name="Power + Wind Direction/Yaw Error History (Sin/Cos)",
            summary="Use target power history plus direction-aware covariates.",
            past_covariates=("wind_direction_sin", "wind_direction_cos", "yaw_error_sin", "yaw_error_cos"),
            derived_source_columns=("Wind direction (°)", "Nacelle position (°)"),
            dataset_specific_notes=("Angle transform uses repository yaw-error convention.",),
        ),
    )

    payload = generator_module.build_dataset_notebook_payload(
        "toy_dataset",
        official_name="Toy Dataset",
        task_id="next_6h_from_24h",
        protocol_metadata=protocol_metadata,
        omitted_protocol_ids=("power_wd_yaw_lrpm_hist_sincos",),
    )

    cells = payload["cells"]
    assert payload["nbformat"] == 4
    assert len(cells) == 8
    assert cells[0]["cell_type"] == "markdown"
    assert "Unsupported protocol sections omitted" in "".join(cells[0]["source"])
    assert cells[2]["cell_type"] == "markdown"
    assert "## Site Layout" in "".join(cells[2]["source"])
    assert "load_turbine_static_for_visualization" in "".join(cells[3]["source"])
    assert "## Power Only" in "".join(cells[4]["source"])
    assert "Protocol-covered task columns: _none_" in "".join(cells[4]["source"])
    assert "## Power + Wind Direction/Yaw Error History (Sin/Cos)" in "".join(cells[6]["source"])
    assert "`wind_direction_sin`" in "".join(cells[6]["source"])
    assert "`Wind direction (°)`" in "".join(cells[6]["source"])
    assert "load_farm_status_tile" in "".join(cells[7]["source"])


def test_load_farm_status_tile_reads_series_and_context_without_requiring_turbine_id_column(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    series_path = tmp_path / "series.parquet"
    task_context_path = tmp_path / "task_context.json"
    pl.DataFrame(
        {
            "dataset": ["toy"] * 4,
            "turbine_id": ["T02", "T01", "T02", "T01"],
            "timestamp": [
                datetime(2024, 1, 1, 0, 10),
                datetime(2024, 1, 1, 0, 0),
                datetime(2024, 1, 1, 0, 0),
                datetime(2024, 1, 1, 0, 10),
            ],
            "quality_flags": ["", "missing_target", "", ""],
            "feature_quality_flags": ["", "", "", "missing_past_covariates|feature_source_conflict__grid"],
        }
    ).write_parquet(series_path)
    task_context_path.write_text('{"task":{"task_id":"next_6h_from_24h"}}', encoding="utf-8")

    paths = TaskBundlePaths(
        dataset_id="toy",
        task_id="next_6h_from_24h",
        feature_protocol_id="power_ws_hist",
        task_dir=tmp_path,
        series_path=series_path,
        known_future_path=tmp_path / "known_future.parquet",
        static_path=tmp_path / "static.parquet",
        window_index_path=tmp_path / "window_index.parquet",
        task_context_path=task_context_path,
        task_report_path=tmp_path / "task_report.json",
        build_meta_path=tmp_path / "_build_meta.json",
    )

    monkeypatch.setattr(
        "wind_datasets.visualization._ensure_task_bundle_paths",
        lambda dataset_id, feature_protocol_id, task_spec, cache_root: paths,
    )

    from wind_datasets.visualization import load_farm_status_tile

    tile = load_farm_status_tile("toy", "power_ws_hist", cache_root=tmp_path)

    assert tile.dataset_id == "toy"
    assert tile.task_id == "next_6h_from_24h"
    assert tile.total_points == 2
    assert tile.any_issue_points == 2
    assert tile.target_issue_points == 1
    assert tile.feature_issue_points == 1


def _load_notebook_generator_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "generate_visualization_notebooks.py"
    spec = importlib.util.spec_from_file_location("generate_visualization_notebooks", module_path)
    if spec is None or spec.loader is None:
        raise AssertionError("Failed to load generate_visualization_notebooks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

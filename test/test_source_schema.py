from __future__ import annotations

import polars as pl

from wind_datasets.source_schema import normalize_source_frame, normalize_source_header


def test_normalize_source_header_trims_and_drops_empty_columns() -> None:
    assert normalize_source_header(
        ["Grid Meter Performance Ratio ", "", " Rated power (kW) "],
        drop_empty=True,
    ) == ["Grid Meter Performance Ratio", "Rated power (kW)"]


def test_normalize_source_frame_trims_names_and_drops_blank_header_columns() -> None:
    frame = pl.DataFrame(
        {
            "Grid Meter Performance Ratio ": [0.98],
            "": ["unused"],
        }
    )

    normalized = normalize_source_frame(frame, drop_empty=True)

    assert normalized.columns == ["Grid Meter Performance Ratio"]
    assert normalized["Grid Meter Performance Ratio"][0] == 0.98

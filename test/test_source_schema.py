from __future__ import annotations

import polars as pl

from wind_datasets.source_schema import (
    _read_plain_csv_header,
    normalize_source_frame,
    normalize_source_header,
)


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


def test_read_plain_csv_header_uses_lossy_fallback_for_bad_bytes(tmp_path) -> None:
    path = tmp_path / "bad.csv"
    path.write_bytes(b"good,\x81bad\n1,2\n")

    header = _read_plain_csv_header(path)

    assert header[0] == "good"
    assert len(header) == 2
    assert "bad" in header[1]

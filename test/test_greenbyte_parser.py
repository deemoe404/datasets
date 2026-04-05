from __future__ import annotations

import polars as pl

from wind_datasets.datasets.greenbyte import GreenbyteDatasetBuilder, _parse_status_csv

from .helpers import build_greenbyte_fixture


def test_greenbyte_parser_merges_duplicate_rows_and_records_conflicts(tmp_path) -> None:
    source_root = tmp_path / "kelmarsh_raw"
    spec = build_greenbyte_fixture(source_root, "Kelmarsh", "Kelmarsh 1")
    builder = GreenbyteDatasetBuilder(spec=spec, cache_root=tmp_path / "cache")

    silver_path = builder.build_silver()

    continuous_files = sorted((silver_path / "continuous").glob("*.parquet"))
    assert continuous_files
    merged = pl.read_parquet(continuous_files[0]).sort("Date and time")
    conflicts = pl.read_parquet(silver_path / "conflicts.parquet")
    event_frames = [
        pl.read_parquet(path)
        for path in sorted((silver_path / "events").glob("Status_*.parquet"))
    ]
    events = pl.concat(event_frames, how="diagonal_relaxed")

    assert merged.height == 8
    assert merged.filter(pl.col("Date and time") == "2024-01-01 00:00:00")["source_row_count"][0] == 2
    assert merged.filter(pl.col("Date and time") == "2024-01-01 00:00:00")["row_conflict_count"][0] == 1
    assert merged.filter(pl.col("Date and time") == "2024-01-01 00:00:00")[
        "Blade angle (pitch position) C (°)"
    ][0] == "0.9"
    assert conflicts.height == 1
    assert conflicts["column_name"][0] == "Wind speed (m/s)"
    assert events.height == 4

    rebuilt_path = builder.build_silver()
    rebuilt_conflicts = pl.read_parquet(rebuilt_path / "conflicts.parquet")
    rebuilt_stats = pl.read_parquet(rebuilt_path / "meta" / "continuous_build_stats.parquet")

    assert rebuilt_conflicts.height == conflicts.height
    assert rebuilt_stats.height == 1
    assert rebuilt_stats["merged_row_count"][0] == 8


def test_greenbyte_status_parser_supports_non_turbine_assets(tmp_path) -> None:
    status_path = tmp_path / "Status_Kelmarsh_Grid_Meter.csv"
    status_path.write_text(
        "\n".join(
            [
                "# This file was exported by Greenbyte at 2023-08-15 15:39:25.",
                "#",
                "# Time zone: UTC",
                "# Time interval: 2016-01-01 00:00:00 - 2023-01-01 00:00:00 (2557 days)",
                "#",
                "# Kelmarsh Grid Meter Sum production: NaN kWh",
                "#",
                "Timestamp start,Timestamp end,Duration,Status,Code,Message,Comment,Service contract category,IEC category",
                "2019-06-30 22:30:00,,,Communication,9997,Data communication unavailable,,,",
            ]
        )
        + "\n"
    )

    frame = _parse_status_csv(status_path)

    assert frame["asset_id"][0] == "Kelmarsh Grid Meter Sum"
    assert frame["asset_type"][0] == "asset"
    assert frame["turbine_id"][0] == "Kelmarsh Grid Meter Sum"

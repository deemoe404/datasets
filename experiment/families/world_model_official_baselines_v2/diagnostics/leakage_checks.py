from __future__ import annotations


def assert_no_future_target_columns(columns: list[str]) -> None:
    forbidden = [column for column in columns if "future_target" in column or column == "target_future"]
    if forbidden:
        raise ValueError(f"Future target leakage columns are forbidden: {forbidden!r}")

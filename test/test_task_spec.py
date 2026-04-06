from __future__ import annotations

import pytest

from wind_datasets.models import TaskSpec


def test_default_preset_derives_steps_for_10_and_30_minute_datasets() -> None:
    task = TaskSpec.next_6h_from_24h()

    ten_minute = task.resolve(10)
    thirty_minute = task.resolve(30)

    assert task.granularity == "farm"
    assert ten_minute.history_steps == 144
    assert ten_minute.forecast_steps == 36
    assert ten_minute.stride_steps == 1
    assert thirty_minute.history_steps == 48
    assert thirty_minute.forecast_steps == 12
    assert thirty_minute.stride_steps == 1


def test_default_preset_supports_explicit_turbine_granularity() -> None:
    task = TaskSpec.next_6h_from_24h(granularity="turbine")

    assert task.granularity == "turbine"


def test_non_divisible_duration_raises() -> None:
    task = TaskSpec(history_duration="25m", forecast_duration="1h")
    with pytest.raises(ValueError):
        task.resolve(10)

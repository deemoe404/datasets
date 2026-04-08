from __future__ import annotations

from datetime import datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np
import polars as pl

from wind_datasets.models import DatasetSpec, TaskSpec


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "experiment"
        / "chronos-2"
        / "chronos2_power_only.py"
    )
    spec = spec_from_file_location("chronos2_power_only", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _penmanshiel_turbine_ids() -> tuple[str, ...]:
    return tuple(f"Penmanshiel {index:02d}" for index in (1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15))


def test_apply_target_policy_masks_invalid_and_clips_remaining_values() -> None:
    module = _load_module()

    masked = module.apply_target_policy(
        [100.0, -5.0, 2600.0, 90.0, None],
        [False, False, False, True, True],
        rated_power_kw=2050.0,
    )

    assert masked.tolist()[:3] == [100.0, 0.0, 2050.0]
    assert np.isnan(masked[3])
    assert np.isnan(masked[4])


def test_select_device_prefers_cuda_then_mps_then_cpu() -> None:
    module = _load_module()

    class _Backend:
        def __init__(self, available: bool) -> None:
            self._available = available

        def is_available(self) -> bool:
            return self._available

    class _Cuda:
        def __init__(self, available: bool) -> None:
            self._available = available

        def is_available(self) -> bool:
            return self._available

    class _Torch:
        def __init__(self, cuda_available: bool, mps_available: bool) -> None:
            self.cuda = _Cuda(cuda_available)
            self.backends = type("Backends", (), {"mps": _Backend(mps_available)})()

    assert module.select_device(_Torch(cuda_available=True, mps_available=True)) == "cuda"
    assert module.select_device(_Torch(cuda_available=False, mps_available=True)) == "mps"
    assert module.select_device(_Torch(cuda_available=False, mps_available=False)) == "cpu"


def test_prepare_power_only_series_masks_flagged_values_and_clips_clean_values() -> None:
    module = _load_module()
    timestamps = pl.datetime_range(
        start=datetime(2024, 1, 1, 0, 0, 0),
        end=datetime(2024, 1, 1, 0, 40, 0),
        interval="10m",
        eager=True,
    )
    series = pl.DataFrame(
        {
            "dataset": ["kelmarsh"] * len(timestamps),
            "turbine_id": ["Kelmarsh 1"] * len(timestamps),
            "timestamp": timestamps,
            "target_kw": [10.0, -2.0, 2200.0, 15.0, None],
            "quality_flags": ["", "", "", "conflict_resolved", ""],
        }
    )

    prepared = module.prepare_power_only_series(series, rated_power_kw=2050.0)

    assert prepared["invalid_target"].to_list() == [False, False, False, True, True]
    masked = prepared["target_kw_masked"].to_numpy()
    np.testing.assert_allclose(masked[:3], np.array([10.0, 0.0, 2050.0], dtype=np.float64))
    assert np.isnan(masked[3])
    assert np.isnan(masked[4])


def test_prepare_power_stats_series_masks_invalid_rows_and_clips_covariates() -> None:
    module = _load_module()
    timestamps = pl.datetime_range(
        start=datetime(2024, 1, 1, 0, 0, 0),
        end=datetime(2024, 1, 1, 0, 40, 0),
        interval="10m",
        eager=True,
    )
    series = pl.DataFrame(
        {
            "dataset": ["kelmarsh"] * len(timestamps),
            "turbine_id": ["Kelmarsh 1"] * len(timestamps),
            "timestamp": timestamps,
            "target_kw": [10.0, -2.0, 2200.0, 15.0, None],
            "quality_flags": ["", "", "", "conflict_resolved", ""],
            "Power, Minimum (kW)": [-1.0, 0.0, 2100.0, 4.0, 5.0],
            "Power, Maximum (kW)": [11.0, 22.0, None, 44.0, 55.0],
            "Power, Standard deviation (kW)": [1.0, -3.0, 3000.0, 5.0, 6.0],
        }
    )

    prepared = module.prepare_power_stats_series(
        series,
        dataset_id="kelmarsh",
        rated_power_kw=2050.0,
    )

    assert prepared["invalid_target"].to_list() == [False, False, False, True, True]
    np.testing.assert_allclose(
        prepared["target_kw_masked"].to_numpy(),
        np.array([10.0, 0.0, 2050.0, np.nan, np.nan], dtype=np.float64),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        prepared["Power, Minimum (kW)"].to_numpy(),
        np.array([0.0, 0.0, 2050.0, np.nan, np.nan], dtype=np.float64),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        prepared["Power, Maximum (kW)"].to_numpy(),
        np.array([11.0, 22.0, np.nan, np.nan, np.nan], dtype=np.float64),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        prepared["Power, Standard deviation (kW)"].to_numpy(),
        np.array([1.0, 0.0, 2050.0, np.nan, np.nan], dtype=np.float64),
        equal_nan=True,
    )


def test_load_power_only_series_frame_projects_only_required_columns(monkeypatch, tmp_path) -> None:
    module = _load_module()
    spec = DatasetSpec(
        dataset_id="hill_of_towie",
        source_root=Path("."),
        resolution_minutes=10,
        turbine_ids=("T01", "T02"),
        target_column="target_kw",
        target_unit="kW",
        timezone_policy="naive",
        timestamp_convention="naive",
        default_feature_groups=("main",),
        handler="synthetic",
    )
    series_path = tmp_path / "series.parquet"
    pl.DataFrame(
        {
            "dataset": ["hill_of_towie", "hill_of_towie"],
            "turbine_id": ["T01", "T02"],
            "timestamp": [datetime(2024, 1, 1, 0, 0, 0), datetime(2024, 1, 1, 0, 10, 0)],
            "target_kw": [100.0, 200.0],
            "quality_flags": ["", "flagged"],
            "unused_feature": [1.0, 2.0],
        }
    ).write_parquet(series_path)

    selected_columns: list[tuple[str, ...]] = []
    original_scan_parquet = module.pl.scan_parquet

    class _RecordingScan:
        def __init__(self, wrapped) -> None:
            self._wrapped = wrapped

        def select(self, columns):
            selected_columns.append(tuple(columns))
            self._wrapped = self._wrapped.select(columns)
            return self

        def filter(self, expr):
            self._wrapped = self._wrapped.filter(expr)
            return self

        def collect(self):
            return self._wrapped.collect()

    monkeypatch.setattr(
        module,
        "resolve_power_only_series_path",
        lambda dataset_id, *, cache_root: (spec, series_path),
    )
    monkeypatch.setattr(module.pl, "scan_parquet", lambda path: _RecordingScan(original_scan_parquet(path)))

    loaded_spec, loaded_path, loaded = module.load_power_only_series_frame("hill_of_towie", turbine_ids=("T01",))

    assert loaded_spec == spec
    assert loaded_path == series_path
    assert selected_columns == [module._POWER_ONLY_COLUMNS]
    assert loaded.columns == list(module._POWER_ONLY_COLUMNS)
    assert loaded["turbine_id"].to_list() == ["T01"]


def test_load_power_only_series_frame_projects_power_stats_columns_and_requires_declared_covariates(monkeypatch, tmp_path) -> None:
    module = _load_module()
    spec = DatasetSpec(
        dataset_id="hill_of_towie",
        source_root=Path("."),
        resolution_minutes=10,
        turbine_ids=("T01", "T02"),
        target_column="target_kw",
        target_unit="kW",
        timezone_policy="naive",
        timestamp_convention="naive",
        default_feature_groups=("main",),
        handler="synthetic",
    )
    series_path = tmp_path / "series.parquet"
    pl.DataFrame(
        {
            "dataset": ["hill_of_towie", "hill_of_towie"],
            "turbine_id": ["T01", "T02"],
            "timestamp": [datetime(2024, 1, 1, 0, 0, 0), datetime(2024, 1, 1, 0, 10, 0)],
            "target_kw": [100.0, 200.0],
            "quality_flags": ["", ""],
            "wtc_ActPower_min": [95.0, 195.0],
            "wtc_ActPower_max": [110.0, 210.0],
            "wtc_ActPower_stddev": [3.0, 4.0],
            "wtc_ActPower_endvalue": [101.0, 201.0],
            "unused_feature": [1.0, 2.0],
        }
    ).write_parquet(series_path)

    selected_columns: list[tuple[str, ...]] = []
    original_scan_parquet = module.pl.scan_parquet

    class _RecordingScan:
        def __init__(self, wrapped) -> None:
            self._wrapped = wrapped

        def select(self, columns):
            selected_columns.append(tuple(columns))
            self._wrapped = self._wrapped.select(columns)
            return self

        def filter(self, expr):
            self._wrapped = self._wrapped.filter(expr)
            return self

        def collect(self):
            return self._wrapped.collect()

    monkeypatch.setattr(
        module,
        "resolve_power_only_series_path",
        lambda dataset_id, *, cache_root: (spec, series_path),
    )
    monkeypatch.setattr(module.pl, "scan_parquet", lambda path: _RecordingScan(original_scan_parquet(path)))

    expected_columns = module.resolve_power_only_columns("hill_of_towie", include_power_stats=True)
    loaded_spec, loaded_path, loaded = module.load_power_only_series_frame(
        "hill_of_towie",
        turbine_ids=("T01",),
        include_power_stats=True,
    )

    assert loaded_spec == spec
    assert loaded_path == series_path
    assert selected_columns == [expected_columns]
    assert loaded.columns == list(expected_columns)
    assert loaded["turbine_id"].to_list() == ["T01"]

    missing_series_path = tmp_path / "series_missing.parquet"
    pl.DataFrame(
        {
            "dataset": ["hill_of_towie"],
            "turbine_id": ["T01"],
            "timestamp": [datetime(2024, 1, 1, 0, 0, 0)],
            "target_kw": [100.0],
            "quality_flags": [""],
            "wtc_ActPower_min": [95.0],
            "wtc_ActPower_max": [110.0],
            "wtc_ActPower_stddev": [3.0],
        }
    ).write_parquet(missing_series_path)
    monkeypatch.setattr(
        module,
        "resolve_power_only_series_path",
        lambda dataset_id, *, cache_root: (spec, missing_series_path),
    )

    try:
        module.load_power_only_series_frame("hill_of_towie", include_power_stats=True)
    except ValueError as exc:
        assert "missing required columns" in str(exc)
        assert "wtc_ActPower_endvalue" in str(exc)
    else:
        raise AssertionError("Expected power_stats loading to fail when a declared covariate column is missing.")


def test_iter_univariate_batches_keeps_partial_nan_windows_and_drops_empty_ones() -> None:
    module = _load_module()
    timestamps_us = np.arange(10, dtype=np.int64) * 600_000_000
    turbine_series = module.TurbineSeries(
        timestamps_us=timestamps_us,
        target_kw_masked=np.array([1.0, np.nan, 3.0, 4.0, np.nan, 6.0, np.nan, np.nan, 9.0, 10.0], dtype=np.float32),
    )

    batches = list(
        module._iter_univariate_batches(
            turbine_series=turbine_series,
            history_steps=3,
            forecast_steps=2,
            stride_steps=2,
            batch_size=8,
        )
    )

    assert len(batches) == 1
    contexts, actual_batch, future_timestamps = batches[0]
    assert contexts.shape == (3, 1, 3)
    np.testing.assert_allclose(contexts[0, 0], np.array([1.0, np.nan, 3.0], dtype=np.float32), equal_nan=True)
    np.testing.assert_allclose(actual_batch[0], np.array([4.0, np.nan], dtype=np.float64), equal_nan=True)
    np.testing.assert_array_equal(future_timestamps[0], np.array([1_800_000_000, 2_400_000_000], dtype=np.int64))


def test_evaluate_univariate_dataset_masks_invalid_future_positions(monkeypatch) -> None:
    module = _load_module()
    spec = DatasetSpec(
        dataset_id="kelmarsh",
        source_root=Path("."),
        resolution_minutes=10,
        turbine_ids=("Kelmarsh 1",),
        target_column="target_kw",
        target_unit="kW",
        timezone_policy="naive",
        timestamp_convention="naive",
        default_feature_groups=("main",),
        handler="synthetic",
    )
    task_spec = TaskSpec(
        history_duration="30m",
        forecast_duration="20m",
        stride_duration="20m",
        task_id="synthetic_task",
        granularity="turbine",
    )
    resolved_task = task_spec.resolve(spec.resolution_minutes)
    timestamps = pl.datetime_range(
        start=datetime(2024, 1, 1, 0, 0, 0),
        end=datetime(2024, 1, 1, 0, 50, 0),
        interval="10m",
        eager=True,
    )
    series = pl.DataFrame(
        {
            "dataset": ["kelmarsh"] * len(timestamps),
            "turbine_id": ["Kelmarsh 1"] * len(timestamps),
            "timestamp": timestamps,
            "target_kw": [10.0, -10.0, 2200.0, 40.0, 50.0, 60.0],
            "quality_flags": ["", "", "", "", "conflict_resolved", ""],
        }
    )

    def _fake_load_dataset_inputs(dataset_id, *, cache_root, task_spec, turbine_ids=None):
        assert dataset_id == "kelmarsh"
        assert turbine_ids == ("Kelmarsh 1",)
        return spec, resolved_task, series

    class _FakePipeline:
        def __init__(self) -> None:
            self.calls: list[np.ndarray] = []
            self.batch_sizes: list[int] = []

        def predict_quantiles(self, *, inputs, prediction_length, quantile_levels, batch_size, limit_prediction_length):
            assert prediction_length == resolved_task.forecast_steps
            assert quantile_levels == [0.5]
            self.calls.append(np.asarray(inputs))
            self.batch_sizes.append(batch_size)
            return [np.array([[[35.0], [70.0]]], dtype=np.float32)], [np.array([[35.0, 70.0]], dtype=np.float32)]

    pipeline = _FakePipeline()
    monkeypatch.setattr(module, "load_dataset_inputs", _fake_load_dataset_inputs)
    monkeypatch.setattr(module, "resolve_dataset_task", lambda dataset_id, *, task_spec=None: (spec, resolved_task))

    result = module.evaluate_univariate_dataset(
        "kelmarsh",
        pipeline=pipeline,
        task_spec=task_spec,
        batch_size=8,
        device="cpu",
    )

    assert len(pipeline.calls) == 1
    np.testing.assert_allclose(
        pipeline.calls[0],
        np.array([[[10.0, 0.0, 2050.0]]], dtype=np.float32),
        equal_nan=True,
    )
    assert pipeline.batch_sizes == [1]
    assert result["dataset_id"] == "kelmarsh_univariate"
    assert result["window_count"] == 1
    assert result["prediction_count"] == 1
    assert result["mae_kw"] == 5.0
    assert result["rmse_kw"] == 5.0
    assert result["start_timestamp"] == "2024-01-01 00:30:00"
    assert result["end_timestamp"] == "2024-01-01 00:30:00"


def test_evaluate_univariate_power_stats_dataset_uses_past_covariates_and_scores_target(monkeypatch) -> None:
    module = _load_module()
    spec = DatasetSpec(
        dataset_id="kelmarsh",
        source_root=Path("."),
        resolution_minutes=10,
        turbine_ids=("Kelmarsh 1",),
        target_column="target_kw",
        target_unit="kW",
        timezone_policy="naive",
        timestamp_convention="naive",
        default_feature_groups=("main",),
        handler="synthetic",
    )
    task_spec = TaskSpec(
        history_duration="30m",
        forecast_duration="20m",
        stride_duration="20m",
        task_id="synthetic_task",
        granularity="turbine",
    )
    resolved_task = task_spec.resolve(spec.resolution_minutes)
    timestamps = pl.datetime_range(
        start=datetime(2024, 1, 1, 0, 0, 0),
        end=datetime(2024, 1, 1, 0, 50, 0),
        interval="10m",
        eager=True,
    )
    series = pl.DataFrame(
        {
            "dataset": ["kelmarsh"] * len(timestamps),
            "turbine_id": ["Kelmarsh 1"] * len(timestamps),
            "timestamp": timestamps,
            "target_kw": [10.0, -10.0, 2200.0, 40.0, 50.0, 60.0],
            "quality_flags": ["", "", "", "", "conflict_resolved", ""],
            "Power, Minimum (kW)": [0.0, -3.0, 2201.0, 20.0, 30.0, 40.0],
            "Power, Maximum (kW)": [15.0, 25.0, 2300.0, 45.0, 55.0, 65.0],
            "Power, Standard deviation (kW)": [1.0, -4.0, 2100.0, 4.0, 5.0, 6.0],
        }
    )

    def _fake_load_dataset_inputs(dataset_id, *, cache_root, task_spec, turbine_ids=None, include_power_stats=False):
        assert dataset_id == "kelmarsh"
        assert turbine_ids == ("Kelmarsh 1",)
        assert include_power_stats is True
        return spec, resolved_task, series

    class _FakePipeline:
        def __init__(self) -> None:
            self.calls: list[list[dict[str, object]]] = []
            self.batch_sizes: list[int] = []

        def predict_quantiles(self, *, inputs, prediction_length, quantile_levels, batch_size, limit_prediction_length):
            assert prediction_length == resolved_task.forecast_steps
            assert quantile_levels == [0.5]
            self.calls.append(inputs)
            self.batch_sizes.append(batch_size)
            return [np.array([[[35.0], [70.0]]], dtype=np.float32)], [np.array([[35.0, 70.0]], dtype=np.float32)]

    pipeline = _FakePipeline()
    monkeypatch.setattr(module, "load_dataset_inputs", _fake_load_dataset_inputs)
    monkeypatch.setattr(module, "resolve_dataset_task", lambda dataset_id, *, task_spec=None: (spec, resolved_task))

    result = module.evaluate_univariate_power_stats_dataset(
        "kelmarsh",
        pipeline=pipeline,
        task_spec=task_spec,
        batch_size=8,
        device="cpu",
    )

    assert len(pipeline.calls) == 1
    assert len(pipeline.calls[0]) == 1
    first_call = pipeline.calls[0][0]
    np.testing.assert_allclose(first_call["target"], np.array([10.0, 0.0, 2050.0], dtype=np.float32), equal_nan=True)
    assert set(first_call["past_covariates"]) == {
        "Power, Minimum (kW)",
        "Power, Maximum (kW)",
        "Power, Standard deviation (kW)",
    }
    np.testing.assert_allclose(
        first_call["past_covariates"]["Power, Minimum (kW)"],
        np.array([0.0, 0.0, 2050.0], dtype=np.float32),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        first_call["past_covariates"]["Power, Maximum (kW)"],
        np.array([15.0, 25.0, 2050.0], dtype=np.float32),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        first_call["past_covariates"]["Power, Standard deviation (kW)"],
        np.array([1.0, 0.0, 2050.0], dtype=np.float32),
        equal_nan=True,
    )
    assert pipeline.batch_sizes == [4]
    assert result["dataset_id"] == "kelmarsh_univariate_power_stats"
    assert result["window_count"] == 1
    assert result["prediction_count"] == 1
    assert result["mae_kw"] == 5.0
    assert result["rmse_kw"] == 5.0
    assert result["start_timestamp"] == "2024-01-01 00:30:00"
    assert result["end_timestamp"] == "2024-01-01 00:30:00"


def test_build_knn_neighbor_map_prefers_xy_and_keeps_target_first() -> None:
    module = _load_module()
    turbine_static = pl.DataFrame(
        {
            "turbine_id": ["T2", "T1", "T3", "T4"],
            "coord_x": [1.0, 0.0, 2.0, 0.0],
            "coord_y": [0.0, 0.0, 0.0, 3.0],
            "latitude": [None, None, None, None],
            "longitude": [None, None, None, None],
        }
    )

    neighborhoods = module.build_knn_neighbor_map(
        turbine_static,
        turbine_ids=("T1", "T2", "T3", "T4"),
        max_neighbors=3,
    )

    assert neighborhoods["T1"] == ("T1", "T2", "T3")
    assert neighborhoods["T4"] == ("T4", "T1", "T2")


def test_build_knn_neighbor_map_falls_back_to_haversine_and_uses_all_neighbors_when_needed() -> None:
    module = _load_module()
    turbine_static = pl.DataFrame(
        {
            "turbine_id": ["A", "B", "C"],
            "coord_x": [None, None, None],
            "coord_y": [None, None, None],
            "latitude": [57.0, 57.01, 57.05],
            "longitude": [-2.0, -2.0, -2.0],
        }
    )

    neighborhoods = module.build_knn_neighbor_map(
        turbine_static,
        turbine_ids=("A", "B", "C"),
        max_neighbors=6,
    )

    assert neighborhoods["A"] == ("A", "B", "C")
    assert len(neighborhoods["B"]) == 3


def test_resolve_multivariate_epoch_configs_penmanshiel_splits_post_2023_targets() -> None:
    module = _load_module()
    spec = DatasetSpec(
        dataset_id="penmanshiel",
        source_root=Path("."),
        resolution_minutes=10,
        turbine_ids=_penmanshiel_turbine_ids(),
        target_column="target_kw",
        target_unit="kW",
        timezone_policy="naive",
        timestamp_convention="naive",
        default_feature_groups=("main",),
        handler="synthetic",
    )

    configs = module.resolve_multivariate_epoch_configs(
        spec,
        target_turbine_ids=("Penmanshiel 10", "Penmanshiel 11", "Penmanshiel 12"),
    )

    boundary_us = module._datetime_to_timestamp_us(module._PENMANSHIEL_EPOCH_BOUNDARY)
    assert [config.name for config in configs] == ["pre_2024_full_farm", "post_2023_active_subset"]
    assert configs[0].active_turbine_ids == spec.turbine_ids
    assert configs[0].target_turbine_ids == ("Penmanshiel 10", "Penmanshiel 11", "Penmanshiel 12")
    assert configs[0].forecast_start_us_min is None
    assert configs[0].forecast_start_us_max == boundary_us - 1
    assert configs[1].active_turbine_ids == module._PENMANSHIEL_POST_2023_TURBINE_IDS
    assert configs[1].target_turbine_ids == ("Penmanshiel 11", "Penmanshiel 12")
    assert configs[1].forecast_start_us_min == boundary_us
    assert configs[1].forecast_start_us_max is None


def test_build_epoch_knn_neighbor_map_penmanshiel_post_2023_uses_active_turbine_count() -> None:
    module = _load_module()
    turbine_static = pl.DataFrame(
        {
            "turbine_id": list(module._PENMANSHIEL_POST_2023_TURBINE_IDS),
            "coord_x": [0.0, 1.0, 2.0, 3.0, 4.0],
            "coord_y": [0.0] * 5,
            "latitude": [None] * 5,
            "longitude": [None] * 5,
        }
    )

    neighborhoods = module.build_epoch_knn_neighbor_map(
        turbine_static,
        active_turbine_ids=module._PENMANSHIEL_POST_2023_TURBINE_IDS,
    )

    assert neighborhoods["Penmanshiel 11"] == (
        "Penmanshiel 11",
        "Penmanshiel 12",
        "Penmanshiel 13",
        "Penmanshiel 14",
        "Penmanshiel 15",
    )
    assert len(neighborhoods["Penmanshiel 11"]) == 5


def test_build_local_panel_reindexes_async_neighbors_and_preserves_order() -> None:
    module = _load_module()
    turbine_series_map = {
        "T1": module.TurbineSeries(
            timestamps_us=np.array([0, 600_000_000, 1_800_000_000], dtype=np.int64),
            target_kw_masked=np.array([1.0, 2.0, 4.0], dtype=np.float32),
        ),
        "T2": module.TurbineSeries(
            timestamps_us=np.array([600_000_000, 1_200_000_000, 1_800_000_000], dtype=np.int64),
            target_kw_masked=np.array([10.0, 12.0, 13.0], dtype=np.float32),
        ),
    }

    local_panel = module.build_local_panel(
        turbine_series_map,
        turbine_ids=("T2", "T1"),
        resolution_minutes=10,
    )

    assert local_panel.turbine_ids == ("T2", "T1")
    assert local_panel.target_kw_masked.shape == (4, 2)
    np.testing.assert_array_equal(
        local_panel.timestamps_us,
        np.array([0, 600_000_000, 1_200_000_000, 1_800_000_000], dtype=np.int64),
    )
    np.testing.assert_allclose(
        local_panel.target_kw_masked[:, 0],
        np.array([np.nan, 10.0, 12.0, 13.0], dtype=np.float32),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        local_panel.target_kw_masked[:, 1],
        np.array([1.0, 2.0, np.nan, 4.0], dtype=np.float32),
        equal_nan=True,
    )


def test_build_local_power_stats_panel_reindexes_async_neighbors_and_masks_covariates() -> None:
    module = _load_module()
    timestamps = pl.Series(
        "timestamp",
        [
            datetime(2024, 1, 1, 0, 0, 0),
            datetime(2024, 1, 1, 0, 10, 0),
            datetime(2024, 1, 1, 0, 20, 0),
            datetime(2024, 1, 1, 0, 30, 0),
            datetime(2024, 1, 1, 0, 40, 0),
            datetime(2024, 1, 1, 0, 50, 0),
        ],
    )
    series = pl.DataFrame(
        {
            "dataset": ["kelmarsh"] * 6,
            "turbine_id": ["T1", "T1", "T1", "T2", "T2", "T2"],
            "timestamp": timestamps,
            "target_kw": [1.0, 2.0, 4.0, 10.0, 12.0, 13.0],
            "quality_flags": ["", "", "", "", "", "flagged"],
            "Power, Minimum (kW)": [0.0, 1.0, 3.0, 9.0, 11.0, 12.0],
            "Power, Maximum (kW)": [2.0, 3.0, 5.0, 11.0, 13.0, 14.0],
            "Power, Standard deviation (kW)": [0.1, 0.2, 0.4, 1.0, 1.2, 1.3],
        }
    )

    prepared = module.prepare_power_stats_series(
        series,
        dataset_id="kelmarsh",
        rated_power_kw=2050.0,
    )
    turbine_series_map = module.build_turbine_power_stats_series_map(
        prepared,
        covariate_columns=module.resolve_power_stats_covariate_columns("kelmarsh"),
    )

    local_panel = module.build_local_power_stats_panel(
        turbine_series_map,
        turbine_ids=("T2", "T1"),
        resolution_minutes=10,
        covariate_columns=module.resolve_power_stats_covariate_columns("kelmarsh"),
    )

    assert local_panel.turbine_ids == ("T2", "T1")
    assert local_panel.target_kw_masked.shape == (6, 2)
    np.testing.assert_array_equal(
        local_panel.timestamps_us,
        timestamps.cast(pl.Int64).to_numpy(),
    )
    np.testing.assert_allclose(
        local_panel.target_kw_masked[:, 0],
        np.array([np.nan, np.nan, np.nan, 10.0, 12.0, np.nan], dtype=np.float32),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        local_panel.target_kw_masked[:, 1],
        np.array([1.0, 2.0, 4.0, np.nan, np.nan, np.nan], dtype=np.float32),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        local_panel.past_covariates["Power, Minimum (kW)"][:, 0],
        np.array([np.nan, np.nan, np.nan, 9.0, 11.0, np.nan], dtype=np.float32),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        local_panel.past_covariates["Power, Maximum (kW)"][:, 1],
        np.array([2.0, 3.0, 5.0, np.nan, np.nan, np.nan], dtype=np.float32),
        equal_nan=True,
    )


def test_iter_multivariate_power_stats_batches_for_runs_flattens_neighbor_covariates() -> None:
    module = _load_module()
    farm_panel = module.FarmPowerStatsPanel(
        turbine_ids=("T01", "T02", "T03", "T04", "T05", "T06"),
        timestamps_us=np.array([0, 600_000_000, 1_200_000_000, 1_800_000_000, 2_400_000_000], dtype=np.int64),
        target_kw_masked=np.array(
            [
                [1.0, 10.0, 20.0, 30.0, 40.0, 50.0],
                [2.0, 11.0, 21.0, 31.0, 41.0, 51.0],
                [3.0, 12.0, 22.0, 32.0, 42.0, 52.0],
                [4.0, 13.0, 23.0, 33.0, 43.0, 53.0],
                [5.0, 14.0, 24.0, 34.0, 44.0, 54.0],
            ],
            dtype=np.float32,
        ),
        past_covariates={
            "Power, Minimum (kW)": np.array(
                [
                    [0.5, 9.5, 19.5, 29.5, 39.5, 49.5],
                    [1.5, 10.5, 20.5, 30.5, 40.5, 50.5],
                    [2.5, 11.5, 21.5, 31.5, 41.5, 51.5],
                    [3.5, 12.5, 22.5, 32.5, 42.5, 52.5],
                    [4.5, 13.5, 23.5, 33.5, 43.5, 53.5],
                ],
                dtype=np.float32,
            ),
            "Power, Maximum (kW)": np.array(
                [
                    [1.5, 10.5, 20.5, 30.5, 40.5, 50.5],
                    [2.5, 11.5, 21.5, 31.5, 41.5, 51.5],
                    [3.5, 12.5, 22.5, 32.5, 42.5, 52.5],
                    [4.5, 13.5, 23.5, 33.5, 43.5, 53.5],
                    [5.5, 14.5, 24.5, 34.5, 44.5, 54.5],
                ],
                dtype=np.float32,
            ),
            "Power, Standard deviation (kW)": np.array(
                [
                    [100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
                    [101.0, 111.0, 121.0, 131.0, 141.0, 151.0],
                    [102.0, 112.0, 122.0, 132.0, 142.0, 152.0],
                    [103.0, 113.0, 123.0, 133.0, 143.0, 153.0],
                    [104.0, 114.0, 124.0, 134.0, 144.0, 154.0],
                ],
                dtype=np.float32,
            ),
        },
    )

    batches = list(
        module._iter_multivariate_power_stats_batches_for_runs(
            panel_runs=[module.TargetPowerStatsPanelRun(name="default", farm_panel=farm_panel, scored_row_index=0)],
            history_steps=3,
            forecast_steps=2,
            stride_steps=2,
            batch_size=4,
            covariate_specs=module.resolve_power_stats_covariate_specs("kelmarsh"),
        )
    )

    assert len(batches) == 1
    input_batch, actual_batch, future_timestamps_batch = batches[0]
    assert len(input_batch) == 1
    assert input_batch[0]["target"].shape == (6, 3)
    assert len(input_batch[0]["past_covariates"]) == 18
    assert tuple(input_batch[0]["past_covariates"])[:3] == (
        "neighbor_00__cov00_min",
        "neighbor_00__cov01_max",
        "neighbor_00__cov02_stddev",
    )
    np.testing.assert_allclose(
        input_batch[0]["past_covariates"]["neighbor_00__cov00_min"],
        np.array([0.5, 1.5, 2.5], dtype=np.float32),
    )
    np.testing.assert_allclose(
        input_batch[0]["past_covariates"]["neighbor_05__cov02_stddev"],
        np.array([150.0, 151.0, 152.0], dtype=np.float32),
    )
    assert actual_batch.shape == (1, 6, 2)
    np.testing.assert_array_equal(
        future_timestamps_batch,
        np.array([[1_800_000_000, 2_400_000_000]], dtype=np.int64),
    )


def test_evaluate_multivariate_knn6_dataset_penmanshiel_switches_epoch_neighbors_at_2024_boundary(monkeypatch) -> None:
    module = _load_module()
    turbine_ids = _penmanshiel_turbine_ids()
    spec = DatasetSpec(
        dataset_id="penmanshiel",
        source_root=Path("."),
        resolution_minutes=10,
        turbine_ids=turbine_ids,
        target_column="target_kw",
        target_unit="kW",
        timezone_policy="naive",
        timestamp_convention="naive",
        default_feature_groups=("main",),
        handler="synthetic",
    )
    task_spec = TaskSpec(
        history_duration="30m",
        forecast_duration="20m",
        stride_duration="30m",
        task_id="synthetic_task",
        granularity="turbine",
    )
    resolved_task = task_spec.resolve(spec.resolution_minutes)
    timestamps = pl.datetime_range(
        start=datetime(2023, 12, 31, 23, 0, 0),
        end=datetime(2024, 1, 1, 0, 20, 0),
        interval="10m",
        eager=True,
    )
    retired_timestamps = timestamps[:6]
    rows: list[dict[str, object]] = []
    values_by_turbine = {
        "Penmanshiel 11": list(range(1, 10)),
        "Penmanshiel 12": list(range(11, 20)),
        "Penmanshiel 13": list(range(21, 30)),
        "Penmanshiel 14": list(range(31, 40)),
        "Penmanshiel 15": list(range(41, 50)),
        "Penmanshiel 10": list(range(101, 107)),
        "Penmanshiel 09": list(range(201, 207)),
    }
    for turbine_id in ("Penmanshiel 11", "Penmanshiel 12", "Penmanshiel 13", "Penmanshiel 14", "Penmanshiel 15"):
        for timestamp, target_kw in zip(timestamps, values_by_turbine[turbine_id], strict=True):
            rows.append(
                {
                    "dataset": "penmanshiel",
                    "turbine_id": turbine_id,
                    "timestamp": timestamp,
                    "target_kw": float(target_kw),
                    "quality_flags": "",
                }
            )
    for turbine_id in ("Penmanshiel 10", "Penmanshiel 09"):
        for timestamp, target_kw in zip(retired_timestamps, values_by_turbine[turbine_id], strict=True):
            rows.append(
                {
                    "dataset": "penmanshiel",
                    "turbine_id": turbine_id,
                    "timestamp": timestamp,
                    "target_kw": float(target_kw),
                    "quality_flags": "",
                }
            )
    full_series = pl.DataFrame(rows)
    turbine_static = pl.DataFrame(
        {
            "turbine_id": list(turbine_ids),
            "coord_x": [-50.0, -40.0, -30.0, -25.0, -20.0, -15.0, -10.0, -1.5, -1.0, 0.0, 1.0, 2.0, 2.1, 2.2],
            "coord_y": [0.0] * len(turbine_ids),
            "latitude": [None] * len(turbine_ids),
            "longitude": [None] * len(turbine_ids),
        }
    )
    seen_loader_turbines: list[tuple[str, ...] | None] = []

    def _fake_load_dataset_inputs(dataset_id, *, cache_root, task_spec, turbine_ids=None):
        assert dataset_id == "penmanshiel"
        resolved_turbine_ids = tuple(turbine_ids) if turbine_ids is not None else None
        seen_loader_turbines.append(resolved_turbine_ids)
        series = full_series
        if resolved_turbine_ids is not None:
            series = series.filter(pl.col("turbine_id").is_in(list(resolved_turbine_ids)))
        return spec, resolved_task, series

    class _FakePipeline:
        def __init__(self) -> None:
            self.calls: list[np.ndarray] = []
            self.batch_sizes: list[int] = []

        def predict_quantiles(self, *, inputs, prediction_length, quantile_levels, batch_size, limit_prediction_length, cross_learning):
            assert prediction_length == resolved_task.forecast_steps
            assert quantile_levels == [0.5]
            assert cross_learning is False
            batch_inputs = np.asarray(inputs)
            self.calls.append(batch_inputs)
            self.batch_sizes.append(batch_size)
            forecast = np.zeros((batch_inputs.shape[1], 2), dtype=np.float32)
            target_last_value = float(batch_inputs[0, 0, -1])
            if batch_inputs.shape[1] == 6:
                assert target_last_value == 3.0
                forecast[0] = np.array([4.0, 5.0], dtype=np.float32)
            elif batch_inputs.shape[1] == 5:
                assert target_last_value == 6.0
                forecast[0] = np.array([7.0, 8.0], dtype=np.float32)
            else:
                raise AssertionError(f"Unexpected variate count {batch_inputs.shape[1]}.")
            return [forecast[:, :, None]], [forecast]

    pipeline = _FakePipeline()
    monkeypatch.setattr(module, "load_dataset_inputs", _fake_load_dataset_inputs)
    monkeypatch.setattr(module, "load_dataset_turbine_static", lambda dataset_id, *, cache_root: turbine_static)
    monkeypatch.setattr(module, "resolve_dataset_task", lambda dataset_id, *, task_spec=None: (spec, resolved_task))

    result = module.evaluate_multivariate_knn6_dataset(
        "penmanshiel",
        pipeline=pipeline,
        task_spec=task_spec,
        batch_size=4,
        device="cpu",
        turbine_ids=("Penmanshiel 11",),
    )

    assert seen_loader_turbines == [
        (
            "Penmanshiel 11",
            "Penmanshiel 10",
            "Penmanshiel 12",
            "Penmanshiel 09",
            "Penmanshiel 13",
            "Penmanshiel 14",
        ),
        (
            "Penmanshiel 11",
            "Penmanshiel 12",
            "Penmanshiel 13",
            "Penmanshiel 14",
            "Penmanshiel 15",
        ),
    ]
    assert len(pipeline.calls) == 2
    assert pipeline.batch_sizes == [6, 5]
    np.testing.assert_allclose(
        pipeline.calls[0],
        np.array(
            [[
                [1.0, 2.0, 3.0],
                [101.0, 102.0, 103.0],
                [11.0, 12.0, 13.0],
                [201.0, 202.0, 203.0],
                [21.0, 22.0, 23.0],
                [31.0, 32.0, 33.0],
            ]],
            dtype=np.float32,
        ),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        pipeline.calls[1],
        np.array(
            [[
                [4.0, 5.0, 6.0],
                [14.0, 15.0, 16.0],
                [24.0, 25.0, 26.0],
                [34.0, 35.0, 36.0],
                [44.0, 45.0, 46.0],
            ]],
            dtype=np.float32,
        ),
        equal_nan=True,
    )
    assert result["dataset_id"] == "penmanshiel_multivariate_knn6"
    assert result["window_count"] == 2
    assert result["prediction_count"] == 4
    assert result["mae_kw"] == 0.0
    assert result["rmse_kw"] == 0.0
    assert result["start_timestamp"] == "2023-12-31 23:30:00"
    assert result["end_timestamp"] == "2024-01-01 00:10:00"


def test_evaluate_multivariate_knn6_dataset_scores_only_target_turbine(monkeypatch) -> None:
    module = _load_module()
    spec = DatasetSpec(
        dataset_id="kelmarsh",
        source_root=Path("."),
        resolution_minutes=10,
        turbine_ids=("Kelmarsh 1", "Kelmarsh 2", "Kelmarsh 3"),
        target_column="target_kw",
        target_unit="kW",
        timezone_policy="naive",
        timestamp_convention="naive",
        default_feature_groups=("main",),
        handler="synthetic",
    )
    task_spec = TaskSpec(
        history_duration="30m",
        forecast_duration="20m",
        stride_duration="20m",
        task_id="synthetic_task",
        granularity="turbine",
    )
    resolved_task = task_spec.resolve(spec.resolution_minutes)
    timestamps = pl.datetime_range(
        start=datetime(2024, 1, 1, 0, 0, 0),
        end=datetime(2024, 1, 1, 0, 50, 0),
        interval="10m",
        eager=True,
    )
    series = pl.DataFrame(
        {
            "dataset": ["kelmarsh"] * 17,
            "turbine_id": (["Kelmarsh 1"] * 6) + (["Kelmarsh 2"] * 5) + (["Kelmarsh 3"] * 6),
            "timestamp": list(timestamps) + [timestamps[0], timestamps[2], timestamps[3], timestamps[4], timestamps[5]] + list(timestamps),
            "target_kw": [
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                10.0, 30.0, 40.0, None, 60.0,
                100.0, 200.0, 300.0, 400.0, 500.0, 600.0,
            ],
            "quality_flags": [""] * 17,
        }
    )
    turbine_static = pl.DataFrame(
        {
            "turbine_id": ["Kelmarsh 1", "Kelmarsh 2", "Kelmarsh 3"],
            "coord_x": [0.0, 1.0, 3.0],
            "coord_y": [0.0, 0.0, 0.0],
            "latitude": [None, None, None],
            "longitude": [None, None, None],
        }
    )

    def _fake_load_dataset_inputs(dataset_id, *, cache_root, task_spec, turbine_ids=None):
        assert dataset_id == "kelmarsh"
        return spec, resolved_task, series

    class _FakePipeline:
        def __init__(self) -> None:
            self.calls: list[np.ndarray] = []
            self.batch_sizes: list[int] = []

        def predict_quantiles(self, *, inputs, prediction_length, quantile_levels, batch_size, limit_prediction_length, cross_learning):
            assert prediction_length == resolved_task.forecast_steps
            assert quantile_levels == [0.5]
            assert cross_learning is False
            batch_inputs = np.asarray(inputs)
            self.calls.append(batch_inputs)
            self.batch_sizes.append(batch_size)
            target_last_value = float(batch_inputs[0, 0, -1])
            if target_last_value == 3.0:
                target_forecast = np.array([4.0, 5.0], dtype=np.float32)
            elif target_last_value == 30.0:
                target_forecast = np.array([40.0, 50.0], dtype=np.float32)
            elif target_last_value == 300.0:
                target_forecast = np.array([400.0, 500.0], dtype=np.float32)
            else:
                raise AssertionError(f"Unexpected target signature {target_last_value}.")

            forecast = np.zeros((3, 2), dtype=np.float32)
            forecast[0] = target_forecast
            return [forecast[:, :, None]], [forecast]

    pipeline = _FakePipeline()
    monkeypatch.setattr(module, "load_dataset_inputs", _fake_load_dataset_inputs)
    monkeypatch.setattr(module, "load_dataset_turbine_static", lambda dataset_id, *, cache_root: turbine_static)
    monkeypatch.setattr(module, "resolve_dataset_task", lambda dataset_id, *, task_spec=None: (spec, resolved_task))

    result = module.evaluate_multivariate_knn6_dataset(
        "kelmarsh",
        pipeline=pipeline,
        task_spec=task_spec,
        batch_size=4,
        device="cpu",
    )

    assert len(pipeline.calls) == 3
    np.testing.assert_allclose(
        pipeline.calls[0],
        np.array(
            [[
                [1.0, 2.0, 3.0],
                [10.0, np.nan, 30.0],
                [100.0, 200.0, 300.0],
            ]],
            dtype=np.float32,
        ),
        equal_nan=True,
    )
    assert pipeline.batch_sizes == [3, 3, 3]
    assert result["dataset_id"] == "kelmarsh_multivariate_knn6"
    assert result["window_count"] == 3
    assert result["prediction_count"] == 5
    assert result["mae_kw"] == 0.0
    assert result["rmse_kw"] == 0.0
    assert result["start_timestamp"] == "2024-01-01 00:30:00"
    assert result["end_timestamp"] == "2024-01-01 00:40:00"


def test_evaluate_multivariate_knn6_dataset_accepts_target_turbine_subset(monkeypatch) -> None:
    module = _load_module()
    spec = DatasetSpec(
        dataset_id="kelmarsh",
        source_root=Path("."),
        resolution_minutes=10,
        turbine_ids=("Kelmarsh 1", "Kelmarsh 2", "Kelmarsh 3"),
        target_column="target_kw",
        target_unit="kW",
        timezone_policy="naive",
        timestamp_convention="naive",
        default_feature_groups=("main",),
        handler="synthetic",
    )
    task_spec = TaskSpec(
        history_duration="30m",
        forecast_duration="20m",
        stride_duration="20m",
        task_id="synthetic_task",
        granularity="turbine",
    )
    resolved_task = task_spec.resolve(spec.resolution_minutes)
    timestamps = pl.datetime_range(
        start=datetime(2024, 1, 1, 0, 0, 0),
        end=datetime(2024, 1, 1, 0, 50, 0),
        interval="10m",
        eager=True,
    )
    series = pl.DataFrame(
        {
            "dataset": ["kelmarsh"] * 18,
            "turbine_id": (["Kelmarsh 1"] * 6) + (["Kelmarsh 2"] * 6) + (["Kelmarsh 3"] * 6),
            "timestamp": list(timestamps) * 3,
            "target_kw": [
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                10.0, 20.0, 30.0, 40.0, 50.0, 60.0,
                100.0, 200.0, 300.0, 400.0, 500.0, 600.0,
            ],
            "quality_flags": [""] * 18,
        }
    )
    turbine_static = pl.DataFrame(
        {
            "turbine_id": ["Kelmarsh 1", "Kelmarsh 2", "Kelmarsh 3"],
            "coord_x": [0.0, 1.0, 3.0],
            "coord_y": [0.0, 0.0, 0.0],
            "latitude": [None, None, None],
            "longitude": [None, None, None],
        }
    )

    def _fake_load_dataset_inputs(dataset_id, *, cache_root, task_spec, turbine_ids=None):
        assert dataset_id == "kelmarsh"
        assert turbine_ids == ("Kelmarsh 1", "Kelmarsh 2", "Kelmarsh 3")
        return spec, resolved_task, series

    class _FakePipeline:
        def __init__(self) -> None:
            self.calls: list[np.ndarray] = []

        def predict_quantiles(self, *, inputs, prediction_length, quantile_levels, batch_size, limit_prediction_length, cross_learning):
            assert prediction_length == resolved_task.forecast_steps
            self.calls.append(np.asarray(inputs))
            forecast = np.zeros((3, 2), dtype=np.float32)
            forecast[0] = np.array([4.0, 5.0], dtype=np.float32)
            return [forecast[:, :, None]], [forecast]

    pipeline = _FakePipeline()
    monkeypatch.setattr(module, "load_dataset_inputs", _fake_load_dataset_inputs)
    monkeypatch.setattr(module, "load_dataset_turbine_static", lambda dataset_id, *, cache_root: turbine_static)
    monkeypatch.setattr(module, "resolve_dataset_task", lambda dataset_id, *, task_spec=None: (spec, resolved_task))

    result = module.evaluate_multivariate_knn6_dataset(
        "kelmarsh",
        pipeline=pipeline,
        task_spec=task_spec,
        batch_size=4,
        device="cpu",
        turbine_ids=("Kelmarsh 1",),
    )

    assert len(pipeline.calls) == 1
    assert result["window_count"] == 1
    assert result["prediction_count"] == 2


def test_evaluate_multivariate_knn6_dataset_loads_only_required_neighbor_union(monkeypatch) -> None:
    module = _load_module()
    spec = DatasetSpec(
        dataset_id="hill_of_towie",
        source_root=Path("."),
        resolution_minutes=10,
        turbine_ids=("T01", "T02", "T03", "T04"),
        target_column="target_kw",
        target_unit="kW",
        timezone_policy="naive",
        timestamp_convention="naive",
        default_feature_groups=("main",),
        handler="synthetic",
    )
    task_spec = TaskSpec(
        history_duration="30m",
        forecast_duration="20m",
        stride_duration="20m",
        task_id="synthetic_task",
        granularity="turbine",
    )
    resolved_task = task_spec.resolve(spec.resolution_minutes)
    timestamps = pl.datetime_range(
        start=datetime(2024, 1, 1, 0, 0, 0),
        end=datetime(2024, 1, 1, 0, 50, 0),
        interval="10m",
        eager=True,
    )
    series = pl.DataFrame(
        {
            "dataset": ["hill_of_towie"] * 18,
            "turbine_id": (["T01"] * 6) + (["T02"] * 6) + (["T03"] * 6),
            "timestamp": list(timestamps) * 3,
            "target_kw": [
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                10.0, 20.0, 30.0, 40.0, 50.0, 60.0,
                100.0, 200.0, 300.0, 400.0, 500.0, 600.0,
            ],
            "quality_flags": [""] * 18,
        }
    )
    seen_loader_turbines: list[tuple[str, ...] | None] = []

    def _fake_load_dataset_inputs(dataset_id, *, cache_root, task_spec, turbine_ids=None):
        assert dataset_id == "hill_of_towie"
        seen_loader_turbines.append(tuple(turbine_ids) if turbine_ids is not None else None)
        return spec, resolved_task, series

    class _FakePipeline:
        def predict_quantiles(self, *, inputs, prediction_length, quantile_levels, batch_size, limit_prediction_length, cross_learning):
            assert prediction_length == resolved_task.forecast_steps
            assert cross_learning is False
            forecast = np.zeros((3, 2), dtype=np.float32)
            forecast[0] = np.array([4.0, 5.0], dtype=np.float32)
            return [forecast[:, :, None]], [forecast]

    monkeypatch.setattr(module, "load_dataset_inputs", _fake_load_dataset_inputs)
    monkeypatch.setattr(module, "load_dataset_turbine_static", lambda dataset_id, *, cache_root: pl.DataFrame())
    monkeypatch.setattr(
        module,
        "build_knn_neighbor_map",
        lambda turbine_static, *, turbine_ids, max_neighbors: {
            "T01": ("T01", "T02", "T03"),
            "T02": ("T02", "T01", "T03"),
            "T03": ("T03", "T01", "T02"),
            "T04": ("T04", "T01", "T02"),
        },
    )

    result = module.evaluate_multivariate_knn6_dataset(
        "hill_of_towie",
        pipeline=_FakePipeline(),
        task_spec=task_spec,
        batch_size=4,
        device="cpu",
        turbine_ids=("T01",),
    )

    assert seen_loader_turbines == [("T01", "T02", "T03")]
    assert result["dataset_id"] == "hill_of_towie_multivariate_knn6"
    assert result["window_count"] == 1
    assert result["prediction_count"] == 2


def test_evaluate_multivariate_knn6_dataset_hill_single_target_smoke(monkeypatch) -> None:
    module = _load_module()
    spec = DatasetSpec(
        dataset_id="hill_of_towie",
        source_root=Path("."),
        resolution_minutes=10,
        turbine_ids=("T01", "T02", "T03", "T04", "T05", "T06"),
        target_column="target_kw",
        target_unit="kW",
        timezone_policy="naive",
        timestamp_convention="naive",
        default_feature_groups=("main",),
        handler="synthetic",
    )
    task_spec = TaskSpec(
        history_duration="30m",
        forecast_duration="20m",
        stride_duration="20m",
        task_id="synthetic_task",
        granularity="turbine",
    )
    resolved_task = task_spec.resolve(spec.resolution_minutes)
    timestamps = pl.datetime_range(
        start=datetime(2024, 1, 1, 0, 0, 0),
        end=datetime(2024, 1, 1, 0, 50, 0),
        interval="10m",
        eager=True,
    )
    turbine_ids = list(spec.turbine_ids)
    values_by_turbine = {
        "T01": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "T02": [11.0, 21.0, 31.0, 41.0, 51.0, 61.0],
        "T03": [12.0, 22.0, 32.0, 42.0, 52.0, 62.0],
        "T04": [13.0, 23.0, 33.0, 43.0, 53.0, 63.0],
        "T05": [14.0, 24.0, 34.0, 44.0, 54.0, 64.0],
        "T06": [15.0, 25.0, 35.0, 45.0, 55.0, 65.0],
    }
    rows: list[dict[str, object]] = []
    for turbine_id in turbine_ids:
        for timestamp, target_kw in zip(timestamps, values_by_turbine[turbine_id], strict=True):
            rows.append(
                {
                    "dataset": "hill_of_towie",
                    "turbine_id": turbine_id,
                    "timestamp": timestamp,
                    "target_kw": target_kw,
                    "quality_flags": "",
                }
            )
    series = pl.DataFrame(rows)
    turbine_static = pl.DataFrame(
        {
            "turbine_id": turbine_ids,
            "coord_x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "coord_y": [0.0] * 6,
            "latitude": [None] * 6,
            "longitude": [None] * 6,
        }
    )

    def _fake_load_dataset_inputs(dataset_id, *, cache_root, task_spec, turbine_ids=None):
        assert dataset_id == "hill_of_towie"
        assert turbine_ids == tuple(spec.turbine_ids)
        return spec, resolved_task, series

    class _FakePipeline:
        def __init__(self) -> None:
            self.calls: list[np.ndarray] = []

        def predict_quantiles(self, *, inputs, prediction_length, quantile_levels, batch_size, limit_prediction_length, cross_learning):
            batch_inputs = np.asarray(inputs)
            self.calls.append(batch_inputs)
            forecast = np.zeros((6, 2), dtype=np.float32)
            forecast[0] = np.array([40.0, 50.0], dtype=np.float32)
            return [forecast[:, :, None]], [forecast]

    pipeline = _FakePipeline()
    monkeypatch.setattr(module, "load_dataset_inputs", _fake_load_dataset_inputs)
    monkeypatch.setattr(module, "load_dataset_turbine_static", lambda dataset_id, *, cache_root: turbine_static)
    monkeypatch.setattr(module, "resolve_dataset_task", lambda dataset_id, *, task_spec=None: (spec, resolved_task))

    result = module.evaluate_multivariate_knn6_dataset(
        "hill_of_towie",
        pipeline=pipeline,
        task_spec=task_spec,
        batch_size=1,
        device="cpu",
        turbine_ids=("T01",),
        max_windows_per_dataset=1,
    )

    assert len(pipeline.calls) == 1
    assert pipeline.calls[0].shape == (1, 6, 3)
    assert result["dataset_id"] == "hill_of_towie_multivariate_knn6"
    assert result["window_count"] == 1
    assert result["prediction_count"] == 2
    assert result["mae_kw"] == 0.0
    assert result["rmse_kw"] == 0.0


def test_evaluate_multivariate_knn6_power_stats_dataset_hill_single_target_smoke(monkeypatch) -> None:
    module = _load_module()
    spec = DatasetSpec(
        dataset_id="hill_of_towie",
        source_root=Path("."),
        resolution_minutes=10,
        turbine_ids=("T01", "T02", "T03", "T04", "T05", "T06"),
        target_column="target_kw",
        target_unit="kW",
        timezone_policy="naive",
        timestamp_convention="naive",
        default_feature_groups=("main",),
        handler="synthetic",
    )
    task_spec = TaskSpec(
        history_duration="30m",
        forecast_duration="20m",
        stride_duration="20m",
        task_id="synthetic_task",
        granularity="turbine",
    )
    resolved_task = task_spec.resolve(spec.resolution_minutes)
    timestamps = pl.datetime_range(
        start=datetime(2024, 1, 1, 0, 0, 0),
        end=datetime(2024, 1, 1, 0, 50, 0),
        interval="10m",
        eager=True,
    )
    turbine_ids = list(spec.turbine_ids)
    rows: list[dict[str, object]] = []
    for turbine_index, turbine_id in enumerate(turbine_ids, start=1):
        for step_index, timestamp in enumerate(timestamps, start=1):
            base = float(turbine_index * 10 + step_index)
            rows.append(
                {
                    "dataset": "hill_of_towie",
                    "turbine_id": turbine_id,
                    "timestamp": timestamp,
                    "target_kw": base,
                    "quality_flags": "",
                    "wtc_ActPower_min": base - 1.0,
                    "wtc_ActPower_max": base + 1.0,
                    "wtc_ActPower_stddev": base + 100.0,
                    "wtc_ActPower_endvalue": base + 200.0,
                }
            )
    series = pl.DataFrame(rows)
    turbine_static = pl.DataFrame(
        {
            "turbine_id": turbine_ids,
            "coord_x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "coord_y": [0.0] * 6,
            "latitude": [None] * 6,
            "longitude": [None] * 6,
        }
    )

    def _fake_load_dataset_inputs(dataset_id, *, cache_root, task_spec, turbine_ids=None, include_power_stats=False):
        assert dataset_id == "hill_of_towie"
        assert turbine_ids == tuple(spec.turbine_ids)
        assert include_power_stats is True
        return spec, resolved_task, series

    class _FakePipeline:
        def __init__(self) -> None:
            self.calls: list[list[dict[str, object]]] = []
            self.batch_sizes: list[int] = []

        def predict_quantiles(self, *, inputs, prediction_length, quantile_levels, batch_size, limit_prediction_length, cross_learning):
            assert prediction_length == resolved_task.forecast_steps
            assert quantile_levels == [0.5]
            assert cross_learning is False
            self.calls.append(inputs)
            self.batch_sizes.append(batch_size)
            assert len(inputs) == 1
            assert inputs[0]["target"].shape == (6, 3)
            assert len(inputs[0]["past_covariates"]) == 24
            forecast = np.zeros((6, 2), dtype=np.float32)
            forecast[0] = np.array([14.0, 15.0], dtype=np.float32)
            return [forecast[:, :, None]], [forecast]

    pipeline = _FakePipeline()
    monkeypatch.setattr(module, "load_dataset_inputs", _fake_load_dataset_inputs)
    monkeypatch.setattr(module, "load_dataset_turbine_static", lambda dataset_id, *, cache_root: turbine_static)
    monkeypatch.setattr(module, "resolve_dataset_task", lambda dataset_id, *, task_spec=None: (spec, resolved_task))

    result = module.evaluate_multivariate_knn6_power_stats_dataset(
        "hill_of_towie",
        pipeline=pipeline,
        task_spec=task_spec,
        batch_size=1,
        device="cpu",
        turbine_ids=("T01",),
        max_windows_per_dataset=1,
    )

    assert len(pipeline.calls) == 1
    assert pipeline.batch_sizes == [30]
    assert result["dataset_id"] == "hill_of_towie_multivariate_knn6_power_stats"
    assert result["window_count"] == 1
    assert result["prediction_count"] == 2
    assert result["mae_kw"] == 0.0
    assert result["rmse_kw"] == 0.0


def test_resolve_pipeline_batch_size_counts_series_not_windows() -> None:
    module = _load_module()

    assert module.resolve_pipeline_batch_size(np.zeros((32, 1, 144), dtype=np.float32)) == 32
    assert module.resolve_pipeline_batch_size(np.zeros((32, 6, 144), dtype=np.float32)) == 192
    assert module.resolve_pipeline_batch_size(
        [
            {
                "target": np.zeros(144, dtype=np.float32),
                "past_covariates": {
                    "a": np.zeros(144, dtype=np.float32),
                    "b": np.zeros(144, dtype=np.float32),
                },
            },
            {
                "target": np.zeros(144, dtype=np.float32),
                "past_covariates": {
                    "a": np.zeros(144, dtype=np.float32),
                    "b": np.zeros(144, dtype=np.float32),
                },
            },
        ]
    ) == 6


def _install_run_experiment_fakes(monkeypatch, module):
    class _FakePipeline:
        pass

    def _build_row(resolved: str, suffix: str, *, window_count: int, prediction_count: int) -> dict[str, object]:
        return {
            "dataset_id": f"{resolved}{suffix}",
            "model_id": module.MODEL_ID,
            "task_id": module.TASK_ID,
            "history_steps": 144,
            "forecast_steps": 36,
            "stride_steps": 36,
            "target_policy": module.TARGET_POLICY,
            "window_count": window_count,
            "prediction_count": prediction_count,
            "start_timestamp": "2024-01-01 00:10:00",
            "end_timestamp": "2024-01-10 00:00:00",
            "mae_kw": 1.0,
            "rmse_kw": 2.0,
            "mae_pu": 0.1,
            "rmse_pu": 0.2,
            "device": "cpu",
            "runtime_seconds": 0.5,
        }

    def _fake_evaluate_univariate_dataset(dataset_id, **kwargs):
        return _build_row(
            module.resolve_dataset_id(dataset_id),
            module.UNIVARIATE_SUFFIX,
            window_count=10,
            prediction_count=360,
        )

    def _fake_evaluate_univariate_power_stats_dataset(dataset_id, **kwargs):
        return _build_row(
            module.resolve_dataset_id(dataset_id),
            module.UNIVARIATE_POWER_STATS_SUFFIX,
            window_count=9,
            prediction_count=359,
        )

    def _fake_evaluate_multivariate_knn6_dataset(dataset_id, **kwargs):
        return _build_row(
            module.resolve_dataset_id(dataset_id),
            module.MULTIVARIATE_KNN6_SUFFIX,
            window_count=11,
            prediction_count=361,
        )

    def _fake_evaluate_multivariate_knn6_power_stats_dataset(dataset_id, **kwargs):
        return _build_row(
            module.resolve_dataset_id(dataset_id),
            module.MULTIVARIATE_KNN6_POWER_STATS_SUFFIX,
            window_count=8,
            prediction_count=288,
        )

    monkeypatch.setattr(module, "load_pipeline", lambda **kwargs: _FakePipeline())
    monkeypatch.setattr(module, "evaluate_univariate_dataset", _fake_evaluate_univariate_dataset)
    monkeypatch.setattr(module, "evaluate_univariate_power_stats_dataset", _fake_evaluate_univariate_power_stats_dataset)
    monkeypatch.setattr(module, "evaluate_multivariate_knn6_dataset", _fake_evaluate_multivariate_knn6_dataset)
    monkeypatch.setattr(
        module,
        "evaluate_multivariate_knn6_power_stats_dataset",
        _fake_evaluate_multivariate_knn6_power_stats_dataset,
    )


def test_run_experiment_all_mode_writes_expected_rows(monkeypatch, tmp_path) -> None:
    module = _load_module()
    _install_run_experiment_fakes(monkeypatch, module)

    output_path = tmp_path / "chronos-2.csv"
    result = module.run_experiment(
        dataset_ids=("kelmarsh", "penmanshiel", "hill_of_towie", "sdwpf_kddcup"),
        output_path=output_path,
        device="cpu",
        mode="all",
    )

    assert output_path.exists()
    assert result.columns == module._RESULT_COLUMNS
    assert result.height == 14
    assert sorted(result["dataset_id"].to_list()) == [
        "hill_of_towie_multivariate_knn6",
        "hill_of_towie_multivariate_knn6_power_stats",
        "hill_of_towie_univariate",
        "hill_of_towie_univariate_power_stats",
        "kelmarsh_multivariate_knn6",
        "kelmarsh_multivariate_knn6_power_stats",
        "kelmarsh_univariate",
        "kelmarsh_univariate_power_stats",
        "penmanshiel_multivariate_knn6",
        "penmanshiel_multivariate_knn6_power_stats",
        "penmanshiel_univariate",
        "penmanshiel_univariate_power_stats",
        "sdwpf_kddcup_multivariate_knn6",
        "sdwpf_kddcup_univariate",
    ]


def test_run_experiment_univariate_mode_includes_power_stats_rows(monkeypatch, tmp_path) -> None:
    module = _load_module()
    _install_run_experiment_fakes(monkeypatch, module)

    result = module.run_experiment(
        dataset_ids=("kelmarsh", "penmanshiel", "hill_of_towie", "sdwpf_kddcup"),
        output_path=tmp_path / "chronos-2.csv",
        device="cpu",
        mode="univariate",
    )

    assert result.height == 7
    assert sorted(result["dataset_id"].to_list()) == [
        "hill_of_towie_univariate",
        "hill_of_towie_univariate_power_stats",
        "kelmarsh_univariate",
        "kelmarsh_univariate_power_stats",
        "penmanshiel_univariate",
        "penmanshiel_univariate_power_stats",
        "sdwpf_kddcup_univariate",
    ]


def test_run_experiment_multivariate_mode_writes_expected_rows(monkeypatch, tmp_path) -> None:
    module = _load_module()
    _install_run_experiment_fakes(monkeypatch, module)

    result = module.run_experiment(
        dataset_ids=("kelmarsh", "penmanshiel", "hill_of_towie", "sdwpf_kddcup"),
        output_path=tmp_path / "chronos-2.csv",
        device="cpu",
        mode="multivariate_knn6",
    )

    assert result.height == 7
    assert sorted(result["dataset_id"].to_list()) == [
        "hill_of_towie_multivariate_knn6",
        "hill_of_towie_multivariate_knn6_power_stats",
        "kelmarsh_multivariate_knn6",
        "kelmarsh_multivariate_knn6_power_stats",
        "penmanshiel_multivariate_knn6",
        "penmanshiel_multivariate_knn6_power_stats",
        "sdwpf_kddcup_multivariate_knn6",
    ]


def test_run_experiment_univariate_power_stats_mode_filters_supported_datasets(monkeypatch, tmp_path) -> None:
    module = _load_module()
    _install_run_experiment_fakes(monkeypatch, module)

    result = module.run_experiment(
        dataset_ids=("kelmarsh", "penmanshiel", "hill_of_towie", "sdwpf_kddcup"),
        output_path=tmp_path / "chronos-2.csv",
        device="cpu",
        mode="univariate_power_stats",
    )

    assert result.height == 3
    assert sorted(result["dataset_id"].to_list()) == [
        "hill_of_towie_univariate_power_stats",
        "kelmarsh_univariate_power_stats",
        "penmanshiel_univariate_power_stats",
    ]


def test_run_experiment_univariate_power_stats_rejects_unsupported_only_dataset(monkeypatch, tmp_path) -> None:
    module = _load_module()
    _install_run_experiment_fakes(monkeypatch, module)

    try:
        module.run_experiment(
            dataset_ids=("sdwpf_kddcup",),
            output_path=tmp_path / "chronos-2.csv",
            device="cpu",
            mode="univariate_power_stats",
        )
    except ValueError as exc:
        assert "requires at least one dataset with power_stats support" in str(exc)
    else:
        raise AssertionError("Expected univariate_power_stats mode to reject dataset selections without support.")

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


def test_resolve_pipeline_batch_size_counts_series_not_windows() -> None:
    module = _load_module()

    assert module.resolve_pipeline_batch_size(np.zeros((32, 1, 144), dtype=np.float32)) == 32
    assert module.resolve_pipeline_batch_size(np.zeros((32, 6, 144), dtype=np.float32)) == 192


def test_run_experiment_all_mode_writes_expected_rows(monkeypatch, tmp_path) -> None:
    module = _load_module()

    class _FakePipeline:
        pass

    def _fake_evaluate_univariate_dataset(dataset_id, **kwargs):
        resolved = module.resolve_dataset_id(dataset_id)
        return {
            "dataset_id": f"{resolved}{module.UNIVARIATE_SUFFIX}",
            "model_id": module.MODEL_ID,
            "task_id": module.TASK_ID,
            "history_steps": 144,
            "forecast_steps": 36,
            "stride_steps": 36,
            "target_policy": module.TARGET_POLICY,
            "window_count": 10,
            "prediction_count": 360,
            "start_timestamp": "2024-01-01 00:10:00",
            "end_timestamp": "2024-01-10 00:00:00",
            "mae_kw": 1.0,
            "rmse_kw": 2.0,
            "mae_pu": 0.1,
            "rmse_pu": 0.2,
            "device": "cpu",
            "runtime_seconds": 0.5,
        }

    def _fake_evaluate_multivariate_knn6_dataset(dataset_id, **kwargs):
        resolved = module.resolve_dataset_id(dataset_id)
        return {
            "dataset_id": f"{resolved}{module.MULTIVARIATE_KNN6_SUFFIX}",
            "model_id": module.MODEL_ID,
            "task_id": module.TASK_ID,
            "history_steps": 144,
            "forecast_steps": 36,
            "stride_steps": 36,
            "target_policy": module.TARGET_POLICY,
            "window_count": 11,
            "prediction_count": 361,
            "start_timestamp": "2024-02-01 00:10:00",
            "end_timestamp": "2024-02-10 00:00:00",
            "mae_kw": 3.0,
            "rmse_kw": 4.0,
            "mae_pu": 0.3,
            "rmse_pu": 0.4,
            "device": "cpu",
            "runtime_seconds": 0.6,
        }

    monkeypatch.setattr(module, "load_pipeline", lambda **kwargs: _FakePipeline())
    monkeypatch.setattr(module, "evaluate_univariate_dataset", _fake_evaluate_univariate_dataset)
    monkeypatch.setattr(module, "evaluate_multivariate_knn6_dataset", _fake_evaluate_multivariate_knn6_dataset)

    output_path = tmp_path / "chronos-2.csv"
    result = module.run_experiment(
        dataset_ids=("kelmarsh", "penmanshiel", "hill_of_towie", "sdwpf_kddcup"),
        output_path=output_path,
        device="cpu",
        mode="all",
    )

    assert output_path.exists()
    assert result.columns == module._RESULT_COLUMNS
    assert result.height == 8
    assert sorted(result["dataset_id"].to_list()) == [
        "hill_of_towie_multivariate_knn6",
        "hill_of_towie_univariate",
        "kelmarsh_multivariate_knn6",
        "kelmarsh_univariate",
        "penmanshiel_multivariate_knn6",
        "penmanshiel_univariate",
        "sdwpf_kddcup_multivariate_knn6",
        "sdwpf_kddcup_univariate",
    ]

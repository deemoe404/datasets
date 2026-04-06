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


def test_clip_target_values_clips_negative_and_over_rated_and_preserves_nan() -> None:
    module = _load_module()

    clipped = module.clip_target_values([100.0, -5.0, 2600.0, None], rated_power_kw=2050.0)

    assert clipped.tolist()[:3] == [100.0, 0.0, 2050.0]
    assert np.isnan(clipped[3])


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


def test_evaluate_dataset_uses_clipped_targets_and_skips_null_context_windows(monkeypatch) -> None:
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
    )
    resolved_task = task_spec.resolve(spec.resolution_minutes)
    timestamps = pl.datetime_range(
        start=datetime(2024, 1, 1, 0, 0, 0),
        end=datetime(2024, 1, 1, 1, 50, 0),
        interval="10m",
        eager=True,
    )
    series = pl.DataFrame(
        {
            "dataset": ["kelmarsh"] * len(timestamps),
            "turbine_id": ["Kelmarsh 1"] * len(timestamps),
            "timestamp": timestamps,
            "target_kw": [50.0, -10.0, 2200.0, 100.0, 110.0, None, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0],
            "is_observed": [True] * len(timestamps),
            "quality_flags": [""] * len(timestamps),
        }
    )
    window_index = pl.DataFrame(
        {
            "turbine_id": ["Kelmarsh 1", "Kelmarsh 1", "Kelmarsh 1"],
            "input_end_ts": [timestamps[2], timestamps[4], timestamps[7]],
            "is_complete_input": [True, False, True],
            "is_complete_output": [True, True, True],
        }
    )

    def _fake_load_dataset_inputs(dataset_id, *, cache_root, task_spec):
        assert dataset_id == "kelmarsh"
        return spec, resolved_task, series, window_index

    class _FakePipeline:
        def __init__(self) -> None:
            self.calls: list[list[np.ndarray]] = []

        def predict_quantiles(self, *, inputs, prediction_length, quantile_levels, batch_size, limit_prediction_length):
            assert prediction_length == resolved_task.forecast_steps
            assert quantile_levels == [0.5]
            self.calls.append([np.asarray(item) for item in inputs])
            predictions = []
            for item in inputs:
                last_value = float(np.asarray(item)[-1])
                predictions.append(np.full((1, prediction_length, 1), last_value, dtype=np.float32))
            return predictions, predictions

    pipeline = _FakePipeline()
    monkeypatch.setattr(module, "load_dataset_inputs", _fake_load_dataset_inputs)

    result = module.evaluate_dataset(
        "kelmarsh",
        pipeline=pipeline,
        task_spec=task_spec,
        batch_size=8,
        device="cpu",
    )

    assert len(pipeline.calls) == 1
    assert len(pipeline.calls[0]) == 1
    np.testing.assert_allclose(pipeline.calls[0][0], np.array([50.0, 0.0, 2050.0], dtype=np.float32))
    assert result["window_count"] == 1
    assert result["prediction_count"] == resolved_task.forecast_steps
    assert result["mae_kw"] == 1945.0
    assert result["rmse_kw"] == np.sqrt(((1950.0**2) + (1940.0**2)) / 2)
    assert result["mae_pu"] == 1945.0 / 2050.0


def test_build_farm_panel_reindexes_full_grid_and_preserves_nan() -> None:
    module = _load_module()
    timestamps = pl.datetime_range(
        start=datetime(2024, 1, 1, 0, 0, 0),
        end=datetime(2024, 1, 1, 0, 30, 0),
        interval="10m",
        eager=True,
    )
    series = pl.DataFrame(
        {
            "dataset": ["kelmarsh"] * 6,
            "turbine_id": ["T1", "T1", "T1", "T2", "T2", "T2"],
            "timestamp": [timestamps[0], timestamps[1], timestamps[3], timestamps[0], timestamps[2], timestamps[3]],
            "target_kw": [10.0, 20.0, 40.0, 100.0, 120.0, 130.0],
            "target_kw_clipped": [10.0, 20.0, 40.0, 100.0, 120.0, 130.0],
        }
    )

    farm_panel = module.build_farm_panel(series, turbine_ids=("T1", "T2"), resolution_minutes=10)

    assert farm_panel.target_kw_clipped.shape == (4, 2)
    np.testing.assert_allclose(farm_panel.target_kw_clipped[0], np.array([10.0, 100.0], dtype=np.float32))
    assert np.isnan(farm_panel.target_kw_clipped[1, 1])
    assert np.isnan(farm_panel.target_kw_clipped[2, 0])
    np.testing.assert_allclose(farm_panel.target_kw_clipped[3], np.array([40.0, 130.0], dtype=np.float32))


def test_evaluate_multivariate_dataset_masks_nan_future_targets(monkeypatch) -> None:
    module = _load_module()
    spec = DatasetSpec(
        dataset_id="kelmarsh",
        source_root=Path("."),
        resolution_minutes=10,
        turbine_ids=("Kelmarsh 1", "Kelmarsh 2"),
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
            "dataset": ["kelmarsh"] * 12,
            "turbine_id": ["Kelmarsh 1"] * 6 + ["Kelmarsh 2"] * 6,
            "timestamp": list(timestamps) + list(timestamps),
            "target_kw": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 100.0, None, 120.0, None, 140.0, 150.0],
            "is_observed": [True] * 12,
            "quality_flags": [""] * 12,
        }
    )

    def _fake_load_dataset_inputs(dataset_id, *, cache_root, task_spec):
        assert dataset_id == "kelmarsh"
        return spec, resolved_task, series, pl.DataFrame()

    class _FakePipeline:
        def __init__(self) -> None:
            self.calls: list[list[np.ndarray]] = []

        def predict_quantiles(self, *, inputs, prediction_length, quantile_levels, batch_size, limit_prediction_length, cross_learning):
            assert prediction_length == resolved_task.forecast_steps
            assert quantile_levels == [0.5]
            assert cross_learning is False
            self.calls.append([np.asarray(item) for item in inputs])
            predictions = []
            for item in inputs:
                item_array = np.asarray(item, dtype=np.float32)
                forecast = np.array(
                    [
                        [35.0, 35.0],
                        [130.0, 130.0],
                    ],
                    dtype=np.float32,
                )
                assert item_array.shape == (2, resolved_task.history_steps)
                predictions.append(forecast[:, :, None])
            return predictions, predictions

    pipeline = _FakePipeline()
    monkeypatch.setattr(module, "load_dataset_inputs", _fake_load_dataset_inputs)

    result = module.evaluate_multivariate_dataset(
        "kelmarsh",
        pipeline=pipeline,
        task_spec=task_spec,
        batch_size=4,
        device="cpu",
    )

    assert len(pipeline.calls) == 1
    assert len(pipeline.calls[0]) == 1
    np.testing.assert_allclose(
        pipeline.calls[0][0],
        np.array([[10.0, 20.0, 30.0], [100.0, np.nan, 120.0]], dtype=np.float32),
        equal_nan=True,
    )
    expected_squared_error = (5.0**2) + (15.0**2) + (10.0**2)
    assert result["dataset_id"] == "kelmarsh_multivariate"
    assert result["window_count"] == 1
    assert result["prediction_count"] == 3
    assert result["mae_kw"] == 10.0
    assert result["rmse_kw"] == np.sqrt(expected_squared_error / 3.0)
    assert result["mae_pu"] == 10.0 / 2050.0


def test_run_experiment_writes_expected_csv_schema(monkeypatch, tmp_path) -> None:
    module = _load_module()

    class _FakePipeline:
        pass

    def _fake_evaluate_dataset(dataset_id, **kwargs):
        return {
            "dataset_id": dataset_id,
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

    monkeypatch.setattr(module, "load_pipeline", lambda **kwargs: _FakePipeline())
    monkeypatch.setattr(module, "evaluate_dataset", _fake_evaluate_dataset)

    output_path = tmp_path / "chronos-2.csv"
    result = module.run_experiment(
        dataset_ids=("kelmarsh", "penmanshiel", "hill_of_towie"),
        output_path=output_path,
        device="cpu",
    )

    assert output_path.exists()
    assert result.columns == module._RESULT_COLUMNS
    assert result.height == 3


def test_run_experiment_append_multivariate_merges_existing_csv(monkeypatch, tmp_path) -> None:
    module = _load_module()

    existing = pl.DataFrame(
        [
            {
                "dataset_id": dataset_id,
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
            for dataset_id in ("kelmarsh", "penmanshiel", "hill_of_towie")
        ]
    ).select(module._RESULT_COLUMNS)

    class _FakePipeline:
        pass

    def _fake_evaluate_multivariate_dataset(dataset_id, **kwargs):
        return {
            "dataset_id": f"{dataset_id}{module.MULTIVARIATE_SUFFIX}",
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
    monkeypatch.setattr(module, "evaluate_multivariate_dataset", _fake_evaluate_multivariate_dataset)

    output_path = tmp_path / "chronos-2.csv"
    existing.write_csv(output_path)
    result = module.run_experiment(
        dataset_ids=("kelmarsh", "penmanshiel", "hill_of_towie"),
        output_path=output_path,
        device="cpu",
        mode="append_multivariate",
    )

    assert output_path.exists()
    assert result.columns == module._RESULT_COLUMNS
    assert result.height == 6
    assert sorted(result["dataset_id"].to_list()) == [
        "hill_of_towie",
        "hill_of_towie_multivariate",
        "kelmarsh",
        "kelmarsh_multivariate",
        "penmanshiel",
        "penmanshiel_multivariate",
    ]

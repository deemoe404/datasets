from __future__ import annotations

from datetime import datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np
import polars as pl

from wind_datasets.models import DatasetSpec, TaskSpec


def _load_manifest_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "experiment"
        / "chronos-2-exogenous"
        / "chronos2_exogenous_manifest.py"
    )
    spec = spec_from_file_location("chronos2_exogenous_manifest", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "experiment"
        / "chronos-2-exogenous"
        / "chronos2_exogenous.py"
    )
    spec = spec_from_file_location("chronos2_exogenous", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _synthetic_spec(dataset_id: str, turbine_ids: tuple[str, ...]) -> DatasetSpec:
    return DatasetSpec(
        dataset_id=dataset_id,
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


def _synthetic_task(resolution_minutes: int = 10):
    task_spec = TaskSpec(
        history_duration="30m",
        forecast_duration="20m",
        stride_duration="20m",
        task_id="synthetic_task",
        granularity="turbine",
    )
    return task_spec, task_spec.resolve(resolution_minutes)


def _single_turbine_series(
    *,
    dataset_id: str,
    turbine_id: str,
    covariate_columns: tuple[str, ...],
    bool_columns: set[str] | None = None,
) -> pl.DataFrame:
    bool_columns = bool_columns or set()
    timestamps = pl.datetime_range(
        start=datetime(2024, 1, 1, 0, 0, 0),
        end=datetime(2024, 1, 1, 0, 50, 0),
        interval="10m",
        eager=True,
    )
    rows: dict[str, object] = {
        "dataset": [dataset_id] * len(timestamps),
        "turbine_id": [turbine_id] * len(timestamps),
        "timestamp": timestamps,
        "target_kw": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "quality_flags": ["", "", "", "", "flagged", ""],
        "feature_quality_flags": [""] * len(timestamps),
    }
    for column_index, column in enumerate(covariate_columns, start=1):
        if column in bool_columns:
            rows[column] = [False, True, False, True, False, True]
        else:
            base = float(column_index * 10)
            rows[column] = [base + step for step in range(len(timestamps))]
    return pl.DataFrame(rows)


def test_covariate_pack_selected_columns_include_present_optional_only() -> None:
    manifest = _load_manifest_module()
    pack = manifest.resolve_covariate_pack("kelmarsh", "stage3_regime")
    available = set(pack.required_columns) | {"evt_any_active", "evt_active_count", "farm_evt_warning_active"}

    selected = pack.selected_covariate_columns(available)

    assert selected[: len(pack.required_columns)] == pack.required_columns
    assert selected[-3:] == ("evt_any_active", "evt_active_count", "farm_evt_warning_active")


def test_covariate_pack_selected_columns_reject_missing_required() -> None:
    manifest = _load_manifest_module()
    pack = manifest.resolve_covariate_pack("sdwpf_kddcup", "stage2_ops")
    available = set(pack.required_columns[:-1])

    try:
        pack.selected_covariate_columns(available)
    except ValueError as exc:
        assert "missing required covariates" in str(exc)
    else:
        raise AssertionError("Expected required-column validation to fail.")


def test_prepare_exogenous_series_masks_target_but_preserves_covariates() -> None:
    module = _load_module()
    timestamps = pl.datetime_range(
        start=datetime(2024, 1, 1, 0, 0, 0),
        end=datetime(2024, 1, 1, 0, 20, 0),
        interval="10m",
        eager=True,
    )
    series = pl.DataFrame(
        {
            "dataset": ["kelmarsh"] * 3,
            "turbine_id": ["Kelmarsh 1"] * 3,
            "timestamp": timestamps,
            "target_kw": [10.0, None, 2200.0],
            "quality_flags": ["", "", "conflict_resolved"],
            "feature_quality_flags": ["", "", ""],
            "evt_any_active": [False, True, None],
            "Wind speed (m/s)": [1.0, None, 3.0],
        }
    )

    prepared = module.prepare_exogenous_series(
        series,
        covariate_columns=("evt_any_active", "Wind speed (m/s)"),
        rated_power_kw=2050.0,
    )

    np.testing.assert_allclose(
        prepared["target_kw_masked"].to_numpy(),
        np.array([10.0, np.nan, np.nan], dtype=np.float64),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        prepared["evt_any_active"].to_numpy(),
        np.array([0.0, 1.0, np.nan], dtype=np.float64),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        prepared["Wind speed (m/s)"].to_numpy(),
        np.array([1.0, np.nan, 3.0], dtype=np.float64),
        equal_nan=True,
    )


def test_resolve_window_batch_size_and_pipeline_batch_size_track_series_budget() -> None:
    module = _load_module()

    assert module.resolve_window_batch_size(series_budget=1024, covariate_count=0) == 1024
    assert module.resolve_window_batch_size(series_budget=1024, covariate_count=15) == 64
    assert module.resolve_window_batch_size(series_budget=4, covariate_count=7) == 1
    assert (
        module.resolve_pipeline_batch_size(
            [
                {
                    "target": np.zeros(144, dtype=np.float32),
                    "past_covariates": {f"c{i}": np.zeros(144, dtype=np.float32) for i in range(7)},
                }
            ]
        )
        == 8
    )


def test_iter_univariate_covariate_batches_builds_past_covariates() -> None:
    module = _load_module()
    turbine_series = module.TurbineExogenousSeries(
        timestamps_us=np.array([0, 600_000_000, 1_200_000_000, 1_800_000_000, 2_400_000_000, 3_000_000_000], dtype=np.int64),
        target_kw_masked=np.array([10.0, 20.0, 30.0, 40.0, np.nan, 60.0], dtype=np.float32),
        past_covariates={
            "Wspd": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
            "evt_any_active": np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        },
    )

    batches = list(
        module._iter_univariate_covariate_batches(
            turbine_series=turbine_series,
            covariate_columns=("Wspd", "evt_any_active"),
            history_steps=3,
            forecast_steps=2,
            stride_steps=2,
            window_batch_size=8,
        )
    )

    assert len(batches) == 1
    input_batch, actual_batch, future_timestamps_batch = batches[0]
    assert len(input_batch) == 1
    np.testing.assert_allclose(
        input_batch[0]["target"],
        np.array([10.0, 20.0, 30.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        input_batch[0]["past_covariates"]["Wspd"],
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        input_batch[0]["past_covariates"]["evt_any_active"],
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        actual_batch,
        np.array([[40.0, np.nan]], dtype=np.float64),
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        future_timestamps_batch,
        np.array([[1_800_000_000, 2_400_000_000]], dtype=np.int64),
    )


def test_evaluate_univariate_covariate_pack_kelmarsh_stage1_core(monkeypatch) -> None:
    module = _load_module()
    manifest = _load_manifest_module()
    pack = manifest.resolve_covariate_pack("kelmarsh", "stage1_core")
    spec = _synthetic_spec("kelmarsh", ("Kelmarsh 1",))
    task_spec, resolved_task = _synthetic_task()
    series = _single_turbine_series(
        dataset_id="kelmarsh",
        turbine_id="Kelmarsh 1",
        covariate_columns=pack.required_columns,
    )

    class _FakePipeline:
        def __init__(self) -> None:
            self.calls: list[list[dict[str, object]]] = []
            self.batch_sizes: list[int] = []

        def predict_quantiles(self, *, inputs, prediction_length, quantile_levels, batch_size, limit_prediction_length):
            assert prediction_length == resolved_task.forecast_steps
            assert quantile_levels == [0.5]
            self.calls.append(inputs)
            self.batch_sizes.append(batch_size)
            return [np.array([[[40.0], [55.0]]], dtype=np.float32)], [np.array([[40.0, 55.0]], dtype=np.float32)]

    pipeline = _FakePipeline()
    monkeypatch.setattr(module, "resolve_dataset_task", lambda dataset_id, *, task_spec=None: (spec, resolved_task))
    monkeypatch.setattr(
        module,
        "load_covariate_series_frame",
        lambda dataset_id, *, pack, cache_root, turbine_ids=None: (spec, pack.feature_set, Path("series.parquet"), pack.required_columns, series),
    )

    result = module.evaluate_univariate_covariate_pack(
        "kelmarsh",
        pack=pack,
        pipeline=pipeline,
        task_spec=task_spec,
        series_budget=1024,
        device="cpu",
    )

    assert len(pipeline.calls) == 1
    assert len(pipeline.calls[0][0]["past_covariates"]) == len(pack.required_columns)
    assert pipeline.batch_sizes == [14]
    assert result["dataset_id"] == "kelmarsh"
    assert result["covariate_stage"] == "stage1_core"
    assert result["covariate_pack"] == "stage1_core"
    assert result["feature_set"] == "lightweight"
    assert result["covariate_count"] == len(pack.required_columns)
    assert result["window_count"] == 1
    assert result["prediction_count"] == 1
    assert result["mae_kw"] == 0.0


def test_evaluate_univariate_covariate_pack_penmanshiel_stage2_ops(monkeypatch) -> None:
    module = _load_module()
    manifest = _load_manifest_module()
    pack = manifest.resolve_covariate_pack("penmanshiel", "stage2_ops")
    spec = _synthetic_spec("penmanshiel", ("Penmanshiel 11",))
    task_spec, resolved_task = _synthetic_task()
    series = _single_turbine_series(
        dataset_id="penmanshiel",
        turbine_id="Penmanshiel 11",
        covariate_columns=pack.required_columns,
    )

    class _FakePipeline:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def predict_quantiles(self, *, inputs, prediction_length, quantile_levels, batch_size, limit_prediction_length):
            self.batch_sizes.append(batch_size)
            return [np.array([[[40.0], [55.0]]], dtype=np.float32)], [np.array([[40.0, 55.0]], dtype=np.float32)]

    pipeline = _FakePipeline()
    monkeypatch.setattr(module, "resolve_dataset_task", lambda dataset_id, *, task_spec=None: (spec, resolved_task))
    monkeypatch.setattr(
        module,
        "load_covariate_series_frame",
        lambda dataset_id, *, pack, cache_root, turbine_ids=None: (spec, pack.feature_set, Path("series.parquet"), pack.required_columns, series),
    )

    result = module.evaluate_univariate_covariate_pack(
        "penmanshiel",
        pack=pack,
        pipeline=pipeline,
        task_spec=task_spec,
        device="cpu",
    )

    assert pipeline.batch_sizes == [22]
    assert result["covariate_stage"] == "stage2_ops"
    assert result["covariate_count"] == len(pack.required_columns)


def test_evaluate_univariate_covariate_pack_hill_stage3_regime(monkeypatch) -> None:
    module = _load_module()
    manifest = _load_manifest_module()
    pack = manifest.resolve_covariate_pack("hill_of_towie", "stage3_regime")
    spec = _synthetic_spec("hill_of_towie", ("T01",))
    task_spec, resolved_task = _synthetic_task()
    bool_columns = {"aeroup_in_install_window", "aeroup_post_install", "tuneup_in_deployment_window", "tuneup_post_effective"}
    series = _single_turbine_series(
        dataset_id="hill_of_towie",
        turbine_id="T01",
        covariate_columns=pack.required_columns,
        bool_columns=bool_columns,
    )

    class _FakePipeline:
        def __init__(self) -> None:
            self.first_call: dict[str, object] | None = None
            self.batch_sizes: list[int] = []

        def predict_quantiles(self, *, inputs, prediction_length, quantile_levels, batch_size, limit_prediction_length):
            self.first_call = inputs[0]
            self.batch_sizes.append(batch_size)
            return [np.array([[[40.0], [55.0]]], dtype=np.float32)], [np.array([[40.0, 55.0]], dtype=np.float32)]

    pipeline = _FakePipeline()
    monkeypatch.setattr(module, "resolve_dataset_task", lambda dataset_id, *, task_spec=None: (spec, resolved_task))
    monkeypatch.setattr(
        module,
        "load_covariate_series_frame",
        lambda dataset_id, *, pack, cache_root, turbine_ids=None: (spec, pack.feature_set, Path("series.parquet"), pack.required_columns, series),
    )

    result = module.evaluate_univariate_covariate_pack(
        "hill_of_towie",
        pack=pack,
        pipeline=pipeline,
        task_spec=task_spec,
        device="cpu",
    )

    assert pipeline.first_call is not None
    np.testing.assert_allclose(
        pipeline.first_call["past_covariates"]["aeroup_in_install_window"],
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )
    assert pipeline.batch_sizes == [31]
    assert result["covariate_stage"] == "stage3_regime"
    assert result["covariate_count"] == len(pack.required_columns)


def test_evaluate_univariate_covariate_pack_sdwpf_stage3_regime(monkeypatch) -> None:
    module = _load_module()
    manifest = _load_manifest_module()
    pack = manifest.resolve_covariate_pack("sdwpf_kddcup", "stage3_regime")
    spec = _synthetic_spec("sdwpf_kddcup", ("1",))
    task_spec, resolved_task = _synthetic_task()
    bool_columns = {"sdwpf_is_unknown", "sdwpf_is_abnormal", "sdwpf_is_masked"}
    series = _single_turbine_series(
        dataset_id="sdwpf_kddcup",
        turbine_id="1",
        covariate_columns=pack.required_columns,
        bool_columns=bool_columns,
    )

    class _FakePipeline:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def predict_quantiles(self, *, inputs, prediction_length, quantile_levels, batch_size, limit_prediction_length):
            self.batch_sizes.append(batch_size)
            return [np.array([[[40.0], [55.0]]], dtype=np.float32)], [np.array([[40.0, 55.0]], dtype=np.float32)]

    pipeline = _FakePipeline()
    monkeypatch.setattr(module, "resolve_dataset_task", lambda dataset_id, *, task_spec=None: (spec, resolved_task))
    monkeypatch.setattr(
        module,
        "load_covariate_series_frame",
        lambda dataset_id, *, pack, cache_root, turbine_ids=None: (spec, pack.feature_set, Path("series.parquet"), pack.required_columns, series),
    )

    result = module.evaluate_univariate_covariate_pack(
        "sdwpf_kddcup",
        pack=pack,
        pipeline=pipeline,
        task_spec=task_spec,
        device="cpu",
    )

    assert pipeline.batch_sizes == [13]
    assert result["covariate_stage"] == "stage3_regime"
    assert result["covariate_count"] == len(pack.required_columns)


def test_run_experiment_includes_reference_and_stage_rows(monkeypatch, tmp_path) -> None:
    module = _load_module()

    class _FakePipeline:
        pass

    def _fake_evaluate(dataset_id, *, pack, **kwargs):
        return {
            "dataset_id": dataset_id,
            "model_id": module.MODEL_ID,
            "task_id": module.TASK_ID,
            "history_steps": 144,
            "forecast_steps": 36,
            "stride_steps": 36,
            "target_policy": module.TARGET_POLICY,
            "window_count": 1,
            "prediction_count": 1,
            "start_timestamp": "2024-01-01 00:10:00",
            "end_timestamp": "2024-01-01 00:10:00",
            "mae_kw": 1.0,
            "rmse_kw": 1.0,
            "mae_pu": 0.1,
            "rmse_pu": 0.1,
            "device": "cpu",
            "runtime_seconds": 0.1,
            "layout": module.LAYOUT,
            "covariate_stage": pack.stage,
            "covariate_pack": pack.pack_name,
            "feature_set": pack.feature_set,
            "covariate_count": len(pack.required_columns),
            "covariate_policy": module.COVARIATE_POLICY,
        }

    monkeypatch.setattr(module, "load_pipeline", lambda **kwargs: _FakePipeline())
    monkeypatch.setattr(module, "evaluate_univariate_covariate_pack", _fake_evaluate)

    output_path = tmp_path / "chronos-2-exogenous.csv"
    result = module.run_experiment(
        dataset_ids=("kelmarsh", "sdwpf_kddcup"),
        output_path=output_path,
        device="cpu",
        include_power_only_reference=True,
    )

    assert output_path.exists()
    assert result.columns == module._RESULT_COLUMNS
    assert result.height == 8
    assert sorted(result["covariate_stage"].to_list()) == [
        "reference",
        "reference",
        "stage1_core",
        "stage1_core",
        "stage2_ops",
        "stage2_ops",
        "stage3_regime",
        "stage3_regime",
    ]

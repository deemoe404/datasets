from __future__ import annotations

from datetime import datetime, timedelta
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import polars as pl

from wind_datasets.models import DatasetSpec, TaskSpec


def _load_manifest_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "experiment"
        / "families"
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
        / "families"
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


def _timestamp_us(timestamps) -> list[int]:
    return pl.Series("timestamp", list(timestamps), dtype=pl.Datetime).cast(pl.Int64).to_list()


def _patch_split_windows(
    monkeypatch,
    module,
    timestamps,
    *,
    target_indices: Sequence[int],
    turbine_indices: Sequence[int] | None = None,
    train_window_count: int = 10,
    val_window_count: int = 5,
):
    timestamp_us = _timestamp_us(timestamps)
    resolved_turbine_indices = tuple(turbine_indices or [0] * len(target_indices))
    windows = module.WindowDescriptorIndex(
        turbine_indices=np.asarray(resolved_turbine_indices, dtype=np.int32),
        target_indices=np.asarray(list(target_indices), dtype=np.int32),
        output_start_us=np.asarray([timestamp_us[index] for index in target_indices], dtype=np.int64),
        output_end_us=np.asarray([timestamp_us[index + 1] for index in target_indices], dtype=np.int64),
    )
    split_windows = module.PreparedSplitWindowSet(
        train_window_count=train_window_count,
        val_window_count=val_window_count,
        test_window_count=len(target_indices),
        test_rolling_windows=windows,
        test_non_overlap_windows=windows,
    )
    monkeypatch.setattr(
        module,
        "ensure_task_cache",
        lambda dataset_id, *, cache_root=module._CACHE_ROOT: module.TaskCachePaths(
            task_window_path=Path("task_windows.parquet"),
            dataset_path=Path("dataset.parquet"),
        ),
    )
    monkeypatch.setattr(module, "load_strict_window_index", lambda *args, **kwargs: pl.DataFrame())
    monkeypatch.setattr(module, "prepare_split_window_set", lambda *args, **kwargs: split_windows)
    return split_windows


def _rows_frame(module, rows) -> pl.DataFrame:
    if isinstance(rows, pl.DataFrame):
        return rows
    return module.sort_result_frame(pl.DataFrame(rows).select(module._RESULT_COLUMNS))


def _overall_row(module, rows, *, eval_protocol: str | None = None) -> dict[str, object]:
    frame = _rows_frame(module, rows)
    selected_protocol = eval_protocol or module.ROLLING_EVAL_PROTOCOL
    return frame.filter(
        (pl.col("eval_protocol") == selected_protocol)
        & (pl.col("metric_scope") == module.OVERALL_METRIC_SCOPE)
    ).to_dicts()[0]


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


def _single_turbine_series_with_targets(
    *,
    dataset_id: str,
    turbine_id: str,
    covariate_columns: tuple[str, ...],
    target_values: Sequence[float],
) -> pl.DataFrame:
    timestamps = pl.datetime_range(
        start=datetime(2024, 1, 1, 0, 0, 0),
        end=datetime(2024, 1, 1, 0, 0, 0) + timedelta(minutes=10 * (len(target_values) - 1)),
        interval="10m",
        eager=True,
    )
    rows: dict[str, object] = {
        "dataset": [dataset_id] * len(target_values),
        "turbine_id": [turbine_id] * len(target_values),
        "timestamp": timestamps,
        "target_kw": list(target_values),
        "quality_flags": [""] * len(target_values),
        "feature_quality_flags": [""] * len(target_values),
    }
    for column_index, column in enumerate(covariate_columns, start=1):
        base = float(column_index * 10)
        rows[column] = [base + step for step in range(len(target_values))]
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


def test_count_retained_windows_matches_batch_iteration_rules() -> None:
    module = _load_module()
    turbine_series = module.TurbineExogenousSeries(
        timestamps_us=np.arange(9, dtype=np.int64) * 600_000_000,
        target_kw_masked=np.array([10.0, 20.0, 30.0, 40.0, 50.0, np.nan, 70.0, 80.0, 90.0], dtype=np.float32),
        past_covariates={},
    )

    counted_windows = module._count_retained_windows_for_turbine(
        turbine_series=turbine_series,
        history_steps=3,
        forecast_steps=2,
        stride_steps=2,
        window_offset=1,
        max_windows_per_dataset=1,
    )
    iterated_windows = sum(
        len(input_batch)
        for input_batch, _, _ in module._iter_univariate_covariate_batches(
            turbine_series=turbine_series,
            covariate_columns=(),
            history_steps=3,
            forecast_steps=2,
            stride_steps=2,
            window_batch_size=8,
            window_offset=1,
            max_windows_per_dataset=1,
        )
    )

    assert counted_windows == 1
    assert iterated_windows == counted_windows


def test_resolve_dataset_progress_plan_counts_batches_from_series_budget(monkeypatch) -> None:
    module = _load_module()
    manifest = _load_manifest_module()
    stage1 = manifest.resolve_covariate_pack("kelmarsh", "stage1_core")
    stage2 = manifest.resolve_covariate_pack("kelmarsh", "stage2_ops")
    spec = _synthetic_spec("kelmarsh", ("Kelmarsh 1",))
    task_spec, resolved_task = _synthetic_task()
    stage2_series = _single_turbine_series_with_targets(
        dataset_id="kelmarsh",
        turbine_id="Kelmarsh 1",
        covariate_columns=stage2.required_columns,
        target_values=(10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0),
    )
    target_series = stage2_series.select(module._BASE_COLUMNS)

    monkeypatch.setattr(module, "resolve_dataset_task", lambda dataset_id, *, task_spec=None: (spec, resolved_task))
    monkeypatch.setattr(
        module,
        "load_target_series_frame",
        lambda dataset_id, *, cache_root, turbine_ids=None: (spec, Path("series.parquet"), target_series),
    )
    monkeypatch.setattr(module.pl, "read_parquet_schema", lambda path: stage2_series.schema)

    plan = module._resolve_dataset_progress_plan(
        "kelmarsh",
        packs=(stage1, stage2),
        task_spec=task_spec,
        series_budget=28,
        turbine_ids=("Kelmarsh 1",),
    )

    assert plan.dataset_id == "kelmarsh"
    assert plan.retained_windows_by_turbine == {"Kelmarsh 1": 2}
    assert [stage_plan.pack_name for stage_plan in plan.stage_plans] == ["stage1_core", "stage2_ops"]
    assert [stage_plan.window_batch_size for stage_plan in plan.stage_plans] == [2, 1]
    assert [stage_plan.total_batches for stage_plan in plan.stage_plans] == [1, 2]
    assert plan.total_batches == 3


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
        lambda dataset_id, *, pack, cache_root, turbine_ids=None: (spec, Path("series.parquet"), pack.required_columns, series),
    )
    _patch_split_windows(monkeypatch, module, series["timestamp"], target_indices=[3])

    result = module.evaluate_univariate_covariate_pack(
        "kelmarsh",
        pack=pack,
        pipeline=pipeline,
        task_spec=task_spec,
        series_budget=1024,
        device="cpu",
        eval_protocols=(module.ROLLING_EVAL_PROTOCOL,),
    )
    overall = _overall_row(module, result)

    assert len(pipeline.calls) == 1
    assert len(pipeline.calls[0][0]["past_covariates"]) == len(pack.required_columns)
    assert pipeline.batch_sizes == [13]
    assert _rows_frame(module, result).height == 3
    assert overall["dataset_id"] == "kelmarsh"
    assert overall["covariate_stage"] == "stage1_core"
    assert overall["covariate_pack"] == "stage1_core"
    assert overall["covariate_count"] == len(pack.required_columns)
    assert overall["window_count"] == 1
    assert overall["prediction_count"] == 1
    assert overall["mae_kw"] == 0.0


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
        lambda dataset_id, *, pack, cache_root, turbine_ids=None: (spec, Path("series.parquet"), pack.required_columns, series),
    )
    _patch_split_windows(monkeypatch, module, series["timestamp"], target_indices=[3])

    result = module.evaluate_univariate_covariate_pack(
        "penmanshiel",
        pack=pack,
        pipeline=pipeline,
        task_spec=task_spec,
        device="cpu",
        eval_protocols=(module.ROLLING_EVAL_PROTOCOL,),
    )
    overall = _overall_row(module, result)

    assert pipeline.batch_sizes == [len(pack.required_columns) + 1]
    assert overall["covariate_stage"] == "stage2_ops"
    assert overall["covariate_count"] == len(pack.required_columns)


def test_evaluate_univariate_covariate_pack_hill_stage3_regime(monkeypatch) -> None:
    module = _load_module()
    manifest = _load_manifest_module()
    pack = manifest.resolve_covariate_pack("hill_of_towie", "stage3_regime")
    spec = _synthetic_spec("hill_of_towie", ("T01",))
    task_spec, resolved_task = _synthetic_task()
    bool_columns = {"aeroup_in_install_window", "aeroup_post_install"}
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
        lambda dataset_id, *, pack, cache_root, turbine_ids=None: (spec, Path("series.parquet"), pack.required_columns, series),
    )
    _patch_split_windows(monkeypatch, module, series["timestamp"], target_indices=[3])

    result = module.evaluate_univariate_covariate_pack(
        "hill_of_towie",
        pack=pack,
        pipeline=pipeline,
        task_spec=task_spec,
        device="cpu",
        eval_protocols=(module.ROLLING_EVAL_PROTOCOL,),
    )
    overall = _overall_row(module, result)

    assert pipeline.first_call is not None
    np.testing.assert_allclose(
        pipeline.first_call["past_covariates"]["aeroup_in_install_window"],
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )
    assert pipeline.batch_sizes == [24]
    assert overall["covariate_stage"] == "stage3_regime"
    assert overall["covariate_count"] == len(pack.required_columns)


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
        lambda dataset_id, *, pack, cache_root, turbine_ids=None: (spec, Path("series.parquet"), pack.required_columns, series),
    )
    _patch_split_windows(monkeypatch, module, series["timestamp"], target_indices=[3])

    result = module.evaluate_univariate_covariate_pack(
        "sdwpf_kddcup",
        pack=pack,
        pipeline=pipeline,
        task_spec=task_spec,
        device="cpu",
        eval_protocols=(module.ROLLING_EVAL_PROTOCOL,),
    )
    overall = _overall_row(module, result)

    assert pipeline.batch_sizes == [13]
    assert overall["covariate_stage"] == "stage3_regime"
    assert overall["covariate_count"] == len(pack.required_columns)


def test_run_experiment_emits_exact_progress_events(monkeypatch, tmp_path) -> None:
    module = _load_module()
    manifest = _load_manifest_module()
    stage1 = manifest.resolve_covariate_pack("kelmarsh", "stage1_core")
    stage2 = manifest.resolve_covariate_pack("kelmarsh", "stage2_ops")
    spec = _synthetic_spec("kelmarsh", ("Kelmarsh 1",))
    task_spec, resolved_task = _synthetic_task()
    stage2_series = _single_turbine_series_with_targets(
        dataset_id="kelmarsh",
        turbine_id="Kelmarsh 1",
        covariate_columns=stage2.required_columns,
        target_values=(10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0),
    )
    target_series = stage2_series.select(module._BASE_COLUMNS)
    progress_events: list[dict[str, object]] = []

    class _FakePipeline:
        def predict_quantiles(self, *, inputs, prediction_length, quantile_levels, batch_size, limit_prediction_length):
            del quantile_levels, batch_size, limit_prediction_length
            forecasts = [
                np.array([[[40.0 + index], [50.0 + index]]], dtype=np.float32)
                for index in range(len(inputs))
            ]
            return forecasts, [np.array([[40.0, 50.0]], dtype=np.float32)] * len(inputs)

    monkeypatch.setattr(module, "resolve_dataset_task", lambda dataset_id, *, task_spec=None: (spec, resolved_task))
    monkeypatch.setattr(
        module,
        "load_covariate_series_frame",
        lambda dataset_id, *, pack, cache_root, turbine_ids=None: (
            spec,
            Path("series.parquet"),
            pack.required_columns,
            stage2_series.select(list(module._BASE_COLUMNS) + list(pack.required_columns)),
        ),
    )
    monkeypatch.setattr(
        module,
        "load_target_series_frame",
        lambda dataset_id, *, cache_root, turbine_ids=None: (spec, Path("series.parquet"), target_series),
    )
    monkeypatch.setattr(module.pl, "read_parquet_schema", lambda path: stage2_series.schema)
    monkeypatch.setattr(
        module,
        "_profile_log",
        lambda dataset_id, phase, **fields: progress_events.append({"dataset_id": dataset_id, "phase": phase, **fields}),
    )
    _patch_split_windows(monkeypatch, module, stage2_series["timestamp"], target_indices=[3])

    output_path = tmp_path / "chronos-2-exogenous.csv"
    result = module.run_experiment(
        dataset_ids=("kelmarsh",),
        covariate_stages=("stage1_core", "stage2_ops"),
        output_path=output_path,
        device="cpu",
        task_spec=task_spec,
        pipeline=_FakePipeline(),
        turbine_ids=("Kelmarsh 1",),
        series_budget=14,
        eval_protocols=(module.ROLLING_EVAL_PROTOCOL,),
        emit_progress_events=True,
    )

    progress_only = [event for event in progress_events if str(event["phase"]).startswith("progress_")]

    assert output_path.exists()
    assert result.height == 6
    assert [event["phase"] for event in progress_only] == [
        "progress_stage_start",
        "progress_batch",
        "progress_stage_complete",
        "progress_stage_start",
        "progress_batch",
        "progress_stage_complete",
    ]
    assert progress_only[0]["chunk_total_batches"] == 1
    assert [event["completed_chunk_batches"] for event in progress_only if event["phase"] == "progress_batch"] == [1, 2]
    assert [event["stage_total_batches"] for event in progress_only if event["phase"] == "progress_stage_start"] == [1, 1]
    assert [event["completed_stage_batches"] for event in progress_only if event["phase"] == "progress_stage_complete"] == [1, 1]
    assert all(event["dataset_id"] == "kelmarsh" for event in progress_only)


def test_run_experiment_includes_reference_and_stage_rows(monkeypatch, tmp_path) -> None:
    module = _load_module()

    class _FakePipeline:
        pass

    def _fake_evaluate(dataset_id, *, pack, **kwargs):
        resolved_task = module.build_task_spec(window_protocol=module.DEFAULT_WINDOW_PROTOCOL).resolve(10)
        base_row = {
            "dataset_id": dataset_id,
            "model_id": module.MODEL_ID,
            "task_id": resolved_task.task_id,
            "window_protocol": module.DEFAULT_WINDOW_PROTOCOL,
            "history_steps": resolved_task.history_steps,
            "forecast_steps": resolved_task.forecast_steps,
            "stride_steps": resolved_task.stride_steps,
            "split_protocol": module.SPLIT_PROTOCOL,
            "split_name": "test",
            "target_policy": module.TARGET_POLICY,
            "start_timestamp": "2024-01-01 00:10:00",
            "end_timestamp": "2024-01-01 00:10:00",
            "device": "cpu",
            "runtime_seconds": 0.1,
            "layout": module.LAYOUT,
            "covariate_stage": pack.stage,
            "covariate_pack": pack.pack_name,
            "covariate_count": len(pack.required_columns),
            "covariate_policy": module.COVARIATE_POLICY,
            "train_window_count": 100,
            "val_window_count": 20,
            "test_window_count": 1,
        }
        rows: list[dict[str, object]] = []
        for eval_protocol in (module.ROLLING_EVAL_PROTOCOL, module.NON_OVERLAP_EVAL_PROTOCOL):
            rows.append(
                {
                    **base_row,
                    "eval_protocol": eval_protocol,
                    "metric_scope": module.OVERALL_METRIC_SCOPE,
                    "lead_step": None,
                    "lead_minutes": None,
                    "window_count": 1,
                    "prediction_count": 1,
                    "mae_kw": 1.0,
                    "rmse_kw": 1.0,
                    "mae_pu": 0.1,
                    "rmse_pu": 0.1,
                }
            )
            for lead_step in range(1, resolved_task.forecast_steps + 1):
                rows.append(
                    {
                        **base_row,
                        "eval_protocol": eval_protocol,
                        "metric_scope": module.HORIZON_METRIC_SCOPE,
                        "lead_step": lead_step,
                        "lead_minutes": lead_step * 10,
                        "window_count": 1,
                        "prediction_count": 1,
                        "mae_kw": 1.0,
                        "rmse_kw": 1.0,
                        "mae_pu": 0.1,
                        "rmse_pu": 0.1,
                    }
                )
        return rows

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
    assert result.height == 8 * 74
    assert result["window_protocol"].unique().to_list() == [module.DEFAULT_WINDOW_PROTOCOL]
    assert sorted(result["covariate_stage"].unique().to_list()) == [
        "reference",
        "stage1_core",
        "stage2_ops",
        "stage3_regime",
    ]


def test_run_experiment_emits_dense_test_only_metadata(monkeypatch, tmp_path) -> None:
    module = _load_module()

    class _FakePipeline:
        pass

    def _fake_evaluate(dataset_id, *, pack, **kwargs):
        del dataset_id, pack, kwargs
        resolved_task = module.build_task_spec(window_protocol=module.DEFAULT_WINDOW_PROTOCOL).resolve(10)
        return [
            {
                "dataset_id": "kelmarsh",
                "model_id": module.MODEL_ID,
                "task_id": resolved_task.task_id,
                "window_protocol": module.DEFAULT_WINDOW_PROTOCOL,
                "history_steps": resolved_task.history_steps,
                "forecast_steps": resolved_task.forecast_steps,
                "stride_steps": resolved_task.stride_steps,
                "split_protocol": module.SPLIT_PROTOCOL,
                "split_name": "test",
                "eval_protocol": module.ROLLING_EVAL_PROTOCOL,
                "metric_scope": module.OVERALL_METRIC_SCOPE,
                "lead_step": None,
                "lead_minutes": None,
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
                "covariate_stage": "stage1_core",
                "covariate_pack": "stage1_core",
                "covariate_count": 1,
                "covariate_policy": module.COVARIATE_POLICY,
                "train_window_count": 100,
                "val_window_count": 20,
                "test_window_count": 1,
            }
        ]

    monkeypatch.setattr(module, "load_pipeline", lambda **kwargs: _FakePipeline())
    monkeypatch.setattr(module, "evaluate_univariate_covariate_pack", _fake_evaluate)

    output_path = tmp_path / "chronos-2-exogenous.csv"
    result = module.run_experiment(
        dataset_ids=("kelmarsh",),
        covariate_stages=("stage1_core",),
        output_path=output_path,
        device="cpu",
    )

    assert output_path.exists()
    assert result["window_protocol"].unique().to_list() == [module.DEFAULT_WINDOW_PROTOCOL]
    assert result["task_id"].unique().to_list() == [module.TASK_ID]
    assert result["split_protocol"].unique().to_list() == [module.SPLIT_PROTOCOL]
    assert result["split_name"].unique().to_list() == ["test"]

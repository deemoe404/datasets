from __future__ import annotations

from datetime import datetime, timedelta
from importlib.util import module_from_spec, spec_from_file_location
import json
import math
from pathlib import Path
import sys

import numpy as np
import polars as pl


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "experiment"
        / "ltsf-linear"
        / "ltsf_linear.py"
    )
    spec = spec_from_file_location("ltsf_linear", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _TqdmRecorder:
    instances: list["_TqdmRecorder"] = []

    def __init__(self, *args, **kwargs) -> None:
        del args
        self.total = kwargs.get("total")
        self.desc = kwargs.get("desc")
        self.disable = kwargs.get("disable")
        self.n = 0
        self.postfixes: list[str] = []
        self.closed = False
        _TqdmRecorder.instances.append(self)

    def update(self, value=1) -> None:
        self.n += int(value)

    def set_postfix_str(self, value: str, refresh: bool = True) -> None:
        del refresh
        self.postfixes.append(value)

    def close(self) -> None:
        self.closed = True


def _build_temp_cache(
    cache_root: Path,
    *,
    dataset_id: str = "toy_dataset",
    history_steps: int = 144,
    forecast_steps: int = 36,
) -> None:
    dataset_root = cache_root / dataset_id
    task_dir = dataset_root / "tasks" / "default" / "turbine" / "next_6h_from_24h"
    default_series_dir = dataset_root / "gold_base" / "default" / "turbine" / "default"
    lightweight_series_dir = dataset_root / "gold_base" / "default" / "turbine" / "lightweight"
    static_dir = dataset_root / "silver" / "meta"
    task_dir.mkdir(parents=True, exist_ok=True)
    default_series_dir.mkdir(parents=True, exist_ok=True)
    lightweight_series_dir.mkdir(parents=True, exist_ok=True)
    static_dir.mkdir(parents=True, exist_ok=True)

    timestamps = pl.datetime_range(
        datetime(2024, 1, 1, 0, 0, 0),
        datetime(2024, 1, 14, 21, 10, 0),
        interval="10m",
        eager=True,
    )
    wind_speed_t01 = [float(index) for index in range(len(timestamps))]
    wind_speed_t02 = [float(index + 5000) for index in range(len(timestamps))]
    ambient_t01 = [float(index % 17) for index in range(len(timestamps))]
    ambient_t02 = [float((index % 17) + 100) for index in range(len(timestamps))]
    status_t01 = [index % 2 == 0 for index in range(len(timestamps))]
    status_t02 = [index % 3 == 0 for index in range(len(timestamps))]
    wind_speed_t01[151] = None
    ambient_t02[152] = None
    status_t01[153] = None

    rows: dict[str, object] = {
        "dataset": [dataset_id] * (len(timestamps) * 2),
        "turbine_id": ["T01"] * len(timestamps) + ["T02"] * len(timestamps),
        "timestamp": list(timestamps) + list(timestamps),
        "target_kw": (
            [float(index) for index in range(len(timestamps))]
            + [float(index + 1000) for index in range(len(timestamps))]
        ),
        "quality_flags": [""] * (len(timestamps) * 2),
        "wind_speed": wind_speed_t01 + wind_speed_t02,
        "ambient_temp": ambient_t01 + ambient_t02,
        "status_flag": status_t01 + status_t02,
    }
    series = pl.DataFrame(rows)
    series.write_parquet(default_series_dir / "series.parquet")
    series.write_parquet(lightweight_series_dir / "series.parquet")

    window_rows = []
    for turbine_id in ("T01", "T02"):
        for start_index in range(history_steps, len(timestamps) - forecast_steps + 1):
            window_rows.append(
                {
                    "dataset": dataset_id,
                    "turbine_id": turbine_id,
                    "output_start_ts": timestamps[start_index],
                    "output_end_ts": timestamps[start_index + forecast_steps - 1],
                    "is_complete_input": True,
                    "is_complete_output": True,
                    "quality_flags": "",
                }
            )
    pl.DataFrame(window_rows).write_parquet(task_dir / "window_index.parquet")

    (task_dir / "task_context.json").write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "quality_profile": "default",
                "turbine_ids": ["T01", "T02"],
                "task": {
                    "task_id": "next_6h_from_24h",
                    "history_steps": history_steps,
                    "forecast_steps": forecast_steps,
                    "stride_steps": 1,
                },
                "series_layout": "turbine",
            }
        ),
        encoding="utf-8",
    )
    pl.DataFrame(
        {
            "turbine_id": ["T01", "T02"],
            "rated_power_kw": [2000.0, 2000.0],
        }
    ).write_parquet(static_dir / "turbine_static.parquet")


def _descriptor(module, *, turbine_indices, target_indices, forecast_steps: int, step_us: int = 600_000_000):
    target_indices_array = np.asarray(target_indices, dtype=np.int32)
    output_start_us = target_indices_array.astype(np.int64) * step_us
    output_end_us = output_start_us + (forecast_steps - 1) * step_us
    return module.WindowDescriptorIndex(
        turbine_indices=np.asarray(turbine_indices, dtype=np.int32),
        target_indices=target_indices_array,
        output_start_us=output_start_us,
        output_end_us=output_end_us,
    )


def _small_prepared_dataset(
    module,
    *,
    dataset_id: str = "kelmarsh",
    covariate_stage: str = "reference",
    covariate_pack: str = "power_only",
    feature_set: str = "default",
    covariate_columns: tuple[str, ...] = (),
    forecast_steps: int = 36,
):
    timestamps_us = np.arange(0, 500 * 600_000_000, 600_000_000, dtype=np.int64)
    turbine_series = (
        module.TurbineSeries(
            timestamps_us=timestamps_us,
            target_pu=np.linspace(0.0, 1.0, len(timestamps_us), dtype=np.float32),
            past_covariates={
                column: (np.linspace(0.0, 1.0, len(timestamps_us), dtype=np.float32) + index)
                for index, column in enumerate(covariate_columns)
            },
        ),
    )
    covariate_count = len(covariate_columns)
    return module.PreparedDataset(
        dataset_id=dataset_id,
        resolution_minutes=10,
        rated_power_kw=2050.0,
        history_steps=144,
        forecast_steps=forecast_steps,
        stride_steps=1,
        covariate_stage=covariate_stage,
        covariate_pack=covariate_pack,
        feature_set=feature_set,
        covariate_columns=covariate_columns,
        covariate_count=covariate_count,
        covariate_policy=module.COVARIATE_POLICY if covariate_count else module.REFERENCE_COVARIATE_POLICY,
        turbine_ids=("T01",),
        turbine_series=turbine_series,
        train_windows=_descriptor(module, turbine_indices=[0, 0, 0, 0], target_indices=[200, 201, 202, 203], forecast_steps=forecast_steps),
        val_rolling_windows=_descriptor(module, turbine_indices=[0, 0], target_indices=[300, 301], forecast_steps=forecast_steps),
        val_non_overlap_windows=_descriptor(module, turbine_indices=[0], target_indices=[300], forecast_steps=forecast_steps),
        test_rolling_windows=_descriptor(module, turbine_indices=[0, 0, 0], target_indices=[360, 361, 362], forecast_steps=forecast_steps),
        test_non_overlap_windows=_descriptor(module, turbine_indices=[0], target_indices=[360], forecast_steps=forecast_steps),
        covariate_means=np.zeros((covariate_count,), dtype=np.float32),
        covariate_stds=np.ones((covariate_count,), dtype=np.float32),
    )


def _evaluation_metrics(module, *, window_count: int, forecast_steps: int, base: float) -> object:
    horizon_window_count = np.full((forecast_steps,), window_count, dtype=np.int64)
    horizon_prediction_count = np.full((forecast_steps,), window_count, dtype=np.int64)
    horizon_values = np.asarray([base + 0.001 * (lead + 1) for lead in range(forecast_steps)], dtype=np.float64)
    return module.EvaluationMetrics(
        window_count=window_count,
        prediction_count=window_count * forecast_steps,
        mae_kw=base * 10,
        rmse_kw=base * 11,
        mae_pu=base,
        rmse_pu=base + 0.01,
        horizon_window_count=horizon_window_count,
        horizon_prediction_count=horizon_prediction_count,
        horizon_mae_kw=horizon_values * 10,
        horizon_rmse_kw=horizon_values * 11,
        horizon_mae_pu=horizon_values,
        horizon_rmse_pu=horizon_values + 0.01,
    )


def _result_row(
    module,
    *,
    dataset_id: str,
    covariate_stage: str,
    covariate_pack: str,
    model_variant: str,
    split_name: str,
    eval_protocol: str,
    metric_scope: str,
    lead_step: int | None,
) -> dict[str, object]:
    lead_minutes = None if lead_step is None else lead_step * 10
    return {
        "dataset_id": dataset_id,
        "model_id": module.MODEL_ID,
        "model_variant": model_variant,
        "task_id": module.TASK_ID,
        "window_protocol": module.WINDOW_PROTOCOL,
        "history_steps": 144,
        "forecast_steps": 36,
        "stride_steps": 1,
        "split_protocol": module.SPLIT_PROTOCOL,
        "split_name": split_name,
        "eval_protocol": eval_protocol,
        "metric_scope": metric_scope,
        "lead_step": lead_step,
        "lead_minutes": lead_minutes,
        "covariate_stage": covariate_stage,
        "covariate_pack": covariate_pack,
        "feature_set": "lightweight" if dataset_id in {"kelmarsh", "penmanshiel"} else "default",
        "covariate_count": 0 if covariate_stage == "reference" else 3,
        "covariate_policy": "none" if covariate_stage == "reference" else module.COVARIATE_POLICY,
        "window_count": 10,
        "prediction_count": 360 if metric_scope == module.OVERALL_METRIC_SCOPE else 10,
        "start_timestamp": "2024-01-01 00:00:00",
        "end_timestamp": "2024-01-02 00:00:00",
        "mae_kw": 1.0,
        "rmse_kw": 1.1,
        "mae_pu": 0.1,
        "rmse_pu": 0.11,
        "device": "cpu",
        "runtime_seconds": 1.0,
        "train_window_count": 70,
        "val_window_count": 10,
        "test_window_count": 20,
        "best_epoch": 2,
        "epochs_ran": 3,
        "best_val_rmse_pu": 0.09,
        "seed": 42,
        "batch_size": 1024,
        "learning_rate": 1e-3,
    }


def test_build_chrono_split_lookup_uses_floor_boundaries() -> None:
    module = _load_module()
    base = datetime(2024, 1, 1, 0, 0, 0)

    lookup = module.build_chrono_split_lookup(
        [base + timedelta(hours=index) for index in range(10)] + [base + timedelta(hours=0)]
    )

    assert lookup.height == 10
    assert lookup["split"].to_list() == [
        "train",
        "train",
        "train",
        "train",
        "train",
        "train",
        "train",
        "val",
        "test",
        "test",
    ]


def test_split_window_index_enforces_strict_containment() -> None:
    module = _load_module()
    raw_timestamps = pl.datetime_range(
        datetime(2024, 1, 1, 0, 0, 0),
        datetime(2024, 1, 14, 21, 10, 0),
        interval="10m",
        eager=True,
    )
    rows = [
        {
            "dataset": "toy",
            "turbine_id": "T01",
            "output_start_ts": raw_timestamps[index],
            "output_end_ts": raw_timestamps[index + module.FORECAST_STEPS - 1],
            "is_complete_input": True,
            "is_complete_output": True,
            "quality_flags": "",
        }
        for index in (144, 1365, 1543, 1544, 1565, 1744)
    ]
    window_index = pl.DataFrame(rows)

    split_frames = module.split_window_index(
        window_index,
        raw_timestamps=raw_timestamps.to_list(),
        resolution_minutes=10,
    )

    assert split_frames["train"]["output_start_ts"].to_list() == [raw_timestamps[144]]
    assert split_frames["val"]["output_start_ts"].to_list() == [raw_timestamps[1544]]
    assert split_frames["test"]["output_start_ts"].to_list() == [raw_timestamps[1744]]


def test_clip_target_values_clips_bounds_and_preserves_nan() -> None:
    module = _load_module()

    clipped = module.clip_target_values([10.0, -5.0, 2500.0, None], rated_power_kw=2050.0)

    np.testing.assert_allclose(clipped[:3], np.asarray([10.0, 0.0, 2050.0], dtype=np.float32))
    assert np.isnan(clipped[3])


def test_fit_covariate_statistics_and_transform_fill_zero_and_mask() -> None:
    module = _load_module()
    train_raw = np.asarray(
        [
            [[1.0], [2.0]],
            [[3.0], [np.nan]],
        ],
        dtype=np.float32,
    )

    means, stds = module.fit_covariate_statistics(train_raw)
    transformed = module.transform_exogenous_inputs(
        np.asarray([[[5.0], [np.nan]]], dtype=np.float32),
        means=means,
        stds=stds,
    )

    np.testing.assert_allclose(means, np.asarray([2.0], dtype=np.float32))
    np.testing.assert_allclose(stds, np.asarray([math.sqrt(2.0 / 3.0)], dtype=np.float32))
    assert transformed.shape == (1, 2, 2)
    np.testing.assert_allclose(transformed[0, 0, 0], (5.0 - 2.0) / math.sqrt(2.0 / 3.0))
    assert transformed[0, 1, 0] == 0.0
    np.testing.assert_allclose(transformed[0, :, 1], np.asarray([0.0, 1.0], dtype=np.float32))


def test_thin_non_overlap_window_index_keeps_every_forecast_step_per_turbine() -> None:
    module = _load_module()
    windows = _descriptor(
        module,
        turbine_indices=[0] * 10 + [1] * 10,
        target_indices=list(range(10)) + list(range(20, 30)),
        forecast_steps=36,
    )

    thinned = module.thin_non_overlap_window_index(
        windows,
        turbine_ids=("T01", "T02"),
        forecast_steps=3,
    )

    assert thinned.turbine_indices.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert thinned.target_indices.tolist() == [0, 3, 6, 9, 20, 23, 26, 29]


def test_fit_covariate_statistics_from_windows_matches_raw_batch_path() -> None:
    module = _load_module()
    turbine_series = (
        module.TurbineSeries(
            timestamps_us=np.arange(0, 8, dtype=np.int64),
            target_pu=np.linspace(0.0, 0.7, 8, dtype=np.float32),
            past_covariates={
                "wind_speed": np.asarray([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32),
                "ambient_temp": np.asarray([10.0, 11.0, 12.0, 13.0, np.nan, 15.0, 16.0, 17.0], dtype=np.float32),
            },
        ),
        module.TurbineSeries(
            timestamps_us=np.arange(0, 8, dtype=np.int64),
            target_pu=np.linspace(0.0, 0.7, 8, dtype=np.float32),
            past_covariates={
                "wind_speed": np.asarray([20.0, 21.0, 22.0, 23.0, 24.0, np.nan, 26.0, 27.0], dtype=np.float32),
                "ambient_temp": np.asarray([30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, np.nan], dtype=np.float32),
            },
        ),
    )
    windows = _descriptor(
        module,
        turbine_indices=[0, 1],
        target_indices=[3, 4],
        forecast_steps=2,
        step_us=1,
    )

    raw_batch = module.build_raw_exogenous_batch(
        turbine_series,
        windows,
        history_steps=3,
        covariate_columns=("wind_speed", "ambient_temp"),
    )
    expected_means, expected_stds = module.fit_covariate_statistics(raw_batch)
    means, stds = module.fit_covariate_statistics_from_windows(
        turbine_series,
        windows,
        history_steps=3,
        covariate_columns=("wind_speed", "ambient_temp"),
        chunk_size=1,
    )

    np.testing.assert_allclose(means, expected_means)
    np.testing.assert_allclose(stds, expected_stds)


def test_fit_covariate_statistics_from_windows_updates_progress_bar(monkeypatch) -> None:
    module = _load_module()
    _TqdmRecorder.instances.clear()
    turbine_series = (
        module.TurbineSeries(
            timestamps_us=np.arange(0, 8, dtype=np.int64),
            target_pu=np.linspace(0.0, 0.7, 8, dtype=np.float32),
            past_covariates={
                "wind_speed": np.asarray([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32),
            },
        ),
    )
    windows = _descriptor(
        module,
        turbine_indices=[0, 0],
        target_indices=[3, 4],
        forecast_steps=2,
        step_us=1,
    )
    monkeypatch.setattr(module, "tqdm", _TqdmRecorder)
    monkeypatch.setattr(module, "progress_is_enabled", lambda: True)

    module.fit_covariate_statistics_from_windows(
        turbine_series,
        windows,
        history_steps=3,
        covariate_columns=("wind_speed",),
        chunk_size=1,
        progress_label="kelmarsh/stage1_core",
    )

    assert len(_TqdmRecorder.instances) == 1
    progress = _TqdmRecorder.instances[0]
    assert progress.total == 2
    assert progress.n == 2
    assert progress.closed is True
    assert progress.desc == "kelmarsh/stage1_core stats"


def test_prepare_dataset_reads_dense_cache_and_builds_dual_eval_windows(tmp_path) -> None:
    module = _load_module()
    cache_root = tmp_path / "cache"
    _build_temp_cache(cache_root)
    pack = module.CovariatePackSpec(
        dataset_id="toy_dataset",
        stage="stage1_core",
        pack_name="stage1_core",
        feature_set="default",
        required_columns=("wind_speed", "ambient_temp", "status_flag"),
    )

    prepared = module.prepare_dataset("toy_dataset", pack=pack, cache_root=cache_root)

    assert prepared.dataset_id == "toy_dataset"
    assert prepared.resolution_minutes == 10
    assert prepared.rated_power_kw == 2000.0
    assert prepared.stride_steps == 1
    assert prepared.covariate_count == 3
    assert prepared.covariate_policy == module.COVARIATE_POLICY
    assert len(prepared.train_windows) == 2442
    assert len(prepared.val_rolling_windows) == 42
    assert len(prepared.val_non_overlap_windows) == 2
    assert len(prepared.test_rolling_windows) == 442
    assert len(prepared.test_non_overlap_windows) == 14
    assert prepared.covariate_means.shape == (3,)
    assert prepared.covariate_stds.shape == (3,)
    val_groups = {}
    for turbine_index, target_index in zip(
        prepared.val_non_overlap_windows.turbine_indices.tolist(),
        prepared.val_non_overlap_windows.target_indices.tolist(),
        strict=True,
    ):
        val_groups.setdefault(turbine_index, []).append(target_index)
    assert val_groups == {0: [1544], 1: [1544]}
    test_groups = {}
    for turbine_index, target_index in zip(
        prepared.test_non_overlap_windows.turbine_indices.tolist(),
        prepared.test_non_overlap_windows.target_indices.tolist(),
        strict=True,
    ):
        test_groups.setdefault(turbine_index, []).append(target_index)
    assert test_groups == {
        0: [1744, 1780, 1816, 1852, 1888, 1924, 1960],
        1: [1744, 1780, 1816, 1852, 1888, 1924, 1960],
    }


def test_build_requested_packs_includes_stage3_regime_for_ltsf() -> None:
    module = _load_module()

    packs = module.build_requested_packs("kelmarsh", covariate_stages=("stage3_regime",))

    assert [pack.pack_name for pack in packs] == ["power_only", "stage3_regime"]


def test_select_device_prefers_cuda_then_mps_then_cpu() -> None:
    module = _load_module()

    class _Backend:
        def __init__(self, available: bool) -> None:
            self._available = available

        def is_available(self) -> bool:
            return self._available

    class _Torch:
        def __init__(self, *, cuda_available: bool, mps_available: bool) -> None:
            self.cuda = _Backend(cuda_available)
            self.backends = type("_Backends", (), {"mps": _Backend(mps_available)})()

    assert module.select_device(torch_module=_Torch(cuda_available=True, mps_available=True)) == "cuda"
    assert module.select_device(torch_module=_Torch(cuda_available=False, mps_available=True)) == "mps"
    assert module.select_device(torch_module=_Torch(cuda_available=False, mps_available=False)) == "cpu"


def test_progress_is_enabled_requires_tqdm_and_tty(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "HAS_TQDM", True)
    monkeypatch.setattr(module.sys.stderr, "isatty", lambda: True)
    assert module.progress_is_enabled() is True

    monkeypatch.setattr(module, "HAS_TQDM", False)
    assert module.progress_is_enabled() is False

    monkeypatch.setattr(module, "HAS_TQDM", True)
    monkeypatch.setattr(module.sys.stderr, "isatty", lambda: False)
    assert module.progress_is_enabled() is False


def test_train_model_uses_val_rolling_rmse_for_early_stopping(monkeypatch) -> None:
    module = _load_module()
    prepared = _small_prepared_dataset(module, forecast_steps=2)
    loader_calls = []
    rmse_values = iter([0.9, 0.8, 0.85, 0.86])

    class _Optimizer:
        def zero_grad(self, *, set_to_none: bool = True) -> None:
            del set_to_none

        def step(self) -> None:
            return None

    class _Loss:
        def backward(self) -> None:
            return None

    class _MSELoss:
        def __call__(self, *_args, **_kwargs):
            return _Loss()

    class _Adam:
        def __init__(self, *_args, **_kwargs) -> None:
            self.optimizer = _Optimizer()

        def __getattr__(self, name):
            return getattr(self.optimizer, name)

    class _Torch:
        float32 = "float32"
        optim = type("_Optim", (), {"Adam": _Adam})
        nn = type("_NN", (), {"MSELoss": _MSELoss})

    class _Model:
        def __init__(self) -> None:
            self.loaded_state = None

        def to(self, device: str):
            del device
            return self

        def parameters(self):
            return []

        def train(self) -> None:
            return None

        def eval(self) -> None:
            return None

        def state_dict(self) -> dict[str, int]:
            return {"best": 1}

        def load_state_dict(self, state: dict[str, int]) -> None:
            self.loaded_state = state

    model = _Model()

    def _fake_loader(prepared_dataset, *, windows, batch_size, shuffle, seed):
        del prepared_dataset, batch_size, shuffle, seed
        if windows is prepared.train_windows:
            return []
        if windows is prepared.val_rolling_windows:
            return "val_loader"
        raise AssertionError("train_model should only build train and val rolling loaders")

    def _fake_evaluate(_model, loader, *, device, rated_power_kw, forecast_steps, progress_label=None):
        del _model, device, rated_power_kw, progress_label
        loader_calls.append(loader)
        return _evaluation_metrics(module, window_count=1, forecast_steps=forecast_steps, base=next(rmse_values) - 0.01)

    monkeypatch.setattr(module, "require_torch", lambda: (_Torch(), module.nn, module.DataLoader, module.Dataset))
    monkeypatch.setattr(module, "_set_random_seed", lambda seed: None)
    monkeypatch.setattr(module, "build_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(module, "_build_dataloader", _fake_loader)
    monkeypatch.setattr(module, "evaluate_model", _fake_evaluate)

    outcome = module.train_model(
        "nlinear",
        prepared,
        device="cpu",
        max_epochs=5,
        early_stopping_patience=2,
    )

    assert loader_calls == ["val_loader", "val_loader", "val_loader", "val_loader"]
    assert outcome.best_epoch == 2
    assert outcome.epochs_ran == 4
    assert outcome.best_val_rmse_pu == 0.8
    assert outcome.model is model
    assert model.loaded_state == {"best": 1}


def test_execute_training_job_emits_long_rows_for_all_eval_protocols(monkeypatch) -> None:
    module = _load_module()
    prepared = _small_prepared_dataset(module, covariate_columns=("wind_speed",))

    def _fake_train_model(*args, **kwargs):
        del args, kwargs
        return module.TrainingOutcome(
            best_epoch=3,
            epochs_ran=5,
            best_val_rmse_pu=0.123,
            device="cpu",
            model="trained_model",
        )

    def _fake_loader(prepared_dataset, *, windows, batch_size, shuffle, seed):
        del prepared_dataset, batch_size, shuffle, seed
        return windows

    def _fake_evaluate(model, loader, *, device, rated_power_kw, forecast_steps, progress_label=None):
        del model, device, rated_power_kw, progress_label
        return _evaluation_metrics(module, window_count=len(loader), forecast_steps=forecast_steps, base=len(loader) / 100.0)

    monkeypatch.setattr(module, "train_model", _fake_train_model)
    monkeypatch.setattr(module, "_build_dataloader", _fake_loader)
    monkeypatch.setattr(module, "evaluate_model", _fake_evaluate)

    rows = module.execute_training_job(prepared, model_variant="dlinear")

    assert len(rows) == 148
    overall_rows = [row for row in rows if row["metric_scope"] == module.OVERALL_METRIC_SCOPE]
    horizon_rows = [row for row in rows if row["metric_scope"] == module.HORIZON_METRIC_SCOPE]
    assert len(overall_rows) == 4
    assert len(horizon_rows) == 144
    assert {row["split_name"] for row in overall_rows} == {"val", "test"}
    assert {row["eval_protocol"] for row in overall_rows} == {
        module.ROLLING_EVAL_PROTOCOL,
        module.NON_OVERLAP_EVAL_PROTOCOL,
    }
    assert {row["lead_step"] for row in overall_rows} == {None}
    assert {row["lead_step"] for row in horizon_rows} == set(range(1, 37))
    assert all(row["window_protocol"] == module.WINDOW_PROTOCOL for row in rows)
    assert all(row["train_window_count"] == 4 for row in rows)
    assert all(row["val_window_count"] == 2 for row in rows)
    assert all(row["test_window_count"] == 3 for row in rows)


def test_sort_result_frame_orders_long_results() -> None:
    module = _load_module()
    frame = pl.DataFrame(
        [
            _result_row(
                module,
                dataset_id="sdwpf_kddcup",
                covariate_stage="stage1_core",
                covariate_pack="stage1_core",
                model_variant="dlinear",
                split_name="test",
                eval_protocol=module.NON_OVERLAP_EVAL_PROTOCOL,
                metric_scope=module.HORIZON_METRIC_SCOPE,
                lead_step=2,
            ),
            _result_row(
                module,
                dataset_id="kelmarsh",
                covariate_stage="reference",
                covariate_pack="power_only",
                model_variant="nlinear",
                split_name="val",
                eval_protocol=module.ROLLING_EVAL_PROTOCOL,
                metric_scope=module.OVERALL_METRIC_SCOPE,
                lead_step=None,
            ),
            _result_row(
                module,
                dataset_id="kelmarsh",
                covariate_stage="reference",
                covariate_pack="power_only",
                model_variant="nlinear",
                split_name="val",
                eval_protocol=module.ROLLING_EVAL_PROTOCOL,
                metric_scope=module.HORIZON_METRIC_SCOPE,
                lead_step=1,
            ),
        ]
    )

    sorted_frame = module.sort_result_frame(frame)

    assert list(
        zip(
            sorted_frame["dataset_id"].to_list(),
            sorted_frame["split_name"].to_list(),
            sorted_frame["eval_protocol"].to_list(),
            sorted_frame["metric_scope"].to_list(),
            sorted_frame["lead_step"].to_list(),
            strict=True,
        )
    ) == [
        ("kelmarsh", "val", module.ROLLING_EVAL_PROTOCOL, module.OVERALL_METRIC_SCOPE, None),
        ("kelmarsh", "val", module.ROLLING_EVAL_PROTOCOL, module.HORIZON_METRIC_SCOPE, 1),
        ("sdwpf_kddcup", "test", module.NON_OVERLAP_EVAL_PROTOCOL, module.HORIZON_METRIC_SCOPE, 2),
    ]


def test_run_experiment_aggregates_runner_rows(tmp_path) -> None:
    module = _load_module()

    def _fake_loader(dataset_id, *, pack, cache_root, max_windows_per_split):
        del cache_root, max_windows_per_split
        return _small_prepared_dataset(
            module,
            dataset_id=dataset_id,
            covariate_stage=pack.stage,
            covariate_pack=pack.pack_name,
        )

    def _fake_runner(prepared, *, model_variant, device, seed, batch_size, learning_rate, max_epochs, early_stopping_patience):
        del device, seed, batch_size, learning_rate, max_epochs, early_stopping_patience
        return [
            _result_row(
                module,
                dataset_id=prepared.dataset_id,
                covariate_stage=prepared.covariate_stage,
                covariate_pack=prepared.covariate_pack,
                model_variant=model_variant,
                split_name="val",
                eval_protocol=module.ROLLING_EVAL_PROTOCOL,
                metric_scope=module.OVERALL_METRIC_SCOPE,
                lead_step=None,
            ),
            _result_row(
                module,
                dataset_id=prepared.dataset_id,
                covariate_stage=prepared.covariate_stage,
                covariate_pack=prepared.covariate_pack,
                model_variant=model_variant,
                split_name="test",
                eval_protocol=module.NON_OVERLAP_EVAL_PROTOCOL,
                metric_scope=module.HORIZON_METRIC_SCOPE,
                lead_step=1,
            ),
        ]

    output_path = tmp_path / "ltsf-linear.csv"
    results = module.run_experiment(
        dataset_ids=("kelmarsh",),
        model_variants=("dlinear", "nlinear"),
        covariate_stages=(),
        include_power_only_reference=True,
        output_path=output_path,
        dataset_loader=_fake_loader,
        job_runner=_fake_runner,
    )

    assert results.height == 4
    assert output_path.exists()
    assert list(
        zip(
            results["model_variant"].to_list(),
            results["split_name"].to_list(),
            results["metric_scope"].to_list(),
            strict=True,
        )
    ) == [
        ("nlinear", "val", module.OVERALL_METRIC_SCOPE),
        ("nlinear", "test", module.HORIZON_METRIC_SCOPE),
        ("dlinear", "val", module.OVERALL_METRIC_SCOPE),
        ("dlinear", "test", module.HORIZON_METRIC_SCOPE),
    ]


def test_run_experiment_updates_job_progress_bar(monkeypatch, tmp_path) -> None:
    module = _load_module()
    _TqdmRecorder.instances.clear()

    def _fake_loader(dataset_id, *, pack, cache_root, max_windows_per_split):
        del cache_root, max_windows_per_split
        return _small_prepared_dataset(
            module,
            dataset_id=dataset_id,
            covariate_stage=pack.stage,
            covariate_pack=pack.pack_name,
        )

    def _fake_runner(prepared, *, model_variant, device, seed, batch_size, learning_rate, max_epochs, early_stopping_patience):
        del device, seed, batch_size, learning_rate, max_epochs, early_stopping_patience
        return [
            _result_row(
                module,
                dataset_id=prepared.dataset_id,
                covariate_stage=prepared.covariate_stage,
                covariate_pack=prepared.covariate_pack,
                model_variant=model_variant,
                split_name="val",
                eval_protocol=module.ROLLING_EVAL_PROTOCOL,
                metric_scope=module.OVERALL_METRIC_SCOPE,
                lead_step=None,
            ),
        ]

    monkeypatch.setattr(module, "tqdm", _TqdmRecorder)
    monkeypatch.setattr(module, "progress_is_enabled", lambda: True)

    module.run_experiment(
        dataset_ids=("kelmarsh",),
        model_variants=("nlinear", "dlinear"),
        covariate_stages=(),
        include_power_only_reference=True,
        output_path=tmp_path / "ltsf-linear.csv",
        dataset_loader=_fake_loader,
        job_runner=_fake_runner,
    )

    assert len(_TqdmRecorder.instances) == 1
    progress = _TqdmRecorder.instances[0]
    assert progress.total == 2
    assert progress.n == 2
    assert progress.closed is True
    assert progress.desc == "ltsf-linear jobs"
    assert "kelmarsh/power_only/dlinear" in progress.postfixes

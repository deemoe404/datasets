from __future__ import annotations

from datetime import datetime, timedelta
from importlib.util import module_from_spec, spec_from_file_location
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import polars as pl


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "experiment"
        / "tft"
        / "tft.py"
    )
    spec = spec_from_file_location("tft", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _build_temp_cache(
    cache_root: Path,
    *,
    dataset_id: str = "kelmarsh",
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
    row_count = len(timestamps)
    rows: list[dict[str, object]] = []
    for turbine_index, turbine_id in enumerate(("T01", "T02")):
        offset = turbine_index * 10_000
        for index, timestamp in enumerate(timestamps):
            rows.append(
                {
                    "dataset": dataset_id,
                    "turbine_id": turbine_id,
                    "timestamp": timestamp,
                    "target_kw": float(offset + index),
                    "quality_flags": "",
                    "Wind speed (m/s)": float(index),
                    "Wind direction (°)": float((index * 3) % 360),
                    "Nacelle position (°)": float((index * 5) % 360),
                    "Generator RPM (RPM)": float(1000 + index),
                    "Rotor speed (RPM)": float(800 + index),
                    "Ambient temperature (converter) (°C)": float((index % 20) - 5),
                    "Nacelle temperature (°C)": float((index % 15) + 10),
                    "Power factor (cosphi)": float(0.9 + ((index % 5) * 0.01)),
                    "Reactive power (kvar)": float(index % 100),
                    "Blade angle (pitch position) A (°)": float(index % 30),
                    "Blade angle (pitch position) B (°)": float((index + 1) % 30),
                    "Blade angle (pitch position) C (°)": float((index + 2) % 30),
                    "farm_pmu__gms_power_kw": float(500 + (index % 60)),
                    "farm_pmu__gms_reactive_power_kvar": float(100 + (index % 25)),
                    "farm_pmu__gms_current_a": float(200 + (index % 40)),
                    "farm_pmu__production_meter_data_availability": bool(index % 2 == 0),
                    "farm_grid_meter__grid_meter_data_availability": bool(index % 3 == 0),
                }
            )
    series = pl.DataFrame(rows)
    series = series.with_columns(
        pl.when(pl.arange(0, row_count * 2) == 151)
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(pl.col("Wind speed (m/s)"))
        .alias("Wind speed (m/s)"),
        pl.when(pl.arange(0, row_count * 2) == 302)
        .then(pl.lit(None, dtype=pl.Boolean))
        .otherwise(pl.col("farm_grid_meter__grid_meter_data_availability"))
        .alias("farm_grid_meter__grid_meter_data_availability"),
    )
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
            "rated_power_kw": [2050.0, 2050.0],
            "coord_x": [0.0, 100.0],
            "coord_y": [10.0, 110.0],
            "latitude": [52.0, 52.1],
            "longitude": [-1.0, -1.1],
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


def _small_prepared_dataset(module, *, input_pack_spec):
    windows = {
        "train": _descriptor(module, turbine_indices=[0, 0], target_indices=[200, 212], forecast_steps=36),
        "val_roll": _descriptor(module, turbine_indices=[0], target_indices=[300], forecast_steps=36),
        "val_non": _descriptor(module, turbine_indices=[0], target_indices=[300], forecast_steps=36),
        "test_roll": _descriptor(module, turbine_indices=[0], target_indices=[360], forecast_steps=36),
        "test_non": _descriptor(module, turbine_indices=[0], target_indices=[360], forecast_steps=36),
    }
    historical_count = input_pack_spec.historical_covariate_count
    static_count = 2 if input_pack_spec.uses_static_covariates else 0
    known_count = 5 if input_pack_spec.uses_known_future_covariates else 0
    historical_columns = tuple(f"hist_{index:02d}" for index in range(historical_count))
    historical_masks = tuple(f"{column}_missing" for column in historical_columns)
    return module.PreparedDataset(
        dataset_id="kelmarsh",
        resolution_minutes=10,
        rated_power_kw=2050.0,
        history_steps=144,
        forecast_steps=36,
        stride_steps=1,
        input_pack=input_pack_spec.input_pack,
        historical_covariate_stage=input_pack_spec.historical_covariate_stage,
        feature_set=input_pack_spec.feature_set,
        uses_static_covariates=input_pack_spec.uses_static_covariates,
        uses_known_future_covariates=input_pack_spec.uses_known_future_covariates,
        static_covariate_columns=("static_coord_1", "static_coord_2") if static_count else (),
        known_covariate_columns=module.KNOWN_FUTURE_COLUMNS if known_count else (),
        historical_covariate_columns=historical_columns,
        historical_mask_columns=historical_masks,
        static_covariate_count=static_count,
        known_covariate_count=known_count,
        historical_covariate_count=historical_count,
        covariate_policy=module.COVARIATE_POLICY if (static_count or known_count or historical_count) else module.REFERENCE_COVARIATE_POLICY,
        turbine_ids=("T01",),
        model_frame=pd.DataFrame({"split": ["train"], "series_id": ["kelmarsh::T01"], "time_idx": [0], "target_pu": [0.0]}),
        train_windows=windows["train"],
        val_rolling_windows=windows["val_roll"],
        val_non_overlap_windows=windows["val_non"],
        test_rolling_windows=windows["test_roll"],
        test_non_overlap_windows=windows["test_non"],
        train_origin_frame=module.window_descriptor_to_origin_frame("kelmarsh", ("T01",), windows["train"]),
        val_rolling_origin_frame=module.window_descriptor_to_origin_frame("kelmarsh", ("T01",), windows["val_roll"]),
        val_non_overlap_origin_frame=module.window_descriptor_to_origin_frame("kelmarsh", ("T01",), windows["val_non"]),
        test_rolling_origin_frame=module.window_descriptor_to_origin_frame("kelmarsh", ("T01",), windows["test_roll"]),
        test_non_overlap_origin_frame=module.window_descriptor_to_origin_frame("kelmarsh", ("T01",), windows["test_non"]),
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


def test_resolve_input_pack_configures_flags_and_counts() -> None:
    module = _load_module()

    reference = module.resolve_input_pack("kelmarsh", "reference")
    known_static = module.resolve_input_pack("kelmarsh", "known_static")
    hist_stage1 = module.resolve_input_pack("kelmarsh", "hist_stage1")
    mixed_stage2 = module.resolve_input_pack("kelmarsh", "mixed_stage2")

    assert reference.historical_covariate_count == 0
    assert reference.uses_static_covariates is False
    assert reference.uses_known_future_covariates is False

    assert known_static.historical_covariate_count == 0
    assert known_static.uses_static_covariates is True
    assert known_static.uses_known_future_covariates is True

    assert hist_stage1.historical_covariate_stage == "stage1_core"
    assert hist_stage1.historical_covariate_count == 12
    assert hist_stage1.feature_set == "lightweight"

    assert mixed_stage2.historical_covariate_stage == "stage2_ops"
    assert mixed_stage2.historical_covariate_count == 17
    assert mixed_stage2.uses_static_covariates is True
    assert mixed_stage2.uses_known_future_covariates is True


def test_build_known_future_feature_frame_uses_timestamp_cycles() -> None:
    module = _load_module()
    frame = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1, 0, 0, 0),
                datetime(2024, 1, 1, 12, 0, 0),
                datetime(2024, 1, 2, 0, 0, 0),
            ]
        }
    )

    enriched = module.build_known_future_feature_frame(frame)

    np.testing.assert_allclose(enriched["tod_sin"].to_numpy(), np.asarray([0.0, 0.0, 0.0]), atol=1e-12)
    np.testing.assert_allclose(enriched["tod_cos"].to_numpy(), np.asarray([1.0, -1.0, 1.0]), atol=1e-12)
    np.testing.assert_allclose(enriched["dow_sin"].to_numpy()[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(enriched["dow_cos"].to_numpy()[0], 1.0, atol=1e-12)
    np.testing.assert_allclose(
        enriched["dow_sin"].to_numpy()[2],
        math.sin(2.0 * math.pi / 7.0),
        atol=1e-12,
    )


def test_resolve_best_epoch_requests_full_checkpoint_load() -> None:
    module = _load_module()
    calls: list[dict[str, object]] = []

    class FakeTorch:
        def load(self, path, map_location=None, **kwargs):
            calls.append(
                {
                    "path": path,
                    "map_location": map_location,
                    "kwargs": kwargs,
                }
            )
            return {"epoch": 8}

    best_epoch = module._resolve_best_epoch("/tmp/model.ckpt", FakeTorch())

    assert best_epoch == 9
    assert calls == [
        {
            "path": "/tmp/model.ckpt",
            "map_location": "cpu",
            "kwargs": {"weights_only": False},
        }
    ]


def test_resolve_best_epoch_falls_back_for_legacy_torch_load() -> None:
    module = _load_module()
    calls: list[dict[str, object]] = []

    class LegacyFakeTorch:
        def load(self, path, map_location=None):
            calls.append(
                {
                    "path": path,
                    "map_location": map_location,
                }
            )
            return {"epoch": 3}

    best_epoch = module._resolve_best_epoch("/tmp/model.ckpt", LegacyFakeTorch())

    assert best_epoch == 4
    assert calls == [
        {
            "path": "/tmp/model.ckpt",
            "map_location": "cpu",
        }
    ]


def test_resolve_static_coordinate_columns_prefers_coord_xy_then_latlon() -> None:
    module = _load_module()
    coord_frame = pl.DataFrame(
        {
            "turbine_id": ["T01", "T02"],
            "coord_x": [0.0, 1.0],
            "coord_y": [2.0, 3.0],
            "latitude": [50.0, 51.0],
            "longitude": [-1.0, -2.0],
        }
    )
    latlon_frame = coord_frame.with_columns(
        pl.lit(None, dtype=pl.Float64).alias("coord_x"),
        pl.lit(None, dtype=pl.Float64).alias("coord_y"),
    )

    assert module.resolve_static_coordinate_columns(coord_frame) == ("coord_x", "coord_y")
    assert module.resolve_static_coordinate_columns(latlon_frame) == ("latitude", "longitude")


def test_filter_decoded_origin_index_matches_pairwise_allowed_origins() -> None:
    module = _load_module()
    decoded_index = pd.DataFrame(
        {
            "series_id": ["kelmarsh::T01", "kelmarsh::T01", "kelmarsh::T02"],
            "time_idx_first_prediction": [144, 145, 144],
        }
    )
    allowed_origins = pd.DataFrame(
        {
            "series_id": ["kelmarsh::T01", "kelmarsh::T02"],
            "time_idx_first_prediction": [145, 144],
        }
    )

    mask = module.filter_decoded_origin_index(decoded_index, allowed_origins)

    assert mask.tolist() == [False, True, True]


def test_prepare_dataset_reads_dense_cache_and_downsamples_train_only(tmp_path) -> None:
    module = _load_module()
    cache_root = tmp_path / "cache"
    _build_temp_cache(cache_root)
    input_pack_spec = module.resolve_input_pack("kelmarsh", "mixed_stage1")

    prepared = module.prepare_dataset(
        "kelmarsh",
        input_pack_spec=input_pack_spec,
        cache_root=cache_root,
    )

    assert prepared.input_pack == "mixed_stage1"
    assert prepared.feature_set == "lightweight"
    assert prepared.static_covariate_count == 2
    assert prepared.known_covariate_count == 5
    assert prepared.historical_covariate_count == 12
    assert len(prepared.historical_covariate_columns) == 12
    assert len(prepared.historical_mask_columns) == 12
    assert len(prepared.train_windows) == 204
    assert len(prepared.val_rolling_windows) == 42
    assert len(prepared.val_non_overlap_windows) == 2
    assert len(prepared.test_rolling_windows) == 442
    assert len(prepared.test_non_overlap_windows) == 14
    grouped = prepared.train_origin_frame.groupby("series_id")["time_idx_first_prediction"].apply(list).to_dict()
    assert grouped["kelmarsh::T01"][:4] == [144, 156, 168, 180]
    assert grouped["kelmarsh::T02"][:4] == [144, 156, 168, 180]


def test_build_frame_for_windows_trims_unused_missing_targets() -> None:
    module = _load_module()
    prepared = module.PreparedDataset(
        dataset_id="kelmarsh",
        resolution_minutes=10,
        rated_power_kw=2050.0,
        history_steps=2,
        forecast_steps=2,
        stride_steps=1,
        input_pack="reference",
        historical_covariate_stage=None,
        feature_set="default",
        uses_static_covariates=False,
        uses_known_future_covariates=False,
        static_covariate_columns=(),
        known_covariate_columns=(),
        historical_covariate_columns=(),
        historical_mask_columns=(),
        static_covariate_count=0,
        known_covariate_count=0,
        historical_covariate_count=0,
        covariate_policy=module.REFERENCE_COVARIATE_POLICY,
        turbine_ids=("T01",),
        model_frame=pd.DataFrame(
            {
                "series_id": ["kelmarsh::T01"] * 8,
                "time_idx": list(range(8)),
                "target_pu": [0.0, None, 0.2, 0.3, 0.4, 0.5, None, None],
                "split": ["train"] * 8,
            }
        ),
        train_windows=_descriptor(module, turbine_indices=[0], target_indices=[4], forecast_steps=2, step_us=1),
        val_rolling_windows=_descriptor(module, turbine_indices=[0], target_indices=[4], forecast_steps=2, step_us=1),
        val_non_overlap_windows=_descriptor(module, turbine_indices=[0], target_indices=[4], forecast_steps=2, step_us=1),
        test_rolling_windows=_descriptor(module, turbine_indices=[0], target_indices=[4], forecast_steps=2, step_us=1),
        test_non_overlap_windows=_descriptor(module, turbine_indices=[0], target_indices=[4], forecast_steps=2, step_us=1),
        train_origin_frame=module.window_descriptor_to_origin_frame("kelmarsh", ("T01",), _descriptor(module, turbine_indices=[0], target_indices=[4], forecast_steps=2, step_us=1)),
        val_rolling_origin_frame=module.window_descriptor_to_origin_frame("kelmarsh", ("T01",), _descriptor(module, turbine_indices=[0], target_indices=[4], forecast_steps=2, step_us=1)),
        val_non_overlap_origin_frame=module.window_descriptor_to_origin_frame("kelmarsh", ("T01",), _descriptor(module, turbine_indices=[0], target_indices=[4], forecast_steps=2, step_us=1)),
        test_rolling_origin_frame=module.window_descriptor_to_origin_frame("kelmarsh", ("T01",), _descriptor(module, turbine_indices=[0], target_indices=[4], forecast_steps=2, step_us=1)),
        test_non_overlap_origin_frame=module.window_descriptor_to_origin_frame("kelmarsh", ("T01",), _descriptor(module, turbine_indices=[0], target_indices=[4], forecast_steps=2, step_us=1)),
    )

    trimmed = module.build_frame_for_windows(prepared, prepared.train_windows)

    assert trimmed["time_idx"].to_list() == [2, 3, 4, 5]
    assert trimmed["target_pu"].isna().sum() == 0


def test_run_experiment_all_input_packs_emits_expected_888_row_grid(tmp_path) -> None:
    module = _load_module()

    def _fake_loader(dataset_id: str, *, input_pack_spec, cache_root, max_train_origins):
        del dataset_id, cache_root, max_train_origins
        return _small_prepared_dataset(module, input_pack_spec=input_pack_spec)

    def _fake_runner(
        prepared_dataset,
        *,
        device,
        seed,
        batch_size,
        learning_rate,
        max_epochs,
        early_stopping_patience,
        hidden_size,
        attention_head_size,
        hidden_continuous_size,
        dropout,
        gradient_clip_val,
        num_workers,
        trainer_precision,
        matmul_precision,
    ):
        del device, max_epochs, early_stopping_patience, num_workers, trainer_precision, matmul_precision
        evaluation_results = [
            ("val", module.ROLLING_EVAL_PROTOCOL, prepared_dataset.val_rolling_windows, _evaluation_metrics(module, window_count=1, forecast_steps=36, base=0.10)),
            ("val", module.NON_OVERLAP_EVAL_PROTOCOL, prepared_dataset.val_non_overlap_windows, _evaluation_metrics(module, window_count=1, forecast_steps=36, base=0.11)),
            ("test", module.ROLLING_EVAL_PROTOCOL, prepared_dataset.test_rolling_windows, _evaluation_metrics(module, window_count=1, forecast_steps=36, base=0.12)),
            ("test", module.NON_OVERLAP_EVAL_PROTOCOL, prepared_dataset.test_non_overlap_windows, _evaluation_metrics(module, window_count=1, forecast_steps=36, base=0.13)),
        ]
        return module.build_result_rows(
            prepared_dataset,
            training_outcome=module.TrainingOutcome(
                best_epoch=2,
                epochs_ran=3,
                best_val_rmse_pu=0.09,
                device="cpu",
                model=None,
            ),
            runtime_seconds=1.0,
            seed=seed,
            batch_size=batch_size or 128,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            hidden_continuous_size=hidden_continuous_size,
            dropout=dropout,
            gradient_clip_val=gradient_clip_val,
            num_workers=0,
            trainer_precision="32-true",
            matmul_precision=None,
            evaluation_results=evaluation_results,
        )

    output_path = tmp_path / "tft-pilot.csv"
    results = module.run_experiment(
        dataset_ids=("kelmarsh",),
        input_packs=module.DEFAULT_INPUT_PACKS,
        output_path=output_path,
        dataset_loader=_fake_loader,
        job_runner=_fake_runner,
    )

    assert output_path.exists()
    assert results.height == 888
    assert results["input_pack"].head(4).to_list() == [
        "reference",
        "reference",
        "reference",
        "reference",
    ]
    assert results.filter(pl.col("input_pack") == "mixed_stage2").height == 148


def test_resolve_batch_size_uses_cpu_default_for_mps() -> None:
    module = _load_module()

    assert module.resolve_batch_size("cpu") == 128
    assert module.resolve_batch_size("mps") == 128
    assert module.resolve_batch_size("cuda") == 512


def test_runtime_auto_resolution_prefers_cuda_fast_path() -> None:
    module = _load_module()

    assert module.resolve_num_workers("cpu") == 0
    assert module.resolve_num_workers("mps") == 0
    assert module.resolve_num_workers("cuda") >= 4
    assert module.resolve_trainer_precision("cpu") == "32-true"
    assert module.resolve_trainer_precision("cuda") == "bf16-mixed"
    assert module.resolve_matmul_precision("cpu") is None
    assert module.resolve_matmul_precision("cuda") == "high"


def test_evaluate_model_disables_return_index(monkeypatch) -> None:
    module = _load_module()
    captured: dict[str, object] = {}

    class FakePrediction:
        output = np.zeros((2, 36), dtype=np.float32)
        y = np.ones((2, 36), dtype=np.float32)

    class FakeModel:
        def predict(self, loader, **kwargs):
            captured["loader"] = loader
            captured["kwargs"] = kwargs
            return FakePrediction()

    def _fake_loader(dataset, *, batch_size, train, device, num_workers):
        captured["loader_args"] = {
            "dataset": dataset,
            "batch_size": batch_size,
            "train": train,
            "device": device,
            "num_workers": num_workers,
        }
        return "loader"

    expected_metrics = _evaluation_metrics(module, window_count=2, forecast_steps=36, base=0.12)

    def _fake_evaluate_predictions(predictions, actual, *, rated_power_kw):
        captured["predictions_shape"] = predictions.shape
        captured["actual_shape"] = actual.shape
        captured["rated_power_kw"] = rated_power_kw
        return expected_metrics

    monkeypatch.setattr(module, "_build_dataloader", _fake_loader)
    monkeypatch.setattr(module, "evaluate_predictions", _fake_evaluate_predictions)

    metrics = module.evaluate_model(
        FakeModel(),
        dataset="dataset",
        device="mps",
        batch_size=64,
        rated_power_kw=2050.0,
    )

    assert metrics is expected_metrics
    assert captured["loader_args"] == {
        "dataset": "dataset",
        "batch_size": 64,
        "train": False,
        "device": "mps",
        "num_workers": None,
    }
    assert captured["kwargs"]["return_index"] is False
    assert captured["kwargs"]["return_y"] is True
    assert captured["predictions_shape"] == (2, 36)
    assert captured["actual_shape"] == (2, 36)
    assert captured["rated_power_kw"] == 2050.0


def test_execute_training_job_reuses_built_datasets(monkeypatch) -> None:
    module = _load_module()
    prepared = _small_prepared_dataset(
        module,
        input_pack_spec=module.resolve_input_pack("kelmarsh", "reference"),
    )
    build_calls: list[object] = []
    train_calls: list[dict[str, object]] = []
    evaluate_calls: list[object] = []

    fake_built = object()

    def _fake_build(prepared_dataset):
        build_calls.append(prepared_dataset)
        return fake_built

    def _fake_train_model(prepared_dataset, **kwargs):
        train_calls.append(
            {
                "prepared_dataset": prepared_dataset,
                "built_datasets": kwargs["built_datasets"],
                "batch_size": kwargs["batch_size"],
                "device": kwargs["device"],
                "num_workers": kwargs["num_workers"],
                "trainer_precision": kwargs["trainer_precision"],
                "matmul_precision": kwargs["matmul_precision"],
            }
        )
        return module.TrainingOutcome(
            best_epoch=2,
            epochs_ran=3,
            best_val_rmse_pu=0.09,
            device="cpu",
            model="model",
        )

    def _fake_iter_specs(prepared_dataset, built_datasets):
        assert prepared_dataset is prepared
        assert built_datasets is fake_built
        return (
            ("val", module.ROLLING_EVAL_PROTOCOL, prepared.val_rolling_windows, "val-roll"),
            ("val", module.NON_OVERLAP_EVAL_PROTOCOL, prepared.val_non_overlap_windows, "val-non"),
            ("test", module.ROLLING_EVAL_PROTOCOL, prepared.test_rolling_windows, "test-roll"),
            ("test", module.NON_OVERLAP_EVAL_PROTOCOL, prepared.test_non_overlap_windows, "test-non"),
        )

    def _fake_evaluate(model, dataset, **kwargs):
        evaluate_calls.append(
            (
                model,
                dataset,
                kwargs["batch_size"],
                kwargs["device"],
                kwargs["num_workers"],
            )
        )
        return _evaluation_metrics(module, window_count=1, forecast_steps=36, base=0.12)

    class _Progress:
        def update(self, _value):
            return None

        def set_postfix_str(self, _value):
            return None

        def close(self):
            return None

    monkeypatch.setattr(module, "build_timeseries_datasets", _fake_build)
    monkeypatch.setattr(module, "train_model", _fake_train_model)
    monkeypatch.setattr(module, "iter_evaluation_specs", _fake_iter_specs)
    monkeypatch.setattr(module, "evaluate_model", _fake_evaluate)
    monkeypatch.setattr(module, "_create_progress_bar", lambda **kwargs: _Progress())

    rows = module.execute_training_job(prepared, device="mps")

    assert len(build_calls) == 1
    assert train_calls == [
        {
            "prepared_dataset": prepared,
            "built_datasets": fake_built,
            "batch_size": 128,
            "device": "mps",
            "num_workers": 0,
            "trainer_precision": "32-true",
            "matmul_precision": None,
        }
    ]
    assert len(evaluate_calls) == 4
    assert rows[0]["batch_size"] == 128
    assert rows[0]["num_workers"] == 0
    assert rows[0]["trainer_precision"] == "32-true"
    assert rows[0]["matmul_precision"] is None


def test_normalize_cli_input_packs_expands_all_once() -> None:
    module = _load_module()

    normalized = module._normalize_cli_input_packs(["all", "reference", "mixed_stage1"])

    assert normalized == module.DEFAULT_INPUT_PACKS


def test_build_dataloader_enables_cuda_prefetching() -> None:
    module = _load_module()
    captured: dict[str, object] = {}

    class FakeDataset:
        def to_dataloader(self, **kwargs):
            captured.update(kwargs)
            return "loader"

    loader = module._build_dataloader(
        FakeDataset(),
        batch_size=512,
        train=True,
        device="cuda",
        num_workers=6,
    )

    assert loader == "loader"
    assert captured == {
        "train": True,
        "batch_size": 512,
        "num_workers": 6,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": module.DEFAULT_PREFETCH_FACTOR,
    }

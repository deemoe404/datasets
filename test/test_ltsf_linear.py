from __future__ import annotations

from datetime import datetime, timedelta
from importlib.util import module_from_spec, spec_from_file_location
import json
import math
from pathlib import Path
import sys

import numpy as np
import polars as pl
import pytest


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


def _build_temp_cache(cache_root: Path, *, dataset_id: str = "toy_dataset") -> None:
    dataset_root = cache_root / dataset_id
    task_dir = dataset_root / "tasks" / "default" / "turbine" / "next_6h_from_24h_stride_6h"
    default_series_dir = dataset_root / "gold_base" / "default" / "turbine" / "default"
    lightweight_series_dir = dataset_root / "gold_base" / "default" / "turbine" / "lightweight"
    static_dir = dataset_root / "silver" / "meta"
    task_dir.mkdir(parents=True, exist_ok=True)
    default_series_dir.mkdir(parents=True, exist_ok=True)
    lightweight_series_dir.mkdir(parents=True, exist_ok=True)
    static_dir.mkdir(parents=True, exist_ok=True)

    timestamps = pl.datetime_range(
        datetime(2024, 1, 1, 0, 0, 0),
        datetime(2024, 1, 2, 9, 10, 0),
        interval="10m",
        eager=True,
    )
    wind_speed_t01 = [float(index) for index in range(len(timestamps))]
    wind_speed_t02 = [float(index + 500) for index in range(len(timestamps))]
    ambient_t01 = [float(index % 13) for index in range(len(timestamps))]
    ambient_t02 = [float((index % 13) + 100) for index in range(len(timestamps))]
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

    output_start_values = [timestamps[150 + index] for index in range(10)]
    output_end_values = [timestamps[150 + index + 35] for index in range(10)]
    window_rows = []
    for turbine_id in ("T01", "T02"):
        for output_start_ts, output_end_ts in zip(output_start_values, output_end_values, strict=True):
            window_rows.append(
                {
                    "dataset": dataset_id,
                    "turbine_id": turbine_id,
                    "output_start_ts": output_start_ts,
                    "output_end_ts": output_end_ts,
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
                    "task_id": "next_6h_from_24h_stride_6h",
                    "history_steps": 144,
                    "forecast_steps": 36,
                    "stride_steps": 36,
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


def _small_prepared_dataset(
    module,
    *,
    dataset_id: str = "kelmarsh",
    covariate_stage: str = "reference",
    covariate_pack: str = "power_only",
    feature_set: str = "default",
    covariate_count: int = 0,
):
    exogenous_width = covariate_count * 2
    train = module.SplitData(
        target_inputs=np.asarray(
            [
                [0.0, 0.1, 0.2, 0.3],
                [0.1, 0.2, 0.3, 0.4],
                [0.2, 0.3, 0.4, 0.5],
                [0.3, 0.4, 0.5, 0.6],
            ],
            dtype=np.float32,
        ),
        exogenous_inputs=np.zeros((4, 4, exogenous_width), dtype=np.float32),
        targets=np.asarray(
            [
                [0.4, 0.5],
                [0.5, 0.6],
                [0.6, 0.7],
                [0.7, 0.8],
            ],
            dtype=np.float32,
        ),
        output_start_us=np.asarray([1, 2, 3, 4], dtype=np.int64),
        output_end_us=np.asarray([2, 3, 4, 5], dtype=np.int64),
    )
    val = module.SplitData(
        target_inputs=np.asarray([[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]], dtype=np.float32),
        exogenous_inputs=np.zeros((2, 4, exogenous_width), dtype=np.float32),
        targets=np.asarray([[0.5, 0.6], [0.6, 0.7]], dtype=np.float32),
        output_start_us=np.asarray([6, 7], dtype=np.int64),
        output_end_us=np.asarray([7, 8], dtype=np.int64),
    )
    test = module.SplitData(
        target_inputs=np.asarray([[0.2, 0.3, 0.4, 0.5], [0.3, 0.4, 0.5, 0.6]], dtype=np.float32),
        exogenous_inputs=np.zeros((2, 4, exogenous_width), dtype=np.float32),
        targets=np.asarray([[0.6, 0.7], [0.7, 0.8]], dtype=np.float32),
        output_start_us=np.asarray([8, 9], dtype=np.int64),
        output_end_us=np.asarray([9, 10], dtype=np.int64),
    )
    return module.PreparedDataset(
        dataset_id=dataset_id,
        rated_power_kw=2050.0,
        history_steps=4,
        forecast_steps=2,
        stride_steps=2,
        covariate_stage=covariate_stage,
        covariate_pack=covariate_pack,
        feature_set=feature_set,
        covariate_count=covariate_count,
        covariate_policy=module.COVARIATE_POLICY if covariate_count else module.REFERENCE_COVARIATE_POLICY,
        train=train,
        val=val,
        test=test,
    )


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


def test_split_window_index_keeps_same_output_start_in_one_split() -> None:
    module = _load_module()
    base = datetime(2024, 1, 1, 0, 0, 0)
    rows = []
    for index in range(10):
        for turbine_id in ("T01", "T02"):
            rows.append(
                {
                    "dataset": "toy",
                    "turbine_id": turbine_id,
                    "output_start_ts": base + timedelta(hours=index),
                    "output_end_ts": base + timedelta(hours=index, minutes=10),
                    "is_complete_input": True,
                    "is_complete_output": True,
                    "quality_flags": "",
                }
            )
    window_index = pl.DataFrame(rows)

    split_frames = module.split_window_index(window_index)

    assert split_frames["train"]["output_start_ts"].n_unique() == 7
    assert split_frames["val"]["output_start_ts"].n_unique() == 1
    assert split_frames["test"]["output_start_ts"].n_unique() == 2
    assert split_frames["train"].group_by("output_start_ts").len()["len"].to_list() == [2] * 7


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


def test_build_raw_split_samples_scales_targets_and_keeps_reference_empty_exogenous() -> None:
    module = _load_module()
    timestamps = pl.datetime_range(
        datetime(2024, 1, 1, 0, 0, 0),
        datetime(2024, 1, 1, 0, 50, 0),
        interval="10m",
        eager=True,
    )
    series = pl.DataFrame(
        {
            "dataset": ["toy"] * 12,
            "turbine_id": ["T01"] * 6 + ["T02"] * 6,
            "timestamp": list(timestamps) + list(timestamps),
            "target_kw": [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 0.0, 10.0, None, 30.0, 40.0, 50.0],
            "quality_flags": [""] * 12,
        }
    )
    series_map = module.build_turbine_series_map(series, rated_power_kw=100.0, covariate_columns=())
    window_index = pl.DataFrame(
        {
            "dataset": ["toy", "toy"],
            "turbine_id": ["T01", "T02"],
            "output_start_ts": [timestamps[3], timestamps[3]],
            "output_end_ts": [timestamps[4], timestamps[4]],
            "is_complete_input": [True, True],
            "is_complete_output": [True, True],
            "quality_flags": ["", ""],
        }
    )

    split = module.build_raw_split_samples(
        window_index,
        series_map,
        history_steps=3,
        forecast_steps=2,
        covariate_columns=(),
    )

    assert split.target_inputs.shape == (1, 3)
    assert split.targets.shape == (1, 2)
    assert split.raw_exogenous_inputs.shape == (1, 3, 0)
    np.testing.assert_allclose(split.target_inputs[0], np.asarray([0.0, 0.1, 0.2], dtype=np.float32))
    np.testing.assert_allclose(split.targets[0], np.asarray([0.3, 0.4], dtype=np.float32))


def test_prepare_dataset_reads_existing_cache_and_expands_exogenous_inputs(tmp_path) -> None:
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
    assert prepared.rated_power_kw == 2000.0
    assert prepared.covariate_count == 3
    assert prepared.covariate_policy == module.COVARIATE_POLICY
    assert prepared.train.target_inputs.shape == (14, 144)
    assert prepared.train.targets.shape == (14, 36)
    assert prepared.train.exogenous_inputs.shape == (14, 144, 6)
    assert prepared.val.target_inputs.shape[0] == 2
    assert prepared.test.target_inputs.shape[0] == 4
    assert np.any(prepared.train.exogenous_inputs[:, :, 3:] == 1.0)


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


def test_run_experiment_writes_expected_rows_with_injected_runner(tmp_path) -> None:
    module = _load_module()

    def _fake_loader(dataset_id: str, *, pack, cache_root, max_windows_per_split):
        del cache_root, max_windows_per_split
        covariate_count = 0 if pack.stage == "reference" else 2
        return _small_prepared_dataset(
            module,
            dataset_id=dataset_id,
            covariate_stage=pack.stage,
            covariate_pack=pack.pack_name,
            feature_set=pack.feature_set,
            covariate_count=covariate_count,
        )

    def _fake_runner(prepared_dataset, *, model_variant, device, seed, batch_size, learning_rate, max_epochs, early_stopping_patience):
        del device, seed, batch_size, learning_rate, max_epochs, early_stopping_patience
        return {
            "dataset_id": prepared_dataset.dataset_id,
            "model_id": module.MODEL_ID,
            "model_variant": model_variant,
            "task_id": module.TASK_ID,
            "history_steps": prepared_dataset.history_steps,
            "forecast_steps": prepared_dataset.forecast_steps,
            "stride_steps": prepared_dataset.stride_steps,
            "split_protocol": module.SPLIT_PROTOCOL,
            "covariate_stage": prepared_dataset.covariate_stage,
            "covariate_pack": prepared_dataset.covariate_pack,
            "feature_set": prepared_dataset.feature_set,
            "covariate_count": prepared_dataset.covariate_count,
            "covariate_policy": prepared_dataset.covariate_policy,
            "window_count": 2,
            "prediction_count": 4,
            "start_timestamp": "2024-01-01 00:00:00",
            "end_timestamp": "2024-01-01 00:10:00",
            "mae_kw": 1.0,
            "rmse_kw": 1.0,
            "mae_pu": 0.1,
            "rmse_pu": 0.1,
            "device": "cpu",
            "runtime_seconds": 0.1,
            "train_window_count": 4,
            "val_window_count": 2,
            "test_window_count": 2,
            "best_epoch": 1,
            "epochs_ran": 1,
            "best_val_rmse_pu": 0.1,
            "seed": module.DEFAULT_SEED,
            "batch_size": module.DEFAULT_BATCH_SIZE,
            "learning_rate": module.DEFAULT_LEARNING_RATE,
        }

    output_path = tmp_path / "results.csv"
    results = module.run_experiment(
        dataset_ids=("kelmarsh", "penmanshiel"),
        model_variants=("nlinear", "dlinear"),
        covariate_stages=("stage1_core",),
        include_power_only_reference=True,
        output_path=output_path,
        dataset_loader=_fake_loader,
        job_runner=_fake_runner,
    )

    assert output_path.exists()
    assert results.height == 8
    assert list(
        zip(
            results["dataset_id"].to_list(),
            results["covariate_stage"].to_list(),
            results["model_variant"].to_list(),
            strict=True,
        )
    ) == [
        ("kelmarsh", "reference", "nlinear"),
        ("kelmarsh", "reference", "dlinear"),
        ("kelmarsh", "stage1_core", "nlinear"),
        ("kelmarsh", "stage1_core", "dlinear"),
        ("penmanshiel", "reference", "nlinear"),
        ("penmanshiel", "reference", "dlinear"),
        ("penmanshiel", "stage1_core", "nlinear"),
        ("penmanshiel", "stage1_core", "dlinear"),
    ]


def test_nlinearx_matches_backbone_without_covariates() -> None:
    torch = pytest.importorskip("torch")
    module = _load_module()
    backbone = module.NLinear(history_steps=4, forecast_steps=2)
    model = module.NLinearX(history_steps=4, forecast_steps=2, exogenous_channels=0)
    model.target_backbone.load_state_dict(backbone.state_dict())
    inputs = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)

    backbone_output = backbone(inputs)
    model_output = model(inputs, torch.zeros((1, 4, 0), dtype=torch.float32))

    torch.testing.assert_close(model_output, backbone_output)


def test_nlinearx_uses_exogenous_residual_head() -> None:
    torch = pytest.importorskip("torch")
    module = _load_module()
    model = module.NLinearX(history_steps=2, forecast_steps=1, exogenous_channels=2)
    with torch.no_grad():
        model.target_backbone.linear.weight.zero_()
        model.target_backbone.linear.bias.zero_()
        model.exogenous_head.linear.weight.fill_(1.0)
        model.exogenous_head.linear.bias.zero_()

    output = model(
        torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32),
    )

    torch.testing.assert_close(output, torch.tensor([[10.0]], dtype=torch.float32))


def test_dlinearx_forward_returns_expected_shape() -> None:
    torch = pytest.importorskip("torch")
    module = _load_module()
    model = module.DLinearX(history_steps=6, forecast_steps=3, exogenous_channels=2)

    output = model(
        torch.zeros((2, 6), dtype=torch.float32),
        torch.zeros((2, 6, 2), dtype=torch.float32),
    )

    assert output.shape == (2, 3)


def test_train_model_is_deterministic_on_cpu() -> None:
    pytest.importorskip("torch")
    module = _load_module()
    prepared = _small_prepared_dataset(module, covariate_stage="stage1_core", covariate_pack="stage1_core", covariate_count=1)

    first = module.train_model(
        "nlinear",
        prepared,
        device="cpu",
        seed=7,
        batch_size=2,
        learning_rate=1e-2,
        max_epochs=4,
        early_stopping_patience=2,
    )
    second = module.train_model(
        "nlinear",
        prepared,
        device="cpu",
        seed=7,
        batch_size=2,
        learning_rate=1e-2,
        max_epochs=4,
        early_stopping_patience=2,
    )

    assert first.best_epoch == second.best_epoch
    assert first.epochs_ran == second.epochs_ran
    assert first.best_val_rmse_pu == second.best_val_rmse_pu
    assert first.metrics == second.metrics

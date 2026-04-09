from __future__ import annotations

from datetime import datetime, timedelta
from importlib.util import module_from_spec, spec_from_file_location
import json
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
    series_dir = dataset_root / "gold_base" / "default" / "turbine" / "default"
    static_dir = dataset_root / "silver" / "meta"
    task_dir.mkdir(parents=True, exist_ok=True)
    series_dir.mkdir(parents=True, exist_ok=True)
    static_dir.mkdir(parents=True, exist_ok=True)

    timestamps = pl.datetime_range(
        datetime(2024, 1, 1, 0, 0, 0),
        datetime(2024, 1, 2, 9, 10, 0),
        interval="10m",
        eager=True,
    )
    rows: dict[str, object] = {
        "dataset": [dataset_id] * (len(timestamps) * 2),
        "turbine_id": ["T01"] * len(timestamps) + ["T02"] * len(timestamps),
        "timestamp": list(timestamps) + list(timestamps),
        "target_kw": (
            [float(index) for index in range(len(timestamps))]
            + [float(index + 1000) for index in range(len(timestamps))]
        ),
        "is_observed": [True] * (len(timestamps) * 2),
        "quality_flags": [""] * (len(timestamps) * 2),
    }
    pl.DataFrame(rows).write_parquet(series_dir / "series.parquet")

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


def _small_prepared_dataset(module, *, dataset_id: str = "kelmarsh"):
    train = module.SplitData(
        inputs=np.asarray(
            [
                [0.0, 0.1, 0.2, 0.3],
                [0.1, 0.2, 0.3, 0.4],
                [0.2, 0.3, 0.4, 0.5],
                [0.3, 0.4, 0.5, 0.6],
            ],
            dtype=np.float32,
        ),
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
        inputs=np.asarray([[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]], dtype=np.float32),
        targets=np.asarray([[0.5, 0.6], [0.6, 0.7]], dtype=np.float32),
        output_start_us=np.asarray([6, 7], dtype=np.int64),
        output_end_us=np.asarray([7, 8], dtype=np.int64),
    )
    test = module.SplitData(
        inputs=np.asarray([[0.2, 0.3, 0.4, 0.5], [0.3, 0.4, 0.5, 0.6]], dtype=np.float32),
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


def test_build_split_samples_scales_targets_and_skips_nan_windows() -> None:
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
            "is_observed": [True] * 12,
            "quality_flags": [""] * 12,
        }
    )
    series_map = module.build_turbine_series_map(series, rated_power_kw=100.0)
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

    split = module.build_split_samples(window_index, series_map, history_steps=3, forecast_steps=2)

    assert split.inputs.shape == (1, 3)
    assert split.targets.shape == (1, 2)
    np.testing.assert_allclose(split.inputs[0], np.asarray([0.0, 0.1, 0.2], dtype=np.float32))
    np.testing.assert_allclose(split.targets[0], np.asarray([0.3, 0.4], dtype=np.float32))


def test_prepare_dataset_reads_existing_cache(tmp_path) -> None:
    module = _load_module()
    cache_root = tmp_path / "cache"
    _build_temp_cache(cache_root)

    prepared = module.prepare_dataset("toy_dataset", cache_root=cache_root)

    assert prepared.dataset_id == "toy_dataset"
    assert prepared.rated_power_kw == 2000.0
    assert prepared.train.inputs.shape[0] == 14
    assert prepared.val.inputs.shape[0] == 2
    assert prepared.test.inputs.shape[0] == 4


def test_run_experiment_writes_expected_rows_with_injected_runner(tmp_path) -> None:
    module = _load_module()

    def _fake_loader(dataset_id: str, *, cache_root, max_windows_per_split):
        del cache_root, max_windows_per_split
        return _small_prepared_dataset(module, dataset_id=dataset_id)

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
        output_path=output_path,
        dataset_loader=_fake_loader,
        job_runner=_fake_runner,
    )

    assert output_path.exists()
    assert results.height == 4
    assert list(zip(results["dataset_id"].to_list(), results["model_variant"].to_list(), strict=True)) == [
        ("kelmarsh", "nlinear"),
        ("kelmarsh", "dlinear"),
        ("penmanshiel", "nlinear"),
        ("penmanshiel", "dlinear"),
    ]


def test_nlinear_forward_applies_last_value_shift() -> None:
    torch = pytest.importorskip("torch")
    module = _load_module()
    model = module.NLinear(history_steps=4, forecast_steps=2)
    with torch.no_grad():
        model.linear.weight.zero_()
        model.linear.bias.zero_()

    output = model(torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32))

    torch.testing.assert_close(output, torch.tensor([[4.0, 4.0]], dtype=torch.float32))


def test_dlinear_forward_returns_expected_shape() -> None:
    torch = pytest.importorskip("torch")
    module = _load_module()
    model = module.DLinear(history_steps=6, forecast_steps=3)

    output = model(torch.zeros((2, 6), dtype=torch.float32))

    assert output.shape == (2, 3)


def test_train_model_is_deterministic_on_cpu() -> None:
    pytest.importorskip("torch")
    module = _load_module()
    prepared = _small_prepared_dataset(module)

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

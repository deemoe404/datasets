from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys

import numpy as np
import polars as pl
import pytest

from test.test_agcrn import (
    _TqdmRecorder,
    _build_temp_cache,
    _descriptor,
    _patch_temp_bundle_loader,
    _require_torch,
    _work_root,
)


_FEATURE_PROTOCOL_ID = "power_wd_yaw_pmean_hist_sincos_masked"
_TARGET_HISTORY_MASK_COLUMNS = ("target_kw__mask",)
_PAST_COVARIATE_COLUMNS = (
    "wind_direction_sin",
    "wind_direction_cos",
    "yaw_error_sin",
    "yaw_error_cos",
    "pitch_mean",
    "wind_direction_sin__mask",
    "wind_direction_cos__mask",
    "yaw_error_sin__mask",
    "yaw_error_cos__mask",
    "pitch_mean__mask",
)
_PAST_COVARIATE_VALUE_COLUMNS = _PAST_COVARIATE_COLUMNS[:5]
_PAST_COVARIATE_MASK_COLUMNS = _PAST_COVARIATE_COLUMNS[5:]


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "experiment"
        / "families"
        / "agcrn_masked"
        / "agcrn_masked.py"
    )
    spec = spec_from_file_location("agcrn_masked", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _build_masked_temp_cache(cache_root: Path, *, dataset_id: str = "toy_dataset") -> None:
    _build_temp_cache(
        cache_root,
        dataset_id=dataset_id,
        feature_protocol_id=_FEATURE_PROTOCOL_ID,
        target_history_mask_columns=_TARGET_HISTORY_MASK_COLUMNS,
        past_covariate_columns=_PAST_COVARIATE_COLUMNS,
    )
    task_dir = cache_root / dataset_id / "tasks" / "next_6h_from_24h" / _FEATURE_PROTOCOL_ID
    series = pl.read_parquet(task_dir / "series.parquet")
    timestamps = series["timestamp"].unique().sort().to_list()
    target_missing_ts = timestamps[200]
    covariate_missing_ts = timestamps[201]
    dirty_window_ts = timestamps[144]

    series = series.with_columns(
        pl.when(pl.col("timestamp") == target_missing_ts)
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(pl.col("target_kw"))
        .alias("target_kw"),
        pl.when(pl.col("timestamp") == target_missing_ts)
        .then(pl.lit(1, dtype=pl.Int8))
        .otherwise(pl.lit(0, dtype=pl.Int8))
        .alias("target_kw__mask"),
        *[
            pl.when(pl.col("timestamp") == covariate_missing_ts)
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(pl.col(column))
            .alias(column)
            for column in _PAST_COVARIATE_VALUE_COLUMNS
        ],
        *[
            pl.when(pl.col("timestamp") == covariate_missing_ts)
            .then(pl.lit(1, dtype=pl.Int8))
            .otherwise(pl.lit(0, dtype=pl.Int8))
            .alias(column)
            for column in _PAST_COVARIATE_MASK_COLUMNS
        ],
    )
    series.write_parquet(task_dir / "series.parquet")

    window_index = pl.read_parquet(task_dir / "window_index.parquet").with_columns(
        pl.when(pl.col("output_start_ts") == dirty_window_ts)
        .then(pl.lit("row_quality_issues"))
        .otherwise(pl.col("quality_flags"))
        .alias("quality_flags")
    )
    window_index.write_parquet(task_dir / "window_index.parquet")

    task_context = json.loads((task_dir / "task_context.json").read_text(encoding="utf-8"))
    task_context["column_groups"]["past_covariate_values"] = list(_PAST_COVARIATE_VALUE_COLUMNS)
    task_context["column_groups"]["past_covariate_masks"] = list(_PAST_COVARIATE_MASK_COLUMNS)
    (task_dir / "task_context.json").write_text(json.dumps(task_context), encoding="utf-8")


def test_prepare_dataset_builds_finite_masked_source_tensor(tmp_path, monkeypatch) -> None:
    module = _load_module()
    cache_root = tmp_path / "cache"
    _build_masked_temp_cache(cache_root)
    _patch_temp_bundle_loader(
        monkeypatch,
        module,
        cache_root,
        feature_protocol_id=_FEATURE_PROTOCOL_ID,
    )

    prepared = module.prepare_dataset("toy_dataset", cache_root=cache_root)

    assert prepared.model_variant == module.MODEL_VARIANT
    assert prepared.feature_protocol_id == module.FEATURE_PROTOCOL_ID
    assert prepared.input_channel_names == (
        "target_pu",
        "target_kw__mask",
        "wind_direction_sin",
        "wind_direction_cos",
        "yaw_error_sin",
        "yaw_error_cos",
        "pitch_mean",
        "wind_direction_sin__mask",
        "wind_direction_cos__mask",
        "yaw_error_sin__mask",
        "yaw_error_cos__mask",
        "pitch_mean__mask",
    )
    assert prepared.input_channels == 12
    assert np.isfinite(prepared.source_tensor).all()
    assert set(np.unique(prepared.source_tensor[:, :, 1]).tolist()).issubset({0.0, 1.0})
    assert prepared.source_tensor[200, 0, 0] == pytest.approx(0.0)
    assert prepared.source_tensor[201, 0, 2:7].tolist() == pytest.approx([0.0] * 5)
    assert prepared.source_tensor[201, 0, 7:12].tolist() == pytest.approx([1.0] * 5)


def test_prepare_dataset_reconstructs_target_valid_mask_without_reading_window_quality_flags(
    tmp_path,
    monkeypatch,
) -> None:
    module = _load_module()
    cache_root = tmp_path / "cache"
    _build_masked_temp_cache(cache_root)
    _patch_temp_bundle_loader(
        monkeypatch,
        module,
        cache_root,
        feature_protocol_id=_FEATURE_PROTOCOL_ID,
    )

    prepared = module.prepare_dataset("toy_dataset", cache_root=cache_root)

    assert prepared.target_valid_mask[200].tolist() == [False, False, False]
    assert prepared.target_valid_mask[202].tolist() == [True, True, True]
    assert 144 in prepared.train_windows.target_indices.tolist()


def test_filter_windows_with_valid_targets_drops_only_fully_invalid_outputs() -> None:
    module = _load_module()
    target_valid_mask = np.asarray(
        [
            [True, True],
            [False, False],
            [True, False],
            [False, False],
            [False, False],
        ],
        dtype=bool,
    )
    windows = _descriptor(module, target_indices=[1, 2, 3], forecast_steps=2, step_us=1)

    filtered = module.filter_windows_with_valid_targets(
        windows,
        target_valid_mask=target_valid_mask,
        forecast_steps=2,
    )

    assert filtered.target_indices.tolist() == [1, 2]


def test_masked_mse_loss_and_evaluate_model_ignore_invalid_targets() -> None:
    module = _load_module()
    torch_module = _require_torch(module)
    history = torch_module.zeros((1, 4, 2, 3), dtype=torch_module.float32)
    targets = torch_module.tensor(
        [[[[1.0], [2.0]], [[3.0], [4.0]]]],
        dtype=torch_module.float32,
    )
    valid_mask = torch_module.tensor(
        [[[[1.0], [0.0]], [[1.0], [1.0]]]],
        dtype=torch_module.float32,
    )
    predictions = targets + 1.0

    class _FakeModel:
        def eval(self):
            return self

        def __call__(self, batch_history, batch_targets, teacher_forcing_ratio=0.0):
            del batch_history, teacher_forcing_ratio
            return batch_targets + 1.0

    loss = module.masked_mse_loss(
        predictions,
        targets,
        valid_mask,
        torch_module=torch_module,
    )
    metrics = module.evaluate_model(
        _FakeModel(),
        [(history, targets, valid_mask)],
        device="cpu",
        rated_power_kw=2.0,
        forecast_steps=2,
    )

    assert float(loss.item()) == pytest.approx(1.0)
    assert metrics.window_count == 1
    assert metrics.prediction_count == 3
    assert metrics.mae_pu == pytest.approx(1.0)
    assert metrics.rmse_pu == pytest.approx(1.0)
    assert metrics.mae_kw == pytest.approx(2.0)
    assert metrics.rmse_kw == pytest.approx(2.0)
    assert metrics.horizon_prediction_count.tolist() == [1, 2]


def test_run_experiment_smoke_trains_masked_family_on_toy_cache(tmp_path, monkeypatch) -> None:
    module = _load_module()
    _require_torch(module)
    cache_root = tmp_path / "cache"
    _build_masked_temp_cache(cache_root)
    _patch_temp_bundle_loader(
        monkeypatch,
        module,
        cache_root,
        feature_protocol_id=_FEATURE_PROTOCOL_ID,
    )
    monkeypatch.setattr(module, "tqdm", _TqdmRecorder)
    output_path = tmp_path / "agcrn-masked.csv"

    results = module.run_experiment(
        dataset_ids=("toy_dataset",),
        cache_root=cache_root,
        output_path=output_path,
        work_root=_work_root(tmp_path),
        device="cpu",
        max_epochs=1,
        max_train_origins=8,
        max_eval_origins=4,
        batch_size=4,
        learning_rate=1e-3,
        early_stopping_patience=1,
        hidden_dim=8,
        embed_dim=4,
        num_layers=1,
        cheb_k=2,
    )

    assert output_path.exists()
    assert results.height == 148
    assert results["model_variant"].unique().to_list() == [module.MODEL_VARIANT]
    assert results["prediction_count"].min() > 0

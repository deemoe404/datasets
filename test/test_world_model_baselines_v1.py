from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np
import polars as pl
import pytest

from test.test_agcrn import _require_torch
from test.test_world_model_rollout_v1 import (
    _KNOWN_FUTURE_COLUMNS,
    _build_world_model_temp_cache,
    _patch_bundle_loader,
)


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "experiment"
        / "families"
        / "world_model_baselines_v1"
        / "world_model_baselines_v1.py"
    )
    spec = spec_from_file_location("world_model_baselines_v1", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _prepare_temp_dataset(
    module,
    tmp_path: Path,
    monkeypatch,
    *,
    dataset_id: str = "kelmarsh",
    max_train_origins: int = 4,
    max_eval_origins: int = 2,
    variant_spec=None,
):
    cache_root = tmp_path / "cache"
    _build_world_model_temp_cache(cache_root, dataset_id=dataset_id)
    _patch_bundle_loader(monkeypatch, module.state_base.rollout_base, cache_root, dataset_id=dataset_id)
    return module.prepare_dataset(
        dataset_id,
        variant_spec=variant_spec,
        cache_root=cache_root,
        max_train_origins=max_train_origins,
        max_eval_origins=max_eval_origins,
    )


def _timexer_variant_spec(module):
    return module.resolve_variant_specs((module.TIMEXER_VARIANT,))[0]


def _dgcrn_variant_spec(module):
    return module.resolve_variant_specs((module.DGCRN_VARIANT,))[0]


def _chronos_variant_spec(module):
    return module.resolve_variant_specs((module.CHRONOS_VARIANT,))[0]


def _itransformer_variant_spec(module):
    return module.resolve_variant_specs((module.ITRANSFORMER_VARIANT,))[0]


class _FakeSummaryWriter:
    instances: list["_FakeSummaryWriter"] = []

    def __init__(self, log_dir: str, flush_secs: int = 10) -> None:
        self.log_dir = log_dir
        self.flush_secs = flush_secs
        self.scalars: list[tuple[str, float, int]] = []
        self.texts: list[tuple[str, str, int]] = []
        self.closed = False
        _FakeSummaryWriter.instances.append(self)

    def add_scalar(self, tag: str, value: float, global_step: int) -> None:
        self.scalars.append((tag, float(value), int(global_step)))

    def add_text(self, tag: str, text: str, global_step: int = 0) -> None:
        self.texts.append((tag, text, int(global_step)))

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def test_registry_declares_kelmarsh_only_and_six_variants() -> None:
    registry_path = (
        Path(__file__).resolve().parents[1]
        / "experiment"
        / "infra"
        / "registry"
        / "families"
        / "world_model_baselines_v1.toml"
    )
    text = registry_path.read_text(encoding="utf-8")

    assert 'family_id = "world_model_baselines_v1"' in text
    assert 'dataset_scope = ["kelmarsh"]' in text
    assert 'supported_feature_protocols = ["world_model_v1"]' in text
    assert 'world_model_persistence_last_value_v1_farm_sync = "world_model_v1"' in text
    assert 'world_model_shared_weight_tft_no_graph_v1_farm_sync = "world_model_v1"' in text
    assert 'world_model_shared_weight_timexer_no_graph_v1_farm_sync = "world_model_v1"' in text
    assert 'world_model_dgcrn_v1_farm_sync = "world_model_v1"' in text
    assert 'world_model_chronos_2_zero_shot_v1_farm_sync = "world_model_v1"' in text
    assert 'world_model_itransformer_no_graph_v1_farm_sync = "world_model_v1"' in text


def test_non_kelmarsh_dataset_is_rejected() -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="only supports"):
        module.run_experiment(dataset_ids=("penmanshiel",), output_path=Path("/tmp/unused.csv"))


def test_tft_default_hyperparameters_are_more_conservative() -> None:
    module = _load_module()
    profile = module.resolve_hyperparameter_profile(module.TFT_VARIANT, dataset_id="kelmarsh")

    assert profile.batch_size == 256
    assert profile.learning_rate == pytest.approx(1e-4)
    assert profile.d_model == 64
    assert profile.lstm_hidden_dim == 64
    assert profile.dropout == pytest.approx(0.2)
    assert profile.weight_decay == pytest.approx(1e-3)


def test_timexer_default_hyperparameters_include_patching() -> None:
    module = _load_module()
    profile = module.resolve_hyperparameter_profile(module.TIMEXER_VARIANT, dataset_id="kelmarsh")

    assert profile.patch_len == 24
    assert profile.encoder_layers == 2
    assert profile.ff_hidden_dim == 256


def test_dgcrn_default_hyperparameters_match_family_contract() -> None:
    module = _load_module()
    profile = module.resolve_hyperparameter_profile(module.DGCRN_VARIANT, dataset_id="kelmarsh")

    assert profile.batch_size == 256
    assert profile.learning_rate == pytest.approx(5e-4)
    assert profile.hidden_dim == 64
    assert profile.embed_dim == 16
    assert profile.num_layers == 2
    assert profile.cheb_k == 2
    assert profile.teacher_forcing_ratio == pytest.approx(0.5)
    assert profile.d_model is None


def test_itransformer_default_hyperparameters_are_kelmarsh_specific() -> None:
    module = _load_module()
    profile = module.resolve_hyperparameter_profile(module.ITRANSFORMER_VARIANT, dataset_id="kelmarsh")

    assert profile.batch_size == 64
    assert profile.learning_rate == pytest.approx(1e-4)
    assert profile.d_model == 64
    assert profile.lstm_hidden_dim is None
    assert profile.attention_heads == 4
    assert profile.dropout == pytest.approx(0.1)
    assert profile.weight_decay == pytest.approx(1e-4)


def test_tensorboard_root_uses_output_hash_by_default(tmp_path) -> None:
    module = _load_module()
    output_path = tmp_path / "published" / "latest.csv"
    work_root = tmp_path / ".work"

    root = module.resolve_tensorboard_root(output_path=output_path, work_root=work_root)
    expected = module._resume_paths_for_output(output_path=output_path, work_root=work_root).slot_dir / "tensorboard"

    assert root == expected


def test_persistence_uses_last_history_value_without_future_leak(tmp_path, monkeypatch) -> None:
    module = _load_module()
    prepared = _prepare_temp_dataset(module, tmp_path, monkeypatch)
    windows = prepared.val_rolling_windows
    target_index = int(windows.target_indices[0])
    history_slice = slice(target_index - prepared.history_steps, target_index)
    future_slice = slice(target_index, target_index + prepared.forecast_steps)

    predictions, _targets, _valid = module.persistence_predictions(prepared, windows)
    values = prepared.local_history_tensor[history_slice, :, module.state_base._LOCAL_VALUE_START]
    unavailable = prepared.local_history_tensor[history_slice, :, module.state_base._LOCAL_MASK_START]
    fallback = module._train_history_target_mean(prepared)
    expected = fallback.copy()
    for node_index in range(prepared.node_count):
        valid_positions = np.flatnonzero(unavailable[:, node_index] < 0.5)
        if valid_positions.size:
            expected[node_index] = values[int(valid_positions[-1]), node_index]

    assert np.allclose(predictions[0, :, :, 0], expected[None, :])

    mutated_targets = prepared.target_pu_filled.copy()
    mutated_targets[future_slice] = 1.0 - mutated_targets[future_slice]
    mutated_prepared = replace(prepared, target_pu_filled=mutated_targets)
    mutated_predictions, _mutated_targets, _mutated_valid = module.persistence_predictions(mutated_prepared, windows)

    assert np.allclose(mutated_predictions, predictions)


def test_persistence_uses_train_only_fallback_when_history_target_missing(tmp_path, monkeypatch) -> None:
    module = _load_module()
    prepared = _prepare_temp_dataset(module, tmp_path, monkeypatch)
    fallback = module._train_history_target_mean(prepared)
    windows = prepared.val_rolling_windows
    target_index = int(windows.target_indices[0])
    history_slice = slice(target_index - prepared.history_steps, target_index)

    local_history = prepared.local_history_tensor.copy()
    local_history[history_slice, 0, module.state_base._LOCAL_VALUE_START] = 0.9876
    local_history[history_slice, 0, module.state_base._LOCAL_MASK_START] = 1.0
    mutated_prepared = replace(prepared, local_history_tensor=local_history)

    predictions, _targets, _valid = module.persistence_predictions(mutated_prepared, windows, train_fallback=fallback)

    assert np.allclose(predictions[0, :, 0, 0], fallback[0])


def test_tft_dataset_expands_to_turbine_instances_and_calendar_future(tmp_path, monkeypatch) -> None:
    module = _load_module()
    _require_torch(module)
    prepared = _prepare_temp_dataset(module, tmp_path, monkeypatch, variant_spec=module.VARIANT_SPECS[1])

    dataset = module.TurbineWindowDataset(prepared, prepared.val_rolling_windows, include_indices=True)
    sample = dataset[0]

    assert len(dataset) == len(prepared.val_rolling_windows) * prepared.node_count
    assert sample[0].shape == (prepared.history_steps, 54)
    assert sample[1].shape == (prepared.history_steps, 40)
    assert sample[2].shape == (prepared.forecast_steps, 7)
    assert sample[3].shape == (6,)
    assert sample[4].shape == (prepared.forecast_steps, 1)
    assert sample[5].shape == (prepared.forecast_steps, 1)
    assert int(sample[6]) == 0
    assert int(sample[7]) == 0
    assert prepared.context_future_feature_names == _KNOWN_FUTURE_COLUMNS
    assert not any("farm" in name.lower() for name in prepared.context_future_feature_names)


def test_chronos_input_adapter_restores_nan_and_calendar_covariates(tmp_path, monkeypatch) -> None:
    module = _load_module()
    prepared = _prepare_temp_dataset(
        module,
        tmp_path,
        monkeypatch,
        variant_spec=_chronos_variant_spec(module),
    )
    target_index = int(prepared.val_rolling_windows.target_indices[0])
    history_slice = slice(target_index - prepared.history_steps, target_index)
    future_slice = slice(target_index, target_index + prepared.forecast_steps)

    local_history = prepared.local_history_tensor.copy()
    context_history = prepared.context_history_tensor.copy()
    local_history[history_slice.start, 0, module.state_base._LOCAL_VALUE_START] = 0.4321
    local_history[history_slice.start, 0, module.state_base._LOCAL_MASK_START] = 1.0
    local_history[history_slice.start, 0, module.state_base._LOCAL_VALUE_START + 1] = 0.6543
    local_history[history_slice.start, 0, module.state_base._LOCAL_MASK_START + 1] = 1.0
    context_history[history_slice.start, module.state_base._CONTEXT_GLOBAL_VALUE_START] = 0.9876
    context_history[history_slice.start, module.state_base._CONTEXT_GLOBAL_MASK_START] = 1.0
    mutated_prepared = replace(prepared, local_history_tensor=local_history, context_history_tensor=context_history)

    chronos_input = module.build_chronos_zero_shot_input(
        mutated_prepared,
        target_index=target_index,
        node_index=0,
    )

    past_covariates = chronos_input["past_covariates"]
    future_covariates = chronos_input["future_covariates"]
    assert isinstance(past_covariates, dict)
    assert isinstance(future_covariates, dict)
    assert np.isnan(np.asarray(chronos_input["target"])[0])
    assert np.isnan(
        np.asarray(past_covariates[prepared.local_input_feature_names[module.state_base._LOCAL_VALUE_START + 1]])[0]
    )
    assert np.isnan(
        np.asarray(past_covariates[prepared.context_history_feature_names[module.state_base._CONTEXT_GLOBAL_VALUE_START]])[0]
    )
    assert tuple(future_covariates) == _KNOWN_FUTURE_COLUMNS
    assert set(future_covariates).issubset(set(past_covariates))
    for column_name in _KNOWN_FUTURE_COLUMNS:
        assert np.allclose(
            np.asarray(future_covariates[column_name]),
            prepared.context_future_tensor[future_slice, _KNOWN_FUTURE_COLUMNS.index(column_name)],
        )


def test_timexer_dataset_splits_endogenous_exogenous_and_history_marks(tmp_path, monkeypatch) -> None:
    module = _load_module()
    _require_torch(module)
    prepared = _prepare_temp_dataset(
        module,
        tmp_path,
        monkeypatch,
        variant_spec=_timexer_variant_spec(module),
    )

    layout = module.resolve_timexer_feature_layout(prepared)
    dataset = module.TimeXerWindowDataset(prepared, prepared.val_rolling_windows, include_indices=True)
    sample = dataset[0]

    assert len(dataset) == len(prepared.val_rolling_windows) * prepared.node_count
    assert layout.endogenous_history_names == ("target_pu",)
    assert layout.history_mark_names == _KNOWN_FUTURE_COLUMNS
    assert len(layout.exogenous_history_names) == 86
    assert layout.exogenous_history_names[0] == "target_kw__mask"
    assert "farm_pmu__gms_power_kw" in layout.exogenous_history_names
    assert sample[0].shape == (prepared.history_steps, 1)
    assert sample[1].shape == (prepared.history_steps, 86)
    assert sample[2].shape == (prepared.history_steps, 7)
    assert sample[3].shape == (prepared.forecast_steps, 1)
    assert sample[4].shape == (prepared.forecast_steps, 1)
    assert int(sample[5]) == 0
    assert int(sample[6]) == 0


def test_tft_forward_shape_and_bounded_output(tmp_path, monkeypatch) -> None:
    module = _load_module()
    torch_module = _require_torch(module)
    prepared = _prepare_temp_dataset(module, tmp_path, monkeypatch, variant_spec=module.VARIANT_SPECS[1])
    model = module.build_tft_model(
        prepared_dataset=prepared,
        d_model=8,
        lstm_hidden_dim=8,
        attention_heads=2,
        dropout=0.0,
        bounded_output_epsilon=module.DEFAULT_BOUNDED_OUTPUT_EPSILON,
    )
    sample = module.TurbineWindowDataset(prepared, prepared.val_rolling_windows)[0]

    with torch_module.no_grad():
        outputs = model(
            torch_module.from_numpy(sample[0][None]),
            torch_module.from_numpy(sample[1][None]),
            torch_module.from_numpy(sample[2][None]),
            torch_module.from_numpy(sample[3][None]),
        )

    assert outputs.shape == (1, prepared.forecast_steps, 1)
    assert float(outputs.min().item()) >= 0.0
    assert float(outputs.max().item()) <= 1.05


def test_timexer_forward_shape_and_bounded_output(tmp_path, monkeypatch) -> None:
    module = _load_module()
    torch_module = _require_torch(module)
    prepared = _prepare_temp_dataset(
        module,
        tmp_path,
        monkeypatch,
        variant_spec=_timexer_variant_spec(module),
    )
    model = module.build_timexer_model(
        prepared_dataset=prepared,
        d_model=8,
        attention_heads=2,
        patch_len=24,
        encoder_layers=1,
        ff_hidden_dim=16,
        dropout=0.0,
        bounded_output_epsilon=module.DEFAULT_BOUNDED_OUTPUT_EPSILON,
    )
    sample = module.TimeXerWindowDataset(prepared, prepared.val_rolling_windows)[0]

    with torch_module.no_grad():
        outputs = model(
            torch_module.from_numpy(sample[0][None]),
            torch_module.from_numpy(sample[1][None]),
            torch_module.from_numpy(sample[2][None]),
        )

    assert outputs.shape == (1, prepared.forecast_steps, 1)
    assert float(outputs.min().item()) >= 0.0
    assert float(outputs.max().item()) <= 1.05


def test_itransformer_dataset_returns_full_window_tensors(tmp_path, monkeypatch) -> None:
    module = _load_module()
    _require_torch(module)
    prepared = _prepare_temp_dataset(
        module,
        tmp_path,
        monkeypatch,
        variant_spec=_itransformer_variant_spec(module),
    )

    dataset = module.ITransformerWindowDataset(prepared, prepared.val_rolling_windows, include_indices=True)
    sample = dataset[0]

    assert len(dataset) == len(prepared.val_rolling_windows)
    assert sample[0].shape == (prepared.history_steps, prepared.node_count, prepared.local_input_channels)
    assert sample[1].shape == (prepared.history_steps, prepared.context_history_channels)
    assert sample[2].shape == (prepared.forecast_steps, prepared.context_future_channels)
    assert sample[3].shape == (prepared.node_count, prepared.static_feature_count)
    assert sample[4].shape == (prepared.forecast_steps, prepared.node_count, 1)
    assert sample[5].shape == (prepared.forecast_steps, prepared.node_count, 1)
    assert int(sample[6]) == 0


def test_itransformer_forward_shape_and_bounded_output(tmp_path, monkeypatch) -> None:
    module = _load_module()
    torch_module = _require_torch(module)
    prepared = _prepare_temp_dataset(
        module,
        tmp_path,
        monkeypatch,
        variant_spec=_itransformer_variant_spec(module),
    )
    model = module.build_itransformer_model(
        prepared_dataset=prepared,
        d_model=8,
        attention_heads=2,
        dropout=0.0,
        bounded_output_epsilon=module.DEFAULT_BOUNDED_OUTPUT_EPSILON,
    )
    sample = module.ITransformerWindowDataset(prepared, prepared.val_rolling_windows)[0]

    with torch_module.no_grad():
        outputs = model(
            torch_module.from_numpy(sample[0][None]),
            torch_module.from_numpy(sample[1][None]),
            torch_module.from_numpy(sample[2][None]),
            torch_module.from_numpy(sample[3][None]),
        )

    assert outputs.shape == (1, prepared.forecast_steps, prepared.node_count, 1)
    assert float(outputs.min().item()) >= 0.0
    assert float(outputs.max().item()) <= 1.05


def test_tft_eval_reaggregates_turbine_instances(tmp_path, monkeypatch) -> None:
    module = _load_module()
    torch_module = _require_torch(module)
    prepared = _prepare_temp_dataset(
        module,
        tmp_path,
        monkeypatch,
        max_train_origins=2,
        max_eval_origins=1,
        variant_spec=module.VARIANT_SPECS[1],
    )
    captured: dict[str, tuple[int, ...]] = {}
    original_metrics = module._metrics_from_arrays

    def _capture_metrics(predictions, targets, valid_mask, *, rated_power_kw):
        captured["predictions"] = predictions.shape
        captured["targets"] = targets.shape
        captured["valid"] = valid_mask.shape
        return original_metrics(predictions, targets, valid_mask, rated_power_kw=rated_power_kw)

    class _ZeroModel(torch_module.nn.Module):
        def forward(self, local_history, context_history, context_future, static_features):
            del local_history, context_history, static_features
            return torch_module.zeros(
                (context_future.shape[0], context_future.shape[1], 1),
                dtype=context_future.dtype,
                device=context_future.device,
            )

    monkeypatch.setattr(module, "_metrics_from_arrays", _capture_metrics)
    metrics = module.evaluate_tft_model(
        _ZeroModel(),
        prepared,
        prepared.val_rolling_windows,
        batch_size=2,
        device="cpu",
        seed=123,
        num_workers=0,
    )

    expected_shape = (len(prepared.val_rolling_windows), prepared.forecast_steps, prepared.node_count, 1)
    assert captured["predictions"] == expected_shape
    assert captured["targets"] == expected_shape
    assert captured["valid"] == expected_shape
    assert metrics.window_count == len(prepared.val_rolling_windows)


def test_itransformer_eval_reaggregates_full_window_predictions(tmp_path, monkeypatch) -> None:
    module = _load_module()
    torch_module = _require_torch(module)
    prepared = _prepare_temp_dataset(
        module,
        tmp_path,
        monkeypatch,
        max_train_origins=2,
        max_eval_origins=1,
        variant_spec=_itransformer_variant_spec(module),
    )
    captured: dict[str, tuple[int, ...]] = {}
    original_metrics = module._metrics_from_arrays

    def _capture_metrics(predictions, targets, valid_mask, *, rated_power_kw):
        captured["predictions"] = predictions.shape
        captured["targets"] = targets.shape
        captured["valid"] = valid_mask.shape
        return original_metrics(predictions, targets, valid_mask, rated_power_kw=rated_power_kw)

    class _ZeroModel(torch_module.nn.Module):
        def forward(self, local_history, context_history, context_future, static_features):
            del local_history, context_history, static_features
            return torch_module.zeros(
                (context_future.shape[0], context_future.shape[1], prepared.node_count, 1),
                dtype=context_future.dtype,
                device=context_future.device,
            )

    monkeypatch.setattr(module, "_metrics_from_arrays", _capture_metrics)
    metrics = module.evaluate_itransformer_model(
        _ZeroModel(),
        prepared,
        prepared.val_rolling_windows,
        batch_size=2,
        device="cpu",
        seed=123,
        num_workers=0,
    )

    expected_shape = (len(prepared.val_rolling_windows), prepared.forecast_steps, prepared.node_count, 1)
    assert captured["predictions"] == expected_shape
    assert captured["targets"] == expected_shape
    assert captured["valid"] == expected_shape
    assert metrics.window_count == len(prepared.val_rolling_windows)


def test_timexer_eval_reaggregates_turbine_instances(tmp_path, monkeypatch) -> None:
    module = _load_module()
    torch_module = _require_torch(module)
    prepared = _prepare_temp_dataset(
        module,
        tmp_path,
        monkeypatch,
        max_train_origins=2,
        max_eval_origins=1,
        variant_spec=_timexer_variant_spec(module),
    )
    captured: dict[str, tuple[int, ...]] = {}
    original_metrics = module._metrics_from_arrays

    def _capture_metrics(predictions, targets, valid_mask, *, rated_power_kw):
        captured["predictions"] = predictions.shape
        captured["targets"] = targets.shape
        captured["valid"] = valid_mask.shape
        return original_metrics(predictions, targets, valid_mask, rated_power_kw=rated_power_kw)

    class _ZeroModel(torch_module.nn.Module):
        def forward(self, endogenous_history, _exogenous_history, _history_marks):
            return torch_module.zeros(
                (endogenous_history.shape[0], prepared.forecast_steps, 1),
                dtype=endogenous_history.dtype,
                device=endogenous_history.device,
            )

    monkeypatch.setattr(module, "_metrics_from_arrays", _capture_metrics)
    metrics = module.evaluate_timexer_model(
        _ZeroModel(),
        prepared,
        prepared.val_rolling_windows,
        batch_size=2,
        device="cpu",
        seed=123,
        num_workers=0,
    )

    expected_shape = (len(prepared.val_rolling_windows), prepared.forecast_steps, prepared.node_count, 1)
    assert captured["predictions"] == expected_shape
    assert captured["targets"] == expected_shape
    assert captured["valid"] == expected_shape
    assert metrics.window_count == len(prepared.val_rolling_windows)


def test_dgcrn_dataset_builds_farm_panel_history_and_future(tmp_path, monkeypatch) -> None:
    module = _load_module()
    _require_torch(module)
    prepared = _prepare_temp_dataset(
        module,
        tmp_path,
        monkeypatch,
        variant_spec=_dgcrn_variant_spec(module),
    )

    dataset = module.FarmPanelWindowDataset(prepared, prepared.val_rolling_windows)
    sample = dataset[0]

    assert len(dataset) == len(prepared.val_rolling_windows)
    assert sample[0].shape == (prepared.history_steps, prepared.node_count, 94)
    assert sample[1].shape == (prepared.forecast_steps, 7)
    assert sample[2].shape == (prepared.forecast_steps, prepared.node_count, 1)
    assert sample[3].shape == (prepared.forecast_steps, prepared.node_count, 1)


def test_dgcrn_forward_shape_and_dynamic_graph_changes_with_state(tmp_path, monkeypatch) -> None:
    module = _load_module()
    torch_module = _require_torch(module)
    prepared = _prepare_temp_dataset(
        module,
        tmp_path,
        monkeypatch,
        variant_spec=_dgcrn_variant_spec(module),
    )
    model = module.build_dgcrn_model(
        prepared_dataset=prepared,
        hidden_dim=8,
        embed_dim=4,
        num_layers=1,
        cheb_k=2,
    )
    sample = module.FarmPanelWindowDataset(prepared, prepared.val_rolling_windows)[0]

    with torch_module.no_grad():
        outputs = model(
            torch_module.from_numpy(sample[0][None]),
            torch_module.from_numpy(sample[1][None]),
        )
        zero_state = torch_module.zeros((1, prepared.node_count, 8), dtype=torch_module.float32)
        one_state = torch_module.ones((1, prepared.node_count, 8), dtype=torch_module.float32)
        adjacency_zero = model.compute_dynamic_adjacency(zero_state)
        adjacency_one = model.compute_dynamic_adjacency(one_state)

    assert outputs.shape == (1, prepared.forecast_steps, prepared.node_count, 1)
    assert adjacency_zero.shape == (1, prepared.node_count, prepared.node_count)
    assert not torch_module.allclose(adjacency_zero, adjacency_one)


def test_persistence_job_writes_synthetic_training_history(tmp_path, monkeypatch) -> None:
    module = _load_module()
    prepared = _prepare_temp_dataset(
        module,
        tmp_path,
        monkeypatch,
        max_train_origins=2,
        max_eval_origins=1,
        variant_spec=module.VARIANT_SPECS[0],
    )
    history_path = tmp_path / "persistence.training_history.csv"

    rows = module.execute_training_job(
        prepared,
        variant_spec=module.VARIANT_SPECS[0],
        seed=123,
        training_history_path=history_path,
    )

    history = pl.read_csv(history_path)
    assert history["epoch"].to_list() == [0]
    assert history["baseline_type"].to_list() == ["persistence_last_value"]
    assert history["train_loss_mean"].null_count() == 1
    assert history["val_rmse_pu"].null_count() == 0
    assert len(rows) == 148
    assert rows[0]["model_variant"] == module.PERSISTENCE_VARIANT
    assert rows[0]["uses_graph"] is False


def test_chronos_zero_shot_job_reaggregates_batches_and_writes_synthetic_history(tmp_path, monkeypatch) -> None:
    module = _load_module()
    prepared = _prepare_temp_dataset(
        module,
        tmp_path,
        monkeypatch,
        max_train_origins=2,
        max_eval_origins=1,
        variant_spec=_chronos_variant_spec(module),
    )
    history_path = tmp_path / "chronos.training_history.csv"
    captured_shapes: list[tuple[int, ...]] = []
    original_metrics = module._metrics_from_arrays

    def _capture_metrics(predictions, targets, valid_mask, *, rated_power_kw):
        captured_shapes.append(predictions.shape)
        assert predictions.shape == targets.shape == valid_mask.shape
        return original_metrics(predictions, targets, valid_mask, rated_power_kw=rated_power_kw)

    class _FakeChronosPipeline:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def predict_quantiles(
            self,
            inputs,
            *,
            prediction_length,
            batch_size,
            context_length,
            cross_learning,
            limit_prediction_length,
        ):
            self.calls.append(
                {
                    "count": len(inputs),
                    "prediction_length": prediction_length,
                    "batch_size": batch_size,
                    "context_length": context_length,
                    "cross_learning": cross_learning,
                    "limit_prediction_length": limit_prediction_length,
                }
            )
            means = []
            for item in inputs:
                target = np.asarray(item["target"], dtype=np.float32)
                anchor = float(np.nan_to_num(target[-1], nan=0.0))
                means.append(np.full((1, prediction_length), anchor, dtype=np.float32))
            return [None] * len(means), means

    fake_pipeline = _FakeChronosPipeline()
    monkeypatch.setattr(module, "_metrics_from_arrays", _capture_metrics)
    monkeypatch.setattr(module, "_load_chronos_zero_shot_pipeline", lambda *, device: fake_pipeline)

    rows = module.execute_training_job(
        prepared,
        variant_spec=_chronos_variant_spec(module),
        device="cpu",
        seed=123,
        batch_size=3,
        eval_batch_size=2,
        training_history_path=history_path,
    )

    expected_shape = (len(prepared.val_rolling_windows), prepared.forecast_steps, prepared.node_count, 1)
    assert captured_shapes
    assert all(shape == expected_shape for shape in captured_shapes)
    assert len(fake_pipeline.calls) == 8
    assert all(call["batch_size"] == 2 for call in fake_pipeline.calls)
    assert all(call["context_length"] == prepared.history_steps for call in fake_pipeline.calls)
    assert all(call["cross_learning"] is False for call in fake_pipeline.calls)
    history = pl.read_csv(history_path)
    assert history["epoch"].to_list() == [0]
    assert history["baseline_type"].to_list() == ["chronos_2_zero_shot"]
    assert history["train_loss_mean"].null_count() == 1
    assert history["device"].to_list() == ["cpu"]
    assert len(rows) == 148
    assert rows[0]["model_variant"] == module.CHRONOS_VARIANT
    assert rows[0]["baseline_type"] == "chronos_2_zero_shot"


def test_persistence_logs_tensorboard_scalars_with_fake_writer(tmp_path, monkeypatch) -> None:
    module = _load_module()
    _FakeSummaryWriter.instances.clear()
    monkeypatch.setattr(module, "SummaryWriter", _FakeSummaryWriter)
    prepared = _prepare_temp_dataset(
        module,
        tmp_path,
        monkeypatch,
        max_train_origins=2,
        max_eval_origins=1,
        variant_spec=module.VARIANT_SPECS[0],
    )
    log_dir = tmp_path / "tensorboard" / "persistence"

    module.execute_training_job(
        prepared,
        variant_spec=module.VARIANT_SPECS[0],
        seed=123,
        tensorboard_log_dir=log_dir,
    )

    assert log_dir.exists()
    assert len(_FakeSummaryWriter.instances) == 1
    writer = _FakeSummaryWriter.instances[0]
    assert Path(writer.log_dir) == log_dir.resolve()
    assert writer.closed is True
    assert any(tag == "run/config_json" for tag, _text, _step in writer.texts)
    assert any(
        tag == "final/val/rolling_origin_no_refit/overall/mae_pu"
        for tag, _value, _step in writer.scalars
    )


def test_tft_one_epoch_cpu_smoke_writes_history(tmp_path, monkeypatch) -> None:
    module = _load_module()
    _require_torch(module)
    prepared = _prepare_temp_dataset(
        module,
        tmp_path,
        monkeypatch,
        max_train_origins=2,
        max_eval_origins=1,
        variant_spec=module.VARIANT_SPECS[1],
    )
    history_path = tmp_path / "tft.training_history.csv"

    rows = module.execute_training_job(
        prepared,
        variant_spec=module.VARIANT_SPECS[1],
        device="cpu",
        seed=123,
        batch_size=3,
        eval_batch_size=3,
        learning_rate=1e-3,
        max_epochs=1,
        early_stopping_patience=1,
        d_model=8,
        lstm_hidden_dim=8,
        attention_heads=2,
        dropout=0.0,
        num_workers=0,
        training_history_path=history_path,
    )

    history = pl.read_csv(history_path)
    assert history["epoch"].to_list() == [1]
    assert history["baseline_type"].to_list() == ["shared_weight_tft_no_graph"]
    assert history["train_loss_mean"].null_count() == 0
    assert history["val_rmse_pu"].null_count() == 0
    assert len(rows) == 148
    assert rows[0]["model_variant"] == module.TFT_VARIANT
    assert rows[0]["d_model"] == 8
    assert rows[0]["uses_future_observations"] is False


def test_timexer_one_epoch_cpu_smoke_writes_history(tmp_path, monkeypatch) -> None:
    module = _load_module()
    _require_torch(module)
    prepared = _prepare_temp_dataset(
        module,
        tmp_path,
        monkeypatch,
        max_train_origins=2,
        max_eval_origins=1,
        variant_spec=_timexer_variant_spec(module),
    )
    history_path = tmp_path / "timexer.training_history.csv"

    rows = module.execute_training_job(
        prepared,
        variant_spec=_timexer_variant_spec(module),
        device="cpu",
        seed=123,
        batch_size=3,
        eval_batch_size=3,
        learning_rate=1e-3,
        max_epochs=1,
        early_stopping_patience=1,
        d_model=8,
        attention_heads=2,
        patch_len=24,
        encoder_layers=1,
        ff_hidden_dim=16,
        dropout=0.0,
        num_workers=0,
        training_history_path=history_path,
    )

    history = pl.read_csv(history_path)
    assert history["epoch"].to_list() == [1]
    assert history["baseline_type"].to_list() == ["shared_weight_timexer_no_graph"]
    assert history["patch_len"].to_list() == [24]
    assert history["encoder_layers"].to_list() == [1]
    assert history["ff_hidden_dim"].to_list() == [16]
    assert history["train_loss_mean"].null_count() == 0
    assert history["val_rmse_pu"].null_count() == 0
    assert len(rows) == 148
    assert rows[0]["model_variant"] == module.TIMEXER_VARIANT
    assert rows[0]["patch_len"] == 24
    assert rows[0]["encoder_layers"] == 1
    assert rows[0]["ff_hidden_dim"] == 16


def test_itransformer_one_epoch_cpu_smoke_writes_history(tmp_path, monkeypatch) -> None:
    module = _load_module()
    _require_torch(module)
    prepared = _prepare_temp_dataset(
        module,
        tmp_path,
        monkeypatch,
        max_train_origins=2,
        max_eval_origins=1,
        variant_spec=_itransformer_variant_spec(module),
    )
    history_path = tmp_path / "itransformer.training_history.csv"

    rows = module.execute_training_job(
        prepared,
        variant_spec=_itransformer_variant_spec(module),
        device="cpu",
        seed=123,
        batch_size=2,
        eval_batch_size=2,
        learning_rate=1e-3,
        max_epochs=1,
        early_stopping_patience=1,
        d_model=8,
        attention_heads=2,
        dropout=0.0,
        num_workers=0,
        training_history_path=history_path,
    )

    history = pl.read_csv(history_path)
    assert history["epoch"].to_list() == [1]
    assert history["baseline_type"].to_list() == ["itransformer_no_graph"]
    assert history["train_loss_mean"].null_count() == 0
    assert history["val_rmse_pu"].null_count() == 0
    assert len(rows) == 148
    assert rows[0]["model_variant"] == module.ITRANSFORMER_VARIANT
    assert rows[0]["d_model"] == 8
    assert rows[0]["lstm_hidden_dim"] is None
    assert rows[0]["uses_future_observations"] is False


def test_dgcrn_one_epoch_cpu_smoke_writes_history(tmp_path, monkeypatch) -> None:
    module = _load_module()
    _require_torch(module)
    prepared = _prepare_temp_dataset(
        module,
        tmp_path,
        monkeypatch,
        max_train_origins=2,
        max_eval_origins=1,
        variant_spec=_dgcrn_variant_spec(module),
    )
    history_path = tmp_path / "dgcrn.training_history.csv"

    rows = module.execute_training_job(
        prepared,
        variant_spec=_dgcrn_variant_spec(module),
        device="cpu",
        seed=123,
        batch_size=2,
        eval_batch_size=2,
        learning_rate=1e-3,
        max_epochs=1,
        early_stopping_patience=1,
        hidden_dim=8,
        embed_dim=4,
        num_layers=1,
        cheb_k=2,
        teacher_forcing_ratio=0.0,
        num_workers=0,
        training_history_path=history_path,
    )

    history = pl.read_csv(history_path)
    assert history["epoch"].to_list() == [1]
    assert history["baseline_type"].to_list() == ["dgcrn_dynamic_graph"]
    assert history["hidden_dim"].to_list() == [8]
    assert history["embed_dim"].to_list() == [4]
    assert history["num_layers"].to_list() == [1]
    assert history["cheb_k"].to_list() == [2]
    assert history["teacher_forcing_ratio"].to_list() == [0.0]
    assert history["train_loss_mean"].null_count() == 0
    assert history["val_rmse_pu"].null_count() == 0
    assert len(rows) == 148
    assert rows[0]["model_variant"] == module.DGCRN_VARIANT
    assert rows[0]["hidden_dim"] == 8
    assert rows[0]["uses_graph"] is True
    assert rows[0]["uses_pairwise"] is True


def test_run_experiment_writes_results_and_hashed_resume_state(tmp_path, monkeypatch) -> None:
    module = _load_module()
    prepared = _prepare_temp_dataset(module, tmp_path, monkeypatch, max_train_origins=2, max_eval_origins=1)
    output_path = tmp_path / "published" / "latest.csv"
    work_root = tmp_path / ".work"

    def _dataset_loader(*_args, **_kwargs):
        return prepared

    def _job_runner(prepared_dataset, **kwargs):
        variant_spec = kwargs["variant_spec"]
        profile = module.resolve_hyperparameter_profile(
            variant_spec.model_variant,
            dataset_id=prepared_dataset.dataset_id,
            batch_size=kwargs["batch_size"],
            learning_rate=kwargs["learning_rate"],
            max_epochs=kwargs["max_epochs"],
            early_stopping_patience=kwargs["early_stopping_patience"],
            d_model=kwargs["d_model"],
            lstm_hidden_dim=kwargs["lstm_hidden_dim"],
            attention_heads=kwargs["attention_heads"],
            patch_len=kwargs["patch_len"],
            encoder_layers=kwargs["encoder_layers"],
            ff_hidden_dim=kwargs["ff_hidden_dim"],
            hidden_dim=kwargs["hidden_dim"],
            embed_dim=kwargs["embed_dim"],
            num_layers=kwargs["num_layers"],
            cheb_k=kwargs["cheb_k"],
            teacher_forcing_ratio=kwargs["teacher_forcing_ratio"],
            dropout=kwargs["dropout"],
            grad_clip_norm=kwargs["grad_clip_norm"],
            weight_decay=kwargs["weight_decay"],
            bounded_output_epsilon=kwargs["bounded_output_epsilon"],
        )
        metrics = module.EvaluationMetrics(
            window_count=1,
            prediction_count=prepared_dataset.forecast_steps * prepared_dataset.node_count,
            mae_kw=1.0,
            rmse_kw=2.0,
            mae_pu=0.1,
            rmse_pu=0.2,
            horizon_window_count=np.ones((prepared_dataset.forecast_steps,), dtype=np.int64),
            horizon_prediction_count=np.full(
                (prepared_dataset.forecast_steps,),
                prepared_dataset.node_count,
                dtype=np.int64,
            ),
            horizon_mae_kw=np.ones((prepared_dataset.forecast_steps,), dtype=np.float64),
            horizon_rmse_kw=np.full((prepared_dataset.forecast_steps,), 2.0, dtype=np.float64),
            horizon_mae_pu=np.full((prepared_dataset.forecast_steps,), 0.1, dtype=np.float64),
            horizon_rmse_pu=np.full((prepared_dataset.forecast_steps,), 0.2, dtype=np.float64),
        )
        evaluation_results = [
            (split_name, eval_protocol, windows, metrics)
            for split_name, eval_protocol, windows in module.iter_evaluation_specs(prepared_dataset)
        ]
        training_outcome = module.TrainingOutcome(
            best_epoch=0,
            epochs_ran=0,
            best_val_rmse_pu=0.2,
            best_val_mae_pu=0.1,
            device="cpu",
            amp_enabled=False,
            model=None,
        )
        return module.build_result_rows(
            prepared_dataset,
            variant_spec=variant_spec,
            training_outcome=training_outcome,
            runtime_seconds=0.5,
            seed=kwargs["seed"],
            profile=profile,
            evaluation_results=evaluation_results,
        )

    results = module.run_experiment(
        dataset_ids=(prepared.dataset_id,),
        output_path=output_path,
        device="cpu",
        max_epochs=1,
        seed=7,
        batch_size=2,
        learning_rate=1e-3,
        d_model=8,
        lstm_hidden_dim=8,
        attention_heads=2,
        patch_len=24,
        encoder_layers=1,
        ff_hidden_dim=16,
        hidden_dim=8,
        embed_dim=4,
        num_layers=1,
        cheb_k=2,
        teacher_forcing_ratio=0.0,
        dropout=0.0,
        work_root=work_root,
        dataset_loader=_dataset_loader,
        job_runner=_job_runner,
    )

    assert output_path.exists()
    assert module.training_history_output_path(output_path).exists()
    assert results.height == 888
    assert set(results["model_variant"].unique().to_list()) == {
        module.PERSISTENCE_VARIANT,
        module.TFT_VARIANT,
        module.TIMEXER_VARIANT,
        module.DGCRN_VARIANT,
        module.CHRONOS_VARIANT,
        module.ITRANSFORMER_VARIANT,
    }
    paths = module._resume_paths_for_output(output_path=output_path, work_root=work_root)
    expected_slot = hashlib.sha256(str(output_path.resolve()).encode("utf-8")).hexdigest()
    assert paths.slot_dir.name == expected_slot
    assert paths.partial_results_path.exists()
    assert paths.checkpoints_dir.exists()
    assert module._job_checkpoint_path(
        paths,
        dataset_id=prepared.dataset_id,
        model_variant=module.TFT_VARIANT,
    ).name == f"{prepared.dataset_id}__{module.TFT_VARIANT}.pt"
    state_payload = json.loads(paths.state_path.read_text(encoding="utf-8"))
    assert state_payload["status"] == "complete"

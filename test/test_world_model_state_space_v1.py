from __future__ import annotations

import hashlib
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from test.test_agcrn import _require_torch
from test.test_world_model_rollout_v1 import (
    _GLOBAL_OBSERVATION_VALUE_COLUMNS,
    _KNOWN_FUTURE_COLUMNS,
    _LOCAL_OBSERVATION_VALUE_COLUMNS,
    _build_world_model_temp_cache,
    _patch_bundle_loader,
)


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "experiment"
        / "families"
        / "world_model_state_space_v1"
        / "world_model_state_space_v1.py"
    )
    spec = spec_from_file_location("world_model_state_space_v1", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_search_module():
    _load_module()
    module_path = (
        Path(__file__).resolve().parents[1]
        / "experiment"
        / "families"
        / "world_model_state_space_v1"
        / "search_world_model_state_space_v1.py"
    )
    spec = spec_from_file_location("search_world_model_state_space_v1", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_main_defaults_output_to_timestamped_publish_path(monkeypatch, capsys) -> None:
    module = _load_module()
    expected_output_path = (
        module._REPO_ROOT
        / "experiment"
        / "artifacts"
        / "published"
        / module.FAMILY_ID
        / "20260418-070809.csv"
    )
    recorded: dict[str, Path] = {}

    monkeypatch.setattr(module, "generate_run_stem", lambda: "20260418-070809")

    def _fake_run_experiment(**kwargs):
        recorded["output_path"] = kwargs["output_path"]
        return pl.DataFrame({"row": [1]})

    monkeypatch.setattr(module, "run_experiment", _fake_run_experiment)

    assert module.main(["--no-record-run", "--disable-tensorboard"]) == 0
    assert recorded["output_path"] == expected_output_path
    assert str(expected_output_path) in capsys.readouterr().out


def test_main_requires_output_path_for_resume_or_force_rerun(capsys) -> None:
    module = _load_module()

    for flag in ("--resume", "--force-rerun"):
        with pytest.raises(SystemExit) as exc:
            module.main([flag, "--no-record-run", "--disable-tensorboard"])
        assert exc.value.code == 2
        assert "requires --output-path" in capsys.readouterr().err


def _prepare_temp_dataset(
    module,
    tmp_path: Path,
    monkeypatch,
    *,
    dataset_id: str = "kelmarsh",
    variant_name: str | None = None,
    max_train_origins: int = 4,
    max_eval_origins: int = 2,
):
    cache_root = tmp_path / "cache"
    _build_world_model_temp_cache(cache_root, dataset_id=dataset_id)
    _patch_bundle_loader(monkeypatch, module.rollout_base, cache_root, dataset_id=dataset_id)
    variant_spec = None
    if variant_name is not None:
        variant_spec = module.resolve_variant_specs((variant_name,))[0]
    return module.prepare_dataset(
        dataset_id,
        variant_spec=variant_spec,
        cache_root=cache_root,
        max_train_origins=max_train_origins,
        max_eval_origins=max_eval_origins,
    )


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


def _tiny_model_kwargs(module, prepared, *, variant_name: str | None = None) -> dict[str, object]:
    resolved_variant_name = prepared.model_variant if variant_name is None else variant_name
    variant_spec = module.resolve_variant_specs((resolved_variant_name,))[0]
    return {
        "node_count": prepared.node_count,
        "local_input_channels": prepared.local_input_channels,
        "context_history_channels": prepared.context_history_channels,
        "context_future_channels": prepared.context_future_channels,
        "static_tensor": prepared.static_tensor,
        "turbine_indices": prepared.turbine_indices,
        "pairwise_tensor": prepared.pairwise_tensor,
        "wake_geometry_tensor": prepared.wake_geometry_tensor,
        "persistence_fallback_pu": prepared.persistence_train_fallback_pu,
        "z_dim": 4,
        "h_dim": 6,
        "global_state_dim": 8,
        "obs_encoding_dim": 8,
        "innovation_dim": 4,
        "source_summary_dim": 4,
        "edge_message_dim": 4,
        "edge_hidden_dim": 8,
        "tau_embed_dim": 4,
        "met_summary_dim": module.DEFAULT_MET_SUMMARY_DIM,
        "turbine_embed_dim": 3,
        "forecast_steps": prepared.forecast_steps,
        "dropout": 0.0,
        "uses_graph": variant_spec.uses_graph,
        "uses_wake_dynamic": variant_spec.uses_wake_dynamic,
        "uses_persistence_residual_head": variant_spec.uses_persistence_residual_head,
        "wake_lambda_x": module.DEFAULT_WAKE_LAMBDA_X,
        "wake_lambda_y": module.DEFAULT_WAKE_LAMBDA_Y,
        "wake_kappa": module.DEFAULT_WAKE_KAPPA,
        "bounded_output_epsilon": module.DEFAULT_BOUNDED_OUTPUT_EPSILON,
    }


def _tiny_profile(module, prepared, *, variant_name: str | None = None, selection_metric: str | None = None):
    resolved_variant_name = prepared.model_variant if variant_name is None else variant_name
    return module.resolve_hyperparameter_profile(
        resolved_variant_name,
        dataset_id=prepared.dataset_id,
        batch_size=2,
        learning_rate=1e-3,
        max_epochs=1,
        early_stopping_patience=1,
        z_dim=4,
        h_dim=6,
        global_state_dim=8,
        obs_encoding_dim=8,
        innovation_dim=4,
        source_summary_dim=4,
        edge_message_dim=4,
        edge_hidden_dim=8,
        tau_embed_dim=4,
        met_summary_dim=module.DEFAULT_MET_SUMMARY_DIM,
        turbine_embed_dim=3,
        dropout=0.0,
    )


def _take_first_windows(module, windows, count: int):
    return module.world_model_base.FarmWindowDescriptorIndex(
        target_indices=windows.target_indices[:count].copy(),
        output_start_us=windows.output_start_us[:count].copy(),
        output_end_us=windows.output_end_us[:count].copy(),
    )


def test_registry_declares_kelmarsh_only() -> None:
    registry_path = (
        Path(__file__).resolve().parents[1]
        / "experiment"
        / "infra"
        / "registry"
        / "families"
        / "world_model_state_space_v1.toml"
    )
    text = registry_path.read_text(encoding="utf-8")

    assert 'family_id = "world_model_state_space_v1"' in text
    assert 'dataset_scope = ["kelmarsh"]' in text
    assert 'supported_feature_protocols = ["world_model_v1"]' in text
    assert 'world_model_state_space_v1_farm_sync = "world_model_v1"' in text
    assert 'world_model_state_space_v1_residual_persistence_farm_sync = "world_model_v1"' in text
    assert 'world_model_state_space_v1_wake_off_farm_sync = "world_model_v1"' in text
    assert 'world_model_state_space_v1_graph_off_farm_sync = "world_model_v1"' in text
    assert 'world_model_state_space_v1_no_farm_aux_farm_sync = "world_model_v1"' in text
    assert 'world_model_state_space_v1_no_met_aux_farm_sync = "world_model_v1"' in text


def test_non_kelmarsh_dataset_is_rejected() -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="only supports"):
        module.run_experiment(dataset_ids=("penmanshiel",), output_path=Path("/tmp/unused.csv"))


def test_tensorboard_root_uses_output_hash_by_default(tmp_path) -> None:
    module = _load_module()
    output_path = tmp_path / "published" / "latest.csv"
    work_root = tmp_path / ".work"

    root = module.resolve_tensorboard_root(output_path=output_path, work_root=work_root)
    expected = module._resume_paths_for_output(output_path=output_path, work_root=work_root).slot_dir / "tensorboard"

    assert root == expected


def test_default_hyperparameters_use_tuned_kelmarsh_batch_size() -> None:
    module = _load_module()
    profile = module.resolve_hyperparameter_profile(module.MODEL_VARIANT, dataset_id="kelmarsh")

    assert profile.batch_size == 432
    assert module.resolve_loader_num_workers(device="cuda") == 4


def test_default_variants_remain_canonical_only() -> None:
    module = _load_module()

    assert module.DEFAULT_VARIANTS == (module.MODEL_VARIANT,)
    assert set(module.ALL_VARIANTS) == {
        module.MODEL_VARIANT,
        module.RESIDUAL_PERSISTENCE_MODEL_VARIANT,
        module.WAKE_OFF_MODEL_VARIANT,
        module.GRAPH_OFF_MODEL_VARIANT,
        module.NO_FARM_AUX_MODEL_VARIANT,
        module.NO_MET_AUX_MODEL_VARIANT,
    }
    parser = module.build_arg_parser()
    variant_action = next(action for action in parser._actions if action.dest == "variants")
    assert set(variant_action.choices) == set(module.ALL_VARIANTS)


def test_horizon_rmse_group_means_cover_expected_ranges() -> None:
    module = _load_module()
    metrics = module.EvaluationMetrics(
        window_count=1,
        prediction_count=36,
        mae_kw=0.0,
        rmse_kw=0.0,
        mae_pu=0.0,
        rmse_pu=0.0,
        horizon_window_count=np.ones((36,), dtype=np.int64),
        horizon_prediction_count=np.ones((36,), dtype=np.int64),
        horizon_mae_kw=np.zeros((36,), dtype=np.float64),
        horizon_rmse_kw=np.zeros((36,), dtype=np.float64),
        horizon_mae_pu=np.zeros((36,), dtype=np.float64),
        horizon_rmse_pu=np.arange(1.0, 37.0, dtype=np.float64),
    )

    summary = module._horizon_rmse_pu_group_means(metrics)

    assert summary["val_rmse_pu_leads_13_24_mean"] == pytest.approx(18.5)
    assert summary["val_rmse_pu_leads_25_36_mean"] == pytest.approx(30.5)


def test_resolve_hyperparameter_profile_applies_ablation_guardrails() -> None:
    module = _load_module()

    wake_profile = module.resolve_hyperparameter_profile(module.WAKE_OFF_MODEL_VARIANT, dataset_id="kelmarsh")
    graph_profile = module.resolve_hyperparameter_profile(module.GRAPH_OFF_MODEL_VARIANT, dataset_id="kelmarsh")
    no_farm_profile = module.resolve_hyperparameter_profile(module.NO_FARM_AUX_MODEL_VARIANT, dataset_id="kelmarsh")
    no_met_profile = module.resolve_hyperparameter_profile(module.NO_MET_AUX_MODEL_VARIANT, dataset_id="kelmarsh")

    assert wake_profile.wake_lambda_x == 0.0
    assert wake_profile.wake_lambda_y == 0.0
    assert wake_profile.wake_kappa == 0.0
    assert graph_profile.wake_lambda_x == 0.0
    assert graph_profile.wake_lambda_y == 0.0
    assert graph_profile.wake_kappa == 0.0
    assert no_farm_profile.farm_loss_weight == 0.0
    assert no_met_profile.met_loss_weight == 0.0

    with pytest.raises(ValueError, match="farm_loss_weight=0.0"):
        module.resolve_hyperparameter_profile(
            module.NO_FARM_AUX_MODEL_VARIANT,
            dataset_id="kelmarsh",
            farm_loss_weight=0.1,
        )
    with pytest.raises(ValueError, match="met_loss_weight=0.0"):
        module.resolve_hyperparameter_profile(
            module.NO_MET_AUX_MODEL_VARIANT,
            dataset_id="kelmarsh",
            met_loss_weight=0.05,
        )
    with pytest.raises(ValueError, match="wake_lambda_x"):
        module.resolve_hyperparameter_profile(
            module.WAKE_OFF_MODEL_VARIANT,
            dataset_id="kelmarsh",
            wake_lambda_x=99.0,
        )


def test_prepare_dataset_builds_state_space_contract(tmp_path, monkeypatch) -> None:
    module = _load_module()
    prepared = _prepare_temp_dataset(module, tmp_path, monkeypatch)

    assert prepared.model_variant == module.MODEL_VARIANT
    assert prepared.feature_protocol_id == module.FEATURE_PROTOCOL_ID
    assert prepared.local_input_channels == 54
    assert prepared.context_history_channels == 40
    assert prepared.context_future_channels == 7
    assert prepared.static_feature_count == 6
    assert prepared.pairwise_feature_count == 7
    assert prepared.persistence_train_fallback_pu.shape == (prepared.node_count,)
    assert prepared.local_input_feature_names[:17] == ("target_pu", *_LOCAL_OBSERVATION_VALUE_COLUMNS)
    assert prepared.local_input_feature_names[17:34] == (
        "target_kw__mask",
        *(f"{column}__mask" for column in _LOCAL_OBSERVATION_VALUE_COLUMNS),
    )
    assert prepared.local_input_feature_names[-3:] == ("is_observed", "row_bad", "feat_bad")
    assert prepared.context_history_feature_names[:9] == _GLOBAL_OBSERVATION_VALUE_COLUMNS
    assert prepared.context_history_feature_names[33:] == _KNOWN_FUTURE_COLUMNS
    assert prepared.context_future_feature_names == _KNOWN_FUTURE_COLUMNS
    assert prepared.future_met_feature_names == module.SITE_SUMMARY_FEATURE_NAMES
    assert prepared.future_met_tensor.shape[1] == module.DEFAULT_MET_SUMMARY_DIM
    assert prepared.future_farm_target_pu.shape[1] == 1

    target_mask = prepared.local_history_tensor[200, 0, module._LOCAL_MASK_START]
    target_available = 1.0 - target_mask
    assert target_mask == pytest.approx(1.0)
    assert target_available == pytest.approx(0.0)
    assert prepared.local_history_tensor[200, 0, module._LOCAL_DELTA_START] == pytest.approx(1.0 / 72.0)
    assert prepared.local_history_tensor[201, 0, module._LOCAL_DELTA_START] == pytest.approx(0.0)
    assert prepared.local_history_tensor[201, 0, module._LOCAL_DELTA_START + 1] == pytest.approx(1.0 / 72.0)


def test_wake_geometry_uses_raw_pairwise_values(tmp_path, monkeypatch) -> None:
    module = _load_module()
    prepared = _prepare_temp_dataset(module, tmp_path, monkeypatch)

    # Directed edge src=T01 (0) -> dst=T02 (1) in the toy cache.
    assert prepared.wake_geometry_tensor[1, 0, 0] == pytest.approx(100.0)
    assert prepared.wake_geometry_tensor[1, 0, 1] == pytest.approx(0.0)
    assert prepared.wake_geometry_tensor[1, 0, 2] == pytest.approx(100.0 / 90.0)

    # Directed edge src=T01 (0) -> dst=T03 (2) in the toy cache.
    assert prepared.wake_geometry_tensor[2, 0, 0] == pytest.approx(220.0)
    assert prepared.wake_geometry_tensor[2, 0, 1] == pytest.approx(30.0)
    assert prepared.wake_geometry_tensor[2, 0, 2] == pytest.approx(np.hypot(220.0, 30.0) / 90.0)

    # The normalized pairwise tensor keeps the same directed indexing but should not
    # equal the raw geometry values for these channels.
    assert prepared.pairwise_tensor[1, 0, 0] != pytest.approx(prepared.wake_geometry_tensor[1, 0, 0])
    assert prepared.pairwise_tensor[2, 0, 1] != pytest.approx(prepared.wake_geometry_tensor[2, 0, 1])
    assert prepared.pairwise_tensor[2, 0, 6] != pytest.approx(prepared.wake_geometry_tensor[2, 0, 2])


def test_future_inputs_exclude_real_observations(tmp_path, monkeypatch) -> None:
    module = _load_module()
    prepared = _prepare_temp_dataset(module, tmp_path, monkeypatch)
    target_index = int(prepared.val_rolling_windows.target_indices[0])
    future_slice = slice(target_index, target_index + prepared.forecast_steps)
    context_future = prepared.context_future_tensor[future_slice]
    future_met_targets = prepared.future_met_tensor[future_slice]
    future_farm_targets = prepared.future_farm_target_pu[future_slice]

    assert context_future.shape == (prepared.forecast_steps, len(_KNOWN_FUTURE_COLUMNS))
    assert future_met_targets.shape == (prepared.forecast_steps, module.DEFAULT_MET_SUMMARY_DIM)
    assert future_farm_targets.shape == (prepared.forecast_steps, 1)


def test_forward_outputs_shapes_and_bounded_forecast(tmp_path, monkeypatch) -> None:
    module = _load_module()
    torch_module = _require_torch(module)
    prepared = _prepare_temp_dataset(module, tmp_path, monkeypatch)
    model = module.build_model(**_tiny_model_kwargs(module, prepared))
    sample = module.PanelWindowDataset(prepared, prepared.val_rolling_windows)[0]
    local_history = torch_module.from_numpy(sample[0][None])
    context_history = torch_module.from_numpy(sample[1][None])
    context_future = torch_module.from_numpy(sample[2][None])

    with torch_module.no_grad():
        outputs = model(local_history, context_history, context_future)

    assert outputs.future_predictions.shape == (1, prepared.forecast_steps, prepared.node_count, 1)
    assert outputs.hist_prior_observations.shape == (
        1,
        prepared.history_steps,
        prepared.node_count,
        module.LOCAL_VALUE_COUNT,
    )
    assert outputs.hist_post_observations.shape == outputs.hist_prior_observations.shape
    assert outputs.future_farm_predictions.shape == (1, prepared.forecast_steps, 1)
    assert outputs.future_met_predictions.shape == (1, prepared.forecast_steps, module.DEFAULT_MET_SUMMARY_DIM)
    assert float(outputs.future_predictions.min().item()) >= 0.0
    assert float(outputs.future_predictions.max().item()) <= 1.05


def test_residual_persistence_zero_decoder_matches_persistence_anchor(tmp_path, monkeypatch) -> None:
    module = _load_module()
    torch_module = _require_torch(module)
    prepared = _prepare_temp_dataset(
        module,
        tmp_path,
        monkeypatch,
        variant_name=module.RESIDUAL_PERSISTENCE_MODEL_VARIANT,
    )
    model = module.build_model(
        **_tiny_model_kwargs(module, prepared, variant_name=module.RESIDUAL_PERSISTENCE_MODEL_VARIANT)
    )
    for parameter in model.forecast_decoder.parameters():
        parameter.data.zero_()
    sample = module.PanelWindowDataset(prepared, prepared.val_rolling_windows)[0]
    local_history = torch_module.from_numpy(sample[0][None])
    context_history = torch_module.from_numpy(sample[1][None])
    context_future = torch_module.from_numpy(sample[2][None])

    with torch_module.no_grad():
        persistence_anchor = model._persistence_anchor(local_history)
        outputs = model(local_history, context_history, context_future)

    expected = persistence_anchor[:, None, :, None].expand(-1, prepared.forecast_steps, -1, -1)
    assert torch_module.allclose(outputs.future_predictions, expected)


def test_persistence_anchor_backtracks_to_last_available_history_value(tmp_path, monkeypatch) -> None:
    module = _load_module()
    torch_module = _require_torch(module)
    prepared = _prepare_temp_dataset(
        module,
        tmp_path,
        monkeypatch,
        variant_name=module.RESIDUAL_PERSISTENCE_MODEL_VARIANT,
    )
    model = module.build_model(
        **_tiny_model_kwargs(module, prepared, variant_name=module.RESIDUAL_PERSISTENCE_MODEL_VARIANT)
    )
    sample = module.PanelWindowDataset(prepared, prepared.val_rolling_windows)[0]
    local_history = torch_module.from_numpy(sample[0][None])
    local_history[:, -1, 0, module._LOCAL_MASK_START] = 1.0
    local_history[:, -2, 0, module._LOCAL_MASK_START] = 0.0
    local_history[:, -2, 0, module._LOCAL_VALUE_START] = 0.37

    anchor = model._persistence_anchor(local_history)

    assert float(anchor[0, 0].item()) == pytest.approx(0.37)


def test_persistence_anchor_uses_train_fallback_when_history_missing(tmp_path, monkeypatch) -> None:
    module = _load_module()
    torch_module = _require_torch(module)
    prepared = _prepare_temp_dataset(
        module,
        tmp_path,
        monkeypatch,
        variant_name=module.RESIDUAL_PERSISTENCE_MODEL_VARIANT,
    )
    model = module.build_model(
        **_tiny_model_kwargs(module, prepared, variant_name=module.RESIDUAL_PERSISTENCE_MODEL_VARIANT)
    )
    sample = module.PanelWindowDataset(prepared, prepared.val_rolling_windows)[0]
    local_history = torch_module.from_numpy(sample[0][None])
    local_history[:, :, 0, module._LOCAL_MASK_START] = 1.0

    anchor = model._persistence_anchor(local_history)

    assert float(anchor[0, 0].item()) == pytest.approx(float(prepared.persistence_train_fallback_pu[0]))


def test_canonical_zero_decoder_keeps_absolute_head_behavior(tmp_path, monkeypatch) -> None:
    module = _load_module()
    torch_module = _require_torch(module)
    prepared = _prepare_temp_dataset(module, tmp_path, monkeypatch)
    model = module.build_model(**_tiny_model_kwargs(module, prepared))
    for parameter in model.forecast_decoder.parameters():
        parameter.data.zero_()
    sample = module.PanelWindowDataset(prepared, prepared.val_rolling_windows)[0]
    local_history = torch_module.from_numpy(sample[0][None])
    context_history = torch_module.from_numpy(sample[1][None])
    context_future = torch_module.from_numpy(sample[2][None])

    with torch_module.no_grad():
        outputs = model(local_history, context_history, context_future)

    expected = (1.0 + module.DEFAULT_BOUNDED_OUTPUT_EPSILON) * 0.5
    assert torch_module.allclose(outputs.future_predictions, torch_module.full_like(outputs.future_predictions, expected))


def test_wake_off_zeros_dynamic_wake_but_keeps_graph_path(tmp_path, monkeypatch) -> None:
    module = _load_module()
    torch_module = _require_torch(module)
    prepared = _prepare_temp_dataset(module, tmp_path, monkeypatch, variant_name=module.WAKE_OFF_MODEL_VARIANT)
    model = module.build_model(**_tiny_model_kwargs(module, prepared, variant_name=module.WAKE_OFF_MODEL_VARIANT))
    met = torch_module.randn(2, module.DEFAULT_MET_SUMMARY_DIM)

    dynamic = model._wake(met)

    assert model.uses_graph is True
    assert model.uses_wake_dynamic is False
    assert torch_module.count_nonzero(dynamic) == 0
    assert float(model.pairwise_features.abs().sum().item()) > 0.0


def test_graph_off_aggregate_returns_zero_message(tmp_path, monkeypatch) -> None:
    module = _load_module()
    torch_module = _require_torch(module)
    prepared = _prepare_temp_dataset(module, tmp_path, monkeypatch, variant_name=module.GRAPH_OFF_MODEL_VARIANT)
    model = module.build_model(**_tiny_model_kwargs(module, prepared, variant_name=module.GRAPH_OFF_MODEL_VARIANT))
    static = model._static(batch_size=1)
    source_summary = torch_module.randn(1, prepared.node_count, 4)
    g = torch_module.randn(1, 8)
    met = torch_module.randn(1, module.DEFAULT_MET_SUMMARY_DIM)
    calendar = torch_module.randn(1, prepared.context_future_channels)

    edge_message = model._aggregate(source_summary, g, met, calendar, model.update_edge, model.update_gate, static=static)

    assert model.uses_graph is False
    assert model.uses_wake_dynamic is False
    assert edge_message.shape == (1, prepared.node_count, 4)
    assert torch_module.count_nonzero(edge_message) == 0


def test_all_missing_node_keeps_prior_in_update(tmp_path, monkeypatch) -> None:
    module = _load_module()
    torch_module = _require_torch(module)
    prepared = _prepare_temp_dataset(module, tmp_path, monkeypatch)
    model = module.build_model(**_tiny_model_kwargs(module, prepared))
    local_observations = torch_module.from_numpy(prepared.local_history_tensor[144][None].copy())
    local_observations[:, 0, module._LOCAL_MASK_START : module._LOCAL_MASK_START + module.LOCAL_MASK_COUNT] = 1.0
    context = torch_module.from_numpy(prepared.context_history_tensor[144][None].copy())
    z = torch_module.randn(1, prepared.node_count, 4)
    h = torch_module.randn(1, prepared.node_count, 6)
    g = torch_module.randn(1, 8)

    z_post, h_post, _g_post, _prior_obs, _post_obs, _innovation = model._correct(z, h, g, local_observations, context)

    assert torch_module.allclose(z_post[:, 0, :], z[:, 0, :])
    assert torch_module.allclose(h_post[:, 0, :], h[:, 0, :])
    available = 1.0 - local_observations[0, :, module._LOCAL_MASK_START : module._LOCAL_MASK_START + module.LOCAL_MASK_COUNT]
    observed_nodes = torch_module.nonzero(available.sum(dim=-1) > 0, as_tuple=False).flatten().tolist()
    observed_nodes = [index for index in observed_nodes if index != 0]
    assert observed_nodes, "toy batch should still contain at least one observed node after masking node 0"
    assert any(not torch_module.allclose(z_post[:, index, :], z[:, index, :]) for index in observed_nodes)


def test_selection_metric_identity_isolation(tmp_path) -> None:
    module = _load_module()
    paths = module._resume_paths_for_output(output_path=tmp_path / "latest.csv", work_root=tmp_path / ".work")

    rmse_identity = module._job_identity(
        dataset_id="kelmarsh",
        model_variant=module.MODEL_VARIANT,
        feature_protocol_id=module.FEATURE_PROTOCOL_ID,
        selection_metric="val_rmse_pu",
    )
    mae_identity = module._job_identity(
        dataset_id="kelmarsh",
        model_variant=module.MODEL_VARIANT,
        feature_protocol_id=module.FEATURE_PROTOCOL_ID,
        selection_metric="val_mae_pu",
    )

    assert rmse_identity != mae_identity
    assert module._job_checkpoint_path(
        paths,
        dataset_id="kelmarsh",
        model_variant=module.MODEL_VARIANT,
        selection_metric="val_rmse_pu",
    ).name == f"kelmarsh__{module.MODEL_VARIANT}__rmse_pu.pt"
    assert module._job_checkpoint_path(
        paths,
        dataset_id="kelmarsh",
        model_variant=module.MODEL_VARIANT,
        selection_metric="val_mae_pu",
    ).name == f"kelmarsh__{module.MODEL_VARIANT}__mae_pu.pt"
    assert module._tensorboard_job_log_dir(
        tmp_path / "tb",
        dataset_id="kelmarsh",
        model_variant=module.MODEL_VARIANT,
        selection_metric="val_rmse_pu",
    ) != module._tensorboard_job_log_dir(
        tmp_path / "tb",
        dataset_id="kelmarsh",
        model_variant=module.MODEL_VARIANT,
        selection_metric="val_mae_pu",
    )


def test_filter_state_prefix_equivalence_with_multi_step_reveal(tmp_path, monkeypatch) -> None:
    module = _load_module()
    torch_module = _require_torch(module)
    prepared = _prepare_temp_dataset(module, tmp_path, monkeypatch, max_eval_origins=4)
    model = module.build_model(**_tiny_model_kwargs(module, prepared))
    module.initialize_model_parameters(model)
    model.eval()
    start_target_index = int(prepared.val_rolling_windows.target_indices[0])
    end_target_index = start_target_index + 3
    assert end_target_index - start_target_index > 1

    start_local_history, start_context_history, _start_future, _start_targets, _start_valid = module._window_batch_tensors(
        prepared,
        target_index=start_target_index,
        torch_module=torch_module,
        device="cpu",
    )
    end_local_history, end_context_history, _end_future, _end_targets, _end_valid = module._window_batch_tensors(
        prepared,
        target_index=end_target_index,
        torch_module=torch_module,
        device="cpu",
    )
    revealed_local, revealed_context = module._revealed_observation_tensors(
        prepared,
        start_target_index=start_target_index,
        end_target_index=end_target_index,
        torch_module=torch_module,
        device="cpu",
    )
    assert revealed_local.shape[1] == end_target_index - start_target_index
    assert revealed_local.shape[1] > 1

    with torch_module.no_grad():
        cold_filter_state = model.filter_history(end_local_history, end_context_history)
        cold_cache = model.build_observation_cache(end_local_history)
        warm_filter_state = model.filter_history(start_local_history, start_context_history)
        warm_cache = model.build_observation_cache(start_local_history)
        advanced_state = model.advance_with_observations(
            warm_filter_state,
            warm_cache,
            revealed_local,
            revealed_context,
        )

    assert torch_module.allclose(advanced_state.filter_state.z, cold_filter_state.z, rtol=1e-5, atol=1e-6)
    assert torch_module.allclose(advanced_state.filter_state.h, cold_filter_state.h, rtol=1e-5, atol=1e-6)
    assert torch_module.allclose(advanced_state.filter_state.g, cold_filter_state.g, rtol=1e-5, atol=1e-6)
    assert torch_module.allclose(
        advanced_state.observation_cache.persistence_anchor,
        cold_cache.persistence_anchor,
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "variant_name",
    (
        "world_model_state_space_v1_farm_sync",
        "world_model_state_space_v1_residual_persistence_farm_sync",
        "world_model_state_space_v1_graph_off_farm_sync",
        "world_model_state_space_v1_no_met_aux_farm_sync",
    ),
)
def test_execute_training_job_smoke_writes_history(tmp_path, monkeypatch, variant_name: str) -> None:
    module = _load_module()
    _require_torch(module)
    prepared = _prepare_temp_dataset(
        module,
        tmp_path,
        monkeypatch,
        variant_name=variant_name,
        max_train_origins=2,
        max_eval_origins=1,
    )
    history_path = tmp_path / "training_history.csv"

    rows = module.execute_training_job(
        prepared,
        device="cpu",
        seed=123,
        batch_size=2,
        eval_batch_size=1,
        learning_rate=1e-3,
        max_epochs=1,
        early_stopping_patience=1,
        z_dim=4,
        h_dim=6,
        global_state_dim=8,
        obs_encoding_dim=8,
        innovation_dim=4,
        source_summary_dim=4,
        edge_message_dim=4,
        edge_hidden_dim=8,
        tau_embed_dim=4,
        met_summary_dim=module.DEFAULT_MET_SUMMARY_DIM,
        turbine_embed_dim=3,
        dropout=0.0,
        grad_clip_norm=1.0,
        training_history_path=history_path,
    )

    assert history_path.exists()
    history = pl.read_csv(history_path)
    assert history["epoch"].to_list() == [1]
    assert history["train_loss_mean"].null_count() == 0
    assert history["train_forecast_loss_mean"].null_count() == 0
    assert history["train_hist_recon_loss_mean"].null_count() == 0
    assert history["train_farm_loss_mean"].null_count() == 0
    assert history["val_rmse_pu_leads_13_24_mean"].null_count() == 0
    assert history["val_rmse_pu_leads_25_36_mean"].null_count() == 0
    assert history["selection_metric"].to_list() == [module.DEFAULT_SELECTION_METRIC]
    assert len(rows) == 148
    assert {row["split_name"] for row in rows} == {"val", "test"}
    assert rows[0]["model_variant"] == variant_name
    assert rows[0]["z_dim"] == 4
    assert rows[0]["h_dim"] == 6
    assert rows[0]["amp_enabled"] is False
    if variant_name == module.GRAPH_OFF_MODEL_VARIANT:
        assert rows[0]["wake_lambda_x"] == 0.0
        assert rows[0]["wake_lambda_y"] == 0.0
        assert rows[0]["wake_kappa"] == 0.0
    if variant_name == module.NO_MET_AUX_MODEL_VARIANT:
        assert rows[0]["met_loss_weight"] == 0.0


def test_execute_training_job_logs_tensorboard_with_fake_writer(tmp_path, monkeypatch) -> None:
    module = _load_module()
    _require_torch(module)
    _FakeSummaryWriter.instances.clear()
    monkeypatch.setattr(module, "SummaryWriter", _FakeSummaryWriter)
    prepared = _prepare_temp_dataset(
        module,
        tmp_path,
        monkeypatch,
        max_train_origins=2,
        max_eval_origins=1,
    )
    history_path = tmp_path / "training_history.csv"
    log_dir = tmp_path / "tensorboard" / "state_space"

    module.execute_training_job(
        prepared,
        device="cpu",
        seed=123,
        batch_size=2,
        eval_batch_size=1,
        learning_rate=1e-3,
        max_epochs=1,
        early_stopping_patience=1,
        z_dim=4,
        h_dim=6,
        global_state_dim=8,
        obs_encoding_dim=8,
        innovation_dim=4,
        source_summary_dim=4,
        edge_message_dim=4,
        edge_hidden_dim=8,
        tau_embed_dim=4,
        met_summary_dim=module.DEFAULT_MET_SUMMARY_DIM,
        turbine_embed_dim=3,
        dropout=0.0,
        grad_clip_norm=1.0,
        training_history_path=history_path,
        tensorboard_log_dir=log_dir,
    )

    assert len(_FakeSummaryWriter.instances) == 1
    writer = _FakeSummaryWriter.instances[0]
    assert Path(writer.log_dir) == log_dir.resolve()
    assert writer.closed is True
    scalar_tags = {tag for tag, _value, _step in writer.scalars}
    text_tags = {tag for tag, _text, _step in writer.texts}
    assert "train/loss_mean" in scalar_tags
    assert "train/forecast_loss_mean" in scalar_tags
    assert "val/overall/rmse_pu" in scalar_tags
    assert "val/horizon_group/rmse_pu/leads_13_24_mean" in scalar_tags
    assert "final/test/rolling_origin_no_refit/overall/rmse_pu" in scalar_tags
    assert "final/test/rolling_origin_no_refit/horizon_group/rmse_pu/leads_25_36_mean" in scalar_tags
    assert "run/config_json" in text_tags
    assert "final/summary_json" in text_tags
    assert any('"selection_metric": "val_rmse_pu"' in text for _tag, text, _step in writer.texts)


def test_read_training_history_backfills_selection_metric_and_horizon_columns(tmp_path) -> None:
    module = _load_module()
    history_path = tmp_path / "legacy.training_history.csv"
    legacy_columns = [
        column
        for column in module._TRAINING_HISTORY_COLUMNS
        if column not in {"selection_metric", "val_rmse_pu_leads_13_24_mean", "val_rmse_pu_leads_25_36_mean"}
    ]
    pl.DataFrame(
        {
            column: [1 if column == "epoch" else "x" if column in {"dataset_id", "model_id", "model_variant", "feature_protocol_id", "task_id", "window_protocol", "device"} else 0]
            for column in legacy_columns
        }
    ).write_csv(history_path)

    history = module._read_training_history(history_path)

    assert history.columns == module._TRAINING_HISTORY_COLUMNS
    assert history["selection_metric"].to_list() == [module.DEFAULT_SELECTION_METRIC]
    assert history["val_rmse_pu_leads_13_24_mean"].null_count() == 1
    assert history["val_rmse_pu_leads_25_36_mean"].null_count() == 1


def test_run_experiment_writes_results_and_hashed_resume_state(tmp_path, monkeypatch) -> None:
    module = _load_module()
    prepared = _prepare_temp_dataset(module, tmp_path, monkeypatch, max_train_origins=2, max_eval_origins=1)
    output_path = tmp_path / "published" / "latest.csv"
    work_root = tmp_path / ".work"

    def _dataset_loader(*_args, **_kwargs):
        return prepared

    def _job_runner(prepared_dataset, **kwargs):
        profile = module.resolve_hyperparameter_profile(
            prepared_dataset.model_variant,
            dataset_id=prepared_dataset.dataset_id,
            batch_size=kwargs["batch_size"],
            learning_rate=kwargs["learning_rate"],
            max_epochs=kwargs["max_epochs"],
            early_stopping_patience=kwargs["early_stopping_patience"],
            z_dim=kwargs["z_dim"],
            h_dim=kwargs["h_dim"],
            global_state_dim=kwargs["global_state_dim"],
            obs_encoding_dim=kwargs["obs_encoding_dim"],
            innovation_dim=kwargs["innovation_dim"],
            source_summary_dim=kwargs["source_summary_dim"],
            edge_message_dim=kwargs["edge_message_dim"],
            edge_hidden_dim=kwargs["edge_hidden_dim"],
            tau_embed_dim=kwargs["tau_embed_dim"],
            met_summary_dim=kwargs["met_summary_dim"],
            turbine_embed_dim=kwargs["turbine_embed_dim"],
            dropout=kwargs["dropout"],
            grad_clip_norm=kwargs["grad_clip_norm"],
            hist_recon_loss_weight=kwargs["hist_recon_loss_weight"],
            farm_loss_weight=kwargs["farm_loss_weight"],
            met_loss_weight=kwargs["met_loss_weight"],
            innovation_loss_weight=kwargs["innovation_loss_weight"],
            weight_decay=kwargs["weight_decay"],
            wake_lambda_x=kwargs["wake_lambda_x"],
            wake_lambda_y=kwargs["wake_lambda_y"],
            wake_kappa=kwargs["wake_kappa"],
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
            best_epoch=1,
            epochs_ran=1,
            best_val_rmse_pu=0.2,
            best_val_mae_pu=0.1,
            selection_metric=kwargs["selection_metric"],
            device="cpu",
            amp_enabled=False,
            model=None,
        )
        return module.build_result_rows(
            prepared_dataset,
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
        selection_metric="val_mae_pu",
        batch_size=2,
        learning_rate=1e-3,
        z_dim=4,
        h_dim=6,
        global_state_dim=8,
        obs_encoding_dim=8,
        innovation_dim=4,
        source_summary_dim=4,
        edge_message_dim=4,
        edge_hidden_dim=8,
        tau_embed_dim=4,
        met_summary_dim=module.DEFAULT_MET_SUMMARY_DIM,
        turbine_embed_dim=3,
        dropout=0.0,
        work_root=work_root,
        dataset_loader=_dataset_loader,
        job_runner=_job_runner,
    )

    assert output_path.exists()
    assert module.training_history_output_path(output_path).exists()
    assert results.height == 148
    assert set(results["metric_scope"].unique().to_list()) == {"overall", "horizon"}
    paths = module._resume_paths_for_output(output_path=output_path, work_root=work_root)
    expected_slot = hashlib.sha256(str(output_path.resolve()).encode("utf-8")).hexdigest()
    assert paths.slot_dir.name == expected_slot
    assert paths.partial_results_path.exists()
    assert paths.checkpoints_dir.exists()
    assert module._job_checkpoint_path(
        paths,
        dataset_id=prepared.dataset_id,
        model_variant=prepared.model_variant,
        selection_metric="val_mae_pu",
    ).name == f"{prepared.dataset_id}__{prepared.model_variant}__mae_pu.pt"
    state_payload = json.loads(paths.state_path.read_text(encoding="utf-8"))
    assert state_payload["status"] == "complete"
    assert state_payload["effective_config"]["selection_metric"] == "val_mae_pu"


def test_saved_checkpoint_single_origin_carry_over_matches_cold_start(tmp_path, monkeypatch) -> None:
    module = _load_module()
    _require_torch(module)
    prepared = _prepare_temp_dataset(module, tmp_path, monkeypatch, max_eval_origins=1)
    model = module.build_model(**_tiny_model_kwargs(module, prepared))
    module.initialize_model_parameters(model)
    profile = _tiny_profile(module, prepared)
    checkpoint_path = tmp_path / "best.pt"
    module.save_best_checkpoint(
        checkpoint_path,
        prepared_dataset=prepared,
        training_outcome=module.TrainingOutcome(
            best_epoch=1,
            epochs_ran=1,
            best_val_rmse_pu=0.2,
            best_val_mae_pu=0.1,
            selection_metric="val_rmse_pu",
            device="cpu",
            amp_enabled=False,
            model=model,
        ),
        profile=profile,
        seed=123,
        runtime_seconds=0.5,
    )
    loaded_checkpoint = module.load_best_checkpoint(
        checkpoint_path,
        prepared_dataset=prepared,
        device="cpu",
    )
    single_window = _take_first_windows(module, prepared.val_rolling_windows, 1)

    rolling_metrics, rolling_diagnostics, rolling_summary = module.evaluate_saved_checkpoint_windows(
        loaded_checkpoint,
        prepared,
        windows=single_window,
        split_name="val",
        eval_protocol=module.ROLLING_EVAL_PROTOCOL,
    )
    carry_metrics, carry_diagnostics, carry_summary = module.evaluate_saved_checkpoint_windows(
        loaded_checkpoint,
        prepared,
        windows=single_window,
        split_name="val",
        eval_protocol=module.CARRY_OVER_EVAL_PROTOCOL,
    )

    assert carry_metrics.mae_pu == pytest.approx(rolling_metrics.mae_pu)
    assert carry_metrics.rmse_pu == pytest.approx(rolling_metrics.rmse_pu)
    assert carry_metrics.horizon_mae_pu.tolist() == pytest.approx(rolling_metrics.horizon_mae_pu.tolist())
    assert carry_metrics.horizon_rmse_pu.tolist() == pytest.approx(rolling_metrics.horizon_rmse_pu.tolist())
    assert rolling_diagnostics.drop("eval_protocol").to_dicts() == carry_diagnostics.drop("eval_protocol").to_dicts()
    assert rolling_summary.drop("eval_protocol").to_dicts() == carry_summary.drop("eval_protocol").to_dicts()


@pytest.mark.parametrize(
    ("variant_name", "expect_residual_metric"),
    (
        ("world_model_state_space_v1_farm_sync", False),
        ("world_model_state_space_v1_residual_persistence_farm_sync", True),
    ),
)
def test_run_saved_checkpoint_scratch_evaluation_writes_diagnostics(
    tmp_path,
    monkeypatch,
    variant_name: str,
    expect_residual_metric: bool,
) -> None:
    module = _load_module()
    _require_torch(module)
    prepared = _prepare_temp_dataset(
        module,
        tmp_path,
        monkeypatch,
        variant_name=variant_name,
        max_eval_origins=2,
    )
    model = module.build_model(**_tiny_model_kwargs(module, prepared, variant_name=variant_name))
    module.initialize_model_parameters(model)
    profile = _tiny_profile(module, prepared, variant_name=variant_name)
    checkpoint_path = tmp_path / f"{variant_name}.pt"
    module.save_best_checkpoint(
        checkpoint_path,
        prepared_dataset=prepared,
        training_outcome=module.TrainingOutcome(
            best_epoch=1,
            epochs_ran=1,
            best_val_rmse_pu=0.2,
            best_val_mae_pu=0.1,
            selection_metric="val_rmse_pu",
            device="cpu",
            amp_enabled=False,
            model=model,
        ),
        profile=profile,
        seed=123,
        runtime_seconds=0.5,
    )
    output_dir = tmp_path / "scratch"

    results, diagnostics, summaries = module.run_saved_checkpoint_scratch_evaluation(
        checkpoint_path=checkpoint_path,
        prepared_dataset=prepared,
        output_dir=output_dir,
        device="cpu",
        include_carry_over=True,
    )

    config_name = module._best_checkpoint_config_name(
        dataset_id=prepared.dataset_id,
        model_variant=prepared.model_variant,
        selection_metric="val_rmse_pu",
        seed=123,
    )
    assert (output_dir / f"{config_name}.csv").exists()
    assert (output_dir / f"{config_name}.diagnostics.csv").exists()
    assert (output_dir / f"{config_name}.summary.csv").exists()
    assert results.height == 148
    assert set(results["eval_protocol"].unique().to_list()) == {
        module.ROLLING_EVAL_PROTOCOL,
        module.CARRY_OVER_EVAL_PROTOCOL,
    }
    assert set(summaries["bucket_name"].unique().to_list()) == {"overall", "leads_1_12", "leads_13_24", "leads_25_36"}
    clamp_values = diagnostics["clamp_hit_rate"].to_numpy()
    assert np.all((clamp_values >= 0.0) & (clamp_values <= 1.0))
    residual_values = diagnostics["mean_abs_residual"].to_numpy()
    if expect_residual_metric:
        assert np.isfinite(residual_values).any()
    else:
        assert not np.isfinite(residual_values).any()


def test_main_eval_only_writes_scratch_outputs(tmp_path, monkeypatch, capsys) -> None:
    module = _load_module()
    prepared = _prepare_temp_dataset(module, tmp_path, monkeypatch)
    expected_results = pl.DataFrame({"dataset_id": [prepared.dataset_id], "split_name": ["val"], "eval_protocol": [module.ROLLING_EVAL_PROTOCOL]})
    captured: dict[str, object] = {}

    def _fake_prepare_dataset(*args, **kwargs):
        del args, kwargs
        return prepared

    def _fake_load_best_checkpoint(checkpoint_path, *, prepared_dataset, device):
        captured["load_best_checkpoint"] = {
            "checkpoint_path": Path(checkpoint_path),
            "prepared_dataset": prepared_dataset,
            "device": device,
        }
        return module.LoadedBestCheckpoint(
            model=None,
            profile=_tiny_profile(module, prepared),
            job_identity=module._job_identity_for_prepared_dataset(prepared, selection_metric="val_rmse_pu"),
            best_epoch=1,
            epochs_ran=1,
            best_val_rmse_pu=0.2,
            best_val_mae_pu=0.1,
            selection_metric="val_rmse_pu",
            seed=123,
            runtime_seconds=0.5,
            device="cpu",
            amp_enabled=False,
        )

    def _fake_run_saved_checkpoint_scratch_evaluation(**kwargs):
        captured["scratch_eval"] = kwargs
        return expected_results, pl.DataFrame({"x": [1]}), pl.DataFrame({"y": [1]})

    monkeypatch.setattr(module, "prepare_dataset", _fake_prepare_dataset)
    monkeypatch.setattr(module, "load_best_checkpoint", _fake_load_best_checkpoint)
    monkeypatch.setattr(module, "run_saved_checkpoint_scratch_evaluation", _fake_run_saved_checkpoint_scratch_evaluation)

    scratch_output_dir = tmp_path / "scratch"
    checkpoint_path = tmp_path / "checkpoint.pt"
    assert module.main(
        [
            "--dataset",
            prepared.dataset_id,
            "--variant",
            prepared.model_variant,
            "--load-best-checkpoint",
            str(checkpoint_path),
            "--scratch-output-dir",
            str(scratch_output_dir),
            "--no-record-run",
        ]
    ) == 0

    assert captured["load_best_checkpoint"]["checkpoint_path"] == checkpoint_path
    assert captured["scratch_eval"]["checkpoint_path"] == checkpoint_path
    assert captured["scratch_eval"]["prepared_dataset"] is prepared
    assert captured["scratch_eval"]["output_dir"] == scratch_output_dir
    assert captured["scratch_eval"]["include_carry_over"] is True
    assert str(scratch_output_dir / f"{prepared.dataset_id}__{prepared.model_variant}__rmse_pu__seed123.csv") in capsys.readouterr().out



def test_search_screen_one_uses_training_device_for_eval_loaders(monkeypatch, tmp_path) -> None:
    module = _load_module()
    search_module = _load_search_module()
    prepared = _prepare_temp_dataset(module, tmp_path, monkeypatch, max_train_origins=2, max_eval_origins=1)
    loader_devices: list[str] = []

    def _fake_train_model(*args, **kwargs):
        del args, kwargs
        return SimpleNamespace(
            model=object(),
            device="cuda",
            amp_enabled=True,
            best_epoch=1,
            epochs_ran=1,
            best_val_rmse_pu=0.2,
        )

    def _fake_build_dataloader(
        prepared_dataset,
        *,
        windows,
        batch_size,
        device,
        shuffle,
        seed,
    ):
        del prepared_dataset, windows, batch_size, shuffle, seed
        loader_devices.append(device)
        return object()

    def _fake_evaluate_model(model, loader, *, device, rated_power_kw, forecast_steps, amp_enabled, progress_label):
        del model, loader, rated_power_kw, forecast_steps, amp_enabled, progress_label
        assert device == "cuda"
        return module.EvaluationMetrics(
            window_count=1,
            prediction_count=prepared.forecast_steps * prepared.node_count,
            mae_kw=1.0,
            rmse_kw=2.0,
            mae_pu=0.1,
            rmse_pu=0.2,
            horizon_window_count=np.ones((prepared.forecast_steps,), dtype=np.int64),
            horizon_prediction_count=np.ones((prepared.forecast_steps,), dtype=np.int64),
            horizon_mae_kw=np.ones((prepared.forecast_steps,), dtype=np.float64),
            horizon_rmse_kw=np.ones((prepared.forecast_steps,), dtype=np.float64),
            horizon_mae_pu=np.ones((prepared.forecast_steps,), dtype=np.float64) * 0.1,
            horizon_rmse_pu=np.linspace(0.1, 0.36, prepared.forecast_steps, dtype=np.float64),
        )

    monkeypatch.setattr(search_module.state_space, "train_model", _fake_train_model)
    monkeypatch.setattr(search_module.state_space, "_build_dataloader", _fake_build_dataloader)
    monkeypatch.setattr(search_module.state_space, "evaluate_model", _fake_evaluate_model)

    row = search_module._screen_one(
        prepared,
        config=search_module.COMMON_SCREEN_CONFIGS[0],
        device="auto",
        seed=42,
        max_epochs=1,
        patience=1,
    )

    assert loader_devices == ["cuda", "cuda"]
    assert row["device"] == "cuda"
    assert row["val_rmse_pu_leads_13_24_mean"] > 0.0
    assert row["val_rmse_pu_leads_25_36_mean"] > row["val_rmse_pu_leads_13_24_mean"]


def test_search_harness_writes_selected_defaults(tmp_path, monkeypatch) -> None:
    module = _load_module()
    search_module = _load_search_module()
    prepared = _prepare_temp_dataset(module, tmp_path, monkeypatch, max_train_origins=2, max_eval_origins=1)
    selected_configs = tuple(search_module.COMMON_SCREEN_CONFIGS[:2])
    metrics_by_config = {
        selected_configs[0].name: {
            "best_val_rmse_pu": 0.18,
            "val_non_overlap_rmse_pu": 0.19,
            "val_rmse_pu_leads_13_24_mean": 0.21,
            "val_rmse_pu_leads_25_36_mean": 0.24,
            "runtime_seconds": 9.0,
            "test_rolling_rmse_pu": 0.2,
        },
        selected_configs[1].name: {
            "best_val_rmse_pu": 0.22,
            "val_non_overlap_rmse_pu": 0.23,
            "val_rmse_pu_leads_13_24_mean": 0.25,
            "val_rmse_pu_leads_25_36_mean": 0.29,
            "runtime_seconds": 5.0,
            "test_rolling_rmse_pu": 0.24,
        },
    }

    def _fake_prepare_dataset(dataset_id: str):
        assert dataset_id == prepared.dataset_id
        return prepared

    def _screen_row(prepared_dataset, *, config, device, seed, max_epochs, patience):
        del device, seed, max_epochs, patience
        values = metrics_by_config[config.name]
        return {
            "dataset_id": prepared_dataset.dataset_id,
            "model_variant": prepared_dataset.model_variant,
            "feature_protocol_id": prepared_dataset.feature_protocol_id,
            "stage": "screen",
            "config_name": config.name,
            "farm_loss_weight": config.farm_loss_weight,
            "best_epoch": 1,
            "epochs_ran": 1,
            "best_val_rmse_pu": values["best_val_rmse_pu"],
            "val_rolling_window_count": len(prepared_dataset.val_rolling_windows),
            "val_non_overlap_window_count": len(prepared_dataset.val_non_overlap_windows),
            "val_rolling_rmse_pu": values["best_val_rmse_pu"],
            "val_rolling_mae_pu": values["best_val_rmse_pu"] / 2.0,
            "val_non_overlap_rmse_pu": values["val_non_overlap_rmse_pu"],
            "val_non_overlap_mae_pu": values["val_non_overlap_rmse_pu"] / 2.0,
            "val_rmse_pu_leads_13_24_mean": values["val_rmse_pu_leads_13_24_mean"],
            "val_rmse_pu_leads_25_36_mean": values["val_rmse_pu_leads_25_36_mean"],
            "train_window_count": len(prepared_dataset.train_windows),
            "runtime_seconds": values["runtime_seconds"],
            "device": "cpu",
            "seed": 42,
        }

    def _final_row(prepared_dataset, *, config, device, seed, max_epochs, patience):
        del device, seed, max_epochs, patience
        values = metrics_by_config[config.name]
        summary = {
            "dataset_id": prepared_dataset.dataset_id,
            "model_variant": prepared_dataset.model_variant,
            "feature_protocol_id": prepared_dataset.feature_protocol_id,
            "stage": "final",
            "config_name": config.name,
            "farm_loss_weight": config.farm_loss_weight,
            "best_epoch": 1,
            "epochs_ran": 1,
            "best_val_rmse_pu": values["best_val_rmse_pu"],
            "train_window_count": len(prepared_dataset.train_windows),
            "val_rolling_window_count": len(prepared_dataset.val_rolling_windows),
            "val_non_overlap_window_count": len(prepared_dataset.val_non_overlap_windows),
            "test_rolling_window_count": len(prepared_dataset.test_rolling_windows),
            "test_non_overlap_window_count": len(prepared_dataset.test_non_overlap_windows),
            "val_rolling_rmse_pu": values["best_val_rmse_pu"],
            "val_rolling_mae_pu": values["best_val_rmse_pu"] / 2.0,
            "val_non_overlap_rmse_pu": values["val_non_overlap_rmse_pu"],
            "val_non_overlap_mae_pu": values["val_non_overlap_rmse_pu"] / 2.0,
            "val_rmse_pu_leads_13_24_mean": values["val_rmse_pu_leads_13_24_mean"],
            "val_rmse_pu_leads_25_36_mean": values["val_rmse_pu_leads_25_36_mean"],
            "test_rolling_rmse_pu": values["test_rolling_rmse_pu"],
            "test_rolling_mae_pu": values["test_rolling_rmse_pu"] / 2.0,
            "test_non_overlap_rmse_pu": values["test_rolling_rmse_pu"] + 0.01,
            "test_non_overlap_mae_pu": (values["test_rolling_rmse_pu"] + 0.01) / 2.0,
            "runtime_seconds": values["runtime_seconds"],
            "device": "cpu",
            "seed": 42,
        }
        detail_frame = pl.DataFrame(
            {
                "dataset_id": [prepared_dataset.dataset_id],
                "config_name": [config.name],
                "test_rolling_rmse_pu": [values["test_rolling_rmse_pu"]],
            }
        )
        return summary, detail_frame

    monkeypatch.setattr(search_module, "_prepare_dataset", _fake_prepare_dataset)
    monkeypatch.setattr(search_module, "_screen_one", _screen_row)
    monkeypatch.setattr(search_module, "_final_one", _final_row)

    output_dir = tmp_path / "search"
    screen_frame, final_frame = search_module.run_search(
        dataset_ids=(prepared.dataset_id,),
        config_names=tuple(config.name for config in selected_configs),
        device="cpu",
        seed=42,
        screen_epochs=1,
        screen_patience=1,
        full_epochs=1,
        full_patience=1,
        top_k=1,
        output_dir=output_dir,
    )

    assert screen_frame.height == 2
    assert final_frame.height == 1
    assert (output_dir / "screen_summary.csv").exists()
    assert (output_dir / "final_summary.csv").exists()
    assert (output_dir / "final_detailed_rows.csv").exists()
    assert (output_dir / "search_plan.json").exists()
    assert (output_dir / "selected_defaults.json").exists()
    assert final_frame["config_name"].to_list() == [selected_configs[0].name]
    assert final_frame["val_rmse_pu_leads_25_36_mean"].to_list() == [0.24]

    selected_defaults = json.loads((output_dir / "selected_defaults.json").read_text(encoding="utf-8"))
    assert selected_defaults["selection_rule"] == ["best_val_rmse_pu", "config_name"]
    chosen = selected_defaults["selected_defaults"][prepared.dataset_id][module.MODEL_VARIANT]
    assert chosen["config_name"] == selected_configs[0].name
    assert chosen["farm_loss_weight"] == selected_configs[0].farm_loss_weight

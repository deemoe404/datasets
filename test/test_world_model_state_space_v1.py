from __future__ import annotations

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


def _prepare_temp_dataset(
    module,
    tmp_path: Path,
    monkeypatch,
    *,
    dataset_id: str = "kelmarsh",
    max_train_origins: int = 4,
    max_eval_origins: int = 2,
):
    cache_root = tmp_path / "cache"
    _build_world_model_temp_cache(cache_root, dataset_id=dataset_id)
    _patch_bundle_loader(monkeypatch, module.rollout_base, cache_root, dataset_id=dataset_id)
    return module.prepare_dataset(
        dataset_id,
        cache_root=cache_root,
        max_train_origins=max_train_origins,
        max_eval_origins=max_eval_origins,
    )


def _tiny_model_kwargs(module, prepared) -> dict[str, object]:
    return {
        "node_count": prepared.node_count,
        "local_input_channels": prepared.local_input_channels,
        "context_history_channels": prepared.context_history_channels,
        "context_future_channels": prepared.context_future_channels,
        "static_tensor": prepared.static_tensor,
        "turbine_indices": prepared.turbine_indices,
        "pairwise_tensor": prepared.pairwise_tensor,
        "wake_geometry_tensor": prepared.wake_geometry_tensor,
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
        "wake_lambda_x": module.DEFAULT_WAKE_LAMBDA_X,
        "wake_lambda_y": module.DEFAULT_WAKE_LAMBDA_Y,
        "wake_kappa": module.DEFAULT_WAKE_KAPPA,
        "bounded_output_epsilon": module.DEFAULT_BOUNDED_OUTPUT_EPSILON,
    }


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


def test_non_kelmarsh_dataset_is_rejected() -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="only supports"):
        module.run_experiment(dataset_ids=("penmanshiel",), output_path=Path("/tmp/unused.csv"))


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
    assert not torch_module.allclose(z_post[:, 1, :], z[:, 1, :])


def test_execute_training_job_smoke_writes_history(tmp_path, monkeypatch) -> None:
    module = _load_module()
    _require_torch(module)
    prepared = _prepare_temp_dataset(
        module,
        tmp_path,
        monkeypatch,
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
    assert len(rows) == 148
    assert {row["split_name"] for row in rows} == {"val", "test"}
    assert rows[0]["model_variant"] == module.MODEL_VARIANT
    assert rows[0]["z_dim"] == 4
    assert rows[0]["h_dim"] == 6
    assert rows[0]["amp_enabled"] is False


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
    ).name == f"{prepared.dataset_id}__{prepared.model_variant}.pt"
    state_payload = json.loads(paths.state_path.read_text(encoding="utf-8"))
    assert state_payload["status"] == "complete"

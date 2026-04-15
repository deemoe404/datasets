from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from test.test_agcrn import _build_temp_cache, _require_torch


_FEATURE_PROTOCOL_ID = "world_model_v1"
_TARGET_HISTORY_MASK_COLUMNS = ("target_kw__mask",)
_LOCAL_OBSERVATION_VALUE_COLUMNS = (
    "Wind speed (m/s)",
    "wind_direction_sin",
    "wind_direction_cos",
    "yaw_error_sin",
    "yaw_error_cos",
    "pitch_mean",
    "Rotor speed (RPM)",
    "Generator RPM (RPM)",
    "Nacelle ambient temperature (°C)",
    "Nacelle temperature (°C)",
    "evt_any_active",
    "evt_active_count",
    "evt_total_overlap_seconds",
    "evt_stop_active",
    "evt_warning_active",
    "evt_informational_active",
)
_GLOBAL_OBSERVATION_VALUE_COLUMNS = (
    "farm_pmu__gms_current_a",
    "farm_pmu__gms_power_kw",
    "farm_pmu__gms_reactive_power_kvar",
    "farm_evt_any_active",
    "farm_evt_active_count",
    "farm_evt_total_overlap_seconds",
    "farm_evt_stop_active",
    "farm_evt_warning_active",
    "farm_evt_informational_active",
)
_PAST_COVARIATE_VALUE_COLUMNS = _LOCAL_OBSERVATION_VALUE_COLUMNS + _GLOBAL_OBSERVATION_VALUE_COLUMNS
_LOCAL_OBSERVATION_MASK_COLUMNS = tuple(f"{column}__mask" for column in _LOCAL_OBSERVATION_VALUE_COLUMNS)
_GLOBAL_OBSERVATION_MASK_COLUMNS = tuple(f"{column}__mask" for column in _GLOBAL_OBSERVATION_VALUE_COLUMNS)
_PAST_COVARIATE_MASK_COLUMNS = tuple(f"{column}__mask" for column in _PAST_COVARIATE_VALUE_COLUMNS)
_KNOWN_FUTURE_COLUMNS = (
    "calendar_hour_sin",
    "calendar_hour_cos",
    "calendar_weekday_sin",
    "calendar_weekday_cos",
    "calendar_month_sin",
    "calendar_month_cos",
    "calendar_is_weekend",
)
_PAIRWISE_COLUMNS = (
    "src_turbine_id",
    "dst_turbine_id",
    "src_turbine_index",
    "dst_turbine_index",
    "delta_x_m",
    "delta_y_m",
    "distance_m",
    "bearing_deg",
    "elevation_diff_m",
    "distance_in_rotor_diameters",
)


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "experiment"
        / "families"
        / "world_model_rollout_v1"
        / "world_model_rollout_v1.py"
    )
    spec = spec_from_file_location("world_model_rollout_v1", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _static_frame(*, dataset_id: str, missing_coordinates: bool = False) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "dataset": [dataset_id, dataset_id, dataset_id],
            "turbine_id": ["T01", "T02", "T03"],
            "turbine_index": [0, 1, 2],
            "latitude": [None, None, None] if missing_coordinates else [52.4, 52.401, 52.402],
            "longitude": [None, None, None] if missing_coordinates else [-0.94, -0.941, -0.942],
            "coord_x": [None, None, None],
            "coord_y": [None, None, None],
            "coord_kind": ["local", "local", "local"],
            "coord_crs": ["epsg:32630", "epsg:32630", "epsg:32630"],
            "elevation_m": [10.0, 12.0, 16.0],
            "rated_power_kw": [2050.0, 2050.0, 2050.0],
            "hub_height_m": [80.0, 80.0, 80.0],
            "rotor_diameter_m": [90.0, 90.0, 90.0],
        }
    )


def _pairwise_frame(*, dataset_id: str) -> pl.DataFrame:
    turbine_ids = ("T01", "T02", "T03")
    coordinates = {
        "T01": (0.0, 0.0),
        "T02": (100.0, 0.0),
        "T03": (220.0, 30.0),
    }
    elevations = {"T01": 10.0, "T02": 12.0, "T03": 16.0}
    rotor_diameter_m = 90.0
    rows: list[dict[str, object]] = []
    for src_index, src_turbine_id in enumerate(turbine_ids):
        src_x, src_y = coordinates[src_turbine_id]
        for dst_index, dst_turbine_id in enumerate(turbine_ids):
            if src_index == dst_index:
                continue
            dst_x, dst_y = coordinates[dst_turbine_id]
            delta_x = float(dst_x - src_x)
            delta_y = float(dst_y - src_y)
            distance = float(math.hypot(delta_x, delta_y))
            rows.append(
                {
                    "src_turbine_id": src_turbine_id,
                    "dst_turbine_id": dst_turbine_id,
                    "src_turbine_index": src_index,
                    "dst_turbine_index": dst_index,
                    "delta_x_m": delta_x,
                    "delta_y_m": delta_y,
                    "distance_m": distance,
                    "bearing_deg": float(math.degrees(math.atan2(delta_y, delta_x))),
                    "elevation_diff_m": float(elevations[dst_turbine_id] - elevations[src_turbine_id]),
                    "distance_in_rotor_diameters": distance / rotor_diameter_m,
                }
            )
    return pl.DataFrame(rows)


def _build_world_model_temp_cache(
    cache_root: Path,
    *,
    dataset_id: str = "kelmarsh",
    missing_coordinates: bool = False,
) -> None:
    _build_temp_cache(
        cache_root,
        dataset_id=dataset_id,
        feature_protocol_id=_FEATURE_PROTOCOL_ID,
        target_history_mask_columns=_TARGET_HISTORY_MASK_COLUMNS,
        past_covariate_columns=_PAST_COVARIATE_VALUE_COLUMNS,
    )
    task_dir = cache_root / dataset_id / "tasks" / "next_6h_from_24h" / _FEATURE_PROTOCOL_ID
    series = pl.read_parquet(task_dir / "series.parquet")
    timestamps = series["timestamp"].unique().sort().to_list()
    target_missing_ts = timestamps[200]
    covariate_missing_ts = timestamps[201]
    updates = [
        pl.when(pl.col("timestamp") == target_missing_ts)
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(pl.col("target_kw"))
        .alias("target_kw"),
        pl.when(pl.col("timestamp") == target_missing_ts)
        .then(pl.lit(1, dtype=pl.Int8))
        .otherwise(pl.lit(0, dtype=pl.Int8))
        .alias("target_kw__mask"),
        pl.when(pl.col("timestamp") == target_missing_ts)
        .then(pl.lit(False))
        .otherwise(pl.lit(True))
        .alias("is_observed"),
        pl.when(pl.col("timestamp") == target_missing_ts)
        .then(pl.lit("row_bad"))
        .otherwise(pl.lit(""))
        .alias("quality_flags"),
        pl.when(pl.col("timestamp") == covariate_missing_ts)
        .then(pl.lit("feat_bad"))
        .otherwise(pl.lit(""))
        .alias("feature_quality_flags"),
    ]
    updates.extend(
        pl.when(pl.col("timestamp") == covariate_missing_ts)
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(pl.col(column))
        .alias(column)
        for column in _LOCAL_OBSERVATION_VALUE_COLUMNS
    )
    updates.extend(
        pl.when(pl.col("timestamp") == covariate_missing_ts)
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(pl.col(column).min().over("timestamp"))
        .alias(column)
        for column in _GLOBAL_OBSERVATION_VALUE_COLUMNS
    )
    updates.extend(
        pl.when(pl.col("timestamp") == covariate_missing_ts)
        .then(pl.lit(1, dtype=pl.Int8))
        .otherwise(pl.lit(0, dtype=pl.Int8))
        .alias(column)
        for column in _PAST_COVARIATE_MASK_COLUMNS
    )
    series = series.with_columns(*updates)
    series.write_parquet(task_dir / "series.parquet")

    _static_frame(dataset_id=dataset_id, missing_coordinates=missing_coordinates).write_parquet(
        task_dir / "static.parquet"
    )
    _pairwise_frame(dataset_id=dataset_id).write_parquet(task_dir / "pairwise.parquet")

    task_context = json.loads((task_dir / "task_context.json").read_text(encoding="utf-8"))
    task_context["feature_protocol_id"] = _FEATURE_PROTOCOL_ID
    task_context["column_groups"] = {
        "series": [
            "dataset",
            "turbine_id",
            "timestamp",
            "target_kw",
            "is_observed",
            "quality_flags",
            "feature_quality_flags",
            *_TARGET_HISTORY_MASK_COLUMNS,
            *_PAST_COVARIATE_VALUE_COLUMNS,
            *_PAST_COVARIATE_MASK_COLUMNS,
        ],
        "target_history_masks": list(_TARGET_HISTORY_MASK_COLUMNS),
        "past_covariates": list(_PAST_COVARIATE_VALUE_COLUMNS + _PAST_COVARIATE_MASK_COLUMNS),
        "past_covariate_values": list(_PAST_COVARIATE_VALUE_COLUMNS),
        "past_covariate_masks": list(_PAST_COVARIATE_MASK_COLUMNS),
        "local_observation_values": list(_LOCAL_OBSERVATION_VALUE_COLUMNS),
        "local_observation_masks": list(_LOCAL_OBSERVATION_MASK_COLUMNS),
        "global_observation_values": list(_GLOBAL_OBSERVATION_VALUE_COLUMNS),
        "global_observation_masks": list(_GLOBAL_OBSERVATION_MASK_COLUMNS),
        "known_future": ["dataset", "timestamp", *_KNOWN_FUTURE_COLUMNS],
        "static": [
            "dataset",
            "turbine_id",
            "turbine_index",
            "latitude",
            "longitude",
            "coord_x",
            "coord_y",
            "coord_kind",
            "coord_crs",
            "elevation_m",
            "rated_power_kw",
            "hub_height_m",
            "rotor_diameter_m",
        ],
        "pairwise": list(_PAIRWISE_COLUMNS),
        "target_derived_covariates": [],
        "audit": [],
    }
    (task_dir / "task_context.json").write_text(json.dumps(task_context), encoding="utf-8")


def _read_world_model_temp_bundle(
    cache_root: Path,
    *,
    dataset_id: str = "kelmarsh",
) -> SimpleNamespace:
    task_dir = cache_root / dataset_id / "tasks" / "next_6h_from_24h" / _FEATURE_PROTOCOL_ID
    return SimpleNamespace(
        series=pl.read_parquet(task_dir / "series.parquet"),
        static=pl.read_parquet(task_dir / "static.parquet"),
        window_index=pl.read_parquet(task_dir / "window_index.parquet"),
        task_context=json.loads((task_dir / "task_context.json").read_text(encoding="utf-8")),
        known_future=pl.read_parquet(task_dir / "known_future.parquet"),
        pairwise=pl.read_parquet(task_dir / "pairwise.parquet"),
    )


def _patch_bundle_loader(monkeypatch, module, cache_root: Path, *, dataset_id: str = "toy_dataset") -> None:
    def _fake_load_task_bundle(requested_dataset_id: str, *, feature_protocol_id: str, cache_root: str | Path):
        assert requested_dataset_id == dataset_id
        assert feature_protocol_id == _FEATURE_PROTOCOL_ID
        assert Path(cache_root) == cache_root_path
        return _read_world_model_temp_bundle(cache_root_path, dataset_id=dataset_id)

    cache_root_path = cache_root
    monkeypatch.setattr(module.world_model_base, "_load_task_bundle", _fake_load_task_bundle)


def _prepare_temp_dataset(module, tmp_path: Path, monkeypatch, *, dataset_id: str = "kelmarsh"):
    cache_root = tmp_path / "cache"
    _build_world_model_temp_cache(cache_root, dataset_id=dataset_id)
    _patch_bundle_loader(monkeypatch, module, cache_root, dataset_id=dataset_id)
    return module.prepare_dataset(
        dataset_id,
        cache_root=cache_root,
        max_train_origins=4,
        max_eval_origins=2,
    )


def test_prepare_dataset_builds_rollout_tensors(tmp_path, monkeypatch) -> None:
    module = _load_module()
    prepared = _prepare_temp_dataset(module, tmp_path, monkeypatch)

    assert prepared.model_variant == module.MODEL_VARIANT
    assert prepared.feature_protocol_id == module.FEATURE_PROTOCOL_ID
    assert prepared.local_input_channels == 37
    assert prepared.context_channels == 25
    assert prepared.static_feature_count == 6
    assert prepared.pairwise_feature_count == 7
    assert prepared.local_input_feature_names[:2] == ("target_pu", "target_kw__mask")
    assert prepared.local_input_feature_names[-3:] == ("is_observed", "row_bad", "feat_bad")
    assert prepared.context_feature_names[:9] == _GLOBAL_OBSERVATION_VALUE_COLUMNS
    assert prepared.context_feature_names[9:18] == _GLOBAL_OBSERVATION_MASK_COLUMNS
    assert prepared.context_feature_names[18:] == _KNOWN_FUTURE_COLUMNS
    assert np.allclose(prepared.context_future_tensor[:, :9], 0.0)
    assert np.allclose(prepared.context_future_tensor[:, 9:18], 1.0)
    assert set(np.unique(prepared.local_history_tensor[:, :, 1]).tolist()).issubset({0.0, 1.0})
    assert prepared.local_history_tensor[200, 0, 0] == pytest.approx(0.0)
    assert prepared.local_history_tensor[200, 0, -3] == pytest.approx(0.0)
    assert prepared.local_history_tensor[200, 0, -2] == pytest.approx(1.0)
    assert prepared.local_history_tensor[201, 0, -1] == pytest.approx(1.0)
    assert prepared.pairwise_tensor[1, 0, 0] > 0.0
    assert prepared.pairwise_tensor[0, 1, 0] < 0.0


def test_correct_step_skips_nodes_without_local_observations(tmp_path, monkeypatch) -> None:
    module = _load_module()
    torch_module = _require_torch(module)
    prepared = _prepare_temp_dataset(module, tmp_path, monkeypatch)
    model = module.build_model(
        node_count=prepared.node_count,
        local_input_channels=prepared.local_input_channels,
        context_channels=prepared.context_channels,
        static_tensor=prepared.static_tensor,
        pairwise_tensor=prepared.pairwise_tensor,
        local_observation_value_count=len(_LOCAL_OBSERVATION_VALUE_COLUMNS),
        node_state_dim=8,
        hidden_state_dim=6,
        latent_state_dim=2,
        global_state_dim=4,
        message_dim=8,
        edge_hidden_dim=8,
        tau_embed_dim=4,
        forecast_steps=prepared.forecast_steps,
        dropout=0.0,
    )
    local_observations = torch_module.from_numpy(prepared.local_history_tensor[144][None].copy())
    local_observations[:, 0, module._OBS_TARGET_MASK_INDEX] = 1.0
    local_observations[
        :,
        0,
        model.local_mask_start : model.local_mask_start + len(_LOCAL_OBSERVATION_VALUE_COLUMNS),
    ] = 1.0
    node_state_prior = torch_module.randn(1, prepared.node_count, 8)
    global_state_prior = torch_module.randn(1, 4)
    node_state_post, _global_state_post = model.correct_step(
        node_state_prior,
        global_state_prior,
        local_observations,
        torch_module.from_numpy(prepared.context_history_tensor[144][None]),
    )

    assert torch_module.allclose(node_state_post[:, 0, :], node_state_prior[:, 0, :])
    assert not torch_module.allclose(node_state_post[:, 1, :], node_state_prior[:, 1, :])


def test_dataset_default_profiles_use_safe_cuda_batch_sizes() -> None:
    module = _load_module()

    kelmarsh_profile = module.resolve_hyperparameter_profile(
        module.MODEL_VARIANT,
        dataset_id="kelmarsh",
    )
    penmanshiel_profile = module.resolve_hyperparameter_profile(
        module.MODEL_VARIANT,
        dataset_id="penmanshiel",
    )
    explicit_penmanshiel_profile = module.resolve_hyperparameter_profile(
        module.MODEL_VARIANT,
        dataset_id="penmanshiel",
        batch_size=96,
    )

    assert kelmarsh_profile.batch_size == 256
    assert penmanshiel_profile.batch_size == 64
    assert explicit_penmanshiel_profile.batch_size == 96


def test_execute_training_job_smoke_returns_result_rows(tmp_path, monkeypatch) -> None:
    module = _load_module()
    _require_torch(module)
    prepared = _prepare_temp_dataset(module, tmp_path, monkeypatch)
    history_path = tmp_path / "training_history.csv"

    rows = module.execute_training_job(
        prepared,
        device="cpu",
        seed=123,
        batch_size=2,
        eval_batch_size=2,
        learning_rate=1e-3,
        max_epochs=1,
        early_stopping_patience=1,
        node_state_dim=16,
        hidden_state_dim=12,
        latent_state_dim=4,
        global_state_dim=8,
        message_dim=8,
        edge_hidden_dim=8,
        tau_embed_dim=4,
        dropout=0.0,
        grad_clip_norm=1.0,
        training_history_path=history_path,
    )

    assert history_path.exists()
    history = pl.read_csv(history_path)
    assert history["epoch"].to_list() == [1]
    assert history["train_loss_mean"].null_count() == 0
    assert history["train_future_loss_mean"].null_count() == 0
    assert history["train_consistency_loss_mean"].null_count() == 0
    assert len(rows) == 148
    assert {row["split_name"] for row in rows} == {"val", "test"}
    assert {row["eval_protocol"] for row in rows} == {"rolling_origin_no_refit", "non_overlap"}
    assert rows[0]["model_variant"] == module.MODEL_VARIANT
    assert rows[0]["node_state_dim"] == 16
    assert rows[0]["latent_state_dim"] == 4
    assert rows[0]["message_dim"] == 8


def test_run_experiment_writes_results_with_fake_runner(tmp_path, monkeypatch) -> None:
    module = _load_module()
    prepared = _prepare_temp_dataset(module, tmp_path, monkeypatch)
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
            node_state_dim=kwargs["node_state_dim"],
            hidden_state_dim=kwargs["hidden_state_dim"],
            latent_state_dim=kwargs["latent_state_dim"],
            global_state_dim=kwargs["global_state_dim"],
            message_dim=kwargs["message_dim"],
            edge_hidden_dim=kwargs["edge_hidden_dim"],
            tau_embed_dim=kwargs["tau_embed_dim"],
            dropout=kwargs["dropout"],
            grad_clip_norm=kwargs["grad_clip_norm"],
            hist_loss_weight=kwargs["hist_loss_weight"],
            aux_turbine_loss_weight=kwargs["aux_turbine_loss_weight"],
            aux_site_loss_weight=kwargs["aux_site_loss_weight"],
            consistency_loss_weight=kwargs["consistency_loss_weight"],
            weight_decay=kwargs["weight_decay"],
        )
        metrics = module.EvaluationMetrics(
            window_count=2,
            prediction_count=prepared_dataset.forecast_steps * prepared_dataset.node_count,
            mae_kw=1.0,
            rmse_kw=2.0,
            mae_pu=0.1,
            rmse_pu=0.2,
            horizon_window_count=np.full((prepared_dataset.forecast_steps,), 2, dtype=np.int64),
            horizon_prediction_count=np.full((prepared_dataset.forecast_steps,), prepared_dataset.node_count, dtype=np.int64),
            horizon_mae_kw=np.full((prepared_dataset.forecast_steps,), 1.0, dtype=np.float64),
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
            device="cpu",
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
        node_state_dim=16,
        hidden_state_dim=12,
        latent_state_dim=4,
        global_state_dim=8,
        message_dim=8,
        edge_hidden_dim=8,
        tau_embed_dim=4,
        dropout=0.0,
        work_root=work_root,
        dataset_loader=_dataset_loader,
        job_runner=_job_runner,
    )

    assert output_path.exists()
    assert results.height == 148
    assert set(results["metric_scope"].unique().to_list()) == {"overall", "horizon"}
    assert set(results["model_variant"].unique().to_list()) == {module.MODEL_VARIANT}
    state_path = next(work_root.glob("*/run_state.json"))
    state_payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert state_payload["status"] == "complete"

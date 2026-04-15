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

from test.test_agcrn import (
    _TqdmRecorder,
    _build_temp_cache,
    _descriptor,
    _require_torch,
    _work_root,
)


_FEATURE_PROTOCOL_ID = "world_model_v1"
_TARGET_HISTORY_MASK_COLUMNS = ("target_kw__mask",)
_PAST_COVARIATE_VALUE_COLUMNS = (
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
_PAST_COVARIATE_MASK_COLUMNS = tuple(f"{column}__mask" for column in _PAST_COVARIATE_VALUE_COLUMNS)
_PAST_COVARIATE_COLUMNS = _PAST_COVARIATE_VALUE_COLUMNS + _PAST_COVARIATE_MASK_COLUMNS
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
        / "world_model_agcrn_v1"
        / "world_model_agcrn_v1.py"
    )
    spec = spec_from_file_location("world_model_agcrn_v1", module_path)
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
        / "world_model_agcrn_v1"
        / "search_world_model_agcrn_v1.py"
    )
    spec = spec_from_file_location("search_world_model_agcrn_v1", module_path)
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
            "latitude": [None, None, None] if not missing_coordinates else [None, None, None],
            "longitude": [None, None, None] if not missing_coordinates else [None, None, None],
            "coord_x": [0.0, 100.0, 220.0] if not missing_coordinates else [None, None, None],
            "coord_y": [0.0, 0.0, 30.0] if not missing_coordinates else [None, None, None],
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
    elevations = {
        "T01": 10.0,
        "T02": 12.0,
        "T03": 16.0,
    }
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
    dataset_id: str = "toy_dataset",
    missing_coordinates: bool = False,
) -> None:
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

    static = _static_frame(dataset_id=dataset_id, missing_coordinates=missing_coordinates)
    static.write_parquet(task_dir / "static.parquet")
    _pairwise_frame(dataset_id=dataset_id).write_parquet(task_dir / "pairwise.parquet")

    task_context = json.loads((task_dir / "task_context.json").read_text(encoding="utf-8"))
    task_context["feature_protocol_id"] = _FEATURE_PROTOCOL_ID
    task_context["column_groups"]["target_history_masks"] = list(_TARGET_HISTORY_MASK_COLUMNS)
    task_context["column_groups"]["past_covariates"] = list(_PAST_COVARIATE_COLUMNS)
    task_context["column_groups"]["known_future"] = ["dataset", "timestamp", *_KNOWN_FUTURE_COLUMNS]
    task_context["column_groups"]["static"] = list(static.columns)
    task_context["column_groups"]["pairwise"] = list(_PAIRWISE_COLUMNS)
    (task_dir / "task_context.json").write_text(json.dumps(task_context), encoding="utf-8")


def _read_world_model_temp_bundle(
    cache_root: Path,
    *,
    dataset_id: str = "toy_dataset",
    feature_protocol_id: str = _FEATURE_PROTOCOL_ID,
) -> SimpleNamespace:
    task_dir = cache_root / dataset_id / "tasks" / "next_6h_from_24h" / feature_protocol_id
    payload = {
        "series": pl.read_parquet(task_dir / "series.parquet"),
        "static": pl.read_parquet(task_dir / "static.parquet"),
        "window_index": pl.read_parquet(task_dir / "window_index.parquet"),
        "task_context": json.loads((task_dir / "task_context.json").read_text(encoding="utf-8")),
    }
    known_future_path = task_dir / "known_future.parquet"
    pairwise_path = task_dir / "pairwise.parquet"
    if known_future_path.exists():
        payload["known_future"] = pl.read_parquet(known_future_path)
    if pairwise_path.exists():
        payload["pairwise"] = pl.read_parquet(pairwise_path)
    return SimpleNamespace(**payload)


def _patch_world_model_bundle_loader(
    monkeypatch,
    module,
    cache_root: Path,
    *,
    dataset_id: str = "toy_dataset",
    feature_protocol_id: str = _FEATURE_PROTOCOL_ID,
) -> None:
    def _fake_load_task_bundle(requested_dataset_id: str, *, feature_protocol_id: str, cache_root: str | Path):
        assert requested_dataset_id == dataset_id
        assert feature_protocol_id == expected_feature_protocol_id
        assert Path(cache_root) == cache_root_path
        return _read_world_model_temp_bundle(
            cache_root_path,
            dataset_id=dataset_id,
            feature_protocol_id=feature_protocol_id,
        )

    cache_root_path = cache_root
    expected_feature_protocol_id = feature_protocol_id
    monkeypatch.setattr(module, "_load_task_bundle", _fake_load_task_bundle)


def _small_prepared_dataset(module, *, dataset_id: str = "kelmarsh"):
    timestamps_us = np.arange(0, 500 * 600_000_000, 600_000_000, dtype=np.int64)
    target_pu = np.stack(
        [
            np.linspace(0.0, 1.0, len(timestamps_us), dtype=np.float32),
            np.linspace(0.1, 0.9, len(timestamps_us), dtype=np.float32),
            np.linspace(0.2, 0.8, len(timestamps_us), dtype=np.float32),
        ],
        axis=1,
    )
    source_channels = [target_pu]
    for channel_index in range(1, module.INPUT_CHANNELS):
        source_channels.append(target_pu + np.float32(channel_index * 0.01))
    source_tensor = np.stack(source_channels, axis=-1).astype(np.float32, copy=False)
    known_future_base = np.asarray([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    known_future_tensor = np.tile(known_future_base, (len(timestamps_us), 1))
    static_tensor = np.asarray(
        [
            [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
            [0.5, -0.5, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    pairwise_tensor = np.zeros((3, 3, len(module.PAIRWISE_NUMERIC_COLUMNS)), dtype=np.float32)
    for src_index in range(3):
        for dst_index in range(3):
            if src_index == dst_index:
                continue
            pairwise_tensor[src_index, dst_index, :] = np.asarray(
                [
                    float(dst_index - src_index),
                    float((dst_index - src_index) * 2),
                    float(abs(dst_index - src_index) + 1),
                    float(dst_index - src_index),
                    float(abs(dst_index - src_index) / 2),
                    math.sin(dst_index - src_index),
                    math.cos(dst_index - src_index),
                ],
                dtype=np.float32,
            )
    return module.PreparedDataset(
        dataset_id=dataset_id,
        model_variant=module.MODEL_VARIANT,
        feature_protocol_id=module.FEATURE_PROTOCOL_ID,
        resolution_minutes=10,
        rated_power_kw=2050.0,
        history_steps=144,
        forecast_steps=36,
        stride_steps=1,
        turbine_ids=("T01", "T02", "T03"),
        coordinate_mode="coord_xy",
        node_count=3,
        timestamps_us=timestamps_us,
        input_channel_names=(
            "target_pu",
            "target_kw__mask",
            *tuple(f"covariate_{index}" for index in range(module.INPUT_CHANNELS - 2)),
        ),
        known_future_channel_names=module.KNOWN_FUTURE_FEATURE_COLUMNS,
        static_feature_names=("coord_x", "coord_y", "elevation_m", "rated_power_kw", "hub_height_m", "rotor_diameter_m"),
        pairwise_feature_names=module.PAIRWISE_NUMERIC_COLUMNS,
        source_tensor=source_tensor,
        known_future_tensor=known_future_tensor,
        static_tensor=static_tensor,
        pairwise_tensor=pairwise_tensor,
        target_pu=target_pu,
        target_pu_filled=target_pu.copy(),
        target_valid_mask=np.ones_like(target_pu, dtype=bool),
        train_windows=_descriptor(module, target_indices=[200, 201, 202, 203], forecast_steps=36),
        val_rolling_windows=_descriptor(module, target_indices=[300, 301], forecast_steps=36),
        val_non_overlap_windows=_descriptor(module, target_indices=[300], forecast_steps=36),
        test_rolling_windows=_descriptor(module, target_indices=[360, 361, 362], forecast_steps=36),
        test_non_overlap_windows=_descriptor(module, target_indices=[360], forecast_steps=36),
    )


def test_prepare_dataset_builds_world_model_tensors(tmp_path, monkeypatch) -> None:
    module = _load_module()
    cache_root = tmp_path / "cache"
    _build_world_model_temp_cache(cache_root)
    _patch_world_model_bundle_loader(monkeypatch, module, cache_root)

    prepared = module.prepare_dataset("toy_dataset", cache_root=cache_root)

    value_start = 2
    value_end = value_start + len(_PAST_COVARIATE_VALUE_COLUMNS)
    mask_start = value_end
    assert prepared.model_variant == module.MODEL_VARIANT
    assert prepared.feature_protocol_id == module.FEATURE_PROTOCOL_ID
    assert prepared.input_channels == 52
    assert prepared.known_future_channels == 7
    assert prepared.static_feature_count == 6
    assert prepared.pairwise_feature_count == 7
    assert prepared.input_channel_names[0] == "target_pu"
    assert prepared.input_channel_names[1] == "target_kw__mask"
    assert prepared.input_channel_names[-1] == "farm_evt_informational_active__mask"
    assert np.isfinite(prepared.source_tensor).all()
    assert np.isfinite(prepared.known_future_tensor).all()
    assert np.isfinite(prepared.static_tensor).all()
    assert np.isfinite(prepared.pairwise_tensor).all()
    assert set(np.unique(prepared.source_tensor[:, :, 1]).tolist()).issubset({0.0, 1.0})
    assert prepared.source_tensor[200, 0, 0] == pytest.approx(0.0)
    assert prepared.source_tensor[201, 0, value_start:value_end].tolist() == pytest.approx(
        [0.0] * len(_PAST_COVARIATE_VALUE_COLUMNS)
    )
    assert prepared.source_tensor[201, 0, mask_start : mask_start + len(_PAST_COVARIATE_MASK_COLUMNS)].tolist() == pytest.approx(
        [1.0] * len(_PAST_COVARIATE_MASK_COLUMNS)
    )


def test_runtime_throughput_helpers_prefer_larger_cuda_batches_and_workers(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module.os, "cpu_count", lambda: 16)

    assert module.resolve_eval_batch_size(512, device="cuda") == 2048
    assert module.resolve_eval_batch_size(512, device="cpu") == 512
    assert module.resolve_eval_batch_size(512, device="cuda", eval_batch_size=1536) == 1536
    assert module.resolve_loader_num_workers(device="cuda") == 4
    assert module.resolve_loader_num_workers(device="cpu") == 0
    assert module.resolve_loader_num_workers(device="cuda", num_workers=2) == 2


def test_prepare_dataset_requires_known_future_sidecar(tmp_path, monkeypatch) -> None:
    module = _load_module()
    cache_root = tmp_path / "cache"
    _build_world_model_temp_cache(cache_root)
    task_dir = cache_root / "toy_dataset" / "tasks" / "next_6h_from_24h" / _FEATURE_PROTOCOL_ID
    (task_dir / "known_future.parquet").unlink()
    _patch_world_model_bundle_loader(monkeypatch, module, cache_root)

    with pytest.raises(ValueError, match="missing known_future sidecar"):
        module.prepare_dataset("toy_dataset", cache_root=cache_root)


def test_prepare_dataset_requires_pairwise_sidecar(tmp_path, monkeypatch) -> None:
    module = _load_module()
    cache_root = tmp_path / "cache"
    _build_world_model_temp_cache(cache_root)
    task_dir = cache_root / "toy_dataset" / "tasks" / "next_6h_from_24h" / _FEATURE_PROTOCOL_ID
    (task_dir / "pairwise.parquet").unlink()
    _patch_world_model_bundle_loader(monkeypatch, module, cache_root)

    with pytest.raises(ValueError, match="missing pairwise sidecar"):
        module.prepare_dataset("toy_dataset", cache_root=cache_root)


def test_prepare_dataset_rejects_static_without_coordinates(tmp_path, monkeypatch) -> None:
    module = _load_module()
    cache_root = tmp_path / "cache"
    _build_world_model_temp_cache(cache_root, missing_coordinates=True)
    _patch_world_model_bundle_loader(monkeypatch, module, cache_root)

    with pytest.raises(ValueError, match="coord_x/coord_y or full latitude/longitude"):
        module.prepare_dataset("toy_dataset", cache_root=cache_root)


def test_build_pairwise_tensor_rejects_misaligned_ids(tmp_path, monkeypatch) -> None:
    module = _load_module()
    cache_root = tmp_path / "cache"
    _build_world_model_temp_cache(cache_root)
    _patch_world_model_bundle_loader(monkeypatch, module, cache_root)
    bundle = _read_world_model_temp_bundle(cache_root)
    pairwise = bundle.pairwise.with_columns(
        pl.when(
            (pl.col("src_turbine_index") == 0)
            & (pl.col("dst_turbine_index") == 1)
        )
        .then(pl.lit("T03"))
        .otherwise(pl.col("src_turbine_id"))
        .alias("src_turbine_id")
    )

    with pytest.raises(ValueError, match="src/dst turbine ids do not match"):
        module.build_pairwise_tensor(
            pairwise,
            turbine_ids=("T01", "T02", "T03"),
            pairwise_feature_names=module.PAIRWISE_NUMERIC_COLUMNS,
        )


def test_seq2seq_model_forward_and_graph_context_react_to_static_and_pairwise() -> None:
    module = _load_module()
    torch_module = _require_torch(module)
    prepared = _small_prepared_dataset(module)
    batch_history = torch_module.from_numpy(prepared.source_tensor[56:200][None])
    batch_known_future = torch_module.from_numpy(prepared.known_future_tensor[200:236][None])
    batch_targets = torch_module.from_numpy(prepared.target_pu_filled[200:236][:, :, None][None])

    torch_module.manual_seed(123)
    model = module.build_model(
        node_count=prepared.node_count,
        input_channels=prepared.input_channels,
        known_future_channels=prepared.known_future_channels,
        static_tensor=prepared.static_tensor,
        pairwise_tensor=prepared.pairwise_tensor,
        hidden_dim=8,
        embed_dim=4,
        num_layers=1,
        cheb_k=2,
        forecast_steps=prepared.forecast_steps,
    )
    model.train()
    teacher_forced = model(batch_history, batch_known_future, batch_targets, teacher_forcing_ratio=1.0)
    model.eval()
    autoregressive = model(batch_history, batch_known_future, targets=None, teacher_forcing_ratio=0.0)

    assert tuple(teacher_forced.shape) == (1, prepared.forecast_steps, prepared.node_count, 1)
    assert tuple(autoregressive.shape) == (1, prepared.forecast_steps, prepared.node_count, 1)

    static_changed = prepared.static_tensor.copy()
    static_changed[0, 0] += 1.5
    torch_module.manual_seed(999)
    model_static_a = module.build_model(
        node_count=prepared.node_count,
        input_channels=prepared.input_channels,
        known_future_channels=prepared.known_future_channels,
        static_tensor=prepared.static_tensor,
        pairwise_tensor=prepared.pairwise_tensor,
        hidden_dim=8,
        embed_dim=4,
        num_layers=1,
        cheb_k=2,
        forecast_steps=prepared.forecast_steps,
    )
    torch_module.manual_seed(999)
    model_static_b = module.build_model(
        node_count=prepared.node_count,
        input_channels=prepared.input_channels,
        known_future_channels=prepared.known_future_channels,
        static_tensor=static_changed,
        pairwise_tensor=prepared.pairwise_tensor,
        hidden_dim=8,
        embed_dim=4,
        num_layers=1,
        cheb_k=2,
        forecast_steps=prepared.forecast_steps,
    )
    embeddings_a, _geometry_a, _supports_a = model_static_a.compute_graph_context()
    embeddings_b, _geometry_b, _supports_b = model_static_b.compute_graph_context()
    assert not torch_module.allclose(embeddings_a, embeddings_b)

    pairwise_changed = prepared.pairwise_tensor.copy()
    pairwise_changed[0, 1, 0] += 5.0
    torch_module.manual_seed(2024)
    model_pairwise_a = module.build_model(
        node_count=prepared.node_count,
        input_channels=prepared.input_channels,
        known_future_channels=prepared.known_future_channels,
        static_tensor=prepared.static_tensor,
        pairwise_tensor=prepared.pairwise_tensor,
        hidden_dim=8,
        embed_dim=4,
        num_layers=1,
        cheb_k=2,
        forecast_steps=prepared.forecast_steps,
    )
    torch_module.manual_seed(2024)
    model_pairwise_b = module.build_model(
        node_count=prepared.node_count,
        input_channels=prepared.input_channels,
        known_future_channels=prepared.known_future_channels,
        static_tensor=prepared.static_tensor,
        pairwise_tensor=pairwise_changed,
        hidden_dim=8,
        embed_dim=4,
        num_layers=1,
        cheb_k=2,
        forecast_steps=prepared.forecast_steps,
    )
    _embeddings_a, geometry_a, supports_a = model_pairwise_a.compute_graph_context()
    _embeddings_b, geometry_b, supports_b = model_pairwise_b.compute_graph_context()
    assert not torch_module.allclose(geometry_a, geometry_b)
    assert not torch_module.allclose(supports_a, supports_b)


def test_masked_mse_loss_and_evaluate_model_ignore_invalid_targets() -> None:
    module = _load_module()
    torch_module = _require_torch(module)
    history = torch_module.zeros((1, 4, 2, module.INPUT_CHANNELS), dtype=torch_module.float32)
    known_future = torch_module.zeros((1, 2, len(module.KNOWN_FUTURE_FEATURE_COLUMNS)), dtype=torch_module.float32)
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

        def __call__(self, batch_history, batch_known_future, batch_targets, teacher_forcing_ratio=0.0):
            del batch_history, batch_known_future, teacher_forcing_ratio
            return batch_targets + 1.0

    loss = module.masked_mse_loss(
        predictions,
        targets,
        valid_mask,
        torch_module=torch_module,
    )
    metrics = module.evaluate_model(
        _FakeModel(),
        [(history, known_future, targets, valid_mask)],
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


def test_run_experiment_smoke_trains_world_model_family_on_toy_cache(tmp_path, monkeypatch) -> None:
    module = _load_module()
    _require_torch(module)
    cache_root = tmp_path / "cache"
    _build_world_model_temp_cache(cache_root, dataset_id="kelmarsh")
    _patch_world_model_bundle_loader(monkeypatch, module, cache_root, dataset_id="kelmarsh")
    monkeypatch.setattr(module, "tqdm", _TqdmRecorder)
    output_path = tmp_path / "world-model-agcrn.csv"

    results = module.run_experiment(
        dataset_ids=("kelmarsh",),
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
    history_path = module.training_history_output_path(output_path)
    assert history_path.exists()
    history = pl.read_csv(history_path)
    assert history["epoch"].to_list() == [1]
    assert history["train_loss_mean"].null_count() == 0
    assert history["val_rmse_pu"].null_count() == 0
    assert results.height == 148
    assert results["model_variant"].unique().to_list() == [module.MODEL_VARIANT]
    assert results["graph_source"].unique().to_list() == [module.GRAPH_SOURCE]
    assert results["prediction_count"].min() > 0


def test_search_harness_writes_selected_defaults(tmp_path, monkeypatch) -> None:
    module = _load_module()
    search_module = _load_search_module()
    prepared_by_dataset = {
        dataset_id: _small_prepared_dataset(module, dataset_id=dataset_id)
        for dataset_id in module.DEFAULT_DATASETS
    }
    selected_config_names = tuple(config.name for config in search_module.COMMON_SCREEN_CONFIGS[:2])
    metrics_by_dataset_and_config = {
        "kelmarsh": {
            selected_config_names[0]: (0.20, 0.22, 9.0, 0.60),
            selected_config_names[1]: (0.25, 0.26, 3.0, 0.10),
        },
        "penmanshiel": {
            selected_config_names[0]: (0.33, 0.36, 6.0, 0.40),
            selected_config_names[1]: (0.21, 0.24, 5.0, 0.80),
        },
    }

    def _fake_prepared_by_variant(dataset_id: str, *, max_train_origins, max_eval_origins):
        del max_train_origins, max_eval_origins
        return {module.MODEL_VARIANT: prepared_by_dataset[dataset_id]}

    def _screen_row(prepared_dataset, config, *, device, seed, max_epochs, patience):
        del device, seed, max_epochs, patience
        val_rolling, val_non_overlap, runtime_seconds, _test_rmse = metrics_by_dataset_and_config[
            prepared_dataset.dataset_id
        ][config.name]
        return {
            "dataset_id": prepared_dataset.dataset_id,
            "model_variant": prepared_dataset.model_variant,
            "feature_protocol_id": prepared_dataset.feature_protocol_id,
            "stage": "screen",
            "config_name": config.name,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "hidden_dim": config.hidden_dim,
            "embed_dim": config.embed_dim,
            "num_layers": config.num_layers,
            "cheb_k": config.cheb_k,
            "teacher_forcing_ratio": config.teacher_forcing_ratio,
            "weight_decay": config.weight_decay,
            "best_epoch": 1,
            "epochs_ran": 1,
            "best_val_rmse_pu": val_rolling,
            "val_rolling_window_count": len(prepared_dataset.val_rolling_windows),
            "val_non_overlap_window_count": len(prepared_dataset.val_non_overlap_windows),
            "val_rolling_rmse_pu": val_rolling,
            "val_rolling_mae_pu": val_rolling / 2.0,
            "val_non_overlap_rmse_pu": val_non_overlap,
            "val_non_overlap_mae_pu": val_non_overlap / 2.0,
            "train_window_count": len(prepared_dataset.train_windows),
            "runtime_seconds": runtime_seconds,
            "device": "cpu",
            "seed": 42,
        }

    def _final_row(prepared_dataset, config, *, device, seed, max_epochs, patience):
        del device, seed, max_epochs, patience
        val_rolling, val_non_overlap, runtime_seconds, test_rmse = metrics_by_dataset_and_config[
            prepared_dataset.dataset_id
        ][config.name]
        summary = {
            "dataset_id": prepared_dataset.dataset_id,
            "model_variant": prepared_dataset.model_variant,
            "feature_protocol_id": prepared_dataset.feature_protocol_id,
            "stage": "final",
            "config_name": config.name,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "hidden_dim": config.hidden_dim,
            "embed_dim": config.embed_dim,
            "num_layers": config.num_layers,
            "cheb_k": config.cheb_k,
            "teacher_forcing_ratio": config.teacher_forcing_ratio,
            "weight_decay": config.weight_decay,
            "best_epoch": 1,
            "epochs_ran": 1,
            "best_val_rmse_pu": val_rolling,
            "train_window_count": len(prepared_dataset.train_windows),
            "val_rolling_window_count": len(prepared_dataset.val_rolling_windows),
            "val_non_overlap_window_count": len(prepared_dataset.val_non_overlap_windows),
            "test_rolling_window_count": len(prepared_dataset.test_rolling_windows),
            "test_non_overlap_window_count": len(prepared_dataset.test_non_overlap_windows),
            "val_rolling_rmse_pu": val_rolling,
            "val_rolling_mae_pu": val_rolling / 2.0,
            "val_non_overlap_rmse_pu": val_non_overlap,
            "val_non_overlap_mae_pu": val_non_overlap / 2.0,
            "test_rolling_rmse_pu": test_rmse,
            "test_rolling_mae_pu": test_rmse / 2.0,
            "test_non_overlap_rmse_pu": test_rmse + 0.01,
            "test_non_overlap_mae_pu": (test_rmse + 0.01) / 2.0,
            "runtime_seconds": runtime_seconds,
            "device": "cpu",
            "seed": 42,
        }
        detail_frame = pl.DataFrame(
            {
                "dataset_id": [prepared_dataset.dataset_id],
                "config_name": [config.name],
                "test_rolling_rmse_pu": [test_rmse],
            }
        )
        return summary, detail_frame

    monkeypatch.setattr(search_module, "_prepared_by_variant", _fake_prepared_by_variant)
    monkeypatch.setattr(search_module, "_screen_one", _screen_row)
    monkeypatch.setattr(search_module, "_final_one", _final_row)

    output_dir = tmp_path / "search"
    screen_frame, final_frame = search_module.run_search(
        dataset_ids=module.DEFAULT_DATASETS,
        tuned_variants=(module.MODEL_VARIANT,),
        config_names=selected_config_names,
        device="cpu",
        seed=42,
        screen_train_origins=32,
        screen_eval_origins=8,
        screen_epochs=1,
        screen_patience=1,
        full_epochs=1,
        full_patience=1,
        top_k=2,
        output_dir=output_dir,
        skip_final=False,
        full_only=False,
    )

    assert not screen_frame.is_empty()
    assert not final_frame.is_empty()
    assert (output_dir / "screen_summary.csv").exists()
    assert (output_dir / "final_summary.csv").exists()
    assert (output_dir / "final_detailed_rows.csv").exists()
    assert (output_dir / "search_plan.json").exists()
    assert (output_dir / "selected_defaults.json").exists()

    selected_defaults = json.loads((output_dir / "selected_defaults.json").read_text(encoding="utf-8"))
    assert selected_defaults["selection_rule"] == [
        "val_rolling_rmse_pu",
        "val_non_overlap_rmse_pu",
        "runtime_seconds",
        "config_name",
    ]
    assert (
        selected_defaults["selected_defaults"]["kelmarsh"][module.MODEL_VARIANT]["config_name"]
        == selected_config_names[0]
    )
    assert (
        selected_defaults["selected_defaults"]["kelmarsh"][module.MODEL_VARIANT]["teacher_forcing_ratio"]
        == search_module.COMMON_SCREEN_CONFIGS[0].teacher_forcing_ratio
    )
    assert (
        selected_defaults["selected_defaults"]["penmanshiel"][module.MODEL_VARIANT]["config_name"]
        == selected_config_names[1]
    )
    assert (
        selected_defaults["selected_defaults"]["penmanshiel"][module.MODEL_VARIANT]["weight_decay"]
        == search_module.COMMON_SCREEN_CONFIGS[1].weight_decay
    )


def test_search_screen_one_uses_training_device_for_eval_loaders(monkeypatch) -> None:
    module = _load_module()
    search_module = _load_search_module()
    prepared = _small_prepared_dataset(module, dataset_id="kelmarsh")
    loader_devices: list[str] = []

    def _fake_train_model(*args, **kwargs):
        del args, kwargs
        return SimpleNamespace(
            model=object(),
            device="cuda",
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

    def _fake_evaluate_model(model, loader, *, device, rated_power_kw, forecast_steps, progress_label):
        del model, loader, rated_power_kw, forecast_steps, progress_label
        assert device == "cuda"
        return SimpleNamespace(rmse_pu=0.2, mae_pu=0.1)

    monkeypatch.setattr(search_module.agcrn, "train_model", _fake_train_model)
    monkeypatch.setattr(search_module.agcrn, "_build_dataloader", _fake_build_dataloader)
    monkeypatch.setattr(search_module.agcrn, "evaluate_model", _fake_evaluate_model)

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


def test_search_harness_skips_oom_configs(tmp_path, monkeypatch) -> None:
    module = _load_module()
    search_module = _load_search_module()
    prepared = _small_prepared_dataset(module, dataset_id="kelmarsh")
    selected_configs = tuple(search_module.COMMON_SCREEN_CONFIGS[:2])

    def _fake_prepared_by_variant(dataset_id: str, *, max_train_origins, max_eval_origins):
        del dataset_id, max_train_origins, max_eval_origins
        return {module.MODEL_VARIANT: prepared}

    def _screen_row(prepared_dataset, config, *, device, seed, max_epochs, patience):
        del prepared_dataset, device, seed, max_epochs, patience
        if config.name == selected_configs[0].name:
            raise RuntimeError("CUDA out of memory while screening")
        return {
            "dataset_id": "kelmarsh",
            "model_variant": module.MODEL_VARIANT,
            "feature_protocol_id": module.FEATURE_PROTOCOL_ID,
            "stage": "screen",
            "config_name": config.name,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "hidden_dim": config.hidden_dim,
            "embed_dim": config.embed_dim,
            "num_layers": config.num_layers,
            "cheb_k": config.cheb_k,
            "teacher_forcing_ratio": config.teacher_forcing_ratio,
            "weight_decay": config.weight_decay,
            "best_epoch": 1,
            "epochs_ran": 1,
            "best_val_rmse_pu": 0.2,
            "val_rolling_window_count": len(prepared.val_rolling_windows),
            "val_non_overlap_window_count": len(prepared.val_non_overlap_windows),
            "val_rolling_rmse_pu": 0.2,
            "val_rolling_mae_pu": 0.1,
            "val_non_overlap_rmse_pu": 0.21,
            "val_non_overlap_mae_pu": 0.11,
            "train_window_count": len(prepared.train_windows),
            "runtime_seconds": 1.0,
            "device": "cuda",
            "seed": 42,
        }

    def _final_row(prepared_dataset, config, *, device, seed, max_epochs, patience):
        del prepared_dataset, device, seed, max_epochs, patience
        summary = {
            "dataset_id": "kelmarsh",
            "model_variant": module.MODEL_VARIANT,
            "feature_protocol_id": module.FEATURE_PROTOCOL_ID,
            "stage": "final",
            "config_name": config.name,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "hidden_dim": config.hidden_dim,
            "embed_dim": config.embed_dim,
            "num_layers": config.num_layers,
            "cheb_k": config.cheb_k,
            "teacher_forcing_ratio": config.teacher_forcing_ratio,
            "weight_decay": config.weight_decay,
            "best_epoch": 1,
            "epochs_ran": 1,
            "best_val_rmse_pu": 0.2,
            "train_window_count": len(prepared.train_windows),
            "val_rolling_window_count": len(prepared.val_rolling_windows),
            "val_non_overlap_window_count": len(prepared.val_non_overlap_windows),
            "test_rolling_window_count": len(prepared.test_rolling_windows),
            "test_non_overlap_window_count": len(prepared.test_non_overlap_windows),
            "val_rolling_rmse_pu": 0.2,
            "val_rolling_mae_pu": 0.1,
            "val_non_overlap_rmse_pu": 0.21,
            "val_non_overlap_mae_pu": 0.11,
            "test_rolling_rmse_pu": 0.22,
            "test_rolling_mae_pu": 0.12,
            "test_non_overlap_rmse_pu": 0.23,
            "test_non_overlap_mae_pu": 0.13,
            "runtime_seconds": 2.0,
            "device": "cuda",
            "seed": 42,
        }
        return summary, pl.DataFrame({"config_name": [config.name]})

    monkeypatch.setattr(search_module, "_prepared_by_variant", _fake_prepared_by_variant)
    monkeypatch.setattr(search_module, "_screen_one", _screen_row)
    monkeypatch.setattr(search_module, "_final_one", _final_row)

    screen_frame, final_frame = search_module.run_search(
        dataset_ids=("kelmarsh",),
        tuned_variants=(module.MODEL_VARIANT,),
        config_names=tuple(config.name for config in selected_configs),
        device="cuda",
        seed=42,
        screen_train_origins=32,
        screen_eval_origins=8,
        screen_epochs=1,
        screen_patience=1,
        full_epochs=1,
        full_patience=1,
        top_k=2,
        output_dir=tmp_path / "search",
        skip_final=False,
        full_only=False,
    )

    assert screen_frame.height == 1
    assert final_frame.height == 1
    assert screen_frame["config_name"].to_list() == [selected_configs[1].name]
    assert final_frame["config_name"].to_list() == [selected_configs[1].name]

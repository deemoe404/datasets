from __future__ import annotations

from datetime import datetime, timedelta
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "experiment"
        / "families"
        / "agcrn"
        / "agcrn.py"
    )
    spec = spec_from_file_location("agcrn", module_path)
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
    feature_protocol_id: str = "power_only",
    past_covariate_columns: tuple[str, ...] = (),
    missing_past_covariate_indices: tuple[int, ...] = (),
    history_steps: int = 144,
    forecast_steps: int = 36,
    task_static_layout: str = "complete",
) -> None:
    dataset_root = cache_root / dataset_id
    task_dir = dataset_root / "tasks" / "next_6h_from_24h" / feature_protocol_id
    task_dir.mkdir(parents=True, exist_ok=True)

    timestamps = pl.datetime_range(
        datetime(2024, 1, 1, 0, 0, 0),
        datetime(2024, 1, 14, 21, 10, 0),
        interval="10m",
        eager=True,
    )
    turbine_ids = ("T01", "T02", "T03")
    rows: list[dict[str, object]] = []
    for turbine_index, turbine_id in enumerate(turbine_ids):
        offset = turbine_index * 10_000
        for index, timestamp in enumerate(timestamps):
            row = {
                "dataset": dataset_id,
                "turbine_id": turbine_id,
                "timestamp": timestamp,
                "target_kw": float(offset + index),
            }
            for column_index, column_name in enumerate(past_covariate_columns, start=1):
                row[column_name] = (
                    None
                    if index in missing_past_covariate_indices
                    else float((column_index * 1000) + offset + index)
                )
            rows.append(row)
    pl.DataFrame(rows).write_parquet(task_dir / "series.parquet")

    missing_indices = set(missing_past_covariate_indices)
    window_rows = []
    for start_index in range(history_steps, len(timestamps) - forecast_steps + 1):
        input_indices = range(start_index - history_steps, start_index)
        output_indices = range(start_index, start_index + forecast_steps)
        feature_flags: list[str] = []
        if any(index in missing_indices for index in input_indices):
            feature_flags.append("feature_quality_issues_input")
        if any(index in missing_indices for index in output_indices):
            feature_flags.append("feature_quality_issues_output")
        window_rows.append(
            {
                "dataset": dataset_id,
                "output_start_ts": timestamps[start_index],
                "output_end_ts": timestamps[start_index + forecast_steps - 1],
                "is_complete_input": True,
                "is_complete_output": True,
                "is_fully_synchronous_input": True,
                "is_fully_synchronous_output": True,
                "quality_flags": "",
                "feature_quality_flags": "|".join(feature_flags),
            }
        )
    pl.DataFrame(window_rows).write_parquet(task_dir / "window_index.parquet")

    (task_dir / "task_context.json").write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "schema_version": "task_bundle.v1",
                "feature_protocol_id": feature_protocol_id,
                "turbine_ids": list(turbine_ids),
                "task": {
                    "task_id": "next_6h_from_24h",
                    "history_steps": history_steps,
                    "forecast_steps": forecast_steps,
                    "stride_steps": 1,
                },
                "time_axis_semantics": "farm_synchronous_long_panel",
                "column_groups": {
                    "series": ["dataset", "turbine_id", "timestamp", "target_kw", *past_covariate_columns],
                    "past_covariates": list(past_covariate_columns),
                    "target_derived_covariates": [],
                    "known_future": ["dataset", "timestamp"],
                    "static": [],
                    "audit": [],
                },
            }
        ),
        encoding="utf-8",
    )
    pl.DataFrame(
        {
            "dataset": [dataset_id for _ in timestamps],
            "timestamp": timestamps,
            "calendar_hour_sin": [0.0] * len(timestamps),
            "calendar_hour_cos": [1.0] * len(timestamps),
            "calendar_weekday_sin": [0.0] * len(timestamps),
            "calendar_weekday_cos": [1.0] * len(timestamps),
            "calendar_month_sin": [0.0] * len(timestamps),
            "calendar_month_cos": [1.0] * len(timestamps),
            "calendar_is_weekend": [0] * len(timestamps),
        }
    ).write_parquet(task_dir / "known_future.parquet")

    if task_static_layout == "complete":
        task_static = pl.DataFrame(
            {
                "dataset": [dataset_id] * len(turbine_ids),
                "turbine_id": list(turbine_ids),
                "turbine_index": [0, 1, 2],
                "rated_power_kw": [2050.0, 2050.0, 2050.0],
                "coord_x": [0.0, 100.0, 210.0],
                "coord_y": [0.0, 0.0, 0.0],
                "latitude": [None, None, None],
                "longitude": [None, None, None],
            }
        )
    elif task_static_layout == "missing_rated_power":
        task_static = pl.DataFrame(
            {
                "dataset": [dataset_id] * len(turbine_ids),
                "turbine_id": list(turbine_ids),
                "turbine_index": [0, 1, 2],
                "coord_x": [0.0, 100.0, 210.0],
                "coord_y": [0.0, 0.0, 0.0],
                "latitude": [None, None, None],
                "longitude": [None, None, None],
            }
        )
    elif task_static_layout == "missing_coordinates":
        task_static = pl.DataFrame(
            {
                "dataset": [dataset_id] * len(turbine_ids),
                "turbine_id": list(turbine_ids),
                "turbine_index": [0, 1, 2],
                "rated_power_kw": [2050.0, 2050.0, 2050.0],
                "coord_x": [None, None, None],
                "coord_y": [None, None, None],
                "latitude": [None, None, None],
                "longitude": [None, None, None],
            }
        )
    else:
        raise ValueError(f"Unsupported task_static_layout {task_static_layout!r}.")

    task_context = json.loads((task_dir / "task_context.json").read_text(encoding="utf-8"))
    task_context["column_groups"]["static"] = task_static.columns
    (task_dir / "task_context.json").write_text(json.dumps(task_context), encoding="utf-8")
    task_static.write_parquet(task_dir / "static.parquet")


def _read_temp_bundle(
    cache_root: Path,
    *,
    dataset_id: str = "toy_dataset",
    feature_protocol_id: str = "power_only",
) -> SimpleNamespace:
    task_dir = cache_root / dataset_id / "tasks" / "next_6h_from_24h" / feature_protocol_id
    return SimpleNamespace(
        series=pl.read_parquet(task_dir / "series.parquet"),
        known_future=pl.read_parquet(task_dir / "known_future.parquet"),
        static=pl.read_parquet(task_dir / "static.parquet"),
        window_index=pl.read_parquet(task_dir / "window_index.parquet"),
        task_context=json.loads((task_dir / "task_context.json").read_text(encoding="utf-8")),
    )


def _patch_temp_bundle_loader(
    monkeypatch,
    module,
    cache_root: Path,
    *,
    dataset_id: str = "toy_dataset",
    feature_protocol_id: str = "power_only",
) -> None:
    def _fake_load_task_bundle(requested_dataset_id: str, *, feature_protocol_id: str, cache_root: str | Path):
        assert requested_dataset_id == dataset_id
        assert feature_protocol_id == expected_feature_protocol_id
        assert Path(cache_root) == cache_root_path
        return _read_temp_bundle(
            cache_root_path,
            dataset_id=dataset_id,
            feature_protocol_id=feature_protocol_id,
        )

    cache_root_path = cache_root
    expected_feature_protocol_id = feature_protocol_id
    monkeypatch.setattr(module, "_load_task_bundle", _fake_load_task_bundle)


def _patch_temp_bundle_loader_for_protocols(
    monkeypatch,
    module,
    cache_root: Path,
    *,
    dataset_id: str = "toy_dataset",
    feature_protocol_ids: tuple[str, ...],
) -> None:
    def _fake_load_task_bundle(requested_dataset_id: str, *, feature_protocol_id: str, cache_root: str | Path):
        assert requested_dataset_id == dataset_id
        assert feature_protocol_id in allowed_feature_protocol_ids
        assert Path(cache_root) == cache_root_path
        return _read_temp_bundle(
            cache_root_path,
            dataset_id=dataset_id,
            feature_protocol_id=feature_protocol_id,
        )

    allowed_feature_protocol_ids = set(feature_protocol_ids)
    cache_root_path = cache_root
    monkeypatch.setattr(module, "_load_task_bundle", _fake_load_task_bundle)


def _descriptor(module, *, target_indices, forecast_steps: int, step_us: int = 600_000_000):
    target_indices_array = np.asarray(target_indices, dtype=np.int32)
    output_start_us = target_indices_array.astype(np.int64) * step_us
    output_end_us = output_start_us + (forecast_steps - 1) * step_us
    return module.FarmWindowDescriptorIndex(
        target_indices=target_indices_array,
        output_start_us=output_start_us,
        output_end_us=output_end_us,
    )


def _window_descriptor_signature(descriptor) -> tuple[list[int], list[int], list[int]]:
    return (
        descriptor.target_indices.tolist(),
        descriptor.output_start_us.tolist(),
        descriptor.output_end_us.tolist(),
    )


def _small_prepared_dataset(
    module,
    *,
    dataset_id: str = "kelmarsh",
    forecast_steps: int = 36,
    model_variant: str | None = None,
    feature_protocol_id: str | None = None,
    input_channels: int = 1,
):
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
    for channel_index in range(1, input_channels):
        source_channels.append(target_pu + (channel_index * 0.05))
    return module.PreparedDataset(
        dataset_id=dataset_id,
        model_variant=model_variant or module.MODEL_VARIANT,
        feature_protocol_id=feature_protocol_id or module.FEATURE_PROTOCOL_ID,
        resolution_minutes=10,
        rated_power_kw=2050.0,
        history_steps=144,
        forecast_steps=forecast_steps,
        stride_steps=1,
        turbine_ids=("T01", "T02", "T03"),
        coordinate_mode="coord_xy",
        node_count=3,
        timestamps_us=timestamps_us,
        input_channel_names=tuple(
            ["target_pu", *[f"covariate_{index}" for index in range(1, input_channels)]]
        ),
        source_tensor=np.stack(source_channels, axis=-1).astype(np.float32, copy=False),
        target_pu=target_pu,
        train_windows=_descriptor(module, target_indices=[200, 201, 202, 203], forecast_steps=forecast_steps),
        val_rolling_windows=_descriptor(module, target_indices=[300, 301], forecast_steps=forecast_steps),
        val_non_overlap_windows=_descriptor(module, target_indices=[300], forecast_steps=forecast_steps),
        test_rolling_windows=_descriptor(module, target_indices=[360, 361, 362], forecast_steps=forecast_steps),
        test_non_overlap_windows=_descriptor(module, target_indices=[360], forecast_steps=forecast_steps),
    )


def _evaluation_metrics(module, *, window_count: int, forecast_steps: int, node_count: int, base: float) -> object:
    horizon_window_count = np.full((forecast_steps,), window_count, dtype=np.int64)
    horizon_prediction_count = np.full((forecast_steps,), window_count * node_count, dtype=np.int64)
    horizon_values = np.asarray([base + 0.001 * (lead + 1) for lead in range(forecast_steps)], dtype=np.float64)
    return module.EvaluationMetrics(
        window_count=window_count,
        prediction_count=window_count * forecast_steps * node_count,
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
    split_name: str,
    eval_protocol: str,
    metric_scope: str,
    lead_step: int | None,
    model_variant: str | None = None,
    input_channels: int | None = None,
) -> dict[str, object]:
    lead_minutes = None if lead_step is None else lead_step * 10
    return {
        "dataset_id": dataset_id,
        "model_id": module.MODEL_ID,
        "model_variant": model_variant or module.MODEL_VARIANT,
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
        "window_count": 10,
        "prediction_count": 360 if metric_scope == module.OVERALL_METRIC_SCOPE else 30,
        "start_timestamp": "2024-01-01 00:00:00",
        "end_timestamp": "2024-01-02 00:00:00",
        "mae_kw": 1.0,
        "rmse_kw": 1.1,
        "mae_pu": 0.1,
        "rmse_pu": 0.11,
        "graph_mode": module.GRAPH_MODE,
        "graph_source": module.GRAPH_SOURCE,
        "coordinate_mode": "coord_xy",
        "node_count": 3,
        "input_channels": input_channels or module.INPUT_CHANNELS,
        "hidden_dim": module.DEFAULT_HIDDEN_DIM,
        "embed_dim": module.DEFAULT_EMBED_DIM,
        "num_layers": module.DEFAULT_NUM_LAYERS,
        "cheb_k": module.DEFAULT_CHEB_K,
        "grad_clip_norm": module.DEFAULT_GRAD_CLIP_NORM,
        "device": "cpu",
        "runtime_seconds": 1.0,
        "train_window_count": 70,
        "val_window_count": 10,
        "test_window_count": 20,
        "best_epoch": 2,
        "epochs_ran": 3,
        "best_val_rmse_pu": 0.09,
        "seed": 42,
        "batch_size": module.DEFAULT_BATCH_SIZE,
        "learning_rate": 1e-3,
    }


def _require_torch(module):
    if module.torch is None:
        pytest.skip("PyTorch is unavailable in the root dataset environment.")
    return module.torch


def _build_reference_classes(torch_module):
    nn_module = torch_module.nn
    functional = torch_module.nn.functional

    class ReferenceAVWGCN(nn_module.Module):
        def __init__(self, dim_in, dim_out, cheb_k, embed_dim) -> None:
            super().__init__()
            self.cheb_k = cheb_k
            self.weights_pool = nn_module.Parameter(
                torch_module.empty(embed_dim, cheb_k, dim_in, dim_out)
            )
            self.bias_pool = nn_module.Parameter(torch_module.empty(embed_dim, dim_out))

        def forward(self, x, node_embeddings):
            node_num = node_embeddings.shape[0]
            supports = functional.softmax(
                functional.relu(torch_module.mm(node_embeddings, node_embeddings.transpose(0, 1))),
                dim=1,
            )
            support_set = [
                torch_module.eye(node_num, device=supports.device, dtype=supports.dtype),
                supports,
            ]
            for _ in range(2, self.cheb_k):
                support_set.append(torch_module.matmul(2 * supports, support_set[-1]) - support_set[-2])
            supports = torch_module.stack(support_set, dim=0)
            weights = torch_module.einsum("nd,dkio->nkio", node_embeddings, self.weights_pool)
            bias = torch_module.matmul(node_embeddings, self.bias_pool)
            x_g = torch_module.einsum("knm,bmc->bknc", supports, x)
            x_g = x_g.permute(0, 2, 1, 3)
            return torch_module.einsum("bnki,nkio->bno", x_g, weights) + bias

    class ReferenceAGCRNCell(nn_module.Module):
        def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim) -> None:
            super().__init__()
            self.node_num = node_num
            self.hidden_dim = dim_out
            self.gate = ReferenceAVWGCN(dim_in + self.hidden_dim, 2 * dim_out, cheb_k, embed_dim)
            self.update = ReferenceAVWGCN(dim_in + self.hidden_dim, dim_out, cheb_k, embed_dim)

        def forward(self, x, state, node_embeddings):
            state = state.to(device=x.device, dtype=x.dtype)
            input_and_state = torch_module.cat((x, state), dim=-1)
            z_r = torch_module.sigmoid(self.gate(input_and_state, node_embeddings))
            z, r = torch_module.split(z_r, self.hidden_dim, dim=-1)
            candidate = torch_module.cat((x, z * state), dim=-1)
            hc = torch_module.tanh(self.update(candidate, node_embeddings))
            return r * state + (1.0 - r) * hc

        def init_hidden_state(self, batch_size):
            return torch_module.zeros(batch_size, self.node_num, self.hidden_dim)

    class ReferenceAVWDCRNN(nn_module.Module):
        def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1) -> None:
            super().__init__()
            self.node_num = node_num
            self.input_dim = dim_in
            self.num_layers = num_layers
            self.dcrnn_cells = nn_module.ModuleList()
            self.dcrnn_cells.append(ReferenceAGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
            for _ in range(1, num_layers):
                self.dcrnn_cells.append(ReferenceAGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

        def forward(self, x, init_state, node_embeddings):
            current_inputs = x
            output_hidden = []
            for layer_index in range(self.num_layers):
                state = init_state[layer_index]
                inner_states = []
                for time_index in range(x.shape[1]):
                    state = self.dcrnn_cells[layer_index](
                        current_inputs[:, time_index, :, :],
                        state,
                        node_embeddings,
                    )
                    inner_states.append(state)
                output_hidden.append(state)
                current_inputs = torch_module.stack(inner_states, dim=1)
            return current_inputs, torch_module.stack(output_hidden, dim=0)

        def init_hidden(self, batch_size):
            init_states = []
            for layer_index in range(self.num_layers):
                init_states.append(self.dcrnn_cells[layer_index].init_hidden_state(batch_size))
            return torch_module.stack(init_states, dim=0)

    class ReferenceAGCRN(nn_module.Module):
        def __init__(
            self,
            *,
            node_count,
            input_channels,
            hidden_dim,
            forecast_steps,
            embed_dim,
            num_layers,
            cheb_k,
            output_channels=1,
        ) -> None:
            super().__init__()
            self.num_node = node_count
            self.input_dim = input_channels
            self.hidden_dim = hidden_dim
            self.output_dim = output_channels
            self.horizon = forecast_steps
            self.num_layers = num_layers
            self.node_embeddings = nn_module.Parameter(
                torch_module.randn(self.num_node, embed_dim),
                requires_grad=True,
            )
            self.encoder = ReferenceAVWDCRNN(
                self.num_node,
                self.input_dim,
                self.hidden_dim,
                cheb_k,
                embed_dim,
                self.num_layers,
            )
            self.end_conv = nn_module.Conv2d(
                1,
                self.horizon * self.output_dim,
                kernel_size=(1, self.hidden_dim),
                bias=True,
            )

        def forward(self, source, targets=None, teacher_forcing_ratio=0.5):
            del targets, teacher_forcing_ratio
            init_state = self.encoder.init_hidden(source.shape[0]).to(device=source.device, dtype=source.dtype)
            output, _ = self.encoder(source, init_state, self.node_embeddings)
            output = output[:, -1:, :, :]
            output = self.end_conv(output)
            output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
            return output.permute(0, 1, 3, 2)

    return ReferenceAVWGCN, ReferenceAGCRNCell, ReferenceAGCRN


def test_resolve_static_coordinate_columns_prefers_xy_then_latlon() -> None:
    module = _load_module()

    with_xy = pl.DataFrame(
        {
            "coord_x": [0.0, 1.0],
            "coord_y": [2.0, 3.0],
            "latitude": [10.0, 11.0],
            "longitude": [20.0, 21.0],
        }
    )
    with_latlon = pl.DataFrame(
        {
            "coord_x": [None, None],
            "coord_y": [None, None],
            "latitude": [10.0, 11.0],
            "longitude": [20.0, 21.0],
        }
    )

    assert module.resolve_static_coordinate_columns(with_xy) == ("coord_x", "coord_y")
    assert module.resolve_static_coordinate_columns(with_latlon) == ("latitude", "longitude")


def test_build_distance_sanity_frame_respects_requested_order() -> None:
    module = _load_module()
    turbine_static = pl.DataFrame(
        {
            "turbine_id": ["T01", "T02", "T03"],
            "coord_x": [0.0, 100.0, 250.0],
            "coord_y": [0.0, 0.0, 0.0],
            "latitude": [None, None, None],
            "longitude": [None, None, None],
        }
    )

    coordinate_mode, frame = module.build_distance_sanity_frame(
        turbine_static,
        ordered_turbine_ids=("T02", "T03", "T01"),
    )

    assert coordinate_mode == "coord_xy"
    assert frame["turbine_id"].to_list() == ["T02", "T03", "T01"]
    assert frame["nearest_turbine_id"].to_list() == ["T01", "T02", "T02"]


def test_thin_non_overlap_window_index_keeps_every_forecast_step() -> None:
    module = _load_module()
    windows = _descriptor(module, target_indices=list(range(10)), forecast_steps=3, step_us=1)

    thinned = module.thin_non_overlap_window_index(windows, forecast_steps=3)

    assert thinned.target_indices.tolist() == [0, 3, 6, 9]


def test_prepare_dataset_reads_farm_cache_and_builds_dual_eval_windows(tmp_path, monkeypatch) -> None:
    module = _load_module()
    cache_root = tmp_path / "cache"
    _build_temp_cache(cache_root)
    assert not (cache_root / "toy_dataset" / "silver").exists()
    _patch_temp_bundle_loader(monkeypatch, module, cache_root)

    prepared = module.prepare_dataset("toy_dataset", cache_root=cache_root)

    assert prepared.dataset_id == "toy_dataset"
    assert prepared.resolution_minutes == 10
    assert prepared.rated_power_kw == 2050.0
    assert prepared.coordinate_mode == "coord_xy"
    assert prepared.node_count == 3
    assert prepared.input_channels == 1
    assert prepared.target_pu.shape == (2000, 3)
    assert prepared.source_tensor.shape == (2000, 3, 1)
    assert len(prepared.train_windows) == 1221
    assert len(prepared.val_rolling_windows) == 21
    assert len(prepared.val_non_overlap_windows) == 1
    assert len(prepared.test_rolling_windows) == 221
    assert len(prepared.test_non_overlap_windows) == 7
    assert prepared.val_non_overlap_windows.target_indices.tolist() == [1544]
    assert prepared.test_non_overlap_windows.target_indices.tolist() == [1744, 1780, 1816, 1852, 1888, 1924, 1960]
    assert not (cache_root / "toy_dataset" / "silver").exists()


def test_prepare_dataset_applies_train_and_eval_limits(tmp_path, monkeypatch) -> None:
    module = _load_module()
    cache_root = tmp_path / "cache"
    _build_temp_cache(cache_root)
    _patch_temp_bundle_loader(monkeypatch, module, cache_root)

    prepared = module.prepare_dataset(
        "toy_dataset",
        cache_root=cache_root,
        max_train_origins=5,
        max_eval_origins=4,
    )

    assert len(prepared.train_windows) == 5
    assert len(prepared.val_rolling_windows) == 4
    assert len(prepared.val_non_overlap_windows) == 1
    assert len(prepared.test_rolling_windows) == 4
    assert len(prepared.test_non_overlap_windows) == 1


def test_prepare_dataset_builds_multichannel_source_tensor_for_power_ws_hist(tmp_path, monkeypatch) -> None:
    module = _load_module()
    cache_root = tmp_path / "cache"
    _build_temp_cache(
        cache_root,
        feature_protocol_id="power_ws_hist",
        past_covariate_columns=("Wind speed (m/s)",),
    )
    _patch_temp_bundle_loader(monkeypatch, module, cache_root, feature_protocol_id="power_ws_hist")

    prepared = module.prepare_dataset(
        "toy_dataset",
        variant_spec=module.resolve_variant_specs((module.POWER_WS_HIST_MODEL_VARIANT,))[0],
        cache_root=cache_root,
    )

    assert prepared.model_variant == module.POWER_WS_HIST_MODEL_VARIANT
    assert prepared.feature_protocol_id == module.POWER_WS_HIST_FEATURE_PROTOCOL_ID
    assert prepared.input_channel_names == ("target_pu", "Wind speed (m/s)")
    assert prepared.input_channels == 2
    assert prepared.source_tensor.shape == (2000, 3, 2)


def test_prepare_dataset_drops_windows_with_input_feature_quality_issues(tmp_path, monkeypatch) -> None:
    module = _load_module()
    cache_root = tmp_path / "cache"
    _build_temp_cache(
        cache_root,
        feature_protocol_id="power_ws_hist",
        past_covariate_columns=("Wind speed (m/s)",),
        missing_past_covariate_indices=(1563,),
    )
    _patch_temp_bundle_loader(monkeypatch, module, cache_root, feature_protocol_id="power_ws_hist")

    prepared = module.prepare_dataset(
        "toy_dataset",
        variant_spec=module.resolve_variant_specs((module.POWER_WS_HIST_MODEL_VARIANT,))[0],
        cache_root=cache_root,
    )

    assert len(prepared.val_rolling_windows) == 20
    assert len(prepared.val_non_overlap_windows) == 1


def test_prepare_dataset_single_variant_keeps_own_windows_even_when_other_variant_is_stricter(
    tmp_path,
    monkeypatch,
) -> None:
    module = _load_module()
    cache_root = tmp_path / "cache"
    _build_temp_cache(cache_root, feature_protocol_id="power_only")
    _build_temp_cache(
        cache_root,
        feature_protocol_id="power_ws_hist",
        past_covariate_columns=("Wind speed (m/s)",),
        missing_past_covariate_indices=(1563,),
    )
    _patch_temp_bundle_loader_for_protocols(
        monkeypatch,
        module,
        cache_root,
        feature_protocol_ids=("power_only", "power_ws_hist"),
    )

    power_only = module.prepare_dataset("toy_dataset", cache_root=cache_root)
    power_ws_hist = module.prepare_dataset(
        "toy_dataset",
        variant_spec=module.resolve_variant_specs((module.POWER_WS_HIST_MODEL_VARIANT,))[0],
        cache_root=cache_root,
    )

    assert len(power_only.val_rolling_windows) == 21
    assert len(power_ws_hist.val_rolling_windows) == 20


def test_prepare_variant_datasets_aligns_multi_variant_windows_to_shared_strict_subset(
    tmp_path,
    monkeypatch,
) -> None:
    module = _load_module()
    cache_root = tmp_path / "cache"
    _build_temp_cache(cache_root, feature_protocol_id="power_only")
    _build_temp_cache(
        cache_root,
        feature_protocol_id="power_ws_hist",
        past_covariate_columns=("Wind speed (m/s)",),
        missing_past_covariate_indices=(200, 1563, 1963),
    )
    _patch_temp_bundle_loader_for_protocols(
        monkeypatch,
        module,
        cache_root,
        feature_protocol_ids=("power_only", "power_ws_hist"),
    )

    power_only_direct = module.prepare_dataset("toy_dataset", cache_root=cache_root)
    power_ws_direct = module.prepare_dataset(
        "toy_dataset",
        variant_spec=module.resolve_variant_specs((module.POWER_WS_HIST_MODEL_VARIANT,))[0],
        cache_root=cache_root,
    )
    power_only_aligned, power_ws_aligned = module._prepare_datasets_for_variants(
        "toy_dataset",
        variant_specs=module.resolve_variant_specs(),
        cache_root=cache_root,
    )

    assert len(power_only_direct.train_windows) > len(power_only_aligned.train_windows)
    assert len(power_only_direct.val_rolling_windows) > len(power_only_aligned.val_rolling_windows)
    assert len(power_only_direct.test_rolling_windows) > len(power_only_aligned.test_rolling_windows)
    assert len(power_ws_direct.train_windows) == len(power_ws_aligned.train_windows)
    assert len(power_ws_direct.val_rolling_windows) == len(power_ws_aligned.val_rolling_windows)
    assert len(power_ws_direct.test_rolling_windows) == len(power_ws_aligned.test_rolling_windows)

    for attribute in (
        "train_windows",
        "val_rolling_windows",
        "val_non_overlap_windows",
        "test_rolling_windows",
        "test_non_overlap_windows",
    ):
        assert _window_descriptor_signature(getattr(power_only_aligned, attribute)) == _window_descriptor_signature(
            getattr(power_ws_aligned, attribute)
        )
        assert _window_descriptor_signature(getattr(power_ws_direct, attribute)) == _window_descriptor_signature(
            getattr(power_ws_aligned, attribute)
        )


def test_prepare_variant_datasets_requires_shared_panel_axis(tmp_path, monkeypatch) -> None:
    module = _load_module()
    cache_root = tmp_path / "cache"
    _build_temp_cache(cache_root, feature_protocol_id="power_only")
    _build_temp_cache(
        cache_root,
        feature_protocol_id="power_ws_hist",
        past_covariate_columns=("Wind speed (m/s)",),
    )
    _patch_temp_bundle_loader_for_protocols(
        monkeypatch,
        module,
        cache_root,
        feature_protocol_ids=("power_only", "power_ws_hist"),
    )

    series_path = cache_root / "toy_dataset" / "tasks" / "next_6h_from_24h" / "power_ws_hist" / "series.parquet"
    series = pl.read_parquet(series_path)
    last_timestamp = series["timestamp"].max()
    assert last_timestamp is not None
    series.filter(pl.col("timestamp") < last_timestamp).write_parquet(series_path)

    with pytest.raises(ValueError, match="do not share raw_timestamps"):
        module._prepare_datasets_for_variants(
            "toy_dataset",
            variant_specs=module.resolve_variant_specs(),
            cache_root=cache_root,
        )


def test_prepare_variant_datasets_rejects_empty_shared_split(tmp_path, monkeypatch) -> None:
    module = _load_module()
    cache_root = tmp_path / "cache"
    _build_temp_cache(cache_root, feature_protocol_id="power_only")
    _build_temp_cache(
        cache_root,
        feature_protocol_id="power_ws_hist",
        past_covariate_columns=("Wind speed (m/s)",),
    )
    _patch_temp_bundle_loader_for_protocols(
        monkeypatch,
        module,
        cache_root,
        feature_protocol_ids=("power_only", "power_ws_hist"),
    )

    reference_context = module._load_variant_dataset_context(
        "toy_dataset",
        variant_spec=module.resolve_variant_specs((module.MODEL_VARIANT,))[0],
        cache_root=cache_root,
    )
    val_keys = (
        module._split_context_window_index(reference_context)["val"]
        .select(["output_start_ts", "output_end_ts"])
        .with_row_index("align_idx")
    )
    for feature_protocol_id, keep_remainder in (("power_only", 0), ("power_ws_hist", 1)):
        window_index_path = (
            cache_root / "toy_dataset" / "tasks" / "next_6h_from_24h" / feature_protocol_id / "window_index.parquet"
        )
        window_index = pl.read_parquet(window_index_path)
        filtered = (
            window_index
            .join(val_keys, on=["output_start_ts", "output_end_ts"], how="left")
            .filter(pl.col("align_idx").is_null() | (pl.col("align_idx") % 2 == keep_remainder))
            .drop("align_idx")
        )
        filtered.write_parquet(window_index_path)

    with pytest.raises(ValueError, match="no shared strict windows for split 'val'.*power_only.*power_ws_hist"):
        module._prepare_datasets_for_variants(
            "toy_dataset",
            variant_specs=module.resolve_variant_specs(),
            cache_root=cache_root,
        )


def test_prepare_dataset_requires_rated_power_in_task_bundle_static(tmp_path, monkeypatch) -> None:
    module = _load_module()
    cache_root = tmp_path / "cache"
    _build_temp_cache(cache_root, task_static_layout="missing_rated_power")
    _patch_temp_bundle_loader(monkeypatch, module, cache_root)

    with pytest.raises(ValueError, match="missing non-null rated_power_kw"):
        module.prepare_dataset("toy_dataset", cache_root=cache_root)


def test_prepare_dataset_requires_coordinates_in_task_bundle_static(tmp_path, monkeypatch) -> None:
    module = _load_module()
    cache_root = tmp_path / "cache"
    _build_temp_cache(cache_root, task_static_layout="missing_coordinates")
    _patch_temp_bundle_loader(monkeypatch, module, cache_root)

    with pytest.raises(ValueError, match="coord_x/coord_y or full latitude/longitude"):
        module.prepare_dataset("toy_dataset", cache_root=cache_root)


def test_avwgcn_matches_official_reference() -> None:
    module = _load_module()
    torch_module = _require_torch(module)
    ReferenceAVWGCN, _, _ = _build_reference_classes(torch_module)

    torch_module.manual_seed(7)
    local = module.AVWGCN(dim_in=3, dim_out=5, cheb_k=module.DEFAULT_CHEB_K, embed_dim=4)
    module.initialize_official_aligned_parameters(local)
    reference = ReferenceAVWGCN(3, 5, module.DEFAULT_CHEB_K, 4)
    reference.load_state_dict(local.state_dict())
    x = torch_module.randn(2, 6, 3)
    node_embeddings = torch_module.randn(6, 4)

    torch_module.testing.assert_close(
        local(x, node_embeddings),
        reference(x, node_embeddings),
        rtol=1e-5,
        atol=1e-6,
    )


def test_agcrn_cell_matches_official_reference() -> None:
    module = _load_module()
    torch_module = _require_torch(module)
    _, ReferenceAGCRNCell, _ = _build_reference_classes(torch_module)

    torch_module.manual_seed(11)
    local = module.AGCRNCell(node_num=4, input_dim=1, hidden_dim=8, cheb_k=module.DEFAULT_CHEB_K, embed_dim=3)
    module.initialize_official_aligned_parameters(local)
    reference = ReferenceAGCRNCell(4, 1, 8, module.DEFAULT_CHEB_K, 3)
    reference.load_state_dict(local.state_dict())
    x = torch_module.randn(2, 4, 1)
    state = torch_module.randn(2, 4, 8)
    node_embeddings = torch_module.randn(4, 3)

    torch_module.testing.assert_close(
        local(x, state, node_embeddings),
        reference(x, state, node_embeddings),
        rtol=1e-5,
        atol=1e-6,
    )


def test_agcrn_matches_official_reference_and_outputs_4d() -> None:
    module = _load_module()
    torch_module = _require_torch(module)
    _, _, ReferenceAGCRN = _build_reference_classes(torch_module)

    torch_module.manual_seed(13)
    local = module.build_model(
        node_count=3,
        hidden_dim=module.DEFAULT_HIDDEN_DIM,
        embed_dim=module.DEFAULT_EMBED_DIM,
        num_layers=module.DEFAULT_NUM_LAYERS,
        cheb_k=module.DEFAULT_CHEB_K,
        forecast_steps=module.FORECAST_STEPS,
    )
    module.initialize_official_aligned_parameters(local)
    reference = ReferenceAGCRN(
        node_count=3,
        input_channels=module.INPUT_CHANNELS,
        hidden_dim=module.DEFAULT_HIDDEN_DIM,
        forecast_steps=module.FORECAST_STEPS,
        embed_dim=module.DEFAULT_EMBED_DIM,
        num_layers=module.DEFAULT_NUM_LAYERS,
        cheb_k=module.DEFAULT_CHEB_K,
        output_channels=module.OUTPUT_CHANNELS,
    )
    reference.load_state_dict(local.state_dict())
    history = torch_module.randn(2, module.HISTORY_STEPS, 3, module.INPUT_CHANNELS)
    targets = torch_module.randn(2, module.FORECAST_STEPS, 3, module.OUTPUT_CHANNELS)

    local_output = local(history, targets, teacher_forcing_ratio=0.0)
    reference_output = reference(history, targets, teacher_forcing_ratio=0.0)

    assert tuple(local_output.shape) == (2, module.FORECAST_STEPS, 3, module.OUTPUT_CHANNELS)
    torch_module.testing.assert_close(local_output, reference_output, rtol=1e-5, atol=1e-6)


def test_execute_training_job_emits_long_rows_for_all_eval_protocols(monkeypatch) -> None:
    module = _load_module()
    prepared = _small_prepared_dataset(
        module,
        model_variant=module.POWER_WS_HIST_MODEL_VARIANT,
        feature_protocol_id=module.POWER_WS_HIST_FEATURE_PROTOCOL_ID,
        input_channels=2,
    )

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
        return _evaluation_metrics(
            module,
            window_count=len(loader),
            forecast_steps=forecast_steps,
            node_count=prepared.node_count,
            base=len(loader) / 100.0,
        )

    monkeypatch.setattr(module, "train_model", _fake_train_model)
    monkeypatch.setattr(module, "_build_dataloader", _fake_loader)
    monkeypatch.setattr(module, "evaluate_model", _fake_evaluate)

    rows = module.execute_training_job(prepared)

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
    assert all(row["graph_mode"] == module.GRAPH_MODE for row in rows)
    assert all(row["model_variant"] == module.POWER_WS_HIST_MODEL_VARIANT for row in rows)
    assert all(row["node_count"] == 3 for row in rows)
    assert all(row["input_channels"] == 2 for row in rows)
    assert all(row["train_window_count"] == 4 for row in rows)
    assert all(row["val_window_count"] == 2 for row in rows)
    assert all(row["test_window_count"] == 3 for row in rows)


def test_sort_result_frame_orders_long_results() -> None:
    module = _load_module()
    frame = pl.DataFrame(
        [
            _result_row(
                module,
                dataset_id="kelmarsh",
                split_name="test",
                eval_protocol=module.NON_OVERLAP_EVAL_PROTOCOL,
                metric_scope=module.HORIZON_METRIC_SCOPE,
                lead_step=2,
            ),
            _result_row(
                module,
                dataset_id="kelmarsh",
                split_name="val",
                eval_protocol=module.ROLLING_EVAL_PROTOCOL,
                metric_scope=module.OVERALL_METRIC_SCOPE,
                lead_step=None,
            ),
            _result_row(
                module,
                dataset_id="kelmarsh",
                split_name="val",
                eval_protocol=module.ROLLING_EVAL_PROTOCOL,
                metric_scope=module.OVERALL_METRIC_SCOPE,
                lead_step=None,
                model_variant=module.POWER_WS_HIST_MODEL_VARIANT,
                input_channels=2,
            ),
            _result_row(
                module,
                dataset_id="kelmarsh",
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
            sorted_frame["model_variant"].to_list(),
            sorted_frame["split_name"].to_list(),
            sorted_frame["eval_protocol"].to_list(),
            sorted_frame["metric_scope"].to_list(),
            sorted_frame["lead_step"].to_list(),
            strict=True,
        )
    ) == [
        (module.MODEL_VARIANT, "val", module.ROLLING_EVAL_PROTOCOL, module.OVERALL_METRIC_SCOPE, None),
        (module.MODEL_VARIANT, "val", module.ROLLING_EVAL_PROTOCOL, module.HORIZON_METRIC_SCOPE, 1),
        (module.MODEL_VARIANT, "test", module.NON_OVERLAP_EVAL_PROTOCOL, module.HORIZON_METRIC_SCOPE, 2),
        (
            module.POWER_WS_HIST_MODEL_VARIANT,
            "val",
            module.ROLLING_EVAL_PROTOCOL,
            module.OVERALL_METRIC_SCOPE,
            None,
        ),
    ]


def test_run_experiment_aggregates_runner_rows(tmp_path) -> None:
    module = _load_module()

    def _fake_loader(dataset_id, *, variant_spec, cache_root, max_train_origins, max_eval_origins):
        del cache_root, max_train_origins, max_eval_origins
        return _small_prepared_dataset(
            module,
            dataset_id=dataset_id,
            model_variant=variant_spec.model_variant,
            feature_protocol_id=variant_spec.feature_protocol_id,
            input_channels=2 if variant_spec.feature_protocol_id == module.POWER_WS_HIST_FEATURE_PROTOCOL_ID else 1,
        )

    def _fake_runner(prepared, **kwargs):
        del kwargs
        return [
            _result_row(
                module,
                dataset_id=prepared.dataset_id,
                split_name="val",
                eval_protocol=module.ROLLING_EVAL_PROTOCOL,
                metric_scope=module.OVERALL_METRIC_SCOPE,
                lead_step=None,
                model_variant=prepared.model_variant,
                input_channels=prepared.input_channels,
            ),
            _result_row(
                module,
                dataset_id=prepared.dataset_id,
                split_name="test",
                eval_protocol=module.NON_OVERLAP_EVAL_PROTOCOL,
                metric_scope=module.HORIZON_METRIC_SCOPE,
                lead_step=1,
                model_variant=prepared.model_variant,
                input_channels=prepared.input_channels,
            ),
        ]

    output_path = tmp_path / "agcrn-official-aligned.csv"
    results = module.run_experiment(
        dataset_ids=("kelmarsh",),
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
        (module.MODEL_VARIANT, "val", module.OVERALL_METRIC_SCOPE),
        (module.MODEL_VARIANT, "test", module.HORIZON_METRIC_SCOPE),
        (module.POWER_WS_HIST_MODEL_VARIANT, "val", module.OVERALL_METRIC_SCOPE),
        (module.POWER_WS_HIST_MODEL_VARIANT, "test", module.HORIZON_METRIC_SCOPE),
    ]


def test_run_experiment_aligns_default_multi_variant_windows(tmp_path, monkeypatch) -> None:
    module = _load_module()
    cache_root = tmp_path / "cache"
    _build_temp_cache(cache_root, feature_protocol_id="power_only")
    _build_temp_cache(
        cache_root,
        feature_protocol_id="power_ws_hist",
        past_covariate_columns=("Wind speed (m/s)",),
        missing_past_covariate_indices=(200, 1563, 1963),
    )
    _patch_temp_bundle_loader_for_protocols(
        monkeypatch,
        module,
        cache_root,
        feature_protocol_ids=("power_only", "power_ws_hist"),
    )

    def _fake_runner(prepared, **kwargs):
        del kwargs
        val_row = _result_row(
            module,
            dataset_id=prepared.dataset_id,
            split_name="val",
            eval_protocol=module.ROLLING_EVAL_PROTOCOL,
            metric_scope=module.OVERALL_METRIC_SCOPE,
            lead_step=None,
            model_variant=prepared.model_variant,
            input_channels=prepared.input_channels,
        )
        val_row.update(
            {
                "window_count": len(prepared.val_rolling_windows),
                "prediction_count": len(prepared.val_rolling_windows) * prepared.forecast_steps * prepared.node_count,
                "train_window_count": len(prepared.train_windows),
                "val_window_count": len(prepared.val_rolling_windows),
                "test_window_count": len(prepared.test_rolling_windows),
            }
        )
        test_row = _result_row(
            module,
            dataset_id=prepared.dataset_id,
            split_name="test",
            eval_protocol=module.NON_OVERLAP_EVAL_PROTOCOL,
            metric_scope=module.OVERALL_METRIC_SCOPE,
            lead_step=None,
            model_variant=prepared.model_variant,
            input_channels=prepared.input_channels,
        )
        test_row.update(
            {
                "window_count": len(prepared.test_non_overlap_windows),
                "prediction_count": len(prepared.test_non_overlap_windows) * prepared.forecast_steps * prepared.node_count,
                "train_window_count": len(prepared.train_windows),
                "val_window_count": len(prepared.val_rolling_windows),
                "test_window_count": len(prepared.test_rolling_windows),
            }
        )
        return [val_row, test_row]

    results = module.run_experiment(
        dataset_ids=("toy_dataset",),
        cache_root=cache_root,
        output_path=tmp_path / "agcrn-official-aligned.csv",
        job_runner=_fake_runner,
    )

    assert results.height == 4
    assert results["train_window_count"].n_unique() == 1
    assert results["val_window_count"].n_unique() == 1
    assert results["test_window_count"].n_unique() == 1
    val_rows = results.filter(
        (pl.col("split_name") == "val")
        & (pl.col("eval_protocol") == module.ROLLING_EVAL_PROTOCOL)
    )
    test_rows = results.filter(
        (pl.col("split_name") == "test")
        & (pl.col("eval_protocol") == module.NON_OVERLAP_EVAL_PROTOCOL)
    )
    assert val_rows.height == 2
    assert test_rows.height == 2
    assert val_rows["window_count"].n_unique() == 1
    assert test_rows["window_count"].n_unique() == 1
    assert results["input_channels"].to_list() == [1, 1, 2, 2]


def test_run_experiment_updates_job_progress_bar(monkeypatch, tmp_path) -> None:
    module = _load_module()
    _TqdmRecorder.instances.clear()

    def _fake_loader(dataset_id, *, variant_spec, cache_root, max_train_origins, max_eval_origins):
        del cache_root, max_train_origins, max_eval_origins
        return _small_prepared_dataset(
            module,
            dataset_id=dataset_id,
            model_variant=variant_spec.model_variant,
            feature_protocol_id=variant_spec.feature_protocol_id,
            input_channels=2 if variant_spec.feature_protocol_id == module.POWER_WS_HIST_FEATURE_PROTOCOL_ID else 1,
        )

    def _fake_runner(prepared, **kwargs):
        del kwargs
        return [
            _result_row(
                module,
                dataset_id=prepared.dataset_id,
                split_name="val",
                eval_protocol=module.ROLLING_EVAL_PROTOCOL,
                metric_scope=module.OVERALL_METRIC_SCOPE,
                lead_step=None,
                model_variant=prepared.model_variant,
                input_channels=prepared.input_channels,
            ),
        ]

    monkeypatch.setattr(module, "tqdm", _TqdmRecorder)
    monkeypatch.setattr(module, "progress_is_enabled", lambda: True)

    module.run_experiment(
        dataset_ids=("kelmarsh",),
        output_path=tmp_path / "agcrn-official-aligned.csv",
        dataset_loader=_fake_loader,
        job_runner=_fake_runner,
    )

    assert len(_TqdmRecorder.instances) == 1
    progress = _TqdmRecorder.instances[0]
    assert progress.total == 2
    assert progress.n == 2
    assert progress.closed is True
    assert progress.desc == "agcrn jobs"
    assert any("kelmarsh" in value for value in progress.postfixes)
    assert any(module.POWER_WS_HIST_MODEL_VARIANT in value for value in progress.postfixes)


def test_run_experiment_can_limit_variants(tmp_path) -> None:
    module = _load_module()

    def _fake_loader(dataset_id, *, variant_spec, cache_root, max_train_origins, max_eval_origins):
        del cache_root, max_train_origins, max_eval_origins
        return _small_prepared_dataset(
            module,
            dataset_id=dataset_id,
            model_variant=variant_spec.model_variant,
            feature_protocol_id=variant_spec.feature_protocol_id,
            input_channels=2 if variant_spec.feature_protocol_id == module.POWER_WS_HIST_FEATURE_PROTOCOL_ID else 1,
        )

    def _fake_runner(prepared, **kwargs):
        del kwargs
        return [
            _result_row(
                module,
                dataset_id=prepared.dataset_id,
                split_name="val",
                eval_protocol=module.ROLLING_EVAL_PROTOCOL,
                metric_scope=module.OVERALL_METRIC_SCOPE,
                lead_step=None,
                model_variant=prepared.model_variant,
                input_channels=prepared.input_channels,
            ),
        ]

    results = module.run_experiment(
        dataset_ids=("kelmarsh",),
        variant_names=(module.POWER_WS_HIST_MODEL_VARIANT,),
        output_path=tmp_path / "agcrn-official-aligned.csv",
        dataset_loader=_fake_loader,
        job_runner=_fake_runner,
    )

    assert results.height == 1
    assert results["model_variant"].to_list() == [module.POWER_WS_HIST_MODEL_VARIANT]
    assert results["input_channels"].to_list() == [2]


def test_run_experiment_uses_tuned_variant_defaults(tmp_path) -> None:
    module = _load_module()
    observed_kwargs: dict[str, dict[str, object]] = {}

    def _fake_loader(dataset_id, *, variant_spec, cache_root, max_train_origins, max_eval_origins):
        del cache_root, max_train_origins, max_eval_origins
        return _small_prepared_dataset(
            module,
            dataset_id=dataset_id,
            model_variant=variant_spec.model_variant,
            feature_protocol_id=variant_spec.feature_protocol_id,
            input_channels=2 if variant_spec.feature_protocol_id == module.POWER_WS_HIST_FEATURE_PROTOCOL_ID else 1,
        )

    def _fake_runner(prepared, **kwargs):
        observed_kwargs[prepared.model_variant] = kwargs
        return [
            _result_row(
                module,
                dataset_id=prepared.dataset_id,
                split_name="val",
                eval_protocol=module.ROLLING_EVAL_PROTOCOL,
                metric_scope=module.OVERALL_METRIC_SCOPE,
                lead_step=None,
                model_variant=prepared.model_variant,
                input_channels=prepared.input_channels,
            ),
        ]

    module.run_experiment(
        dataset_ids=("kelmarsh",),
        output_path=tmp_path / "agcrn-official-aligned.csv",
        dataset_loader=_fake_loader,
        job_runner=_fake_runner,
    )

    power_only_profile = module.resolve_hyperparameter_profile(module.MODEL_VARIANT)
    power_ws_hist_profile = module.resolve_hyperparameter_profile(module.POWER_WS_HIST_MODEL_VARIANT)

    assert observed_kwargs[module.MODEL_VARIANT]["batch_size"] == power_only_profile.batch_size
    assert observed_kwargs[module.MODEL_VARIANT]["learning_rate"] == power_only_profile.learning_rate
    assert observed_kwargs[module.MODEL_VARIANT]["max_epochs"] == power_only_profile.max_epochs
    assert observed_kwargs[module.MODEL_VARIANT]["early_stopping_patience"] == power_only_profile.early_stopping_patience
    assert observed_kwargs[module.MODEL_VARIANT]["hidden_dim"] == power_only_profile.hidden_dim
    assert observed_kwargs[module.MODEL_VARIANT]["embed_dim"] == power_only_profile.embed_dim
    assert observed_kwargs[module.MODEL_VARIANT]["num_layers"] == power_only_profile.num_layers
    assert observed_kwargs[module.MODEL_VARIANT]["cheb_k"] == power_only_profile.cheb_k
    assert observed_kwargs[module.MODEL_VARIANT]["grad_clip_norm"] == power_only_profile.grad_clip_norm

    assert observed_kwargs[module.POWER_WS_HIST_MODEL_VARIANT]["batch_size"] == power_ws_hist_profile.batch_size
    assert observed_kwargs[module.POWER_WS_HIST_MODEL_VARIANT]["learning_rate"] == power_ws_hist_profile.learning_rate
    assert observed_kwargs[module.POWER_WS_HIST_MODEL_VARIANT]["max_epochs"] == power_ws_hist_profile.max_epochs
    assert (
        observed_kwargs[module.POWER_WS_HIST_MODEL_VARIANT]["early_stopping_patience"]
        == power_ws_hist_profile.early_stopping_patience
    )
    assert observed_kwargs[module.POWER_WS_HIST_MODEL_VARIANT]["hidden_dim"] == power_ws_hist_profile.hidden_dim
    assert observed_kwargs[module.POWER_WS_HIST_MODEL_VARIANT]["embed_dim"] == power_ws_hist_profile.embed_dim
    assert observed_kwargs[module.POWER_WS_HIST_MODEL_VARIANT]["num_layers"] == power_ws_hist_profile.num_layers
    assert observed_kwargs[module.POWER_WS_HIST_MODEL_VARIANT]["cheb_k"] == power_ws_hist_profile.cheb_k
    assert observed_kwargs[module.POWER_WS_HIST_MODEL_VARIANT]["grad_clip_norm"] == power_ws_hist_profile.grad_clip_norm


def test_run_experiment_overrides_tuned_defaults_field_by_field(tmp_path) -> None:
    module = _load_module()
    observed_kwargs: dict[str, dict[str, object]] = {}

    def _fake_loader(dataset_id, *, variant_spec, cache_root, max_train_origins, max_eval_origins):
        del cache_root, max_train_origins, max_eval_origins
        return _small_prepared_dataset(
            module,
            dataset_id=dataset_id,
            model_variant=variant_spec.model_variant,
            feature_protocol_id=variant_spec.feature_protocol_id,
            input_channels=2 if variant_spec.feature_protocol_id == module.POWER_WS_HIST_FEATURE_PROTOCOL_ID else 1,
        )

    def _fake_runner(prepared, **kwargs):
        observed_kwargs[prepared.model_variant] = kwargs
        return [
            _result_row(
                module,
                dataset_id=prepared.dataset_id,
                split_name="val",
                eval_protocol=module.ROLLING_EVAL_PROTOCOL,
                metric_scope=module.OVERALL_METRIC_SCOPE,
                lead_step=None,
                model_variant=prepared.model_variant,
                input_channels=prepared.input_channels,
            ),
        ]

    module.run_experiment(
        dataset_ids=("kelmarsh",),
        output_path=tmp_path / "agcrn-official-aligned.csv",
        batch_size=256,
        max_epochs=7,
        dataset_loader=_fake_loader,
        job_runner=_fake_runner,
    )

    assert observed_kwargs[module.MODEL_VARIANT]["batch_size"] == 256
    assert observed_kwargs[module.POWER_WS_HIST_MODEL_VARIANT]["batch_size"] == 256
    assert observed_kwargs[module.MODEL_VARIANT]["max_epochs"] == 7
    assert observed_kwargs[module.POWER_WS_HIST_MODEL_VARIANT]["max_epochs"] == 7

    assert observed_kwargs[module.MODEL_VARIANT]["embed_dim"] == 10
    assert observed_kwargs[module.MODEL_VARIANT]["cheb_k"] == 2
    assert observed_kwargs[module.POWER_WS_HIST_MODEL_VARIANT]["embed_dim"] == 16
    assert observed_kwargs[module.POWER_WS_HIST_MODEL_VARIANT]["cheb_k"] == 3

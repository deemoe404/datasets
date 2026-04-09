from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import polars as pl

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - exercised in the root env where torch is absent
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


MODEL_ID = "LTSF-Linear"
TASK_ID = "next_6h_from_24h_stride_6h"
SPLIT_PROTOCOL = "chrono_70_10_20"
DEFAULT_DATASETS = ("kelmarsh", "penmanshiel", "hill_of_towie", "sdwpf_kddcup")
MODEL_VARIANTS = ("nlinear", "dlinear")
HISTORY_STEPS = 144
FORECAST_STEPS = 36
STRIDE_STEPS = 36
DEFAULT_BATCH_SIZE = 1024
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_MAX_EPOCHS = 50
DEFAULT_EARLY_STOPPING_PATIENCE = 5
DEFAULT_SEED = 42
DLINEAR_KERNEL_SIZE = 25
QUALITY_PROFILE = "default"
SERIES_LAYOUT = "turbine"
FEATURE_SET = "default"
PROFILE_LOG_PREFIX = "[ltsf_linear] "

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CACHE_ROOT = _REPO_ROOT / "cache"
_OUTPUT_PATH = _REPO_ROOT / "experiment" / "ltsf-linear.csv"
_TASK_DIR_NAME = TASK_ID
_TASK_WINDOW_COLUMNS = (
    "dataset",
    "turbine_id",
    "output_start_ts",
    "output_end_ts",
    "is_complete_input",
    "is_complete_output",
    "quality_flags",
)
_SERIES_COLUMNS = (
    "dataset",
    "turbine_id",
    "timestamp",
    "target_kw",
    "is_observed",
    "quality_flags",
)
_RESULT_COLUMNS = [
    "dataset_id",
    "model_id",
    "model_variant",
    "task_id",
    "history_steps",
    "forecast_steps",
    "stride_steps",
    "split_protocol",
    "window_count",
    "prediction_count",
    "start_timestamp",
    "end_timestamp",
    "mae_kw",
    "rmse_kw",
    "mae_pu",
    "rmse_pu",
    "device",
    "runtime_seconds",
    "train_window_count",
    "val_window_count",
    "test_window_count",
    "best_epoch",
    "epochs_ran",
    "best_val_rmse_pu",
    "seed",
    "batch_size",
    "learning_rate",
]
_DATASET_ORDER = {dataset_id: index for index, dataset_id in enumerate(DEFAULT_DATASETS)}
_MODEL_ORDER = {model_variant: index for index, model_variant in enumerate(MODEL_VARIANTS)}


@dataclass(frozen=True)
class TaskCachePaths:
    dataset_id: str
    task_dir: Path
    window_index_path: Path
    task_context_path: Path
    series_path: Path
    turbine_static_path: Path


@dataclass(frozen=True)
class DatasetMetadata:
    dataset_id: str
    turbine_ids: tuple[str, ...]
    rated_power_kw: float
    task_paths: TaskCachePaths


@dataclass(frozen=True)
class TurbineTargetSeries:
    timestamps_us: np.ndarray
    target_pu: np.ndarray


@dataclass(frozen=True)
class SplitData:
    inputs: np.ndarray
    targets: np.ndarray
    output_start_us: np.ndarray
    output_end_us: np.ndarray


@dataclass(frozen=True)
class PreparedDataset:
    dataset_id: str
    rated_power_kw: float
    history_steps: int
    forecast_steps: int
    stride_steps: int
    train: SplitData
    val: SplitData
    test: SplitData


@dataclass(frozen=True)
class TrainingOutcome:
    best_epoch: int
    epochs_ran: int
    best_val_rmse_pu: float
    device: str
    metrics: dict[str, float | int]


def _profile_log(dataset_id: str, phase: str, **fields: object) -> None:
    payload = {"dataset_id": dataset_id, "phase": phase, **fields}
    print(
        f"{PROFILE_LOG_PREFIX}{json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True)}",
        file=sys.stderr,
        flush=True,
    )


def _ensure_repo_src_on_path() -> None:
    src_path = _REPO_ROOT / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def build_task_spec():
    _ensure_repo_src_on_path()
    from wind_datasets import TaskSpec

    return TaskSpec(
        history_duration="24h",
        forecast_duration="6h",
        stride_duration="6h",
        task_id=TASK_ID,
        granularity="turbine",
    )


def require_torch() -> tuple[Any, Any, Any, Any]:
    if torch is None or nn is None or DataLoader is None or TensorDataset is None:
        raise ImportError(
            "PyTorch is unavailable in the current environment. "
            "Create experiment/ltsf-linear/.conda with ./create_env.sh."
        )
    return torch, nn, DataLoader, TensorDataset


def select_device(torch_module: Any | None = None) -> str:
    resolved_torch = torch_module or torch
    if resolved_torch is None:
        return "cpu"
    if bool(resolved_torch.cuda.is_available()):
        return "cuda"
    mps_backend = getattr(getattr(resolved_torch, "backends", None), "mps", None)
    if mps_backend is not None and bool(mps_backend.is_available()):
        return "mps"
    return "cpu"


def resolve_device(device: str | None = None, torch_module: Any | None = None) -> str:
    if device is None or device == "auto":
        return select_device(torch_module=torch_module)
    return device


def clip_target_values(values: Sequence[float | None], rated_power_kw: float) -> np.ndarray:
    clipped = np.asarray(values, dtype=np.float32).copy()
    valid = ~np.isnan(clipped)
    clipped[valid] = np.clip(clipped[valid], 0.0, rated_power_kw)
    return clipped


def _normalize_target_values(values: Sequence[float | None], rated_power_kw: float) -> np.ndarray:
    clipped = clip_target_values(values, rated_power_kw)
    normalized = clipped.copy()
    valid = ~np.isnan(normalized)
    normalized[valid] = normalized[valid] / rated_power_kw
    return normalized


def resolve_cache_paths(dataset_id: str, *, cache_root: str | Path = _CACHE_ROOT) -> TaskCachePaths:
    dataset_root = Path(cache_root) / dataset_id
    task_dir = dataset_root / "tasks" / QUALITY_PROFILE / SERIES_LAYOUT / _TASK_DIR_NAME
    return TaskCachePaths(
        dataset_id=dataset_id,
        task_dir=task_dir,
        window_index_path=task_dir / "window_index.parquet",
        task_context_path=task_dir / "task_context.json",
        series_path=dataset_root / "gold_base" / QUALITY_PROFILE / SERIES_LAYOUT / FEATURE_SET / "series.parquet",
        turbine_static_path=dataset_root / "silver" / "meta" / "turbine_static.parquet",
    )


def ensure_task_cache(dataset_id: str, *, cache_root: str | Path = _CACHE_ROOT) -> TaskCachePaths:
    paths = resolve_cache_paths(dataset_id, cache_root=cache_root)
    required_paths = (
        paths.window_index_path,
        paths.task_context_path,
        paths.series_path,
        paths.turbine_static_path,
    )
    if all(path.exists() for path in required_paths):
        return paths

    _ensure_repo_src_on_path()
    try:
        from wind_datasets import build_task_cache
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError("Unable to import wind_datasets for task cache construction.") from exc

    try:
        build_task_cache(dataset_id, build_task_spec(), cache_root=cache_root)
    except Exception as exc:  # pragma: no cover - exercised only when cache is missing
        raise RuntimeError(
            f"Task cache for dataset {dataset_id!r} is missing and could not be rebuilt. "
            "Either prebuild the cache artifacts or configure wind_datasets.local.toml."
        ) from exc

    if not all(path.exists() for path in required_paths):
        raise RuntimeError(f"Dataset cache for {dataset_id!r} is incomplete after rebuild.")
    return paths


def load_dataset_metadata(dataset_id: str, *, cache_root: str | Path = _CACHE_ROOT) -> DatasetMetadata:
    paths = ensure_task_cache(dataset_id, cache_root=cache_root)
    task_context = json.loads(paths.task_context_path.read_text(encoding="utf-8"))
    task = task_context.get("task", {})
    if (
        int(task.get("history_steps", HISTORY_STEPS)) != HISTORY_STEPS
        or int(task.get("forecast_steps", FORECAST_STEPS)) != FORECAST_STEPS
        or int(task.get("stride_steps", STRIDE_STEPS)) != STRIDE_STEPS
    ):
        raise ValueError(
            f"Cached task context for dataset {dataset_id!r} does not match the expected "
            f"{HISTORY_STEPS}/{FORECAST_STEPS}/{STRIDE_STEPS} task."
        )

    turbine_static = pl.read_parquet(paths.turbine_static_path).select(["turbine_id", "rated_power_kw"])
    rated_powers = sorted(
        {
            float(value)
            for value in turbine_static["rated_power_kw"].drop_nulls().to_list()
        }
    )
    if len(rated_powers) != 1:
        raise ValueError(f"Dataset {dataset_id!r} must have a single rated_power_kw, found {rated_powers!r}.")
    return DatasetMetadata(
        dataset_id=dataset_id,
        turbine_ids=tuple(task_context["turbine_ids"]),
        rated_power_kw=rated_powers[0],
        task_paths=paths,
    )


def load_target_series_frame(dataset_id: str, *, cache_root: str | Path = _CACHE_ROOT) -> pl.DataFrame:
    paths = ensure_task_cache(dataset_id, cache_root=cache_root)
    load_started = time.monotonic()
    frame = (
        pl.scan_parquet(paths.series_path)
        .select(list(_SERIES_COLUMNS))
        .sort(["turbine_id", "timestamp"])
        .collect()
    )
    _profile_log(
        dataset_id,
        "load_target_series",
        rows=frame.height,
        columns=len(frame.columns),
        duration_seconds=round(time.monotonic() - load_started, 6),
    )
    return frame


def load_strict_window_index(dataset_id: str, *, cache_root: str | Path = _CACHE_ROOT) -> pl.DataFrame:
    paths = ensure_task_cache(dataset_id, cache_root=cache_root)
    load_started = time.monotonic()
    frame = (
        pl.scan_parquet(paths.window_index_path)
        .select(list(_TASK_WINDOW_COLUMNS))
        .filter(
            pl.col("is_complete_input")
            & pl.col("is_complete_output")
            & (pl.col("quality_flags").fill_null("") == "")
        )
        .sort(["output_start_ts", "turbine_id"])
        .collect()
    )
    _profile_log(
        dataset_id,
        "load_window_index",
        strict_windows=frame.height,
        duration_seconds=round(time.monotonic() - load_started, 6),
    )
    if frame.is_empty():
        raise ValueError(f"Dataset {dataset_id!r} has no strict windows for {TASK_ID}.")
    return frame


def build_chrono_split_lookup(output_start_timestamps: Sequence[datetime]) -> pl.DataFrame:
    unique_sorted = sorted(dict.fromkeys(output_start_timestamps))
    total = len(unique_sorted)
    train_count = math.floor(total * 0.7)
    val_count = math.floor(total * 0.1)
    test_count = total - train_count - val_count
    if min(train_count, val_count, test_count) <= 0:
        raise ValueError(
            f"Chronological split {SPLIT_PROTOCOL!r} requires non-empty train/val/test, found "
            f"{train_count}/{val_count}/{test_count}."
        )
    return pl.DataFrame(
        {
            "output_start_ts": unique_sorted,
            "split": (
                ["train"] * train_count
                + ["val"] * val_count
                + ["test"] * test_count
            ),
        }
    )


def split_window_index(
    window_index: pl.DataFrame,
    *,
    max_windows_per_split: int | None = None,
) -> dict[str, pl.DataFrame]:
    split_lookup = build_chrono_split_lookup(window_index["output_start_ts"].to_list())
    frames = (
        window_index.join(split_lookup, on="output_start_ts", how="inner")
        .sort(["output_start_ts", "turbine_id"])
    )
    split_frames: dict[str, pl.DataFrame] = {}
    for split_name in ("train", "val", "test"):
        split_frame = frames.filter(pl.col("split") == split_name)
        if max_windows_per_split is not None:
            split_frame = split_frame.head(max_windows_per_split)
        if split_frame.is_empty():
            raise ValueError(f"Split {split_name!r} is empty after window selection.")
        split_frames[split_name] = split_frame
    return split_frames


def build_turbine_series_map(
    series: pl.DataFrame,
    *,
    rated_power_kw: float,
) -> dict[str, TurbineTargetSeries]:
    turbines: dict[str, TurbineTargetSeries] = {}
    for turbine_frame in series.partition_by("turbine_id", maintain_order=True):
        turbine_id = turbine_frame["turbine_id"][0]
        turbines[turbine_id] = TurbineTargetSeries(
            timestamps_us=turbine_frame["timestamp"].cast(pl.Int64).to_numpy(),
            target_pu=_normalize_target_values(turbine_frame["target_kw"].to_list(), rated_power_kw),
        )
    return turbines


def build_split_samples(
    window_index: pl.DataFrame,
    turbine_series_map: Mapping[str, TurbineTargetSeries],
    *,
    history_steps: int,
    forecast_steps: int,
) -> SplitData:
    sample_inputs: list[np.ndarray] = []
    sample_targets: list[np.ndarray] = []
    output_start_values: list[int] = []
    output_end_values: list[int] = []

    working = window_index.with_columns(
        pl.col("output_start_ts").cast(pl.Int64).alias("output_start_us"),
        pl.col("output_end_ts").cast(pl.Int64).alias("output_end_us"),
    )
    skipped_nan_windows = 0

    for turbine_frame in working.partition_by("turbine_id", maintain_order=True):
        turbine_id = turbine_frame["turbine_id"][0]
        series = turbine_series_map[turbine_id]
        timestamp_index = {int(timestamp): index for index, timestamp in enumerate(series.timestamps_us.tolist())}
        for output_start_us, output_end_us in zip(
            turbine_frame["output_start_us"].to_list(),
            turbine_frame["output_end_us"].to_list(),
            strict=True,
        ):
            target_index = timestamp_index.get(int(output_start_us))
            if target_index is None:
                raise KeyError(
                    f"Output start timestamp {output_start_us!r} is missing for turbine {turbine_id!r}."
                )
            context = series.target_pu[target_index - history_steps : target_index]
            future = series.target_pu[target_index : target_index + forecast_steps]
            if context.shape[0] != history_steps or future.shape[0] != forecast_steps:
                raise ValueError(
                    f"Window for turbine {turbine_id!r} does not match expected history/forecast sizes."
                )
            if np.isnan(context).any() or np.isnan(future).any():
                skipped_nan_windows += 1
                continue
            sample_inputs.append(context.astype(np.float32, copy=True))
            sample_targets.append(future.astype(np.float32, copy=True))
            output_start_values.append(int(output_start_us))
            output_end_values.append(int(output_end_us))

    if not sample_inputs:
        raise ValueError("No valid samples remain after filtering NaN target windows.")

    return SplitData(
        inputs=np.stack(sample_inputs),
        targets=np.stack(sample_targets),
        output_start_us=np.asarray(output_start_values, dtype=np.int64),
        output_end_us=np.asarray(output_end_values, dtype=np.int64),
    )


def prepare_dataset(
    dataset_id: str,
    *,
    cache_root: str | Path = _CACHE_ROOT,
    max_windows_per_split: int | None = None,
) -> PreparedDataset:
    metadata = load_dataset_metadata(dataset_id, cache_root=cache_root)
    strict_window_index = load_strict_window_index(dataset_id, cache_root=cache_root)
    split_frames = split_window_index(strict_window_index, max_windows_per_split=max_windows_per_split)
    series = load_target_series_frame(dataset_id, cache_root=cache_root)
    turbine_series_map = build_turbine_series_map(series, rated_power_kw=metadata.rated_power_kw)
    split_data = {
        split_name: build_split_samples(
            split_frame,
            turbine_series_map,
            history_steps=HISTORY_STEPS,
            forecast_steps=FORECAST_STEPS,
        )
        for split_name, split_frame in split_frames.items()
    }
    _profile_log(
        dataset_id,
        "prepare_dataset_complete",
        train_windows=split_data["train"].inputs.shape[0],
        val_windows=split_data["val"].inputs.shape[0],
        test_windows=split_data["test"].inputs.shape[0],
        rated_power_kw=metadata.rated_power_kw,
    )
    return PreparedDataset(
        dataset_id=dataset_id,
        rated_power_kw=metadata.rated_power_kw,
        history_steps=HISTORY_STEPS,
        forecast_steps=FORECAST_STEPS,
        stride_steps=STRIDE_STEPS,
        train=split_data["train"],
        val=split_data["val"],
        test=split_data["test"],
    )


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn_backend = getattr(torch.backends, "cudnn", None)
    if cudnn_backend is not None:
        cudnn_backend.deterministic = True
        cudnn_backend.benchmark = False


def _initialize_linear(layer: Any) -> None:
    if layer is None:  # pragma: no cover - defensive
        return
    with torch.no_grad():
        layer.weight.fill_(1.0 / layer.in_features)
        if layer.bias is not None:
            layer.bias.zero_()


if nn is not None:

    class MovingAverage(nn.Module):
        def __init__(self, kernel_size: int) -> None:
            super().__init__()
            self.kernel_size = kernel_size
            self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

        def forward(self, x):
            padding = (self.kernel_size - 1) // 2
            front = x[:, :1].repeat(1, padding)
            back = x[:, -1:].repeat(1, padding)
            padded = torch.cat([front, x, back], dim=1)
            return self.avg(padded.unsqueeze(1)).squeeze(1)


    class SeriesDecomposition(nn.Module):
        def __init__(self, kernel_size: int) -> None:
            super().__init__()
            self.moving_average = MovingAverage(kernel_size)

        def forward(self, x):
            trend = self.moving_average(x)
            seasonal = x - trend
            return seasonal, trend


    class NLinear(nn.Module):
        def __init__(self, history_steps: int, forecast_steps: int) -> None:
            super().__init__()
            self.linear = nn.Linear(history_steps, forecast_steps)
            _initialize_linear(self.linear)

        def forward(self, x):
            last = x[:, -1:].detach()
            normalized = x - last
            return self.linear(normalized) + last


    class DLinear(nn.Module):
        def __init__(self, history_steps: int, forecast_steps: int, kernel_size: int = DLINEAR_KERNEL_SIZE) -> None:
            super().__init__()
            self.decomposition = SeriesDecomposition(kernel_size)
            self.linear_seasonal = nn.Linear(history_steps, forecast_steps)
            self.linear_trend = nn.Linear(history_steps, forecast_steps)
            _initialize_linear(self.linear_seasonal)
            _initialize_linear(self.linear_trend)

        def forward(self, x):
            seasonal, trend = self.decomposition(x)
            return self.linear_seasonal(seasonal) + self.linear_trend(trend)

else:

    class MovingAverage:  # pragma: no cover - exercised only when torch is missing
        def __init__(self, *_args, **_kwargs) -> None:
            require_torch()


    class SeriesDecomposition:  # pragma: no cover - exercised only when torch is missing
        def __init__(self, *_args, **_kwargs) -> None:
            require_torch()


    class NLinear:  # pragma: no cover - exercised only when torch is missing
        def __init__(self, *_args, **_kwargs) -> None:
            require_torch()


    class DLinear:  # pragma: no cover - exercised only when torch is missing
        def __init__(self, *_args, **_kwargs) -> None:
            require_torch()


def build_model(model_variant: str, *, history_steps: int, forecast_steps: int):
    require_torch()
    if model_variant == "nlinear":
        return NLinear(history_steps, forecast_steps)
    if model_variant == "dlinear":
        return DLinear(history_steps, forecast_steps, kernel_size=DLINEAR_KERNEL_SIZE)
    raise ValueError(f"Unsupported model_variant {model_variant!r}.")


def _build_dataloader(
    split_data: SplitData,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
):
    resolved_torch, _, resolved_loader, resolved_dataset = require_torch()
    dataset = resolved_dataset(
        resolved_torch.from_numpy(split_data.inputs),
        resolved_torch.from_numpy(split_data.targets),
    )
    generator = resolved_torch.Generator()
    generator.manual_seed(seed)
    return resolved_loader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def evaluate_model(model, loader, *, device: str, rated_power_kw: float) -> dict[str, float | int]:
    resolved_torch, _, _, _ = require_torch()
    metrics = {
        "window_count": 0,
        "prediction_count": 0,
        "abs_error_sum": 0.0,
        "squared_error_sum": 0.0,
        "normalized_abs_error_sum": 0.0,
        "normalized_squared_error_sum": 0.0,
    }
    model.eval()
    with resolved_torch.no_grad():
        for batch_inputs, batch_targets in loader:
            batch_inputs = batch_inputs.to(device=device, dtype=resolved_torch.float32)
            batch_targets = batch_targets.to(device=device, dtype=resolved_torch.float32)
            predictions = model(batch_inputs)
            errors_pu = predictions - batch_targets
            errors_kw = errors_pu * rated_power_kw
            metrics["window_count"] = int(metrics["window_count"]) + int(batch_inputs.shape[0])
            metrics["prediction_count"] = int(metrics["prediction_count"]) + int(batch_targets.numel())
            metrics["abs_error_sum"] = float(metrics["abs_error_sum"]) + float(resolved_torch.abs(errors_kw).sum().item())
            metrics["squared_error_sum"] = float(metrics["squared_error_sum"]) + float(
                resolved_torch.square(errors_kw).sum().item()
            )
            metrics["normalized_abs_error_sum"] = float(metrics["normalized_abs_error_sum"]) + float(
                resolved_torch.abs(errors_pu).sum().item()
            )
            metrics["normalized_squared_error_sum"] = float(metrics["normalized_squared_error_sum"]) + float(
                resolved_torch.square(errors_pu).sum().item()
            )
    prediction_count = int(metrics["prediction_count"])
    return {
        **metrics,
        "mae_kw": _safe_divide(float(metrics["abs_error_sum"]), prediction_count),
        "rmse_kw": _safe_rmse(float(metrics["squared_error_sum"]), prediction_count),
        "mae_pu": _safe_divide(float(metrics["normalized_abs_error_sum"]), prediction_count),
        "rmse_pu": _safe_rmse(float(metrics["normalized_squared_error_sum"]), prediction_count),
    }


def train_model(
    model_variant: str,
    prepared_dataset: PreparedDataset,
    *,
    device: str,
    seed: int = DEFAULT_SEED,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
) -> TrainingOutcome:
    resolved_torch, _, _, _ = require_torch()
    _set_random_seed(seed)
    resolved_device = resolve_device(device)
    model = build_model(
        model_variant,
        history_steps=prepared_dataset.history_steps,
        forecast_steps=prepared_dataset.forecast_steps,
    ).to(device=resolved_device)
    optimizer = resolved_torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = resolved_torch.nn.MSELoss()
    train_loader = _build_dataloader(prepared_dataset.train, batch_size=batch_size, shuffle=True, seed=seed)
    val_loader = _build_dataloader(prepared_dataset.val, batch_size=batch_size, shuffle=False, seed=seed)
    test_loader = _build_dataloader(prepared_dataset.test, batch_size=batch_size, shuffle=False, seed=seed)

    best_state: dict[str, Any] | None = None
    best_epoch = 0
    best_val_rmse_pu = float("inf")
    epochs_without_improvement = 0
    epochs_ran = 0

    for epoch_index in range(1, max_epochs + 1):
        model.train()
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device=resolved_device, dtype=resolved_torch.float32)
            batch_targets = batch_targets.to(device=resolved_device, dtype=resolved_torch.float32)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_inputs)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()

        epochs_ran = epoch_index
        val_metrics = evaluate_model(
            model,
            val_loader,
            device=resolved_device,
            rated_power_kw=prepared_dataset.rated_power_kw,
        )
        val_rmse_pu = float(val_metrics["rmse_pu"])
        if val_rmse_pu < best_val_rmse_pu - 1e-12:
            best_val_rmse_pu = val_rmse_pu
            best_epoch = epoch_index
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                break

    if best_state is None:  # pragma: no cover - defensive
        raise RuntimeError("Training completed without a best checkpoint.")

    model.load_state_dict(best_state)
    test_metrics = evaluate_model(
        model,
        test_loader,
        device=resolved_device,
        rated_power_kw=prepared_dataset.rated_power_kw,
    )
    return TrainingOutcome(
        best_epoch=best_epoch,
        epochs_ran=epochs_ran,
        best_val_rmse_pu=best_val_rmse_pu,
        device=resolved_device,
        metrics=test_metrics,
    )


def _safe_divide(numerator: float, denominator: int) -> float:
    if denominator <= 0:
        return float("nan")
    return numerator / denominator


def _safe_rmse(squared_error_sum: float, denominator: int) -> float:
    if denominator <= 0:
        return float("nan")
    return math.sqrt(squared_error_sum / denominator)


def _timestamp_us_to_string(value: int | None) -> str | None:
    if value is None:
        return None
    return (
        datetime.fromtimestamp(value / 1_000_000, tz=UTC)
        .replace(tzinfo=None)
        .strftime("%Y-%m-%d %H:%M:%S")
    )


def build_result_row(
    prepared_dataset: PreparedDataset,
    *,
    model_variant: str,
    training_outcome: TrainingOutcome,
    runtime_seconds: float,
    seed: int,
    batch_size: int,
    learning_rate: float,
) -> dict[str, object]:
    test_split = prepared_dataset.test
    metrics = training_outcome.metrics
    return {
        "dataset_id": prepared_dataset.dataset_id,
        "model_id": MODEL_ID,
        "model_variant": model_variant,
        "task_id": TASK_ID,
        "history_steps": prepared_dataset.history_steps,
        "forecast_steps": prepared_dataset.forecast_steps,
        "stride_steps": prepared_dataset.stride_steps,
        "split_protocol": SPLIT_PROTOCOL,
        "window_count": int(metrics["window_count"]),
        "prediction_count": int(metrics["prediction_count"]),
        "start_timestamp": _timestamp_us_to_string(int(test_split.output_start_us.min())),
        "end_timestamp": _timestamp_us_to_string(int(test_split.output_end_us.max())),
        "mae_kw": float(metrics["mae_kw"]),
        "rmse_kw": float(metrics["rmse_kw"]),
        "mae_pu": float(metrics["mae_pu"]),
        "rmse_pu": float(metrics["rmse_pu"]),
        "device": training_outcome.device,
        "runtime_seconds": round(runtime_seconds, 6),
        "train_window_count": int(prepared_dataset.train.inputs.shape[0]),
        "val_window_count": int(prepared_dataset.val.inputs.shape[0]),
        "test_window_count": int(prepared_dataset.test.inputs.shape[0]),
        "best_epoch": training_outcome.best_epoch,
        "epochs_ran": training_outcome.epochs_ran,
        "best_val_rmse_pu": training_outcome.best_val_rmse_pu,
        "seed": seed,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    }


def execute_training_job(
    prepared_dataset: PreparedDataset,
    *,
    model_variant: str,
    device: str | None = None,
    seed: int = DEFAULT_SEED,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
) -> dict[str, object]:
    dataset_start = time.monotonic()
    training_outcome = train_model(
        model_variant,
        prepared_dataset,
        device=resolve_device(device),
        seed=seed,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        early_stopping_patience=early_stopping_patience,
    )
    runtime_seconds = time.monotonic() - dataset_start
    _profile_log(
        prepared_dataset.dataset_id,
        "training_complete",
        model_variant=model_variant,
        best_epoch=training_outcome.best_epoch,
        epochs_ran=training_outcome.epochs_ran,
        best_val_rmse_pu=training_outcome.best_val_rmse_pu,
        test_rmse_pu=training_outcome.metrics["rmse_pu"],
        runtime_seconds=round(runtime_seconds, 6),
    )
    return build_result_row(
        prepared_dataset,
        model_variant=model_variant,
        training_outcome=training_outcome,
        runtime_seconds=runtime_seconds,
        seed=seed,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )


def sort_result_frame(frame: pl.DataFrame) -> pl.DataFrame:
    return (
        frame.with_columns(
            pl.col("dataset_id")
            .replace_strict(_DATASET_ORDER, default=len(_DATASET_ORDER))
            .alias("__dataset_order"),
            pl.col("model_variant")
            .replace_strict(_MODEL_ORDER, default=len(_MODEL_ORDER))
            .alias("__model_order"),
        )
        .sort(["__dataset_order", "__model_order"])
        .drop(["__dataset_order", "__model_order"])
    )


def run_experiment(
    *,
    dataset_ids: Sequence[str] = DEFAULT_DATASETS,
    model_variants: Sequence[str] = MODEL_VARIANTS,
    cache_root: str | Path = _CACHE_ROOT,
    output_path: str | Path = _OUTPUT_PATH,
    device: str | None = None,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    max_windows_per_split: int | None = None,
    seed: int = DEFAULT_SEED,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    dataset_loader: Callable[..., PreparedDataset] | None = None,
    job_runner: Callable[..., dict[str, object]] | None = None,
) -> pl.DataFrame:
    unknown_variants = [variant for variant in model_variants if variant not in MODEL_VARIANTS]
    if unknown_variants:
        raise ValueError(f"Unsupported model variants: {unknown_variants!r}")
    loader = dataset_loader or prepare_dataset
    runner = job_runner or execute_training_job
    prepared_datasets: dict[str, PreparedDataset] = {}
    rows: list[dict[str, object]] = []

    for dataset_id in dataset_ids:
        prepared = loader(
            dataset_id,
            cache_root=cache_root,
            max_windows_per_split=max_windows_per_split,
        )
        prepared_datasets[dataset_id] = prepared
        for model_variant in model_variants:
            rows.append(
                runner(
                    prepared,
                    model_variant=model_variant,
                    device=device,
                    seed=seed,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    max_epochs=max_epochs,
                    early_stopping_patience=early_stopping_patience,
                )
            )

    results = sort_result_frame(pl.DataFrame(rows).select(_RESULT_COLUMNS))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.write_csv(output)
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local LTSF-Linear training benchmarks.")
    parser.add_argument(
        "--dataset",
        action="append",
        choices=list(DEFAULT_DATASETS),
        dest="datasets",
        help="Limit execution to one or more datasets. Defaults to all supported datasets.",
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=list(MODEL_VARIANTS),
        dest="models",
        help="Limit execution to one or more model variants. Defaults to both nlinear and dlinear.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Training device. Defaults to auto (cuda -> mps -> cpu).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_MAX_EPOCHS,
        help="Maximum training epochs.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=_OUTPUT_PATH,
        help="Output CSV path. Defaults to experiment/ltsf-linear.csv in the repo root.",
    )
    parser.add_argument(
        "--max-windows-per-split",
        type=int,
        default=None,
        help="Optional smoke-test limit applied independently to train/val/test after split assignment.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_experiment(
        dataset_ids=tuple(args.datasets) if args.datasets else DEFAULT_DATASETS,
        model_variants=tuple(args.models) if args.models else MODEL_VARIANTS,
        device=args.device,
        max_epochs=args.epochs,
        output_path=args.output_path,
        max_windows_per_split=args.max_windows_per_split,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

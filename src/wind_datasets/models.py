from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from .utils import duration_to_steps, format_duration, parse_duration


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    source_root: Path
    resolution_minutes: int
    turbine_ids: tuple[str, ...]
    target_column: str
    target_unit: str
    timezone_policy: str
    timestamp_convention: str
    default_feature_groups: tuple[str, ...]
    handler: str
    default_quality_profile: str = "default"


@dataclass(frozen=True)
class ResolvedTaskSpec:
    task_id: str
    history_duration: timedelta
    forecast_duration: timedelta
    stride_duration: timedelta
    target_mode: str
    granularity: str
    allow_partial_input: bool
    allow_partial_output: bool
    resolution_minutes: int
    history_steps: int
    forecast_steps: int
    stride_steps: int

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "history_duration": format_duration(self.history_duration),
            "forecast_duration": format_duration(self.forecast_duration),
            "stride_duration": format_duration(self.stride_duration),
            "target_mode": self.target_mode,
            "granularity": self.granularity,
            "allow_partial_input": self.allow_partial_input,
            "allow_partial_output": self.allow_partial_output,
            "resolution_minutes": self.resolution_minutes,
            "history_steps": self.history_steps,
            "forecast_steps": self.forecast_steps,
            "stride_steps": self.stride_steps,
        }


@dataclass(frozen=True)
class TaskSpec:
    history_duration: timedelta | str
    forecast_duration: timedelta | str
    task_id: str | None = None
    stride_duration: timedelta | str | None = None
    target_mode: str = "multi_step"
    granularity: str = "turbine"
    allow_partial_input: bool = False
    allow_partial_output: bool = False

    def __post_init__(self) -> None:
        history = parse_duration(self.history_duration)
        forecast = parse_duration(self.forecast_duration)
        stride = parse_duration(self.stride_duration) if self.stride_duration is not None else None
        generated_task_id = self.task_id or (
            f"{self.target_mode}_{format_duration(history)}_to_{format_duration(forecast)}"
        )
        object.__setattr__(self, "history_duration", history)
        object.__setattr__(self, "forecast_duration", forecast)
        object.__setattr__(self, "stride_duration", stride)
        object.__setattr__(self, "task_id", generated_task_id)

    @classmethod
    def next_6h_from_24h(cls) -> "TaskSpec":
        return cls(
            task_id="next_6h_from_24h",
            history_duration="24h",
            forecast_duration="6h",
            stride_duration=None,
            target_mode="multi_step",
            granularity="turbine",
        )

    def resolve(self, resolution_minutes: int) -> ResolvedTaskSpec:
        stride = self.stride_duration or timedelta(minutes=resolution_minutes)
        return ResolvedTaskSpec(
            task_id=self.task_id or "task",
            history_duration=self.history_duration,
            forecast_duration=self.forecast_duration,
            stride_duration=stride,
            target_mode=self.target_mode,
            granularity=self.granularity,
            allow_partial_input=self.allow_partial_input,
            allow_partial_output=self.allow_partial_output,
            resolution_minutes=resolution_minutes,
            history_steps=duration_to_steps(self.history_duration, resolution_minutes),
            forecast_steps=duration_to_steps(self.forecast_duration, resolution_minutes),
            stride_steps=duration_to_steps(stride, resolution_minutes),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "history_duration": format_duration(self.history_duration),
            "forecast_duration": format_duration(self.forecast_duration),
            "stride_duration": format_duration(self.stride_duration),
            "target_mode": self.target_mode,
            "granularity": self.granularity,
            "allow_partial_input": self.allow_partial_input,
            "allow_partial_output": self.allow_partial_output,
        }

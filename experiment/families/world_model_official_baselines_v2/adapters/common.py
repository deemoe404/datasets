from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class AdapterForwardOutput:
    prediction: np.ndarray
    source_file: str


def repeat_last_value(last_value: Sequence[Sequence[float]] | np.ndarray, forecast_steps: int) -> np.ndarray:
    anchor = np.asarray(last_value, dtype=np.float64)
    return np.repeat(anchor[:, None, :], forecast_steps, axis=1)


def source_file_for(path: str | Path) -> str:
    return str(Path(path).resolve())

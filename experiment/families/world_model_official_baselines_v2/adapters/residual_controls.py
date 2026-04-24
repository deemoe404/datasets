from __future__ import annotations

import numpy as np

from .common import AdapterForwardOutput, repeat_last_value


class _ZeroResidualAdapter:
    source_file = "repo://official-baselines-v2-controls"

    def predict(self, history_target: np.ndarray, forecast_steps: int = 36) -> AdapterForwardOutput:
        prediction = repeat_last_value(history_target[:, -1, :], forecast_steps)
        return AdapterForwardOutput(prediction=prediction, source_file=self.source_file)


class RidgeResidualAdapter(_ZeroResidualAdapter):
    pass


class MLPResidualAdapter(_ZeroResidualAdapter):
    pass


class GRUResidualAdapter(_ZeroResidualAdapter):
    pass


class TCNResidualAdapter(_ZeroResidualAdapter):
    pass

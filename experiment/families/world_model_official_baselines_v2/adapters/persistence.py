from __future__ import annotations

import numpy as np

from .common import AdapterForwardOutput, repeat_last_value


class LastValuePersistenceAdapter:
    source_file = "analytic://last-value-persistence"

    def predict(self, history_target: np.ndarray, forecast_steps: int = 36) -> AdapterForwardOutput:
        prediction = repeat_last_value(history_target[:, -1, :], forecast_steps)
        return AdapterForwardOutput(prediction=prediction, source_file=self.source_file)


class SeasonalPersistenceAdapter:
    source_file = "analytic://seasonal-persistence"

    def predict(self, history_target: np.ndarray, forecast_steps: int = 36, seasonal_lag: int = 144) -> AdapterForwardOutput:
        if history_target.shape[1] < seasonal_lag:
            seasonal_anchor = history_target[:, -1, :]
        else:
            seasonal_anchor = history_target[:, -seasonal_lag, :]
        prediction = repeat_last_value(seasonal_anchor, forecast_steps)
        return AdapterForwardOutput(prediction=prediction, source_file=self.source_file)

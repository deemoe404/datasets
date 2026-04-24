from __future__ import annotations

from typing import Mapping


def gate_b_overfit_passed(train_rmse_pu: float, persistence_train_rmse_pu: float) -> bool:
    return train_rmse_pu <= 0.03 or train_rmse_pu <= 0.5 * persistence_train_rmse_pu


def gate_c_continuity_passed(metrics_10min: Mapping[str, float], persistence_10min: Mapping[str, float]) -> bool:
    return (
        metrics_10min["rmse_pu"] <= 1.05 * persistence_10min["rmse_pu"]
        and metrics_10min["mae_pu"] <= 1.05 * persistence_10min["mae_pu"]
    )

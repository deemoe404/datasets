from __future__ import annotations

from typing import Sequence

import numpy as np


def horizon_bucket_means(leadwise_rmse_pu: Sequence[float]) -> dict[str, float]:
    values = np.asarray(leadwise_rmse_pu, dtype=np.float64)
    return {
        "short_horizon_rmse_pu": float(np.nanmean(values[0:6])),
        "mid_horizon_rmse_pu": float(np.nanmean(values[6:18])),
        "long_horizon_rmse_pu": float(np.nanmean(values[18:36])),
    }

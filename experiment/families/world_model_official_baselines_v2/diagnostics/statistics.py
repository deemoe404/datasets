from __future__ import annotations

import numpy as np


def paired_bootstrap_delta(
    *,
    baseline_errors: list[float] | np.ndarray,
    proposed_errors: list[float] | np.ndarray,
    repeats: int = 5000,
    seed: int = 3407,
) -> dict[str, float]:
    baseline = np.asarray(baseline_errors, dtype=np.float64)
    proposed = np.asarray(proposed_errors, dtype=np.float64)
    if baseline.shape != proposed.shape:
        raise ValueError("baseline_errors and proposed_errors must have the same shape.")
    if baseline.ndim != 1 or baseline.size == 0:
        raise ValueError("paired bootstrap expects a non-empty 1D origin-level error array.")
    rng = np.random.default_rng(seed)
    deltas = np.empty(repeats, dtype=np.float64)
    origin_count = baseline.size
    for repeat_index in range(repeats):
        indices = rng.integers(0, origin_count, size=origin_count)
        deltas[repeat_index] = float(np.mean(baseline[indices] - proposed[indices]))
    return {
        "delta_mean": float(np.mean(deltas)),
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
        "prob_delta_gt_zero": float(np.mean(deltas > 0.0)),
    }


def block_bootstrap_indices(origin_count: int, *, block_length: int, repeats: int, seed: int = 3407) -> np.ndarray:
    if origin_count <= 0 or block_length <= 0 or repeats <= 0:
        raise ValueError("origin_count, block_length, and repeats must be positive.")
    rng = np.random.default_rng(seed)
    result = np.empty((repeats, origin_count), dtype=np.int64)
    max_start = max(0, origin_count - block_length)
    for repeat_index in range(repeats):
        chosen: list[int] = []
        while len(chosen) < origin_count:
            start = int(rng.integers(0, max_start + 1))
            chosen.extend(range(start, min(start + block_length, origin_count)))
        result[repeat_index] = np.asarray(chosen[:origin_count], dtype=np.int64)
    return result


def error_quantiles(abs_errors: list[float] | np.ndarray) -> dict[str, float]:
    values = np.asarray(abs_errors, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("error_quantiles expects a non-empty 1D absolute-error array.")
    return {
        "ae_p50": float(np.round(np.quantile(values, 0.50), 12)),
        "ae_p90": float(np.round(np.quantile(values, 0.90), 12)),
        "ae_p95": float(np.round(np.quantile(values, 0.95), 12)),
    }

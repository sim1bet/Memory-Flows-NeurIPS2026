"""Confidence intervals that respect independent pattern-set sampling."""

from __future__ import annotations

import numpy as np


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0:
        return np.nan, np.nan
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    radius = z * np.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return float(max(0.0, center - radius)), float(min(1.0, center + radius))


def bootstrap_mean_interval(
    values: np.ndarray,
    rng: np.random.Generator,
    repetitions: int = 2000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    if values.size == 1:
        return float(values[0]), float(values[0])
    indices = rng.integers(0, values.size, size=(int(repetitions), values.size))
    estimates = values[indices].mean(axis=1)
    tail = 0.5 * (1.0 - confidence)
    return float(np.quantile(estimates, tail)), float(np.quantile(estimates, 1.0 - tail))

"""Small, dependency-light statistics for paired agent experiments."""

from __future__ import annotations

import math

import numpy as np


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def bootstrap_ci(x, stat=np.mean, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boots = np.array([stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n_boot)])
    return (float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2)))


def paired_bootstrap_diff(a, b, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    """CI of mean(b - a) over paired items (e.g. tasks)."""
    d = np.asarray(b, float) - np.asarray(a, float)
    return float(d.mean()) if len(d) else float("nan"), bootstrap_ci(d, n_boot=n_boot, alpha=alpha, seed=seed)


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value from discordant counts."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)

"""Bootstrap confidence intervals and related statistical utilities.

All functions are pure (no I/O).
"""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    data: np.ndarray,
    func,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_state: int | None = None,
) -> tuple[float, float]:
    """Bootstrap confidence interval for a statistic.

    Repeatedly resamples *data* with replacement, applies *func* to each
    resample, and returns the empirical percentile interval.

    Args:
        data:         1-D array of observations.
        func:         Callable that accepts a 1-D numpy array and returns a scalar.
        n_bootstrap:  Number of bootstrap iterations (default 1000).
        ci:           Coverage probability (default 0.95 → 95% CI).
        random_state: Optional integer seed for reproducibility.

    Returns:
        Tuple (lower, upper) for the confidence interval.
    """
    data = np.asarray(data, dtype=float)
    rng = np.random.default_rng(random_state)
    samples = [
        func(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_bootstrap)
    ]
    lo = float(np.percentile(samples, (1.0 - ci) / 2.0 * 100))
    hi = float(np.percentile(samples, (1.0 + ci) / 2.0 * 100))
    return lo, hi

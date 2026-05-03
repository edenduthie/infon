"""Statistical utilities for ablation experiments.

Provides paired bootstrap confidence intervals for paired-sample comparisons
where both arms evaluate the same scenarios (paired structure improves power).
"""

from __future__ import annotations

import numpy as np


def paired_bootstrap_ci(
    differences: np.ndarray,
    n_resamples: int = 10_000,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Compute a percentile bootstrap confidence interval for the mean difference.

    Uses a fixed internal seed (0) so results are reproducible given the same
    input array. Returns Python ``float`` values (not ``np.float64``) so that
    ``isinstance(result, float)`` is ``True``.

    Args:
        differences: 1-D array of per-scenario differences (arm A minus arm B).
        n_resamples:  Number of bootstrap resamples (default 10 000).
        ci:           Nominal coverage level, e.g. 0.95 for a 95 % CI.

    Returns:
        ``(low, high)`` — the lower and upper bounds of the percentile CI.
    """
    n = len(differences)
    rng = np.random.default_rng(seed=0)

    bootstrap_means = np.empty(n_resamples)
    for i in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        bootstrap_means[i] = np.mean(differences[indices])

    alpha = 1.0 - ci
    low = float(np.percentile(bootstrap_means, 100 * alpha / 2))
    high = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))
    return low, high

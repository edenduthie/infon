"""Red-phase TDD tests for paired_bootstrap_ci.

This file is intentionally written BEFORE the implementation exists in
``reference_v2.experiments.stats``. All tests MUST fail today with an
ImportError; once the green phase lands the implementation, all tests
should pass.

Spec:
- Epic 02 — Synthetic stress dataset + full ablation matrix
- Task infon-8pa.13 (red) -> green phase follows
- Function signature: paired_bootstrap_ci(differences, n_resamples=10_000, ci=0.95)
  returns (low, high)

The paired bootstrap is used (not a t-test) because both ablation arms
evaluate the same scenarios — the paired structure is real and ignoring
it loses statistical power. Coverage must match the requested 95% level
to within 2 percentage points (acceptance range [90%, 99%]).
"""

from __future__ import annotations

import numpy as np
import pytest

from reference_v2.experiments.stats import paired_bootstrap_ci


# ── Constants ─────────────────────────────────────────────────────────────

TRUE_DIFF = 0.1   # known mean difference (mu = A - B)
N_SCENARIOS = 50  # per-experiment sample size
N_BOOTSTRAP = 1000  # bootstrap resamples per CI (low for test speed)
CI_LEVEL = 0.95   # nominal coverage level
N_EXPERIMENTS = 200  # simulated experiments for coverage check
COVERAGE_LOW = 0.90   # minimum acceptable empirical coverage
COVERAGE_HIGH = 0.99  # maximum acceptable empirical coverage


# ── Helpers ────────────────────────────────────────────────────────────────


def _generate_differences(rng: np.random.Generator, n: int = N_SCENARIOS) -> np.ndarray:
    """Generate an array of per-scenario differences with mean TRUE_DIFF.

    Uses unit-variance Gaussian noise so the signal-to-noise is controlled
    and the CI should bracket the true value near the nominal rate.
    """
    return rng.normal(loc=TRUE_DIFF, scale=1.0, size=n)


# ── Test 1 — single CI brackets the true difference ────────────────────────


def test_ci_brackets_true_difference():
    """A single run with seed=42 produces a CI that brackets TRUE_DIFF.

    Phase RED: ImportError because experiments.stats does not exist yet.

    Phase GREEN: the function should return (ci_low, ci_high) such that
    ci_low <= TRUE_DIFF <= ci_high for this particular seed.
    """
    rng = np.random.default_rng(42)
    differences = _generate_differences(rng)

    ci_low, ci_high = paired_bootstrap_ci(
        differences,
        n_resamples=N_BOOTSTRAP,
        ci=CI_LEVEL,
    )

    assert ci_low <= TRUE_DIFF <= ci_high, (
        f"CI ({ci_low:.4f}, {ci_high:.4f}) does not bracket "
        f"true difference {TRUE_DIFF}"
    )


# ── Test 2 — CI is a valid interval ────────────────────────────────────────


def test_ci_is_valid_interval():
    """The returned CI is a proper interval (low < high) with finite values.

    Phase RED: ImportError.

    Phase GREEN: basic sanity checks on the returned tuple.
    """
    rng = np.random.default_rng(42)
    differences = _generate_differences(rng)

    ci_low, ci_high = paired_bootstrap_ci(
        differences,
        n_resamples=N_BOOTSTRAP,
        ci=CI_LEVEL,
    )

    assert isinstance(ci_low, float), f"ci_low is not float: {type(ci_low)}"
    assert isinstance(ci_high, float), f"ci_high is not float: {type(ci_high)}"
    assert np.isfinite(ci_low), f"ci_low is not finite: {ci_low}"
    assert np.isfinite(ci_high), f"ci_high is not finite: {ci_high}"
    assert ci_low < ci_high, (
        f"CI is not a proper interval: ci_low={ci_low:.4f} >= ci_high={ci_high:.4f}"
    )


# ── Test 3 — empirical coverage across 200 simulated experiments ───────────


def test_coverage_rate_near_nominal():
    """Over 200 simulated experiments, the CI brackets TRUE_DIFF in
    approximately 95% of cases.

    Acceptance range: [90%, 99%] (i.e., 2 percentage points around nominal).

    Phase RED: ImportError because experiments.stats does not exist yet.

    Phase GREEN: the bootstrap implementation must achieve near-nominal
    coverage. Each experiment draws a fresh sample from the null distribution
    with mean TRUE_DIFF, constructs a 95% CI, and checks whether the CI
    contains TRUE_DIFF. The fraction of experiments where this holds is the
    empirical coverage rate.
    """
    covered = 0

    for seed in range(N_EXPERIMENTS):
        rng = np.random.default_rng(seed)
        differences = _generate_differences(rng)

        ci_low, ci_high = paired_bootstrap_ci(
            differences,
            n_resamples=N_BOOTSTRAP,
            ci=CI_LEVEL,
        )

        if ci_low <= TRUE_DIFF <= ci_high:
            covered += 1

    empirical_coverage = covered / N_EXPERIMENTS

    assert COVERAGE_LOW <= empirical_coverage <= COVERAGE_HIGH, (
        f"Empirical coverage {empirical_coverage:.3f} ({covered}/{N_EXPERIMENTS}) "
        f"is outside the acceptable range [{COVERAGE_LOW}, {COVERAGE_HIGH}]. "
        f"Nominal CI level is {CI_LEVEL}."
    )


if __name__ == "__main__":
    # Allow direct invocation for quick manual verification.
    test_ci_brackets_true_difference()
    test_ci_is_valid_interval()
    test_coverage_rate_near_nominal()
    print("All tests passed.")

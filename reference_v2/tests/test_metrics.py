"""Red-phase TDD tests for reference_v2.experiments.metrics.

All six metric functions are tested with synthetic data whose expected
values have been computed by hand.  Each assertion uses a tolerance of
1e-6 to catch numerical drift without being sensitive to floating-point
representation details.

Module under test: reference_v2/experiments/metrics.py  (does not exist yet).

Metrics tested
--------------
1. polarity_accuracy_3way(pred, true)        — 3-way categorical accuracy
2. ece(probs, labels, n_bins=15)             — Expected Calibration Error
3. brier_3way(probs, labels)                 — Multi-class Brier score
4. aurc(risks, coverages)                    — Area under risk-coverage curve
5. selective_accuracy_at_coverage(scores, preds, labels, coverage)
6. spearman_rho(x, y)                        — Spearman rank correlation

All functions must be vectorised over numpy arrays; no scikit-learn dependency.

Phase A.6 (RED): every test below fails with ImportError because
    reference_v2/experiments/metrics.py  does not exist.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# ── Import under test — intentionally absent until Green phase ─────────────
from reference_v2.experiments.metrics import (  # noqa: E402
    aurc,
    brier_3way,
    ece,
    polarity_accuracy_3way,
    selective_accuracy_at_coverage,
    spearman_rho,
)

# ──────────────────────────────────────────────────────────────────────────
# 1. polarity_accuracy_3way
# ──────────────────────────────────────────────────────────────────────────


def test_polarity_accuracy_3way_exact():
    """10 predictions, 8 correct → accuracy = 0.8 exactly.

    Hand computation
    ----------------
    correct = 8, total = 10 → 8 / 10 = 0.8
    """
    # Labels: SUPPORTS=0, REFUTES=1, NEI=2
    true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    pred = np.array([0, 1, 2, 0, 1, 2, 0, 1, 0, 1])  # last 2 wrong

    result = polarity_accuracy_3way(pred, true)

    assert abs(result - 0.8) < 1e-6, f"expected 0.8, got {result}"


def test_polarity_accuracy_3way_all_correct():
    """All predictions correct → accuracy = 1.0."""
    labels = np.array([0, 1, 2, 0, 1])
    result = polarity_accuracy_3way(labels, labels)
    assert abs(result - 1.0) < 1e-6, f"expected 1.0, got {result}"


def test_polarity_accuracy_3way_all_wrong():
    """All predictions wrong → accuracy = 0.0."""
    true = np.array([0, 0, 0])
    pred = np.array([1, 2, 1])
    result = polarity_accuracy_3way(pred, true)
    assert abs(result - 0.0) < 1e-6, f"expected 0.0, got {result}"


# ──────────────────────────────────────────────────────────────────────────
# 2. ece
# ──────────────────────────────────────────────────────────────────────────


def test_ece_confident_correct():
    """15 examples all confidently correct at p=0.9 → ECE = 0.1.

    Hand computation
    ----------------
    All 15 examples have max-class probability 0.9 and are correctly
    classified.  With n_bins=15 they all fall into the same confidence
    bin whose midpoint is near 0.9.

    Within that bin:
        accuracy_in_bin  = 1.0   (every example is correct)
        confidence_in_bin = 0.9  (every example has confidence 0.9)
        fraction_in_bin  = 15/15 = 1.0

    ECE = fraction_in_bin * |accuracy - confidence|
        = 1.0 * |1.0 - 0.9|
        = 0.1
    """
    n = 15
    # 3-class probs: class-0 is always the predicted class with p=0.9
    probs = np.zeros((n, 3))
    probs[:, 0] = 0.9
    probs[:, 1] = 0.05
    probs[:, 2] = 0.05
    labels = np.zeros(n, dtype=int)  # all true label = 0 (correct)

    result = ece(probs, labels, n_bins=15)

    assert abs(result - 0.1) < 1e-6, f"expected ECE=0.1, got {result}"


def test_ece_perfect_calibration():
    """Perfect calibration: two groups at p=0.5 and p=1.0 with matching
    empirical accuracy → ECE = 0.0.

    Hand computation
    ----------------
    Group A (5 examples): confidence=1.0, all correct → |acc - conf| = 0
    Group B (5 examples): confidence=0.5, exactly half correct (2/4 is not
        possible with 5 examples, so we use 5 examples with acc=0.6 …
        actually use 10 examples split 5+5 at conf=1.0 and 5 at conf=0.5
        with 2 or 3 correct).

    Simpler: use only the confidence=1.0 group (all correct).
        fraction=1.0, |1.0 - 1.0| = 0 → ECE = 0.
    """
    probs = np.zeros((5, 3))
    probs[:, 0] = 1.0
    labels = np.zeros(5, dtype=int)

    result = ece(probs, labels, n_bins=15)
    assert abs(result - 0.0) < 1e-6, f"expected ECE=0.0, got {result}"


# ──────────────────────────────────────────────────────────────────────────
# 3. brier_3way
# ──────────────────────────────────────────────────────────────────────────


def test_brier_3way_hand_computed():
    """3 examples with known probabilities → Brier score computed by hand.

    Hand computation
    ----------------
    Brier score (multi-class) = mean over n of sum_k (p_k - y_k)^2

    Example 0: true=0 (SUPPORTS), probs=[0.7, 0.2, 0.1]
        (0.7-1)^2 + (0.2-0)^2 + (0.1-0)^2 = 0.09 + 0.04 + 0.01 = 0.14

    Example 1: true=1 (REFUTES),  probs=[0.1, 0.8, 0.1]
        (0.1-0)^2 + (0.8-1)^2 + (0.1-0)^2 = 0.01 + 0.04 + 0.01 = 0.06

    Example 2: true=2 (NEI),      probs=[0.2, 0.2, 0.6]
        (0.2-0)^2 + (0.2-0)^2 + (0.6-1)^2 = 0.04 + 0.04 + 0.16 = 0.24

    Mean = (0.14 + 0.06 + 0.24) / 3 = 0.44 / 3
    """
    probs = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.2, 0.6],
    ])
    labels = np.array([0, 1, 2])

    expected = 0.44 / 3  # ≈ 0.14666...

    result = brier_3way(probs, labels)

    assert abs(result - expected) < 1e-6, f"expected {expected}, got {result}"


def test_brier_3way_perfect():
    """Perfect predictions → Brier score = 0.0.

    Hand computation
    ----------------
    One-hot probs exactly matching labels → each squared term = 0.
    """
    probs = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    labels = np.array([0, 1, 2])

    result = brier_3way(probs, labels)
    assert abs(result - 0.0) < 1e-6, f"expected 0.0, got {result}"


# ──────────────────────────────────────────────────────────────────────────
# 4. aurc
# ──────────────────────────────────────────────────────────────────────────


def test_aurc_hand_computed():
    """Construct a risk-coverage curve with known area.

    Hand computation
    ----------------
    Use 4 operating points:
        coverages = [0.25, 0.50, 0.75, 1.00]
        risks     = [0.00, 0.00, 1/3,  0.50]

    These arise from 4 examples ordered by decreasing confidence:
        rank 1 (conf=0.95): correct → cumulative: 1/1=0, risk=0
        rank 2 (conf=0.80): correct → cumulative: 2/2=0, risk=0
        rank 3 (conf=0.60): wrong   → cumulative: 2/3 correct, risk=1/3
        rank 4 (conf=0.30): wrong   → cumulative: 2/4=0.5 correct, risk=0.5

    AURC = np.trapz(risks, coverages)
         = 0.5*(0+0)*0.25 + 0.5*(0+1/3)*0.25 + 0.5*(1/3+0.5)*0.25
         = 0 + 1/24 + 5/48
         = 2/48 + 5/48
         = 7/48
    """
    coverages = np.array([0.25, 0.50, 0.75, 1.00])
    risks = np.array([0.0, 0.0, 1.0 / 3.0, 0.5])

    expected = 7.0 / 48.0  # ≈ 0.145833...

    result = aurc(risks, coverages)

    assert abs(result - expected) < 1e-6, f"expected {expected}, got {result}"


def test_aurc_zero_risk():
    """All predictions correct → AURC = 0.

    Hand computation
    ----------------
    risks = [0, 0, 0, 0] at any coverages → integral = 0.
    """
    coverages = np.array([0.25, 0.50, 0.75, 1.00])
    risks = np.zeros(4)

    result = aurc(risks, coverages)
    assert abs(result - 0.0) < 1e-6, f"expected 0.0, got {result}"


# ──────────────────────────────────────────────────────────────────────────
# 5. selective_accuracy_at_coverage
# ──────────────────────────────────────────────────────────────────────────


def test_selective_accuracy_at_coverage_exact():
    """Top-70% (7 of 10) by confidence; known accuracy among those 7.

    Hand computation
    ----------------
    10 examples with scores, predictions, and labels:

    idx | score | pred | label | correct
     0  |  0.95 |  0   |   0   |   yes
     1  |  0.90 |  1   |   1   |   yes
     2  |  0.85 |  2   |   2   |   yes
     3  |  0.80 |  0   |   0   |   yes
     4  |  0.75 |  1   |   1   |   yes
     5  |  0.70 |  2   |   0   |   no   ← rank 6 (top-7 boundary)
     6  |  0.65 |  0   |   1   |   no   ← rank 7 (top-7 boundary)
     7  |  0.40 |  1   |   2   |   no   ← excluded
     8  |  0.30 |  2   |   0   |   no   ← excluded
     9  |  0.20 |  0   |   1   |   no   ← excluded

    Top-7 by score: indices 0,1,2,3,4,5,6
    Correct among top-7: indices 0,1,2,3,4 → 5 correct out of 7
    selective_accuracy = 5 / 7
    """
    scores = np.array([0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.40, 0.30, 0.20])
    preds  = np.array([0,    1,    2,    0,    1,    2,    0,    1,    2,    0   ])
    labels = np.array([0,    1,    2,    0,    1,    0,    1,    2,    0,    1   ])

    expected = 5.0 / 7.0  # ≈ 0.714285...

    result = selective_accuracy_at_coverage(scores, preds, labels, coverage=0.7)

    assert abs(result - expected) < 1e-6, f"expected {expected}, got {result}"


def test_selective_accuracy_at_coverage_full():
    """Coverage=1.0 should equal standard accuracy."""
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    preds  = np.array([0,   1,   2,   0  ])
    labels = np.array([0,   1,   0,   0  ])  # 3 correct out of 4

    expected = 3.0 / 4.0

    result = selective_accuracy_at_coverage(scores, preds, labels, coverage=1.0)

    assert abs(result - expected) < 1e-6, f"expected {expected}, got {result}"


# ──────────────────────────────────────────────────────────────────────────
# 6. spearman_rho
# ──────────────────────────────────────────────────────────────────────────


def test_spearman_rho_known_correlation():
    """Known rank correlation between m_theta and planted_thinness.

    Hand computation
    ----------------
    x = [1, 2, 3, 4, 5]  →  ranks = [1, 2, 3, 4, 5]  (already ranked)
    y = [5, 4, 3, 2, 1]  →  ranks = [5, 4, 3, 2, 1]  (reverse order)

    Spearman rho for perfectly inverse rank ordering = -1.0

    Formula: rho = 1 - (6 * sum(d_i^2)) / (n*(n^2 - 1))
        d = [1-5, 2-4, 3-3, 4-2, 5-1] = [-4, -2, 0, 2, 4]
        sum(d^2) = 16 + 4 + 0 + 4 + 16 = 40
        rho = 1 - (6*40)/(5*24) = 1 - 240/120 = 1 - 2 = -1.0
    """
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])

    result = spearman_rho(x, y)

    assert abs(result - (-1.0)) < 1e-6, f"expected -1.0, got {result}"


def test_spearman_rho_perfect_positive():
    """Identical ordering → Spearman rho = +1.0."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = spearman_rho(x, x)
    assert abs(result - 1.0) < 1e-6, f"expected 1.0, got {result}"


def test_spearman_rho_partial_known():
    """Partial known rank correlation → verified by formula.

    Hand computation
    ----------------
    x = [1, 2, 3, 4]  →  ranks_x = [1, 2, 3, 4]
    y = [1, 3, 2, 4]  →  ranks_y = [1, 3, 2, 4]

    d = [0, -1, 1, 0]
    sum(d^2) = 0 + 1 + 1 + 0 = 2
    rho = 1 - (6*2)/(4*15) = 1 - 12/60 = 1 - 0.2 = 0.8
    """
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([1.0, 3.0, 2.0, 4.0])

    expected = 0.8

    result = spearman_rho(x, y)

    assert abs(result - expected) < 1e-6, f"expected {expected}, got {result}"


# ──────────────────────────────────────────────────────────────────────────
# Allow direct invocation for quick manual red-phase verification
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

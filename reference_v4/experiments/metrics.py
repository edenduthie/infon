"""Evaluation metrics for the Cognition benchmark harness.

All functions operate on plain Python lists or numpy arrays.
No I/O is performed here.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    """Fraction of correctly predicted labels.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        Accuracy in [0, 1].

    Raises:
        ZeroDivisionError: If lists are empty.
    """
    if len(y_true) == 0:
        raise ZeroDivisionError("accuracy requires at least one sample")
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def ece(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error (ECE).

    Partitions predictions into equal-width bins by confidence, then computes
    the weighted average of |avg_confidence - avg_accuracy| per bin.

    Args:
        confidences: 1-D array of confidence scores in [0, 1].
        correct:     1-D boolean array (True if prediction was correct).
        n_bins:      Number of equal-width bins (default 15).

    Returns:
        ECE in [0, 1].
    """
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    n = len(confidences)
    if n == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece_sum = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Include right edge in last bin
        if i < n_bins - 1:
            mask = (confidences >= lo) & (confidences < hi)
        else:
            mask = (confidences >= lo) & (confidences <= hi)
        if not np.any(mask):
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = correct[mask].mean()
        bin_count = mask.sum()
        ece_sum += (bin_count / n) * abs(bin_conf - bin_acc)

    return float(ece_sum)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Brier score: mean squared error of probability estimates.

    Lower is better (0 = perfect, 1 = worst).

    Args:
        probs:  1-D array of predicted probabilities in [0, 1].
        labels: 1-D array of binary ground-truth labels (0 or 1).

    Returns:
        Brier score in [0, 1].
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if len(probs) == 0:
        return 0.0
    return float(np.mean((probs - labels) ** 2))


def aurc(confidences: np.ndarray, correct: np.ndarray) -> float:
    """Area Under the Risk-Coverage curve (AURC).

    Risk = error rate. Coverage = fraction of predictions retained (sorted
    by descending confidence). Lower AURC indicates better selective prediction.

    Args:
        confidences: 1-D array of confidence scores in [0, 1].
        correct:     1-D boolean array (True if prediction was correct).

    Returns:
        AURC in [0, 1].
    """
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    n = len(confidences)
    if n == 0:
        return 0.0

    # Sort by confidence descending (highest confidence first = full coverage)
    order = np.argsort(-confidences)
    sorted_correct = correct[order]

    # Risk at each coverage level: cumulative error rate
    cumulative_errors = np.cumsum(1.0 - sorted_correct)
    coverages = np.arange(1, n + 1, dtype=float)
    risks = cumulative_errors / coverages

    # Area under risk-coverage curve via trapezoidal integration
    # Coverage ranges from 1/n to 1 in steps of 1/n
    coverage_fractions = coverages / n
    return float(np.trapezoid(risks, coverage_fractions))


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation coefficient.

    Args:
        x: First 1-D array.
        y: Second 1-D array.

    Returns:
        Spearman rho in [-1, 1].
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    result = scipy_stats.spearmanr(x, y)
    return float(result.statistic)

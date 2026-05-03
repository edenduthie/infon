"""Evaluation metrics for three-class polarity experiments.

All functions operate on numpy arrays and have no scikit-learn dependency.
Each function includes a docstring reference to the originating publication.

Metrics
-------
polarity_accuracy_3way  — categorical accuracy for 3-class labels
ece                     — Expected Calibration Error (Guo et al. 2017)
brier_3way              — multi-class Brier score (Brier 1950)
aurc                    — Area Under Risk-Coverage curve (Geifman & El-Yaniv 2017)
selective_accuracy_at_coverage — accuracy at a fixed coverage level
spearman_rho            — Spearman rank correlation (Spearman 1904)
"""

from __future__ import annotations

import numpy as np


def polarity_accuracy_3way(pred: np.ndarray, true: np.ndarray) -> float:
    """Categorical accuracy for 3-class (SUPPORTS / REFUTES / NEI) labels.

    Parameters
    ----------
    pred:
        Predicted class indices, shape (n,).
    true:
        Ground-truth class indices, shape (n,).

    Returns
    -------
    float
        Fraction of predictions that equal the ground-truth label.
    """
    return float(np.mean(pred == true))


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error.

    Reference: Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).
    "On Calibration of Modern Neural Networks." ICML 2017.

    The predicted confidence is the maximum class probability.  Samples are
    binned into ``n_bins`` equal-width bins over [0, 1] by their confidence.
    Within each bin the absolute gap between mean accuracy and mean confidence
    is weighted by the fraction of samples in that bin.

    Parameters
    ----------
    probs:
        Predicted probabilities, shape (n, num_classes).
    labels:
        Ground-truth class indices, shape (n,).
    n_bins:
        Number of equal-width bins to use (default 15).

    Returns
    -------
    float
        Expected Calibration Error in [0, 1].
    """
    confidence = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    n_total = len(labels)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece_value = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # include the right edge only for the last bin so that p=1.0 is captured
        if i < n_bins - 1:
            in_bin = (confidence >= lo) & (confidence < hi)
        else:
            in_bin = (confidence >= lo) & (confidence <= hi)

        n_in_bin = int(in_bin.sum())
        if n_in_bin == 0:
            continue

        accuracy_bin = float(np.mean(predictions[in_bin] == labels[in_bin]))
        confidence_bin = float(np.mean(confidence[in_bin]))
        fraction_bin = n_in_bin / n_total
        ece_value += fraction_bin * abs(accuracy_bin - confidence_bin)

    return float(ece_value)


def brier_3way(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multi-class Brier score.

    Reference: Brier, G. W. (1950). "Verification of Forecasts Expressed in
    Terms of Probability." Monthly Weather Review, 78(1), 1–3.

    The score is the mean squared Euclidean distance between the predicted
    probability vector and the one-hot encoding of the true label.

    Parameters
    ----------
    probs:
        Predicted probabilities, shape (n, num_classes).
    labels:
        Ground-truth class indices, shape (n,).

    Returns
    -------
    float
        Brier score in [0, 2].  Lower is better.
    """
    num_classes = probs.shape[1]
    y_onehot = np.eye(num_classes)[labels]
    return float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))


def aurc(risks: np.ndarray, coverages: np.ndarray) -> float:
    """Area Under the Risk-Coverage curve.

    Reference: Geifman, Y., & El-Yaniv, R. (2017). "Selective Classification
    for Deep Neural Networks." NeurIPS 2017.

    Integration is performed using the trapezoidal rule.

    Parameters
    ----------
    risks:
        Risk values at each operating point, shape (m,).
    coverages:
        Coverage values at each operating point, shape (m,).
        Must be non-decreasing and in [0, 1].

    Returns
    -------
    float
        Area under the risk-coverage curve (lower is better).
    """
    return float(np.trapezoid(risks, coverages))


def selective_accuracy_at_coverage(
    scores: np.ndarray,
    preds: np.ndarray,
    labels: np.ndarray,
    coverage: float,
) -> float:
    """Accuracy restricted to the top-k highest-confidence predictions.

    Samples are ranked by descending ``scores``.  The top
    ``round(n * coverage)`` samples are retained and their accuracy is
    returned.

    Parameters
    ----------
    scores:
        Confidence scores, shape (n,).  Higher means more confident.
    preds:
        Predicted class indices, shape (n,).
    labels:
        Ground-truth class indices, shape (n,).
    coverage:
        Fraction of samples to include (0 < coverage <= 1.0).

    Returns
    -------
    float
        Accuracy among the top-k retained samples.
    """
    n = len(scores)
    k = round(n * coverage)
    # argsort ascending; reverse to get descending confidence order
    top_k_idx = np.argsort(scores)[::-1][:k]
    return float(np.mean(preds[top_k_idx] == labels[top_k_idx]))


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation coefficient.

    Reference: Spearman, C. (1904). "The Proof and Measurement of Association
    between Two Things." The American Journal of Psychology, 15(1), 72–101.

    Ranks are assigned via double argsort (i.e. ordinal ranking without
    tie-breaking).  The closed-form d^2 formula is used.

    Parameters
    ----------
    x:
        First variable, shape (n,).
    y:
        Second variable, shape (n,).

    Returns
    -------
    float
        Spearman rho in [-1, +1].
    """
    rank_x = np.argsort(np.argsort(x)) + 1.0
    rank_y = np.argsort(np.argsort(y)) + 1.0
    d = rank_x - rank_y
    n = len(x)
    rho = 1.0 - 6.0 * np.sum(d ** 2) / (n * (n ** 2 - 1))
    return float(rho)

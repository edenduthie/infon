#!/usr/bin/env python3
"""Generate four headline figures for the phase-2 memo."""
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless — no display available
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).parent.parent  # reference_v2/experiments/
CANONICAL_DIR = BASE / "results/canonical_cells/"
FIGURES_DIR = BASE / "results/figures/"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Verdict colour palette shared across figures
VERDICT_COLORS = {"SUPPORTS": "steelblue", "REFUTES": "firebrick", "NEI": "gray"}


def _load_per_scenario(cell: str, seed: int) -> list[dict]:
    """Load per_scenario list from a cell result file."""
    path = CANONICAL_DIR / f"{cell}__seed={seed}.json"
    with path.open() as fh:
        data = json.load(fh)
    return data.get("per_scenario", [])


def _load_metrics(cell: str, seed: int) -> dict:
    path = CANONICAL_DIR / f"{cell}__seed={seed}.json"
    with path.open() as fh:
        data = json.load(fh)
    return data.get("metrics", {})


# ---------------------------------------------------------------------------
# Figure 1: H1 accuracy by hop count
# ---------------------------------------------------------------------------

def build_figure1_h1_accuracy_by_hop() -> None:
    """Bar chart: polarity accuracy at hop_count=2 for typed_ikl vs uniform_mean."""
    h1_path = FIGURES_DIR / "h1_accuracy_by_hop.json"
    with h1_path.open() as fh:
        rows = json.load(fh)

    if not rows:
        print("WARNING: h1_accuracy_by_hop.json is empty — skipping Figure 1")
        return

    aggregators = [r["aggregator"] for r in rows]
    means = [r["polarity_acc_mean"] for r in rows]
    ci_lows = [r["polarity_acc_ci_low"] for r in rows]
    ci_highs = [r["polarity_acc_ci_high"] for r in rows]

    # Asymmetric error bars: [below, above]
    err_below = [m - lo for m, lo in zip(means, ci_lows)]
    err_above = [hi - m for m, hi in zip(means, ci_highs)]
    yerr = [err_below, err_above]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(aggregators))
    bars = ax.bar(
        x,
        means,
        yerr=yerr,
        capsize=6,
        color=["steelblue", "darkorange"],
        edgecolor="black",
        width=0.5,
        error_kw={"elinewidth": 1.5, "ecolor": "black"},
    )

    ax.set_xticks(x)
    ax.set_xticklabels(aggregators, fontsize=11)
    ax.set_xlabel("Aggregator", fontsize=12)
    ax.set_ylabel("Polarity Accuracy", fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title("H1: Polarity Accuracy at Hop Count = 2", fontsize=13)
    ax.axhline(1 / 3, color="gray", linestyle="--", linewidth=1, label="Random baseline (1/3)")
    ax.legend(fontsize=9)

    # Annotation note
    ax.text(
        0.5,
        -0.18,
        "Note: single hop_count=2; full H1 curve pending more compositional_depth variation",
        ha="center",
        va="top",
        transform=ax.transAxes,
        fontsize=8,
        style="italic",
        color="dimgray",
    )

    fig.tight_layout()
    out_path = FIGURES_DIR / "h1_accuracy_by_hop.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: H2 m(Theta) vs planted thinness (scatter)
# ---------------------------------------------------------------------------

def build_figure2_theta_vs_thinness() -> None:
    """Scatter: m(Theta) = mass[3] vs planted_thinness, coloured by oracle_verdict."""
    scenarios = _load_per_scenario("canonical", 42)
    metrics = _load_metrics("canonical", 42)

    if not scenarios:
        print("WARNING: no per_scenario data for canonical seed=42 — skipping Figure 2")
        return

    thinness_vals = np.array([s["planted_thinness"] for s in scenarios])
    theta_vals = np.array([s["mass"][3] for s in scenarios])
    verdicts = [s["oracle_verdict"] for s in scenarios]

    spearman_rho = metrics.get("spearman_thinness", float("nan"))

    fig, ax = plt.subplots(figsize=(7, 5))

    for verdict, color in VERDICT_COLORS.items():
        mask = [v == verdict for v in verdicts]
        ax.scatter(
            thinness_vals[mask],
            theta_vals[mask],
            c=color,
            label=verdict,
            alpha=0.4,
            s=18,
            edgecolors="none",
        )

    ax.set_xlabel("Planted Thinness", fontsize=12)
    ax.set_ylabel("m(Θ) = mass[3]", fontsize=12)
    ax.set_title("H2: m(Θ) vs Planted Thinness (canonical, seed=42)", fontsize=13)
    ax.legend(title="Oracle Verdict", fontsize=10)

    rho_label = f"Spearman ρ = {spearman_rho:.3f}" if not np.isnan(spearman_rho) else "Spearman ρ = N/A"
    ax.annotate(
        rho_label,
        xy=(0.97, 0.05),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"),
    )

    fig.tight_layout()
    out_path = FIGURES_DIR / "h2_theta_vs_thinness.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: Reliability diagrams (calibration)
# ---------------------------------------------------------------------------

def _prediction_confidence(scenario: dict, cell: str) -> float:
    """Return scalar confidence from mass array.

    For ds_4mass cells (canonical): max(mass[0], mass[1], mass[2] + mass[3])
    For others (softmax, dirichlet): max(mass)
    """
    mass = scenario["mass"]
    if cell in ("canonical", "coherence_off", "single_layer", "top1_fusion",
                "uniform_aggregator", "teacher_only"):
        return float(max(mass[0], mass[1], mass[2] + mass[3]))
    return float(max(mass))


def _compute_calibration(scenarios: list[dict], cell: str, n_bins: int = 15):
    """Bin by confidence and return (mean_confidence, fraction_correct) arrays."""
    if not scenarios:
        return np.array([]), np.array([])

    confidences = np.array([_prediction_confidence(s, cell) for s in scenarios])
    correct = np.array(
        [int(s["predicted_verdict"] == s["oracle_verdict"]) for s in scenarios],
        dtype=float,
    )

    bin_edges = np.linspace(0, 1, n_bins + 1)
    mean_confs, frac_corrects = [], []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        mean_confs.append(confidences[mask].mean())
        frac_corrects.append(correct[mask].mean())

    return np.array(mean_confs), np.array(frac_corrects)


def build_figure3_reliability_diagrams() -> None:
    """3-panel reliability diagram — one per readout family."""
    readout_cells = [
        ("canonical", "ds_4mass"),
        ("softmax_readout", "softmax_readout"),
        ("dirichlet_edl_readout", "dirichlet_edl_readout"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    fig.suptitle("Reliability Diagrams (calibration) — seed=42", fontsize=13)

    for ax, (cell, label) in zip(axes, readout_cells):
        scenarios = _load_per_scenario(cell, 42)

        mean_confs, frac_corrects = _compute_calibration(scenarios, cell)

        # Perfect calibration diagonal
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Perfect")

        if mean_confs.size > 0:
            ax.bar(
                mean_confs,
                frac_corrects,
                width=0.04,
                align="center",
                color="steelblue",
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
                label="Model",
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence", fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel("Fraction Correct", fontsize=11)
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=8)

    fig.tight_layout()
    out_path = FIGURES_DIR / "reliability_diagrams.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 4: Risk-coverage curves
# ---------------------------------------------------------------------------

def _compute_risk_coverage(scenarios: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Compute risk-coverage curve using 1 - mass[3] as confidence.

    Sorts by descending confidence, computes cumulative accuracy
    at each threshold increment.
    Returns (coverage, risk) arrays, both length = len(scenarios).
    """
    if not scenarios:
        return np.array([]), np.array([])

    confidences = np.array([1.0 - s["mass"][3] for s in scenarios])
    correct = np.array(
        [int(s["predicted_verdict"] == s["oracle_verdict"]) for s in scenarios],
        dtype=float,
    )

    order = np.argsort(confidences)[::-1]  # descending confidence
    correct_sorted = correct[order]

    n = len(correct_sorted)
    coverage = np.arange(1, n + 1) / n
    cumulative_acc = np.cumsum(correct_sorted) / np.arange(1, n + 1)
    risk = 1.0 - cumulative_acc

    return coverage, risk


def build_figure4_risk_coverage() -> None:
    """Risk-coverage curves for the three readout families."""
    readout_cells = [
        ("canonical", "ds_4mass"),
        ("softmax_readout", "softmax_readout"),
        ("dirichlet_edl_readout", "dirichlet_edl_readout"),
    ]
    colors = ["steelblue", "darkorange", "seagreen"]

    fig, ax = plt.subplots(figsize=(7, 5))

    for (cell, label), color in zip(readout_cells, colors):
        scenarios = _load_per_scenario(cell, 42)
        coverage, risk = _compute_risk_coverage(scenarios)

        if coverage.size == 0:
            print(f"WARNING: no scenarios for {cell} — skipping curve")
            continue

        ax.plot(coverage, risk, label=label, color=color, linewidth=1.8)

    ax.set_xlabel("Coverage (fraction included)", fontsize=12)
    ax.set_ylabel("Risk (1 - accuracy)", fontsize=12)
    ax.set_title("Risk-Coverage Curves (seed=42)", fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(title="Readout", fontsize=10)
    ax.text(
        0.97,
        0.97,
        "Lower is better",
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=9,
        style="italic",
        color="dimgray",
    )

    fig.tight_layout()
    out_path = FIGURES_DIR / "risk_coverage.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    build_figure1_h1_accuracy_by_hop()
    build_figure2_theta_vs_thinness()
    build_figure3_reliability_diagrams()
    build_figure4_risk_coverage()
    print("All figures generated.")


if __name__ == "__main__":
    main()

"""Render per-infon Θ distribution before/after fusion (C.4 Figure 1).

Story: pre-fusion per-infon m(Θ) is HEALTHY (mean ≈0.23, in the
acceptance band [0.20, 0.40]); the audited collapse is fusion-side. The
canonical fix (``top1, decisive_top_k=2, coherence_weight=1.0``) keeps
m(Θ) inside the band by avoiding aggressive fusion.

Output: ``experiments/results/canonical_v0_2_figures/per_infon_mass_distribution.png``
A side-by-side two-panel figure (Toyota, Honda). Each panel overlays:

  * histogram of pre-fusion per-infon m(Θ) at seed=42 (from B.2 diagnostic);
  * vertical dashed line at the post-fusion baseline m(Θ) (Dempster
    k=5, cw=0.2; from the audit reproducer);
  * vertical dashed line at the post-fusion canonical m(Θ) (top1 k=2,
    cw=1.0; the chosen fix);
  * shaded acceptance band [0.20, 0.40].

Determinism: all data is loaded from committed JSON; PNG metadata is
suppressed via ``metadata={"Software": None, ...}``.

Usage (from ``reference_v2/``)::

    PYTHONPATH=src python3 -m experiments.figures.per_infon_distribution
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering, deterministic
import matplotlib.pyplot as plt

# --- Constants ---------------------------------------------------------------

THETA_INDEX = 3  # m(Θ) is index 3 of the 4-frame {S, R, U, Θ}.
QUERIES = ("toyota", "honda")
SEED = 42

ACCEPTANCE_LOW = 0.20
ACCEPTANCE_HIGH = 0.40

# Color palette (consistent with sweep_summary.py / tab10).
COLOR_HIST = "#1f77b4"  # tab:blue   — pre-fusion per-infon distribution
COLOR_BASELINE = "#d62728"  # tab:red — collapsed post-fusion baseline
COLOR_CANONICAL = "#2ca02c"  # tab:green — canonical post-fusion fix
COLOR_BAND = "0.85"

REPO_ROOT = Path(__file__).resolve().parents[2]
DIAG_DIR = REPO_ROOT / "experiments" / "results" / "diagnostic"
BASELINE_PATH = REPO_ROOT / "experiments" / "results" / "baseline" / f"baseline__seed={SEED}.json"
CANONICAL_PATH = (
    REPO_ROOT
    / "experiments"
    / "results"
    / "sweep_collapse"
    / f"sweep_collapse__cw=1.0__fr=top1__tk=2__seed={SEED}.json"
)
OUTPUT_DIR = REPO_ROOT / "experiments" / "results" / "canonical_v0_2_figures"

PNG_METADATA = {"Software": None, "Creation Time": None}


# --- Data loading ------------------------------------------------------------


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _per_infon_thetas(query: str) -> list[float]:
    record = _load_json(DIAG_DIR / f"{query}__seed={SEED}.json")
    return [r["mass"][THETA_INDEX] for r in record["per_infon_records"]]


def _post_fusion_theta(path: Path, query: str) -> float:
    record = _load_json(path)
    return float(record["queries"][query]["mass"][THETA_INDEX])


# --- Plotting ----------------------------------------------------------------


def _plot_panel(ax, query: str) -> None:
    thetas = _per_infon_thetas(query)
    baseline_theta = _post_fusion_theta(BASELINE_PATH, query)
    canonical_theta = _post_fusion_theta(CANONICAL_PATH, query)

    ax.axvspan(
        ACCEPTANCE_LOW,
        ACCEPTANCE_HIGH,
        color=COLOR_BAND,
        alpha=0.6,
        zorder=0,
        label=f"acceptance band [{ACCEPTANCE_LOW:.2f}, {ACCEPTANCE_HIGH:.2f}]",
    )

    ax.hist(
        thetas,
        bins=15,
        range=(0.0, 0.5),
        color=COLOR_HIST,
        edgecolor="white",
        alpha=0.85,
        zorder=2,
        label=f"pre-fusion per-infon (n={len(thetas)})",
    )

    ax.axvline(
        baseline_theta,
        color=COLOR_BASELINE,
        linestyle="--",
        linewidth=2.0,
        zorder=3,
        label=f"post-fusion baseline (Dempster k=5 cw=0.2): m(Θ)={baseline_theta:.4f}",
    )
    ax.axvline(
        canonical_theta,
        color=COLOR_CANONICAL,
        linestyle="--",
        linewidth=2.0,
        zorder=3,
        label=f"post-fusion canonical (top1 k=2 cw=1.0): m(Θ)={canonical_theta:.4f}",
    )

    ax.set_title(f"{query.capitalize()}", fontsize=12)
    ax.set_xlabel("m(Θ)")
    ax.set_xlim(0.0, 0.5)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)


def _make_figure() -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, query in zip(axes, QUERIES):
        _plot_panel(ax, query)

    axes[0].set_ylabel("count (per-infon records)")

    # Build one shared legend. Each panel's labels embed query-specific m(Θ)
    # values, so collect handles from both panels.
    handles_l, labels_l = axes[0].get_legend_handles_labels()
    handles_r, labels_r = axes[1].get_legend_handles_labels()
    # The acceptance-band and pre-fusion-histogram entries are identical
    # across panels — keep the left-panel copy and drop the right's
    # duplicates (first two labels).
    handles = handles_l + handles_r[2:]
    labels = labels_l + labels_r[2:]
    fig.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
        fontsize=9,
    )

    fig.suptitle(
        "Per-infon m(Θ) distribution vs. post-fusion m(Θ) (seed=42)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.22, 1, 0.95))
    return fig


def _save_png(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, format="png", metadata=PNG_METADATA)
    plt.close(fig)


# --- Entry point -------------------------------------------------------------


def main() -> None:
    matplotlib.rcParams["svg.hashsalt"] = "infon-6o3.21"
    matplotlib.rcParams["pdf.compression"] = 0

    fig = _make_figure()
    out_path = OUTPUT_DIR / "per_infon_mass_distribution.png"
    _save_png(fig, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

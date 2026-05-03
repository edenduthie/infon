"""Render m(Θ) vs coherence_weight under top1/k=2 (C.4 Figure 2).

Story: under the chosen ``fusion_rule="top1"`` and ``decisive_top_k=2``,
how does post-fusion m(Θ) respond to ``coherence_weight``? This is the
dose-response curve. cw=0 sits below the acceptance band, cw≥1.0 enters
it, cw=5.0 overshoots; cw=1.0 is the canonical pick.

Output: ``experiments/results/canonical_v0_2_figures/theta_vs_coherence.png``
A single-panel line plot with two lines (Toyota, Honda), mean ± std
across 5 seeds, the [0.20, 0.40] acceptance band shaded, and the
canonical-config point (cw=1.0) marked.

Data: ``experiments/results/sweep_collapse/aggregate.json``, filtered to
``fusion_rule="top1"`` and ``decisive_top_k=2``.

Determinism: data is loaded from committed JSON; PNG metadata is
suppressed via ``metadata={"Software": None, ...}``.

Usage (from ``reference_v2/``)::

    PYTHONPATH=src python3 -m experiments.figures.theta_vs_coherence
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering, deterministic
import matplotlib.pyplot as plt

# --- Constants ---------------------------------------------------------------

THETA_INDEX = 3
QUERIES = ("toyota", "honda")
COHERENCE_WEIGHTS = (0.0, 0.2, 0.5, 1.0, 2.0, 5.0)
CANONICAL_CW = 1.0
FUSION_RULE = "top1"
TOP_K = 2

ACCEPTANCE_LOW = 0.20
ACCEPTANCE_HIGH = 0.40

# Per-query colors (tab10).
QUERY_COLORS = {
    "toyota": "#1f77b4",  # tab:blue
    "honda": "#ff7f0e",  # tab:orange
}
COLOR_BAND = "0.85"
COLOR_CANONICAL_MARKER = "#2ca02c"  # tab:green

REPO_ROOT = Path(__file__).resolve().parents[2]
AGGREGATE_PATH = REPO_ROOT / "experiments" / "results" / "sweep_collapse" / "aggregate.json"
OUTPUT_DIR = REPO_ROOT / "experiments" / "results" / "canonical_v0_2_figures"

PNG_METADATA = {"Software": None, "Creation Time": None}


# --- Data loading ------------------------------------------------------------


def _load_aggregate() -> list[dict]:
    with AGGREGATE_PATH.open("r", encoding="utf-8") as fh:
        rows = json.load(fh)
    return [
        r
        for r in rows
        if r["fusion_rule"] == FUSION_RULE and int(r["decisive_top_k"]) == TOP_K
    ]


def _series(rows: list[dict], query: str) -> tuple[list[float], list[float], list[float]]:
    by_cw = {float(r["coherence_weight"]): r for r in rows}
    xs: list[float] = []
    means: list[float] = []
    stds: list[float] = []
    for cw in COHERENCE_WEIGHTS:
        row = by_cw.get(cw)
        if row is None:
            continue
        q = row["queries"][query]
        xs.append(cw)
        means.append(q["mean_mass"][THETA_INDEX])
        stds.append(q["std_mass"][THETA_INDEX])
    return xs, means, stds


# --- Plotting ----------------------------------------------------------------


def _make_figure(rows: list[dict]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.axhspan(
        ACCEPTANCE_LOW,
        ACCEPTANCE_HIGH,
        color=COLOR_BAND,
        alpha=0.6,
        zorder=0,
        label=f"acceptance band [{ACCEPTANCE_LOW:.2f}, {ACCEPTANCE_HIGH:.2f}]",
    )

    for query in QUERIES:
        xs, means, stds = _series(rows, query)
        color = QUERY_COLORS[query]
        ax.errorbar(
            xs,
            means,
            yerr=stds,
            color=color,
            linewidth=1.6,
            capsize=3.0,
            elinewidth=1.0,
            marker="o",
            markersize=5,
            label=f"{query.capitalize()} (mean ± std, n=5 seeds)",
            zorder=2,
        )
        # Annotate canonical-config point per query.
        if CANONICAL_CW in xs:
            i = xs.index(CANONICAL_CW)
            ax.annotate(
                f"m(Θ)={means[i]:.3f}",
                xy=(xs[i], means[i]),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=8,
                color=color,
            )

    # Highlight canonical cw=1.0 with a vertical guide + marker.
    ax.axvline(
        CANONICAL_CW,
        color=COLOR_CANONICAL_MARKER,
        linestyle=":",
        linewidth=1.4,
        zorder=1,
        label=f"canonical config (cw={CANONICAL_CW})",
    )

    ax.set_title(
        f"m(Θ) vs coherence_weight  (fusion_rule={FUSION_RULE}, decisive_top_k={TOP_K})",
        fontsize=12,
    )
    ax.set_xlabel("coherence_weight")
    ax.set_ylabel("m(Θ)  [mean ± std, n=5 seeds]")
    ax.set_xticks(list(COHERENCE_WEIGHTS))
    ax.set_xticklabels([str(c) for c in COHERENCE_WEIGHTS])
    ax.set_ylim(-0.05, 0.75)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(loc="upper left", fontsize=9, frameon=False)

    fig.tight_layout()
    return fig


def _save_png(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, format="png", metadata=PNG_METADATA)
    plt.close(fig)


# --- Entry point -------------------------------------------------------------


def main() -> None:
    matplotlib.rcParams["svg.hashsalt"] = "infon-6o3.21"
    matplotlib.rcParams["pdf.compression"] = 0

    rows = _load_aggregate()
    fig = _make_figure(rows)
    out_path = OUTPUT_DIR / "theta_vs_coherence.png"
    _save_png(fig, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

"""Render sweep summary figures from the 480-cell B.3 sweep.

Reads ``experiments/results/sweep_collapse/aggregate.json`` and writes one
PNG per diagnostic query (Honda, Toyota) showing m(Θ) across the
``coherence_weight × fusion_rule × decisive_top_k`` grid (mean ± std over
5 seeds), with an acceptance band at ``0.20 ≤ m(Θ) ≤ 0.40`` and markers
indicating verdict-polarity correctness.

Usage (from ``reference_v2/``)::

    PYTHONPATH=src python3 -m experiments.figures.sweep_summary
    # or
    PYTHONPATH=src python3 experiments/figures/sweep_summary.py

Dependency note: ``matplotlib`` is a study-time tool, not a runtime
dependency of ``cognition``. Install ad-hoc when reproducing figures::

    pip install matplotlib

The script is deterministic given a fixed ``aggregate.json``: the figure
contents (axes, lines, markers) are fully data-driven, and PNG metadata is
suppressed via ``metadata={"Software": None, ...}`` in ``savefig`` so that
two runs produce byte-identical files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")  # headless rendering, deterministic
import matplotlib.pyplot as plt

# --- Constants ---------------------------------------------------------------

THETA_INDEX = 3  # mean_mass[3] / std_mass[3] is m(Θ) on the 4-frame {S, R, U, Θ}.
QUERIES = ("honda", "toyota")
FUSION_RULES = ("dempster", "yager", "murphy", "top1")
TOP_K_VALUES = (1, 2, 3, 5)
COHERENCE_WEIGHTS = (0.0, 0.2, 0.5, 1.0, 2.0, 5.0)

# tab10 first 4 colors are reasonably color-blind-distinguishable for line plots.
TOP_K_COLORS = {
    1: "#1f77b4",  # tab:blue
    2: "#ff7f0e",  # tab:orange
    3: "#2ca02c",  # tab:green
    5: "#d62728",  # tab:red
}

ACCEPTANCE_LOW = 0.20
ACCEPTANCE_HIGH = 0.40

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "experiments" / "results" / "sweep_collapse"
AGGREGATE_PATH = RESULTS_DIR / "aggregate.json"

# PNG metadata kept empty so two runs produce byte-identical output.
PNG_METADATA = {"Software": None, "Creation Time": None}


# --- Data loading ------------------------------------------------------------


def _load_aggregate(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        rows = json.load(fh)
    if not isinstance(rows, list):
        raise ValueError(f"Expected list at {path}, got {type(rows).__name__}")
    return rows


def _row_index(rows: Iterable[dict]) -> dict[tuple[str, int, float], dict]:
    """Index by (fusion_rule, decisive_top_k, coherence_weight)."""
    index: dict[tuple[str, int, float], dict] = {}
    for row in rows:
        key = (row["fusion_rule"], int(row["decisive_top_k"]), float(row["coherence_weight"]))
        index[key] = row
    return index


# --- Plotting ---------------------------------------------------------------


def _plot_panel(ax, rows_by_key: dict, query: str, fusion_rule: str) -> None:
    ax.axhspan(
        ACCEPTANCE_LOW,
        ACCEPTANCE_HIGH,
        color="0.85",
        alpha=0.6,
        zorder=0,
        label="_acceptance",
    )

    for top_k in TOP_K_VALUES:
        xs: list[float] = []
        means: list[float] = []
        stds: list[float] = []
        polarity_correct: list[bool] = []
        for cw in COHERENCE_WEIGHTS:
            row = rows_by_key.get((fusion_rule, top_k, cw))
            if row is None:
                continue
            q = row["queries"][query]
            xs.append(cw)
            means.append(q["mean_mass"][THETA_INDEX])
            stds.append(q["std_mass"][THETA_INDEX])
            polarity_correct.append(bool(q["polarity_correct"]))

        color = TOP_K_COLORS[top_k]
        # Connecting line + error bars (no markers — markers drawn per-point below
        # so we can use different shapes for polarity-correct vs incorrect cells).
        ax.errorbar(
            xs,
            means,
            yerr=stds,
            color=color,
            linewidth=1.4,
            capsize=2.5,
            elinewidth=0.8,
            marker="",
            label=f"k={top_k}",
            zorder=2,
        )
        for x, y, ok in zip(xs, means, polarity_correct):
            if ok:
                ax.plot(
                    x, y,
                    marker="o", markersize=6,
                    markerfacecolor=color, markeredgecolor=color,
                    linestyle="None", zorder=3,
                )
            else:
                ax.plot(
                    x, y,
                    marker="s", markersize=6,
                    markerfacecolor="white", markeredgecolor=color,
                    markeredgewidth=1.2,
                    linestyle="None", zorder=3,
                )

    ax.set_title(f"{fusion_rule}", fontsize=11)
    ax.set_xlabel("coherence_weight")
    ax.set_xticks(list(COHERENCE_WEIGHTS))
    ax.set_xticklabels([str(c) for c in COHERENCE_WEIGHTS], fontsize=8, rotation=30)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)


def _make_figure(rows_by_key: dict, query: str) -> plt.Figure:
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
    for ax, fusion_rule in zip(axes, FUSION_RULES):
        _plot_panel(ax, rows_by_key, query, fusion_rule)

    axes[0].set_ylabel("m(Θ)  [mean ± std, n=5 seeds]")

    # Build one shared legend (line colors + polarity markers).
    line_handles = [
        plt.Line2D([0], [0], color=TOP_K_COLORS[k], linewidth=1.6, label=f"top_k = {k}")
        for k in TOP_K_VALUES
    ]
    polarity_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="None",
                   markerfacecolor="0.3", markeredgecolor="0.3", markersize=6,
                   label="polarity correct"),
        plt.Line2D([0], [0], marker="s", linestyle="None",
                   markerfacecolor="white", markeredgecolor="0.3", markersize=6,
                   markeredgewidth=1.2, label="polarity incorrect"),
        plt.Line2D([0], [0], color="0.85", linewidth=8, alpha=0.6,
                   label=f"acceptance band [{ACCEPTANCE_LOW:.2f}, {ACCEPTANCE_HIGH:.2f}]"),
    ]
    fig.legend(
        handles=line_handles + polarity_handles,
        loc="lower center",
        ncol=7,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=9,
    )

    fig.suptitle(
        f"{query.capitalize()} diagnostic — m(Θ) across 480-cell sweep",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    return fig


def _save_png(fig: plt.Figure, out_path: Path) -> None:
    fig.savefig(
        out_path,
        dpi=150,
        format="png",
        metadata=PNG_METADATA,
    )
    plt.close(fig)


# --- Entry point ------------------------------------------------------------


def main() -> None:
    rows = _load_aggregate(AGGREGATE_PATH)
    rows_by_key = _row_index(rows)

    # Determinism: matplotlib sometimes records timestamps in PNG metadata.
    # We pass metadata=PNG_METADATA at savefig time to suppress them.
    matplotlib.rcParams["svg.hashsalt"] = "infon-6o3.17"
    matplotlib.rcParams["pdf.compression"] = 0

    for query in QUERIES:
        fig = _make_figure(rows_by_key, query)
        out_path = RESULTS_DIR / f"sweep_summary_{query}.png"
        _save_png(fig, out_path)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

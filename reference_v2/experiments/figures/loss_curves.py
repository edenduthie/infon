"""Render training loss curves under each fusion_rule (C.4 Figure 3).

Story: the canonical config converges; it does not diverge under the
``coherence_weight=1.0`` regularizer. Defends against "you just turned
up regularization until something happened."

Output: ``experiments/results/canonical_v0_2_figures/loss_curves.png``
A single panel: x = epoch (0–29), y = training loss; one line per
fusion_rule (dempster, yager, murphy, top1) at the canonical
``coherence_weight=1.0, decisive_top_k=2`` slice. The seed=42 trace is
plotted; a thin ±std band across 5 seeds is shaded behind each line.
The canonical (top1) curve is highlighted with a thicker stroke.

Note (recorded as a Stage-C learning): in the current implementation,
``fusion_rule`` only affects ``reason()``, not ``fit()``. As a result,
all four loss traces at fixed (seed, coherence_weight, decisive_top_k)
are identically equal. The figure still tells the intended story —
training is stable under cw=1.0 — but the four lines overlap by
construction.

Determinism: traces are loaded from committed JSON; PNG metadata is
suppressed via ``metadata={"Software": None, ...}``.

Usage (from ``reference_v2/``)::

    PYTHONPATH=src python3 -m experiments.figures.loss_curves
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering, deterministic
import matplotlib.pyplot as plt

# --- Constants ---------------------------------------------------------------

FUSION_RULES = ("dempster", "yager", "murphy", "top1")
SEEDS = (0, 1, 7, 13, 42)
HIGHLIGHT_SEED = 42
CANONICAL_RULE = "top1"
CANONICAL_CW = 1.0
CANONICAL_TK = 2

# tab10 — same palette as sweep_summary.py for top_k.
RULE_COLORS = {
    "dempster": "#1f77b4",  # tab:blue
    "yager": "#ff7f0e",  # tab:orange
    "murphy": "#2ca02c",  # tab:green
    "top1": "#d62728",  # tab:red — the canonical pick, drawn last/thicker
}

REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_DIR = REPO_ROOT / "experiments" / "results" / "sweep_collapse"
OUTPUT_DIR = REPO_ROOT / "experiments" / "results" / "canonical_v0_2_figures"

PNG_METADATA = {"Software": None, "Creation Time": None}


# --- Data loading ------------------------------------------------------------


def _cell_path(rule: str, seed: int) -> Path:
    return SWEEP_DIR / (
        f"sweep_collapse__cw={CANONICAL_CW}__fr={rule}__tk={CANONICAL_TK}__seed={seed}.json"
    )


def _load_trace(rule: str, seed: int) -> list[float]:
    with _cell_path(rule, seed).open("r", encoding="utf-8") as fh:
        return list(json.load(fh)["loss_trace"])


def _seed_band(rule: str) -> tuple[list[float], list[float], list[float]]:
    """Return (highlighted_trace, mean_per_epoch, std_per_epoch) across seeds."""
    traces = [_load_trace(rule, s) for s in SEEDS]
    n_epochs = len(traces[0])
    if not all(len(t) == n_epochs for t in traces):
        raise ValueError(f"loss_trace length mismatch across seeds for rule={rule}")
    means = [sum(t[e] for t in traces) / len(traces) for e in range(n_epochs)]
    # Population std (matches sweep aggregate convention).
    stds: list[float] = []
    for e in range(n_epochs):
        m = means[e]
        var = sum((t[e] - m) ** 2 for t in traces) / len(traces)
        stds.append(var ** 0.5)
    return _load_trace(rule, HIGHLIGHT_SEED), means, stds


# --- Plotting ----------------------------------------------------------------


def _make_figure() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))

    for rule in FUSION_RULES:
        trace, means, stds = _seed_band(rule)
        epochs = list(range(len(trace)))
        color = RULE_COLORS[rule]

        is_canonical = rule == CANONICAL_RULE
        linewidth = 2.6 if is_canonical else 1.4
        zorder = 4 if is_canonical else 2
        label = (
            f"{rule}  ← canonical" if is_canonical else rule
        )

        # Thin ±std band across 5 seeds.
        lo = [m - s for m, s in zip(means, stds)]
        hi = [m + s for m, s in zip(means, stds)]
        ax.fill_between(epochs, lo, hi, color=color, alpha=0.15, zorder=zorder - 1)

        # Seed=42 trace.
        ax.plot(
            epochs,
            trace,
            color=color,
            linewidth=linewidth,
            label=label,
            zorder=zorder,
        )

    ax.set_title(
        f"Loss curves under coherence_weight={CANONICAL_CW}, "
        f"decisive_top_k={CANONICAL_TK} (seed={HIGHLIGHT_SEED}; band = ±std over 5 seeds)",
        fontsize=11,
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss")
    ax.grid(True, axis="both", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(loc="upper right", fontsize=9, frameon=False)

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

    fig = _make_figure()
    out_path = OUTPUT_DIR / "loss_curves.png"
    _save_png(fig, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

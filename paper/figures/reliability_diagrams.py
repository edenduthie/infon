"""
reliability_diagrams.py — H3 panel figure.

Reads h3_panel.json and produces 15-bin reliability (calibration) diagrams,
one subplot per system.

Usage:
    python reliability_diagrams.py [--input PATH] [--output PATH]

Defaults:
    --input  paper/results/h3_panel.json
    --output paper/figures/reliability_diagrams.pdf
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Color-blind-safe palette per system (cycled)
SYSTEM_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00"]

DISPLAY_NAMES = {
    "cognition_symbolic": "Infon-symbolic",
    "cognition_gnn": "Infon+GNN",
    "flat_retrieval": "Flat retrieval",
    "symbolic_floor": "Symbolic floor",
    "nli_classifier": "NLI classifier",
    "llm_zeroshot": "LLM zero-shot",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Produce reliability_diagrams.pdf")
    parser.add_argument(
        "--input",
        default=os.path.join(os.path.dirname(__file__), "..", "results", "h3_panel.json"),
        help="Path to h3_panel.json",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "reliability_diagrams.pdf"),
        help="Path for output PDF",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.input, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    systems: list[str] = data["systems"]
    calibration: dict = data["calibration"]

    n_systems = len(systems)
    fig, axes = plt.subplots(1, n_systems, figsize=(4 * n_systems, 4), sharey=True)
    if n_systems == 1:
        axes = [axes]

    for idx, (ax, system) in enumerate(zip(axes, systems)):
        color = SYSTEM_COLORS[idx % len(SYSTEM_COLORS)]
        sys_cal = calibration[system]
        bins: list[float] = sys_cal["bins"]
        accuracies: list[float] = sys_cal["accuracies"]

        # Diagonal perfect-calibration reference
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect calibration")

        # Bar chart of calibration bins (gap from diagonal)
        bin_width = bins[0] if len(bins) > 1 else 0.1
        if len(bins) > 1:
            bin_width = bins[1] - bins[0]
        bin_lefts = [b - bin_width for b in bins]
        ax.bar(
            bin_lefts,
            accuracies,
            width=bin_width,
            align="edge",
            color=color,
            alpha=0.6,
            edgecolor="white",
            linewidth=0.5,
            label="Accuracy",
        )
        # Confidence line
        ax.step(
            [0.0] + list(np.array(bins) - bin_width / 2) + [1.0],
            [accuracies[0]] + list(accuracies) + [accuracies[-1]],
            where="post",
            color=color,
            linewidth=1.2,
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence", fontsize=10)
        ax.set_title(DISPLAY_NAMES.get(system, system.replace("_", " ")), fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx == 0:
            ax.set_ylabel("Accuracy", fontsize=11)
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Reliability diagrams (H3)", fontsize=12)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output, format="pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

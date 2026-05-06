"""
accuracy_by_hop.py — H1 panel figure.

Reads h1_panel.json and produces a grouped bar chart of accuracy vs num_hops,
one series per system.

Usage:
    python accuracy_by_hop.py [--input PATH] [--output PATH]

Defaults:
    --input  paper/results/h1_panel.json
    --output paper/figures/accuracy_by_hop.pdf
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Color-blind-safe palette (Wong 2011)
PALETTE = {
    "cognition_symbolic": "#0072B2",
    "cognition_gnn": "#E69F00",
    "flat_retrieval": "#009E73",
    "symbolic_floor": "#D55E00",
}
DEFAULT_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Produce accuracy_by_hop.pdf")
    parser.add_argument(
        "--input",
        default=os.path.join(os.path.dirname(__file__), "..", "results", "h1_panel.json"),
        help="Path to h1_panel.json",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "accuracy_by_hop.pdf"),
        help="Path for output PDF",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.input, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    systems: list[str] = data["systems"]
    depth_results: list[dict] = data["depth_results"]

    hops = [row["num_hops"] for row in depth_results]
    n_groups = len(hops)
    n_systems = len(systems)

    bar_width = 0.8 / n_systems
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(7, 4))

    for idx, system in enumerate(systems):
        color = PALETTE.get(system, DEFAULT_COLORS[idx % len(DEFAULT_COLORS)])
        accuracies = [row[system] for row in depth_results]
        offset = (idx - (n_systems - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            accuracies,
            width=bar_width,
            label=system.replace("_", " "),
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xlabel("Number of hops", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Multi-hop accuracy by reasoning depth (H1)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([str(h) for h in hops])
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output, format="pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

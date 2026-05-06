"""
theta_distribution.py — H2 panel figure.

Reads h2_panel.json and produces a box plot of m_theta by ground-truth class,
one subplot per system.

Usage:
    python theta_distribution.py [--input PATH] [--output PATH]

Defaults:
    --input  paper/results/h2_panel.json
    --output paper/figures/theta_distribution.pdf
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Color-blind-safe palette per class
CLASS_COLORS = {
    "SUPPORTS": "#0072B2",
    "REFUTES": "#D55E00",
    "NEI": "#009E73",
}
CLASSES = ["SUPPORTS", "REFUTES", "NEI"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Produce theta_distribution.pdf")
    parser.add_argument(
        "--input",
        default=os.path.join(os.path.dirname(__file__), "..", "results", "h2_panel.json"),
        help="Path to h2_panel.json",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "theta_distribution.pdf"),
        help="Path for output PDF",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.input, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    systems: list[str] = data["systems"]
    theta_by_class: dict = data["theta_by_class"]

    n_systems = len(systems)
    fig, axes = plt.subplots(1, n_systems, figsize=(4 * n_systems, 4), sharey=True)
    if n_systems == 1:
        axes = [axes]

    for ax, system in zip(axes, systems):
        system_data = theta_by_class[system]
        class_values = [system_data.get(cls, []) for cls in CLASSES]
        colors = [CLASS_COLORS[cls] for cls in CLASSES]

        bp = ax.boxplot(
            class_values,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 1.5},
            whiskerprops={"linewidth": 1},
            capprops={"linewidth": 1},
            flierprops={"marker": "o", "markersize": 3, "linestyle": "none"},
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(system.replace("_", " "), fontsize=10)
        ax.set_xticks(range(1, len(CLASSES) + 1))
        ax.set_xticklabels(CLASSES, fontsize=9)
        ax.set_ylim(0, 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(r"$m_\Theta$ (DS vacuous mass)", fontsize=11)
    fig.suptitle(r"$m_\Theta$ distribution by ground-truth class (H2)", fontsize=12)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output, format="pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

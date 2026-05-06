"""
pareto.py — Aggregate panel figure.

Reads aggregate_panel.json and produces a scatter plot of accuracy vs AURC
with a Pareto-front polyline (lower AURC and higher accuracy is better).

Usage:
    python pareto.py [--input PATH] [--output PATH]

Defaults:
    --input  paper/results/aggregate_panel.json
    --output paper/figures/pareto.pdf
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Color-blind-safe palette
POINT_COLOR = "#0072B2"
PARETO_COLOR = "#D55E00"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Produce pareto.pdf")
    parser.add_argument(
        "--input",
        default=os.path.join(
            os.path.dirname(__file__), "..", "results", "aggregate_panel.json"
        ),
        help="Path to aggregate_panel.json",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "pareto.pdf"),
        help="Path for output PDF",
    )
    return parser.parse_args()


def compute_pareto_front(systems: list[dict]) -> list[dict]:
    """Return systems on the Pareto front (maximise accuracy, minimise AURC)."""
    # Sort by accuracy descending, then AURC ascending
    sorted_systems = sorted(systems, key=lambda s: (-s["accuracy"], s["aurc"]))
    pareto: list[dict] = []
    min_aurc = float("inf")
    for s in sorted_systems:
        if s["aurc"] < min_aurc:
            pareto.append(s)
            min_aurc = s["aurc"]
    return pareto


def main() -> None:
    args = parse_args()

    with open(args.input, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    systems: list[dict] = data["systems"]

    accuracies = [s["accuracy"] for s in systems]
    aurcs = [s["aurc"] for s in systems]
    names = [s["name"].replace("_", " ") for s in systems]

    pareto = compute_pareto_front(systems)
    pareto_sorted = sorted(pareto, key=lambda s: s["aurc"])
    pareto_aurc = [s["aurc"] for s in pareto_sorted]
    pareto_acc = [s["accuracy"] for s in pareto_sorted]

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(aurcs, accuracies, color=POINT_COLOR, s=60, zorder=3, label="Systems")

    # Label each point
    for name, aurc, acc in zip(names, aurcs, accuracies):
        ax.annotate(
            name,
            xy=(aurc, acc),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )

    # Pareto front polyline
    ax.plot(
        pareto_aurc,
        pareto_acc,
        color=PARETO_COLOR,
        linewidth=1.5,
        marker="D",
        markersize=5,
        label="Pareto front",
        zorder=4,
    )

    ax.set_xlabel("AURC (lower is better)", fontsize=11)
    ax.set_ylabel("Accuracy (higher is better)", fontsize=11)
    ax.set_title("Accuracy vs AURC — Pareto front", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output, format="pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

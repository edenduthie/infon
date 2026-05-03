"""Generate the four headline tables from ablation results."""
import csv
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent.parent  # reference_v2/experiments/
CANONICAL_AGG = BASE / "results/canonical_cells/aggregate.json"
CANONICAL_CELLS_DIR = BASE / "results/canonical_cells/"
FIGURES_DIR = BASE / "results/figures/"


def save_table(rows: list[dict], stem: str) -> None:
    """Save rows as both CSV and JSON files under FIGURES_DIR."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    (FIGURES_DIR / f"{stem}.json").write_text(json.dumps(rows, indent=2))
    if rows:
        with open(FIGURES_DIR / f"{stem}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def build_h1_table(cell_map: dict) -> list[dict]:
    """Table 1: H1 accuracy by hop_count (ablation: aggregator).

    Only hop_count=2 exists in the current dataset, so one row per aggregator.
    Rows: typed_ikl=canonical, uniform_mean=uniform_aggregator.
    """
    rows = []
    for cell_name, agg_cell in [("typed_ikl", "canonical"), ("uniform_mean", "uniform_aggregator")]:
        row = cell_map[agg_cell]
        rows.append(
            {
                "aggregator": cell_name,
                "hop_count": 2,
                "polarity_acc_mean": row["polarity_acc_mean"],
                "polarity_acc_ci_low": row["polarity_acc_ci_low"],
                "polarity_acc_ci_high": row["polarity_acc_ci_high"],
                "note": "single hop_count=2 (full range pending)",
            }
        )
    return rows


def build_h2_table(agg: list[dict]) -> list[dict]:
    """Table 2: H2 rho — spearman_thinness for all 8 cells with CIs."""
    return [
        {
            "cell": row["cell"],
            "spearman_thinness_mean": row.get("spearman_thinness", 0),
            "spearman_thinness_ci_low": row.get("spearman_thinness_ci_low", 0),
            "spearman_thinness_ci_high": row.get("spearman_thinness_ci_high", 0),
        }
        for row in agg
    ]


def build_aurc_table(cell_map: dict) -> list[dict]:
    """Table 3: AURC mean/std for the three readout cells.

    Reads individual per-seed JSON files so that std can be computed.
    """
    readout_cells = ["canonical", "softmax_readout", "dirichlet_edl_readout"]
    rows = []
    for cell_name in readout_cells:
        cell_files = sorted(CANONICAL_CELLS_DIR.glob(f"{cell_name}__seed=*.json"))
        aurcs = [json.loads(f.read_text())["metrics"]["aurc"] for f in cell_files]
        rows.append(
            {
                "cell": cell_name,
                "readout": cell_map[cell_name]["config"].get("readout", "ds_4mass"),
                "aurc_mean": float(np.mean(aurcs)) if aurcs else 0.0,
                "aurc_std": float(np.std(aurcs)) if len(aurcs) > 1 else 0.0,
                "n_seeds": len(aurcs),
            }
        )
    return rows


def build_ece_brier_table(agg: list[dict]) -> list[dict]:
    """Table 4: ECE and Brier score for all 8 cells."""
    return [
        {
            "cell": row["cell"],
            "ece": row.get("ece", 0),
            "brier": row.get("brier", 0),
        }
        for row in agg
    ]


def main() -> None:
    agg: list[dict] = json.loads(CANONICAL_AGG.read_text())
    cell_map: dict[str, dict] = {row["cell"]: row for row in agg}

    save_table(build_h1_table(cell_map), "h1_accuracy_by_hop")
    save_table(build_h2_table(agg), "h2_rho_table")
    save_table(build_aurc_table(cell_map), "aurc_table")
    save_table(build_ece_brier_table(agg), "ece_brier_table")

    print("Tables generated:")
    for f in sorted(FIGURES_DIR.glob("*.csv")):
        print(f"  {f}")


if __name__ == "__main__":
    main()

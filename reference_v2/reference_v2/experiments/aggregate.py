"""reference_v2.experiments.aggregate — aggregate per-cell ablation run results.

Reads per-cell JSON files from a directory, groups them by cell identifier,
computes statistical summaries (mean, std, 95% CI), and returns one summary
dict per unique cell.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _ci_95(values: np.ndarray) -> tuple[float, float]:
    """Compute a 95% confidence interval using mean ± 1.96 * std / sqrt(n).

    Uses population std (ddof=0). When n == 1, std is 0 and CI collapses to
    the point estimate (both bounds equal the mean), which is finite.
    """
    n = len(values)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=0))
    margin = 1.96 * std / np.sqrt(n)
    return float(mean - margin), float(mean + margin)


def aggregate_runs(input_dir: str | Path) -> list[dict]:
    """Aggregate per-cell JSON result files into one summary dict per cell.

    Parameters
    ----------
    input_dir:
        Directory containing ``*.json`` files, each representing a single
        (cell, seed) evaluation run.

    Returns
    -------
    list[dict]
        One dict per unique cell value found in the JSON files.  Each dict
        contains the 15 keys documented in the module docstring.
    """
    input_dir = Path(input_dir)

    # Group runs by cell name, preserving sorted file order for reproducibility.
    cell_runs: dict[str, list[dict]] = {}
    for json_file in sorted(input_dir.glob("*.json")):
        with open(json_file) as fh:
            run = json.load(fh)
        cell = run["cell"]
        cell_runs.setdefault(cell, []).append(run)

    results: list[dict] = []
    for cell, runs in cell_runs.items():
        polarity_accs = np.array([r["polarity_acc"] for r in runs])
        spearman_vals = np.array([r["spearman_thinness"] for r in runs])

        pa_ci_low, pa_ci_high = _ci_95(polarity_accs)
        st_ci_low, st_ci_high = _ci_95(spearman_vals)

        row: dict = {
            "cell": cell,
            "n_seeds": len(runs),
            # polarity_acc statistics
            "polarity_acc_mean": float(np.mean(polarity_accs)),
            "polarity_acc_std": float(np.std(polarity_accs, ddof=0)),
            "polarity_acc_ci_low": pa_ci_low,
            "polarity_acc_ci_high": pa_ci_high,
            # scalar metrics averaged across seeds
            "ece": float(np.mean([r["ece"] for r in runs])),
            "brier": float(np.mean([r["brier"] for r in runs])),
            "aurc": float(np.mean([r["aurc"] for r in runs])),
            "sel_acc_at_50": float(np.mean([r["sel_acc_at_50"] for r in runs])),
            "sel_acc_at_70": float(np.mean([r["sel_acc_at_70"] for r in runs])),
            "sel_acc_at_90": float(np.mean([r["sel_acc_at_90"] for r in runs])),
            # spearman_thinness statistics
            "spearman_thinness": float(np.mean(spearman_vals)),
            "spearman_thinness_ci_low": st_ci_low,
            "spearman_thinness_ci_high": st_ci_high,
            # config from the first run for this cell
            "config": runs[0]["config"],
        }
        results.append(row)

    return results

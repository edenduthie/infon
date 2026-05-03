"""reference_v2.experiments.aggregate — aggregate per-cell ablation run results.

Reads per-cell JSON files from a directory, groups them by cell identifier,
computes statistical summaries (mean, std, 95% CI), and returns one summary
dict per unique cell.

Input per-cell JSON schema (written by ablation_matrix.run_cell):
    {
      "cell": <str>,
      "seed": <int>,
      "config": {...},
      "metrics": {
        "polarity_acc_mean": <float>,
        "ece": <float>,
        "brier": <float>,
        "aurc": <float>,
        "sel_acc_at_50": <float>,
        "sel_acc_at_70": <float>,
        "sel_acc_at_90": <float>,
        "spearman_thinness": <float>,
        ...
      },
      ...
    }

Legacy flat format (also accepted for backward compatibility):
    {
      "cell": <str>,
      "seed": <int>,
      "config": {...},
      "polarity_acc": <float>,
      "ece": <float>,
      ...
    }
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


def _extract_scalar(run: dict, key: str, metrics_key: str | None = None) -> float:
    """Extract a scalar metric from a run dict.

    Supports both the flat legacy format (key at top level) and the nested
    format produced by run_cell (key inside run["metrics"]).

    Parameters
    ----------
    run:
        The run dict loaded from a JSON file.
    key:
        The key to look up at the top level (legacy format).
    metrics_key:
        The key to look up inside run["metrics"] (new format). Defaults to
        the same as ``key``.
    """
    if metrics_key is None:
        metrics_key = key
    # Try nested metrics dict first (new format from run_cell)
    if "metrics" in run and metrics_key in run["metrics"]:
        return float(run["metrics"][metrics_key])
    # Fall back to flat top-level key (legacy test format)
    return float(run[key])


def aggregate_runs(input_dir: str | Path) -> list[dict]:
    """Aggregate per-cell JSON result files into one summary dict per cell.

    Parameters
    ----------
    input_dir:
        Directory containing ``*.json`` files, each representing a single
        (cell, seed) evaluation run.  The ``aggregate.json`` file itself
        is excluded from processing.

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
        # Skip the aggregate output file itself to avoid recursion
        if json_file.name == "aggregate.json":
            continue
        with open(json_file) as fh:
            run = json.load(fh)
        cell = run["cell"]
        cell_runs.setdefault(cell, []).append(run)

    results: list[dict] = []
    for cell, runs in cell_runs.items():
        polarity_accs = np.array([
            _extract_scalar(r, "polarity_acc", "polarity_acc_mean") for r in runs
        ])
        spearman_vals = np.array([
            _extract_scalar(r, "spearman_thinness") for r in runs
        ])

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
            "ece": float(np.mean([_extract_scalar(r, "ece") for r in runs])),
            "brier": float(np.mean([_extract_scalar(r, "brier") for r in runs])),
            "aurc": float(np.mean([_extract_scalar(r, "aurc") for r in runs])),
            "sel_acc_at_50": float(np.mean([_extract_scalar(r, "sel_acc_at_50") for r in runs])),
            "sel_acc_at_70": float(np.mean([_extract_scalar(r, "sel_acc_at_70") for r in runs])),
            "sel_acc_at_90": float(np.mean([_extract_scalar(r, "sel_acc_at_90") for r in runs])),
            # spearman_thinness statistics
            "spearman_thinness": float(np.mean(spearman_vals)),
            "spearman_thinness_ci_low": st_ci_low,
            "spearman_thinness_ci_high": st_ci_high,
            # config from the first run for this cell
            "config": runs[0]["config"],
        }
        results.append(row)

    return results


def main() -> None:
    """CLI entrypoint: aggregate per-cell JSON files into a summary JSON."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Aggregate per-cell ablation result JSONs into a summary."
    )
    parser.add_argument(
        "--input", required=True,
        help="Directory containing per-cell JSON files",
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to write the aggregate JSON output",
    )
    args = parser.parse_args()

    results = aggregate_runs(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"Wrote aggregate with {len(results)} cell(s) to {output_path}")
    for row in results:
        print(
            f"  {row['cell']:30s}  n_seeds={row['n_seeds']}  "
            f"polarity_acc={row['polarity_acc_mean']:.3f}±{row['polarity_acc_std']:.3f}"
        )


if __name__ == "__main__":
    main()

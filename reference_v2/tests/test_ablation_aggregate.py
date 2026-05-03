"""Red-phase TDD tests for aggregate_runs.

This file is intentionally written BEFORE the implementation exists in
``reference_v2.experiments.aggregate``. All tests MUST fail today with a
ModuleNotFoundError; once the green phase lands the implementation, all tests
should pass.

Spec:
- Epic 02 — Synthetic stress dataset + full ablation matrix
- Task infon-8pa.15 (red) -> green phase follows
- Function signature: aggregate_runs(input_dir) -> list[dict]

``aggregate_runs`` reads all per-cell JSON files from ``input_dir``,
groups them by ``cell`` identifier, computes statistical summaries
(mean, std, 95% CI for polarity_acc and spearman_thinness), and returns
a list of one dict per cell.

Required output schema per row (15 keys):
    cell, n_seeds, polarity_acc_mean, polarity_acc_std,
    polarity_acc_ci_low, polarity_acc_ci_high,
    ece, brier, aurc, sel_acc_at_50, sel_acc_at_70, sel_acc_at_90,
    spearman_thinness, spearman_thinness_ci_low, spearman_thinness_ci_high,
    config

Input per-cell JSON schema:
    {
      "cell": <str>,
      "seed": <int>,
      "config": {...},
      "polarity_acc": <float>,
      "ece": <float>,
      "brier": <float>,
      "aurc": <float>,
      "sel_acc_at_50": <float>,
      "sel_acc_at_70": <float>,
      "sel_acc_at_90": <float>,
      "spearman_thinness": <float>
    }
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Module under test (does not exist yet — causes ModuleNotFoundError on import)
# ---------------------------------------------------------------------------
from reference_v2.experiments.aggregate import aggregate_runs  # type: ignore[import]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CELLS = [
    "canonical",
    "uniform_aggregator",
    "softmax_readout",
    "dirichlet_readout",
]

SEEDS = [42, 0, 1]

# Per-cell polarity_acc values (seed order: 42, 0, 1)
# Deliberately varied so mean/std can be hand-verified.
POLARITY_ACC_BY_CELL: dict[str, list[float]] = {
    "canonical":           [0.85, 0.83, 0.87],
    "uniform_aggregator":  [0.70, 0.72, 0.68],
    "softmax_readout":     [0.78, 0.76, 0.80],
    "dirichlet_readout":   [0.91, 0.89, 0.93],
}

SPEARMAN_THINNESS_BY_CELL: dict[str, list[float]] = {
    "canonical":           [0.72, 0.70, 0.74],
    "uniform_aggregator":  [0.60, 0.58, 0.62],
    "softmax_readout":     [0.65, 0.63, 0.67],
    "dirichlet_readout":   [0.80, 0.78, 0.82],
}

REQUIRED_KEYS = frozenset({
    "cell",
    "n_seeds",
    "polarity_acc_mean",
    "polarity_acc_std",
    "polarity_acc_ci_low",
    "polarity_acc_ci_high",
    "ece",
    "brier",
    "aurc",
    "sel_acc_at_50",
    "sel_acc_at_70",
    "sel_acc_at_90",
    "spearman_thinness",
    "spearman_thinness_ci_low",
    "spearman_thinness_ci_high",
    "config",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_per_cell_jsons(output_dir: Path) -> None:
    """Write 12 fake per-cell JSON files (4 cells × 3 seeds) into output_dir."""
    config_by_cell = {
        "canonical":           {"fusion_rule": "top1",    "decisive_top_k": 2, "coherence_weight": 1.0},
        "uniform_aggregator":  {"fusion_rule": "murphy",  "decisive_top_k": 3, "coherence_weight": 0.5},
        "softmax_readout":     {"fusion_rule": "yager",   "decisive_top_k": 2, "coherence_weight": 0.2},
        "dirichlet_readout":   {"fusion_rule": "dempster","decisive_top_k": 1, "coherence_weight": 0.0},
    }

    for cell in CELLS:
        for i, seed in enumerate(SEEDS):
            record = {
                "cell": cell,
                "seed": seed,
                "config": config_by_cell[cell],
                "polarity_acc": POLARITY_ACC_BY_CELL[cell][i],
                "ece": 0.05,
                "brier": 0.10,
                "aurc": 0.08,
                "sel_acc_at_50": 0.90,
                "sel_acc_at_70": 0.88,
                "sel_acc_at_90": 0.85,
                "spearman_thinness": SPEARMAN_THINNESS_BY_CELL[cell][i],
            }
            fname = f"{cell}__seed={seed}.json"
            (output_dir / fname).write_text(json.dumps(record))


def _hand_computed_mean(cell: str) -> float:
    """Arithmetic mean of polarity_acc for the three seeds of a cell."""
    vals = POLARITY_ACC_BY_CELL[cell]
    return sum(vals) / len(vals)


def _hand_computed_std(cell: str) -> float:
    """Population standard deviation of polarity_acc for the three seeds."""
    vals = POLARITY_ACC_BY_CELL[cell]
    mean = _hand_computed_mean(cell)
    variance = sum((v - mean) ** 2 for v in vals) / len(vals)
    return math.sqrt(variance)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def per_cell_dir(tmp_path: Path) -> Path:
    """Return a temporary directory containing 12 per-cell JSON files."""
    _write_per_cell_jsons(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Test 1 — output is a list with one row per cell
# ---------------------------------------------------------------------------


def test_aggregate_returns_list_of_dicts(per_cell_dir: Path) -> None:
    """``aggregate_runs(input_dir)`` returns a list of dicts, one per cell.

    Phase RED: ModuleNotFoundError because reference_v2.experiments.aggregate
    does not exist yet.

    Phase GREEN: the function reads all JSON files, groups by cell, and
    returns exactly one summary dict per unique cell.
    """
    result = aggregate_runs(per_cell_dir)

    assert isinstance(result, list), (
        f"aggregate_runs must return a list, got {type(result).__name__}"
    )
    assert len(result) == len(CELLS), (
        f"expected {len(CELLS)} rows (one per cell), got {len(result)}"
    )
    for row in result:
        assert isinstance(row, dict), (
            f"each row must be a dict, got {type(row).__name__}"
        )


# ---------------------------------------------------------------------------
# Test 2 — all 15 required keys are present in every row
# ---------------------------------------------------------------------------


def test_aggregate_schema_has_required_keys(per_cell_dir: Path) -> None:
    """Every row in the aggregate output contains the 15 required schema keys.

    Phase RED: ModuleNotFoundError.

    Phase GREEN: the aggregate dict for each cell must contain exactly the
    documented set of keys; no keys may be missing.
    """
    result = aggregate_runs(per_cell_dir)

    for row in result:
        missing = REQUIRED_KEYS - set(row.keys())
        assert not missing, (
            f"row for cell {row.get('cell', '<unknown>')!r} is missing keys: "
            f"{sorted(missing)}"
        )


# ---------------------------------------------------------------------------
# Test 3 — n_seeds equals the number of seeds per cell
# ---------------------------------------------------------------------------


def test_aggregate_n_seeds_correct(per_cell_dir: Path) -> None:
    """Each row reports n_seeds == 3 (the number of seed files for that cell).

    Phase RED: ModuleNotFoundError.

    Phase GREEN: the aggregator must count how many JSON files existed per
    cell and store that count in ``n_seeds``.
    """
    result = aggregate_runs(per_cell_dir)

    for row in result:
        assert row["n_seeds"] == len(SEEDS), (
            f"cell {row['cell']!r}: expected n_seeds={len(SEEDS)}, "
            f"got {row['n_seeds']}"
        )


# ---------------------------------------------------------------------------
# Test 4 — polarity_acc_mean matches hand-computed average
# ---------------------------------------------------------------------------


def test_aggregate_polarity_acc_mean(per_cell_dir: Path) -> None:
    """``polarity_acc_mean`` matches the hand-computed arithmetic mean.

    Phase RED: ModuleNotFoundError.

    Phase GREEN: the aggregator averages polarity_acc over all seeds for each
    cell. Tolerance is 1e-9 (floating-point arithmetic, no approximation).
    """
    result = aggregate_runs(per_cell_dir)
    rows_by_cell = {row["cell"]: row for row in result}

    for cell in CELLS:
        expected_mean = _hand_computed_mean(cell)
        actual_mean = rows_by_cell[cell]["polarity_acc_mean"]
        assert abs(actual_mean - expected_mean) < 1e-9, (
            f"cell {cell!r}: polarity_acc_mean={actual_mean:.6f} "
            f"but expected {expected_mean:.6f} "
            f"(values={POLARITY_ACC_BY_CELL[cell]})"
        )


# ---------------------------------------------------------------------------
# Test 5 — polarity_acc_std matches hand-computed population std
# ---------------------------------------------------------------------------


def test_aggregate_polarity_acc_std(per_cell_dir: Path) -> None:
    """``polarity_acc_std`` matches the hand-computed standard deviation.

    Phase RED: ModuleNotFoundError.

    Phase GREEN: the aggregator computes std (population or sample — the
    test accepts either ddof=0 or ddof=1, verified to 1e-6 tolerance after
    normalising for the known values). The primary contract is that the value
    is finite and non-negative.
    """
    result = aggregate_runs(per_cell_dir)
    rows_by_cell = {row["cell"]: row for row in result}

    for cell in CELLS:
        actual_std = rows_by_cell[cell]["polarity_acc_std"]
        assert isinstance(actual_std, float), (
            f"cell {cell!r}: polarity_acc_std must be float, got {type(actual_std).__name__}"
        )
        assert math.isfinite(actual_std), (
            f"cell {cell!r}: polarity_acc_std is not finite: {actual_std}"
        )
        assert actual_std >= 0.0, (
            f"cell {cell!r}: polarity_acc_std is negative: {actual_std}"
        )
        # Verify against population std (ddof=0); allow small floating-point gap.
        expected_std = _hand_computed_std(cell)
        assert abs(actual_std - expected_std) < 1e-6, (
            f"cell {cell!r}: polarity_acc_std={actual_std:.8f} "
            f"but expected population std={expected_std:.8f}"
        )


# ---------------------------------------------------------------------------
# Test 6 — CI bounds are ordered and finite
# ---------------------------------------------------------------------------


def test_aggregate_ci_bounds_are_valid(per_cell_dir: Path) -> None:
    """CI bounds satisfy ci_low <= mean <= ci_high and are finite floats.

    Phase RED: ModuleNotFoundError.

    Phase GREEN: the aggregator computes a 95% CI for both polarity_acc and
    spearman_thinness. The CI must bracket the point estimate and have
    finite bounds.
    """
    result = aggregate_runs(per_cell_dir)

    for row in result:
        cell = row["cell"]

        # polarity_acc CI
        pa_low = row["polarity_acc_ci_low"]
        pa_mean = row["polarity_acc_mean"]
        pa_high = row["polarity_acc_ci_high"]
        assert math.isfinite(pa_low) and math.isfinite(pa_high), (
            f"cell {cell!r}: polarity_acc CI bounds are not finite: "
            f"({pa_low}, {pa_high})"
        )
        assert pa_low <= pa_mean <= pa_high, (
            f"cell {cell!r}: polarity_acc CI ({pa_low:.4f}, {pa_high:.4f}) "
            f"does not bracket mean {pa_mean:.4f}"
        )

        # spearman_thinness CI
        st_low = row["spearman_thinness_ci_low"]
        st_val = row["spearman_thinness"]
        st_high = row["spearman_thinness_ci_high"]
        assert math.isfinite(st_low) and math.isfinite(st_high), (
            f"cell {cell!r}: spearman_thinness CI bounds are not finite: "
            f"({st_low}, {st_high})"
        )
        assert st_low <= st_val <= st_high, (
            f"cell {cell!r}: spearman_thinness CI ({st_low:.4f}, {st_high:.4f}) "
            f"does not bracket point estimate {st_val:.4f}"
        )


if __name__ == "__main__":
    # Allow direct invocation; pytest still runs all tests via -v.
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        tmpdir = Path(d)
        _write_per_cell_jsons(tmpdir)
        test_aggregate_returns_list_of_dicts(tmpdir)
        test_aggregate_schema_has_required_keys(tmpdir)
        test_aggregate_n_seeds_correct(tmpdir)
        test_aggregate_polarity_acc_mean(tmpdir)
        test_aggregate_polarity_acc_std(tmpdir)
        test_aggregate_ci_bounds_are_valid(tmpdir)
    print("All tests passed.")

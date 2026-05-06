"""
pareto.py — Aggregate panel figure (infon wrapper).

Reads aggregate_panel.json and produces a scatter plot of accuracy vs AURC
with a Pareto-front polyline (lower AURC and higher accuracy is better).

The JSON schema is identical to paper/tests/fixtures/aggregate_panel.json,
so we reuse the paper figure implementation directly.

Usage:
    python3 experiments/figures/pareto.py \
        --input experiments/results/aggregate_panel.json \
        --output experiments/figures/pareto.pdf
"""

import sys
from pathlib import Path

# Add repo root so paper.figures is importable
_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from paper.figures.pareto import main  # noqa: E402

if __name__ == "__main__":
    main()

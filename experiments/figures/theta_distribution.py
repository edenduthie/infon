"""
theta_distribution.py — H2 panel figure (infon wrapper).

Reads h2_panel.json and produces a box plot of m_theta by ground-truth class,
one subplot per system.

The JSON schema is identical to paper/tests/fixtures/h2_panel.json,
so we reuse the paper figure implementation directly.

Usage:
    python3 experiments/figures/theta_distribution.py \
        --input experiments/results/h2_panel.json \
        --output experiments/figures/theta_distribution.pdf
"""

import sys
from pathlib import Path

# Add repo root so paper.figures is importable
_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from paper.figures.theta_distribution import main  # noqa: E402

if __name__ == "__main__":
    main()

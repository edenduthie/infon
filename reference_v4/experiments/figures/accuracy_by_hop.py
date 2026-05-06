"""
accuracy_by_hop.py — H1 panel figure (reference_v4 wrapper).

Reads h1_panel.json and produces a grouped bar chart of accuracy vs num_hops,
one series per system.

The JSON schema is identical to paper/tests/fixtures/h1_panel.json,
so we reuse the paper figure implementation directly.

Usage:
    python3 reference_v4/experiments/figures/accuracy_by_hop.py \
        --input reference_v4/experiments/results/h1_panel.json \
        --output reference_v4/experiments/figures/accuracy_by_hop.pdf
"""

import sys
from pathlib import Path

# Add repo root so paper.figures is importable
_REPO_ROOT = Path(__file__).parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from paper.figures.accuracy_by_hop import main  # noqa: E402

if __name__ == "__main__":
    main()

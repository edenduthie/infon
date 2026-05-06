"""
reliability_diagrams.py — H3 panel figure (infon wrapper).

Reads h3_panel.json and produces 15-bin reliability (calibration) diagrams,
one subplot per system.

The JSON schema is identical to paper/tests/fixtures/h3_panel.json,
so we reuse the paper figure implementation directly.

Usage:
    python3 experiments/figures/reliability_diagrams.py \
        --input experiments/results/h3_panel.json \
        --output experiments/figures/reliability_diagrams.pdf
"""

import sys
from pathlib import Path

# Add repo root so paper.figures is importable
_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from paper.figures.reliability_diagrams import main  # noqa: E402

if __name__ == "__main__":
    main()

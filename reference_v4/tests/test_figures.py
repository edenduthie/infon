"""Tests for reference_v4/experiments/figures/*.py scripts.

Tests:
  - Each figure script exists and can be imported.
  - pareto.py with --input and --output args produces a PDF file > 1 KB.
  - Running pareto.py twice produces the same output (deterministic).

Run from repo root:
    pytest reference_v4/tests/test_figures.py -v
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
FIGURES_DIR = REPO_ROOT / "reference_v4" / "experiments" / "figures"
PAPER_FIXTURES = REPO_ROOT / "paper" / "tests" / "fixtures"


# ---------------------------------------------------------------------------
# Import smoke tests
# ---------------------------------------------------------------------------

def test_figures_init_importable():
    """reference_v4.experiments.figures package must be importable."""
    import importlib
    mod = importlib.import_module("reference_v4.experiments.figures")
    assert mod is not None


@pytest.mark.parametrize("script_name", [
    "accuracy_by_hop",
    "theta_distribution",
    "reliability_diagrams",
    "pareto",
])
def test_figure_module_importable(script_name):
    """Each figure sub-module must be importable."""
    import importlib
    mod = importlib.import_module(f"reference_v4.experiments.figures.{script_name}")
    assert mod is not None


# ---------------------------------------------------------------------------
# pareto.py: functional test
# ---------------------------------------------------------------------------

class TestPareto:
    def test_pareto_creates_pdf(self, tmp_path):
        """pareto.py --input <fixture> --output <tmp> must create a PDF > 1 KB."""
        input_path = PAPER_FIXTURES / "aggregate_panel.json"
        output_path = tmp_path / "pareto.pdf"

        result = subprocess.run(
            [
                sys.executable,
                str(FIGURES_DIR / "pareto.py"),
                "--input", str(input_path),
                "--output", str(output_path),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"pareto.py failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert output_path.exists(), "pareto.py must create the output PDF"
        size = output_path.stat().st_size
        assert size > 1024, f"PDF too small: {size} bytes (expected > 1 KB)"

    def test_pareto_deterministic(self, tmp_path):
        """Running pareto.py twice must produce identical output."""
        input_path = PAPER_FIXTURES / "aggregate_panel.json"
        output1 = tmp_path / "pareto1.pdf"
        output2 = tmp_path / "pareto2.pdf"

        for output_path in (output1, output2):
            result = subprocess.run(
                [
                    sys.executable,
                    str(FIGURES_DIR / "pareto.py"),
                    "--input", str(input_path),
                    "--output", str(output_path),
                ],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0

        size1 = output1.stat().st_size
        size2 = output2.stat().st_size
        assert size1 == size2, (
            f"Pareto PDF sizes differ between runs: {size1} vs {size2} bytes"
        )


# ---------------------------------------------------------------------------
# accuracy_by_hop.py: functional test
# ---------------------------------------------------------------------------

class TestAccuracyByHop:
    def test_accuracy_by_hop_creates_pdf(self, tmp_path):
        """accuracy_by_hop.py --input <fixture> --output <tmp> must create a PDF > 1 KB."""
        input_path = PAPER_FIXTURES / "h1_panel.json"
        output_path = tmp_path / "accuracy_by_hop.pdf"

        result = subprocess.run(
            [
                sys.executable,
                str(FIGURES_DIR / "accuracy_by_hop.py"),
                "--input", str(input_path),
                "--output", str(output_path),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"accuracy_by_hop.py failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert output_path.exists()
        assert output_path.stat().st_size > 1024


# ---------------------------------------------------------------------------
# theta_distribution.py: functional test
# ---------------------------------------------------------------------------

class TestThetaDistribution:
    def test_theta_distribution_creates_pdf(self, tmp_path):
        """theta_distribution.py --input <fixture> --output <tmp> must create a PDF > 1 KB."""
        input_path = PAPER_FIXTURES / "h2_panel.json"
        output_path = tmp_path / "theta_distribution.pdf"

        result = subprocess.run(
            [
                sys.executable,
                str(FIGURES_DIR / "theta_distribution.py"),
                "--input", str(input_path),
                "--output", str(output_path),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"theta_distribution.py failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert output_path.exists()
        assert output_path.stat().st_size > 1024


# ---------------------------------------------------------------------------
# reliability_diagrams.py: functional test
# ---------------------------------------------------------------------------

class TestReliabilityDiagrams:
    def test_reliability_diagrams_creates_pdf(self, tmp_path):
        """reliability_diagrams.py --input <fixture> --output <tmp> must create a PDF > 1 KB."""
        input_path = PAPER_FIXTURES / "h3_panel.json"
        output_path = tmp_path / "reliability_diagrams.pdf"

        result = subprocess.run(
            [
                sys.executable,
                str(FIGURES_DIR / "reliability_diagrams.py"),
                "--input", str(input_path),
                "--output", str(output_path),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"reliability_diagrams.py failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert output_path.exists()
        assert output_path.stat().st_size > 1024

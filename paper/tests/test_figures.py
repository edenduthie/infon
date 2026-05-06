"""
test_figures.py

For each figure script:
  1. Load fixture JSON from paper/tests/fixtures/
  2. Invoke the script with --input and --output pointing at fixture/tmp paths
  3. Assert the output PDF exists and is > 1 KB
  4. Invoke a second time and assert the file is recreated with the same size
     (within a small tolerance) — a proxy for deterministic output.

PDF timestamp bytes are stripped before size comparison to guard against
any metadata variation between runs.
"""

import importlib.util
import re
import sys
from pathlib import Path

import pytest

# Paths
FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
MIN_PDF_BYTES = 1024  # 1 KB


def _load_script(script_path: Path):
    """Dynamically load a figure script as a module."""
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _strip_pdf_timestamps(pdf_bytes: bytes) -> bytes:
    """Remove /CreationDate and /ModDate values from PDF bytes for comparison."""
    cleaned = re.sub(rb"/CreationDate\s*\([^)]*\)", b"/CreationDate()", pdf_bytes)
    cleaned = re.sub(rb"/ModDate\s*\([^)]*\)", b"/ModDate()", cleaned)
    return cleaned


def _run_figure_script(script_path: Path, input_path: Path, output_path: Path) -> None:
    """Invoke a figure script's main() with patched sys.argv."""
    original_argv = sys.argv[:]
    try:
        sys.argv = [
            str(script_path),
            "--input", str(input_path),
            "--output", str(output_path),
        ]
        module = _load_script(script_path)
        module.main()
    finally:
        sys.argv = original_argv


# ---------------------------------------------------------------------------
# accuracy_by_hop
# ---------------------------------------------------------------------------

class TestAccuracyByHop:
    SCRIPT = FIGURES_DIR / "accuracy_by_hop.py"
    FIXTURE = FIXTURES_DIR / "h1_panel.json"

    def test_pdf_created_and_large_enough(self, tmp_path: Path) -> None:
        output = tmp_path / "accuracy_by_hop.pdf"
        _run_figure_script(self.SCRIPT, self.FIXTURE, output)
        assert output.exists(), "accuracy_by_hop.pdf was not created"
        assert output.stat().st_size > MIN_PDF_BYTES, (
            f"accuracy_by_hop.pdf too small: {output.stat().st_size} bytes"
        )

    def test_second_run_same_size(self, tmp_path: Path) -> None:
        output = tmp_path / "accuracy_by_hop.pdf"
        _run_figure_script(self.SCRIPT, self.FIXTURE, output)
        first_bytes = _strip_pdf_timestamps(output.read_bytes())

        output.unlink()
        _run_figure_script(self.SCRIPT, self.FIXTURE, output)
        second_bytes = _strip_pdf_timestamps(output.read_bytes())

        assert abs(len(first_bytes) - len(second_bytes)) < 512, (
            f"PDF size changed significantly between runs: "
            f"{len(first_bytes)} vs {len(second_bytes)}"
        )


# ---------------------------------------------------------------------------
# theta_distribution
# ---------------------------------------------------------------------------

class TestThetaDistribution:
    SCRIPT = FIGURES_DIR / "theta_distribution.py"
    FIXTURE = FIXTURES_DIR / "h2_panel.json"

    def test_pdf_created_and_large_enough(self, tmp_path: Path) -> None:
        output = tmp_path / "theta_distribution.pdf"
        _run_figure_script(self.SCRIPT, self.FIXTURE, output)
        assert output.exists(), "theta_distribution.pdf was not created"
        assert output.stat().st_size > MIN_PDF_BYTES, (
            f"theta_distribution.pdf too small: {output.stat().st_size} bytes"
        )

    def test_second_run_same_size(self, tmp_path: Path) -> None:
        output = tmp_path / "theta_distribution.pdf"
        _run_figure_script(self.SCRIPT, self.FIXTURE, output)
        first_bytes = _strip_pdf_timestamps(output.read_bytes())

        output.unlink()
        _run_figure_script(self.SCRIPT, self.FIXTURE, output)
        second_bytes = _strip_pdf_timestamps(output.read_bytes())

        assert abs(len(first_bytes) - len(second_bytes)) < 512, (
            f"PDF size changed significantly between runs: "
            f"{len(first_bytes)} vs {len(second_bytes)}"
        )


# ---------------------------------------------------------------------------
# reliability_diagrams
# ---------------------------------------------------------------------------

class TestReliabilityDiagrams:
    SCRIPT = FIGURES_DIR / "reliability_diagrams.py"
    FIXTURE = FIXTURES_DIR / "h3_panel.json"

    def test_pdf_created_and_large_enough(self, tmp_path: Path) -> None:
        output = tmp_path / "reliability_diagrams.pdf"
        _run_figure_script(self.SCRIPT, self.FIXTURE, output)
        assert output.exists(), "reliability_diagrams.pdf was not created"
        assert output.stat().st_size > MIN_PDF_BYTES, (
            f"reliability_diagrams.pdf too small: {output.stat().st_size} bytes"
        )

    def test_second_run_same_size(self, tmp_path: Path) -> None:
        output = tmp_path / "reliability_diagrams.pdf"
        _run_figure_script(self.SCRIPT, self.FIXTURE, output)
        first_bytes = _strip_pdf_timestamps(output.read_bytes())

        output.unlink()
        _run_figure_script(self.SCRIPT, self.FIXTURE, output)
        second_bytes = _strip_pdf_timestamps(output.read_bytes())

        assert abs(len(first_bytes) - len(second_bytes)) < 512, (
            f"PDF size changed significantly between runs: "
            f"{len(first_bytes)} vs {len(second_bytes)}"
        )


# ---------------------------------------------------------------------------
# pareto
# ---------------------------------------------------------------------------

class TestPareto:
    SCRIPT = FIGURES_DIR / "pareto.py"
    FIXTURE = FIXTURES_DIR / "aggregate_panel.json"

    def test_pdf_created_and_large_enough(self, tmp_path: Path) -> None:
        output = tmp_path / "pareto.pdf"
        _run_figure_script(self.SCRIPT, self.FIXTURE, output)
        assert output.exists(), "pareto.pdf was not created"
        assert output.stat().st_size > MIN_PDF_BYTES, (
            f"pareto.pdf too small: {output.stat().st_size} bytes"
        )

    def test_second_run_same_size(self, tmp_path: Path) -> None:
        output = tmp_path / "pareto.pdf"
        _run_figure_script(self.SCRIPT, self.FIXTURE, output)
        first_bytes = _strip_pdf_timestamps(output.read_bytes())

        output.unlink()
        _run_figure_script(self.SCRIPT, self.FIXTURE, output)
        second_bytes = _strip_pdf_timestamps(output.read_bytes())

        assert abs(len(first_bytes) - len(second_bytes)) < 512, (
            f"PDF size changed significantly between runs: "
            f"{len(first_bytes)} vs {len(second_bytes)}"
        )

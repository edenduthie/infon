"""
test_numbers_audit.py

Tests for paper/tools/check_numbers_audit.py.

Scenarios:
  - Pass: direct match (value=0.87, json acc=0.87)
  - Pass: percentage match (value=87.0, json acc=0.87, tolerance=0.5)
  - Fail: clear mismatch (value=0.87, json acc=0.72, tolerance=0.005)
         -> exits non-zero, stderr mentions "delta" or the offending row
  - Pass: header-only audit table exits 0 (vacuously)
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# Path to the checker script
CHECKER = Path(__file__).parent.parent / "tools" / "check_numbers_audit.py"


def _build_audit_md(
    value: str,
    json_relpath: str,
    json_key: str,
    tolerance: str,
    section: str = "S5",
    location: str = "p.7",
    description: str = "test value",
) -> str:
    """Return a complete numbers_audit.md string with one data row."""
    return textwrap.dedent(f"""\
        # Numbers Audit

        | value | section | location | json_path | json_key | tolerance | description |
        |-------|---------|----------|-----------|----------|-----------|-------------|
        | {value} | {section} | {location} | {json_relpath} | {json_key} | {tolerance} | {description} |
    """)


def _run_checker(tmp_path: Path, audit_text: str) -> subprocess.CompletedProcess:
    """
    Write audit text to tmp_path/paper/numbers_audit.md and run the checker
    with --repo-root pointing at tmp_path so it loads from the temp tree.
    """
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir(exist_ok=True)
    (paper_dir / "numbers_audit.md").write_text(audit_text, encoding="utf-8")

    return subprocess.run(
        [sys.executable, str(CHECKER), "--repo-root", str(tmp_path)],
        capture_output=True,
        text=True,
    )


class TestPassDirectMatch:
    """value=0.87, JSON acc=0.87, tolerance=0.005 — should exit 0."""

    def test_exit_zero(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        fixture_json = data_dir / "results.json"
        fixture_json.write_text(json.dumps({"acc": 0.87}), encoding="utf-8")

        audit_text = _build_audit_md(
            value="0.87",
            json_relpath="data/results.json",
            json_key="acc",
            tolerance="0.005",
        )
        result = _run_checker(tmp_path, audit_text)
        assert result.returncode == 0, (
            f"Expected exit 0 (direct match). stdout={result.stdout!r} stderr={result.stderr!r}"
        )


class TestPassPercentageMatch:
    """value=87.0 (percentage), JSON acc=0.87, tolerance=0.5 — should exit 0."""

    def test_exit_zero(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        fixture_json = data_dir / "results.json"
        fixture_json.write_text(json.dumps({"acc": 0.87}), encoding="utf-8")

        audit_text = _build_audit_md(
            value="87.0",
            json_relpath="data/results.json",
            json_key="acc",
            tolerance="0.5",
        )
        result = _run_checker(tmp_path, audit_text)
        assert result.returncode == 0, (
            f"Expected exit 0 (percentage match). stdout={result.stdout!r} stderr={result.stderr!r}"
        )


class TestFailMismatch:
    """value=0.87, JSON acc=0.72, tolerance=0.005 — should exit non-zero."""

    def test_exit_nonzero(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        fixture_json = data_dir / "results.json"
        fixture_json.write_text(json.dumps({"acc": 0.72}), encoding="utf-8")

        audit_text = _build_audit_md(
            value="0.87",
            json_relpath="data/results.json",
            json_key="acc",
            tolerance="0.005",
        )
        result = _run_checker(tmp_path, audit_text)
        assert result.returncode != 0, (
            f"Expected non-zero exit (mismatch). stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        combined = result.stdout + result.stderr
        assert "delta" in combined.lower() or "mismatch" in combined.lower(), (
            f"Expected 'delta' or 'mismatch' in output. Got: {combined!r}"
        )


class TestEmptyAudit:
    """Header-only audit table should exit 0 (vacuously passes)."""

    def test_header_only_exits_zero(self, tmp_path: Path) -> None:
        audit_text = (
            "# Numbers Audit\n\n"
            "| value | section | location | json_path | json_key | tolerance | description |\n"
            "|-------|---------|----------|-----------|----------|-----------|-------------|\n"
        )
        result = _run_checker(tmp_path, audit_text)
        assert result.returncode == 0, (
            f"Expected exit 0 for empty audit. stdout={result.stdout!r} stderr={result.stderr!r}"
        )

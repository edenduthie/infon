"""
test_reproducer.py

Tests for paper/reproducer.sh.

Scenarios:
  - dry-run: the script lists all stage names and exits 0.
  - resumability: create a stage_1 checkpoint sentinel and verify the script
    skips stage_1 on re-run (i.e., the sentinel was pre-existing and no
    install command is emitted for stage_1 in dry-run mode from a fresh state,
    and in live mode the sentinel prevents re-execution).
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPRODUCER = Path(__file__).parent.parent / "reproducer.sh"
REPO_ROOT = Path(__file__).parent.parent.parent


class TestDryRun:
    """--dry-run should list all stage names and exit 0."""

    def test_exits_zero(self) -> None:
        result = subprocess.run(
            ["bash", str(REPRODUCER), "--dry-run"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0, (
            f"Expected exit 0 with --dry-run. "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )

    def test_lists_all_stage_names(self) -> None:
        result = subprocess.run(
            ["bash", str(REPRODUCER), "--dry-run"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        combined = result.stdout + result.stderr
        expected_stages = [
            "stage_1",
            "stage_2",
            "stage_3",
            "stage_4",
            "stage_5",
            "stage_6",
        ]
        for stage in expected_stages:
            assert stage in combined, (
                f"Expected '{stage}' in dry-run output. Got:\n{combined}"
            )

    def test_stage_descriptions_present(self) -> None:
        """Verify meaningful stage descriptions appear in dry-run output."""
        result = subprocess.run(
            ["bash", str(REPRODUCER), "--dry-run"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        combined = result.stdout + result.stderr
        descriptions = [
            "install",
            "figure",
            "lint",
            "pdf",
        ]
        for desc in descriptions:
            assert desc.lower() in combined.lower(), (
                f"Expected '{desc}' in dry-run output. Got:\n{combined}"
            )


class TestResumability:
    """
    Create a stage_1 checkpoint sentinel and verify the script skips stage_1.

    We use --dry-run in a temp directory with a pre-existing sentinel so we
    don't actually invoke pip. Instead we test the real execution path by
    checking that when the sentinel exists, the script prints "already done"
    for stage 1.
    """

    def test_skips_stage_with_existing_sentinel(self, tmp_path: Path) -> None:
        # Set up a minimal fake repo root that the reproducer can run from
        paper_dir = tmp_path / "paper"
        paper_dir.mkdir()
        figures_dir = paper_dir / "figures"
        figures_dir.mkdir()

        # Copy the reproducer into the temp paper dir
        import shutil
        shutil.copy(str(REPRODUCER), str(paper_dir / "reproducer.sh"))

        # Create the stage_1 checkpoint sentinel
        checkpoint_dir = tmp_path / ".reproducer_checkpoints"
        checkpoint_dir.mkdir()
        (checkpoint_dir / ".stage_1_done").touch()

        # Run the reproducer with --dry-run from the temp repo root.
        # Even with --dry-run, the sentinel check happens before the dry-run
        # early-exit, so we test via a live run that only does stage_1 logic.
        #
        # Strategy: run without --dry-run but patch the environment so that
        # stages 2-6 are all already done too, making the script complete
        # without actually invoking pip/make/latexmk.
        for stage_num in range(1, 7):
            (checkpoint_dir / f".stage_{stage_num}_done").touch()

        result = subprocess.run(
            ["bash", str(paper_dir / "reproducer.sh")],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
        )
        combined = result.stdout + result.stderr
        assert result.returncode == 0, (
            f"Expected exit 0 when all stages already done. "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        # All stages should be reported as skipped
        assert "already done" in combined, (
            f"Expected 'already done' skip message. Got:\n{combined}"
        )
        # stage_1 specifically
        assert "stage_1 already done" in combined or "stage_1: already done" in combined or \
               "stage_1 already done" in combined.replace("==> ", ""), (
            f"Expected stage_1 skip message. Got:\n{combined}"
        )

    def test_stage1_sentinel_prevents_pip_invocation(self, tmp_path: Path) -> None:
        """
        When .stage_1_done sentinel exists, the script must not attempt to
        run pip for stage_1. We verify by checking 'pip install' does NOT
        appear in the output when the sentinel is present but we are in
        dry-run mode (dry-run prints everything it would do).

        NOTE: In dry-run mode the sentinel check is bypassed (dry-run prints
        all stages regardless). So this test verifies the sentinel logic in
        live mode by marking all stages done and confirming pip is not called.
        """
        import shutil
        paper_dir = tmp_path / "paper"
        paper_dir.mkdir()
        figures_dir = paper_dir / "figures"
        figures_dir.mkdir()
        shutil.copy(str(REPRODUCER), str(paper_dir / "reproducer.sh"))

        # Mark only stage_1 as done
        checkpoint_dir = tmp_path / ".reproducer_checkpoints"
        checkpoint_dir.mkdir()
        (checkpoint_dir / ".stage_1_done").touch()
        # Mark remaining stages done too to avoid running pip/make
        for stage_num in range(2, 7):
            (checkpoint_dir / f".stage_{stage_num}_done").touch()

        result = subprocess.run(
            ["bash", str(paper_dir / "reproducer.sh")],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
        )
        combined = result.stdout + result.stderr
        assert result.returncode == 0, (
            f"Expected exit 0. stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        # pip install should NOT appear because stage_1 is already done
        assert "pip install" not in combined, (
            f"Expected 'pip install' to be skipped when sentinel exists. Got:\n{combined}"
        )

"""
test_arxiv_tarball.py

Tests for paper/arxiv-submission/build.sh.

Scenarios:
  - Creates a minimal temp paper structure (main.tex with one \\input and one
    \\includegraphics), runs build.sh on it, and verifies the resulting tarball:
      * Contains expected files (main.tex, the input'd .tex file)
      * Does NOT contain .aux, .log, or .bib files
      * Total size <= 50 MB
"""

import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

BUILD_SH = Path(__file__).parent.parent / "arxiv-submission" / "build.sh"


def _build_minimal_paper(paper_dir: Path, figure_file: str = "figures/fig1.pdf") -> None:
    """
    Create a minimal paper directory structure for testing.

    Produces:
      paper_dir/
        main.tex               (references sections/intro.tex and a figure)
        sections/intro.tex
        figures/fig1.pdf       (or whatever figure_file specifies)
        references.bbl         (minimal .bbl)
        junk.aux               (should NOT appear in tarball)
        junk.log               (should NOT appear in tarball)
        junk.bib               (should NOT appear in tarball)
    """
    paper_dir.mkdir(parents=True, exist_ok=True)
    (paper_dir / "sections").mkdir(exist_ok=True)
    (paper_dir / "figures").mkdir(exist_ok=True)

    # main.tex
    (paper_dir / "main.tex").write_text(
        "\\documentclass{article}\n"
        "\\usepackage{graphicx}\n"
        "\\begin{document}\n"
        "\\input{sections/intro}\n"
        "\\includegraphics{figures/fig1.pdf}\n"
        "\\end{document}\n",
        encoding="utf-8",
    )

    # sections/intro.tex
    (paper_dir / "sections" / "intro.tex").write_text(
        "This is the introduction.\n",
        encoding="utf-8",
    )

    # figures/fig1.pdf (fake — just needs to exist)
    (paper_dir / "figures" / "fig1.pdf").write_bytes(b"%PDF-1.4 fake\n")

    # references.bbl (minimal)
    (paper_dir / "references.bbl").write_text(
        "\\begin{thebibliography}{99}\n\\end{thebibliography}\n",
        encoding="utf-8",
    )

    # Junk files that must NOT appear in the tarball
    (paper_dir / "main.aux").write_text("aux junk\n", encoding="utf-8")
    (paper_dir / "main.log").write_text("log junk\n", encoding="utf-8")
    (paper_dir / "references.bib").write_text(
        "@article{foo, title={Foo}}\n", encoding="utf-8"
    )


class TestArxivTarball:
    def _run_build(self, paper_dir: Path, out_tarball: Path) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(BUILD_SH), "--paper-dir", str(paper_dir), "--out", str(out_tarball)],
            capture_output=True,
            text=True,
        )

    def test_tarball_created(self, tmp_path: Path) -> None:
        """build.sh must create paper.tar.gz."""
        paper_dir = tmp_path / "paper"
        _build_minimal_paper(paper_dir)
        out = tmp_path / "paper.tar.gz"

        result = self._run_build(paper_dir, out)
        assert result.returncode == 0, (
            f"build.sh failed.\nstdout={result.stdout}\nstderr={result.stderr}"
        )
        assert out.exists(), "paper.tar.gz was not created"

    def test_main_tex_in_tarball(self, tmp_path: Path) -> None:
        """main.tex must be present in the tarball."""
        paper_dir = tmp_path / "paper"
        _build_minimal_paper(paper_dir)
        out = tmp_path / "paper.tar.gz"

        self._run_build(paper_dir, out)

        with tarfile.open(out, "r:gz") as tf:
            names = tf.getnames()
        # Normalise: strip leading ./
        names_clean = {n.lstrip("./") for n in names}
        assert "main.tex" in names_clean, (
            f"main.tex not found in tarball. Contents: {sorted(names_clean)}"
        )

    def test_input_tex_in_tarball(self, tmp_path: Path) -> None:
        """sections/intro.tex (referenced by \\input) must be in the tarball."""
        paper_dir = tmp_path / "paper"
        _build_minimal_paper(paper_dir)
        out = tmp_path / "paper.tar.gz"

        self._run_build(paper_dir, out)

        with tarfile.open(out, "r:gz") as tf:
            names = tf.getnames()
        names_clean = {n.lstrip("./") for n in names}
        assert "sections/intro.tex" in names_clean, (
            f"sections/intro.tex not in tarball. Contents: {sorted(names_clean)}"
        )

    def test_bbl_in_tarball(self, tmp_path: Path) -> None:
        """references.bbl must be included (not .bib)."""
        paper_dir = tmp_path / "paper"
        _build_minimal_paper(paper_dir)
        out = tmp_path / "paper.tar.gz"

        self._run_build(paper_dir, out)

        with tarfile.open(out, "r:gz") as tf:
            names = tf.getnames()
        names_clean = {n.lstrip("./") for n in names}
        assert "references.bbl" in names_clean, (
            f"references.bbl not in tarball. Contents: {sorted(names_clean)}"
        )

    def test_excluded_files_absent(self, tmp_path: Path) -> None:
        """
        .aux, .log, .bib files must NOT appear in the tarball.
        """
        paper_dir = tmp_path / "paper"
        _build_minimal_paper(paper_dir)
        out = tmp_path / "paper.tar.gz"

        self._run_build(paper_dir, out)

        with tarfile.open(out, "r:gz") as tf:
            names = tf.getnames()

        forbidden_extensions = (".aux", ".log", ".bib")
        offenders = [n for n in names if any(n.endswith(ext) for ext in forbidden_extensions)]
        assert offenders == [], (
            f"Forbidden file(s) found in tarball: {offenders}"
        )

    def test_size_within_50mb(self, tmp_path: Path) -> None:
        """Tarball must be <= 50 MB."""
        paper_dir = tmp_path / "paper"
        _build_minimal_paper(paper_dir)
        out = tmp_path / "paper.tar.gz"

        self._run_build(paper_dir, out)

        size_mb = out.stat().st_size / (1024 * 1024)
        assert size_mb <= 50, (
            f"Tarball size {size_mb:.2f} MB exceeds 50 MB arXiv limit"
        )

    def test_build_exits_zero(self, tmp_path: Path) -> None:
        """build.sh must exit 0 on a valid paper structure."""
        paper_dir = tmp_path / "paper"
        _build_minimal_paper(paper_dir)
        out = tmp_path / "paper.tar.gz"

        result = self._run_build(paper_dir, out)
        assert result.returncode == 0, (
            f"build.sh returned non-zero exit code {result.returncode}.\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )

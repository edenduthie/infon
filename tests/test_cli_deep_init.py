"""End-to-end tests for the v0.1.2 default ``infon init`` behaviour: schema
discovery + docstring extraction + markdown extraction.

The default ``infon init`` (no flags) MUST:
  1. Run schema discovery so the kb has actor anchors (not just the eight
     built-in relations) — without these, conceptual queries can't activate
     anything besides ``raises``/``calls``/etc.
  2. Extract docstrings from .py files into text infons grounded in the
     source file. Conceptual queries like "how does X work" fall back to
     these instead of returning random AST junk.
  3. Extract sentences from .md/.rst/.txt files into text infons.
  4. Still do everything ``infon init`` did before (AST extraction,
     consolidation, .mcp.json, .gitignore).

``infon init --shallow`` MUST opt out of (1)-(3) and behave like the
v0.1.1 fast path.

These tests are real-stack: real subprocess, real DuckDB, real SPLADE
encoder. They are slow (~minute each) because schema discovery encodes
every line of the corpus through SPLADE — that's the explicit cost of
the new default and the reason for the opt-out.
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke


def _run_infon(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "infon.cli", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
        timeout=600,  # discovery is slow, give it room
    )


@pytest.fixture
def docstring_repo(tmp_path: Path) -> Path:
    """A tiny repo with a deliberately distinctive docstring so we can prove
    text extraction landed in the kb."""
    (tmp_path / "src").mkdir()

    # The docstring contains tokens we expect to find via search.
    (tmp_path / "src" / "spladey.py").write_text(textwrap.dedent('''
        """The spladey module computes sparse vectors via mlm logits."""

        class SpladeyEncoder:
            """Encodes text into thirty thousand dimensional sparse vectors.

            Uses log of one plus relu of the masked language model logits,
            max pooled across token positions.
            """

            def encode(self, text):
                """Returns a sparse activation dictionary."""
                return {}


        def make_encoder():
            """Factory that constructs a SpladeyEncoder instance."""
            return SpladeyEncoder()
    '''))

    (tmp_path / "src" / "noisy.py").write_text(textwrap.dedent('''
        """Unrelated module that should NOT match SPLADE-related queries."""

        def add(a, b):
            return a + b
    '''))

    # Markdown content — should also be extracted.
    (tmp_path / "README.md").write_text(textwrap.dedent("""
        # Spladey Project

        This project implements sparse vector retrieval using SPLADE.
        The encoder converts text into high dimensional sparse activations.
    """))

    return tmp_path


def _stats_counts(stats_output: str) -> dict[str, int]:
    """Parse ``infon stats`` output into {Infons, Edges, Constraints, Documents}."""
    out: dict[str, int] = {}
    for line in stats_output.splitlines():
        line = line.strip()
        for label in ("Infons", "Edges", "Constraints", "Documents"):
            if line.startswith(f"{label}:"):
                out[label] = int(line.split(":")[1].strip())
    return out


def _kb_has_text_infon_from(db_path: Path, file_path_substr: str) -> bool:
    """Return True iff the kb contains at least one text-grounded infon whose
    grounding doc_id mentions ``file_path_substr``. Uses a real DuckDB read."""
    import duckdb
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = conn.execute(
            "SELECT grounding_json FROM infons WHERE grounding_type = 'text'"
        ).fetchall()
    finally:
        conn.close()
    for (gj,) in rows:
        g = json.loads(gj)
        if file_path_substr in g.get("doc_id", ""):
            return True
    return False


def test_init_default_runs_schema_discovery_and_extracts_text(
    docstring_repo: Path,
) -> None:
    """Default ``infon init`` (no flags) must produce a discovered schema
    AND text-grounded infons from docstrings + markdown."""
    init = _run_infon(["init"], docstring_repo)
    assert init.returncode == 0, f"init failed: {init.stderr}\n{init.stdout}"

    # Schema must be the discovered one — version field is "auto-1.0".
    schema = json.loads((docstring_repo / ".infon" / "schema.json").read_text())
    assert schema["version"] == "auto-1.0", (
        f"expected discovered schema version 'auto-1.0', got "
        f"{schema['version']!r} — did discovery actually run?"
    )
    # Must have more than the eight built-in relation anchors.
    assert len(schema["anchors"]) > 8, (
        f"discovered schema should have >8 anchors (8 builtins + N actors); "
        f"got {len(schema['anchors'])}"
    )
    # Built-in relations must still be present.
    for relation in ("calls", "imports", "defines", "returns"):
        assert relation in schema["anchors"], f"missing built-in relation {relation!r}"

    # Stats: text extraction must produce at least some text infons.
    stats = _run_infon(["stats"], docstring_repo)
    counts = _stats_counts(stats.stdout)
    assert counts.get("Infons", 0) > 0
    assert counts.get("Documents", 0) >= 1  # README.md at minimum

    # Concrete check: the SpladeyEncoder docstring must have produced at least
    # one text-grounded infon.
    db_path = docstring_repo / ".infon" / "kb.ddb"
    assert _kb_has_text_infon_from(db_path, "spladey.py"), (
        "expected at least one text-grounded infon from spladey.py docstrings"
    )

    # Markdown extraction sanity check — README content should have produced
    # at least one text-grounded infon.
    assert _kb_has_text_infon_from(db_path, "README.md"), (
        "expected at least one text-grounded infon from README.md"
    )


def test_init_shallow_skips_discovery_and_text_extraction(
    docstring_repo: Path,
) -> None:
    """``infon init --shallow`` must use the static code schema and skip text
    extraction (matches v0.1.1 fast path)."""
    init = _run_infon(["init", "--shallow"], docstring_repo)
    assert init.returncode == 0, f"shallow init failed: {init.stderr}\n{init.stdout}"

    # Schema must be the static code schema — version "0.1.0", exactly the
    # eight built-in relation anchors.
    schema = json.loads((docstring_repo / ".infon" / "schema.json").read_text())
    assert schema["version"] == "0.1.0", (
        f"shallow init must use static schema version '0.1.0', got "
        f"{schema['version']!r}"
    )
    assert len(schema["anchors"]) == 8, (
        f"shallow schema must have exactly the 8 built-in relations; got "
        f"{len(schema['anchors'])}"
    )

    # No text-grounded infons may exist — only AST.
    db_path = docstring_repo / ".infon" / "kb.ddb"
    assert not _kb_has_text_infon_from(db_path, "spladey.py"), (
        "shallow init must not extract docstring text"
    )
    assert not _kb_has_text_infon_from(db_path, "README.md"), (
        "shallow init must not extract markdown text"
    )


def test_search_finds_docstring_content_after_default_init(
    docstring_repo: Path,
) -> None:
    """The whole point of the new default: a conceptual query must surface
    a result grounded in the matching docstring/markdown rather than only AST
    structural infons.
    """
    _run_infon(["init"], docstring_repo)

    # "computes sparse vectors" appears verbatim in spladey.py module-level
    # docstring. Search should find it.
    search = _run_infon(["search", "computes sparse vectors"], docstring_repo)
    assert search.returncode == 0, f"search failed: {search.stderr}"
    out = search.stdout
    assert "No results found" not in out, (
        f"conceptual query returned nothing: {out!r}"
    )
    # At least one result line must reference the spladey.py docstring source.
    assert "spladey.py" in out, (
        f"expected at least one result grounded in spladey.py; got: {out!r}"
    )

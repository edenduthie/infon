"""End-to-end smoke test: init → ingest → search returns useful results.

This test would have caught the v0.1.0 regression where ``infon search``
was a TODO stub (infon-84t.14) and ``infon ingest`` skipped consolidation
(infon-84t.15) and document registration (infon-84t.16). Each of those
showed up as "everything looks fine in stats but search returns nothing."

Real-stack: spawns the full ``infon`` CLI as a subprocess against a real
Python source tree, against a real DuckDB store, hitting the real SPLADE
encoder. No mocks. Real subprocess timing is ~30-90 seconds depending on
whether the SPLADE model is in the HF cache.
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# Slow encoder runs; skippable via ``pytest -m 'not smoke'`` in tight loops.
# Marker is registered in pyproject.toml ``[tool.pytest.ini_options]``.
pytestmark = pytest.mark.smoke


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """Real Python+JS source tree the smoke test will index and search."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "store.py").write_text(textwrap.dedent("""
        '''Toy infon-store-like module for smoke testing.'''
        import duckdb

        class StubStore:
            def __init__(self, db_path):
                self._conn = duckdb.connect(db_path)

            def upsert(self, infon):
                self._conn.execute('INSERT INTO infons VALUES (?)', [infon])

        def make_store(path):
            return StubStore(path)
    """))
    (tmp_path / "src" / "service.py").write_text(textwrap.dedent("""
        '''Service that uses the StubStore.'''
        from store import StubStore, make_store

        class UserService:
            def __init__(self):
                self.store = make_store('users.db')

            def save_user(self, user):
                self.store.upsert(user)
    """))
    (tmp_path / "README.md").write_text(textwrap.dedent("""
        # Sample
        This is a sample README that should be registered as a document.
    """))
    return tmp_path


def _run_infon(args: list[str], cwd: Path, env: dict | None = None) -> subprocess.CompletedProcess:
    """Invoke the infon CLI as a subprocess so the test exercises the real
    entry point, not a function-level shortcut."""
    return subprocess.run(
        [sys.executable, "-m", "infon.cli", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
        env=env,
    )


def test_init_then_search_returns_grounded_results(sample_repo: Path) -> None:
    """init must populate infons, edges, AND documents; search must return
    real ranked results with file:line grounding."""
    init = _run_infon(["init"], sample_repo)
    assert init.returncode == 0, f"init failed: {init.stderr}"

    # Stats must reflect a populated kb across all four tables.
    stats = _run_infon(["stats"], sample_repo)
    assert stats.returncode == 0, f"stats failed: {stats.stderr}"
    out = stats.stdout
    assert "Infons:" in out and "Edges:" in out
    assert "Documents:" in out

    # Documents count must be > 0 — README.md should have been registered.
    doc_line = next(line for line in out.splitlines() if "Documents:" in line)
    doc_count = int(doc_line.split(":")[1].strip())
    assert doc_count >= 1, f"expected at least 1 document, got {doc_count}: {out!r}"

    # Edges count must be > 0 — consolidate() should have produced NEXT edges
    # since multiple infons share anchors (e.g. "StubStore" appears in several).
    edge_line = next(line for line in out.splitlines() if "Edges:" in line)
    edge_count = int(edge_line.split(":")[1].strip())
    assert edge_count > 0, f"expected NEXT edges from consolidation, got {edge_count}"

    # Search by symbol name must surface the AST infon for that symbol —
    # the v0.1.0 regression had this returning "No results found".
    search = _run_infon(["search", "StubStore"], sample_repo)
    assert search.returncode == 0, f"search failed: {search.stderr}"
    s = search.stdout
    assert "No results found" not in s, f"search regressed to empty: {s!r}"
    assert "StubStore" in s
    assert "score=" in s
    # Grounding line must include file:line so users can navigate.
    assert "src/store.py:" in s or "src/service.py:" in s


def test_search_handles_natural_language_query(sample_repo: Path) -> None:
    """Natural-language phrasing (with stop words) must still find symbols.

    Validates that the keyword fallback strips stop words and matches the
    remaining tokens against subject/predicate/object — without it, queries
    like "what calls StubStore" return []. We test against ``StubStore``
    (which the Python AST extractor does record as the object of a
    ``calls``/``defines`` edge); class-name preservation for methods like
    ``UserService.save_user`` is a separate AST limitation tracked in
    follow-up work.
    """
    _run_infon(["init"], sample_repo)

    search = _run_infon(["search", "what calls StubStore"], sample_repo)
    assert search.returncode == 0, f"search failed: {search.stderr}"
    out = search.stdout
    assert "No results found" not in out, f"NL query regressed to empty: {out!r}"
    assert "StubStore" in out


def test_init_writes_mcp_config_and_gitignore(sample_repo: Path) -> None:
    """init must drop a working .mcp.json and protect .infon/ in .gitignore.

    Catches regressions where the MCP server invocation is mis-configured
    or where the kb gets accidentally committed.
    """
    _run_infon(["init"], sample_repo)

    mcp_path = sample_repo / ".mcp.json"
    assert mcp_path.exists()
    config = json.loads(mcp_path.read_text())
    assert "infon" in config["mcpServers"]
    server = config["mcpServers"]["infon"]
    # Either uvx form or absolute binary path — both are valid per 13.0c.
    assert server["command"] == "uvx" or server["command"].startswith("/")
    assert "serve" in server["args"]

    gitignore = (sample_repo / ".gitignore").read_text()
    assert ".infon/" in gitignore

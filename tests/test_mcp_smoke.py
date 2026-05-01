"""End-to-end smoke test: MCP server tools return real, useful results.

Complementary to ``tests/test_cli_smoke.py``: that file proves the CLI
``search`` command works end-to-end against a populated kb. This file proves
the same for the MCP server — i.e. that a Claude Code client speaking JSON-RPC
over stdio gets back non-empty, grounded results when calling the ``search``,
``store_observation``, and ``query_ast`` tools.

Would have caught the bug-14 class of regression at the MCP boundary: the
CLI search was a stub returning nothing, and although the MCP server already
called ``retrieve()``, no test ever verified end-to-end that the MCP tool
returned non-empty results from a freshly ``init``-ed kb.

Real-stack: spawns ``infon init`` to populate a real kb (real SPLADE encoder,
real DuckDB), then spawns the MCP server as a subprocess and drives it over
JSON-RPC stdio. No mocks. ~30-90s depending on HF cache state.
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

# Slow encoder runs; share the smoke marker with the CLI smoke tests.
pytestmark = pytest.mark.smoke


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def populated_repo(tmp_path: Path) -> Path:
    """Real Python source tree that ``infon init`` will index.

    Mirrors ``tests/test_cli_smoke.py::sample_repo`` so the assertions about
    which symbols are searchable (``StubStore``) stay aligned across the two
    smoke suites.
    """
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

    # Run real ``infon init`` to populate .infon/kb.ddb + schema.json.
    init = subprocess.run(
        [sys.executable, "-m", "infon.cli", "init"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert init.returncode == 0, f"infon init failed: {init.stderr}"
    return tmp_path


# ---------------------------------------------------------------------------
# JSON-RPC helpers (adapted from tests/test_mcp.py — kept local so this file
# is self-contained and survives independent refactors of test_mcp.py).
# ---------------------------------------------------------------------------


def _send(proc: subprocess.Popen, request_id: int | None, method: str, params: dict | None = None) -> None:
    request: dict[str, Any] = {"jsonrpc": "2.0", "method": method, "params": params or {}}
    if request_id is not None:
        request["id"] = request_id
    assert proc.stdin is not None
    proc.stdin.write(json.dumps(request) + "\n")
    proc.stdin.flush()


def _recv(proc: subprocess.Popen) -> dict | None:
    assert proc.stdout is not None
    line = proc.stdout.readline()
    if not line:
        return None
    return json.loads(line)


def _handshake(proc: subprocess.Popen) -> int:
    """Send initialize + notifications/initialized. Returns next request id."""
    _send(proc, 1, "initialize", {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "smoke-test", "version": "1.0.0"},
    })
    response = _recv(proc)
    assert response is not None, "no response to initialize"
    assert "error" not in response, f"initialize returned error: {response}"
    _send(proc, None, "notifications/initialized")
    return 2


def _extract_tool_result(response: dict) -> Any:
    """Pull the tool's actual return value out of a tools/call response.

    FastMCP returns one of two shapes depending on the tool's return type:
    - ``structuredContent.result`` when the tool returns a list (search,
      query_ast)
    - ``structuredContent`` directly when the tool returns a dict
      (store_observation)
    Older transports may also fall back to ``content[0].text`` as JSON.
    """
    assert "result" in response, f"no result key in response: {response}"
    result = response["result"]
    if "structuredContent" in result:
        sc = result["structuredContent"]
        # FastMCP wraps list returns under {"result": [...]}, dict returns are
        # serialized as the dict itself.
        if isinstance(sc, dict) and set(sc.keys()) == {"result"}:
            return sc["result"]
        return sc
    if "content" in result and result["content"]:
        # Fallback for transports that don't emit structuredContent.
        return json.loads(result["content"][0]["text"])
    raise AssertionError(f"unexpected tools/call result shape: {result}")


def _spawn_mcp(db_path: Path, cwd: Path) -> subprocess.Popen:
    """Start the MCP server subprocess with stdin/stdout pipes."""
    return subprocess.Popen(
        [sys.executable, "-m", "infon.mcp.server", "--db", str(db_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(cwd),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_mcp_search_returns_grounded_results(populated_repo: Path) -> None:
    """``search`` over a real init-ed kb must return non-empty results with
    the dict shape a Claude Code client expects (subject/predicate/object/
    grounding/score), and grounding must include file:line for AST hits so
    the client can render a navigable reference.
    """
    db_path = populated_repo / ".infon" / "kb.ddb"
    proc = _spawn_mcp(db_path, populated_repo)
    try:
        req_id = _handshake(proc)

        _send(proc, req_id, "tools/call", {
            "name": "search",
            "arguments": {"query": "StubStore", "limit": 5},
        })
        response = _recv(proc)
        assert response is not None, "no response to search"
        assert response["id"] == req_id

        results = _extract_tool_result(response)
        assert isinstance(results, list), f"expected list, got {type(results).__name__}: {results}"
        assert len(results) > 0, "search regressed to empty for known symbol StubStore"

        # Each result must carry the full shape the client renders.
        required_keys = {"subject", "predicate", "object", "grounding", "score"}
        for item in results:
            assert isinstance(item, dict), f"non-dict result element: {item!r}"
            assert "error" not in item, f"server returned error envelope: {item}"
            missing = required_keys - item.keys()
            assert not missing, f"missing keys {missing} in result: {item}"

        # At least one result must mention the symbol we searched for so we
        # know we're not matching unrelated noise.
        assert any(
            "StubStore" in (item["subject"], item["object"])
            for item in results
        ), f"no result mentions StubStore: {results}"

        # Grounding shape contract: type is ast|text; ast must carry
        # file_path + line_number so the client can render file:line.
        for item in results:
            grounding = item["grounding"]
            assert grounding["type"] in {"ast", "text"}, grounding
            if grounding["type"] == "ast":
                assert "file_path" in grounding and grounding["file_path"]
                assert "line_number" in grounding and grounding["line_number"] is not None
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_mcp_store_observation_returns_summary(populated_repo: Path) -> None:
    """``store_observation`` must run extract + persist + consolidate end-to-end
    and return a summary dict carrying both ``infons_added`` and
    ``infons_reinforced`` integer counts.
    """
    db_path = populated_repo / ".infon" / "kb.ddb"
    proc = _spawn_mcp(db_path, populated_repo)
    try:
        req_id = _handshake(proc)

        _send(proc, req_id, "tools/call", {
            "name": "store_observation",
            "arguments": {
                "text": "ServiceA delegates auth to ServiceB",
                "source": "smoke-test",
            },
        })
        response = _recv(proc)
        assert response is not None, "no response to store_observation"
        assert response["id"] == req_id

        summary = _extract_tool_result(response)
        assert isinstance(summary, dict), f"expected dict summary, got {summary!r}"
        assert "error" not in summary, f"server returned error: {summary}"

        # Both keys must be present and numeric — the client UI surfaces both.
        for key in ("infons_added", "infons_reinforced"):
            assert key in summary, f"missing {key} in summary: {summary}"
            assert isinstance(summary[key], int), f"{key} not int: {summary[key]!r}"
            assert summary[key] >= 0, f"{key} is negative: {summary[key]}"
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_mcp_query_ast_finds_known_symbol(populated_repo: Path) -> None:
    """``query_ast`` must surface AST infons for a symbol that the AST
    extractor recorded during ``infon init``. Each result must reference
    the queried symbol on either the subject or object side (the server
    unions both query directions before returning).
    """
    db_path = populated_repo / ".infon" / "kb.ddb"
    proc = _spawn_mcp(db_path, populated_repo)
    try:
        req_id = _handshake(proc)

        _send(proc, req_id, "tools/call", {
            "name": "query_ast",
            "arguments": {"symbol": "StubStore"},
        })
        response = _recv(proc)
        assert response is not None, "no response to query_ast"
        assert response["id"] == req_id

        results = _extract_tool_result(response)
        assert isinstance(results, list), f"expected list, got {results!r}"
        assert len(results) > 0, "query_ast returned empty for known AST symbol StubStore"

        for item in results:
            assert isinstance(item, dict), f"non-dict element: {item!r}"
            assert "error" not in item, f"server returned error envelope: {item}"
            assert (
                item.get("subject") == "StubStore" or item.get("object") == "StubStore"
            ), f"result does not reference StubStore on either side: {item}"
    finally:
        proc.terminate()
        proc.wait(timeout=5)

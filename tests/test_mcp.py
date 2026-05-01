"""
Integration test for MCP server.

Tests the FastMCP-based stdio MCP server:
- Spawn real run_server() subprocess with pre-populated store
- Open real JSON-RPC session over stdin/stdout
- Send tools/list and verify all three tools registered
- Call search/store_observation/query_ast via JSON-RPC
- Verify tool error handling (no process crash)
- Fetch all three resources (infon://stats, infon://schema, infon://recent)

No mocks — real subprocess, real store, real JSON-RPC.
"""

import json
import subprocess
import sys
import uuid
from datetime import UTC, datetime

import pytest

from infon.grounding import ASTGrounding, TextGrounding
from infon.infon import ImportanceScore, Infon
from infon.schema import Anchor, AnchorSchema
from infon.store import InfonStore


@pytest.fixture
def sample_schema() -> AnchorSchema:
    """Create a sample AnchorSchema with code-mode anchors."""
    anchors = {
        "user_service": Anchor(
            key="user_service",
            type="actor",
            tokens=["user", "userservice", "user_service"],
            description="User service component",
            parent=None,
        ),
        "database_pool": Anchor(
            key="database_pool",
            type="actor",
            tokens=["database", "pool", "connection_pool", "database_pool"],
            description="Database connection pool",
            parent=None,
        ),
        "token_validator": Anchor(
            key="token_validator",
            type="actor",
            tokens=["token", "validator", "token_validator"],
            description="Token validation component",
            parent=None,
        ),
        "calls": Anchor(
            key="calls",
            type="relation",
            tokens=["call", "calls", "invoke", "invokes", "uses"],
            description="Function or method invocation",
            parent=None,
        ),
        "imports": Anchor(
            key="imports",
            type="relation",
            tokens=["import", "imports", "require", "requires"],
            description="Module import",
            parent=None,
        ),
        "delegates": Anchor(
            key="delegates",
            type="relation",
            tokens=["delegate", "delegates", "delegating"],
            description="Delegates responsibility",
            parent=None,
        ),
    }
    return AnchorSchema(anchors=anchors, version="1.0.0", language="code")


@pytest.fixture
def populated_store(sample_schema, tmp_path):
    """Create a pre-populated InfonStore for testing."""
    db_path = tmp_path / "kb.ddb"
    schema_path = tmp_path / "schema.json"
    
    # Write schema to disk
    with open(schema_path, "w") as f:
        json.dump(sample_schema.model_dump(), f)
    
    # Create store and populate with test data
    store = InfonStore(db_path)
    
    # Add some test infons
    infon1 = Infon(
        id=str(uuid.uuid4()),
        subject="user_service",
        predicate="calls",
        object="database_pool",
        polarity=True,
        grounding=ASTGrounding(
            file_path="src/services/user.py",
            line_number=42,
            node_type="call_expression",
        ),
        confidence=0.95,
        timestamp=datetime.now(UTC),
        importance=ImportanceScore(
            activation=0.8,
            coherence=0.7,
            specificity=0.9,
            novelty=0.6,
            reinforcement=0.5,
        ),
        kind="extracted",
        reinforcement_count=3,
    )
    
    infon2 = Infon(
        id=str(uuid.uuid4()),
        subject="user_service",
        predicate="imports",
        object="token_validator",
        polarity=True,
        grounding=ASTGrounding(
            file_path="src/services/user.py",
            line_number=5,
            node_type="import_statement",
        ),
        confidence=0.99,
        timestamp=datetime.now(UTC),
        importance=ImportanceScore(
            activation=0.9,
            coherence=0.8,
            specificity=0.95,
            novelty=0.7,
            reinforcement=0.6,
        ),
        kind="extracted",
        reinforcement_count=1,
    )
    
    infon3 = Infon(
        id=str(uuid.uuid4()),
        subject="token_validator",
        predicate="calls",
        object="database_pool",
        polarity=False,  # Negated
        grounding=TextGrounding(
            doc_id="design_doc_1",
            sent_id=0,
            char_start=0,
            char_end=50,
            sentence_text="TokenValidator no longer calls DatabasePool directly.",
        ),
        confidence=0.85,
        timestamp=datetime.now(UTC),
        importance=ImportanceScore(
            activation=0.7,
            coherence=0.6,
            specificity=0.8,
            novelty=0.9,
            reinforcement=0.4,
        ),
        kind="extracted",
        reinforcement_count=1,
    )
    
    store.upsert(infon1)
    store.upsert(infon2)
    store.upsert(infon3)
    
    store.close()
    
    return db_path, schema_path


def send_jsonrpc_request(proc, request_id: int | None, method: str, params: dict = None):
    """Send a JSON-RPC 2.0 request to the MCP server subprocess."""
    request = {
        "jsonrpc": "2.0",
        "method": method,
        "params": params or {},
    }
    if request_id is not None:
        request["id"] = request_id
    message = json.dumps(request) + "\n"
    proc.stdin.write(message)
    proc.stdin.flush()


def read_jsonrpc_response(proc):
    """Read a JSON-RPC 2.0 response from the MCP server subprocess."""
    line = proc.stdout.readline()
    if not line:
        return None
    return json.loads(line)


def init_mcp_session(proc):
    """Initialize an MCP session with handshake."""
    # Send initialize request
    send_jsonrpc_request(proc, 1, "initialize", {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "test-client", "version": "1.0.0"}
    })
    response = read_jsonrpc_response(proc)
    assert response is not None
    assert "result" in response or "error" not in response
    
    # Send initialized notification
    send_jsonrpc_request(proc, None, "notifications/initialized")
    
    return 2  # Next request ID


def test_mcp_server_tools_list(populated_store, tmp_path):
    """
    Test that MCP server exposes all three tools via tools/list.
    
    WHEN: MCP server is started and tools/list is requested
    THEN: Response includes search, store_observation, query_ast
    """
    db_path, schema_path = populated_store
    
    # Start the MCP server subprocess
    # We need to create a simple runner script since we don't have the module yet
    # For now, we'll assume we can import and run it directly
    proc = subprocess.Popen(
        [sys.executable, "-m", "infon.mcp.server", "--db", str(db_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=tmp_path.parent.parent,  # Project root
    )
    
    try:
        # Initialize MCP session
        req_id = init_mcp_session(proc)
        
        # Now send tools/list request
        send_jsonrpc_request(proc, req_id, "tools/list", {})
        
        # Read response
        response = read_jsonrpc_response(proc)
        
        assert response is not None
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 2
        assert "result" in response
        
        # Verify all three tools are present
        tools = response["result"]["tools"]
        tool_names = {tool["name"] for tool in tools}
        
        assert "search" in tool_names
        assert "store_observation" in tool_names
        assert "query_ast" in tool_names
        
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_mcp_server_search_tool(populated_store, tmp_path):
    """
    Test the search tool via JSON-RPC.
    
    WHEN: search tool is called with a query
    THEN: Returns ranked infons matching the query
    """
    db_path, schema_path = populated_store
    
    proc = subprocess.Popen(
        [sys.executable, "-m", "infon.mcp.server", "--db", str(db_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=tmp_path.parent.parent,
    )
    
    try:
        # Initialize MCP session
        req_id = init_mcp_session(proc)
        
        # Call search tool
        send_jsonrpc_request(
            proc,
            req_id,
            "tools/call",
            {"name": "search", "arguments": {"query": "what calls database", "limit": 10}},
        )
        
        response = read_jsonrpc_response(proc)
        
        assert response is not None
        assert response["id"] == req_id
        assert "result" in response
        
        # Verify we got ranked results
        # FastMCP wraps results in structuredContent
        if "structuredContent" in response["result"]:
            results = response["result"]["structuredContent"]["result"]
        else:
            results = response["result"]
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        # Verify result structure
        first_result = results[0]
        assert "subject" in first_result
        assert "predicate" in first_result
        assert "object" in first_result
        assert "polarity" in first_result
        assert "confidence" in first_result
        assert "score" in first_result
        assert "grounding" in first_result
        
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_mcp_server_store_observation_tool(populated_store, tmp_path):
    """
    Test the store_observation tool via JSON-RPC.
    
    WHEN: store_observation is called with text
    THEN: Text is extracted, infons are persisted, consolidation runs
    """
    db_path, schema_path = populated_store
    
    proc = subprocess.Popen(
        [sys.executable, "-m", "infon.mcp.server", "--db", str(db_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=tmp_path.parent.parent,
    )
    
    try:
        # Initialize MCP session
        req_id = init_mcp_session(proc)
        
        # Call store_observation tool
        send_jsonrpc_request(
            proc,
            req_id,
            "tools/call",
            {
                "name": "store_observation",
                "arguments": {
                    "text": "UserService now delegates authentication to TokenValidator",
                    "source": "agent",
                },
            },
        )
        
        response = read_jsonrpc_response(proc)
        
        assert response is not None
        assert response["id"] == req_id
        assert "result" in response
        
        # Verify summary returned
        # FastMCP wraps results in structuredContent
        if "structuredContent" in response["result"]:
            summary = response["result"]["structuredContent"]
        else:
            summary = response["result"]
        
        assert "infons_added" in summary
        assert "infons_reinforced" in summary
        assert isinstance(summary["infons_added"], int)
        assert isinstance(summary["infons_reinforced"], int)
        
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_mcp_server_query_ast_tool(populated_store, tmp_path):
    """
    Test the query_ast tool via JSON-RPC.
    
    WHEN: query_ast is called with a symbol
    THEN: Returns infons matching the symbol sorted by reinforcement_count
    """
    db_path, schema_path = populated_store
    
    proc = subprocess.Popen(
        [sys.executable, "-m", "infon.mcp.server", "--db", str(db_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=tmp_path.parent.parent,
    )
    
    try:
        # Initialize MCP session
        req_id = init_mcp_session(proc)
        
        # Call query_ast tool with known symbol
        send_jsonrpc_request(
            proc,
            req_id,
            "tools/call",
            {"name": "query_ast", "arguments": {"symbol": "user_service", "limit": 20}},
        )
        
        response = read_jsonrpc_response(proc)
        
        assert response is not None
        assert response["id"] == req_id
        assert "result" in response
        
        # Verify results
        # FastMCP wraps results in structuredContent
        if "structuredContent" in response["result"]:
            results = response["result"]["structuredContent"]["result"]
        else:
            results = response["result"]
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        # Verify sorted by reinforcement_count descending
        # (infon1 has reinforcement_count=3, should come first)
        first_result = results[0]
        assert first_result["subject"] == "user_service"
        assert first_result["predicate"] == "calls"
        
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_mcp_server_error_handling_no_crash(populated_store, tmp_path):
    """
    Test that tool errors return JSON error dicts without crashing.
    
    WHEN: query_ast is called with invalid symbol
    THEN: Server returns error dict, process continues running
    """
    db_path, schema_path = populated_store
    
    proc = subprocess.Popen(
        [sys.executable, "-m", "infon.mcp.server", "--db", str(db_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=tmp_path.parent.parent,
    )
    
    try:
        # Initialize MCP session
        req_id = init_mcp_session(proc)
        
        # Call query_ast with invalid symbol
        send_jsonrpc_request(
            proc,
            req_id,
            "tools/call",
            {"name": "query_ast", "arguments": {"symbol": "nonexistent_symbol", "limit": 20}},
        )
        
        response = read_jsonrpc_response(proc)
        
        assert response is not None
        assert response["id"] == req_id
        
        # Should return either error in response or empty result (graceful handling)
        # The server should NOT crash
        assert "result" in response or "error" in response
        
        # Verify server is still responsive by sending another request
        send_jsonrpc_request(proc, req_id + 1, "tools/list", {})
        response2 = read_jsonrpc_response(proc)
        assert response2 is not None
        assert response2["id"] == req_id + 1
        
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_mcp_server_stats_resource(populated_store, tmp_path):
    """
    Test the infon://stats resource.
    
    WHEN: infon://stats resource is fetched
    THEN: Returns Markdown-formatted store statistics
    """
    db_path, schema_path = populated_store
    
    proc = subprocess.Popen(
        [sys.executable, "-m", "infon.mcp.server", "--db", str(db_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=tmp_path.parent.parent,
    )
    
    try:
        # Initialize MCP session
        req_id = init_mcp_session(proc)
        
        # Fetch stats resource
        send_jsonrpc_request(
            proc, req_id, "resources/read", {"uri": "infon://stats"}
        )
        
        response = read_jsonrpc_response(proc)
        
        assert response is not None
        assert response["id"] == req_id
        assert "result" in response
        
        # Verify Markdown content
        content = response["result"]["contents"][0]["text"]
        assert isinstance(content, str)
        assert len(content) > 0
        # Stats should mention infons
        assert "infon" in content.lower() or "triple" in content.lower()
        
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_mcp_server_schema_resource(populated_store, tmp_path):
    """
    Test the infon://schema resource.
    
    WHEN: infon://schema resource is fetched
    THEN: Returns formatted anchor list grouped by type
    """
    db_path, schema_path = populated_store
    
    proc = subprocess.Popen(
        [sys.executable, "-m", "infon.mcp.server", "--db", str(db_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=tmp_path.parent.parent,
    )
    
    try:
        # Initialize MCP session
        req_id = init_mcp_session(proc)
        
        # Fetch schema resource
        send_jsonrpc_request(
            proc, req_id, "resources/read", {"uri": "infon://schema"}
        )
        
        response = read_jsonrpc_response(proc)
        
        assert response is not None
        assert response["id"] == req_id
        assert "result" in response
        
        # Verify content contains schema anchors
        content = response["result"]["contents"][0]["text"]
        assert isinstance(content, str)
        assert len(content) > 0
        # Should mention anchor types
        assert "actor" in content.lower() or "relation" in content.lower()
        
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_mcp_server_recent_resource(populated_store, tmp_path):
    """
    Test the infon://recent resource.
    
    WHEN: infon://recent resource is fetched
    THEN: Returns 20 most recent infons as Markdown table
    """
    db_path, schema_path = populated_store
    
    proc = subprocess.Popen(
        [sys.executable, "-m", "infon.mcp.server", "--db", str(db_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=tmp_path.parent.parent,
    )
    
    try:
        # Initialize MCP session
        req_id = init_mcp_session(proc)
        
        # Fetch recent resource
        send_jsonrpc_request(
            proc, req_id, "resources/read", {"uri": "infon://recent"}
        )
        
        response = read_jsonrpc_response(proc)
        
        assert response is not None
        assert response["id"] == req_id
        assert "result" in response
        
        # Verify Markdown table
        content = response["result"]["contents"][0]["text"]
        assert isinstance(content, str)
        assert len(content) > 0
        # Should be a table with our test data
        assert "user_service" in content or "|" in content  # Markdown table format
        
    finally:
        proc.terminate()
        proc.wait(timeout=5)

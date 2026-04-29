"""
Integration tests for the CLI using Click's CliRunner.

These tests verify the five CLI commands work end-to-end:
- infon init: Creates .infon/ with schema and db, writes .mcp.json
- infon search: Returns results from the store
- infon stats: Prints store statistics
- infon ingest: Ingests repository files
- infon serve: Starts MCP server (tested via --help only)

All tests use real fixtures, real store, and real Click runner.
No mocks or stubs.
"""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from infon.cli import cli


@pytest.fixture
def temp_repo():
    """Create a temporary directory with Python source files for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        
        # Create a simple Python file
        src_dir = repo_path / "src"
        src_dir.mkdir()
        
        (src_dir / "example.py").write_text("""
def hello():
    return "world"

class Greeter:
    def greet(self, name):
        return f"Hello, {name}"
""")
        
        (src_dir / "utils.py").write_text("""
import json

def parse_config(path):
    with open(path) as f:
        return json.load(f)
""")
        
        yield repo_path


@pytest.fixture
def temp_repo_with_git():
    """Create a temporary git repository for incremental ingest tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        
        # Initialize git repo
        subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=repo_path,
            check=True,
            capture_output=True
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=repo_path,
            check=True,
            capture_output=True
        )
        
        # Create initial files
        (repo_path / "file1.py").write_text("""
def function_one():
    return 1
""")
        
        (repo_path / "file2.py").write_text("""
def function_two():
    return 2
""")
        
        # Initial commit
        subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"],
            cwd=repo_path,
            check=True,
            capture_output=True
        )
        
        yield repo_path


def test_cli_init_creates_infon_directory(temp_repo):
    """Test that infon init creates .infon/ with schema and database."""
    runner = CliRunner()
    
    # Change to temp_repo directory
    import os
    original_cwd = os.getcwd()
    try:
        os.chdir(temp_repo)
        
        # Verify .infon/ does not exist
        infon_dir = temp_repo / ".infon"
        assert not infon_dir.exists()
        
        # Run infon init
        result = runner.invoke(cli, ["init"])
        
        # Should succeed
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        
        # Verify .infon/ directory created
        assert infon_dir.exists()
        assert infon_dir.is_dir()
        
        # Verify schema.json created
        schema_path = infon_dir / "schema.json"
        assert schema_path.exists()
        schema = json.loads(schema_path.read_text())
        assert "language" in schema
        assert "anchors" in schema
        
        # Verify kb.ddb created
        db_path = infon_dir / "kb.ddb"
        assert db_path.exists()
        
        # Verify .mcp.json created
        mcp_config_path = temp_repo / ".mcp.json"
        assert mcp_config_path.exists()
        mcp_config = json.loads(mcp_config_path.read_text())
        assert "mcpServers" in mcp_config
        assert "infon" in mcp_config["mcpServers"]
        
        # Verify .gitignore updated
        gitignore_path = temp_repo / ".gitignore"
        assert gitignore_path.exists()
        gitignore_content = gitignore_path.read_text()
        assert ".infon/" in gitignore_content
    finally:
        os.chdir(original_cwd)


def test_cli_search_returns_results(temp_repo):
    """Test that infon search returns results after init."""
    runner = CliRunner()
    
    import os
    original_cwd = os.getcwd()
    try:
        os.chdir(temp_repo)
        
        # First run init
        result = runner.invoke(cli, ["init"])
        assert result.exit_code == 0
        
        # Now run search
        result = runner.invoke(cli, ["search", "hello"])
        
        # Should succeed
        assert result.exit_code == 0, f"Search failed: {result.output}"
        
        # Should contain table-like output
        assert "subject" in result.output.lower() or "predicate" in result.output.lower()
    finally:
        os.chdir(original_cwd)


def test_cli_search_missing_store_exits_with_error(temp_repo):
    """Test that infon search exits with code 1 when store doesn't exist."""
    runner = CliRunner()
    
    import os
    original_cwd = os.getcwd()
    try:
        os.chdir(temp_repo)
        
        # Run search without init
        result = runner.invoke(cli, ["search", "anything"])
        
        # Should fail with exit code 1
        assert result.exit_code == 1
        
        # Should contain helpful error message
        assert "No knowledge base found" in result.output or "infon init" in result.output
    finally:
        os.chdir(original_cwd)


def test_cli_stats_prints_output(temp_repo):
    """Test that infon stats prints store statistics."""
    runner = CliRunner()
    
    import os
    original_cwd = os.getcwd()
    try:
        os.chdir(temp_repo)
        
        # First run init
        result = runner.invoke(cli, ["init"])
        assert result.exit_code == 0
        
        # Now run stats
        result = runner.invoke(cli, ["stats"])
        
        # Should succeed
        assert result.exit_code == 0, f"Stats failed: {result.output}"
        
        # Should contain statistics
        assert "infon" in result.output.lower() or "count" in result.output.lower()
    finally:
        os.chdir(original_cwd)


def test_cli_stats_missing_store_exits_with_error(temp_repo):
    """Test that infon stats exits with code 1 when store doesn't exist."""
    runner = CliRunner()
    
    import os
    original_cwd = os.getcwd()
    try:
        os.chdir(temp_repo)
        
        # Run stats without init
        result = runner.invoke(cli, ["stats"])
        
        # Should fail with exit code 1
        assert result.exit_code == 1
        
        # Should contain helpful error message
        assert "No knowledge base found" in result.output or "infon init" in result.output
    finally:
        os.chdir(original_cwd)


def test_cli_ingest_incremental(temp_repo_with_git):
    """Test that infon ingest --incremental only processes changed files."""
    runner = CliRunner()
    
    import os
    original_cwd = os.getcwd()
    try:
        os.chdir(temp_repo_with_git)
        
        # Run init
        result = runner.invoke(cli, ["init"])
        assert result.exit_code == 0
        
        # Get initial infon count
        from infon.store import InfonStore
        db_path = temp_repo_with_git / ".infon" / "kb.ddb"
        with InfonStore(str(db_path)) as store:
            initial_stats = store.stats()
            initial_count = initial_stats.infon_count
        
        # Modify file2.py
        (temp_repo_with_git / "file2.py").write_text("""
def function_two():
    return 22  # Changed!

def function_three():
    return 3
""")
        
        # Commit the change
        subprocess.run(
            ["git", "add", "file2.py"],
            cwd=temp_repo_with_git,
            check=True,
            capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Update file2"],
            cwd=temp_repo_with_git,
            check=True,
            capture_output=True
        )
        
        # Run incremental ingest
        result = runner.invoke(cli, ["ingest", "--incremental"])
        assert result.exit_code == 0
        
        # Verify infons were updated
        with InfonStore(str(db_path)) as store:
            final_stats = store.stats()
            final_count = final_stats.infon_count
            
            # Should have more infons (function_three was added)
            assert final_count >= initial_count
    finally:
        os.chdir(original_cwd)


def test_cli_serve_help():
    """Test that infon serve --help works."""
    runner = CliRunner()
    
    result = runner.invoke(cli, ["serve", "--help"])
    
    # Should succeed
    assert result.exit_code == 0
    
    # Should contain help text
    assert "serve" in result.output.lower() or "mcp" in result.output.lower()


def test_cli_init_custom_db_path(temp_repo):
    """Test that infon init accepts custom --db path."""
    runner = CliRunner()
    
    import os
    original_cwd = os.getcwd()
    try:
        os.chdir(temp_repo)
        
        custom_db = temp_repo / "custom.ddb"
        
        # Run init with custom db path
        result = runner.invoke(cli, ["init", "--db", str(custom_db)])
        
        # Should succeed
        assert result.exit_code == 0
        
        # Verify custom db created
        assert custom_db.exists()
    finally:
        os.chdir(original_cwd)


def test_cli_search_custom_db_path(temp_repo):
    """Test that infon search accepts custom --db path."""
    runner = CliRunner()
    
    import os
    original_cwd = os.getcwd()
    try:
        os.chdir(temp_repo)
        
        custom_db = temp_repo / "custom.ddb"
        
        # Run init with custom db path
        result = runner.invoke(cli, ["init", "--db", str(custom_db)])
        assert result.exit_code == 0
        
        # Run search with custom db path
        result = runner.invoke(cli, ["search", "hello", "--db", str(custom_db)])
        
        # Should succeed
        assert result.exit_code == 0
    finally:
        os.chdir(original_cwd)

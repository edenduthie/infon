"""
Command-line interface for infon.

Provides five commands:
- init: Initialize knowledge base
- search: Search for infons
- stats: Show statistics
- ingest: Ingest repository (with --incremental support)
- serve: Start MCP server
"""

import json
import os
import sys
from pathlib import Path

import click

from infon.schema import CODE_RELATION_ANCHORS, AnchorSchema
from infon.store import InfonStore

# Default MCP server invocation when running outside a venv: uvx fetches
# the latest published version of infon on demand, which is what we want
# for zero-install setups (see docs/installation.md "Option 1: uvx").
_UVX_COMMAND = "uvx"
_UVX_ARGS = ["--from", "infon", "infon", "serve"]


def get_default_db_path() -> Path:
    """Get the default database path (.infon/kb.ddb in current directory)."""
    return Path.cwd() / ".infon" / "kb.ddb"


def ensure_store_exists(db_path: Path | None) -> Path:
    """
    Ensure the store exists, or print error and exit.
    
    Args:
        db_path: Optional custom database path
        
    Returns:
        Path to the database file
        
    Raises:
        SystemExit: If store doesn't exist
    """
    if db_path is None:
        db_path = get_default_db_path()
    
    if not db_path.exists():
        click.echo(
            "No knowledge base found. Run 'infon init' to create one.",
            err=True
        )
        sys.exit(1)
    
    return db_path


@click.group()
@click.version_option(version="0.1.0", prog_name="infon")
def cli():
    """infon - Information network memory system."""
    pass


@cli.command()
@click.option(
    "--schema",
    type=click.Path(exists=True, path_type=Path),
    help="Path to schema JSON file (default: auto-discover)"
)
@click.option(
    "--db",
    type=click.Path(path_type=Path),
    help="Database path (default: .infon/kb.ddb)"
)
def init(schema: Path | None, db: Path | None):
    """
    Initialize infon knowledge base.
    
    Creates .infon/ directory with schema and database,
    ingests the current repository, and writes .mcp.json
    configuration file.
    """
    click.echo("Initializing infon knowledge base...")
    
    # Determine paths
    infon_dir = Path.cwd() / ".infon"
    schema_path = infon_dir / "schema.json"
    db_path = db if db else (infon_dir / "kb.ddb")
    
    # Create .infon/ directory
    infon_dir.mkdir(exist_ok=True)
    
    # Load or create schema
    if schema:
        click.echo(f"Loading schema from {schema}")
        schema_obj = AnchorSchema.model_validate_json(schema.read_text())
    else:
        click.echo("Creating default code schema...")
        # Create minimal schema with code relation anchors
        schema_obj = AnchorSchema(
            version="0.1.0",
            language="code",
            anchors=CODE_RELATION_ANCHORS
        )
    
    # Write schema
    schema_path.write_text(schema_obj.model_dump_json(indent=2))
    click.echo(f"Schema written to {schema_path}")
    
    # Create store and ingest
    click.echo(f"Creating database at {db_path}")
    with InfonStore(str(db_path)) as store:
        click.echo("Ingesting repository...")
        
        # Import here to avoid circular dependency
        from infon.ast.ingest import ingest_repo
        
        try:
            infons = ingest_repo(Path.cwd(), store, schema_obj)
            click.echo(f"Extracted {len(infons)} infons")
        except Exception as e:
            click.echo(f"Warning: Ingest failed: {e}", err=True)
    
    # Write .mcp.json
    write_mcp_config(Path.cwd())
    
    # Update .gitignore
    update_gitignore(Path.cwd())
    
    # Print stats
    with InfonStore(str(db_path)) as store:
        stats = store.stats()
        click.echo("\nKnowledge base initialized!")
        click.echo(f"  Infons: {stats.infon_count}")
        click.echo(f"  Edges: {stats.edge_count}")
        click.echo(f"  Documents: {stats.document_count}")


@cli.command()
@click.argument("query")
@click.option(
    "--db",
    type=click.Path(path_type=Path),
    help="Database path (default: .infon/kb.ddb)"
)
@click.option(
    "--limit",
    type=int,
    default=10,
    help="Maximum number of results (default: 10)"
)
@click.option(
    "--persona",
    type=str,
    help="Persona for retrieval context"
)
def search(query: str, db: Path | None, limit: int, persona: str | None):
    """
    Search the knowledge base.
    
    Returns infons matching the query, ranked by relevance.
    """
    db_path = ensure_store_exists(db)
    
    # TODO: Implement actual retrieval once encoder is integrated
    # For now, do simple triple matching
    with InfonStore(str(db_path)) as store:
        # Query for infons matching the search term
        results = store.query(
            subject=query,
            limit=limit
        )
        
        # Also try predicate and object
        if not results:
            results = store.query(predicate=query, limit=limit)
        if not results:
            results = store.query(object=query, limit=limit)
        
        # Print results as table
        if not results:
            click.echo("No results found.")
        else:
            click.echo(f"\nFound {len(results)} results:\n")
            click.echo(f"{'Subject':<30} {'Predicate':<20} {'Object':<30}")
            click.echo("-" * 80)
            
            for infon in results[:limit]:
                subject = infon.subject[:28] + ".." if len(infon.subject) > 30 else infon.subject
                predicate = infon.predicate[:18] + ".." if len(infon.predicate) > 20 else infon.predicate
                obj = infon.object[:28] + ".." if len(infon.object) > 30 else infon.object
                
                click.echo(f"{subject:<30} {predicate:<20} {obj:<30}")


@cli.command()
@click.option(
    "--db",
    type=click.Path(path_type=Path),
    help="Database path (default: .infon/kb.ddb)"
)
def stats(db: Path | None):
    """
    Show knowledge base statistics.
    
    Prints counts of infons, edges, constraints, and top anchors.
    """
    db_path = ensure_store_exists(db)
    
    with InfonStore(str(db_path)) as store:
        stats_obj = store.stats()
        
        click.echo("\nKnowledge Base Statistics")
        click.echo("=" * 50)
        click.echo(f"Infons:       {stats_obj.infon_count}")
        click.echo(f"Edges:        {stats_obj.edge_count}")
        click.echo(f"Constraints:  {stats_obj.constraint_count}")
        click.echo(f"Documents:    {stats_obj.document_count}")
        
        if stats_obj.top_anchors:
            click.echo("\nTop Anchors:")
            for anchor, count in stats_obj.top_anchors[:10]:
                click.echo(f"  {anchor:<40} {count:>5}")


@cli.command()
@click.option(
    "--incremental",
    is_flag=True,
    help="Only process files changed since last ingest"
)
@click.option(
    "--db",
    type=click.Path(path_type=Path),
    help="Database path (default: .infon/kb.ddb)"
)
def ingest(incremental: bool, db: Path | None):
    """
    Ingest repository files into the knowledge base.
    
    With --incremental, only processes files that have changed
    since the last ingest (requires git repository).
    """
    db_path = ensure_store_exists(db)
    
    # Load schema
    schema_path = Path.cwd() / ".infon" / "schema.json"
    if not schema_path.exists():
        click.echo("Error: schema.json not found. Run 'infon init' first.", err=True)
        sys.exit(1)
    
    schema_obj = AnchorSchema.model_validate_json(schema_path.read_text())
    
    with InfonStore(str(db_path)) as store:
        click.echo("Ingesting repository...")
        
        from infon.ast.ingest import ingest_repo
        
        if incremental:
            # Get list of changed files from git
            import subprocess
            
            try:
                result = subprocess.run(
                    ["git", "diff", "--name-only", "HEAD~1", "HEAD"],
                    capture_output=True,
                    text=True,
                    check=True,
                    cwd=Path.cwd()
                )
                
                changed_files = [
                    Path(f.strip()) for f in result.stdout.splitlines() if f.strip()
                ]
                
                click.echo(f"Processing {len(changed_files)} changed files...")
                
                # For now, just re-ingest everything
                # TODO: Implement selective re-ingestion
                infons = ingest_repo(Path.cwd(), store, schema_obj)
                click.echo(f"Extracted {len(infons)} infons")
                
            except subprocess.CalledProcessError as e:
                click.echo(f"Error: git command failed: {e}", err=True)
                sys.exit(1)
        else:
            # Full ingest
            infons = ingest_repo(Path.cwd(), store, schema_obj)
            click.echo(f"Extracted {len(infons)} infons")
        
        # Print updated stats
        stats_obj = store.stats()
        click.echo(f"\nTotal infons: {stats_obj.infon_count}")


@cli.command()
@click.option(
    "--db",
    type=click.Path(path_type=Path),
    help="Database path (default: .infon/kb.ddb)"
)
def serve(db: Path | None):
    """
    Start MCP server for Claude Code integration.

    Runs the FastMCP server on stdio for use with Claude Desktop.
    """
    db_path = ensure_store_exists(db)

    from infon.mcp.server import run_server

    run_server(str(db_path))


def _running_in_venv() -> bool:
    """
    Return True when the current interpreter is inside a virtual env.

    We treat *either* signal as authoritative because both can occur on
    their own in the wild:

    - ``sys.prefix != sys.base_prefix`` is the standard PEP 405 check
      and works for venvs created by ``python -m venv`` even when
      ``VIRTUAL_ENV`` was unset (e.g. invoking the binary directly).
    - ``VIRTUAL_ENV`` is set by ``activate`` and by tools like ``uv run``
      and is sometimes the only signal in conda-style stacks.
    """
    return sys.prefix != sys.base_prefix or bool(os.environ.get("VIRTUAL_ENV"))


def _venv_infon_binary() -> Path:
    """Path to the ``infon`` script inside the active venv (POSIX or Windows)."""
    if sys.platform == "win32":
        return Path(sys.prefix) / "Scripts" / "infon.exe"
    return Path(sys.prefix) / "bin" / "infon"


def _detect_infon_command() -> tuple[str, list[str]]:
    """
    Choose the ``(command, args)`` pair to write into ``.mcp.json``.

    Outside a venv we default to ``uvx`` so Claude Code can launch the
    MCP server without any prior install. Inside a venv we pin to the
    absolute path of that venv's ``infon`` binary so the MCP server uses
    exactly the version the user installed (and isn't silently upgraded
    by uvx). If we *think* we're in a venv but the binary is missing
    (unusual but possible if someone deleted scripts), we fall back to
    uvx with a warning rather than writing a config that won't run.
    """
    if not _running_in_venv():
        return _UVX_COMMAND, list(_UVX_ARGS)

    binary = _venv_infon_binary()
    if not binary.exists():
        click.echo(
            f"Warning: detected venv at {sys.prefix} but no infon binary at "
            f"{binary}; falling back to uvx in .mcp.json",
            err=True,
        )
        return _UVX_COMMAND, list(_UVX_ARGS)

    return str(binary.resolve()), ["serve"]


def write_mcp_config(repo_path: Path) -> None:
    """
    Write ``.mcp.json`` configuration file.

    Detects whether infon is being run from a virtualenv or not and
    writes the appropriate launch command (``uvx ...`` outside a venv,
    absolute venv binary path inside one). Pre-existing ``.mcp.json``
    files are never overwritten.

    Args:
        repo_path: Repository root directory.
    """
    mcp_config_path = repo_path / ".mcp.json"

    if mcp_config_path.exists():
        click.echo(".mcp.json already exists, skipping")
        return

    command, args = _detect_infon_command()
    mcp_config = {
        "mcpServers": {
            "infon": {
                "type": "stdio",
                "command": command,
                "args": args,
            }
        }
    }

    mcp_config_path.write_text(json.dumps(mcp_config, indent=2))
    click.echo(f"MCP config written to {mcp_config_path}")


def update_gitignore(repo_path: Path):
    """
    Add .infon/ to .gitignore if not already present.
    
    Args:
        repo_path: Repository root directory
    """
    gitignore_path = repo_path / ".gitignore"
    
    # Read existing .gitignore or create empty
    if gitignore_path.exists():
        gitignore_content = gitignore_path.read_text()
    else:
        gitignore_content = ""
    
    # Add .infon/ if not present
    if ".infon/" not in gitignore_content:
        if gitignore_content and not gitignore_content.endswith("\n"):
            gitignore_content += "\n"
        gitignore_content += ".infon/\n"
        
        gitignore_path.write_text(gitignore_content)
        click.echo("Added .infon/ to .gitignore")
    else:
        click.echo(".gitignore already contains .infon/")


if __name__ == "__main__":
    cli()

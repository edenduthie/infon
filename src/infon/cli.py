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
    help="Path to schema JSON file (overrides discovery)"
)
@click.option(
    "--db",
    type=click.Path(path_type=Path),
    help="Database path (default: .infon/kb.ddb)"
)
@click.option(
    "--shallow",
    is_flag=True,
    help=(
        "Skip schema discovery and text extraction (AST-only). "
        "Faster (~10s vs ~minute) but conceptual queries won't work."
    ),
)
def init(schema: Path | None, db: Path | None, shallow: bool):
    """
    Initialize infon knowledge base.

    Default: runs schema discovery on the corpus, extracts code structure
    (AST), extracts docstring + markdown text, and consolidates. Slow
    on first invocation (downloads SPLADE model + per-line encoding) but
    enables conceptual search like "how does X work".

    Use --shallow for the fast AST-only path (matches v0.1.1 behaviour).
    Use --schema to provide a hand-tuned schema (skips discovery).
    """
    click.echo("Initializing infon knowledge base...")

    # Determine paths
    infon_dir = Path.cwd() / ".infon"
    schema_path = infon_dir / "schema.json"
    db_path = db if db else (infon_dir / "kb.ddb")

    # Create .infon/ directory
    infon_dir.mkdir(exist_ok=True)

    # Load or create schema. Precedence:
    #   --schema (user supplied)  >  shallow (static)  >  discovery (default)
    if schema:
        click.echo(f"Loading schema from {schema}")
        schema_obj = AnchorSchema.model_validate_json(schema.read_text())
    elif shallow:
        click.echo("Using static code schema (--shallow)")
        schema_obj = AnchorSchema(
            version="0.1.0", language="code", anchors=CODE_RELATION_ANCHORS
        )
    else:
        schema_obj = _discover_schema(Path.cwd())

    # Write schema
    schema_path.write_text(schema_obj.model_dump_json(indent=2))
    click.echo(f"Schema written to {schema_path} ({len(schema_obj.anchors)} anchors)")

    # Create store and ingest
    click.echo(f"Creating database at {db_path}")
    with InfonStore(str(db_path)) as store:
        try:
            _run_full_ingest(Path.cwd(), store, schema_obj, shallow=shallow)
        except Exception as e:
            click.echo(f"Warning: Ingest failed: {e}", err=True)

    # Write .mcp.json
    write_mcp_config(Path.cwd())

    # Update .gitignore
    update_gitignore(Path.cwd())

    # Print stats — single source of truth (store.stats), avoids the historic
    # mismatch between extractor return-value and persisted count.
    with InfonStore(str(db_path)) as store:
        _print_stats(store, header="\nKnowledge base initialized!")


def _discover_schema(repo_path: Path) -> AnchorSchema:
    """Run Phase-6 schema discovery against the corpus, in code mode.

    Discovery encodes every line through SPLADE and clusters the
    co-activation matrix — slow on first run (~30-90s) but produces actor
    anchors derived from the actual repo vocabulary, which is what makes
    conceptual queries work later.
    """
    click.echo(
        "Discovering schema from corpus (this may take 30-90s; "
        "use --shallow to skip)..."
    )
    from infon.discovery import SchemaDiscovery

    discovery = SchemaDiscovery()
    schema = discovery.discover(str(repo_path), mode="code")
    return schema


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

    Encodes the query to anchor space (SPLADE) and ranks infons by relevance,
    falling back to keyword matching if the SPLADE path returns nothing.
    Each result shows its grounding so you can jump to the source.
    """
    from infon.retrieve import retrieve

    db_path = ensure_store_exists(db)
    schema = _load_schema_for(db_path)

    with InfonStore(str(db_path)) as store:
        results = retrieve(query, store, schema, limit=limit, persona=persona)

    if not results:
        click.echo("No results found.")
        return

    click.echo(f"\nFound {len(results)} results:\n")
    for i, scored in enumerate(results, 1):
        infon = scored.infon
        polarity = "" if infon.polarity else "  [NEGATED]"
        click.echo(
            f"{i}. {infon.subject} --[{infon.predicate}]--> {infon.object}"
            f"  (score={scored.score:.3f}){polarity}"
        )
        click.echo(f"   {_format_grounding(infon)}")
        if scored.context:
            ctx_summary = ", ".join(
                f"{c.subject}->{c.object}" for c in scored.context[:3]
            )
            click.echo(f"   next: {ctx_summary}")
        click.echo()


def _load_schema_for(db_path: Path) -> AnchorSchema:
    """Read .infon/schema.json sitting next to ``db_path``.

    Falls back to the built-in code schema if the file is missing so the
    CLI keeps working on legacy stores created before init wrote a schema.
    """
    schema_path = db_path.parent / "schema.json"
    if schema_path.exists():
        return AnchorSchema.model_validate_json(schema_path.read_text())
    return AnchorSchema(
        version="0.1.0", language="code", anchors=CODE_RELATION_ANCHORS
    )


def _format_grounding(infon) -> str:
    """One-line human-readable grounding (file:line for AST, doc for text)."""
    g = infon.grounding.root
    if g.grounding_type == "ast":
        return f"{g.file_path}:{g.line_number} ({g.node_type})"
    snippet = g.sentence_text[:80] + ("..." if len(g.sentence_text) > 80 else "")
    return f"{g.doc_id} sent#{g.sent_id}: {snippet}"


# Document file extensions that get walked during ingest.
_DOCUMENT_EXTENSIONS = (".md", ".rst", ".txt")

# Directories the ingest walker skips. Mirrors the AST walker so doc counts
# stay consistent with code counts.
_SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", "dist", "build",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".infon", "site",
}


def _run_full_ingest(
    repo_path: Path,
    store: InfonStore,
    schema: AnchorSchema,
    *,
    shallow: bool = False,
) -> None:
    """Run the full ingest pipeline.

    Always runs:
        - AST extraction (tree-sitter Python/JS)
        - Document registration (records .md/.rst/.txt in the documents table)
        - Consolidation (NEXT edges, constraints)

    Runs by default (skipped with ``shallow=True``):
        - Python docstring text extraction
        - Markdown/RST/TXT content text extraction

    The text-extraction passes are slow (one SPLADE encode per sentence)
    but they're what makes conceptual queries like "how does X work"
    return docstring-grounded results instead of random AST infons.
    """
    from infon.ast.ingest import ingest_repo
    from infon.consolidate import consolidate

    click.echo("Extracting code structure...")
    ingest_repo(repo_path, store, schema)

    click.echo("Registering documents...")
    doc_count = _register_documents(repo_path, store)
    click.echo(f"  {doc_count} documents registered")

    if not shallow:
        click.echo("Extracting Python docstring text...")
        ds_count = _extract_python_docstrings(repo_path, store, schema)
        click.echo(f"  {ds_count} infons from docstrings")

        click.echo("Extracting markdown/RST/TXT text...")
        text_count = _extract_document_text(repo_path, store, schema)
        click.echo(f"  {text_count} infons from documents")

    click.echo("Consolidating (NEXT edges, constraints)...")
    consolidate(store, schema)


def _register_documents(repo_path: Path, store: InfonStore) -> int:
    """Walk the repo for .md/.rst/.txt files and register each as a document.

    Triple-extraction from the content runs separately in
    ``_extract_document_text``; this function just records existence + token
    count so ``infon stats`` Documents shows what was discovered, regardless
    of whether the deep pass was skipped via ``--shallow``.
    """
    count = 0
    for path in _walk_for_documents(repo_path):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = str(path.relative_to(repo_path))
        store.upsert_document(
            doc_id=rel,
            path=rel,
            kind=path.suffix.lstrip("."),
            token_count=len(text.split()),
        )
        count += 1
    return count


# Stricter SPLADE-extraction thresholds for code-mode (docstrings + repo
# markdown). The spec defaults (threshold=0.1, top_k=5) target natural-
# language paragraphs and produce a runaway cartesian product of triples
# on docstring-style text — on the infon repo itself the relaxed defaults
# yielded ~38k infons from docstrings alone, dominating the kb and
# diluting search ranking. These tighter values cut the per-sentence
# triple cap from 250 to 54 and require activations to be meaningful
# rather than incidental.
_CODE_MODE_EXTRACT_THRESHOLD: float = 0.3
_CODE_MODE_EXTRACT_TOP_K: int = 3


def _extract_python_docstrings(
    repo_path: Path, store: InfonStore, schema: AnchorSchema
) -> int:
    """Collect every module/class/function docstring across the repo, then
    run them through ``extract_text_batch`` in one batched SPLADE pass.

    Using stdlib ``ast.get_docstring`` (not tree-sitter) because it handles
    the "first string literal in body" convention correctly across classes,
    functions, and async functions; we don't need incremental parsing here.

    Files that fail to parse (Python 2 syntax in fixtures, weird encoding)
    are skipped with a warning — partial coverage beats no coverage.
    """
    import ast as _ast
    from infon.extract import extract_text_batch

    items: list[tuple[str, str]] = []
    for py_path in _walk_python_files(repo_path):
        try:
            source = py_path.read_text(encoding="utf-8", errors="replace")
            tree = _ast.parse(source, filename=str(py_path))
        except (SyntaxError, ValueError) as e:
            click.echo(f"  Warning: skipping {py_path}: {e}", err=True)
            continue

        rel = str(py_path.relative_to(repo_path))
        for node in [tree, *_ast.walk(tree)]:
            if not isinstance(
                node,
                (_ast.Module, _ast.ClassDef, _ast.FunctionDef, _ast.AsyncFunctionDef),
            ):
                continue
            doc = _ast.get_docstring(node)
            if not doc:
                continue
            node_name = (
                "<module>"
                if isinstance(node, _ast.Module)
                else getattr(node, "name", "<unknown>")
            )
            items.append((doc, f"{rel}:{node_name}"))

    if not items:
        return 0

    try:
        infons = extract_text_batch(
            items,
            schema,
            threshold=_CODE_MODE_EXTRACT_THRESHOLD,
            top_k=_CODE_MODE_EXTRACT_TOP_K,
        )
    except Exception as e:
        click.echo(f"  Warning: batched docstring extraction failed: {e}", err=True)
        return 0

    for infon in infons:
        store.upsert(infon)
    return len(infons)


def _extract_document_text(
    repo_path: Path, store: InfonStore, schema: AnchorSchema
) -> int:
    """Run ``extract_text_batch`` over every .md/.rst/.txt file's content,
    using stricter code-mode thresholds.

    Document registration in ``_register_documents`` runs first and writes
    the path/token_count metadata; this pass adds the actual content infons
    grounded in those documents.
    """
    from infon.extract import extract_text_batch

    items: list[tuple[str, str]] = []
    for path in _walk_for_documents(repo_path):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not text.strip():
            continue
        doc_id = str(path.relative_to(repo_path))
        items.append((text, doc_id))

    if not items:
        return 0

    try:
        infons = extract_text_batch(
            items,
            schema,
            threshold=_CODE_MODE_EXTRACT_THRESHOLD,
            top_k=_CODE_MODE_EXTRACT_TOP_K,
        )
    except Exception as e:
        click.echo(f"  Warning: batched text extraction failed: {e}", err=True)
        return 0

    for infon in infons:
        store.upsert(infon)
    return len(infons)


def _walk_python_files(repo_path: Path) -> list[Path]:
    """Yield .py files under repo_path, honoring the same skip-dir set as the
    AST and document walkers."""
    out: list[Path] = []

    def _walk(p: Path) -> None:
        if not p.is_dir():
            return
        try:
            for item in p.iterdir():
                if item.name.startswith(".") or item.name in _SKIP_DIRS:
                    continue
                if item.is_file() and item.suffix == ".py":
                    out.append(item)
                elif item.is_dir():
                    _walk(item)
        except PermissionError:
            pass

    _walk(repo_path)
    return out


def _walk_for_documents(repo_path: Path) -> list[Path]:
    """Recursive walk yielding ``_DOCUMENT_EXTENSIONS`` files under repo_path."""
    out: list[Path] = []

    def _walk(p: Path) -> None:
        if not p.is_dir():
            return
        try:
            for item in p.iterdir():
                if item.name.startswith(".") or item.name in _SKIP_DIRS:
                    continue
                if item.is_file() and item.suffix.lower() in _DOCUMENT_EXTENSIONS:
                    out.append(item)
                elif item.is_dir():
                    _walk(item)
        except PermissionError:
            pass

    _walk(repo_path)
    return out


def _print_stats(store: InfonStore, *, header: str) -> None:
    """Print the canonical stats block — single source of truth for counts."""
    s = store.stats()
    click.echo(header)
    click.echo(f"  Infons:      {s.infon_count}")
    click.echo(f"  Edges:       {s.edge_count}")
    click.echo(f"  Constraints: {s.constraint_count}")
    click.echo(f"  Documents:   {s.document_count}")


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
@click.option(
    "--shallow",
    is_flag=True,
    help="Skip text extraction (AST + consolidation only). Faster for tight loops.",
)
def ingest(incremental: bool, db: Path | None, shallow: bool):
    """
    Ingest repository files into the knowledge base.

    Default: extracts code structure, registers documents, extracts docstring
    + markdown text, and consolidates. Use ``--shallow`` for the AST-only
    fast path. ``--incremental`` is informational only today (always
    re-ingests; selective re-ingest is a future task).
    """
    db_path = ensure_store_exists(db)
    
    # Load schema
    schema_path = Path.cwd() / ".infon" / "schema.json"
    if not schema_path.exists():
        click.echo("Error: schema.json not found. Run 'infon init' first.", err=True)
        sys.exit(1)
    
    schema_obj = AnchorSchema.model_validate_json(schema_path.read_text())
    
    with InfonStore(str(db_path)) as store:
        if incremental:
            # NOTE: incremental currently re-ingests everything; the diff is
            # informational. Selective re-ingest is a v0.1.2 task.
            import subprocess

            try:
                result = subprocess.run(
                    ["git", "diff", "--name-only", "HEAD~1", "HEAD"],
                    capture_output=True,
                    text=True,
                    check=True,
                    cwd=Path.cwd(),
                )
                changed_files = [
                    Path(f.strip()) for f in result.stdout.splitlines() if f.strip()
                ]
                click.echo(f"Processing {len(changed_files)} changed files...")
            except subprocess.CalledProcessError as e:
                click.echo(f"Error: git command failed: {e}", err=True)
                sys.exit(1)

        _run_full_ingest(Path.cwd(), store, schema_obj, shallow=shallow)
        _print_stats(store, header="\nIngest complete.")


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

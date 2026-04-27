# infon

Structured, interpretable knowledge-base memory for AI agents — backed by DuckDB, tree-sitter AST extraction, and zero-training SPLADE projection.

## What it does

`infon` gives AI agents (Claude Code, Cursor, etc.) **structured memory** backed by a local DuckDB knowledge graph. It transforms a code repository into typed, queryable triples called **infons** — each grounded to a source file and line number — so an agent can reason about code structure, remember observations, and trace every fact back to its origin.

Run one command to index a repo:

```bash
uvx infon init
```

Then connect Claude Code via MCP, and the agent gains the ability to:
- **Query code structure**: "What calls `DatabasePool`?" returns grounded triples with file paths and line numbers.
- **Store observations mid-session**: Decisions, refactorings, and architectural notes become searchable memory.
- **Auto-discover domain vocabulary**: No manual schema needed — `infon init` derives an anchor schema from your repository using spectral clustering.

## Quick start

```bash
# 1. Index the current repository (auto-discovers schema, extracts AST infons)
uvx infon init

# 2. Search the knowledge base
uvx infon search "what calls DatabasePool"

# 3. Start the MCP server (stdio) for Claude Code
uvx infon serve
```

That's it. `.infon/schema.json` and `.infon/kb.ddb` are created in your project root (auto-ignored by `.gitignore`). A `.mcp.json` is written so Claude Code picks up the three tools automatically.

## Features

- **Infon data model** — typed triples `<<predicate, subject, object; polarity>>` grounded to source file + line number, drawn from situation semantics (Barwise & Perry 1983).
- **Multi-language AST extraction** — Python and TypeScript/JavaScript via tree-sitter. Extracts calls, imports, inheritance, mutations, definitions, returns, raises, and decorators into grounded infons.
- **SPLADE-based text extraction** — Converts docstrings, comments, and natural-language observations into infons via change-of-basis projection from BERT token space to anchor concept space. Zero training, bundled model.
- **Schema auto-discovery** — `infon init` runs spectral clustering on a co-activation matrix to derive anchors from your codebase. No hand-authored schema required.
- **Consolidation** — Duplicate merging, NEXT-edge temporal chains, constraint aggregation, and importance decay.
- **Persona-aware retrieval** — Persona-specific valence scoring (`investor`, `engineer`, `executive`, `regulator`, `analyst`) shifts ranking based on who's asking.
- **MCP server** — stdio-based FastMCP server exposing `search`, `store_observation`, and `query_ast` tools, plus `infon://stats`, `infon://schema`, `infon://recent` resources. Connects to Claude Code via `.mcp.json`.
- **DuckDB storage** — Embedded, columnar, zero-configuration. Analytical queries 5-50x faster than row-based alternatives.
- **Three install paths** — `uvx infon` (zero-install), `pip install infon`, or project-local venv.

## Installation

| Path | Command | Best for |
|---|---|---|
| `uvx` (zero-install) | `uvx infon init` | Quick start, Claude Code |
| `pip` | `pip install infon` | Teams with existing Python tooling |
| Project venv | `pip install infon` in venv | CI, isolation |

## CLI commands

| Command | Description |
|---|---|
| `infon init` | Auto-discover schema, ingest repo, configure MCP |
| `infon ingest [PATH]` | Ingest a directory into the knowledge base |
| `infon search QUERY` | Query the knowledge base |
| `infon stats` | Print knowledge base statistics |
| `infon serve` | Start the MCP stdio server |

## Project structure

```
.infon/
├── kb.ddb            # DuckDB knowledge base
└── schema.json       # Discovered anchor schema
.mcp.json             # Claude Code MCP config
```

## License

Apache 2.0

# Change: infon — Open Source Knowledge Base for AI Agent Memory

## Why

AI agents — including Claude Code — currently rely on flat files or short-context summaries for memory. There is no open source, structured, queryable knowledge base that:

- Stores observations as typed, grounded triples (infons) rather than unstructured text
- Understands the AST of a code repository and can answer structural questions ("what calls DatabasePool?", "what inherits BaseModel?")
- Integrates with Claude Code as an MCP server so the agent can proactively query and update its own memory during a session
- Is interpretable by construction — every fact traces back to a source file and line number, every answer to an infon

`infon` fills this gap. It is a local-first, zero-training, Apache 2.0 Python package that gives any Claude Code session structured memory backed by a DuckDB knowledge graph.

## What Changes

- Introduces a new open source Python package `infon` (PyPI: `infon`, GitHub: `github.com/<owner>/infon`).
- Implements the **infon data model** — typed triples `<<predicate, subject, object; polarity>>` grounded to source file + line number, drawn from situation semantics (Barwise & Perry 1983).
- Implements a **typed anchor schema** — a domain vocabulary of actors, relations, features, and markets. For code repos the anchors are symbols, call/import/inherit/mutate relations, and architectural domains.
- Implements **multi-language AST extraction** via tree-sitter, turning Python and TypeScript source files into infons (e.g. `<<calls, UserService, DatabasePool; +>> @ src/services/user.py:42`). New language grammars can be added by dropping in a tree-sitter grammar.
- Implements **SPLADE-based text extraction** for docstrings, comments, and natural-language observations, projecting them into the same infon graph via a change-of-basis operator (no training required).
- Implements **schema auto-discovery** via spectral clustering on the anchor co-activation matrix (left Kan extension from category theory), so `infon init` requires no manually authored schema.
- Uses **DuckDB** as the storage backend — columnar, embedded, zero-configuration, fast for the graph query workload.
- Implements a **consolidation layer** — reinforcement (duplicate merging), NEXT edges (temporal chains), constraint aggregation, and importance decay.
- Implements a **retrieval / query engine** — SPLADE query encoding, anchor expansion, persona-aware valence scoring, and NEXT-edge temporal walking.
- Exposes the knowledge base as a **stdio MCP server** (`infon serve`) using FastMCP, with three tools (`search`, `store_observation`, `query_ast`) and three resources (`infon://stats`, `infon://schema`, `infon://recent`). Claude Code connects to it via `.mcp.json`.
- Provides a **CLI** (`infon init`, `infon ingest`, `infon search`, `infon stats`, `infon serve`).
- Supports **three install paths**: `uvx infon` (zero-install), `pip install infon` (explicit), and `pip install infon` into a project venv (isolated).
- Ships a **full documentation site** using MkDocs Material, hosted for free on GitHub Pages at `https://<owner>.github.io/infon`.
- Ships with **GitHub Actions CI** for tests and automated PyPI publishing on tagged releases.

## Phased Scope

- **v1 (this spec):** Infon data model, DuckDB store, SPLADE encoder + anchor projection, text extraction, multi-language AST extraction, schema auto-discovery, consolidation, retrieval, MCP server, CLI, docs site, packaging.
- **v2 (follow-up spec):** Graph MCTS reasoning — multi-hop belief-calibrated claim verification using Dempster-Shafer and UCB1 tree search.
- **v3 (follow-up spec):** GA imagination — genetic-algorithm counterfactual generation scored by grammar, logic, and corpus health.

## Impact

- New open source repository. No existing code is modified.
- Depends on: `duckdb`, `pydantic`, `click`, `fastmcp`, `tree-sitter`, `tree-sitter-python`, `tree-sitter-javascript`, `transformers`, `torch` (CPU), bundled `splade-tiny` model (17 MB, Apache 2.0).
- Wheel target: ≤ 50 MB including bundled model.
- Python ≥ 3.11.

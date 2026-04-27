# Design: infon

## Context

`infon` must work as a drop-in AI memory layer for Claude Code. The primary user is a developer who clones a repo, runs one command, and then has Claude Code gain structured knowledge of the codebase without any manual schema authoring. Secondary users are AI researchers and hobbyists who want a lightweight, interpretable knowledge graph for their own agents.

Key constraints:
- Zero training required — the system must work on a new repo without fine-tuning anything
- Interpretable by construction — every fact must trace to a source file and line number; no black-box embeddings
- Local-first — no cloud dependency in v1; DuckDB file lives in the project directory
- Fast cold start — `infon init` on a 50k-line repo should complete in under two minutes on a laptop
- Claude Code integration via stdio MCP — must work with `uvx infon serve` in `.mcp.json` with no other configuration

## Goals / Non-Goals

- **Goals:** Infon data model, DuckDB store, SPLADE encoder + anchor projection, text extraction, multi-language AST extraction (Python + TypeScript via tree-sitter), schema auto-discovery (left Kan extension / spectral clustering), consolidation, retrieval, MCP server (stdio, FastMCP), CLI, MkDocs docs site on GitHub Pages, three install paths (uvx / pip / pip+venv), GitHub Actions CI + PyPI publish.
- **Non-Goals:** Graph MCTS reasoning (v2), GA imagination (v3), cloud / remote KB, HTTP MCP transport, multi-user shared KB, fine-tuning SPLADE, paid hosting, GUI.

## Decisions

### Decision: Package name `infon`

Name the PyPI package and GitHub repo `infon`.

**Why:** `infon` is the name of the core data unit (an atomic unit of information grounded in a situation, from Barwise & Perry 1983). It is available on PyPI, five characters, and directly describes what the package is about. Alternatives considered: `ontomem` (taken on PyPI), `onto` (taken on PyPI), `knomem` (available but not evocative), `kbmem` (available but reads as an acronym without meaning).

---

### Decision: New package from scratch using BlackMagic as reference implementation

Write `infon` from scratch rather than forking `BlackMagic`.

**Why:** BlackMagic is an English-only, SQLite-backed research codebase with ~3,970 lines of production code and ~5,700 lines of cognition code. `infon` makes different choices in three areas that would require substantial rework of a fork: storage (DuckDB instead of SQLite), language support (multi-language AST via tree-sitter from day one), and schema setup (`infon init` auto-discovers anchors instead of requiring a hand-authored schema). Writing from scratch with BlackMagic as a reference ensures the design decisions are deliberate rather than inherited, while preserving all the core algorithms (SPLADE projection, infon algebra, consolidation, retrieval) which are well-specified in `BlackMagic/SPEC.md` and the workshop modules.

**Alternatives considered:**
- *Fork BlackMagic and adapt*: Saves initial bootstrap time but creates divergence debt immediately; the SQLite → DuckDB and manual schema → auto-discovery changes touch every layer of the stack.

---

### Decision: Apache 2.0 license

**Why:** Apache 2.0 is the standard license for open source AI/ML infrastructure. It is compatible with the bundled `splade-tiny` model (also Apache 2.0) and with downstream commercial use without copyleft constraints. MIT would also be acceptable but Apache 2.0 provides explicit patent protection.

---

### Decision: DuckDB as storage backend

Use DuckDB (embedded columnar database) rather than SQLite.

**Why:** The knowledge graph query workload is analytical — range scans over importance scores, anchor-indexed lookups, aggregations for consolidation, NEXT-edge traversals that fan out across many rows. DuckDB's columnar storage and vectorised execution outperform SQLite for this workload by 5–50× on realistic KB sizes (10k–500k infons). DuckDB is also embedded (zero server infrastructure, single `.ddb` file), schema-on-write, and has a Python API that is a near-drop-in replacement for SQLite's `dbapi2` interface. The only user-facing difference is the `.infon/kb.ddb` file extension.

**Alternatives considered:**
- *SQLite (BlackMagic default)*: Adequate for small KBs but 10–50× slower on analytical queries; row-oriented storage is a poor fit for the consolidation and retrieval aggregation patterns.
- *PostgreSQL*: Requires a running server; unacceptable for a local-first embedded tool.
- *Lance / Parquet files*: More complex dependency; no single-file transactional semantics.

---

### Decision: Multi-language AST extraction via tree-sitter from v1

Support Python and TypeScript/JavaScript AST extraction in v1, with a pluggable interface for additional languages.

**Why:** The two most common languages in codebases that use Claude Code are Python and TypeScript/JavaScript. Tree-sitter provides a single, uniform AST API across all languages, handles partial/broken files (critical for real repos under active development), and supports 40+ language grammars via installable packages (`tree-sitter-python`, `tree-sitter-javascript`). Adding a new language in a future version is a single grammar install + a thin extractor subclass. Using the stdlib `ast` module for Python-only would be faster to implement but would require a complete architectural replacement to add TypeScript — a worse trade.

**AST-to-infon mapping:** Each language extractor maps specific node types to infon predicates:

| AST node type | Infon predicate | Example |
|---|---|---|
| `function_definition` / `function_declaration` | `defines` | `<<defines, auth_module, verify_token; +>>` |
| `class_definition` / `class_declaration` | `defines` | `<<defines, auth_module, UserModel; +>>` |
| `call_expression` | `calls` | `<<calls, login_handler, verify_token; +>>` |
| `import_statement` / `import_from` | `imports` | `<<imports, auth_module, jwt_library; +>>` |
| `extends_clause` / base class list | `inherits` | `<<inherits, AdminUser, BaseUser; +>>` |
| `assignment` to a class field | `mutates` | `<<mutates, update_cart, CartState; +>>` |
| `return_statement` with type hint | `returns` | `<<returns, get_user, UserModel; +>>` |
| `raise_statement` | `raises` | `<<raises, validate_input, ValueError; +>>` |
| `decorator` | `decorates` | `<<decorates, route_decorator, login_handler; +>>` |

Grounding for AST infons uses `(file_path, line_number)` in place of `(doc_id, sent_id, char_start, char_end)`.

---

### Decision: Schema auto-discovery via spectral clustering (left Kan extension)

`infon init` runs schema discovery automatically — no manual `schema.json` required.

**Why:** Requiring users to define 80 typed anchors before ingesting anything is the single largest barrier to adoption. The left Kan extension approach (spectral clustering on the SPLADE co-activation matrix) discovers anchors from the corpus itself: frequently co-activated vocabulary tokens cluster into concept groups, cluster centroids become anchors, and anchor types are inferred from linguistic features (verbs → `relation`, proper nouns or module names → `actor`, adjectives/attributes → `feature`). For AST mode, symbols discovered in the call graph become `actor` anchors and the relation set is fixed (`calls`, `imports`, `inherits`, `mutates`, `defines`, `returns`, `raises`, `decorates`). The result is a `schema.json` written to `.infon/` that users can inspect and edit; the auto-discovered schema is a starting point, not a black box.

**Alternatives considered:**
- *Require manual schema (BlackMagic approach)*: Maximum control, but fatal for onboarding. A user who has to write 80 anchors before anything works will not adopt the tool.
- *Fixed built-in schema for code*: Works for code-only use but breaks for text-heavy corpora (docstrings, PR descriptions, issue comments) that need domain-specific anchors.

---

### Decision: MCP server only for Claude Code integration (no skills)

Ship an MCP server (`infon serve`) as the sole Claude Code integration mechanism in v1. Do not ship skill files.

**Why:** The MCP server runs as a persistent subprocess for the session and gives Claude Code proactive access to the knowledge base without user-initiated slash commands. Claude can call `search`, `store_observation`, and `query_ast` mid-reasoning in response to its own needs. This is the correct mental model: the KB is the agent's memory substrate, not a tool it remembers to invoke. Skills (slash commands) are a valid lighter-weight alternative but they require the user to explicitly initiate every KB interaction — this breaks the use case of "Claude automatically stores every significant decision it makes". MCP is the right integration point.

**uvx as the primary distribution mechanism:** The `.mcp.json` ships in the repo with `command: uvx` so users who clone the repo get the MCP server automatically without a separate install step.

---

### Decision: Three install paths documented, uvx as primary

Document three install paths in the README and docs, with `uvx` as the recommended quick-start path.

| Path | Command | Best for |
|---|---|---|
| `uvx` (zero-install) | `uvx infon init` | Quick start, open source contributors |
| `pip install infon` | `pip install infon && infon init` | Teams with existing Python tooling, version pinning |
| `pip` + project venv | Create venv, pip, use venv binary | Production, CI, isolation from project deps |

**Why three:** Different users have different constraints. `uvx` requires `uv` (ships with Claude Code's environment) and auto-updates. `pip install` is familiar to any Python user and supports `==version` pinning. The venv path avoids polluting global Python environments. Documenting all three prevents users from getting stuck if their environment doesn't match the primary path.

---

### Decision: Stdio transport only in v1 (no HTTP MCP)

The MCP server uses stdio transport exclusively in v1.

**Why:** Stdio is sufficient for local use (one developer, one machine, one project). It requires no port allocation, no auth, no TLS, and no running daemon — Claude Code spawns and kills the process automatically. HTTP transport (for shared team KBs) is a v2+ concern. Implementing both in v1 would add surface area and operational complexity before there is evidence of demand for the shared-KB use case.

---

### Decision: Per-project KB isolation via `.infon/kb.ddb`

Each project stores its knowledge base in `.infon/kb.ddb` at the repo root. `.infon/` is gitignored.

**Why:** Codebases are distinct knowledge domains. A `UserService` in project A and a `UserService` in project B are different anchors. Mixing them into a global KB would pollute retrieval results and make the `query_ast` tool unreliable. Per-project isolation is also required for the MCP server's `--db` argument: the server is spawned in the project directory and opens the local `.infon/kb.ddb`. Users with multiple projects open multiple Claude Code sessions each with their own MCP subprocess and their own KB.

**Global KB for cross-project observations (future):** A user-level KB at `~/.infon/global.ddb` is reserved for future use (cross-project `user` type memories, such as personal preferences Claude has learned). Not in scope for v1.

---

### Decision: Bundle splade-tiny (17 MB, Apache 2.0) inside the wheel

Include the `splade-tiny` model weights inside the Python wheel rather than downloading them at runtime.

**Why:** A model that requires internet access on first use is a failure mode for air-gapped environments, CI systems, and offline development. splade-tiny (4.4M params, 17 MB) is small enough to bundle without making the wheel unreasonably large. The wheel target is ≤ 50 MB. The Apache 2.0 license is compatible with redistribution. Alternatives (downloading from HuggingFace on first run, or requiring the user to provide a model path) both introduce network dependencies and failure modes that the bundled approach eliminates.

---

### Decision: FastMCP for the MCP server implementation

Use the `fastmcp` Python library to implement the MCP server.

**Why:** FastMCP is the de facto standard library for Python MCP servers. It provides tool, resource, and prompt registration via decorators, handles stdio transport and JSON-RPC plumbing, and integrates naturally with type hints and Pydantic. Writing a raw MCP server from the protocol spec would require ~500 lines of JSON-RPC boilerplate that FastMCP eliminates. The `fastmcp` package is actively maintained and tracks the MCP specification.

---

### Decision: MkDocs Material + GitHub Pages for documentation

Use MkDocs with the Material theme for the documentation site, deployed to GitHub Pages via GitHub Actions.

**Why:** MkDocs Material is the standard documentation stack for Python open source projects (FastAPI, Pydantic, Ruff, Typer all use it). It provides: full-text search out of the box, versioned docs, a polished theme that renders well on mobile, and a single `mkdocs.yml` configuration file. GitHub Pages hosting is free for public repos, triggered on every push to `main` via a two-step GitHub Actions workflow (`mkdocs build` + `gh-pages` deploy). The result is a live docs site at `https://<owner>.github.io/infon` with no paid hosting, no CDN configuration, and no separate deploy pipeline.

**Alternatives considered:**
- *ReadTheDocs*: Also free for open source; requires an account and webhook. MkDocs Material is more visually polished and faster to set up.
- *Docusaurus (React-based)*: Excellent for projects with a JavaScript audience; overkill for a Python library.
- *README-only*: Insufficient for a tool with multiple install paths, a CLI, an MCP integration, and conceptual depth.

---

### Decision: Retrieval and storage in v1; reasoning (MCTS) in v2; imagination (GA) in v3

Scope v1 to ingestion and retrieval only. Defer graph MCTS reasoning and GA imagination to follow-up specs.

**Why:** The highest-value use cases for Claude Code agent memory are "remember this observation" and "what do I know about X?" — both served by v1. Graph MCTS (multi-hop belief-calibrated reasoning over claim verification) and GA imagination (counterfactual generation) are substantially more complex to implement correctly, require the Dempster-Shafer belief calculus and NLI heads as foundations, and address use cases (claim verification, hypothesis generation) that require an already-populated KB to be useful. Building v1 first means the KB will be populated by real usage before reasoning is layered on top, and the v2 design can be informed by real query patterns.

---

### Decision: Vertical-slice execution plan

Execute the implementation as a sequence of phases, each ending with a working vertical slice, rather than building each module to completion before moving on.

**Why:** The same rationale as the agent-harness change: phased vertical delivery surfaces integration issues early, keeps `main` always-working, and enables real usage feedback before all features land. Phase 1 ends with a working DuckDB store and data model. Phase 3 ends with text extraction into the store. Phase 5 ends with AST extraction running on a real Python repo. Phase 7 ends with a working MCP server. Phase 9 ends with a polished CLI. Phase 10 ends with a live docs site and PyPI package.

## Risks / Trade-offs

- **tree-sitter grammar maintenance:** Tree-sitter grammars for Python and TypeScript are community-maintained. If a grammar version breaks the extractor, `infon ingest` will fail on affected files. Mitigation: pin grammar versions in `pyproject.toml`; add a `--skip-errors` flag to `infon ingest` that logs failures and continues.
- **splade-tiny accuracy:** At 4.4M params, splade-tiny trades recall for size. For code-heavy corpora the SPLADE path is secondary to the AST path, so lower recall on text is acceptable. Users with a quality requirement can swap the encoder via `--encoder` flag (future).
- **DuckDB concurrent write limits:** DuckDB allows only one writer at a time. The MCP server and CLI cannot write concurrently. Mitigation: the MCP server is the sole writer during a Claude Code session; CLI commands run in a separate process and should not be run during an active MCP session. Document this constraint clearly.
- **Auto-discovered schema quality:** Spectral clustering may produce noisy anchor clusters on small corpora (< 100 documents). Mitigation: document the minimum corpus size recommendation; allow `infon init --schema <path>` to bypass discovery with a hand-authored schema.

## Open Questions

- Should `infon ingest` support incremental re-indexing (only changed files since last run) using git diff? Yes — implement as a `--incremental` flag using `git diff --name-only` in v1.
- Should the MCP server expose a `reason` tool in v1 that is a stub returning "reasoning not available until v2"? No — omit it entirely to avoid confusion.

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **CLI search no longer returns "No results found" for natural-language
  queries.** `infon search` was a TODO stub doing literal triple matching;
  now wired to `retrieve()` with file:line grounding output. (infon-84t.14)
- **Ingest now runs consolidation.** `infon init` and `infon ingest` no
  longer leave the Edges and Constraints tables empty — both call
  `consolidate(store, schema)` after extraction. (infon-84t.15)
- **Documents are registered during ingest.** `.md`/`.rst`/`.txt` files
  walked by `_register_documents` populate the documents table so
  `infon stats` reports a non-zero count. Triple extraction from doc
  text remains gated on richer schemas (deferred to v0.1.2).
  (infon-84t.16)
- **Consistent infon counts.** `infon init` no longer prints two
  different infon counts (the per-extractor count vs `store.stats()`);
  a shared `_print_stats` helper is the single source of truth.
  (infon-84t.17)
- **DuckDB TIMESTAMPTZ no longer aborts ingest.** Added `pytz` to
  runtime dependencies — without it `store.upsert()` failed mid-batch
  with `ModuleNotFoundError: No module named 'pytz'`. (infon-84t.13)

### Added

- **Keyword fallback in `retrieve()`.** When SPLADE produces no anchor
  candidates (typical with the relation-only default schema), retrieval
  falls back to stop-word-filtered keyword matching against
  subject/predicate/object so search is useful from day one without
  schema discovery.
- **Keyword bonus in SPLADE-path scoring.** Multiplies relevance score
  by `(1 + fraction_of_query_tokens_matched)` so queries like
  "what calls InfonStore" surface InfonStore-mentioning infons rather
  than every "calls" infon at uniform score.
- **End-to-end smoke tests.** `tests/test_cli_smoke.py` and
  `tests/test_mcp_smoke.py` spawn the CLI and MCP server as subprocesses
  against a real populated kb to catch regressions where commands
  silently fall through to stubs.
- **CLI search output format.** Each result shows score, the
  `subject -> [predicate] -> object` triple, file:line grounding, and a
  preview of NEXT-edge neighbors so users can navigate temporal
  context.

## [0.1.0] - 2026-05-01

Initial public release of infon, an information-network memory system that
stores observations as semantic anchors with persona-aware retrieval.

### Added

- **Data model (Phase 1-2):** `Infon`, `Anchor`, `AnchorSchema`, `Grounding`
  (text and AST variants), and composite `ImportanceScore` Pydantic models,
  with hierarchy traversal helpers on `AnchorSchema` and a `replace` helper
  on `Infon` for immutable updates.
- **Persistence (Phase 3):** `InfonStore` backed by DuckDB with four tables
  (infons, anchors, schemas, edges) and a single-file `.ddb` knowledge base.
- **Text extraction (Phase 4):** Extraction pipeline that turns raw text into
  infons, including negation detection.
- **AST extraction (Phase 5):** Tree-sitter based extraction for Python and
  JavaScript/TypeScript source, producing `ASTGrounding` records linked to
  the originating spans.
- **Schema discovery (Phase 6):** Spectral clustering over anchor embeddings
  to discover schemas without supervision.
- **Consolidation (Phase 7):** Merge pass that builds `NEXT` edges between
  related infons and applies importance decay.
- **Retrieval (Phase 8):** Query pipeline that ranks results using persona
  valence in addition to semantic similarity.
- **MCP server (Phase 9):** FastMCP-based server exposing `search`,
  `store_observation`, and `query_ast` tools, with `NEXT`-edge context
  serialised on responses.
- **CLI (Phase 10):** `infon` command built on Click with five subcommands:
  `init`, `ingest`, `search`, `stats`, and `serve` (which launches the MCP
  server).
- **CI/CD (Phase 11):** GitHub Actions workflows for tests and linting plus
  `ruff` and `mypy` configuration.
- **Documentation (Phase 12):** MkDocs Material documentation site published
  to GitHub Pages at https://edenduthie.github.io/infon/.
- **Packaging (Phase 13):** PyPI-ready `pyproject.toml` with runtime and dev
  dependency groups and an `infon` console script entry point.

### Changed

- **SPLADE encoder:** Model weights are now fetched from the HuggingFace Hub
  on first use rather than bundled in the repository, keeping the wheel
  small.

[Unreleased]: https://github.com/edenduthie/infon/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/edenduthie/infon/releases/tag/v0.1.0

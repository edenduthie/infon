# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed (default behaviour change — opt out with `--shallow`)

- **`infon init` now runs full schema discovery and text extraction by
  default.** Previous default was AST-only with a static 8-anchor relation
  schema. The new default also derives actor anchors from the corpus
  (Phase 6 spectral clustering), extracts module/class/function docstrings
  from `.py` files, and extracts sentences from `.md`/`.rst`/`.txt` files.
  This is what makes conceptual queries like "how does the SPLADE encoder
  work" return results grounded in docstrings rather than random AST
  infons.
- **Cost:** default init takes 10-30+ minutes on a moderate (~100-file) repo
  on CPU because every line goes through SPLADE. GPU support would bring
  this to seconds and is filed as v0.1.3 work.
- **Opt out:** `infon init --shallow` (and `infon ingest --shallow`) skip
  schema discovery and text extraction, restoring the v0.1.1 fast path
  (~10s on the same repo). Use it for tight iteration loops or kbs that
  only need structural queries.
- `infon init --schema <path>` continues to skip discovery (you supply the
  schema) but still runs text extraction unless combined with `--shallow`.

### Added

- **`SpladeEncoder.encode_sparse_batch(texts, batch_size, max_length)`** —
  batches multiple texts through one tokenizer + forward pass. Used by
  schema discovery; provides a 5-10x speedup on CPU vs the per-text path.
  `encode_sparse()` now delegates to it for behavioural parity.
- **`INFON_DISCOVERY_LINES` env var** — overrides the default 2000-line
  cap on the discovery corpus. Larger values produce richer schemas at
  proportionally longer wall time.

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

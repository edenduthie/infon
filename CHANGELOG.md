# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

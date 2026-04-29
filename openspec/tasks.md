# Tasks: infon

## Development Rules (applied to every task below)

- **Test-First:** Write the test before implementation. Verify it fails (red), implement, verify it passes (green).
- **No Mocks:** All tests exercise real functionality with real dependencies (real DuckDB, real files, real subprocesses, real tree-sitter, real SPLADE model). `MagicMock`, `mock`, `stub`, `patch` are forbidden.
- **Phase-Boundary Review:** After completing all tasks in a phase, re-read the relevant spec sections, run the full regression test suite (`pytest -v`), verify spec compliance, confirm the system works end-to-end at this phase, resolve inconsistencies, update the openspec plan, and update beads tasks. After each individual task within a phase, only confirm that task's own tests pass — do not run the full regression.

---

## Phase 1 — Repository Bootstrap and Core Data Model ✅

- [x] 1.1 Create repository structure: `src/infon/`, `tests/`, `tests/fixtures/`, `docs/`, `.github/workflows/`, `CHANGELOG.md`, `CONTRIBUTING.md`, `LICENSE` (Apache 2.0), `README.md` (stub), `.gitignore` (include `.infon/`, `*.ddb`, `dist/`, `.venv/`)
- [x] 1.2 Write `pyproject.toml` with package name `infon`, Apache 2.0 license, Python ≥ 3.11, entry point `infon = "infon.cli:cli"`, and runtime dependencies: `duckdb>=0.10`, `pydantic>=2.0`, `click>=8.0`, `fastmcp>=0.4`, `tree-sitter>=0.21`, `tree-sitter-python>=0.21`, `tree-sitter-javascript>=0.21`, `transformers>=4.35`, `torch>=2.0` (CPU-only), `numpy>=1.24`, `scipy>=1.11`
- [x] 1.3 Write dev dependencies in `pyproject.toml`: `pytest>=8.0`, `pytest-asyncio`, `ruff`, `mypy`, `mkdocs-material>=9.0`, `python-build`
- [x] 1.4 Write `tests/test_grounding.py` integration test first: construct `TextGrounding` and `ASTGrounding`, serialize to JSON, deserialize back, assert all fields round-trip, verify frozen immutability. **Verify test fails (red).**
- [x] 1.4b Implement `Grounding` base class and `TextGrounding`, `ASTGrounding` subtypes in `src/infon/grounding.py` — frozen Pydantic models with `grounding_type` discriminator; `TextGrounding` carries `doc_id`, `sent_id`, `char_start`, `char_end`, `sentence_text`; `ASTGrounding` carries `file_path`, `line_number`, `node_type`. **Verify test passes (green).**
- [x] 1.5 Write `tests/test_importance.py` integration test first: construct `ImportanceScore`, assert `composite` returns weighted average, assert frozen immutability. **Verify test fails (red).**
- [x] 1.5b Implement `ImportanceScore` frozen Pydantic model in `src/infon/infon.py` with fields `activation`, `coherence`, `specificity`, `novelty`, `reinforcement` (all float in [0,1]); add `composite` property returning weighted average. **Verify test passes (green).**
- [x] 1.6 Write `tests/test_infon.py` integration test first: construct `Infon`, assert immutability, call `replace()`, assert new instance differs from original, round-trip JSON serialization. **Verify test fails (red).**
- [x] 1.6b Implement `Infon` frozen Pydantic model in `src/infon/infon.py` with all fields from the spec; add `Infon.replace(**kwargs)` helper that returns a new instance with updated fields. **Verify test passes (green).**
- [x] 1.7 Write `tests/test_schema.py` integration test first: write `schema.json` to temp dir, load with `from_json()`, assert anchors deserialize, test `ancestors()`/`descendants()`, verify `CODE_RELATION_ANCHORS` present in code-mode schema, round-trip JSON. **Verify test fails (red).**
- [x] 1.7b Implement `Anchor` and `AnchorSchema` in `src/infon/schema.py`; include the eight built-in code relation anchors as a module-level constant `CODE_RELATION_ANCHORS`; implement `AnchorSchema.from_json()`, `AnchorSchema.to_json()`, `ancestors()`, `descendants()`. **Verify test passes (green).**
- [x] **PHASE-BOUNDARY REVIEW Phase 1:** Run `pytest -v`, verify spec compliance for all Phase 1 requirements, confirm data model and grounding work end-to-end, update openspec plan and beads tasks.

## Phase 2 — DuckDB Store ✅

- [x] 2.1 Write `tests/test_store.py` integration test first: create `InfonStore` in temp dir, verify all four tables created with indexes, test `upsert()` and reinforcement merge, test `get()`/`query()`/`add_edge()`/`get_edges()`/`upsert_constraint()`/`upsert_document()`/`stats()`, test concurrent write detection, test context manager. **Verify test fails (red).**
- [x] 2.1b Implement `InfonStore` in `src/infon/store.py`; create all four tables on first open (`infons`, `edges`, `constraints`, `documents`) with all indexes; use DuckDB WAL mode; detect concurrent writer and raise `ConcurrentWriteError`; implement `upsert()`, `get()`, `query()`, `add_edge()`, `get_edges()`, `upsert_constraint()`, `upsert_document()`, `stats()`, `close()` with context manager. **Verify test passes (green).**
- [x] **PHASE-BOUNDARY REVIEW Phase 2:** Run `pytest -v`, verify spec compliance, confirm data model persists and queries end-to-end through DuckDB, update openspec plan and beads tasks.

## Phase 3 — SPLADE Encoder and Anchor Projection ✅

- [x] 3.1 Download `splade-cocondenser-selfdistil` (splade-tiny, 4.4M params) weights and bundle them inside the package at `src/infon/models/splade-tiny/` — include `config.json`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `pytorch_model.bin` (or `model.safetensors`)
- [x] 3.2 Write `tests/test_encoder.py` integration test first: set `TRANSFORMERS_OFFLINE=1`, call `encode_sparse()` with real text, assert non-empty sparse dict; call `project()` with real `AnchorSchema`, assert anchor keys match; call `encode()` and assert output keys are subset of schema. **Verify test fails (red).**
- [x] 3.2b Implement `SpladeEncoder` in `src/infon/encoder.py` — lazy-init from bundled path; implement `encode_sparse()`; implement `AnchorProjector.project()`; implement module-level `encode()`. **Verify test passes (green).**
- [x] **PHASE-BOUNDARY REVIEW Phase 3:** Run `pytest -v`, verify spec compliance, confirm text encodes to anchor space end-to-end with real bundled model, update openspec plan and beads tasks.

## Phase 4 — Text Extraction Pipeline ✅

- [x] 4.1 Write `tests/test_extract.py` integration test first: call `extract_text()` with affirmative sentence, assert `polarity=True`; with negated sentence, assert `polarity=False`; with multi-sentence input, assert multiple infons; with empty text, assert empty list; with no activated anchors, assert empty list. **Verify test fails (red).**
- [x] 4.1b Implement sentence splitter in `src/infon/extract.py`, then `_form_triples()`, `_find_spans()`, `_detect_negation()`, `_classify_tense()`, `_score_importance()`, and `extract_text()` assembling all stages. **Verify test passes (green).**
- [x] **PHASE-BOUNDARY REVIEW Phase 4:** Run `pytest -v`, verify spec compliance, confirm text encodes to infons and persists through store end-to-end, update openspec plan and beads tasks.

## Phase 5 — AST Extraction Pipeline ✅

- [x] 5.1 Create `tests/fixtures/` with a synthetic Python project (`main.py`, `auth.py`, `db.py`, `models.py`, ~200 lines total) and a synthetic TypeScript project (`app.ts`, `auth.ts`, `db.ts`, ~150 lines total) covering all eight relation types
- [x] 5.2 Write `tests/test_ast.py` integration test first: call `ingest_repo()` on `tests/fixtures/` with real store and real schema, assert >50 infons produced; verify Python `calls` infons with correct grounding; verify TypeScript `imports` infons; verify `inherits` infons; verify unknown extension skipped. **Verify test fails (red).**
- [x] 5.2b Implement `BaseASTExtractor` abstract class, `PythonASTExtractor`, `TypeScriptASTExtractor`, `ExtractorRegistry`, and `ingest_repo()` in their respective modules. **Verify test passes (green).**
- [x] **PHASE-BOUNDARY REVIEW Phase 5:** Run `pytest -v`, verify spec compliance, confirm repo ingestion produces infons with correct grounding through store end-to-end, update openspec plan and beads tasks.

## Phase 6 — Schema Discovery ✅

- [x] 6.1 Write `tests/test_discovery.py` integration test first: run `SchemaDiscovery.discover()` on `tests/fixtures/` in code mode, assert schema contains eight built-in relation anchors; verify JSON round-trip; verify small corpus warning. **Verify test fails (red).**
- [x] 6.1b Implement `_build_coactivation_matrix()`, `_filter_frequent_tokens()`, `_spectral_cluster()`, `_infer_anchor_type()`, and `SchemaDiscovery.discover()` in `src/infon/discovery.py`. **Verify test passes (green).**
- [x] **PHASE-BOUNDARY REVIEW Phase 6:** Run `pytest -v`, verify spec compliance, confirm schema discovery produces valid schema that integrates with encoder and store, update openspec plan and beads tasks.

## Phase 7 — Consolidation ✅

- [x] 7.1 Write `tests/test_consolidate.py` integration test first: populate real store with chronologically ordered infons, call `consolidate()`, assert NEXT edges; call `consolidate()` again for idempotency; verify constraint aggregation; backdate timestamps and verify importance decay. **Verify test fails (red).**
- [x] 7.1b Implement `_build_next_edges()`, `_aggregate_constraints()`, `_apply_importance_decay()`, and `consolidate()` in `src/infon/consolidate.py`. **Verify test passes (green).**
- [x] **PHASE-BOUNDARY REVIEW Phase 7:** Run `pytest -v`, verify spec compliance, confirm ingested infons are enriched with NEXT edges and constraints end-to-end, update openspec plan and beads tasks.

## Phase 8 — Query Engine ✅ / Retrieval

- [x] 8.1 Write `tests/test_retrieve.py` integration test first: populate real store, call `retrieve()` with real query, assert sorted by score descending; verify anchor expansion; verify persona valence shifts ranking; verify empty store returns empty list. **Verify test fails (red).**
- [x] 8.1b Define persona valence tables in `src/infon/personas.py`; implement `ScoredInfon` dataclass; implement `retrieve()` in `src/infon/retrieve.py` with all pipeline stages. **Verify test passes (green).**
- [x] **PHASE-BOUNDARY REVIEW Phase 8:** Run `pytest -v`, verify spec compliance, confirm query returns ranked results with context from ingested and consolidated data end-to-end, update openspec plan and beads tasks.

## Phase 9 — MCP Server ✅

- [x] 9.1 Write `tests/test_mcp.py` integration test first: spawn real `run_server()` subprocess with pre-populated store, open JSON-RPC session, send `tools/list`, call `search`/`store_observation`/`query_ast` via JSON-RPC, verify tool error handling, fetch all three resources. **Verify test fails (red).**
- [x] 9.1b Implement `src/infon/mcp/server.py` using FastMCP with all three tools and three resources; implement `run_server()`; register tools and resources; handle errors as JSON dicts. **Verify test passes (green).**
- [x] **PHASE-BOUNDARY REVIEW Phase 9:** Run `pytest -v`, verify spec compliance, confirm MCP server exposes tools and resources and drives the full pipeline end-to-end, update openspec plan and beads tasks.

## Phase 10 — CLI ✅

- [x] 10.1 Write `tests/test_cli.py` integration test first using Click's `CliRunner`: `infon init` end-to-end on `tests/fixtures/`, `infon search` returns results, `infon stats` prints output, missing store exits 1, `infon ingest --incremental` in real git repo. **Verify test fails (red).**
- [x] 10.1b Implement `src/infon/cli.py` with Click group and all five subcommands; implement `src/infon/mcp_config.py` for `.mcp.json` writer (detects uvx vs venv install path and writes appropriate command); add missing-store guard to all commands. **Verify test passes (green).**
- [x] **PHASE-BOUNDARY REVIEW Phase 10:** Run `pytest -v`, verify spec compliance, confirm CLI drives full init → ingest → search → serve flow end-to-end, update openspec plan and beads tasks.

## Phase 11 — GitHub Actions CI ✅

- [x] 11.1 Write `.github/workflows/ci.yml` — trigger on push and pull request to `main`; steps: checkout, setup Python 3.11, install with `pip install -e ".[dev]"`, run `pytest tests/ -v`, run `ruff check src/`, run `ruff format --check src/`, run `mypy src/infon/`
- [x] 11.2 Write `.github/workflows/publish.yml` — trigger on GitHub Release (`v*.*.*` tag); steps: checkout, setup Python, build wheel + sdist with `python -m build`, publish to PyPI using OIDC trusted publishing
- [x] 11.3 Configure OIDC trusted publisher on PyPI project page (document steps in `CONTRIBUTING.md`)
- [x] 11.4 Write `ruff.toml` (or inline `[tool.ruff]` in `pyproject.toml`) with line length 100, select `E,F,I,UP`, ignore `E501`
- [x] 11.5 Write `mypy.ini` (or inline `[tool.mypy]`) with `strict = true`, `ignore_missing_imports = true`
- [x] **PHASE-BOUNDARY REVIEW Phase 11:** Run `pytest -v`, `ruff check src/`, `ruff format --check src/`, `mypy src/infon/`, verify CI workflow syntax (run `actionlint`), verify spec compliance, update openspec plan and beads tasks.

## Phase 12 — Docs Site ✅

- [x] 12.1 Write `mkdocs.yml` with Material theme, project name `infon`, repo URL, navigation structure matching all required pages; enable search plugin and code block extensions
- [x] 12.2 Write `.github/workflows/docs.yml` — trigger on push to `main`; build and deploy docs site
- [x] 12.3 Write `docs/index.md` — overview, quick start, conceptual summary
- [x] 12.4 Write `docs/installation.md` — three install paths with commands
- [x] 12.5 Write `docs/concepts.md` — infons, anchors, change of basis, consolidation
- [x] 12.6 Write `docs/cli.md` — all five commands with flags and examples
- [x] 12.7 Write `docs/mcp.md` — MCP server setup, tools, resources, CLAUDE.md snippet
- [x] 12.8 Write `docs/ast-extraction.md` — supported languages, mapping table, how to add new language
- [x] 12.9 Write `docs/schema-discovery.md` — algorithm in plain English, schema output, manual override
- [x] 12.10 Write `docs/api-reference.md` — Python API with type signatures and examples
- [x] 12.11 Write `docs/contributing.md` — development setup, testing, adding extractors, PR process
- [x] 12.12 Write `README.md` — tagline, quick start, feature list, install options, badges
- [x] **PHASE-BOUNDARY REVIEW Phase 12:** Run `pytest -v`, `mkdocs build --strict`, verify all pages render with zero warnings, verify docs workflow syntax, verify spec compliance, update openspec plan and beads tasks.

## Phase 13 — Packaging and Release Preparation

- [ ] 13.1 Verify `uvx infon serve` cold start in a clean environment (no prior `infon` install); confirm MCP server starts and responds to a `tools/list` JSON-RPC call within 30 seconds
- [ ] 13.2 Verify `pip install infon` in a fresh virtualenv; confirm `infon --version` and `infon init` work end-to-end
- [ ] 13.3 Verify pip + project venv path; confirm `.mcp.json` written by `infon init` uses the venv binary path
- [ ] 13.4 Run `infon init` on the `infon` repo itself (dog-food test); confirm `.infon/kb.ddb` is populated, `infon search "what calls InfonStore"` returns results
- [ ] 13.5 Confirm wheel size is ≤ 50 MB including bundled splade-tiny model (`python -m build && ls -lh dist/*.whl`)
- [ ] 13.6 Tag `v0.1.0`, create GitHub Release, verify publish workflow uploads to PyPI and `pip install infon==0.1.0` succeeds
- [ ] 13.7 Confirm docs site is live at `https://<owner>.github.io/infon` after merge to `main`
- [ ] **PHASE-BOUNDARY REVIEW Phase 13:** Final end-to-end validation: run `pytest -v`, confirm all three install paths work, verify wheel size, confirm PyPI publish and docs deployment, final spec compliance pass, update openspec plan, update beads tasks, mark project complete.
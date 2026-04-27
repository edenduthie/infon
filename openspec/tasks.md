# Tasks: infon

## Phase 1 — Repository Bootstrap and Core Data Model

- [ ] 1.1 Create repository structure: `src/infon/`, `tests/`, `tests/fixtures/`, `docs/`, `.github/workflows/`, `CHANGELOG.md`, `CONTRIBUTING.md`, `LICENSE` (Apache 2.0), `README.md` (stub), `.gitignore` (include `.infon/`, `*.ddb`, `dist/`, `.venv/`)
- [ ] 1.2 Write `pyproject.toml` with package name `infon`, Apache 2.0 license, Python ≥ 3.11, entry point `infon = "infon.cli:cli"`, and runtime dependencies: `duckdb>=0.10`, `pydantic>=2.0`, `click>=8.0`, `fastmcp>=0.4`, `tree-sitter>=0.21`, `tree-sitter-python>=0.21`, `tree-sitter-javascript>=0.21`, `transformers>=4.35`, `torch>=2.0` (CPU-only), `numpy>=1.24`, `scipy>=1.11`
- [ ] 1.3 Write dev dependencies in `pyproject.toml`: `pytest>=8.0`, `pytest-asyncio`, `ruff`, `mypy`, `mkdocs-material>=9.0`, `python-build`
- [ ] 1.4 Implement `Grounding` base class and `TextGrounding`, `ASTGrounding` subtypes in `src/infon/grounding.py` — frozen Pydantic models with `grounding_type` discriminator; `TextGrounding` carries `doc_id`, `sent_id`, `char_start`, `char_end`, `sentence_text`; `ASTGrounding` carries `file_path`, `line_number`, `node_type`
- [ ] 1.5 Implement `ImportanceScore` frozen Pydantic model in `src/infon/infon.py` with fields `activation`, `coherence`, `specificity`, `novelty`, `reinforcement` (all float in [0,1]); add `composite` property returning weighted average
- [ ] 1.6 Implement `Infon` frozen Pydantic model in `src/infon/infon.py` with all fields from the spec; add `Infon.replace(**kwargs)` helper that returns a new instance with updated fields
- [ ] 1.7 Implement `Anchor` and `AnchorSchema` in `src/infon/schema.py`; include the eight built-in code relation anchors as a module-level constant `CODE_RELATION_ANCHORS`; implement `AnchorSchema.from_json()`, `AnchorSchema.to_json()`, `ancestors()`, `descendants()`
- [ ] 1.8 Write integration tests in `tests/test_infon.py` covering: `Infon` construction and immutability, `Infon.replace()`, `TextGrounding` and `ASTGrounding` round-trip JSON serialisation
- [ ] 1.9 Write integration tests in `tests/test_schema.py` covering: `AnchorSchema` round-trip JSON serialisation, `ancestors()`, `descendants()`, presence of `CODE_RELATION_ANCHORS` in a code-mode schema

## Phase 2 — DuckDB Store

- [ ] 2.1 Implement `InfonStore` in `src/infon/store.py`; create all four tables on first open (`infons`, `edges`, `constraints`, `documents`) with all indexes specified in the spec; use DuckDB WAL mode; detect concurrent writer and raise `ConcurrentWriteError`
- [ ] 2.2 Implement `InfonStore.upsert(infon)` — insert or merge on `(subject, predicate, object, polarity)` match; on merge, increment `reinforcement_count` and average `confidence`
- [ ] 2.3 Implement `InfonStore.get(id)`, `InfonStore.query(subject, predicate, object, min_confidence, limit)`, `InfonStore.add_edge()`, `InfonStore.get_edges()`
- [ ] 2.4 Implement `InfonStore.upsert_constraint()`, `InfonStore.upsert_document()`, `InfonStore.stats()` returning `StoreStats`
- [ ] 2.5 Implement `InfonStore.close()` with context manager support (`__enter__`/`__exit__`)
- [ ] 2.6 Write integration tests in `tests/test_store.py` covering: create store in temp dir, upsert and reinforcement merge, query with subject/predicate/object filters, edge operations, constraint upsert, stats, concurrent write detection, context manager

## Phase 3 — SPLADE Encoder and Anchor Projection

- [ ] 3.1 Download `splade-cocondenser-selfdistil` (splade-tiny, 4.4M params) weights and bundle them inside the package at `src/infon/models/splade-tiny/` — include `config.json`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `pytorch_model.bin` (or `model.safetensors`)
- [ ] 3.2 Implement `SpladeEncoder` in `src/infon/encoder.py` — load model from bundled path on first call (lazy init, cached); implement `encode_sparse(text: str) -> dict[int, float]` returning non-zero `{token_id: activation}` using `log(1 + ReLU(MLM_logits))` max-pooled across token positions; set `TRANSFORMERS_OFFLINE=1` before loading to prevent network calls
- [ ] 3.3 Implement `AnchorProjector` in `src/infon/encoder.py` — `project(sparse_vec: dict[int, float], schema: AnchorSchema) -> dict[str, float]`; for each anchor, max-pool over all token activations matching the anchor's `tokens` list; return only anchors with activation above zero
- [ ] 3.4 Implement module-level `encode(text: str, schema: AnchorSchema) -> dict[str, float]` combining both stages
- [ ] 3.5 Write integration tests in `tests/test_encoder.py` covering: `SpladeEncoder` produces sparse dict with non-zero values for semantically relevant tokens; `AnchorProjector` maps token activations to anchor keys; `encode()` returns only keys present in schema; model loads from bundle without network access (mock socket)

## Phase 4 — Text Extraction Pipeline

- [ ] 4.1 Implement sentence splitter in `src/infon/extract.py` using `re` + heuristics (period + capital, newlines); no NLTK dependency; handle abbreviations and code snippets gracefully
- [ ] 4.2 Implement `_form_triples(anchor_activations: dict[str, float], schema: AnchorSchema, threshold: float) -> list[tuple[str, str, str]]` — cartesian product of activated actors × activated relations × activated (features ∪ actors); filter pairs below threshold
- [ ] 4.3 Implement `_find_spans(triple, sentence_text) -> tuple[int,int,int,int] | None` — exact substring match for subject/object tokens; return `(subj_start, subj_end, obj_start, obj_end)` or `None` if not found
- [ ] 4.4 Implement `_detect_negation(sentence_text: str, predicate_anchor: Anchor) -> bool` — lexical negation detection (not, never, no, without, lack of, fails to) in scope of the predicate token
- [ ] 4.5 Implement `_classify_tense(sentence_text: str) -> tuple[str, str]` — simple present/past/future tense and simple/progressive/perfect aspect from auxiliary verb patterns
- [ ] 4.6 Implement `_score_importance(activation, sentence_position, existing_constraints) -> ImportanceScore`
- [ ] 4.7 Implement `extract_text(text: str, doc_id: str, schema: AnchorSchema) -> list[Infon]` assembling all stages
- [ ] 4.8 Write integration tests in `tests/test_extract.py` covering: positive triple extracted from affirmative sentence; negation sets `polarity=False`; multi-sentence document produces multiple infons; empty text returns empty list; sentence with no activated anchors produces no infons

## Phase 5 — AST Extraction Pipeline

- [ ] 5.1 Create `tests/fixtures/` with a synthetic Python project (`main.py`, `auth.py`, `db.py`, `models.py`, ~200 lines total) and a synthetic TypeScript project (`app.ts`, `auth.ts`, `db.ts`, ~150 lines total) covering all eight relation types
- [ ] 5.2 Implement `BaseASTExtractor` abstract class in `src/infon/extractors/base.py` with abstract method `extract(source: bytes, file_path: str, schema: AnchorSchema) -> list[Infon]` and class attribute `EXTENSIONS: list[str]`
- [ ] 5.3 Implement `PythonASTExtractor(BaseASTExtractor)` in `src/infon/extractors/ast_python.py` using `tree-sitter-python`; handle all eight relation types; extract enclosing function/class context for `calls` infons; use `ASTGrounding` with exact line numbers
- [ ] 5.4 Implement `TypeScriptASTExtractor(BaseASTExtractor)` in `src/infon/extractors/ast_typescript.py` using `tree-sitter-javascript`; handle `.ts`, `.tsx`, `.js`, `.jsx` extensions; handle all eight relation types including TypeScript-specific `extends` and `implements`
- [ ] 5.5 Implement `ExtractorRegistry` in `src/infon/extractors/__init__.py` — maps file extension → extractor class; `register(extractor_class)` to add new languages; `get(extension) -> BaseASTExtractor | None`
- [ ] 5.6 Implement `ingest_repo(root: str, schema: AnchorSchema, store: InfonStore, *, incremental: bool = False, skip_errors: bool = True) -> IngestStats` in `src/infon/ingest.py`; walk directory recursively excluding `__pycache__`, `.git`, `node_modules`, `dist`, `.venv`; dispatch each file to registry; with `incremental=True` use `git diff --name-only HEAD` to get changed files; collect per-file error count in `IngestStats`
- [ ] 5.7 Write integration tests in `tests/test_ast.py` covering: Python `calls` infon extracted with correct file path and line number; TypeScript `imports` infon extracted; `inherits` infon from Python class with base class; unknown extension skipped; `ingest_repo` on `tests/fixtures/` produces >50 infons; incremental mode only processes changed files (mock `git diff` output)

## Phase 6 — Schema Discovery

- [ ] 6.1 Implement `_build_coactivation_matrix(sentences: list[str], encoder: SpladeEncoder) -> scipy.sparse.csr_matrix` — encode each sentence, accumulate NPMI co-activation counts for co-occurring token pairs; return sparse V×V matrix
- [ ] 6.2 Implement `_filter_frequent_tokens(matrix, top_f: int = 2000) -> tuple[np.ndarray, list[int]]` — select top-F token IDs by total activation frequency; return filtered matrix and token ID list
- [ ] 6.3 Implement `_spectral_cluster(matrix: np.ndarray, k: int = 50) -> np.ndarray` — build normalised graph Laplacian, extract bottom-K eigenvectors, run k-means, return cluster label array
- [ ] 6.4 Implement `_infer_anchor_type(top_tokens: list[str]) -> str` — heuristic type inference: all-caps or CamelCase → `actor`; verb form (ends in -s, -ed, -ing) → `relation`; adjective form → `feature`; geographic or domain keyword → `market`; default → `actor`
- [ ] 6.5 Implement `SchemaDiscovery.discover(corpus: list[str] | str, mode: str = "text", k: int = 50) -> AnchorSchema` in `src/infon/discovery.py`; in code mode, walk the repo to extract symbol identifiers as the corpus and fix the eight built-in relation anchors; emit warning if corpus has fewer than 50 items
- [ ] 6.6 Write integration tests in `tests/test_discovery.py` covering: discovery on `tests/fixtures/` Python project produces a schema with at minimum the eight CODE_RELATION_ANCHORS; discovered schema serialises and deserialises correctly; small corpus warning is emitted; `--schema` override skips discovery

## Phase 7 — Consolidation

- [ ] 7.1 Implement `_build_next_edges(store: InfonStore, anchor_key: str)` — for a single anchor key, fetch all infons with that key as subject or object, sort by timestamp, add NEXT edges with computed weights
- [ ] 7.2 Implement `_aggregate_constraints(store: InfonStore)` — for each distinct `(subject, predicate, object)` triple, compute evidence_count, strength, persistence and upsert to constraints
- [ ] 7.3 Implement `_apply_importance_decay(store: InfonStore, decay_factor: float = 0.95)` — identify infons older than 7 days without recent reinforcement; apply exponential decay to `importance.reinforcement`; update via `upsert`
- [ ] 7.4 Implement `consolidate(store: InfonStore, schema: AnchorSchema, *, decay_factor: float = 0.95)` composing all three steps; make idempotent by checking for existing NEXT edges before adding
- [ ] 7.5 Write integration tests in `tests/test_consolidate.py` covering: NEXT edges created for three chronologically ordered infons sharing an anchor; idempotency (two consolidate calls produce same edge count); constraint aggregation with multi-document reinforcement; decay applied to old infons; decay not applied to recently reinforced infons

## Phase 8 — Query Engine / Retrieval

- [ ] 8.1 Implement `retrieve(query, store, schema, *, limit, persona) -> list[ScoredInfon]` in `src/infon/retrieve.py`; implement all pipeline stages: query encoding, anchor expansion, candidate fetch, valence scoring, relevance scoring, NEXT-edge context fetch, ranking and deduplication
- [ ] 8.2 Define persona valence tables in `src/infon/personas.py` for `investor`, `engineer`, `executive`, `regulator`, `analyst` — mapping predicate anchor keys to weights +1, -1, or 0
- [ ] 8.3 Implement `ScoredInfon` dataclass with `infon: Infon`, `score: float`, `context: list[Infon]`
- [ ] 8.4 Write integration tests in `tests/test_retrieve.py` covering: retrieve returns results sorted by score descending; anchor expansion includes descendants; persona valence shifts ranking (engineer persona ranks `optimises` higher than `documents`); empty query returns top-N by importance; retrieve on empty store returns empty list

## Phase 9 — MCP Server

- [ ] 9.1 Implement `src/infon/mcp/server.py` using FastMCP; register `search`, `store_observation`, `query_ast` tools; register `infon://stats`, `infon://schema`, `infon://recent` resources; handle tool errors by returning JSON error dicts rather than raising
- [ ] 9.2 Implement `search` tool: call `retrieve()`, serialise `ScoredInfon` list to JSON-compatible dicts including grounding and context
- [ ] 9.3 Implement `store_observation` tool: call `extract_text()`, call `store.upsert()` for each infon, call `consolidate()`, return `{infons_added, infons_reinforced}`
- [ ] 9.4 Implement `query_ast` tool: query store by `(subject=symbol)` OR `(object=symbol)` with optional predicate filter; sort by `reinforcement_count` descending; serialise to JSON dicts
- [ ] 9.5 Implement `infon://stats`, `infon://schema`, `infon://recent` resources returning formatted Markdown strings
- [ ] 9.6 Implement `run_server(db_path: str)` startup function: open store, load schema, start FastMCP stdio loop, handle SIGTERM cleanly
- [ ] 9.7 Write integration tests in `tests/test_mcp.py` covering: `search` tool returns ranked results from a pre-populated store; `store_observation` persists infons and runs consolidate; `query_ast` returns correct infons for a known symbol; tool error returns JSON error dict, not exception; `infon://stats` resource returns non-empty Markdown

## Phase 10 — CLI

- [ ] 10.1 Implement `src/infon/cli.py` with Click group `cli` and subcommands `init`, `ingest`, `search`, `stats`, `serve`
- [ ] 10.2 Implement `infon init [--schema PATH] [--db PATH]` — run `SchemaDiscovery.discover()` (or load provided schema), write `.infon/schema.json`, create `.infon/kb.ddb`, run `ingest_repo`, write `.mcp.json` (using `uvx` by default, `venv` path if inside a venv), update `.gitignore`
- [ ] 10.3 Implement `.mcp.json` writer in `src/infon/mcp_config.py` — detect if running inside a virtualenv (`sys.prefix != sys.base_prefix`); if venv, write absolute path to `infon` binary; otherwise write `uvx` invocation; never overwrite existing `.mcp.json`
- [ ] 10.4 Implement `infon ingest [PATH] [--db PATH] [--incremental]` — call `ingest_repo` + `consolidate`, print `IngestStats`
- [ ] 10.5 Implement `infon search QUERY [--db PATH] [--limit N] [--persona PERSONA]` — call `retrieve`, print results as a Click-formatted table with truncated grounding paths
- [ ] 10.6 Implement `infon stats [--db PATH]` — call `store.stats()`, print formatted summary
- [ ] 10.7 Implement `infon serve [--db PATH]` — call `run_server(db_path)`
- [ ] 10.8 Add missing-store guard to all commands that open the store — print actionable error and exit 1 if `.infon/kb.ddb` not found
- [ ] 10.9 Write integration tests in `tests/test_cli.py` using Click's `CliRunner`; cover: `infon init` end-to-end on `tests/fixtures/`; `infon search` returns results after init; `infon stats` prints non-empty output; `infon search` on missing store exits 1 with correct message; `infon ingest --incremental` invoked in a git repo

## Phase 11 — GitHub Actions CI

- [ ] 11.1 Write `.github/workflows/ci.yml` — trigger on push and pull request to `main`; steps: checkout, setup Python 3.11, install with `pip install -e ".[dev]"`, run `pytest tests/ -v`, run `ruff check src/`, run `ruff format --check src/`, run `mypy src/infon/`
- [ ] 11.2 Write `.github/workflows/publish.yml` — trigger on GitHub Release (`v*.*.*` tag); steps: checkout, setup Python, build wheel + sdist with `python -m build`, publish to PyPI using OIDC trusted publishing (`pypa/gh-action-pypi-publish`)
- [ ] 11.3 Configure OIDC trusted publisher on PyPI project page (document steps in `CONTRIBUTING.md`)
- [ ] 11.4 Write `ruff.toml` (or inline `[tool.ruff]` in `pyproject.toml`) with line length 100, select `E,F,I,UP`, ignore `E501`
- [ ] 11.5 Write `mypy.ini` (or inline `[tool.mypy]`) with `strict = true`, `ignore_missing_imports = true`

## Phase 12 — Docs Site

- [ ] 12.1 Write `mkdocs.yml` with Material theme, project name `infon`, repo URL, navigation structure matching all required pages from the spec; enable search plugin; enable `pymdownx.highlight`, `pymdownx.superfences`, `pymdownx.tabbed` for code blocks with tabs (used on Installation page for uvx/pip/venv)
- [ ] 12.2 Write `.github/workflows/docs.yml` — trigger on push to `main`; steps: checkout, setup Python, install `mkdocs-material`, run `mkdocs gh-deploy --force`
- [ ] 12.3 Write `docs/index.md` — overview paragraph, 30-second quick start (uvx path, three commands), conceptual summary of infons and anchors, link to full docs
- [ ] 12.4 Write `docs/installation.md` — three tabbed sections (uvx / pip / pip+venv) with copy-pasteable commands; explain what `infon init` does; explain `.mcp.json` and Claude Code connection; explain `.gitignore` update
- [ ] 12.5 Write `docs/concepts.md` — what an infon is (triple + grounding); what an anchor schema is; the change-of-basis metaphor (BERT token space → concept space); what grounding means (file path + line); what consolidation produces (NEXT edges, constraints)
- [ ] 12.6 Write `docs/cli.md` — all five commands with full flag reference, example invocations, and sample output blocks
- [ ] 12.7 Write `docs/mcp.md` — MCP server setup, `.mcp.json` examples for all three install paths, tool reference (`search`, `store_observation`, `query_ast`) with parameter docs, resource reference (`infon://stats`, `infon://schema`, `infon://recent`), Claude Code CLAUDE.md snippet showing how to instruct Claude to use the KB proactively
- [ ] 12.8 Write `docs/ast-extraction.md` — supported languages, AST-to-infon mapping table, `ASTGrounding` explained, how to add a new language (implement `BaseASTExtractor`, register extension)
- [ ] 12.9 Write `docs/schema-discovery.md` — what schema discovery does, the spectral clustering algorithm in plain English (no equations), what the output schema looks like, how to inspect and edit `.infon/schema.json`, when to use a manual schema
- [ ] 12.10 Write `docs/api-reference.md` — Python API for `Infon`, `AnchorSchema`, `InfonStore`, `encode`, `extract_text`, `ingest_repo`, `retrieve`, `consolidate`; include type signatures and one example per function
- [ ] 12.11 Write `docs/contributing.md` — development setup (`git clone`, `pip install -e ".[dev]"`), running tests, adding a language extractor, submitting a PR, release process
- [ ] 12.12 Write `README.md` — project tagline, three-command quick start, feature list, install options table, link to docs site, license badge, CI badge

## Phase 13 — Packaging and Release Preparation

- [ ] 13.1 Verify `uvx infon serve` cold start in a clean environment (no prior `infon` install); confirm MCP server starts and responds to a `tools/list` JSON-RPC call within 30 seconds
- [ ] 13.2 Verify `pip install infon` in a fresh virtualenv; confirm `infon --version` and `infon init` work end-to-end
- [ ] 13.3 Verify pip + project venv path; confirm `.mcp.json` written by `infon init` uses the venv binary path
- [ ] 13.4 Run `infon init` on the `infon` repo itself (dog-food test); confirm `.infon/kb.ddb` is populated, `infon search "what calls InfonStore"` returns results
- [ ] 13.5 Confirm wheel size is ≤ 50 MB including bundled splade-tiny model (`python -m build && ls -lh dist/*.whl`)
- [ ] 13.6 Tag `v0.1.0`, create GitHub Release, verify publish workflow uploads to PyPI and `pip install infon==0.1.0` succeeds
- [ ] 13.7 Confirm docs site is live at `https://<owner>.github.io/infon` after merge to `main`

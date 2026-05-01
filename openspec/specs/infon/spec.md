## Hard Rules

- **TDD only:** Every requirement must be implemented via Test-Driven Development. Write the integration test first, verify it fails (red), write the minimal implementation, verify it passes (green). Iterate until green.
- **No mocks, no stubs, no test doubles:** Never use `MagicMock`, `mock`, `stub`, `patch`, or any form of test double. Tests that exercise a mock are testing the mock, not the system.
- **No isolated unit tests:** Every test must exercise the full stack — real database, real file system, real subprocess, real network. No testing a function in vacuum.
- **Provision real dependencies:** If the feature needs a database, the test creates a real DuckDB instance in a temp directory. If it needs a subprocess, the test spawns a real process. No fakes.
- **Tear down after the test:** All provisions must be cleaned up. No leftover state between test runs.
- **No skips:** `pytest.mark.skip`, `pytest.mark.xfail`, or equivalent bypasses are forbidden. Difficulty writing a test is a signal that the real integration needs to be built.
- **No shortcuts:** A partially working system with skipped tests is not done. Iterate until all tests pass and the full regression suite is green.
- **Phase-boundary review:** After completing all tasks in a phase, re-read the relevant spec sections, run the full regression test suite, verify spec compliance, check the system works end-to-end at this phase, update the openspec plan, and update beads tasks. After each individual task within a phase, only confirm that task's own tests pass — the full review is reserved for the phase boundary.

## ADDED Requirements

### Requirement: Infon Data Model

The `infon` package SHALL implement an `Infon` dataclass representing an atomic unit of information as a typed triple `<<predicate, subject, object; polarity>>` grounded to a source location.

Every `Infon` SHALL carry:
- `id` — UUID, auto-generated
- `subject` — anchor key (str)
- `predicate` — anchor key (str)
- `object` — anchor key (str)
- `polarity` — boolean (`True` = affirmative, `False` = negated)
- `grounding` — `Grounding` subtype: either `TextGrounding(doc_id, sent_id, char_start, char_end, sentence_text)` or `ASTGrounding(file_path, line_number, node_type)`
- `confidence` — float in [0, 1]
- `timestamp` — UTC datetime
- `importance` — `ImportanceScore(activation, coherence, specificity, novelty, reinforcement)` composite, each component in [0, 1]
- `kind` — one of `extracted`, `consolidated`, `imagined` (imagined reserved for v3)
- `reinforcement_count` — int, incremented on merge; starts at 1

The `Infon` dataclass SHALL be immutable (frozen Pydantic model or `@dataclass(frozen=True)`). Mutation SHALL be performed by creating a new instance with updated fields, not by in-place modification. This ensures that the store is the canonical source of truth and infons retrieved from the store are not accidentally mutated in memory.

#### Scenario: Infon created from text extraction
- **WHEN** the text extractor produces a triple from a sentence
- **THEN** an `Infon` is created with `kind=extracted`, `polarity` determined by negation detection, a `TextGrounding` pointing to the source sentence, and `reinforcement_count=1`

#### Scenario: Infon created from AST extraction
- **WHEN** the AST extractor encounters a call expression in a source file
- **THEN** an `Infon` is created with `kind=extracted`, `polarity=True` (AST facts are always affirmative unless the node is a delete/remove operation), an `ASTGrounding` pointing to the source file and line number, and `reinforcement_count=1`

#### Scenario: Infon retrieved from store is immutable
- **WHEN** an `Infon` is retrieved from the DuckDB store
- **THEN** attempting to modify any field SHALL raise an error; callers that need a modified copy SHALL use the `Infon.replace(**kwargs)` helper

**Testability:** A test constructs real `Infon` instances with both `TextGrounding` and `ASTGrounding`, verifies immutability by attempting field assignment (expecting `FrozenInstanceError`), calls `Infon.replace()` and asserts the new instance has the updated fields while the original is unchanged, and round-trips through JSON serialization. No mocks — all real Pydantic model instances.

---

### Requirement: AnchorSchema

The `infon` package SHALL implement an `AnchorSchema` that defines the typed vocabulary of the knowledge base — the set of named anchors (concepts) and their types.

An `Anchor` SHALL carry:
- `key` — unique string identifier (snake_case)
- `type` — one of `actor`, `relation`, `feature`, `market`, `location`
- `tokens` — list of vocabulary strings that map to this anchor (used by the SPLADE projector)
- `description` — human-readable one-line description
- `parent` — optional parent anchor key (enables hierarchy)

An `AnchorSchema` SHALL carry:
- `anchors` — dict mapping key → Anchor
- `version` — semver string
- `language` — `text` or `code`

The `AnchorSchema` SHALL be serialisable to and from JSON. The canonical on-disk location is `.infon/schema.json`. The `AnchorSchema` SHALL provide:
- `actors` — filtered list of actor-type anchors
- `relations` — filtered list of relation-type anchors
- `features` — filtered list of feature-type anchors
- `markets` — filtered list of market-type anchors
- `locations` — filtered list of location-type anchors
- `ancestors(key)` — list of ancestor anchor keys by walking `parent` links
- `descendants(key)` — list of descendant anchor keys

For `language=code`, the `AnchorSchema` SHALL always include the eight built-in relation anchors: `calls`, `imports`, `inherits`, `mutates`, `defines`, `returns`, `raises`, `decorates`. These MUST NOT be removed or overridden by schema discovery.

#### Scenario: Schema loaded from JSON
- **WHEN** `AnchorSchema.from_json(".infon/schema.json")` is called
- **THEN** all anchors are deserialised, types are validated, and parent links are resolved; a missing or malformed schema file SHALL raise `SchemaLoadError`

#### Scenario: Anchor hierarchy traversal
- **WHEN** `schema.descendants("database_layer")` is called on a schema where `postgresql`, `redis`, and `dynamodb` all have `parent="database_layer"`
- **THEN** the returned list contains all three descendant keys

**Testability:** A test writes a real `schema.json` file to a temp directory with parent-child anchor relationships, calls `AnchorSchema.from_json()`, asserts all anchors deserialize with correct types, calls `ancestors()` and `descendants()` and verifies traversal, then writes the schema back to JSON and reloads to prove round-trip fidelity. No mocked file I/O.

---

### Requirement: SPLADE Encoder and Anchor Projection (Change of Basis)

The `infon` package SHALL implement a two-stage encoder that re-expresses text from BERT's 30,522-dimensional token vocabulary into the domain's anchor concept space.

**Stage 1 — SpladeEncoder:** Fetches a SPLADE model from the HuggingFace Hub on first use (default: `naver/splade-cocondenser-ensembledistil`) and caches it locally via the `transformers` library cache (default: `~/.cache/huggingface`). The model identifier is configurable via the `INFON_SPLADE_MODEL` environment variable or the `model_name` constructor argument to `SpladeEncoder`; either may be a Hub model ID or a local filesystem path to a pre-downloaded model directory. Loading is lazy — model and tokenizer weights are materialised on first encode call, not at import time. Given a string, the encoder SHALL produce a sparse 30,522-dimensional vector via `log(1 + ReLU(MLM_logits))` max-pooled across token positions. The output SHALL be a dict mapping `{token_id: float}` containing only non-zero activations.

**Stage 2 — AnchorProjector:** Given a sparse SPLADE vector and an `AnchorSchema`, it SHALL produce a dense anchor-space vector by max-pooling SPLADE activations over each anchor's `tokens` list. The result is a dict mapping `{anchor_key: float}`.

The combined encoder SHALL be accessible via `encode(text: str, schema: AnchorSchema) -> dict[str, float]`. Encoding SHALL be deterministic and stateless — the same text + schema always produces the same output.

The first encode call requires network access to download the model unless the model is already present in the transformers cache or a local path is supplied. After the initial download, subsequent runs operate entirely from the local cache. **Air-gapped deployments** SHALL pre-populate the cache (e.g., `huggingface-cli download naver/splade-cocondenser-ensembledistil`) or pass a local model directory path via `INFON_SPLADE_MODEL=/path/to/model` (or the `model_name` kwarg); in either case the encoder operates without contacting any external service at runtime.

#### Scenario: Text encoded to anchor space
- **WHEN** `encode("UserService calls the database connection pool", schema)` is called on a schema with anchors `user_service` (tokens: ["userservice", "user_service"]) and `database_pool` (tokens: ["pool", "connection_pool", "database_pool"])
- **THEN** the returned dict contains non-zero values for both `user_service` and `database_pool`; keys not in the schema are absent

#### Scenario: Model loaded from HuggingFace Hub on first use, cached locally
- **WHEN** `SpladeEncoder()` is instantiated for the first time on a machine with no prior transformers cache and `encode_sparse()` is invoked
- **THEN** the default model (`naver/splade-cocondenser-ensembledistil`) is downloaded from the HuggingFace Hub and stored in the transformers cache; the encode call returns a non-empty sparse dict
- **AND WHEN** a second `SpladeEncoder()` is instantiated and used in the same environment
- **THEN** the model is loaded from the local transformers cache without re-downloading
- **AND WHEN** `INFON_SPLADE_MODEL` is set to a different Hub ID before instantiation
- **THEN** that alternate model is used in place of the default
- **AND WHEN** `model_name` is set to a local filesystem path containing a pre-downloaded model
- **THEN** the encoder loads weights from that path with no network access required

**Testability:** A test instantiates a real `SpladeEncoder` (using the default model from the transformers cache, or a fixture-cached model directory passed via `model_name`), calls `encode_sparse()` with real text and asserts the output is a non-empty sparse dict, calls `project()` with a real `AnchorSchema` and asserts anchor keys match, and calls the combined `encode()` and verifies output keys are a subset of the schema keys. A second test sets `INFON_SPLADE_MODEL` to a local path and asserts the encoder uses it. No mocked model — real model loaded from the HuggingFace cache or a local directory.

---

### Requirement: Text Extraction Pipeline

The `infon` package SHALL implement a text extraction pipeline that converts natural language text (docstrings, comments, README files, observations) into a list of `Infon` instances.

The pipeline SHALL process text in the following stages:
1. **Sentence splitting** — split document text into sentences using a language-aware splitter
2. **SPLADE encoding** — encode each sentence to the anchor space via `encode()`
3. **Triple formation** — form candidate triples from activated anchor pairs: for each sentence, cartesian-product the top-K activated actors × activated relations × activated features/actors. Retain triples where all three components are above an activation threshold (configurable, default 0.1)
4. **Span finding** — for each triple, find the character spans in the sentence that activated the subject and object anchors (exact token match or semantic span fallback)
5. **Negation detection** — detect lexical negation (not, never, no, without, lack of) in scope of the predicate; if found, set `polarity=False`
6. **Tense/aspect classification** — classify the verb tense (present, past, future) and aspect (simple, progressive, perfect) from the sentence
7. **Importance scoring** — compute `ImportanceScore` from activation strength, sentence position, novelty vs. existing constraints, and specificity of the triple
8. **Infon construction** — assemble `Infon` instances with `kind=extracted` and `TextGrounding`

The pipeline SHALL be accessible as `extract_text(text: str, doc_id: str, schema: AnchorSchema) -> list[Infon]`. The `doc_id` parameter identifies the source document and is stored in the `Infon.doc_id` field and recorded in the `documents` table for incremental ingest tracking. When called from `store_observation`, `doc_id` SHALL be set to `"<source>:<timestamp>"` (e.g., `"agent:2024-01-15T10:30:00Z"`).

#### Scenario: Observation stored via MCP tool produces infons
- **WHEN** `store_observation("UserService now delegates auth to the new TokenValidator")` is called via the MCP server
- **THEN** the text extraction pipeline runs, producing at minimum one `Infon` with `predicate="delegates"` or `predicate="calls"`, `subject` matching `user_service`, and `object` matching `token_validator` (assuming both are anchors in the schema), and the infon is persisted to the DuckDB store

#### Scenario: Negated sentence sets polarity false
- **WHEN** the sentence "UserService no longer calls DatabasePool directly" is extracted
- **THEN** the resulting infon has `polarity=False`

**Testability:** A test calls `extract_text()` with real sentences using a real `AnchorSchema`, asserts that affirmative sentences produce `polarity=True`, negated sentences produce `polarity=False`, multi-sentence input produces multiple infons, and empty input returns an empty list. The encoder and schema are real — no mocked components.

---

### Requirement: Multi-Language AST Extraction Pipeline

The `infon` package SHALL implement an AST extraction pipeline that converts source code files into `Infon` instances by walking the abstract syntax tree via tree-sitter.

**Supported languages in v1:** Python (via `tree-sitter-python`) and TypeScript/JavaScript (via `tree-sitter-javascript`).

**Language detection:** The extractor SHALL detect language from file extension (`.py` → Python; `.ts`, `.tsx`, `.js`, `.jsx` → TypeScript/JavaScript). Unknown extensions SHALL be skipped with a warning.

**Pluggable extractor interface:** `BaseASTExtractor` SHALL define an abstract interface with a single method `extract(source: bytes, file_path: str, schema: AnchorSchema) -> list[Infon]`. Each language extractor SHALL subclass `BaseASTExtractor`. New language support SHALL be addable by implementing `BaseASTExtractor` and registering the subclass for a file extension — no changes to the core pipeline.

**Repo walker:** `ingest_repo(root: str, schema: AnchorSchema, store: InfonStore, *, incremental: bool = False) -> IngestStats` SHALL recursively walk a directory, dispatch each file to the appropriate extractor, and persist the resulting infons. Each ingested file SHALL be recorded in the `documents` table with its path and `ingested_at` timestamp. With `incremental=True`, it SHALL use the `documents` table to identify files whose current mtime is newer than their recorded `ingested_at` (or that are missing from the table), re-ingesting only those files. As a fallback when `documents` is empty (first run), it SHALL use `git diff --name-only HEAD` to identify changed files. Files that fail to parse SHALL be logged and skipped; the walker SHALL NOT raise on individual file failures.

**AST node to infon mapping:** Each language extractor SHALL map the node types listed in the design document to infon predicates. All AST infons use the eight built-in relation anchors. Grounding SHALL use `ASTGrounding(file_path, line_number, node_type)`.

#### Scenario: Python call expression extracted
- **WHEN** `ingest_repo` is run on a Python file containing `result = verify_token(user_id, token)`
- **THEN** an `Infon` with `predicate="calls"`, `subject` = the enclosing function's module/class key, `object="verify_token"` (if `verify_token` is a schema anchor or is discoverable), and `grounding.line_number` pointing to the call site is persisted to the store

#### Scenario: TypeScript import extracted
- **WHEN** `ingest_repo` is run on a TypeScript file containing `import { DatabasePool } from './db'`
- **THEN** an `Infon` with `predicate="imports"`, `subject` = the module name, `object="DatabasePool"` is persisted

#### Scenario: Unknown file extension skipped gracefully
- **WHEN** `ingest_repo` encounters a `.go` file
- **THEN** the file is skipped with a warning; no error is raised; the rest of the repo continues to ingest normally

#### Scenario: Incremental ingest only re-processes changed files
- **WHEN** `infon ingest --incremental` is run after two files have changed since the last ingest
- **THEN** only those two files are re-parsed; infons from unchanged files are preserved in the store

**Testability:** A test provisions a real temp directory with Python and TypeScript fixture files, calls `ingest_repo()` with a real `InfonStore` and real `AnchorSchema`, asserts >50 infons were produced, verifies specific `calls`/`imports`/`inherits` infons have correct grounding. The `tests/fixtures/` directory contains real source files parsed by real tree-sitter. No mocked extractors or parsers.

---

### Requirement: Schema Auto-Discovery (Left Kan Extension)

The `infon` package SHALL implement a `SchemaDiscovery` class that derives an `AnchorSchema` from a corpus without any manual anchor definitions, using spectral clustering on the SPLADE co-activation matrix.

**Algorithm:**

1. **Co-activation matrix:** Encode all sentences (text mode) or all symbol identifiers (AST mode) via `SpladeEncoder`. For each pair of vocabulary tokens that co-activate in the same sentence, accumulate their positive pointwise mutual information (NPMI). The result is a sparse `V × V` NPMI matrix where `V` is the SPLADE vocabulary size (30,522).
2. **Filter frequent tokens:** Retain only the top-F vocabulary tokens by total activation frequency (default F = 2,000).
3. **Graph Laplacian:** Build the filtered `F × F` NPMI adjacency matrix. Compute the normalised graph Laplacian `L = D^{-1/2} (D - A) D^{-1/2}`.
4. **Spectral clustering:** Extract the bottom-K eigenvectors of L (default K = 50). Run k-means on these eigenvectors to produce K anchor clusters.
5. **Anchor labelling:** Label each cluster by its top-3 vocabulary tokens (highest average activation within the cluster). Infer anchor type from linguistic features: clusters whose top tokens are verbs → `relation`; clusters whose top tokens are proper nouns, capitalised identifiers, or module-like tokens → `actor`; clusters whose top tokens are adjectives or attributes → `feature`.
6. **Code mode override:** In code mode, replace the relation clusters with the eight built-in relation anchors (`calls`, `imports`, `inherits`, `mutates`, `defines`, `returns`, `raises`, `decorates`). Actor clusters are derived from the symbol identifier frequency in the repo.

`SchemaDiscovery.discover(corpus, mode) -> AnchorSchema` SHALL produce an `AnchorSchema` serialisable to `.infon/schema.json`. The `version` field SHALL be set to `"auto-1.0"`.

`infon init` SHALL call `SchemaDiscovery.discover()` and write the result to `.infon/schema.json` before running the first ingest.

#### Scenario: Schema discovered from Python repo
- **WHEN** `infon init` is run on a Python repo with 200+ source files
- **THEN** `.infon/schema.json` is created containing at minimum the eight built-in relation anchors, plus actor anchors derived from the most-referenced module and class names in the repo; the process completes in under 60 seconds on a standard laptop

#### Scenario: Manual schema bypasses discovery
- **WHEN** `infon init --schema ./my-schema.json` is called
- **THEN** schema discovery is skipped; the provided schema is validated and copied to `.infon/schema.json`

#### Scenario: Minimum corpus size warning
- **WHEN** `infon init` is run on a corpus with fewer than 50 source files
- **THEN** schema discovery completes but emits a warning: "Corpus has fewer than 50 files; auto-discovered schema may be noisy. Consider providing a manual schema with --schema."

**Testability:** A test runs `SchemaDiscovery.discover()` against the real `tests/fixtures/` Python project with a real `SpladeEncoder`, asserts the produced schema contains at minimum the eight built-in relation anchors, serialises to JSON and reloads, and verifies a small corpus emits a warning (`warnings.catch_warnings`). No mocked encoder or clustering — real spectral clustering on real co-activation data.

---

### Requirement: DuckDB Store

The `infon` package SHALL implement an `InfonStore` backed by a DuckDB database file.

**Schema:** The store SHALL maintain four tables:

- `infons(id UUID, subject VARCHAR, predicate VARCHAR, object VARCHAR, polarity BOOL, grounding_type VARCHAR, grounding_json JSON, confidence FLOAT, timestamp TIMESTAMPTZ, importance_json JSON, kind VARCHAR, reinforcement_count INT, doc_id VARCHAR, created_at TIMESTAMPTZ)` — indexed on `(subject, predicate, object)`, `(subject)`, `(predicate)`, `(doc_id)`, `(timestamp)`, `(kind)`.
- `edges(id UUID, from_infon_id UUID, to_infon_id UUID, edge_type VARCHAR, weight FLOAT, created_at TIMESTAMPTZ)` — `edge_type` is one of `NEXT`, `CONTRADICTS`, `SUPPORTS`, `SIMILAR`. Indexed on `(from_infon_id, edge_type)`.
- `constraints(id UUID, subject VARCHAR, predicate VARCHAR, object VARCHAR, evidence_count INT, strength FLOAT, persistence FLOAT, updated_at TIMESTAMPTZ)` — indexed on `(subject, predicate, object)`.
- `documents(id VARCHAR, path VARCHAR, kind VARCHAR, ingested_at TIMESTAMPTZ, token_count INT)` — indexed on `(path)`.

**API:** `InfonStore` SHALL expose:
- `upsert(infon: Infon) -> Infon` — insert or merge with existing infon having the same `(subject, predicate, object, polarity)`; on merge, increment `reinforcement_count` and average `confidence`
- `get(id: UUID) -> Infon | None`
- `query(subject=None, predicate=None, object=None, min_confidence=0.0, limit=100) -> list[Infon]`
- `add_edge(from_id, to_id, edge_type, weight)`
- `get_edges(infon_id, edge_type=None) -> list[Edge]`
- `upsert_constraint(subject, predicate, object, evidence_count, strength, persistence)`
- `upsert_document(doc_id, path, kind, token_count)`
- `stats() -> StoreStats` — counts of infons, edges, constraints, documents; top-10 most-referenced anchors
- `close()`

All write operations SHALL be wrapped in transactions. The store SHALL use DuckDB's WAL mode. Concurrent reads are supported; concurrent writes are not — the store SHALL raise `ConcurrentWriteError` if a second writer attempts to open the same `.ddb` file while it is already open for writing.

#### Scenario: Infon upserted and reinforced
- **WHEN** `store.upsert(infon_a)` is called twice with two `Infon` instances having the same `(subject, predicate, object, polarity)` but different `doc_id`
- **THEN** the store contains exactly one row for that triple; its `reinforcement_count` is 2; its `confidence` is the average of the two input confidences

#### Scenario: Stats returns correct counts
- **WHEN** 500 infons and 200 edges have been persisted and `store.stats()` is called
- **THEN** the returned `StoreStats` has `infon_count=500` and `edge_count=200`

**Testability:** A test provisions a real DuckDB store in a temp directory, upserts real `Infon` instances, calls `get()` and `query()` with filter combinations, adds real edges, upserts constraints and documents, calls `stats()` and verifies counts, tests concurrent write detection by opening a second writer, and uses the context manager to verify clean close. The database is a real `.ddb` file — no patched database connections.

---

### Requirement: Consolidation

The `infon` package SHALL implement a `consolidate(store: InfonStore, schema: AnchorSchema)` function that enriches the knowledge graph after ingest.

**Consolidation steps:**

1. **Reinforcement (already handled at upsert):** Duplicate triples are merged by `InfonStore.upsert`. `consolidate()` does not re-run this step but does recompute `importance.reinforcement` scores based on updated `reinforcement_count` values.
2. **NEXT edges:** For each anchor key that appears as `subject` or `object` in multiple infons, sort those infons by `timestamp`. Add a `NEXT` edge from each infon to the next chronological infon sharing that anchor. `weight = 1 / (1 + days_between)`.
3. **Constraint aggregation:** For each distinct `(subject, predicate, object)` triple, compute `evidence_count = reinforcement_count`, `strength = average(confidence)`, `persistence = 1 - decay_factor^evidence_count`. Upsert to `constraints`.
4. **Importance decay:** For infons with `timestamp` older than 7 days, apply exponential decay to `importance.reinforcement`: `new_value = old_value * decay_factor^days_since_timestamp`. Default `decay_factor = 0.95`. Infons that are reinforced (their `reinforcement_count` increases) are exempt from decay for 7 days after the last reinforcement.

`consolidate()` SHALL be idempotent — running it twice produces the same result as running it once.

#### Scenario: NEXT edges created for temporal chain
- **WHEN** three infons share the anchor `user_service` as subject and have timestamps T1 < T2 < T3, and `consolidate()` is called
- **THEN** the store contains NEXT edges T1→T2 and T2→T3 for the `user_service` anchor; there is no direct T1→T3 edge

#### Scenario: Consolidation is idempotent
- **WHEN** `consolidate(store, schema)` is called, then called again without any new infons
- **THEN** the edge count and constraint count are identical after both calls; no duplicate edges are created

**Testability:** A test populates a real `InfonStore` with chronologically ordered real infons sharing an anchor, calls `consolidate()`, asserts NEXT edges exist between consecutive infons with correct weights, calls `consolidate()` again and asserts edge count is unchanged (idempotency), verifies constraint aggregation produces correct `evidence_count`/`strength`/`persistence`, and manually backdates infon timestamps to test importance decay. No mocked store — real DuckDB with real infons.

---

### Requirement: Query Engine / Retrieval

The `infon` package SHALL implement a `retrieve(query: str, store: InfonStore, schema: AnchorSchema, *, limit: int = 10, persona: str | None = None) -> list[ScoredInfon]` function.

**Retrieval pipeline:**

1. **Query encoding:** Encode the query string to anchor space via `encode(query, schema)`. Retain anchors above activation threshold.
2. **Anchor expansion:** For each activated anchor key, expand to all descendants via `schema.descendants(key)` and add them to the candidate anchor set.
3. **Candidate fetch:** Retrieve all infons from the store where `subject` OR `object` is in the candidate anchor set.
4. **Valence scoring:** If `persona` is specified, apply persona-specific valence weights to the predicate anchor of each candidate infon. Supported personas: `investor`, `engineer`, `executive`, `regulator`, `analyst`. Each persona has a set of positive predicates (+1), negative predicates (-1), and neutral predicates (0).
5. **Relevance scoring:** Score each candidate infon as `score = query_anchor_overlap × confidence × importance.reinforcement × (1 + valence_weight)`.
6. **NEXT-edge context:** For the top-K candidates, fetch their immediately preceding and following NEXT-edge infons and include them as context in `ScoredInfon.context`.
7. **Ranking and deduplication:** Sort by `score` descending, deduplicate by `(subject, predicate, object)` keeping the highest-scored instance, return top `limit`.

`ScoredInfon` SHALL carry the `Infon` plus `score: float`, `context: list[Infon]` (NEXT-edge neighbours).

#### Scenario: Search returns ranked results
- **WHEN** `retrieve("what calls the database pool", store, schema, limit=5)` is called on a store containing infons about `DatabasePool`
- **THEN** the returned list contains at most 5 `ScoredInfon` instances, sorted by score descending, all with `object` or `subject` matching `database_pool` or a descendant anchor

#### Scenario: Persona valence shifts ranking
- **WHEN** `retrieve("authentication", store, schema, persona="regulator")` is called
- **THEN** infons with `predicate="validates"` or `predicate="enforces"` (positive for regulator persona) rank higher than equivalent infons with `predicate="bypasses"` (negative for regulator)

**Testability:** A test populates a real `InfonStore` with diverse infons and edges, calls `retrieve()` with real queries and asserts results are sorted by score descending, validates anchor expansion includes descendants, verifies persona valence shifts ranking order, and tests empty store returns empty list. Real encoder, real store, real schema — no mocked pipeline stages.

---

### Requirement: MCP Server

The `infon` package SHALL implement a stdio MCP server (`infon serve`) using FastMCP, exposing three tools and three resources.

**Tools:**

- `search(query: str, limit: int = 10) -> list[dict]` — runs `retrieve()` and returns ranked infons as JSON-serialisable dicts. Each dict SHALL include `subject`, `predicate`, `object`, `polarity`, `confidence`, `score`, `grounding` (file path + line or doc + span), and `context` (adjacent NEXT-edge infons).
- `store_observation(text: str, source: str = "agent") -> dict` — constructs a `doc_id` as `"<source>:<current-UTC-timestamp>"`, runs `extract_text()` on the input with that `doc_id`, persists the resulting infons to the store, runs `consolidate()`, and returns a summary `{infons_added: int, infons_reinforced: int}`.
- `query_ast(symbol: str, relation: str | None = None, limit: int = 20) -> list[dict]` — queries the store for infons where `subject` or `object` matches `symbol` (exact or ancestor match), optionally filtered by `predicate=relation`. Returns infons sorted by `reinforcement_count` descending.

**Resources:**

- `infon://stats` — returns the output of `store.stats()` as a Markdown-formatted string.
- `infon://schema` — returns the active `AnchorSchema` as a formatted anchor list grouped by type.
- `infon://recent` — returns the 20 most recently added infons (by `created_at`) as a Markdown table.

**Server startup:**

`infon serve [--db PATH]` SHALL:
1. Open or create the DuckDB store at `PATH` (default: `.infon/kb.ddb` relative to cwd)
2. Load the schema from `PATH/../schema.json`
3. Start the FastMCP stdio server loop
4. On shutdown (SIGTERM or stdin close), flush any pending writes and close the store

The MCP server SHALL handle tool call errors gracefully — if `search` or `query_ast` raises, it SHALL return a JSON error object rather than crashing the server process.

#### Scenario: search tool called by Claude Code
- **WHEN** Claude Code calls the `search` tool with `query="what calls DatabasePool?"`
- **THEN** the server returns a list of infons with `predicate="calls"` and `object` matching the database pool anchor, including their file path and line number groundings

#### Scenario: store_observation persists and consolidates
- **WHEN** Claude Code calls `store_observation("Decided to use the repository pattern for all database access")` during a session
- **THEN** the observation is extracted into infons, persisted to `.infon/kb.ddb`, and consolidation runs; subsequent `search("repository pattern")` calls return this observation

#### Scenario: Server recovers from tool error
- **WHEN** `query_ast` is called with a `symbol` that does not exist in the schema
- **THEN** the server returns `{"error": "symbol not found in schema", "symbol": "<value>"}` as the tool result; the MCP server process continues running

#### Scenario: MCP server configured via .mcp.json with uvx
- **WHEN** a project's `.mcp.json` contains `{"mcpServers": {"infon": {"type": "stdio", "command": "uvx", "args": ["--from", "infon", "infon", "serve"]}}}`
- **AND** the user starts a Claude Code session in that project directory
- **THEN** Claude Code spawns the `infon serve` process via uvx and the three tools are available to the agent without any further configuration

**Testability:** A test spawns a real `run_server()` subprocess with a pre-populated real `InfonStore`, opens a real JSON-RPC session over stdin/stdout, sends `tools/list` and asserts all three tools are registered, calls `search` via JSON-RPC and asserts ranked results match stored infons, calls `store_observation` and verifies new infons are persisted plus consolidation ran, calls `query_ast` for a known symbol and verifies results, sends an invalid `query_ast` payload and asserts a JSON error dict without process crash, and fetches `infon://stats`, `infon://schema`, `infon://recent` resources asserting non-empty Markdown. Real subprocess, real store, real JSON-RPC — no mocked MCP transport.

---

### Requirement: CLI

The `infon` package SHALL expose a command-line interface via the `infon` entry point, implemented with Click.

**Commands:**

- `infon init [--schema PATH] [--db PATH]` — Runs schema discovery (or loads provided schema), writes `.infon/schema.json`, creates `.infon/kb.ddb`, runs `ingest_repo` on the current directory, writes `.mcp.json` if one does not exist, appends `.infon/` to `.gitignore`. Prints a summary on completion. SHALL complete in under 120 seconds on a 100k-line Python repo.
- `infon ingest [PATH] [--db PATH] [--incremental]` — Runs `ingest_repo` on PATH (default: cwd), followed by `consolidate()`. Prints ingestion stats. With `--incremental`, only re-ingests files changed since the last ingest via `git diff --name-only`.
- `infon search QUERY [--db PATH] [--limit N] [--persona PERSONA]` — Runs `retrieve()` and prints results as a formatted table with subject, predicate, object, score, and grounding.
- `infon stats [--db PATH]` — Prints `store.stats()` as a formatted summary.
- `infon serve [--db PATH]` — Starts the FastMCP stdio MCP server.

All commands that open the DuckDB store SHALL print a clear error and exit with code 1 if the store file does not exist and `--db` was not specified — with a hint to run `infon init` first.

#### Scenario: infon init runs end-to-end
- **WHEN** `infon init` is run in a Python repo with no existing `.infon/` directory
- **THEN** `.infon/schema.json` is created, `.infon/kb.ddb` is created and populated with infons from the repo's source files, `.mcp.json` is written (if absent), and `.gitignore` is updated; the exit code is 0

#### Scenario: infon search returns results
- **WHEN** `infon search "what imports jwt"` is run after a successful `infon init`
- **THEN** the output contains at least one row with `predicate=imports` and `object` matching a JWT-related anchor, along with the file path grounding

#### Scenario: Missing store gives actionable error
- **WHEN** `infon search "anything"` is run in a directory with no `.infon/` directory
- **THEN** the CLI prints "No knowledge base found. Run 'infon init' to create one." and exits with code 1

**Testability:** A test uses Click's `CliRunner` to invoke the first four subcommands against `tests/fixtures/`, asserts `infon init` end-to-end produces `.infon/` with schema and populated DB, `infon search` returns tabular output after init, `infon stats` prints non-empty output, missing store exits 1 with expected message. A separate test provisions a real git repo in a temp directory with known commits on two files, runs `infon init` via subprocess to ingest all files, modifies one file and commits, then runs `infon ingest --incremental` via subprocess and asserts only the changed file's infons are re-created. No mocked CLI — real Click runner with real store and real fixtures.

---

### Requirement: Three Install Paths

The `infon` package SHALL support three install and invocation paths, all producing an identical runtime.

**Path A — uvx (zero-install):**
- The package SHALL be published to PyPI as `infon`.
- `uvx infon <command>` SHALL work without any prior `pip install`. `uvx` downloads and caches the package on first run.
- The generated `.mcp.json` from `infon init` SHALL use `command: uvx` as the default MCP server invocation.

**Path B — pip install:**
- `pip install infon` SHALL install the `infon` entry point globally (or in the active virtualenv).
- After installation, all CLI commands SHALL be available as `infon <command>`.
- The package SHALL declare all runtime dependencies in `pyproject.toml` with minimum version constraints. It SHALL NOT use `>=` without an upper bound for dependencies with known breaking changes.

**Path C — pip + project venv:**
- The package SHALL function correctly when installed into a project-local virtualenv.
- `infon serve` invoked from a venv SHALL write a `.mcp.json` that uses the full venv path to the `infon` binary (e.g., `.venv/bin/infon serve`), not `uvx`.

The README SHALL document all three paths with copy-pasteable commands. The docs site SHALL have a dedicated Installation page covering all three paths with explanatory text.

#### Scenario: uvx cold start succeeds
- **WHEN** `uvx infon serve` is run on a machine where `infon` has never been installed
- **THEN** `uv` downloads and caches the package and its dependencies, and the MCP server starts successfully; the total time from cold start to server ready SHALL be under 30 seconds on a standard internet connection

#### Scenario: pip install provides correct entry point
- **WHEN** `pip install infon` completes in a fresh virtualenv
- **THEN** `infon --version` prints the package version and exits 0; `infon --help` lists all commands

**Testability:** A test creates a fresh Python virtualenv, runs `pip install` from the local package, invokes `infon --version` and `infon --help` as subprocesses and asserts correct output, and verifies `pip install` into a project-local venv writes the correct binary path in `.mcp.json`. These are verification tasks (Phase 13) run as real subprocess commands — no mocked pip or venv.

---

### Requirement: Documentation Site

The `infon` package SHALL include a MkDocs Material documentation site at `docs/` in the repository root, automatically deployed to GitHub Pages on every push to `main`.

**Required pages:**
- `index.md` — Overview, 30-second quick start (uvx path), conceptual summary of infons and anchors
- `installation.md` — All three install paths with explanatory text and copy-pasteable commands
- `concepts.md` — Infons, AnchorSchema, change of basis, SPLADE projection, grounding, explained in plain English
- `cli.md` — All CLI commands with flags, examples, and expected output
- `mcp.md` — MCP server setup, the three tools and three resources, `.mcp.json` examples for all three install paths
- `ast-extraction.md` — AST extraction, supported languages, AST-to-infon mapping table, how to add a new language
- `schema-discovery.md` — What schema discovery does, the left Kan extension algorithm in plain English, how to inspect and edit the auto-discovered schema
- `api-reference.md` — Python API reference for `Infon`, `AnchorSchema`, `InfonStore`, `encode`, `extract_text`, `ingest_repo`, `retrieve`, `consolidate`
- `contributing.md` — Development setup, running tests, adding a language extractor, submitting a PR

**GitHub Actions workflow** at `.github/workflows/docs.yml` SHALL build and deploy the docs site on every push to `main` using `mkdocs gh-deploy --force`.

#### Scenario: Docs site deployed on push to main
- **WHEN** a commit is pushed to `main`
- **THEN** the `docs.yml` GitHub Actions workflow runs, `mkdocs build` succeeds with no warnings, and the updated site is live at `https://<owner>.github.io/infon` within 3 minutes

**Testability:** A test runs `mkdocs build` as a subprocess against a real `docs/` directory with all required pages populated, asserts the build succeeds with zero warnings, and verifies the generated HTML contains all expected page titles. The `gh-deploy` step is verified by CI, but the local build is testable in every test run.

---

### Requirement: CI and PyPI Publishing

The `infon` repository SHALL include GitHub Actions workflows for continuous integration and automated PyPI publishing.

**CI workflow** (`.github/workflows/ci.yml`): runs on every push and pull request to `main`.
- `pytest tests/` — all tests must pass
- `ruff check src/` — no linting errors
- `ruff format --check src/` — no formatting errors
- `mypy src/infon/` — no type errors

**PyPI publish workflow** (`.github/workflows/publish.yml`): runs on every GitHub Release (tag push matching `v*.*.*`).
- Builds the wheel and sdist with `python -m build`
- Publishes to PyPI using OIDC trusted publishing (no API token stored in secrets)

All tests SHALL be integration tests — no unit tests with mocks. Tests run against a real DuckDB instance (ephemeral, created in a temp directory per test). Tests that require the SPLADE model SHALL load it from the HuggingFace transformers cache (the CI workflow pre-warms the cache so subsequent runs are offline). Tests that require a repo to ingest SHALL use the `tests/fixtures/` directory which contains a small synthetic Python + TypeScript codebase.

#### Scenario: CI passes on a clean PR
- **WHEN** a pull request is opened with no test failures and no lint errors
- **THEN** all three CI checks (pytest, ruff, mypy) pass and the PR is marked green

#### Scenario: PyPI publish triggered by release
- **WHEN** a GitHub Release is created with tag `v0.1.0`
- **THEN** the publish workflow builds the wheel and uploads it to PyPI; `pip install infon==0.1.0` succeeds within 5 minutes of the release

**Testability:** A test runs `python -m build` as a subprocess and asserts the wheel and sdist are produced in `dist/`, verifies `ruff check src/` and `mypy src/infon/` succeed as subprocesses with no errors, and verifies lint/format pass. The PyPI upload itself is gated by CI OIDC but the build artifact and all CI checks are testable locally as real subprocess commands — no mocked CI runner.

# Spec: Epic 03 (v4) — Evidence Generation + Baselines

## Hard Rules

- **TDD only:** integration test before implementation; red → green; no shortcuts.
- **No mocks for evaluation:** real InfonStore, real cassette ingests, real DS mass computations. The single permitted exception is the LLM cache: tests run in replay-only mode against committed fixtures; the cache is real data, not a mock.
- **No dataset-specific branching in the harness:** the evaluation loop iterates over `EvalClaim` and `EvalSystem` — no `if dataset == "hover"` in `benchmark_eval.py`.
- **Reproduce-before-report:** each NLI baseline must reproduce its published accuracy on its native benchmark to within ±2 pp before any comparative result is reported. The LLM baseline: ±3 pp on any published zero-shot number on the same split.
- **Stage-boundary review:** at each stage's end, run `pytest -v`, re-read this spec, verify compliance.

---

## ADDED Requirements

### Requirement: Dataset Loaders

`reference_v4/benchmarks/hover.py` SHALL provide `HoVerLoader(split: Literal["train","dev"]) → list[EvalClaim]`. Each returned claim SHALL have:
- `ground_truth ∈ {"SUPPORTS", "REFUTES"}` (HoVer has no NEI class)
- `metadata["num_hops"] ∈ {2, 3, 4}`
- `evidence_docs`: the Wikipedia sentences cited as supporting facts

`reference_v4/benchmarks/averitec.py` SHALL provide `AVeriTeCLoader(split, version="v2") → list[EvalClaim]`. Each claim SHALL have:
- `ground_truth ∈ {"SUPPORTS", "REFUTES", "NEI"}` (mapped from AVeriTeC's four-class label: `CONFLICTING_EVIDENCE` → `REFUTES`)
- `metadata["nei_indicator"]: int` (1 if `NOT_ENOUGH_EVIDENCE`, 0 otherwise)
- `evidence_docs`: the question-answer pairs flattened to text

`reference_v4/benchmarks/scifact.py` SHALL provide `SciFactLoader(split) → list[EvalClaim]`. Each claim SHALL have:
- `ground_truth ∈ {"SUPPORTS", "REFUTES", "NEI"}`
- `metadata["rationales"]: list[str]` (the cited abstract sentences)
- `evidence_docs`: the full abstracts of cited documents

All three loaders SHALL cache downloads to `reference_v4/experiments/data/<dataset>/` and SHALL NOT re-download if the cache file exists and its SHA-256 matches the committed `data/SHA256SUMS`.

#### Scenario: HoVer dev count matches published
- **WHEN** `HoVerLoader("dev")` is called
- **THEN** the count matches the HoVer paper's published dev-set count (4,000 claims)

#### Scenario: AVeriTeC NEI class surfaced
- **WHEN** `AVeriTeCLoader("dev")` is called
- **THEN** at least one claim has `ground_truth == "NEI"` and `metadata["nei_indicator"] == 1`

#### Scenario: SciFact rationales non-empty on a SUPPORTS claim
- **WHEN** a SciFact SUPPORTS claim is loaded
- **THEN** `metadata["rationales"]` is non-empty

**Testability:** unit tests on cached fixtures; count assertion on HoVer; structural assertions on the first claim of each dataset.

---

### Requirement: EvalClaim Dataclass

`reference_v4/benchmarks/types.py` SHALL define `EvalClaim` with: `claim_id: str`, `claim_text: str`, `evidence_docs: list[str]`, `ground_truth: Literal["SUPPORTS", "REFUTES", "NEI"]`, `metadata: dict`.

#### Scenario: EvalClaim is uniform across datasets
- **WHEN** the first claim from each of the three loaders is inspected
- **THEN** all three have non-empty `claim_text`, a valid `ground_truth`, and non-empty `evidence_docs`

**Testability:** parametric test over all three loaders.

---

### Requirement: Uniform EvalSystem Interface

`reference_v4/baselines/types.py` SHALL define the `EvalSystem` protocol:

```python
class EvalSystem(Protocol):
    name: str
    def evaluate(self, claim: EvalClaim) -> MassFunction: ...
```

`MassFunction` is a `NamedTuple` or dataclass with fields `supports: float, refutes: float, uncertain: float, theta: float`, all non-negative and summing to 1 within `1e-6`.

#### Scenario: All systems satisfy the interface
- **WHEN** every system class (symbolic floor, NLI, LLM, Cognition symbolic, Cognition+GNN) is instantiated
- **THEN** `evaluate()` on a fixture claim returns a `MassFunction` that sums to 1 within `1e-6`

**Testability:** parametric test over all system classes with a fixture claim.

---

### Requirement: Cognition Evaluation System

`reference_v4/baselines/cognition_system.py` SHALL provide `CognitionSystem(mode: Literal["symbolic", "gnn"])` wrapping `InfonStore`. Its `evaluate()` method SHALL:
1. Create a fresh `InfonStore` in a temporary directory.
2. Run the two-iteration schema bootstrap on `claim.evidence_docs`.
3. Ingest `evidence_docs` under the generated schema.
4. Query with `store.ask(Query().where(...))` derived from `claim.claim_text`.
5. Return the verdict's `MassFunction`.

If schema coverage < 80% after two iterations, return `MassFunction(0, 0, 0, 1.0)` (maximum θ) and log the claim ID in a coverage-failure list.

#### Scenario: Evaluate returns a valid mass on a simple claim
- **GIVEN** a fixture `EvalClaim` with one evidence document containing the subject, predicate, and object of the claim
- **WHEN** `CognitionSystem("gnn").evaluate(claim)` is called
- **THEN** the returned mass sums to 1 within `1e-6` and `supports > 0`

**Testability:** real InfonStore, real ingest, real DS mass; fixture claim uses the same EV scenario from Epic 01.

---

### Requirement: Symbolic Floor Baseline

`reference_v4/baselines/symbolic_floor.py` SHALL provide `SymbolicFloor()`. Its `evaluate()` method SHALL:
1. Retrieve the top-5 sentences from `evidence_docs` by SPLADE cosine similarity to `claim_text`.
2. Apply a rule-based polarity detector: if any retrieved sentence contains a negation token near the claim's key predicate, assign REFUTES; otherwise SUPPORTS.
3. If retrieval score < threshold `τ = 0.15`, return `MassFunction(0, 0, 0, 1.0)` (abstain).
4. Map verdict and retrieval score to a 3-element mass; set `theta = max(0, 1 - top_score)`.

#### Scenario: Abstain on weak retrieval
- **GIVEN** a claim whose claim text shares no vocabulary with the evidence docs
- **WHEN** `SymbolicFloor().evaluate(claim)` is called
- **THEN** the returned `MassFunction.theta == 1.0`

**Testability:** constructive fixture with disjoint vocabularies.

---

### Requirement: NLI Classifier Baseline

`reference_v4/baselines/nli_classifier.py` SHALL provide `NLIClassifier(model_name="cross-encoder/nli-deberta-v3-large")`. Its `evaluate()` method SHALL:
1. Concatenate claim text with each evidence document.
2. Run the NLI model over all premise-hypothesis pairs.
3. Aggregate: majority vote over {ENTAILMENT, CONTRADICTION, NEUTRAL}; map to {SUPPORTS, REFUTES, NEI}.
4. Set `MassFunction` from softmax probabilities; set `uncertain = theta = 0` (the model produces a 3-way mass).
5. Apply temperature scaling (T=1.5, calibration constant from held-out SciFact dev) to soften overconfident predictions.

#### Scenario: NLI output sums to 1
- **WHEN** `NLIClassifier().evaluate(fixture_claim)` is called
- **THEN** the returned `MassFunction` has `theta == 0`, `uncertain == 0`, and `supports + refutes + uncertain == 1` within `1e-6`

**Testability:** fixture claim with a strong entailment signal (claim text is a copy of the first evidence sentence).

---

### Requirement: CognitionSystem Uses Raw InfonStore API — No LLM at Inference

`CognitionSystem.evaluate()` SHALL call `store.ask()`, `store.connect()`, and `store.any_of()` directly on the `InfonStore`. It SHALL NOT instantiate `Analyst` or make any LLM API calls during evaluation. The Strands-powered `Analyst` is the user-facing conversational layer; `InfonStore` is the benchmarking surface.

**Why this must be specified:** `reference_v4/cognition/src/cognition/cassette/analyst.py` uses an LLM to route natural-language questions to the right store primitive. Using `Analyst` in the evaluation loop would (a) introduce an LLM into the "our system" column, conflating the comparison with the LLM baseline, and (b) make the evaluation non-deterministic without caching the Analyst's routing calls. The `ask()` / `connect()` APIs are fully deterministic given a fixed store state and schema.

The schema bootstrap (two-iteration spectral clustering via `extraction_report()`) is algorithmic and does not use an LLM. The `set_schema()` call in the Analyst simply loads a JSON dict — this part is reused.

#### Scenario: CognitionSystem makes zero LLM API calls
- **GIVEN** a `CognitionSystem` with LLM call monitoring enabled
- **WHEN** `evaluate()` is called on any fixture claim
- **THEN** zero calls are made to the Anthropic API or any other LLM provider

**Testability:** patch `anthropic.Anthropic` at the module level; assert it is never instantiated during a `CognitionSystem.evaluate()` call.

---

### Requirement: LLM SDK, Authentication, and Call Parameters

All LLM calls (the `LLMZeroShot` baseline only) SHALL use the `anthropic` Python SDK (`pip install anthropic>=0.28`). The client SHALL be instantiated as:

```python
import anthropic
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment
```

The API key SHALL be read from the environment variable `ANTHROPIC_API_KEY`. It SHALL NOT be hard-coded, logged, or committed. If the key is absent at instantiation time in live (non-replay) mode, `LLMZeroShot.__init__` SHALL raise `EnvironmentError` with a message naming the missing variable. In replay-only mode (cache hit path), the key is not required and its absence SHALL NOT raise.

Every API call SHALL use:

```python
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=256,
    temperature=0,
    system=SYSTEM_PROMPT,
    messages=[{"role": "user", "content": user_prompt}],
)
```

`temperature=0` enforces deterministic output. `max_tokens=256` is sufficient for the JSON response format; claims requiring longer reasoning are handled by the prompt's instruction to be concise.

#### Scenario: Missing API key raises in live mode
- **GIVEN** `ANTHROPIC_API_KEY` is unset
- **WHEN** `LLMZeroShot(replay_only=False)` is instantiated
- **THEN** `EnvironmentError` is raised naming `ANTHROPIC_API_KEY`

#### Scenario: Missing API key does NOT raise in replay mode
- **GIVEN** `ANTHROPIC_API_KEY` is unset
- **WHEN** `LLMZeroShot(replay_only=True)` is instantiated and `evaluate()` is called on a cached claim
- **THEN** no error is raised; the cached mass is returned

**Testability:** real `LLMZeroShot` instantiation; `ANTHROPIC_API_KEY` temporarily removed from environment via `monkeypatch.delenv`.

---

### Requirement: LLM Prompt Template

The `LLMZeroShot` baseline SHALL use the following fixed prompt structure. Templates are module-level constants, not constructed at call time.

**System prompt** (`SYSTEM_PROMPT` constant in `llm_zeroshot.py`):

```
You are a fact-checking assistant. Your task is to determine whether a claim is
supported, refuted, or has insufficient evidence, based solely on the provided
evidence passages. Do not use any knowledge beyond what is given.

Respond with a JSON object on a single line with exactly two fields:
  "verdict": one of "SUPPORTS", "REFUTES", or "NEI" (not enough evidence)
  "confidence": a float in [0.0, 1.0] expressing how certain you are

Use "NEI" whenever the evidence passages do not address the claim, even partially.
A verdict of "NEI" MUST have confidence 1.0 — certainty that evidence is absent
is itself a form of certainty.
```

**User prompt** (`USER_PROMPT_TEMPLATE` constant, formatted per claim):

```
Evidence passages:
{evidence_block}

Claim: {claim_text}

Respond with JSON only. No explanation.
```

Where `{evidence_block}` is constructed as:

```
[1] {evidence_docs[0]}

[2] {evidence_docs[1]}

...
```

Each document is prefixed with a `[N]` index. Documents are separated by a blank line. The block is truncated per the evidence truncation requirement below.

#### Scenario: Prompt template is stable across calls
- **WHEN** `LLMZeroShot._build_prompt(fixture_claim)` is called twice with the same claim
- **THEN** the returned system prompt and user prompt are byte-identical

**Testability:** determinism assertion; no randomness in prompt construction.

---

### Requirement: Evidence Truncation

Long evidence corpora will exceed the context window or inflate token cost. The following truncation rules SHALL be applied before constructing `{evidence_block}`:

1. Each individual document is truncated to its first **600 words** (split on whitespace). If a document exceeds 600 words, the truncated version ends with `[truncated]`.
2. The total `{evidence_block}` is capped at **3,000 words** across all documents. If the sum of (truncated) document lengths exceeds 3,000 words, the last documents in the list are dropped (not truncated mid-document) until the cap is met. Dropped documents are noted as `[N documents omitted — evidence truncated at 3000-word limit]` appended at the end of the block.
3. If after truncation the evidence block is empty (zero documents), the user prompt includes `Evidence passages: [none provided]` and the expected verdict is NEI.

These thresholds are module-level constants (`MAX_WORDS_PER_DOC = 600`, `MAX_TOTAL_WORDS = 3000`) so they can be adjusted without changing logic.

#### Scenario: Long document truncated with marker
- **GIVEN** an evidence document of 1000 words
- **WHEN** `LLMZeroShot._truncate_evidence([long_doc])` is called
- **THEN** the returned string has ≤ 600 words and ends with `[truncated]`

#### Scenario: Total cap drops last documents
- **GIVEN** five documents of 800 words each (4000 total words)
- **WHEN** `LLMZeroShot._truncate_evidence(five_docs)` is called
- **THEN** the block contains 5 documents each truncated to 600 words = 3000 words total; all five fit exactly; the omission marker is absent

#### Scenario: Over-cap case
- **GIVEN** six documents of 600 words each (3600 total after per-doc truncation)
- **WHEN** `LLMZeroShot._truncate_evidence(six_docs)` is called
- **THEN** only five documents appear; the sixth is replaced by the omission marker

**Testability:** unit tests on `_truncate_evidence`; word-count assertions; marker presence/absence.

---

### Requirement: Confidence-to-MassFunction Mapping

The mapping from `(verdict, confidence)` to `MassFunction` is a formal contract. It SHALL be implemented as a pure function `verdict_to_mass(verdict: str, confidence: float) -> MassFunction` in `reference_v4/baselines/llm_zeroshot.py`.

The full mapping:

| LLM output | `m_supports` | `m_refutes` | `m_uncertain` | `m_theta` |
|---|---|---|---|---|
| `SUPPORTS` | `confidence` | `0` | `0` | `1 − confidence` |
| `REFUTES` | `0` | `confidence` | `0` | `1 − confidence` |
| `NEI` | `0` | `0` | `0` | `1.0` |
| Parse failure (any) | `0` | `0` | `0` | `1.0` |

**Rationale for NEI → θ = 1.0 (not `m_uncertain = 1.0`):** NEI means the corpus is silent. In DS terms this is *total ignorance* — the frame of discernment is open; we assign all mass to the vacuous element Θ. Assigning mass to `m_uncertain` would assert that we know something (that the claim's truth value is uncertain), which is epistemically stronger than "no evidence". The distinction is the core of H2: a system that returns `m_theta = 1.0` on NEI claims is **more honest** than one that spreads mass over S/R/U. The LLM baseline's θ on NEI claims (measured in the H2 panel) will be close to 0 if the model is overconfident — because the system prompt's `confidence 1.0` instruction for NEI is often not followed by the model.

**Rationale for parse failure → θ = 1.0:** A response that cannot be parsed is equivalent to "no usable evidence" for this claim. Maximum ignorance is the correct response.

**Validation:** `verdict_to_mass` SHALL assert that the returned mass sums to 1 within `1e-6` before returning; raise `AssertionError` if not (this would indicate a floating-point edge case in the formula).

#### Scenario: SUPPORTS verdict maps correctly
- **GIVEN** `verdict="SUPPORTS"`, `confidence=0.85`
- **WHEN** `verdict_to_mass("SUPPORTS", 0.85)` is called
- **THEN** `MassFunction(supports=0.85, refutes=0, uncertain=0, theta=0.15)`; sum == 1.0

#### Scenario: NEI verdict maps to full theta
- **GIVEN** `verdict="NEI"`, `confidence=0.95`
- **WHEN** `verdict_to_mass("NEI", 0.95)` is called
- **THEN** `MassFunction(supports=0, refutes=0, uncertain=0, theta=1.0)`; confidence is ignored for NEI

#### Scenario: Unknown verdict treated as parse failure
- **GIVEN** `verdict="UNCERTAIN"` (not a valid label)
- **WHEN** `verdict_to_mass("UNCERTAIN", 0.5)` is called
- **THEN** `MassFunction(supports=0, refutes=0, uncertain=0, theta=1.0)`

**Testability:** unit tests on `verdict_to_mass`; all four rows of the mapping table; edge cases (confidence=0.0, confidence=1.0).

---

### Requirement: LLM Record-and-Replay Cache

`reference_v4/baselines/_llm_cache.py` SHALL provide `LLMCache` with `record` and `replay_only` modes.

**Cache key:** `sha256(model_name + "\x00" + system_prompt + "\x00" + user_prompt + "\x00" + str(temperature) + "\x00" + str(max_tokens))`. All six components are included so that a change to any call parameter produces a cache miss (rather than returning a stale response).

**Cache format:** A single JSON Lines file at `reference_v4/baselines/llm_cache.jsonl`. Each line is a JSON object with fields `key` (the SHA-256 hex digest), `response_text` (the raw completion string), `model`, `timestamp_utc`, `input_tokens`, `output_tokens`. The file is append-only in record mode; never modified in replay mode.

**Modes:**
- `record` mode: on a cache miss, calls the Anthropic API, appends the response to `llm_cache.jsonl`, and returns the response text. On a cache hit, returns the cached response without an API call.
- `replay_only` mode: on a cache miss, raises `LLMCacheMissError(key=<hex>, prompt_sha=<first 12 chars>)`. Never calls the API.

**Committed fixture cache:** A small `reference_v4/tests/fixtures/llm_cache_fixture.jsonl` SHALL be committed containing at minimum 5 pre-recorded responses (one SUPPORTS, one REFUTES, one NEI, one with a non-JSON response, one with an NEI-but-low-confidence response) for use in all replay-mode tests.

**Token tracking:** `LLMCache` SHALL maintain a `total_input_tokens: int` and `total_output_tokens: int` counter across all calls in the session. These are read by the evaluation harness's budget guard.

#### Scenario: Replay returns cached completion without network I/O
- **GIVEN** the fixture cache file contains a pre-recorded response for prompt hash H
- **WHEN** `LLMCache(mode="replay_only").get(key=H)` is called
- **THEN** the cached `response_text` is returned; no network call is made

#### Scenario: Cache miss in replay mode raises with informative message
- **WHEN** `LLMCache(mode="replay_only").get(key=unknown_key)` is called
- **THEN** `LLMCacheMissError` is raised; the error message includes the first 12 chars of the key

#### Scenario: Record mode appends to cache file
- **GIVEN** a cache file that does not contain key K
- **WHEN** `LLMCache(mode="record").get(key=K, ...)` is called (with a mock API response)
- **THEN** the cache file gains a new line with `"key": K` and the response text

#### Scenario: Token counter accumulates across calls
- **GIVEN** a `LLMCache` instance in replay mode
- **WHEN** three cached responses (each with `input_tokens=100, output_tokens=50`) are retrieved
- **THEN** `cache.total_input_tokens == 300` and `cache.total_output_tokens == 150`

**Testability:** real file I/O; fixture cache; no MagicMock; `monkeypatch` used only to swap the cache file path.

---

### Requirement: LLM Token Budget Guard

The evaluation harness SHALL accept `--max-input-tokens INT` (default: 5,000,000 — approximately $15 at claude-sonnet-4-6 pricing). When the cumulative `total_input_tokens` across all LLM calls in the session reaches this limit:

1. The current cell is completed (no mid-cell kill).
2. All remaining LLM cells are skipped and recorded in the output as `{"status": "budget_exhausted"}`.
3. A warning is printed to stderr: `LLM budget exhausted after N input tokens (limit M). Skipping K remaining LLM cells.`

The budget guard reads `LLMCache.total_input_tokens` after each claim evaluation.

#### Scenario: Budget exhaustion skips remaining cells gracefully
- **GIVEN** `--max-input-tokens 100` and a dataset of 10 claims each using ~80 tokens
- **WHEN** the harness runs
- **THEN** the first claim completes; the remaining 9 cells are recorded as `budget_exhausted`; no API error is raised

**Testability:** integration test with the fixture cache (replayed tokens count toward budget); small `--max-input-tokens` limit.

---

### Requirement: LLM Zero-Shot Baseline (revised)

`reference_v4/baselines/llm_zeroshot.py` SHALL provide `LLMZeroShot(model="claude-sonnet-4-6", replay_only: bool = True)`. Its `evaluate()` method SHALL:

1. Call `_build_prompt(claim)` to produce `(system_prompt, user_prompt)` using `SYSTEM_PROMPT` and `USER_PROMPT_TEMPLATE`.
2. Call `_truncate_evidence(claim.evidence_docs)` before inserting into the template.
3. Compute the cache key as `sha256(model + "\x00" + system_prompt + "\x00" + user_prompt + "\x00" + "0" + "\x00" + "256")` (temperature=0, max_tokens=256 are constants).
4. Call `LLMCache.get(key)`. On cache hit, use the cached `response_text`. On cache miss in replay mode, let `LLMCacheMissError` propagate. On cache miss in record mode, call the Anthropic API with the parameters from the SDK requirement above.
5. Parse `response_text` as JSON. Extract `verdict` and `confidence`. On any parse error (not JSON, missing field, `verdict` not in `{"SUPPORTS","REFUTES","NEI"}`, `confidence` outside [0,1]), log the claim ID and raw response at DEBUG level and treat as parse failure.
6. Call `verdict_to_mass(verdict, confidence)` and return the result.

The default `replay_only=True` ensures CI and accidental re-runs never make live API calls.

#### Scenario: Full evaluate() path on a SUPPORTS fixture
- **GIVEN** fixture cache entry: verdict=SUPPORTS, confidence=0.82
- **WHEN** `LLMZeroShot(replay_only=True).evaluate(fixture_claim)` is called
- **THEN** `MassFunction(supports=0.82, refutes=0, uncertain=0, theta=0.18)` is returned

#### Scenario: Malformed JSON response → theta=1.0
- **GIVEN** fixture cache entry: `response_text = "I think it supports the claim."`
- **WHEN** `evaluate()` is called
- **THEN** `MassFunction(supports=0, refutes=0, uncertain=0, theta=1.0)` is returned; no exception propagates

#### Scenario: NEI verdict ignores stated confidence
- **GIVEN** fixture cache entry: `{"verdict": "NEI", "confidence": 0.3}`
- **WHEN** `evaluate()` is called
- **THEN** `MassFunction(supports=0, refutes=0, uncertain=0, theta=1.0)` regardless of the 0.3

**Testability:** real cache replay from committed fixture file; real `verdict_to_mass`; no mocks.

---

### Requirement: Evaluation Harness

`reference_v4/experiments/benchmark_eval.py` SHALL drive the Cartesian product `(dataset × system × seed)` with:
- Checkpoint-resumable: a partial run writes per-cell JSON on completion; resumed runs skip completed cells.
- Per-cell timeout: default 300s; configurable via `--timeout`.
- LLM budget guard: configurable `--max-llm-calls`; stops LLM cells before exceeding budget.
- Output: one JSON per cell at `experiments/results/benchmark/<dataset>/<system>/<seed>.json`, plus `aggregate.json` per `(dataset × system)`.

#### Scenario: Resumability
- **GIVEN** a run that completed 3 of 9 cells then was killed
- **WHEN** the runner is invoked again with the same config
- **THEN** the 3 completed cells are skipped; only the 6 remaining cells run

**Testability:** integration test with a 3-cell matrix; kill after 2 cells; resume; assert no recomputation.

---

### Requirement: H1 Panel (HoVer, Depth-Stratified)

The H1 panel SHALL be a JSON table: one row per `(system, num_hops)` cell with `label_accuracy_mean`, `label_accuracy_ci_lo`, `label_accuracy_ci_hi` (95% bootstrap). It SHALL also include one row per system for the **flat SPLADE** variant of `CognitionSystem` (MCTS disabled; top-K retrieval only), enabling the MCTS-vs-flat comparison at each hop count.

The H1 effect size SHALL be defined as `(Cognition_GNN.acc_at_4hop − SymbolicFloor.acc_at_4hop)` with its bootstrap CI; reported regardless of sign.

#### Scenario: H1 panel schema
- **WHEN** `experiments/results/h1_panel.json` is loaded
- **THEN** it contains rows for every `(system, num_hops)` combination, each with `label_accuracy_mean` and CI bounds

**Testability:** schema assertion on the committed JSON.

---

### Requirement: H2 Panel (AVeriTeC, Abstention Quality)

The H2 panel SHALL contain:
1. θ distribution statistics per `(system, ground_truth_class)`: mean, std, 5th and 95th percentile of `m_theta` on NEI claims vs SUPPORTS/REFUTES claims.
2. ρ(`m_theta`, `nei_indicator`) for each system (Spearman correlation; expected negative: NEI → high θ).
3. False positive endorsement rate: fraction of NEI claims where the system returns SUPPORTS with `m_theta < 0.5`.

#### Scenario: H2 panel schema
- **WHEN** `experiments/results/h2_panel.json` is loaded
- **THEN** it contains per-system entries with `theta_on_nei_mean`, `spearman_rho`, and `fp_endorsement_rate`

#### Scenario: Cognition θ on NEI claims exceeds LLM θ
- **WHEN** H2 panel is evaluated against Cognition(GNN) and LLMZeroShot on the AVeriTeC dev set
- **THEN** Cognition's `theta_on_nei_mean > LLMZeroShot.theta_on_nei_mean` (Cognition is more honest about what it does not know)

**Testability:** schema assertion + ordering assertion (the ordering is the core H2 claim; if this test fails the hypothesis is falsified and the paper must report accordingly).

---

### Requirement: H3 Panel (Calibration, All Datasets)

The H3 panel SHALL contain ECE (15-bin), Brier score, and AURC for every `(dataset × system)` cell. It SHALL additionally include reliability diagrams as PNG figures (one per system, faceted by dataset).

The H3 effect size SHALL be defined as `(Cognition_GNN.ECE − NLIClassifier.ECE)` on SciFact test split; expected negative (lower ECE = better calibration). Report regardless of sign.

#### Scenario: H3 panel schema
- **WHEN** `experiments/results/h3_panel.json` is loaded
- **THEN** every `(dataset, system)` cell has `ece`, `brier`, `aurc` fields; all ECE values ∈ [0, 1]

**Testability:** schema assertion + range check.

---

### Requirement: Baseline Reproduction Gate

Before any `ours vs baseline` number is reported, each baseline must pass its reproduction gate:

| Baseline | Native benchmark | Tolerance |
|----------|-----------------|-----------|
| NLI classifier | SciFact dev label accuracy vs Wadden et al. (2020) | ±2 pp |
| LLM zero-shot | AVeriTeC dev vs any published LLM-zero-shot number on same split | ±3 pp |

A `reference_v4/experiments/check_baseline_reproductions.py` script SHALL produce this table. It SHALL exit non-zero if any baseline exceeds its tolerance. The phase-3 memo SHALL include the table at its top.

**Testability:** existence + exit-code check in CI.

---

### Requirement: Pareto-Front Figure

`reference_v4/experiments/figures/pareto.py` SHALL produce `pareto.pdf`: scatter of `label_accuracy_mean` (x) vs `aurc` (y) for all `(dataset × system)` cells; Pareto-optimal subset connected by a polyline; labeled. Regenerates deterministically from `aggregate.json`.

**Testability:** existence + determinism check (two invocations → identical output).

---

### Requirement: Phase-3 Technical Memo

`docs/publication/phase3_evidence_generation.md` SHALL contain:
1. Dataset provenance and statistics (claim count, label distribution, evidence length statistics).
2. Schema auto-generation coverage report (mean coverage %, fraction of claims flagged as < 80%).
3. Baseline reproduction-gate table.
4. Full results table per dataset (all systems, all metrics, CIs).
5. H1 panel with commentary.
6. H2 panel with commentary.
7. H3 panel with commentary.
8. Pareto-front figure with commentary.
9. Null / negative findings reported explicitly (if H1 or H3 effect is not significant, this is stated, not omitted).
10. Explicit claim envelope for Epic 04 (the exact sentences that can be claimed in the paper, based on this evidence).

**Testability:** existence + link from `reference_v4/README.md`.

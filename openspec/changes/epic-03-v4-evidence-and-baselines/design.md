# Design: Epic 03 (v4) — Evidence Generation + Baselines

## Context

The evaluation target is the Cognition system in `reference_v4/cognition/`. The evaluation harness must wrap `InfonStore` cleanly — none of the `HypergraphReasoner` or `experiments/` infrastructure from `reference_v2` is reused. The harness design must support offline-first operation (no live API calls during CI) and checkpoint-resumable runs (a full dataset × system × seed matrix takes hours).

Key constraints:
- Zero schema pre-authoring: the evaluation surface must be general-purpose, matching the system's claimed scope.
- Per-claim isolation: each claim gets its own InfonStore; stores are not shared across claims (no cross-claim contamination).
- Offline-first: LLM API calls are record-and-replayed from a committed cache; dataset downloads are cached locally.
- Resumable: a killed run resumes without recomputing completed cells.

## Decisions

### Decision: Per-claim InfonStore with auto-generated schema

Each claim is evaluated in a fresh `InfonStore` populated with only that claim's associated evidence documents. The schema is auto-generated per claim via SPLADE anchor co-activation spectral clustering (already implemented as `Analyst.set_schema` + extraction report loop in `reference_v4/cognition/src/cognition/cassette/analyst.py`).

**Why:** Dataset-wide shared schemas would require hand-authoring a per-dataset ontology (defeating the system's general-purpose claim) or a dataset-specific training process (defeating the zero-shot claim). Per-claim auto-schemas are the honest evaluation surface: the system does exactly what it advertises, on data it has never seen.

**Tradeoff:** Per-claim InfonStores are slower than a shared store (no cross-claim cassette reuse). On a dataset of 500 claims, this is ~500 ingest calls of ~0.5 s each (~4 min), which is acceptable.

**Alternatives considered:**
- *Dataset-wide single store*: fast, but would require a shared schema that encodes dataset-level entity structure — not zero-shot.
- *Pre-authored per-dataset schema*: accurate but requires expert annotation; contradicts the system's design goals.

---

### Decision: `EvalClaim` dataclass as the uniform claim representation

All three dataset loaders produce `EvalClaim` objects with a uniform interface. Dataset-specific fields (HoVer `num_hops`, AVeriTeC `question_answers`, SciFact `rationales`) are stored in a typed `metadata` dict.

**Why:** A uniform representation lets the evaluation harness iterate over any dataset without dataset-specific branching. Hypothesis panels (H1 stratified by `num_hops`, H2 by `nei_indicator`) read from the metadata dict.

```python
@dataclass
class EvalClaim:
    claim_id: str
    claim_text: str
    evidence_docs: list[str]          # text of associated evidence docs/passages
    ground_truth: Literal["SUPPORTS", "REFUTES", "NEI"]
    metadata: dict                     # dataset-specific: num_hops, rationales, etc.
```

---

### Decision: Uniform `EvalSystem` interface for all evaluated systems

```python
class EvalSystem(Protocol):
    name: str
    def evaluate(self, claim: EvalClaim) -> MassFunction: ...
```

`MassFunction` is `(m_s, m_r, m_u, m_theta)` — the four-element DS mass. Baselines that produce 3-way softmax set `m_u = m_theta = 0`. The LLM baseline maps verbalized confidence to a 4-element mass via a documented rule (see spec).

**Why:** A uniform interface means the evaluation harness has zero system-specific branching. Adding a fourth system later is a single class.

---

### Decision: CognitionSystem uses raw InfonStore API — Analyst is excluded from evaluation

`CognitionSystem.evaluate()` calls `store.ask()`, `store.connect()`, and `store.any_of()` directly. It does not use the `Analyst` class.

**Why:** `Analyst` wraps a Strands-powered LLM that routes natural-language questions to the correct store primitive. Using it in the evaluation loop would mean the "our system" column implicitly calls an LLM to parse the query, which (a) conflates the comparison with the LLM baseline, (b) introduces non-determinism unless the Analyst's LLM calls are also cached, and (c) makes the per-claim latency unpredictable. The `InfonStore` API is the correct benchmarking surface: it is deterministic, has no external dependencies, and is the layer being compared.

Schema bootstrap (the two-iteration `extraction_report()` + spectral clustering loop) is algorithmic and runs without an LLM call. The only LLM in the entire evaluation pipeline is the `LLMZeroShot` baseline.

---

### Decision: LLM call parameters — claude-sonnet-4-6, temperature=0, max_tokens=256

All LLM calls use `claude-sonnet-4-6` (the same model used in the system's Analyst; consistent across paper). `temperature=0` for determinism (the cache records real API responses; determinism means re-running with the same inputs never produces a different result). `max_tokens=256` is sufficient for the JSON response format with a brief verdict and confidence value.

**Why `max_tokens=256` and not more:** The prompt instructs the model to respond with a single JSON line. 256 tokens covers any plausible JSON response. Higher limits waste API budget and make the budget guard calculation noisier.

---

### Decision: LLM record-and-replay cache

LLM calls are cached by `sha256(model + system_prompt + user_prompt + temperature + max_tokens)` — all six components of the call signature. The cache is a JSONL file (`llm_cache.jsonl`) committed alongside results. CI always runs in `replay_only` mode; live calls only happen when an engineer explicitly runs in `record` mode.

**Why all six components in the key:** A change to any parameter (model upgrade, prompt refinement, temperature experiment) produces a cache miss, ensuring the committed cache always reflects exactly the calls that produced the stored results. A key that only hashes the prompt would silently return stale responses if model or parameters change.

**Why JSONL and not SQLite:** JSONL is human-readable (debuggable without tooling), append-only (no locking issues under concurrent writes), and trivially diff-able in git. The cache is O(N claims × datasets) — at most ~10,000 lines — so query speed is not a concern.

---

### Decision: Evidence truncation at 600 words/doc, 3,000 words total

These thresholds were chosen to fit within the effective reasoning window of `claude-sonnet-4-6` while keeping per-claim token cost predictable.

- **600 words/doc (~800 tokens):** Evidence passages in AVeriTeC are QA pairs (short); in HoVer they are Wikipedia sentences (short); in SciFact they are abstracts (typically 150–300 words). The 600-word cap only activates on verbose passages and rarely fires. When it does, the most informative content is almost always in the first 600 words of a scientific abstract or Wikipedia article.
- **3,000 words total (~4,000 tokens):** Leaves ~100,000 tokens of headroom in the 200k context window. At 3,000 words plus the system prompt (~200 words) and user prompt framing (~50 words), total input is ~3,250 words (~4,300 tokens) — predictable cost of roughly $0.013 per claim at claude-sonnet-4-6 pricing. For a 500-claim evaluation this is ~$6.50, well within the default $15 budget guard.

**Alternatives considered:**
- *No truncation:* Unpredictable cost; some AVeriTeC claims have very long evidence chains.
- *Character-based truncation:* Less interpretable than word-based; harder to reason about in the memo.

---

### Decision: NEI verdict → m_theta = 1.0 (not m_uncertain = 1.0)

When the LLM returns NEI, the resulting mass is `(0, 0, 0, 1.0)` — all mass on the vacuous element Θ (total ignorance), not on `m_uncertain`. This is the formally correct mapping in Dempster–Shafer theory and is the crux of H2.

**Why this matters for H2:** The H2 claim is that Cognition's DS mass produces `m_theta → 1.0` on unsupported claims, while the LLM baseline produces `m_theta → 0` (it halluccinates a confident answer). If the LLM baseline mapped NEI to `m_uncertain = 1.0`, it would look well-calibrated on the θ metric even though it is not using DS semantics. The correct mapping (NEI → `m_theta = 1.0`) means the LLM baseline will score well on H2 *only if the model actually returns NEI* with `confidence = 1.0` as instructed — which it often does not. This surfaces the real behavioural difference.

The system prompt explicitly instructs: "A verdict of 'NEI' MUST have confidence 1.0." Even with this instruction, models frequently return NEI with confidence 0.3 or 0.7. The spec's mapping ignores that stated confidence for NEI and always assigns `m_theta = 1.0`. This choice is logged as a design decision in the phase-3 memo so reviewers can evaluate it.

---

### Decision: LLM token budget guard tracks input tokens, not API calls

The budget guard uses cumulative `total_input_tokens` rather than number of API calls.

**Why:** Token cost, not call count, determines API spend. A claim with 3,000 words of evidence costs ~4× more than a claim with 750 words. Counting calls would undercount spend on evidence-heavy datasets (AVeriTeC) and overcount on thin ones (SciFact). Token tracking gives a direct cost proxy.

---

### Decision: Three baselines (not six)

The old Epic 03 spec had six baselines. Three are dropped:

| Dropped | Why |
|---------|-----|
| R-GCN architectural control | Designed to isolate `typed_ikl` vs `uniform_mean` inside the v2 GNN; meaningless without the v2 training loop |
| Dirichlet EDL | Meaningful only as a readout ablation on the same architecture; replaced by ECE comparison in H3 |
| Sufficient-Context / RAG selective generation | Requires dataset-specific retriever setup; adds ~3 days of engineering for one data point; the LLM zero-shot baseline already covers the upper accuracy bound |

**Why:** Every baseline must be justifiable by a hypothesis it helps test. Three baselines cover the three distinct comparisons: symbolic floor (H1 ablation), NLI (H3 ECE comparison), LLM (H2 false endorsement comparison).

---

### Decision: Baseline reproduction gate is per-system-on-native-dataset, not per-dataset

Each baseline must reproduce its own published number on the dataset it was natively evaluated on (RoBERTa-large FEVER → FEVER NLI accuracy; LLM → any published LLM-zero-shot on AVeriTeC dev). We do not require each baseline to reproduce on all three datasets before reporting — only on its own native benchmark.

**Why:** The reproduction gate is a sanity check, not a secondary evaluation. It confirms the baseline implementation is correct; it does not constrain which dataset we run it on for our results.

---

### Decision: Schema auto-generation uses two-iteration Analyst bootstrap

For each claim corpus, schema generation follows the Analyst's documented two-iteration loop:
1. Ingest evidence docs with an empty schema
2. Run `extraction_report()` — surfaces top-N SPLADE tokens by activation frequency
3. Cluster via spectral clustering on the token co-activation matrix
4. Produce `{actor, relation, feature, market}` typed anchors
5. Re-ingest with the generated schema
6. Run final `extraction_report()` — confirm coverage ≥ 80%

If the two-iteration loop does not converge (coverage < 80%), the claim is flagged in the phase-3 memo and reported with `m_theta = 1.0` (maximum uncertainty, honest abstention).

**Why:** The two-iteration loop is the documented usage pattern in `reference_v4/00_quick_start.ipynb` and `reference_v4/01_schema_design.ipynb`. Using it in evaluation matches the production use case.

---

### Decision: Metrics module reused from Epic 02, not rewritten

`reference_v2/experiments/metrics.py` (pure numpy, 6-metric battery) is copied verbatim to `reference_v4/experiments/metrics.py`. The `stats.py` (bootstrap CI) is likewise copied.

**Why:** Both modules are already tested (Epic 02 tests cover boundary conditions and coverage rates). Rewriting introduces risk; copying preserves the tested invariants. The modules have no v2-specific dependencies.

# BlackMagic — Specification v0.1

## Elevator pitch

**BlackMagic** is an English-only sparse-retrieval and reasoning library, forked
from the `cognition` package with multilingual complexity stripped out. Built on
splade-tiny, it takes a typed anchor schema and a document corpus and supports:

- **Retrieval** — sparse anchor-based search with persona-specific valence
- **Temporal reasoning** — NEXT edges, cross-document constraint aggregation
- **Belief fusion** — Dempster-Shafer mass functions, contrary-view inversion
- **Graph search** — MCTS traversal over the infon hypergraph
- **Imagination (new)** — genetic-algorithm proposal of counterfactual infons
  scored by grammar, logic, and health cost functions

The library preserves the `cognition` reasoning surface while narrowing the
input to English documents on splade-tiny. It is the English production-ready
sibling of `cognition`, not a trimmed-down primitive.

## Relationship to `cognition`

| File in `cognition` | In BlackMagic? | Changes |
|---|---|---|
| `encoder.py` | Yes | Strip multilingual code paths; WordPiece only; keep splade-tiny bundle |
| `extract.py` | Yes | Strip CJK sentence splitter and Hangul regex; ASCII-only span matching |
| `schema.py` | Yes | As-is; hierarchy + typed anchors unchanged |
| `infon.py` | Yes | Add `kind` and `parent_infon_ids` fields for imagined infons |
| `dempster_shafer.py` | Yes | As-is |
| `graph_mcts.py` | Yes | As-is, with fallback to heuristic evaluation when heads unavailable |
| `heads.py` | Yes | As-is; bundled heads.pt ships with splade-tiny dims |
| `query.py` | Yes | As-is; persona valence + contrary views preserved |
| `consolidate.py` | Yes | As-is; constraints aggregation preserved |
| `store/local.py` | Yes | Single-file `store.py`, add imagined-infon columns |
| `structural.py` | **No** | Not in v0.1 scope |
| `category.py` | **No** | Not in v0.1 scope |
| `agent_tools.py` | **No** | Not in v0.1 scope |
| `compute/` | **No** | Local only, no cloud |
| `store/cloud.py` | **No** | No DynamoDB backend |
| `cli.py` | Yes | Extended with `imagine` subcommand |
| — | **New** | `imagine.py` — GA imagination layer |

Stripped totals: 5 modules removed (`structural`, `category`, `agent_tools`,
`compute/*`, `store/cloud`). One module added (`imagine`). Net code delta
roughly -2000 lines kept, +300 lines new.

## Non-goals (v0.1)

- Multilingual input. Non-English input will still run but produces undefined
  activation patterns.
- Structural analysis engines (Kano, Feature Gap, Ghost Detection, Kan, etc.).
  These live in `cognition`.
- Category-theoretic extensions.
- Cloud deployment (DynamoDB, Lambda, S3).
- MCP / agent-tool wrappers.
- Training or fine-tuning the encoder / heads.

## Design principles

1. **Whitebox by default.** Every infon traces to a sentence and character
   spans; every anchor activation traces to vocab tokens; every imagined
   triple traces to the real triples that spawned it.
2. **Sparse by contract.** Anchors fire or don't. Binarization thresholds are
   explicit.
3. **No-training baseline.** Schema + corpus → working system. Fine-tuning is
   a `cognition` concern, not a BlackMagic concern.
4. **Reasoning is deliberate.** MCTS and GA imagination are explicit APIs —
   they don't silently run during `search()`. Users opt in.
5. **English assumption is load-bearing.** We remove the ~15% of cognition
   code that existed to handle multilingual edge cases. If you need
   multilingual, use `cognition`.

## API surface

```python
from blackmagic import BlackMagic, BlackMagicConfig

cfg = BlackMagicConfig(
    schema_path="schemas/automotive.json",
    db_path="bm.db",                  # or ":memory:"
    activation_threshold=0.3,
    min_confidence=0.05,
    top_k_per_role=3,
    consolidation_interval=100,       # run every N new infons
    persona="investor",               # default query persona
)

bm = BlackMagic(cfg)

# ── Ingest ────────────────────────────────────────────────────────────
n = bm.ingest([
    {"text": "Toyota invests $13.6B in batteries.", "id": "d1",
     "timestamp": "2026-04-20"},
])

# Optional explicit consolidation — builds NEXT edges + constraints
bm.consolidate()

# ── Retrieval (the primary use case) ──────────────────────────────────
result = bm.search(
    "Which automakers are investing in batteries?",
    top_k=10,
    include_chains=True,             # walk NEXT edges for temporal context
    persona="investor",              # override default
    contrary=False,                  # red-team mode toggle
)
for hit in result.hits:
    print(hit.score, hit.infon, hit.valence)

# ── Temporal / constraint queries ────────────────────────────────────
constraints = bm.store.get_constraints(subject="toyota", limit=20)
chains = bm.store.get_edges(edge_type="NEXT", anchor="toyota")

# ── Belief verification (Dempster-Shafer) ────────────────────────────
verdict = bm.verify_claim(
    "Toyota's battery investments will dominate the Asian EV market by 2030."
)
print(verdict.label, verdict.belief_supports, verdict.belief_refutes)

# ── Graph search (MCTS) ──────────────────────────────────────────────
mcts_result = bm.reason(
    "Does Toyota's battery strategy create cascading supply-chain risk?",
    max_iterations=8, max_depth=4,
)
for chain in mcts_result.chains_discovered:
    print(" → ".join(chain))

# ── Imagination (NEW) ────────────────────────────────────────────────
# Query-time GA proposal of counterfactual triples. Return type is
# isomorphic to MCTSResult so the same renderers (tree panels, iteration
# tables, verdict bars) work on both.
result = bm.imagine(
    "What partnerships might emerge between battery suppliers and Asian OEMs?",
    n_generations=10,
    population_size=50,
    persona="investor",
    cost_weights={"grammar": 1.0, "logic": 1.0, "health": 1.0},
)

# Imagination-native verdict ∈ {"PLAUSIBLE", "CONTRADICTED", "SPECULATIVE"}
print(result.verdict, result.combined_mass)

# MCTS-compatible verdict ∈ {"SUPPORTS", "REFUTES", "UNCERTAIN"} — same data,
# mapped onto the MCTS vocabulary so legacy renderers (e.g. customer demo
# flash_text) work without modification.
print(result.mcts_verdict)

# Traversal tree mirrors MCTS: root = query seed, children = GA offspring
for child in result.traversal_tree.children:
    print(child.anchor_path, child.fitness, child.belief_mass)

# Top imagined infons with provenance
for inf in result.imagined_infons[:10]:
    print(inf.subject, inf.predicate, inf.object,
          "fitness=", inf.fitness,
          "from=", inf.parent_infon_ids)

# Same chains_discovered shape as MCTSResult
for chain in result.chains_discovered:
    print(" → ".join(chain))

# ── Lifecycle ────────────────────────────────────────────────────────
bm.close()
```

### Core types (`blackmagic.infon`)

```python
@dataclass
class Infon:
    infon_id: str
    subject: str
    predicate: str
    object: str
    polarity: int                    # 1=affirmed, 0=negated
    confidence: float
    sentence: str
    doc_id: str
    sent_id: str
    spans: dict                      # {role: {text, start, end}}
    support: dict                    # {role: "direct" | "semantic" | "hierarchical"}
    subject_meta: dict
    predicate_meta: dict
    object_meta: dict
    locations: list
    timestamp: str | None
    precision: str
    temporal_refs: list
    tense: str
    aspect: str
    activation: float
    coherence: float
    specificity: float
    novelty: float
    importance: float
    reinforcement_count: int
    last_reinforced: str | None
    decay_rate: float
    # NEW for imagination:
    kind: str = "observed"           # "observed" | "imagined"
    parent_infon_ids: list[str] = []
    fitness: float | None = None     # populated for imagined only
```

`Edge` and `Constraint` unchanged from `cognition`.

### Search result (extends `cognition.QueryResult`)

```python
@dataclass
class SearchResult:
    query: str
    persona: str
    infons: list[Infon]
    constraints: list[Constraint]
    edges: list[Edge]                # NEXT edges walked
    valence: dict                    # {infon_id: float}
    timeline: list[Infon]            # chronologically ordered
    anchors_activated: dict          # {name: score}
    # NEW:
    hits: list[Hit]                  # scored, ranked retrieval view
```

### Imagination result (MCTS-shaped)

`ImaginationResult` is deliberately isomorphic to `MCTSResult` so that the
same renderers work for both — the customer demo's tree panels, iteration
tables, and verdict bars need minimal adaptation.

```python
@dataclass
class ImaginationNode:
    """One node in the imagination traversal tree — mirrors MCTSNode."""
    node_id: str                        # deterministic hash of path + generation
    anchor_path: list[str]              # anchors touched by this lineage
    infons: list[Infon]                 # GA population members at this node
                                        # (kind="imagined", fitness populated)
    parent: "ImaginationNode | None"
    children: list["ImaginationNode"]
    visit_count: int                    # generation depth
    belief_mass: MassFunction           # DS belief derived from logic_penalty
    fitness: float                      # GA fitness score for this node
    mutator_used: str | None            # name of mutator that produced this node
    is_terminal: bool                   # pruned, converged, or at max depth


@dataclass
class ImaginationResult:
    """Query-scoped GA imagination output, MCTS-shaped."""
    query: str
    seed_anchors: dict[str, float]      # activated anchors from the query

    # ── Dual verdicts ────────────────────────────────────────────────
    # Imagination-native label — describes the imagination outcome:
    #   "PLAUSIBLE"     : top candidates extend the KB without contradiction
    #   "CONTRADICTED"  : top candidates violate high-confidence constraints
    #   "SPECULATIVE"   : plausible extensions exist but weakly supported
    verdict: str
    # MCTS-compatible label — same data, mapped for renderer reuse:
    #   PLAUSIBLE    → SUPPORTS
    #   CONTRADICTED → REFUTES
    #   SPECULATIVE  → UNCERTAIN
    mcts_verdict: str

    combined_mass: MassFunction         # DS fusion over candidate logic penalties
    traversal_tree: ImaginationNode     # root: query seed; children: generations

    iteration_log: list[dict]           # per-generation: max/mean fitness,
                                        # population diversity, convergence signal
    chains_discovered: list[list[str]]  # imagined anchor chains (like MCTS)
    imagined_infons: list[Infon]        # top-K imagined infons (fitness-ranked)

    # ── Metrics matching MCTSResult ──────────────────────────────────
    nodes_explored: int                 # total candidates fitness-evaluated
    infons_evaluated: int               # total unique imagined infons
    generations: int                    # GA iterations run
    iterations: int                     # alias of `generations` for MCTS compat
    elapsed_s: float


# Verdict mapping (stable; callers can reuse directly)
IMAGINATION_TO_MCTS_VERDICT = {
    "PLAUSIBLE":    "SUPPORTS",
    "CONTRADICTED": "REFUTES",
    "SPECULATIVE":  "UNCERTAIN",
}
```

## Imagination layer (new)

### Goal

Given a query or seed, propose plausible-but-not-observed infons that extend
the knowledge graph. This is the "what might be true that we haven't seen?"
capability — genetic-algorithm search over typed triples, scored by a three-
term cost function.

### Module: `blackmagic.imagine`

```python
class Imagination:
    def __init__(self, store, schema, encoder, config):
        self.mutators = [
            HierarchyWalkMutator(schema),
            PredicateSubstitutionMutator(schema),
            PolarityFlipMutator(),
            RoleRecombinationMutator(),
            TemporalProjectionMutator(store),
        ]
        self.fitness = FitnessFunction(store, schema, encoder, config)

    def run(self, query: str,
            seed_infons: list[Infon],
            persona: str = None,
            n_generations: int = 10,
            population_size: int = 50,
            mutation_rate: float = 0.7,
            elitism: float = 0.2,
            top_k: int = 20) -> ImaginationResult:
        """Run the GA loop over candidate imagined triples.

        Returns an ImaginationResult whose shape mirrors MCTSResult —
        traversal_tree, combined_mass, chains_discovered, iteration_log,
        verdict + mcts_verdict (for renderer compatibility), and a
        fitness-ranked list of imagined_infons.

        The tree structure: root = a synthetic "query seed" node whose
        infons are the query-activated seed_infons. Each child represents
        one generation of GA offspring, with mutator_used recording which
        mutation operator produced it. Pruned / converged branches have
        is_terminal=True."""
```

### Mutation operators

Each is a pure function `Infon → Infon | None`; returns `None` when the
mutation is schema-invalid or has no valid variation.

1. **HierarchyWalkMutator** — replace subject or object with its parent,
   sibling, or child in the schema hierarchy. Bias toward siblings (same-level
   substitution is more informative than vertical moves).
2. **PredicateSubstitutionMutator** — substitute predicate with another
   relation-type anchor that shares at least one activation cluster (learned
   from corpus co-occurrence, not a pre-defined synonym list).
3. **PolarityFlipMutator** — invert the polarity bit. Produces the
   counterfactual negation of an observed triple.
4. **RoleRecombinationMutator** — pick two existing infons `A = (s₁, p₁, o₁)`
   and `B = (s₂, p₂, o₂)`; produce `(s₁, p₂, o₂)`, `(s₁, p₁, o₂)`, or
   `(s₂, p₁, o₁)`. The generative operator — truly novel triples.
5. **TemporalProjectionMutator** — take an infon observed at timestamp `t`,
   propose the same triple at `t + Δ` where Δ is drawn from the NEXT-edge
   gap distribution for that subject. Interacts with the temporal graph.

### Fitness function

```
fitness(triple) = grammar(triple)
                × max(0, 1 + logic_penalty(triple, kb))
                × health(triple, kb, persona)
```

- **`grammar(triple)`** — hard 0/1 filter:
  - `subject.type ∈ {actor}`
  - `predicate.type ∈ {relation}`
  - `object.type ∈ {location, feature, market}`
  Fails → fitness = 0, triple discarded.

- **`logic_penalty(triple, kb)`** — soft penalty ∈ `[-1, 0]`:
  - For each existing constraint `c` with same `(subject, predicate, object)`
    but opposite polarity: penalty −= `0.5 × c.strength × c.persistence`.
  - For each Dempster-Shafer mass function over the triple's support set
    where `belief(REFUTES) > 0.6`: penalty −= `0.3 × belief(REFUTES)`.
  - Clipped at −1 (never strictly zero — a contradicted triple can still
    score positively if its health is very high, as that's the signal of an
    interesting counterfactual).

- **`health(triple, kb, persona)`** — composite reward, product of three
  ∈ `[0, 1]` components:

  ```
  health = whisper × bridge × persona_align
  ```

  - **`whisper(triple)`** — at least one anchor in the triple must have
    non-trivial activation somewhere in the corpus. Concretely: each of
    `subject`, `predicate`, `object` must have at least one observed infon.
    Prevents pure hallucination of anchor combinations never seen anywhere.
    Computed as `min(freq(s), freq(p), freq(o))` normalized by corpus size.

  - **`bridge(triple, kb)`** — graph-topology reward. Compute the minimum
    number of existing infons needed to hop from `subject` to `object`
    through the current infon graph. Imagined triples that directly bridge
    disconnected subgraphs score high; triples that restate existing
    adjacencies score low. This is the "novel connection" signal.

  - **`persona_align(triple, persona)`** — if a persona is active, compute
    whether `predicate ∈ persona.positive_predicates` (score = 1.0),
    `negative_predicates` (0.3 — still useful for persona red-teaming),
    else 0.6 (neutral). Without a persona, always 1.0.

### GA loop

```
population = [seed_infons sampled + small_random_mutations]
for generation in range(n_generations):
    fitness_scores = [fitness(ind) for ind in population]
    elite = top(population, elitism_frac)            # carry best unchanged
    parents = tournament_select(population, fitness_scores)
    offspring = [random_mutator(p) for p in parents]
    population = elite ∪ offspring (dedup by triple key)
    if convergence(fitness_scores): break
return top_k(population, k=20)
```

Convergence: if max fitness change over last 3 generations < `ε`, stop.

### Verdict derivation

`ImaginationResult.verdict` is derived from the top-K imagined infons:

- **`PLAUSIBLE`**: ≥ 5 imagined infons with `fitness ≥ 0.6` and
  `logic_penalty ≥ -0.2` (no serious contradictions).
- **`CONTRADICTED`**: top imagined infon has `logic_penalty < -0.5`, OR
  the mean `logic_penalty` across top-10 is below `-0.4`. Interpretation:
  plausible-looking extensions keep violating high-confidence existing
  constraints — the KB's current shape is resisting the imagined direction.
- **`SPECULATIVE`**: neither condition holds. Imagination produced
  candidates but support is thin (low whisper scores) or fitness is
  marginal. This is the "interesting but unconfirmed" bucket.

`mcts_verdict` is the direct mapping via `IMAGINATION_TO_MCTS_VERDICT`.
The `combined_mass` MassFunction is built from the logic_penalty signals
of the top-K candidates, so it can be fed directly into the same DS
fusion path MCTSResult uses downstream.

### Imagined infon persistence

Imagined infons can be optionally stored in the `infons` table with
`kind="imagined"`. `search()` defaults to `include_imagined=False`. A
`cleanup_imagined()` method clears them. This lets callers treat
imagination as either ephemeral (one-shot, never stored) or cumulative
(build up a "what-if" corpus over time).

### Evaluation harness

`tests/test_imagine.py` includes a leave-out benchmark:
1. Ingest a corpus and run consolidation to establish the real KB.
2. Randomly hold out 10% of infons.
3. Run imagination seeded from the remaining 90%.
4. Measure how many held-out infons appear in the top-100 imagined triples.
5. Compare against a random-triple baseline (expected ~0% recall).

This is *not* a claim that imagination is "correct" — it's a claim that the
cost function has pull toward plausible patterns, not gibberish.

## Retrieval semantics (preserved from `cognition`)

- **Anchor expansion**: query activates anchors via SPLADE → AnchorProjector.
  Parent-level anchors expand to their descendants for retrieval.
- **Persona valence**: retrieved infons get a signed score based on the
  active persona's `positive_predicates` / `negative_predicates`.
- **Contrary views**: `contrary=True` inverts the DS frame — `SUPPORTS` and
  `REFUTES` swap. Same data, inverted lens.
- **NEXT-edge walking**: when `include_chains=True`, results include edges
  linking infons across time per shared anchor.
- **Constraint integration**: high-confidence constraints bubble up in
  results — cross-document aggregations of the same triple.

## Package layout

```
BlackMagic/
├── pyproject.toml
├── README.md
├── SPEC.md
├── CHANGELOG.md
├── src/
│   └── blackmagic/
│       ├── __init__.py           # exports main types
│       ├── config.py             # BlackMagicConfig
│       ├── schema.py             # AnchorSchema
│       ├── infon.py              # Infon, Edge, Constraint, Hit, SearchResult
│       ├── encoder.py            # SpladeEncoder, AnchorProjector, Encoder (WordPiece-only)
│       ├── extract.py            # extract_infons pipeline
│       ├── store.py              # LocalStore (single SQLite file)
│       ├── consolidate.py        # NEXT edges + constraints aggregation
│       ├── dempster_shafer.py    # MassFunction, verify_claim
│       ├── heads.py              # NLIHead, RelevanceHead, etc.
│       ├── graph_mcts.py         # GraphMCTS
│       ├── retrieve.py           # search() + persona valence
│       ├── imagine.py            # NEW: GA imagination
│       ├── cli.py                # blackmagic [ingest|search|imagine|stats]
│       └── model/                # bundled splade-tiny + heads.pt
├── tests/
│   ├── conftest.py
│   ├── test_schema.py
│   ├── test_encoder.py
│   ├── test_extract.py
│   ├── test_store.py
│   ├── test_consolidate.py
│   ├── test_dempster_shafer.py
│   ├── test_heads.py
│   ├── test_graph_mcts.py
│   ├── test_retrieve.py
│   ├── test_imagine.py
│   └── test_integration.py
└── examples/
    ├── automotive_schema.json
    └── quickstart.py
```

**Line budget** (not strict, but a sanity check against scope creep):

| Module | Target | Source |
|---|---|---|
| `encoder.py` | ≤ 220 | cognition/encoder.py stripped |
| `extract.py` | ≤ 280 | cognition/extract.py stripped |
| `schema.py` | ≤ 130 | as-is |
| `infon.py` | ≤ 180 | as-is + new fields |
| `store.py` | ≤ 400 | cognition/store/local.py + imagined cols |
| `consolidate.py` | ≤ 260 | as-is |
| `dempster_shafer.py` | ≤ 480 | as-is |
| `heads.py` | ≤ 420 | as-is |
| `graph_mcts.py` | ≤ 700 | as-is |
| `retrieve.py` | ≤ 340 | cognition/query.py adapted |
| `imagine.py` | ≤ 420 | NEW — includes ImaginationNode/ImaginationResult + tree construction |
| `cli.py` | ≤ 150 | extended |
| **Total src** | **≤ 3970** | cognition is ~5700 |

## Data model (SQLite)

Four tables — same as `cognition` plus a small addition:

```sql
CREATE TABLE infons (
    infon_id    TEXT PRIMARY KEY,
    subject     TEXT NOT NULL,
    predicate   TEXT NOT NULL,
    object      TEXT NOT NULL,
    polarity    INTEGER DEFAULT 1,
    direction   TEXT DEFAULT 'forward',
    confidence  REAL DEFAULT 0.0,
    sentence    TEXT,
    doc_id      TEXT,
    sent_id     TEXT,
    spans       TEXT,                -- json
    support     TEXT,                -- json
    subject_meta   TEXT,
    predicate_meta TEXT,
    object_meta    TEXT,
    locations   TEXT,                -- json
    timestamp   TEXT,
    precision   TEXT,
    temporal_refs TEXT,              -- json
    tense       TEXT,
    aspect      TEXT,
    activation  REAL,
    coherence   REAL,
    specificity REAL,
    novelty     REAL,
    importance  REAL,
    reinforcement_count INTEGER,
    last_reinforced TEXT,
    decay_rate  REAL,
    kind        TEXT DEFAULT 'observed',    -- NEW: 'observed' | 'imagined'
    parent_infon_ids TEXT,                  -- NEW: json list for imagined
    fitness     REAL,                       -- NEW: populated for imagined only
    created_at  REAL DEFAULT (unixepoch())
);
CREATE INDEX ix_infons_kind ON infons(kind);

CREATE TABLE edges (
    source      TEXT NOT NULL,
    target      TEXT NOT NULL,
    edge_type   TEXT NOT NULL,
    weight      REAL DEFAULT 1.0,
    metadata    TEXT
);
CREATE INDEX ix_edges_type ON edges(edge_type);

CREATE TABLE constraints (
    subject     TEXT, predicate   TEXT, object      TEXT,
    evidence    INTEGER, doc_count INTEGER,
    strength    REAL, persistence INTEGER, score REAL,
    infon_ids   TEXT
);

CREATE TABLE documents (
    doc_id    TEXT PRIMARY KEY,
    text      TEXT NOT NULL,
    timestamp TEXT,
    n_infons  INTEGER DEFAULT 0,
    ingested_at REAL DEFAULT (unixepoch())
);
```

## Test plan

Test coverage must preserve cognition-level confidence. Every module gets a
test file. Key tests:

**test_imagine.py** (new):
- Grammar filter rejects `(actor, actor, location)` and similar malformed triples.
- Logic penalty reduces fitness for `polarity=1` triple when a
  `polarity=0` constraint exists with high strength.
- Health(whisper) = 0 when any role anchor has zero corpus occurrences.
- Bridge score is monotone — imagined triples connecting disconnected
  subgraphs outscore those restating existing adjacencies.
- Leave-out recall: imagination rediscovers ≥ 5% of held-out infons in
  top-100 (vs < 0.5% for random-triple baseline).
- Reproducibility: same seed → same output.
- Persona alignment bias works as specified.

Other modules follow `cognition`'s existing test structure with minor
adaptations for removed features (no multilingual test cases).

## CLI

```
blackmagic ingest   <schema.json> <docs.jsonl> [--db PATH] [--consolidate]
blackmagic search   <query> [--schema PATH] [--db PATH] [--top-k N]
                    [--persona NAME] [--contrary] [--include-chains]
blackmagic imagine  <query> [--schema PATH] [--db PATH] [--generations N]
                    [--population N] [--persona NAME] [--store-imagined]
blackmagic verify   <claim> [--schema PATH] [--db PATH]
blackmagic reason   <query> [--schema PATH] [--db PATH] [--iterations N]
blackmagic stats    [--db PATH]
```

No `bm` short alias in v0.1 — ambiguous with `bm` in other contexts (Bash
Multiplexer, etc.). Add in v0.2 if users ask.

## Dependencies

```toml
[project]
name = "blackmagic"
version = "0.1.0"
requires-python = ">=3.11"
license = "Apache-2.0"
dependencies = [
    "torch>=2.0",
    "transformers>=4.40",
    "numpy>=1.24",
]

[project.optional-dependencies]
dev = ["pytest>=7", "pytest-cov"]

[project.scripts]
blackmagic = "blackmagic.cli:main"

[tool.setuptools.package-data]
blackmagic = ["model/*"]
```

Wheel size target: ≤ 25MB (splade-tiny 17MB + heads.pt ~1MB + source).

## Defaults I picked where you didn't specify

Flagged explicitly so you can override:

1. **Imagination runs query-scoped, not batch** (user confirmed). Seeded
   from query-activated infons.
2. **Imagination output is MCTS-shaped with dual verdicts** (user confirmed
   Option 3). `verdict` carries imagination-native labels
   (PLAUSIBLE/CONTRADICTED/SPECULATIVE); `mcts_verdict` carries the mapped
   MCTS labels (SUPPORTS/REFUTES/UNCERTAIN) for renderer compatibility.
3. **Imagined infons stored by default** when persisted. `search()`
   defaults to `include_imagined=False`.
4. **Fitness weights start equal** (`grammar=1, logic=1, health=1`).
   Tunable via `cost_weights=` argument.
5. **Leave-out held-out test is the primary imagination eval.** Not a
   correctness claim; a signal-vs-noise claim.
6. **GA population size 50, 10 generations** as defaults. Tunable.
7. **No `bm` CLI alias in v0.1.** Full name only.
8. **`src/` layout.** Standard Python best practice.
9. **`BlackMagic` is the final package name** unless you say otherwise.
10. **No built-in `demo` CLI command in v0.1.** Add later if needed; for now,
    `examples/quickstart.py`.

## Build order

1. `pyproject.toml`, `__init__.py`, `config.py` — package bootstrap
2. `infon.py`, `schema.py` — data types (no external deps)
3. `encoder.py` — port from cognition with multilingual code removed
4. `store.py` — merge `store/{__init__.py, local.py}`, add imagined columns
5. `extract.py` — port from cognition, strip CJK
6. `consolidate.py` — port as-is
7. `heads.py`, `dempster_shafer.py` — port as-is
8. `graph_mcts.py` — port as-is
9. `retrieve.py` — port `query.py` preserving persona valence + contrary
10. `imagine.py` — **new** module, built bottom-up from mutators → fitness → GA
11. `cli.py` — extend with `imagine` + `reason` + `verify`
12. Tests per module as each is written
13. `examples/` + `README.md`
14. `pip install -e .` from fresh venv, run quickstart end-to-end

Estimated effort: 2-3 days for the port + new imagine module + tests.

## Open questions still needing your call

1. **Persona in imagination — filter or weight?** Current spec: soft weight
   (no hard rejection of mismatched predicates). If you want "investor
   persona never imagines `recall` or `cancel` predicates," switch to filter.
2. **Cost-function decomposition** — grammar × logic × health matches the
   prior description, but do you want different term weights as defaults?
3. **BlackMagic as final name** — confirm or propose alternative.

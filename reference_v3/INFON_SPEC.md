# Infon Specification

## 1. What is an Infon?

An infon is a grounded unit of information extracted from text. It captures **who did what to whom, where, when, how confidently, and how it connects to other infons**. It is the atomic thought of the cognition system.

In situation semantics notation: `<<predicate, subject, object; polarity>>`

An infon is always grounded in a source sentence with extractive spans, always typed by an anchor schema with hierarchy metadata, and always scored for importance. Infons link to each other through temporal sequences and semantic entailment, forming a queryable knowledge graph.

## 2. Core Triple

Every infon has exactly three anchors filling three roles:

```
subject ──── predicate ──── object
"toyota"     "invest"       "solid_state"
```

| Field | Type | Description |
|-------|------|-------------|
| `subject` | `str` | Anchor name filling the subject role (actor) |
| `predicate` | `str` | Anchor name filling the predicate role (relation) |
| `object` | `str` | Anchor name filling the object role (feature/market/variant) |
| `polarity` | `int` | `1` = affirmed, `0` = negated |
| `direction` | `str` | `forward` (S acts on O), `reverse` (O acts on S), `neutral` |
| `confidence` | `float` | Geometric mean of role probabilities, range [0, 1] |

### Polarity

Polarity captures negation. "Toyota invests in batteries" has polarity 1. "Toyota has not invested in batteries" has polarity 0. The triple stays the same; only the truth value flips.

### Direction

Direction captures who is acting on whom. Most triples are forward (subject acts on object). "Batteries are invested in by Toyota" is reverse. "Toyota and Honda compete" is neutral (symmetric).

## 3. Grounding

Every infon is grounded in a specific sentence from a specific document, with character-level spans showing exactly where each role was found.

| Field | Type | Description |
|-------|------|-------------|
| `sentence` | `str` | The source sentence text |
| `doc_id` | `str` | Document identifier |
| `sent_id` | `str` | Sentence identifier (`{doc_id}_{sent_idx:04d}`) |
| `spans` | `dict` | Character spans per role (see below) |
| `support` | `dict` | How each role was grounded (see below) |

### Spans

```python
{
    "subject":   {"text": "Toyota",          "start": 0,  "end": 6},
    "predicate": {"text": "investing",       "start": 10, "end": 19},
    "object":    {"text": "solid-state battery tech", "start": 23, "end": 48},
}
```

A span may be `None` if the role was inferred semantically rather than found in the text.

### Support Types

Each role's grounding is classified:

| Support | Meaning |
|---------|---------|
| `direct` | At least one anchor token appears verbatim in the sentence |
| `semantic` | The model activated this anchor but no token appears in text |
| `hierarchical` | Activated via a child/parent anchor (e.g., "Camry" activates `toyota`) |

```python
{
    "subject":   "direct",
    "predicate": "direct",
    "object":    "semantic",
}
```

## 4. Hierarchy Metadata

Each anchor in the schema carries typed hierarchy fields. The infon inherits these from its three roles. This enables multi-level queries without changing the infon itself.

### Subject Metadata (Actor)

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `type` | `str` | `"actor"` | Anchor type |
| `organisation_type` | `str` | `"private-sector"` | From ORGANISATION_TYPES enum |
| `canonical_name` | `str` | `"Toyota Motor Corporation"` | Full legal/display name |
| `country_code` | `str` | `"JP"` | ISO 3166-1 alpha-2 |
| `parent` | `str` | `"japan_auto"` | Parent anchor in hierarchy |

### Predicate Metadata (Relation)

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `type` | `str` | `"relation"` | Anchor type |
| `domain` | `str` | `"economic"` | Semantic domain |
| `level` | `str` | `"action"` | From RELATION_LEVELS enum |
| `parent` | `str` | `"capital_allocation"` | Parent anchor |

### Object Metadata (Feature / Market / Variant)

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `type` | `str` | `"feature"` | Anchor type |
| `level` | `str` | `"product"` | From FEATURE_LEVELS or LOCATION_LEVELS enum |
| `macro_region` | `str` | `"East Asia"` | For market/location types |
| `country_code` | `str` | `"JP"` | For market/location types |
| `parent` | `str` | `"batteries"` | Parent anchor |

### Hierarchy Traversal

The anchor schema supports `get_ancestors(name)` and `get_descendants(name)`. When querying for "japanese automakers invest in X", the system can walk up from `toyota` to `japan_auto` and find all siblings, or walk down from `batteries` to `solid_state`, `lithium_ion`, etc.

## 5. Spatial Context

Where in the world does this infon apply? Locations may be extracted from the text or inferred from anchor metadata.

| Field | Type | Description |
|-------|------|-------------|
| `locations` | `list[dict]` | Detected/inferred locations (see below) |

Each location entry:

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `name` | `str` | `"japan"` | Anchor name or free-text location |
| `level` | `str` | `"country"` | From LOCATION_LEVELS enum |
| `macro_region` | `str` | `"East Asia"` | From MACRO_REGIONS enum |
| `country_code` | `str` | `"JP"` | ISO code |
| `confidence` | `float` | `0.65` | How sure we are this location applies |
| `source` | `str` | `"inferred"` | `extracted` (from text), `inferred` (from anchor metadata), `propagated` (from document metadata) |

### Source Types

- **extracted**: a location token was found in the sentence text ("sold in the US")
- **inferred**: location derived from an anchor's `country_code` or `macro_region` (Toyota -> JP -> East Asia)
- **propagated**: location inherited from document-level metadata

## 6. Temporal Context

When does this infon apply? Temporal information comes from the document timestamp, from temporal expressions in the text, and from linguistic tense/aspect.

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | `str` | Document/event timestamp (ISO 8601) |
| `precision` | `str` | From TEMPORAL_PRECISIONS: `exact`, `day`, `month`, `quarter`, `year`, etc. |
| `temporal_refs` | `list[dict]` | Temporal expressions extracted from text |
| `tense` | `str` | Linguistic tense of the predicate |
| `aspect` | `str` | Linguistic aspect of the predicate |

Each temporal reference:

| Field | Type | Example |
|-------|------|---------|
| `text` | `str` | `"Q1 2026"` |
| `precision` | `str` | `"quarter"` |
| `date_iso` | `str` | `"2026-Q1"` |
| `confidence` | `float` | `0.8` |

### Tense Values

`past`, `present`, `present_continuous`, `future`, `conditional`, `unknown`

### Aspect Values

`punctual` (single event), `durative` (ongoing), `habitual` (repeated), `stative` (state), `unknown`

## 7. Importance

Importance is a composite score that determines whether an infon survives pruning and how it ranks in query results. It maps to the Atomic Architecture's "competition for attention" mechanism.

| Field | Type | Description |
|-------|------|-------------|
| `activation` | `float` | Model confidence (geometric mean of role probs) |
| `coherence` | `float` | Sheaf NPMI consistency of the (S, P, O) combination |
| `specificity` | `float` | Inverse document frequency of the anchors (rare = specific) |
| `novelty` | `float` | How different from existing infons (1.0 = first time seen) |
| `importance` | `float` | Composite score (weighted combination) |
| `reinforcement_count` | `int` | How many times this infon has been re-observed (consolidation) |
| `last_reinforced` | `str` | Timestamp of last reinforcement |
| `decay_rate` | `float` | Per-day importance decay (default 0.01) |

### Lifecycle

1. **Creation**: importance = f(activation, coherence, specificity, novelty)
2. **Consolidation**: each new observation of the same (S, P, O) increments `reinforcement_count` and boosts importance
3. **Decay**: importance decreases by `decay_rate` per day since `last_reinforced`
4. **Pruning**: infons below a threshold are soft-deleted (retained but excluded from queries)

### Composite Formula

```
importance = w_act * activation
           + w_coh * coherence
           + w_spec * specificity
           + w_nov * novelty
           + w_reinf * log(1 + reinforcement_count)
           - decay_rate * days_since_reinforced
```

Default weights: `w_act=0.3, w_coh=0.25, w_spec=0.2, w_nov=0.15, w_reinf=0.1`

## 8. Edges

Infons connect to anchors and to each other through typed edges.

### Spoke Edges (structural)

These connect an infon to its constituent anchors. Created at extraction time.

| Edge Type | From | To | Description |
|-----------|------|----|-------------|
| `INITIATES` | subject anchor | infon | Actor initiates this infon |
| `ASSERTS` | infon | predicate anchor | Infon asserts this relation |
| `TARGETS` | infon | object anchor | Infon targets this entity |
| `LOCATED_AT` | infon | location anchor | Infon is spatially located here |
| `OCCURRED_ON` | infon | temporal anchor | Infon is temporally located here |

### Sequence Edges (experience chains)

These link infons temporally through shared anchors. Created at consolidation time.

| Edge Type | From | To | Description |
|-----------|------|----|-------------|
| `NEXT` | infon_A | infon_B | Same anchor, B is temporally after A |

Metadata on NEXT edges:

| Field | Type | Description |
|-------|------|-------------|
| `anchor` | `str` | The shared anchor that links these infons |
| `anchor_role` | `str` | Role the anchor plays: `subject`, `predicate`, `object` |
| `gap_days` | `int` | Calendar days between the two infons |

A single infon can have multiple NEXT edges — one per shared anchor. "Toyota invests in batteries" links to the next toyota-as-subject infon AND the next batteries-as-object infon. These are separate chains.

### Semantic Edges (inferred)

These capture logical relationships between infons. Created at consolidation time or query time.

| Edge Type | From | To | Description |
|-----------|------|----|-------------|
| `ENTAILS` | infon_A | infon_B | A logically implies B |
| `CONTRADICTS` | infon_A | infon_B | A and B cannot both be true |
| `SUPPORTS` | infon | constraint | Infon is evidence for this constraint |
| `SIMILAR` | infon_A | infon_B | Same situation, different wording |

### Constraint Edge

When multiple infons assert the same (S, P, O), they collectively support a **constraint** — an aggregated claim with evidence strength.

| Field | Type | Description |
|-------|------|-------------|
| `subject` | `str` | Constraint subject |
| `predicate` | `str` | Constraint predicate |
| `object` | `str` | Constraint object |
| `evidence` | `int` | Number of supporting infons |
| `doc_count` | `int` | Number of distinct documents |
| `strength` | `float` | Mean confidence across evidence |
| `persistence` | `int` | Number of distinct time windows |
| `constraint_score` | `float` | Composite score |

## 9. Query-Time Evaluation

These fields are NOT stored on the infon. They are computed at query time based on (infon, persona, goal).

| Field | Type | Description |
|-------|------|-------------|
| `valence` | `float` | +1.0 (positive for persona) to -1.0 (negative) |
| `salience` | `float` | How relevant to the query goal |
| `urgency` | `float` | Does this demand immediate attention |
| `temporal_relevance` | `float` | Recency weighted by query temporal bias |

These are returned in query results but never persisted.

## 10. Serialization

### In-Memory (Python dataclass)

```python
@dataclass
class Infon:
    # Identity
    infon_id: str

    # Core triple
    subject: str
    predicate: str
    object: str
    polarity: int = 1
    direction: str = "forward"
    confidence: float = 0.0

    # Grounding
    sentence: str = ""
    doc_id: str = ""
    sent_id: str = ""
    spans: dict = field(default_factory=dict)
    support: dict = field(default_factory=dict)

    # Hierarchy (looked up from schema, stored for query efficiency)
    subject_meta: dict = field(default_factory=dict)
    predicate_meta: dict = field(default_factory=dict)
    object_meta: dict = field(default_factory=dict)

    # Spatial
    locations: list = field(default_factory=list)

    # Temporal
    timestamp: str | None = None
    precision: str = "unknown"
    temporal_refs: list = field(default_factory=list)
    tense: str = "unknown"
    aspect: str = "unknown"

    # Importance
    activation: float = 0.0
    coherence: float = 0.0
    specificity: float = 0.0
    novelty: float = 1.0
    importance: float = 0.0
    reinforcement_count: int = 0
    last_reinforced: str | None = None
    decay_rate: float = 0.01
```

### SQLite (local store)

```sql
CREATE TABLE infons (
    infon_id    TEXT PRIMARY KEY,

    -- Core triple
    subject     TEXT NOT NULL,
    predicate   TEXT NOT NULL,
    object      TEXT NOT NULL,
    polarity    INTEGER DEFAULT 1,
    direction   TEXT DEFAULT 'forward',
    confidence  REAL DEFAULT 0.0,

    -- Grounding
    sentence    TEXT,
    doc_id      TEXT,
    sent_id     TEXT,
    spans       TEXT,       -- JSON
    support     TEXT,       -- JSON

    -- Hierarchy
    subject_meta  TEXT,     -- JSON
    predicate_meta TEXT,    -- JSON
    object_meta   TEXT,     -- JSON

    -- Spatial
    locations   TEXT,       -- JSON array

    -- Temporal
    timestamp   TEXT,
    precision   TEXT DEFAULT 'unknown',
    temporal_refs TEXT,     -- JSON array
    tense       TEXT DEFAULT 'unknown',
    aspect      TEXT DEFAULT 'unknown',

    -- Importance
    activation  REAL DEFAULT 0.0,
    coherence   REAL DEFAULT 0.0,
    specificity REAL DEFAULT 0.0,
    novelty     REAL DEFAULT 1.0,
    importance  REAL DEFAULT 0.0,
    reinforcement_count INTEGER DEFAULT 0,
    last_reinforced TEXT,
    decay_rate  REAL DEFAULT 0.01
);

-- Query indexes
CREATE INDEX idx_infon_subject ON infons(subject);
CREATE INDEX idx_infon_predicate ON infons(predicate);
CREATE INDEX idx_infon_object ON infons(object);
CREATE INDEX idx_infon_doc ON infons(doc_id);
CREATE INDEX idx_infon_timestamp ON infons(timestamp);
CREATE INDEX idx_infon_importance ON infons(importance);
CREATE INDEX idx_infon_polarity ON infons(polarity);
```

### DynamoDB (cloud store)

Single-table design with adjacency list pattern:

| PK | SK | Attributes |
|----|------|------------|
| `INFON#{infon_id}` | `META` | All infon fields |
| `ANCHOR#{name}` | `INFON#{infon_id}` | role, confidence |
| `INFON#{id_a}` | `NEXT#{id_b}` | anchor, gap_days |
| `INFON#{id_a}` | `ENTAILS#{id_b}` | confidence |
| `CONSTRAINT#{s}#{p}#{o}` | `META` | evidence, strength, persistence |
| `CONSTRAINT#{s}#{p}#{o}` | `INFON#{infon_id}` | confidence |
| `DOC#{doc_id}` | `INFON#{infon_id}` | sent_id |

GSI-1: `SK` as partition key (for reverse lookups: "all infons for anchor X")
GSI-2: `subject` + `timestamp` (for temporal queries on a subject)

## 11. Category-Theoretic Extensions

The category module (`cognition.category`) adds three constructions from category theory that operate on the infon graph.

### Sheaf Coherence

A presheaf on the anchor co-activation graph replaces the placeholder `coherence=0.0` with a real topological signal. Construction:

1. **Observe**: build a sentence × anchor binary co-occurrence matrix from SPLADE activations
2. **Fit**: compute the NPMI (normalized pointwise mutual information) matrix between all anchor pairs, then the graph Laplacian
3. **Score**: for each infon `<<S, P, O>>`, compute mean pairwise NPMI among its three anchors, mapped to [0, 1]
4. **Diagnostics**: Fiedler value (second-smallest Laplacian eigenvalue) measures algebraic connectivity; per-anchor degree centrality identifies hubs; connected components reveal independent sub-categories

High coherence = the triple's anchors genuinely co-occur across the corpus. Low coherence = anchors activated by coincidence, not structural co-occurrence.

| Diagnostic | Meaning |
|-----------|---------|
| `fiedler_value` | Algebraic connectivity — 0 means disconnected components, high means single global section |
| `anchor_centrality` | Degree centrality on positive NPMI graph — hubs are bridge concepts |
| `component_structure` | Connected components — each is a "natural category" of co-occurring anchors |

### Functorial Data Migration

A functor `F: Schema_old → Schema_new` pushes forward all infons, edges, and constraints without re-ingesting documents.

**SchemaFunctor** defines the mapping:
- `rename`: `{old_name: new_name}` — 1:1 (non-injective renames can merge triples)
- `merge`: `{old_name1: target, old_name2: target}` — many:1 (reinforces merged infons)
- `delete`: `{name, ...}` — forget (infons containing deleted anchors are removed)

**FunctorialMigration** applies the functor:
- `migrate_infon(infon)` → remapped infon or `None` (deleted)
- `migrate_all(infons, edges)` → deduplicated migrated infons + edges
- `report()` → statistics on merges, deletions, reinforcements

**Composition preservation**: for functors F1 and F2, `migrate(F2 ∘ F1)` produces the same result as `migrate(F2, migrate(F1, data))`.

### Left Kan Extension — Schema-Free Discovery

Given raw text with no predefined schema, discover the "natural" anchor categories from SPLADE activation patterns.

**Algorithm (SchemaDiscovery.discover)**:
1. Encode corpus through SPLADE (no schema needed)
2. Filter to frequently-activated vocab tokens (document frequency ≥ threshold)
3. Build vocab × vocab co-activation matrix, normalize to PMI
4. Spectral clustering on the co-activation graph (normalized Laplacian → eigenvectors → k-means)
5. Label clusters by top-activated tokens → `DiscoveredAnchor` objects
6. Infer types from token semantics (verbs → relation, country names → market, etc.)
7. Return `AnchorSchema` ready for use with the standard pipeline

The discovered schema can be used directly with `Encoder` and `extract_infons`, or refined through functorial migration to align with an existing schema.

### Category-Theoretic Interpretation

| Category Theory | Cognition System |
|----------------|-----------------|
| Objects | Anchors (typed vocabulary entries) |
| Morphisms | Infons (grounded triples connecting anchors) |
| Composition | NEXT chains (temporal sequencing) |
| Functor | Schema migration (structure-preserving map) |
| Presheaf | Sheaf coherence (local-to-global consistency) |
| Left Kan extension | Schema discovery (free construction from data) |
| Natural transformation | Importance decay (systematic modification of all morphisms) |
| Colimit | Constraints (aggregation of co-supporting infons) |
| Pullback | Query results (intersection of anchor-filtered subgraphs) |

## 12. Mapping to Atomic Architecture

| Atomic Architecture | Infon System | Implementation |
|---------------------|-------------|----------------|
| Atomic Thought | Infon | `Infon` dataclass, one per (sentence, S, P, O) |
| Source -> Type -> Target | Spoke edges | `INITIATES`, `ASSERTS`, `TARGETS` |
| Universal Knowledge Base | Store backends | SQLite (local) or DynamoDB (cloud) |
| Recognition Engine | ModernBERT encoder | Sentence -> anchor activations -> infon |
| Experience Sequences | NEXT edges | Temporal chains per shared anchor |
| Prediction System | Query + sequence walk | Follow NEXT edges to predict future states |
| Well-Being Score | Query-time valence | `(infon, persona, goal) -> valence` |
| Importance Competition | Importance scoring | Composite of activation, coherence, specificity, novelty |
| Consolidation | Reinforcement | `reinforcement_count++`, importance boost |
| Pruning | Importance decay | Below threshold = soft delete |
| Spatial Model | Locations + hierarchy | Egocentric (relative to persona) at query time |

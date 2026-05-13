# Module 11: Category-Theoretic Extensions


> **Status update.** `SchemaFunctor` + Kan-based migration *shipped* in `cognition.cassette.migrate` — see module 15 for the user-facing walkthrough and `store.migrate(functor, schema_path)`. Sheaf coherence remains theory-only at the scoring layer, but H¹ discrepancy is a live feature in the trained sheaf GNN (see notebook 12). Schema discovery via Kan extension is not shipped; the Strands `Analyst.bootstrap_gnn` and agent-driven ontology proposal cover the same user need.

## What You'll Learn

- How sheaf coherence replaces placeholder `coherence=0.0` with a real topological signal
- How to evolve schemas without re-ingesting documents using functorial migration
- How to discover anchor categories from raw text with no predefined schema
- The category-theoretic interpretation of the entire cognition system

## Background

Category theory gives us three tools that address real problems in the cognition pipeline:

1. **Sheaf coherence** — the importance formula has `coherence` as a weight, but extraction sets it to 0.0. A presheaf on the anchor co-activation graph fills this gap with a measure of local-to-global consistency: do these three anchors genuinely co-occur across the corpus, or was this a one-off coincidence?

2. **Functorial migration** — when you need to rename "us" to "united_states", merge "condemn" and "sanction" into "coerce", or delete an anchor entirely, a functor pushes forward all infons without re-ingesting documents. Composition is preserved: F2 after F1 equals the composite.

3. **Left Kan extension** — what if you don't have a schema? Given SPLADE activations on raw text, spectral clustering on the co-activation matrix discovers the "natural" anchor categories. This is the free construction: the best schema that explains the data.

## Prompt 1: Sheaf Coherence

---

```
Build sheaf coherence scoring in cognition/src/cognition/category.py.

The SheafCoherence class should:

1. observe(activation_matrix, threshold=0.3)
   - Take an (n_sentences, n_anchors) matrix from encoder.encode()
   - Build binary co-occurrence (above threshold)
   - Accumulate pair counts and per-anchor counts

2. fit()
   - Compute NPMI matrix from co-occurrence counts
   - Build adjacency from positive NPMI values
   - Compute graph Laplacian L = D - A
   - Extract Fiedler value (second-smallest eigenvalue)

3. score_infon(infon) → float in [0, 1]
   - Average pairwise NPMI among (subject, predicate, object)
   - Map from [-1, 1] to [0, 1]

4. Diagnostics:
   - global_connectivity() → Fiedler value
   - anchor_centrality() → dict of degree centrality scores
   - component_structure() → list of connected components

Test on a geopolitical corpus with 30 anchors. Verify:
- NPMI matrix is symmetric with zeros on diagonal
- Fiedler value > 0 (connected graph)
- Coherence scores are in [0, 1] with reasonable distribution
- Top coherent triples make intuitive sense
```

---

### What the Sheaf Does

Think of the anchor graph as a topological space. Each sentence is a local observation — it activates a subset of anchors. The NPMI matrix tells you which pairs of anchors co-activate reliably (positive NPMI) vs. coincidentally or never (zero/negative NPMI).

A sheaf assigns to each open set (subset of anchors) a vector space of consistent activations. The coherence score measures whether a triple's three anchors form a "consistent section" — whether the local observation (this sentence's activations) is compatible with the global pattern (corpus-wide co-activation).

The Fiedler value tells you about global structure:
- **Fiedler = 0**: the graph is disconnected — there are independent sub-categories that never co-occur
- **Fiedler > 0**: the graph is connected — all anchors participate in a single coherent structure
- **High Fiedler**: strongly connected — the schema is tight and well-integrated

## Prompt 2: Functorial Data Migration

---

```
Build functorial schema migration in cognition/src/cognition/category.py.

SchemaFunctor dataclass:
  - rename: dict[str, str]  (old → new, 1:1)
  - merge: dict[str, str]   (old → merged_target, many:1)
  - delete: set[str]        (remove anchor entirely)
  - map_anchor(name) → new_name or None
  - map_triple(s, p, o) → (new_s, new_p, new_o) or None

FunctorialMigration class:
  - __init__(functor, source_schema, target_schema)
  - migrate_infon(infon) → Infon or None
  - migrate_all(infons, edges) → (migrated_infons, migrated_edges)
  - report(original, migrated) → statistics dict

Key behaviors:
- Non-injective renames (two old names → same new name) merge triples
- Merged triples get reinforcement_count incremented, confidence averaged
- Deleted anchors remove any infon that references them in S, P, or O
- Composition: for functors F1 and F2, migrate(F2∘F1) = migrate(F2, migrate(F1, data))

Test all four: rename, merge, delete, composition.
```

---

### Why Functors

When you rename "us" to "united_states", some triples that were distinct before the rename (e.g., one had "us" as subject, another "united_states") may become duplicates after. The functor handles this: it's a non-injective morphism, and the migration correctly merges the duplicate triples by reinforcing one and discarding the other.

Composition means you can chain migrations safely. Rename in v1, merge in v2, delete in v3 — and the result is the same whether you apply them sequentially or compose into a single functor.

## Prompt 3: Schema-Free Discovery (Left Kan Extension)

---

```
Build schema-free anchor discovery in cognition/src/cognition/category.py.

SchemaDiscovery class:
  - __init__(encoder: SpladeEncoder)  # raw SPLADE, no anchor projector
  - discover(texts, n_anchors=25, min_doc_freq=2, 
             activation_threshold=0.3) → (AnchorSchema, list[DiscoveredAnchor])

Algorithm:
1. Encode corpus through raw SPLADE → sparse vocab matrix
2. Filter to tokens with document frequency ≥ min_doc_freq
3. Build co-activation matrix, normalize to PMI
4. Spectral clustering: normalized Laplacian → eigenvectors → k-means
5. For each cluster: top tokens by mean activation → anchor name + tokens
6. Infer type from token semantics:
   - Verb-like tokens → relation
   - Country/region names → market
   - Organization/entity tokens → actor
   - Everything else → feature

DiscoveredAnchor dataclass:
  - name, inferred_type, tokens, centroid_indices
  - size (tokens in cluster), mean_activation, coherence

Test: give it raw text (no schema), verify it discovers reasonable anchors,
then use the discovered schema with the standard pipeline to extract infons.
```

---

### The Category Theory

The left Kan extension is the "best approximation" — given a subset of observations (SPLADE activations on raw text), it constructs the free algebra (schema) that explains those observations. The spectral clustering finds the natural basis vectors of the co-activation space, and each cluster becomes an anchor.

This is the adjoint to restriction: if you have a schema and restrict to data, you lose information. The Kan extension goes the other way — from data to schema — recovering the structure that was implicit in the activation patterns.

## Prompt 4: Integration Testing

---

```
Write tests that verify the three extensions work together:

1. Sheaf improves importance:
   - Extract infons with default coherence=0.0
   - Compute sheaf coherence scores
   - Update infon.coherence with sheaf scores
   - Verify all infons now have coherence > 0

2. Discover then migrate:
   - Discover a schema from raw text
   - Extract infons using discovered schema
   - Define a migration functor that maps discovered anchors to
     a manually-curated schema
   - Migrate and verify infons are preserved

3. Full pipeline with timeline:
   - Build a corpus with timestamps spanning 2004-2026
   - Extract infons, compute sheaf coherence
   - Query with persona valence
   - Walk NEXT chains for temporal prediction
   - Report: coherence distribution, Fiedler value, components,
     top constraints, timeline of events
```

---

## The Category-Theoretic View of Cognition

After building these three extensions, step back and see the full picture:

| Category Theory | Cognition System |
|----------------|-----------------|
| Objects | Anchors (typed vocabulary entries) |
| Morphisms | Infons (grounded triples connecting anchors) |
| Composition | NEXT chains (temporal sequencing) |
| Functor | Schema migration (structure-preserving map) |
| Presheaf | Sheaf coherence (local-to-global consistency) |
| Left Kan extension | Schema discovery (free construction from data) |
| Natural transformation | Importance decay (systematic modification across all morphisms) |
| Colimit | Constraints (aggregation of co-supporting infons) |
| Pullback | Query results (intersection of anchor-filtered subgraphs) |
| Sheaf morphism (`P_fwd`, `P_bwd`) | Sheaf-GNN restriction maps per relation |
| Sheaf-Laplacian `L_F` | Unsupervised edge-discrepancy regularizer |

The knowledge graph is a category. Schemas are its sketches. Functors preserve structure across schema evolution. Sheaves measure consistency. And the Kan extension discovers structure from nothing.

---

## Extension 4: The sheaf *neural network*

`SheafCoherence` above is an unsupervised signal on the *anchor* graph.
The sheaf neural network goes further: it changes how the GNN itself
propagates messages.

In an R-GCN layer, each relation `r` has a single weight matrix `W_r`
that both endpoints share. A *sheaf* layer instead gives every relation
a pair of **restriction maps**:

```
P_fwd[r] : h_source → edge_stalk[r]
P_bwd[r] : h_target → edge_stalk[r]
```

Every edge `(s -r-> t)` defines an edge stalk where both endpoints
project. When the projections agree, the relation's view of the pair
is coherent — the sheaf admits a *global section* there.

### The regularizer

```
L_F = (1 / |E|) · Σ_{(s,r,t) in E}  w_e · || P_fwd[r]·h_s − P_bwd[r]·h_t ||²
```

`L_F = 0` iff every edge's endpoints project to the same point at the
stalk. Minimising it pushes embeddings toward geometric coherence
without any labels — an unsupervised prior that *structurally valid*
relations should have consistent endpoint views.

In the shipped code this is wired into `fit()`:

```python
from cognition.logic import HypergraphReasoner

reasoner = HypergraphReasoner(
    store, encoder, schema,
    hidden_dim=64, n_layers=2,
    use_sheaf=True,                  # swap R-GCN for sheaf layer
)
reasoner.fit(graph=graph, epochs=15, laplacian_weight=0.1)
```

### Why it's strictly more expressive than R-GCN

A single `W_r` forces every node listening on relation `r` to see the
same linear view. Forward/backward decoupling lets an edge apply one
view when the source *speaks* into the edge and a different view when
the target *reads* from it. Asymmetric relations (`acquires`,
`causes`, `partners`) can finally have asymmetric geometry.

### When does it pay off

On tiny clean corpora, the sheaf layer roughly matches R-GCN — near-
identity initialisation means there's little asymmetry to exploit, and
few conflicting edges for `L_F` to smooth over.

The payoff shows on **larger, messier graphs** with conflicting
relation views: supply-chain networks where a `partners` edge means
different things to each endpoint, financial filings where
counterparty-dependence is asymmetric, claim-verification corpora where
one paragraph *supports* and another *refutes* the same triple.

`tests/test_sheaf.py` runs the head-to-head; see Module 10 for the
benchmark table.

## Checkpoint

- [ ] Sheaf coherence replaces placeholder `coherence=0.0` with real topological scores
- [ ] Fiedler value correctly measures algebraic connectivity
- [ ] Functorial rename/merge/delete all work, including non-injective merges
- [ ] Functor composition is preserved: F2 after F1 = composite
- [ ] Schema discovery finds reasonable anchors from raw text
- [ ] Discovered schema produces usable infons with the standard pipeline
- [ ] All three integrate: discover → extract with coherence → migrate
- [ ] `SheafMessagePassingLayer` instantiates with `NUM_RELATIONS` pairs of restriction maps
- [ ] `sheaf_discrepancy()` returns a non-negative scalar that decreases under Adam training
- [ ] `HypergraphReasoner(use_sheaf=True)` produces a valid `ReasoningResult` end-to-end
- [ ] θ calibration holds for both variants: mean θ on NEI > mean θ on SUPPORTS

# Module 04: Document to Infon Extraction

## What You'll Learn

- The full pipeline from raw text through the change of basis to grounded triples
- How cartesian triples work and why geometric mean is the confidence metric
- Span finding, support classification, and hierarchy metadata lookup
- Negation detection, tense detection, and temporal reference extraction

## Background

Extraction is the full pipeline from raw text through the change of basis to grounded infons. The SPLADE encoder (Module 03) maps each sentence to concept coordinates; extraction takes those coordinates and forms structured observations. For each document:

1. Split into sentences
2. Encode all sentences through the change of basis → anchor activation matrix (concept coordinates)
3. For each sentence, partition activated anchors by role (subject/predicate/object)
4. Take top-K per role above threshold
5. Form the cartesian product of (subjects x predicates x objects) — every combination of activated anchors by role becomes a candidate infon
6. Enrich each infon with spans, support types, hierarchy, spatial, temporal, importance

## Prompt 1: Build the Extraction Pipeline

---

```
Create cognition/src/cognition/extract.py with:

1. split_sentences(text) — regex-based sentence splitter that handles 
   abbreviations like "Dr.", "$3.5B.", "U.S."

2. extract_infons(documents, encoder, schema, config) → (infons, edges)
   
   Pipeline:
   a. Split all documents into sentences, tracking provenance (doc_id, sent_idx, timestamp)
   b. Batch encode all sentences through the encoder
   c. Compute corpus-level IDF for specificity scoring
   d. For each sentence:
      - Partition activations by role using schema.role_for_type()
      - Sort by score, take top_k_per_role (configurable, default 3)
      - Filter by activation_threshold (configurable, default 0.15)
      - Form cartesian triples from (subjects × predicates × objects)
      - For each triple:
        * Compute geometric mean confidence
        * Find character spans for each anchor in the sentence
        * Classify support type (direct/semantic/hierarchical)
        * Look up hierarchy metadata from schema
        * Detect polarity (negation)
        * Detect tense and temporal references
        * Compute importance score
        * Infer spatial locations from anchor metadata
        * Generate deterministic infon_id
        * Create spoke edges (INITIATES, ASSERTS, TARGETS, LOCATED_AT)
   
   Return (list[Infon], list[Edge])

Also create a CognitionConfig dataclass in config.py with all the 
configurable parameters: thresholds, weights, paths, backend choice.
```

---

## Prompt 2: Test on Your Data

---

```
Load the trained model and run extraction on 5-10 sample documents from 
my domain. For each document, show me:
1. How many sentences were split
2. How many infons were extracted
3. The top 5 infons by confidence
4. For each top infon: the sentence, the triple, the confidence, and 
   the support types

Then show aggregate stats:
- Total infons extracted
- Most common subjects, predicates, objects
- Distribution of support types (how many direct vs semantic vs hierarchical)
- Average confidence per support type
```

---

## Key Design Decisions

### Why Cartesian Product?
If a sentence activates [toyota, honda] as subjects, [invest] as predicate, and [solid_state, ev] as objects, we get 4 infons: (toyota, invest, solid_state), (toyota, invest, ev), (honda, invest, solid_state), (honda, invest, ev). This is intentional — the model says all four relationships are present in this sentence.

### Why Geometric Mean?
Geometric mean (cube root of product) penalizes weak links more than arithmetic mean. If subject=0.9, predicate=0.8, object=0.1, the geometric mean is 0.42 while arithmetic is 0.6. The weak object link drags the score down appropriately.

### Why IDF for Specificity?
An anchor that fires on every sentence (like a common relation) is less informative than one that fires rarely. IDF captures this: `specificity = log(N / (1 + df))` where df is how many documents activate this anchor.

## Checkpoint

- [ ] `extract.py` runs without errors on sample documents
- [ ] Infons have non-empty spans for directly grounded roles
- [ ] Support types are correctly classified
- [ ] Negated sentences produce polarity=0
- [ ] Importance scores are reasonable (0.1–0.8 range)

# Module 02: The Infon Data Model


> **Update for the cassette substrate.** The Infon dataclass is unchanged. What changed is where infons live: immutable cassette files (content-addressed, Parquet-indexed, fsspec-addressable) instead of a mutable SQLite table. See module 05 for the substrate.

## What You'll Learn

- What an infon is — the atomic unit of knowledge, expressed in concept coordinates
- How situation semantics maps to a Python dataclass
- Why grounding, hierarchy, spatial, temporal, and importance metadata matter
- The edge types that connect infons into a knowledge graph

## Background

Once the change of basis (Module 03) maps a sentence into concept coordinates, we need a structure to hold what we observe there. An **infon** is that structure: a grounded unit of information `<<predicate, subject, object; polarity>>`, where subject, predicate, and object are anchors in your concept basis. Each infon is a situated observation at specific concept-space coordinates — situation semantics made concrete.

The data model is inspired by:
- **Situation semantics** (Barwise & Perry): facts are always relative to a situation — here, a sentence projected into concept space
- **Brain Simulator III's Atomic Architecture**: infons are "atomic thoughts" that compete for attention, form temporal sequences, and consolidate through reinforcement

## Prompt 1: Build the Infon Data Model

---

```
Create a new Python package called cognition with this structure:
cognition/
  src/cognition/
    __init__.py
    infon.py
    schema.py

In infon.py, create these dataclasses:

1. Infon — the core knowledge unit with fields for:
   - Identity: infon_id (deterministic hash of doc_id + sent_idx + S + P + O)
   - Core triple: subject, predicate, object, polarity (1=affirmed, 0=negated), 
     direction (forward/reverse/neutral), confidence (geometric mean of role probs)
   - Grounding: sentence text, doc_id, sent_id, character spans per role, 
     support type per role (direct/semantic/hierarchical)
   - Hierarchy metadata: dicts for subject_meta, predicate_meta, object_meta
   - Spatial: list of location dicts with name, level, country_code, source
   - Temporal: timestamp, precision, temporal_refs list, tense, aspect
   - Importance: activation, coherence, specificity, novelty, importance 
     (composite), reinforcement_count, last_reinforced, decay_rate
   - Methods: to_dict(), from_dict(), triple_key()

2. Edge — typed directed edge: source, target, edge_type, weight, metadata
   Edge types: INITIATES, ASSERTS, TARGETS, LOCATED_AT, OCCURRED_ON, NEXT, 
   ENTAILS, CONTRADICTS, SUPPORTS, SIMILAR

3. Constraint — aggregated claim: subject, predicate, object, evidence count,
   doc_count, strength, persistence, score, infon_ids

4. QueryResult — query response: query text, persona, infons list, 
   constraints list, edges list, valence dict, timeline list, 
   anchors_activated dict

In schema.py, create AnchorSchema with:
- Load from JSON file
- Type lookup, hierarchy traversal (ancestors, descendants)
- role_for_type() mapping: actor→subject, relation→predicate, else→object

Use pyproject.toml for packaging with torch, transformers, numpy as deps.
```

---

## Prompt 2: Understand the Design

After the code is generated, ask:

---

```
Walk me through infon.py and explain:
1. Why is confidence the geometric mean and not arithmetic mean?
2. What's the difference between direct, semantic, and hierarchical support?
3. How does the importance lifecycle work (creation → consolidation → decay → pruning)?
4. Why are valence and salience NOT stored on the infon?
5. How do NEXT edges form "experience sequences" like Brain Simulator III?

Keep explanations concise — one paragraph each.
```

---

## Key Concepts

### The Core Triple
Every infon captures WHO did WHAT to WHOM:
- Subject: the actor (Toyota, BMW)
- Predicate: the action (invest, launch, compete)
- Object: the target (solid_state, ev_platform, north_america)

### Three Support Types
When we say "Toyota invests in batteries":
- **Direct**: "Toyota" literally appears in the sentence → direct grounding
- **Semantic**: the model activates `toyota` even though only "the automaker" appears → semantic inference
- **Hierarchical**: "Camry" appears, activating `toyota` via child→parent hierarchy → hierarchical

### Importance = Darwinian Competition
Infons compete for survival like thoughts in a brain:
- High confidence + high specificity + novel = important
- Reinforced by multiple documents = consolidated (stronger)
- Not seen for a while = decaying (weaker)
- Below threshold = pruned (forgotten)

## Checkpoint

- [ ] `cognition/src/cognition/infon.py` compiles without errors
- [ ] `Infon.to_dict()` and `Infon.from_dict()` are inverse operations
- [ ] `AnchorSchema.from_file("data/schema.json")` loads your schema
- [ ] You can explain what each field on the Infon is for

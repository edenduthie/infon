# Module 06: Consolidation


> **Replaced by query-time primitives.** The cassette substrate does not materialize NEXT edges at ingest time — a delta ingest would need to rewrite edge files, breaking append-only storage. Instead, `store.trajectory(anchor)` sorts the by_anchor index at query time, `store.next_edges(anchor)` derives consecutive pairs, and `store.constraint(s, p, o)` aggregates evidence on demand. A later retraction updates aggregates automatically — no consolidation run needed. See module 07 for the query-time replacements.

## What You'll Learn

- How duplicate triples get reinforced (like memory consolidation in brains)
- Building NEXT edges that trace trajectories through concept space over time
- Aggregating infons into constraints (corpus-level assertions)
- Importance decay over time and pruning

## Background

Consolidation adds temporal structure to concept space. The change of basis gives us a static snapshot per sentence; consolidation connects those snapshots into trajectories. NEXT edges trace how entities move through concept coordinates over time — Toyota's path from `invest → partner → launch` is a trajectory through predicate space. Reinforcement strengthens recurring patterns at the same coordinates. Decay lets stale coordinates fade.

Concretely, the Atomic Architecture's lifecycle:

1. **Reinforce**: "Toyota invests in batteries" appears in 50 documents → `reinforcement_count=50`, importance boosted
2. **Sequence**: NEXT edges link infons through shared anchors chronologically → trajectories in concept space
3. **Aggregate**: group by (S, P, O) → constraints with evidence counts
4. **Decay**: old, unreinforced infons lose importance
5. **Prune**: below threshold = soft-deleted

## Prompt 1: Build Consolidation

---

```
Create cognition/src/cognition/consolidate.py with:

1. aggregate_constraints(infons) → list[Constraint]
   - Group infons by (subject, predicate, object)
   - For each group: count evidence, count distinct docs, compute mean 
     confidence, count distinct time windows (monthly buckets)
   - Compute composite constraint score using log-scaled normalization

2. build_next_edges(infons) → list[Edge]
   - For each anchor, collect all infons that mention it
   - Sort by timestamp
   - Link consecutive infons with NEXT edges
   - Metadata: shared anchor name, role, gap in days
   - A single infon gets multiple NEXT edges (one per anchor role)

3. reinforce(existing_infon, new_infon, config) → Infon
   - Increment reinforcement_count
   - Update last_reinforced timestamp
   - Recompute importance with reinforcement bonus: 
     w_reinf * log(1 + reinforcement_count)
   - Running average of confidence

4. apply_decay(infons, reference_date) → infons
   - importance -= decay_rate * days_since_last_reinforced
   - Clamp at 0.0

5. consolidate(new_infons, store, config) → (constraints, edges)
   - Full consolidation pass: check duplicates, reinforce or insert,
     aggregate constraints, build NEXT edges
```

---

## Prompt 2: Visualize the Knowledge Graph

---

```
After running consolidation on my documents, show me:

1. Top 10 constraints by score — these are the strongest claims in the corpus
2. For the top constraint, show the NEXT chain for its subject:
   walk forward through time showing how the subject's story evolves
3. Reinforcement distribution: how many infons have been reinforced 
   1x, 2-5x, 5-10x, 10+x?
4. A simple ASCII visualization of the temporal graph for the top 3 actors:
   timeline with infon markers showing key events
```

---

## NEXT Edges: Experience Sequences

This is the key insight from Brain Simulator III. When you walk the NEXT chain for "toyota as subject", you see Toyota's story unfold:

```
Mar 2025: toyota → invest → solid_state (conf=0.85)
    ↓ NEXT (gap=45 days)
May 2025: toyota → partner → panasonic (conf=0.78)
    ↓ NEXT (gap=30 days)
Jun 2025: toyota → launch → ev_platform (conf=0.72)
```

These chains are per-anchor, so `solid_state` has its own chain showing all the companies investing in it. Walking forward from the latest infon gives you **prediction power** — what's likely to happen next.

## Checkpoint

- [ ] Constraints correctly count evidence and doc_count
- [ ] NEXT edges link infons chronologically through shared anchors
- [ ] Reinforcement increases importance for duplicate triples
- [ ] Decay reduces importance over time
- [ ] The consolidate() function handles the full lifecycle

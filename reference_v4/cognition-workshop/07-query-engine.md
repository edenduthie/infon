# Module 07: The Query DSL and Reasoner

## What You'll Learn

- How the cassette query DSL composes grammar (where / mentioning / polarity), timeline (between / before / after), and logic (AND / OR / NOT) into a single immutable Query object
- Why `store.ask(Query)` returns a calibrated Dempster–Shafer verdict, not just hits
- How `store.connect()` and `store.any_of()` handle multi-hop and one-of-many questions via MCTS
- Why trajectories and constraints are computed at query time from the index, not materialized at ingest
- How hierarchy expansion walks the schema tree at query time so parent queries match descendant hits

## Background

Earlier versions of this module taught a persona-plus-valence query engine: users passed a persona name, valence weights were computed at query time from each infon's polarity and the persona's preferences, and the ranking combined importance × salience × urgency × recency. It worked, but the persona layer added state users had to reason about, and valence as a first-class concept made simple questions more complicated than they needed to be.

The cassette query surface is smaller. Four primitives compose into everything:

1. **`Query`** — an immutable filter over the cassette indexes. `where(s, p, o)`, `mentioning(a)`, `affirmed() / negated()`, `between(t1, t2)`, `expand_hierarchy(schema)`. Chained calls AND together; `run_any([q1, q2])` does OR.

2. **`store.ask(Query)`** — single-claim reasoner. Retrieves matching infons, computes Dempster–Shafer masses, returns a Verdict with `(supports, refutes, theta)` and the cited sources.

3. **`store.connect(source, target)`** — multi-hop. Runs MCTS over connective edges (partner / supply / acquire / license / invest), returns the path.

4. **`store.any_of(source, targets)`** — one tree walk, many targets. Cost is roughly O(1) in the number of targets.

Persona and valence don't disappear — they just move to the *query* side. If a user wants an investor's lens, they write it into the Query filters (recency bounds, polarity, confidence threshold). The store stays neutral; bias is a property of the question, not the data.

## Prompt 1: The Query Dataclass

---

```
Build cognition/src/cognition/cassette/dsl.py with an immutable Query
class. Every builder method returns a new instance:

  Query()
    .where(subject=..., predicate=..., object=...)    # AND each call
    .mentioning(a)                                     # role-free match
    .affirmed() / .negated()                           # polarity filter
    .min_conf(c)                                       # confidence floor
    .between(t_start, t_end)                           # time window
    .before(t) / .after(t)                             # open-ended time
    .contradicting()                                   # polarity flip on pinned triple
    .expand_hierarchy(schema)                          # walk descendants per pinned role

Plus:
  run_any(manifest, [q1, q2, ...])   # OR — union dedup'd by infon_id

Rules:
- All methods return a fresh Query (frozen dataclass).
- `.run(manifest) -> list[Hit]` dispatches to the right index (by_triple /
  by_time / by_anchor) based on what's pinned, driving the manifest pruner.
- Multiple `where()` calls accumulate; they don't overwrite.
- `expand_hierarchy(schema)` populates subject_set / predicate_set /
  object_set with anchor + all descendants, then the executor does
  "IN anchor_set" instead of "= anchor".

The DSL is the API surface users see. Make it small, immutable, and
composable so chaining + introspection stay obvious.
```

---

## Prompt 2: The Reasoner

---

```
Build cognition/src/cognition/cassette/reason.py with a single claim
verifier:

  reason(manifest, claim: Query, *,
         fetcher=LocalFetcher(),
         evidence_filter=None,
         max_evidence=20) -> Verdict

Steps:
1. Resolve hits via evidence_filter (default: Query().where(subject=
   claim.subject) — broadest relevant net).
2. Cap to max_evidence, sorted by confidence.
3. Hydrate via range GETs.
4. For each infon, compute a 6-case mass function against the claim:
     - exact triple match, polarity agrees → SUPPORTS
     - exact triple match, polarity disagrees → REFUTES
     - S+P match, different O → REFUTES (semantic contradiction)
     - S only match → small SUPPORTS/REFUTES by polarity
     - P or O only → tiny signal, weighted by confidence
     - no overlap → pure θ
5. Combine the top-5 most decisive via Dempster's rule.
6. Threshold to label.

Return Verdict{label, mass, sources, n_candidates, n_hydrated,
range_gets}. Every verdict carries its sources — the reasoner can be
wrong but never lies about what it saw.

The critical calibration is case (d): S-only matches give a BARELY-
positive mass (0.05 * w, not 0.15 * w). Dempster amplifies positive S
across multiple infons; an aggressive S-only rule tips NEI claims into
spurious SUPPORTS. Low case-(d) weight is what keeps θ → 1.0 on
claims the corpus doesn't answer.
```

---

## Prompt 3: Multi-hop via MCTS

---

```
Build cognition/src/cognition/cassette/reason_path.py with two
entrypoints:

  reason_connectivity(manifest, source, target, budget=20,
                       connective_predicates=None,
                       use_gnn=False, gnn_weight=0.5)
      → Verdict

  reason_any_target(manifest, source, targets: set, budget=20,
                     connective_predicates=None)
      → dict[target → Verdict]

Both run an AlphaGo-style MCTS from `source`. Each node is a partial
chain; each expansion fetches the next hop's infons; UCB balances
|S - R| (decisiveness) against θ * sqrt(ln N / n) (exploration).

Three design choices that are load-bearing — flag them in comments:

1. Chain mass = conjunction, not Dempster:
     S(chain) = min over edges (weakest hop)
     R(chain) = max over edges (any retraction breaks the chain)
     θ       = 1 - S - R, with a per-hop penalty (0.10 * (n - 1)).
   Dempster's additive combine amplifies S as edges accumulate, which
   is wrong for chains.

2. Edge grouping by (s, p, o) triple with retraction folding:
   Multiple infons for the same triple (affirm at t1, refute at t3)
   collapse into ONE edge. Dempster combines their per-infon masses;
   the conflict normalization handles the "affirmed then retracted"
   case correctly — supports and refutes both drop, θ rises.

3. Connective predicates filter at expansion, not just at scoring:
   An edge whose predicate isn't connective (mention, describe) never
   gets added to the frontier. Saves hydration AND prevents graph-
   coincidental chains like "X mentions Y, Y mentions Z".

For any_of, the single tree walk attributes to every target on a leaf's
path — a chain of length 3 can resolve 3 targets at once.

GNN prior is optional — if <root>/_model/gnn.pt exists and
use_gnn=True, we score the final chain with the trained sheaf GNN and
blend into the symbolic mass.
```

---

## Prompt 4: Query-time Trajectory and Constraint

---

```
Add to cognition/src/cognition/cassette/dsl.py:

  trajectory_hits(manifest, anchor) -> list[Hit]
  next_edges(manifest, anchor)       -> list[NextEdge]
  constraint(manifest, s, p, o)       -> Constraint

All computed at read time from the by_anchor and by_triple indexes.
None of these are materialized at ingest.

Why not materialize NEXT edges:
- A later ingest can add an infon for an entity that already has a
  trajectory. Materialized NEXT edges would need to be rewritten,
  breaking the append-only cassette guarantee.
- At read time, sorting an anchor's hits by timestamp is cheap (ms
  per entity) because the index is already scoped to the anchor.

Why constraint is computed not stored:
- A retraction of (s, p, o) that lands in a later ingest updates the
  constraint's polarity_balance automatically. No consolidation step
  to run. No stale aggregates.

Constraint fields: evidence_count, n_affirmed, n_refuted, mean_conf,
t_min, t_max, span_days, polarity_balance, is_contested.

Expose these on InfonStore:
  store.trajectory(anchor, hydrate=True|False)
  store.next_edges(anchor)
  store.constraint(s, p, o)
```

---

## Why the persona/valence layer is gone

The old query engine baked a *perspective* into the store. Every infon was scored against a persona's preferences — an investor saw negative valence on decline infons, a regulator saw positive valence on compliance infons. This was elegant theory, but three real costs showed up:

1. **Persona state became load-bearing.** Users had to remember which persona they had active; agents had to choose one before asking. The choice was rarely obvious.

2. **Valence was computed at query time but tied to infon storage.** Caching valence per (infon, persona) was tempting but wrong, because new evidence could flip the sign. So valence was recomputed on every query, duplicating work.

3. **The reasoner's job got muddled.** Mass functions already encode belief honestly. Layering persona-weighted valence on top either agreed with the mass (redundant) or contradicted it (confusing).

The cassette engine gives users the same expressive power via filters. An investor who cares about battery investments in 2026 writes:

```python
store.any_of(
    "toyota",
    {"batteries", "solid_state", "hbm"},
    connective_predicates={"invest", "partner", "acquire"},
)
```

The query embodies the perspective; the store stays neutral. Agents compose perspective into questions instead of setting a mode.

## Checkpoint

- [ ] `Query().where(subject="toyota").run(manifest)` returns hits without hydration.
- [ ] `Query().where(...).affirmed().run(...)` excludes retractions.
- [ ] `Query().mentioning("catl").run(...)` finds catl in any triple role.
- [ ] `store.ask(Query().where(subject=..., predicate=..., object=...))` returns SUPPORTS / REFUTES / NEI with cited sources.
- [ ] A NEI claim the corpus can't answer returns `θ ≈ 1.0` with `range_gets == 0`.
- [ ] `store.connect(a, b)` finds 2–3 hop chains via connective predicates.
- [ ] `store.constraint(s, p, o).polarity_balance` flips signs correctly when a retraction is appended in a later ingest.

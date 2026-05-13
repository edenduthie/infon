# Module 14: Bootstrap — corpus in, trained reasoner out

## What You'll Learn

- Why the shipping product replaces the full sklearn-style AutoML loop with two focused tools: `extraction_report()` and `bootstrap_gnn()`
- How `extraction_report()` catches corpus-quality failures *before* users waste compute — the cassette equivalent of the old `analyze_corpus` probes
- How `bootstrap_gnn()` lets the Strands `Analyst` peek at docs, design a synthgen config, train a sheaf GNN on synthetic data, and save per-store weights — all in one conversational turn
- Why shipping a trained model (the sheaf GNN on synthgen) beats shipping a sweep-and-select loop for most users

## Background

The original Module 14 pitch was a full sklearn-style AutoML: `analyze_corpus` runs six conditional-entropy probes, `auto_select_ssl` sweeps four SSL modes, `cross_val_score` does K-fold on held-out infons, `ensemble_top_k` combines champions via Dempster's rule, and `random_state` threads determinism through every RNG.

Most of that didn't ship. Here's the honest sequencing of what actually solved the user problem:

1. The biggest leverage came from **diagnosing corpus quality**, not from selecting between reasoner variants. Users whose schemas missed 40% of their corpus weren't helped by sweep — they needed to know their schema was broken. That became `extraction_report()`.

2. The Dempster–Shafer reasoner + sheaf GNN handle almost every use case without per-corpus tuning. The GNN is pre-trained on synthgen's distributional levers, not on the user's corpus. Features are schema-independent (relation-kind, polarity, position) so the trained weights transfer across domains. No sweep needed.

3. The one per-corpus tuning step that remained — "the user's domain has relation types synthgen doesn't cover" — is handled by `bootstrap_gnn()`, which is a single method call, not an AutoML loop.

4. `cross_val_score` and `ensemble_top_k` were elegant but solved a problem users didn't actually have at our scale. We shipped without them; no user has asked.

What remained was a simpler promise: *you give us a corpus, we tell you if the schema is broken, we train the GNN on your domain's patterns, you ask questions.* Two methods, not ten.

## Prompt 1: extraction_report — corpus diagnostics for free

---

```
Build cognition/src/cognition/cassette/diagnostic.py.

Four failure-mode categories, each measured once from the by_triple +
by_anchor indexes. Zero hydration — all counts come from Parquet
aggregates:

1. docs_with_zero_infons
   A document produced no triples. Common causes:
     - schema missed the entities mentioned ("pan" instead of "panasonic")
     - sentence is too telegraphic for SPLADE to disambiguate
     - extractor's role constraints dropped the only candidate triple

2. unused_anchors
   A schema anchor never fired. Common causes:
     - dead anchor left from an earlier schema version
     - tokens are wrong ("u.s." in schema, "US" in corpus)
     - anchor typed as the wrong role for its actual surface form

3. overfit_objects
   One anchor hoarding the object role (>50% of triples). Usually
   means the object vocabulary is too narrow — one feature absorbs
   every sentence because nothing else matches.

4. role_imbalance
   An anchor landing in a role its type shouldn't fill. Relation
   in subject slot means a verb is masquerading as a noun (or the
   schema typing is wrong). Post the actor-as-object extraction fix
   we accept actors in BOTH subject and object slots, so this flags
   only true mis-typings.

The report is attached to every ingest() result. Users see it for free.

Return it from InfonStore.extraction_report() too, for introspection
after the fact.
```

---

## Prompt 2: bootstrap_gnn — per-corpus training in one agent turn

---

```
Add a tool to cognition/src/cognition/cassette/analyst.py:

  @tool
  def bootstrap_gnn(sample_size: int = 5,
                     synth_samples: int = 2000,
                     epochs: int = 25) -> str:
      \"\"\"Peek at the store's corpus, design a synthgen config that
      matches its domain structure, generate labeled hypergraphs,
      train the sheaf GNN, save weights to <root>/_model/gnn.pt.

      Flow:
        1. Read `sample_size` hydrated infons from the store.
        2. Infer RelationSpec per predicate:
             kind = \"connective\" if most objects are actors
                    \"reportive\" if most objects are non-entities
                    \"terminal\" otherwise
        3. Build a SynthGenConfig using the same anchor vocabulary
           and relation kinds as the real corpus.
        4. Generate synth_samples labeled SynthGraphs.
        5. Train a SheafHypergraphEncoder(hidden_dim=64, n_layers=3)
           on held-out synth data.
        6. Save {state_dict, hidden_dim, n_layers, best_val_acc} to
           <root>/_model/gnn.pt.
        7. Return a summary + the best_val_acc.
      \"\"\"

Users invoke this via the Analyst: \"train a reasoner on my corpus.\"
The LLM reads the extraction_report first (coverage is fixable); then
calls bootstrap_gnn; reports val_acc to the user.

Why this is the right shape:
- Sheaf GNN features are schema-independent. Per-corpus training can
  adapt to the user's specific relation-kind distribution without
  redesigning the encoder.
- Training cost is ~60s on CPU. A one-shot operation, not an
  hour-long sweep.
- The trained weights live in the store's _model/ directory —
  portable with the cassettes, invisible to users who don't need it.
- When a later ingest lands that meaningfully changes the distribution,
  the user re-runs bootstrap_gnn. No background retrain daemon.
```

---

## What the original AutoML pitch DID ship, in spirit

The six original themes survived in more focused forms:

| Original Module 14 ambition | How it shipped |
|---|---|
| `analyze_corpus` conditional-entropy probes | `extraction_report()` four-category diagnostic, automatic after every ingest |
| `auto_select_ssl` sweep over SSL modes | Synthgen-trained sheaf GNN with per-relation-kind restriction maps — one architecture, one training regime |
| `sweep(config_grid, budget)` with held-out scoring | `bootstrap_gnn()` — one-shot per-corpus training, no grid |
| `cross_val_score` K-fold | Not shipped; not needed at our typical scale |
| `ensemble_top_k` via Dempster's rule | Not shipped; the single trained GNN + symbolic ensemble via `use_gnn=True, gnn_weight=0.5` is a lighter version of the same idea |
| `random_state` threading | Still used in synthgen and sheaf GNN training — see `SynthGenConfig.seed` and `torch.manual_seed` calls in the training script |

## Honesty about what's NOT automated

Two things users still do manually:

1. **Schema design.** `extraction_report` tells them *when* the schema is wrong, but not *what* to change. The Strands `Analyst` proposes fixes in natural language but the user approves them. Automated schema discovery (via Kan extensions on co-activation clusters) is theory in module 11 and not shipped.

2. **Connective-predicate classification.** `infer_connective_predicates` gives a 0.5-ratio heuristic that works on corpora with clear actor/feature separation. On noisy extraction where actors land in object slots, users may want to pass `connective_predicates={"partner", "supply", ...}` explicitly. The README's benchmarks quote both modes.

These are user decisions by design — they encode domain knowledge that a corpus alone can't reveal.

## Checkpoint

- [ ] `store.ingest(docs)` returns a result whose `report` field flags docs producing zero infons, unused anchors, and overfit objects.
- [ ] `store.extraction_report()` returns the same shape on demand.
- [ ] `analyst("train a reasoner on my corpus")` calls `bootstrap_gnn` and reports held-out val_acc.
- [ ] After `bootstrap_gnn()` the file `<root>/_model/gnn.pt` exists and loads cleanly.
- [ ] `reason_connectivity(..., use_gnn=True)` picks up the trained model without any other configuration.
- [ ] The same query with `use_gnn=False` still works (fallback is symbolic-only).

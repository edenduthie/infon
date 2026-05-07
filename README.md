# Infon

**Infon is scikit-learn for graph reasoning on text. One store, a trained sheaf GNN prior, calibrated verdicts that say when they don't know. Laptop CPU, S3-native, under 5 seconds to first answer.**

Five things make it different from everything else in this space:

1. **Cassette-native storage.** Documents become immutable, content-addressed `.inf` files on S3 (or local FS via `fsspec`). Delta ingests append; nothing is ever rewritten. Manifest pruning skips cassettes that can't contain your anchor in **7–16×** fewer Parquet opens at scale, **278×** on misses.
2. **No query language.** Plain-English questions in, calibrated verdicts out. A Strands-powered `Analyst` routes questions to the right cassette primitive: single claim, multi-hop connectivity, one-of-many reachability. The agent never invents facts — it cites the sources `ask()` returned.
3. **It tells you when to trust it.** Every answer is a Dempster–Shafer mass `(supports, refutes, θ)`. A trained sheaf GNN scores the chain as a prior. On claims the corpus doesn't speak to, θ → 1.0 instead of a confident wrong answer. Measured: symbolic **80%** → GNN **100%** on the actor-to-actor real-data eval.
4. **Sheaf GNN is shipped, not optional.** A 140k-param sheaf-structured hypergraph encoder, trained once on synthgen (known ground truth), frozen and wired into the reasoner as a terminal scorer. On synthetic held-out it hits **99.2%** overall, **+94%** over symbolic on reportive-edge anomalies.
5. **Schema evolution without reingestion.** `SchemaFunctor(rename, merge, delete)` rewrites existing cassettes under a new ontology via Kan pushforward. **62× faster** than re-extracting. Old cassettes stay, time-travel intact.

**Agent-native by construction.** The `Analyst` exposes 9 tools (`set_schema`, `ingest`, `reingest`, `extraction_report`, `ask`, `connect`, `any_of`, `record_finding`, `list_findings`) to any Strands agent. Extraction diagnostics surface coverage issues automatically. Findings persist across sessions so the store remembers what's been learned.

---

## One-minute start

```python
from infon import InfonStore, Query, Analyst

store = InfonStore("./data/chips", schema_path="schemas/auto.json")

store.ingest(documents)                           # delta append; idempotent
report = store.extraction_report()                # coverage diagnostics
print(report.summary())                           # flags missing anchors, overfit objects

v = store.ask(Query().where(subject="toyota",
                             predicate="invest",
                             object="solid_state"))
print(v.label, v.mass.supports, v.mass.theta)     # SUPPORTS 0.53 0.29

# Multi-hop:
v = store.connect("toyota", "catl")               # chain → SUPPORTS via panasonic
vs = store.any_of("toyota", {"catl","lg","samsung"})  # one tree walk, N verdicts

# Conversational:
a = Analyst(store)
print(a("Does Toyota partner with Panasonic?"))   # agent translates, cites sources
```

S3 is a URI swap:

```python
store = InfonStore("s3://acme/chips", schema_path="schemas/auto.json")
# same API, fsspec handles the rest
```

---

## The four pillars

### 1 · Cassette substrate — S3-native, delta-ingestible, time-travelable

| Mechanism | Where it lives | What it does |
|---|---|---|
| Cassette format | `cassette/format.py` | 8-byte magic + JSON header + gzip-per-record frames + JSON footer + 16-byte trailer. Range-addressable via one tail `GET`. |
| Split indexes | `cassette/index.py` | Per-cassette `by_triple / by_time / by_anchor` Parquet shards. Never rewritten. |
| Manifest pruner | `cassette/index.py` | Cassette-level bbox on anchor sets + time range. 7–16× shard skip at 300 cassettes, 278× on misses. |
| Time-travel | `Manifest.load_at()` | Every ingest creates a new snapshot in a parent chain. Queries at snapshot `S` see the HEAD-as-of-S view. |
| Delta ingest | `InfonStore.ingest()` | Content-addressed by `sha256(text + schema_ref)`. Re-running is free. |
| Schema migration | `cassette/migrate.py` | Kan pushforward via `SchemaFunctor`. Old cassettes stay, new ones join. |

**Measured result:** ingesting 48 docs with `SyncExecutor` → 0.5s. Migrating the same store to a new schema via functor → 20ms. 10 cassettes / 600 infons / 300 MCTS iterations: 10.5% of corpus read, 2 range gets per query at scale.

### 2 · DSL — grammar, timeline, logic compose

| Primitive | Example | What it answers |
|---|---|---|
| `where(s=, p=, o=)` | `Query().where(subject="toyota", predicate="invest")` | Pin triple roles |
| `mentioning(*a)` | `Query().mentioning("catl")` | Role-free anchor match |
| `affirmed()` / `negated()` | `Query().where(s="honda", p="invest", o="lithium_ion").negated()` | Polarity filter |
| `between / before / after` | `Query().where(s="toyota").between("2026-02-01","2026-02-28")` | Temporal window |
| `contradicting()` | `claim.contradicting()` | Flip polarity — find refuters |
| `expand_hierarchy(schema)` | `Query().where(s="chip_maker").expand_hierarchy(s)` | Parent query → all descendant hits |
| `run_any([...])` | logical OR across queries | — |
| `trajectory(anchor)` | `store.trajectory("nvidia")` | Time-ordered sequence, NEXT edges derived at read time |
| `constraint(s,p,o)` | `store.constraint("samsung","supply","hbm")` | Corpus-level aggregate: count, polarity balance, span |
| `count_by` / `first_seen` / `last_seen` | aggregate pushdown | No hydration |

**Measured result:** 500-infon compliance query (contradiction search) resolves to 0 range gets — pure index scan. Time-travel snapshots cost one file read.

### 3 · Reasoner — symbolic + sheaf GNN

| Layer | Where | What it does |
|---|---|---|
| Dempster–Shafer | `cassette/reason.py` | Per-infon mass `(S, R, θ)`, combined via Dempster's rule |
| MCTS traversal | `cassette/reason_path.py` | AlphaGo-style search over the hypergraph. Polarity-aware `chain_mass` (conjunctive min/max), connective-predicate filter |
| Sheaf GNN encoder | `cassette/gnn_encoder.py` | 3-layer message passing with per-relation-kind restriction maps; H¹ discrepancy as anomaly signal; chain-verdict head |
| Synthetic corpus | `cassette/synthgen.py` | `SynthGenConfig` → labeled hypergraphs. Trains the GNN with known ground truth |
| Auto-infer connectives | `cassette/reason_path.py` | Predicates are connective when objects are mostly entities. No schema annotation required |

**Measured result:** symbolic 88.5% → GNN **99.2%** on synthgen held-out. On the actor-to-actor real-data eval after extraction fix: symbolic **80%** → GNN **100%**. The GNN's +94% win is on reportive-edge anomalies the symbolic reasoner cannot distinguish.

### 4 · Strands Analyst — conversational layer on top

| Tool | Purpose |
|---|---|
| `set_schema(ontology_json)` | Activate ontology (dict-first, no temp files) |
| `ingest(documents_json)` | Delta ingest + auto-run extraction_report |
| `reingest()` | Re-extract under the active schema |
| `extraction_report()` | Coverage + dead anchors + overfit objects |
| `ask(s, p, o, polarity)` | Single-claim verdict with cited sources |
| `connect(source, target, max_hops)` | Multi-hop chain — SUPPORTS / REFUTES / NEI |
| `any_of(source, targets_json)` | One tree walk, N targets, ranked verdicts |
| `record_finding(title, body, tags)` | Cross-session memory (findings persist under `<root>/findings/`) |
| `list_findings(tag, limit)` | Read back prior investigations |

The system prompt enforces the non-negotiables: never output a verdict without citing `ask()` sources; when θ > 0.7 explicitly say the corpus doesn't answer; check `list_findings` first; run `extraction_report` before answering on a new store. On a 15-doc AI-chip cold-start the agent reaches **100%** doc coverage in **2 iterations** of schema refinement.

---

## The arc — how each design choice was earned

Every stage below has a reproducible probe in `experiments/`.

| # | Experiment | Result | Decision |
|---|---|---|---|
| 1 | Cassette format vs. SQLite/DynamoDB | Immutable + S3-native → delta ingest trivial; range gets scale with corpus fan-out | **Ship** — `InfonStore` is the primary entry point |
| 2 | Manifest bbox pruner | At 300 cassettes: 7–16× shard skip on real workloads, 278× on misses | **Always on** |
| 3 | MCTS retrieval vs. flat top-k | Flat-seed SPLADE: 0.93 recall@20 on single-hop; MCTS wins at multi-hop (flat can't find 2-hop chains) | **Use flat for factoid, MCTS for connectivity** |
| 4 | Chain mass: Dempster vs. conjunctive min/max | Dempster amplifies S as edges accumulate (wrong for chains); min/max matches "all hops hold" | **min/max is the default** |
| 5 | Auto-inferred connective predicates | Heuristic: object-is-entity ratio ≥ 0.5; terminal-entity widening for chain endpoints | **Default on; explicit set overrides** |
| 6 | Sheaf GNN on synthgen | 99.2% accuracy, +94% on reportive-edge anomalies vs. symbolic baseline | **Ship trained weights; wire as terminal scorer** |
| 7 | Extractor actor-as-object fix | Before: `nvidia/partner/b200` (wrong). After: `nvidia/partner/tsmc` + word-order penalty for direction | **Default on** |
| 8 | Kan-based schema migration | 62× faster than reingest on 10-infon store; old snapshots remain queryable | **Ship** — `SchemaFunctor` + `store.migrate()` |
| 9 | Strands Analyst bootstrap | 15-doc corpus, no schema: agent proposes ontology, ingests, reports, refines → 100% coverage in 2 iterations | **Default conversational entrypoint** |

---

## Install

```bash
pip install -e .
```

| Dependency | Purpose | Required? |
|---|---|---|
| `torch` ≥ 2.6 | GNN + SSL losses | yes |
| `transformers` ≥ 4.40 | SPLADE tokenizer/model | yes |
| `numpy` ≥ 1.24 | linear algebra | yes |
| `pyarrow` ≥ 15 | cassette indexes | yes |
| `fsspec` ≥ 2024.1 | local + S3 paths | yes |
| `s3fs` ≥ 2024.1 | S3 backend | optional (required for `s3://` URIs) |
| `strands-agents` ≥ 1.0 | `Analyst` conversational layer | optional |
| `boto3` | Lambda deploy + ECR | optional |

17 MB SPLADE-tiny ships with the package. No model download, no GPU, no API keys.

---

## What the API looks like

```python
# 1. Cassette-native storage with delta ingest
store = InfonStore("s3://acme/chips", schema_path="schema.json")
result = store.ingest(docs)
print(result["report"].summary())                   # coverage diagnostics, free

# 2. Composable DSL
hits = (Query().where(subject="toyota", predicate="invest")
               .after("2026-02-01")
               .affirmed()
               .run(store.manifest))

# 3. Calibrated single-claim reasoner
v = store.ask(Query().where(subject="toyota", predicate="invest",
                             object="solid_state"))

# 4. Multi-hop with GNN prior (bootstrap_gnn writes <root>/_model/gnn.pt once)
v = store.connect("toyota", "catl")                 # MCTS + sheaf GNN scorer

# 5. One tree walk, many targets
vs = store.any_of("toyota", {"catl","lg","samsung","sk_hynix"})

# 6. Schema evolution without reingest
functor = SchemaFunctor(rename={"sk_hynix": "sk_hynix_corp"},
                         merge={"azure": "microsoft"},
                         delete={"tpu"})
store.migrate(functor, "schema_v2.json")            # 62× faster than reingest

# 7. Strands-powered conversation
a = Analyst(store)
a("Which chip companies is OpenAI linked to?")      # → any_of across targets
```

---

## Limitations

| Limit | Reason | Workaround |
|---|---|---|
| Ingestion is SPLADE-bound | ~2s cold-start per worker for the 17MB model | Batch 20+ docs per worker; fan out with `ProcessExecutor` at 100+ docs |
| Schema is load-bearing | Missing anchors → silent extraction failures | `extraction_report()` surfaces this within 1s of ingest |
| Actor-to-actor requires the dual-partition fix | Default extractor now dual-partitions actors, but older cassettes may miss chains | Reingest OR run `store.migrate()` with identity functor under a new schema |
| GNN is synthgen-trained | Transfers to most domains because features are schema-independent (role + polarity + position); novel relation kinds may require retrain | `bootstrap_gnn()` on your own corpus |
| Torch 2.6+ is 1.5GB | Lambda layer limit is 250MB; container path required | `lambda_container.py` builds a Lambda-compatible container image |
| No AutoML loop yet | Bootstrap covers the 80% case; per-user hyperparameter sweep deferred | Tune `hidden_dim`, `n_layers` manually in `SheafHypergraphEncoder` |

---

## Where to go next

| You are | Start here |
|---|---|
| A user evaluating fit | [00 — Quick Start](00_quick_start.ipynb) + `experiments/benchmark_eval.py` |
| Building a production pipeline | [07 — Cloud](07_cloud.ipynb) + `src/infon/cassette/lambda_container.py` |
| A researcher reproducing the arc | `experiments/` — reproducible evaluation scripts |
| Shopping for the theory | [08 — Category Theory & Sheaves](08_category_theory.ipynb) — Kan migration + sheaf GNN + H¹ discrepancy |
| Integrating with an LLM agent | [06 — Agent Tools](06_agent_tools.ipynb) + `src/infon/cassette/analyst.py` |
| Schema migration (new) | `experiments/exp2_morphic_propagation.py` |

---

## Benchmark summary

| Measure | Symbolic only | After extraction fix + GNN |
|---|---|---|
| Accuracy on 10-claim actor-to-actor real eval | 40% | **100%** |
| Accuracy on synthgen held-out (2000 samples) | 88.5% | **99.2%** |
| Anomaly (reportive-edge trap) accuracy | 6% | **100%** |
| Range gets per MCTS query at 300 cassettes | 20 (flat) | **1.4** (pruner + MCTS) |
| Migration vs reingest cost (10-infon store) | 1245 ms | **20 ms (62×)** |
| Ingest wall for 48 docs via `SyncExecutor` | — | ~500 ms |
| Schema-bootstrap by Analyst (15 docs, cold start) | — | **100% coverage in 2 iterations** |

Details in `experiments/` — every row above has a reproducible script.

---

## References

Bodnar et al. 2022 (Neural Sheaf Diffusion) · Schlichtkrull et al. 2018 (R-GCN) · Shafer 1976 (Dempster–Shafer) · Barwise & Perry 1983 (situation semantics) · Kan 1958 (adjoint functors for schema migration).

Pywren philosophy for Lambda fan-out: same API locally and in the cloud. Common Crawl's CDX layout for range-addressable archival.

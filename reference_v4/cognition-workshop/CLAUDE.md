# Cognition Workshop

A Claude Code workshop for building a knowledge graph system grounded in situation semantics and category theory. You define a typed ontology, a pretrained sparse language model performs a **change of basis** from token space to your concept space, and structured knowledge emerges without any model training. Everything lives on immutable cassettes that run local or on S3 with a one-character URI change.

## The Core Idea: Change of Basis

In linear algebra, a change of basis re-expresses vectors in a new coordinate system without losing information. That is exactly what happens here:

1. **Token basis** (dim 30,522) — BERT's WordPiece vocabulary. Every sentence is a point in this space.
2. **Concept basis** (dim *n*, your schema) — your domain's typed anchors: actors, relations, features, markets.
3. **Change-of-basis operator** — [splade-tiny](https://github.com/rasyosef/splade-tiny-msmarco) (4.4M params, Apache 2.0) applies `log(1 + ReLU(MLM_logits))` → max-pool to produce a sparse 30,522-dim vocabulary expansion, then the AnchorProjector maps those activations to your anchors via token ID max-pooling.

The result: every sentence is now expressed in your concept coordinates. No training — just define the target basis and the projector does the rest. This is why the system works across any domain instantly.

[splade-tiny-msmarco](https://github.com/rasyosef/splade-tiny-msmarco) is a knowledge-distilled BERT-Tiny trained on 1.2M MS MARCO passages. At 4.4M parameters (15× smaller than distilbert-based SPLADE), it ships bundled with the package — no model download, no GPU required. It beats BM25 by 65% on passage retrieval because its MLM head learns to activate semantically related vocabulary terms beyond the literal input tokens — this is the *concept basis expansion* that makes schema projection work.

## What You'll Build

A **cognition** system that:

1. Takes raw documents from any domain
2. Applies the change-of-basis via SPLADE sparse encoding (Apache 2.0)
3. Projects vocabulary activations onto a typed anchor schema — no training required
4. Extracts grounded knowledge triples (*infons*) from situation semantics, with actors eligible as both subject and object (dual-partition)
5. Stores them in **immutable cassettes** with split Parquet indexes — content-addressed, delta-appendable, S3-native
6. Supports **time-travel** for free: every ingest creates a manifest snapshot
7. Queries via a composable DSL (grammar, timeline, logic, aggregate, hierarchy expansion)
8. Verifies claims via **Dempster–Shafer belief calculus** — calibrated mass functions over {SUPPORTS, REFUTES, UNCERTAIN} with residual ignorance θ
9. Explores multi-hop connections via **AlphaGo-style MCTS** over the hypergraph
10. Uses a **trained sheaf GNN** as a terminal scorer on MCTS chains (140k params, ships with every store)
11. Evolves schemas via **Kan-based functorial migration** — rename, merge, delete anchors without reingesting
12. Routes through a **Strands Analyst** with 9 tools for conversational use
13. Runs identically **local or S3**; ingestion fans out to Lambda containers via a clean executor protocol

## How This Workshop Works

Each module is a `.md` file containing prompts for Claude Code. You don't write code — you give Claude Code the prompts and it builds everything. You learn by reading what it produces and asking questions along the way.

## Prerequisites

- Python 3.11+
- Claude Code CLI installed
- No GPU required (splade-tiny is 17 MB, runs on CPU)

## Storyline

The workshop follows a six-act arc, each act building on the last:

### Act I — The Basis (Modules 01–03)

Define your domain's coordinate system. Design a typed anchor schema (the target basis), build the infon data model from situation semantics, and implement the SPLADE encoder with anchor projection — the change-of-basis operator that maps token space to concept space.

*After Act I you can encode any sentence into your domain's coordinates.*

### Act II — Extraction & Storage (Modules 04–05)

Build the pipeline that transforms documents into grounded infons: sentence splitting, SPLADE encoding, cartesian triple formation, span finding, negation detection, word-order direction resolution for actor-to-actor triples. Store everything in the **cassette substrate** — immutable content-addressed byte files with per-cassette Parquet indexes and an append-only manifest chain.

*After Act II you have a populated, delta-ingestible knowledge graph.*

### Act III — Intelligence (Modules 06–07)

Replace ingest-time consolidation with query-time primitives. `store.trajectory(anchor)` sorts the by_anchor index by timestamp at read time — no materialized NEXT edges, so delta ingest stays truly append-only. `store.constraint(s, p, o)` aggregates evidence on demand, so a retraction in a later batch updates the aggregate immediately. The query DSL composes grammar, timeline, logic, and hierarchy expansion into one immutable `Query` object; `store.ask(Query)` returns a calibrated Dempster–Shafer verdict.

*After Act III you have a working intelligence system. Modules 08–14 are enrichments.*

### Act IV — Agency & Scale (Modules 08–09)

Expose the graph to an LLM agent via the 9-tool **Strands `Analyst`**: set_schema, ingest, reingest, extraction_report, ask, connect, any_of, record_finding, list_findings. Deploy via **Lambda container images** (not zip layers — torch is 1.5 GB) using `cognition.cassette.lambda_container`'s Python-only build-push-deploy flow. Same InfonStore code runs local or S3.

### Act V — Belief & Reasoning (Modules 10–13)

Evaluate with `extraction_report()` diagnostics that catch schema gaps before they cost compute. Apply Dempster–Shafer belief calculus for single-claim verdicts. Run **Graph MCTS** for multi-hop connectivity with three critical design choices: (a) chain mass uses min/max conjunction, not Dempster across edges; (b) edges are grouped by triple so later retractions cancel earlier affirmations via conflict normalization; (c) connective-predicate filter at expansion time prevents graph-coincidental chains. Layer a **trained sheaf GNN** as a terminal scorer — 140k params, synthgen-trained, wired via `use_gnn=True` in `reason_connectivity`.

### Act VI — Productisation (Modules 14–15)

Turn the reasoner into a library. `extraction_report()` makes corpus-quality diagnostics automatic after every ingest. `bootstrap_gnn()` lets the Analyst peek at docs, design a synthgen config, and train a sheaf GNN on synthetic data matching the user's domain structure — one conversational turn, ~60 s on CPU. **Schema migration** (module 15) via `SchemaFunctor` rewrites existing cassettes under a new ontology without reingesting — 60× faster than re-extraction, old cassettes untouched (time-travel intact).

*After Act VI users give the system a corpus and an ontology and get a calibrated, self-diagnosing reasoner back.*

## Module Order

1. `01-domain-schema.md` — Define your domain's anchor vocabulary (the target basis)
2. `02-data-model.md` — Build the infon data model from situation semantics
3. `03-encoder.md` — The change of basis: SPLADE encoder + anchor projection
4. `04-extraction.md` — Document → sentence → SPLADE → infon pipeline (with dual-partition actor-as-object + word-order disambiguation)
5. `05-storage.md` — **Cassette substrate**: immutable byte files, Parquet indexes, manifest pruner, time-travel
6. `06-consolidation.md` — *Replaced by query-time primitives*: trajectory / next_edges / constraint computed from the index
7. `07-query-engine.md` — **DSL + reasoner**: `Query`, `store.ask`, `store.connect`, `store.any_of`
8. `08-agent-tools.md` — *Superseded by* `cognition.cassette.analyst`: 9-tool Strands Analyst
9. `09-cloud-deploy.md` — S3-native storage + Executor protocol + Lambda container deployment
10. `10-evaluation.md` — Precision/recall audits + the `extraction_report` four-category diagnostic
11. `11-category-theory.md` — Sheaf coherence (theory), **`SchemaFunctor` shipped** — see module 15
12. `12-dempster-shafer.md` — Mass functions + Dempster combination; min/max for chains (see module 13)
13. `13-graph-mcts.md` — MCTS + polarity-aware chain mass + sheaf GNN terminal scorer
14. `14-automl-loop.md` — *Focused as* `extraction_report` + `bootstrap_gnn`: corpus in, trained reasoner out
15. `15-schema-migration.md` — **Kan-based functorial migration**: rename / merge / delete without reingesting

## The Mathematical Stack

| Layer | Math | Implementation |
|---|---|---|
| Encoding | Change of basis (linear algebra) | SPLADE log-ReLU → AnchorProjector max-pool |
| Extraction | Situation semantics (infon theory) | `<<predicate, subject, object; polarity>>`, dual-partition role assignment |
| Storage | Content-addressing + append-only manifests | `sha256(text + schema_ref)` cassette IDs; snapshot chain |
| Indexing | Columnar over fsspec | Per-cassette Parquet shards (by_triple / by_time / by_anchor) + bbox pruner |
| Querying | DSL composition (immutable dataclass) | `Query.where/mentioning/between/contradicting/expand_hierarchy` |
| Aggregation | Read-time folds from indexes | `trajectory`, `next_edges`, `constraint` — no materialization |
| Belief | Dempster–Shafer theory | Per-infon mass functions → Dempster combination → verdict |
| Chain belief | Conjunctive logic | min(S), max(R) across hops (NOT Dempster — see 12/13 for why) |
| Reasoning | Monte Carlo Tree Search | UCB over chains; connective-predicate filter at expansion |
| Prior | Sheaf neural network | Per-relation-kind restriction maps + H¹ discrepancy feature |
| Training | Synthetic supervised (no human labels) | Synthgen labeled hypergraphs → joint chain-verdict + next-anchor |
| Migration | Category theory (functors) | `SchemaFunctor`: rename, merge, delete; `store.migrate` |
| Cloud | fsspec + Lambda container images | S3 URIs; boto3 build-push-deploy in Python |

## Tips

- Read each module's "What You'll Learn" section before starting.
- After each prompt, read the generated code and ask Claude to explain anything unclear.
- Customize the domain schema in Module 01 to your own domain.
- The system works end-to-end after Module 07 — Modules 08–15 are enrichments.
- Modules 11 + 15 cover the shipped category-theoretic tools: the sheaf GNN (in the reasoner) and Kan-based schema migration.
- Module 14 is the productisation capstone: two focused tools (`extraction_report` + `bootstrap_gnn`) rather than a full sklearn-style AutoML.
- The bundled splade-tiny model (17 MB) ships with the package — no download needed.

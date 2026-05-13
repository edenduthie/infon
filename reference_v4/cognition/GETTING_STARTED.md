# Getting started with cognition

A hypergraph-based reasoner that extracts calibrated claims from text,
answers questions about them, and tells you when it doesn't know.

This walkthrough runs end-to-end in **under 5 seconds** on a laptop CPU.

## Install

```bash
pip install torch transformers numpy
# Optional: ddgs (for cog.expand web search), hypothesis (for fuzz tests)
pip install ddgs hypothesis
```

Clone this repo and ensure `src/cognition/` is on your Python path.

## Five-step workflow

```python
from cognition import Cognition, CognitionConfig
from cognition.logic import HypergraphReasoner
import json, tempfile, os

# 1. Define the schema — the anchor vocabulary of your domain
SCHEMA = {
    "toyota":   {"type": "actor",    "tokens": ["toyota"]},
    "invests":  {"type": "relation", "tokens": ["invest", "invests"]},
    "battery":  {"type": "feature",  "tokens": ["battery", "batteries"]},
    # … plus more actors/relations/features/markets
}

with tempfile.TemporaryDirectory() as tmpdir:
    schema_path = os.path.join(tmpdir, "schema.json")
    with open(schema_path, "w") as f:
        json.dump(SCHEMA, f)

    # 2. Create the Cognition instance (loads SPLADE, optionally
    #    trains a per-schema SentenceEmbedder)
    cog = Cognition(CognitionConfig(
        schema_path=schema_path,
        db_path=os.path.join(tmpdir, "cog.db"),
        quality_threshold=0.05,
        max_triples_per_sentence=2,
        use_trained_embedder=True,
    ))

    # 3. Ingest documents
    cog.ingest([{"id": "d1",
                 "text": "Toyota invests in battery technology."}])
    cog.consolidate()

    # 4. Ask a question
    reasoner = HypergraphReasoner(
        cog.store, cog.encoder, cog.schema,
        hidden_dim=64, n_layers=2,
    )
    reasoner.builder.embedder = cog.embedder  # optional
    result = reasoner.reason("Did Toyota invest in batteries?")

    print(result.verdict)          # 'SUPPORTS'
    print(result.mass.supports)    # 0.62
    print(result.mass.theta)       # 0.30  ← residual ignorance

    # 5. Explore — which infons support the verdict?
    #    (Works best after refine() populates CAUSES edges)
    reasoner.refine(verbose=False)
    for cause in reasoner.root_cause(target_infon_id, top_k=3):
        print(f"{cause['subject']}/{cause['predicate']}/{cause['object']}")
```

## What the key knobs do

| Config | Default | What it controls |
|---|---|---|
| `schema_path` | — | JSON file with anchor definitions |
| `quality_threshold` | 0.05 | Joint-score floor for extracted triples |
| `max_triples_per_sentence` | 3 | Hard cap on triples from any sentence |
| `use_trained_embedder` | False | Auto-train a per-schema embedder on a synthetic corpus |
| `embedder_n_synth` | 1500 | Size of the synthetic training corpus |
| `embedder_trunk_dim` | 256 | Hidden size of the embedder |
| `coreference` | True | Resolve pronouns to actor names during extraction |

## Refreshing after new ingest

When new documents arrive, the cached reasoner becomes stale.
Call `refresh()` to rebuild.

```python
cog.ingest([{"id": "d_new",
             "text": "Tesla produces batteries."}])

# The cached reasoner is now stale — refresh it:
summary = cog.refresh(verbose=True)
# → {'rebuilt': True, 'elapsed_s': 0.17, 'n_infons': 2, ...}

# Subsequent queries reflect the new evidence
result = cog.reasoner().reason("Does Tesla produce batteries?")
```

## Active exploration with web search

When the system is uncertain (high θ), you can trigger an
external search to fetch more context, then re-query:

```python
# Uses ddgs (installed via pip install ddgs) — no API key required
result = cog.expand(
    query="Has Toyota partnered with any Chinese battery makers?",
    max_docs=5,
    theta_threshold=0.4,   # trigger search when θ > 0.4
    source="ddgs",          # or "mock" for tests
    verbose=True,
)
print(result["before"])  # θ before search
print(result["after"])   # θ after ingesting + re-querying
print(result["sources"])  # URLs of fetched results
```

## What you get

- **Extraction** of (subject, predicate, object) triples with tense, polarity, evidentiality, modality metadata
- **Dempster-Shafer belief masses** — every claim has (supports, refutes, uncertain, **θ**) with θ = "I don't know"
- **Graph reasoning** — typed hypergraph with IKL-style aggregators (AND, IF-THEN, IST)
- **Schema-driven interpretability** — every output points back to concrete anchors you defined
- **Nine pluggable heads** — next-anchor prediction, time-to-event, risk ranking, anomaly detection, counterfactual, root-cause, recommender, attribution, refutation

## What you don't get

Read [BENCHMARK.md](BENCHMARK.md) for scaling numbers and
[COMPARISON.md](COMPARISON.md) for an honest comparison against a
simulated LLM. Known limitations:

- **Scaling**: works well up to ~5,000 infons on a laptop; beyond that
  you'll want mini-batch GNN training (not shipped yet).
- **Schema is load-bearing**: the system only reasons over anchors you
  define. Missing a key concept = silently missing extractions.
- **Multilingual**: SPLADE is EN-primary; dependency extraction is
  EN-only; CJK paths work but under-tested.
- **Active exploration**: ingestion is batch, not streaming. No web
  search integration (yet).
- **Calibration depends on evidence mass**: on a tiny corpus, θ drifts
  toward zero even for claims the corpus doesn't actually support. The
  calibration gets better the more documents you ingest.

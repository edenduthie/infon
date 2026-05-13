"""Getting started with cognition in 5 steps.

Runs end-to-end in under a minute on a laptop CPU. Each section prints
what it is doing so you can read the output and understand the
pipeline without reading the source.

The example uses a small automotive-industry schema and three sample
documents — enough to demonstrate extraction, training, querying, and
explanation.
"""
from __future__ import annotations

import json
import os
import tempfile
import time


SCHEMA = {
    "toyota":    {"type": "actor",    "tokens": ["toyota"]},
    "honda":     {"type": "actor",    "tokens": ["honda"]},
    "tesla":     {"type": "actor",    "tokens": ["tesla"]},
    "panasonic": {"type": "actor",    "tokens": ["panasonic"]},
    "catl":      {"type": "actor",    "tokens": ["catl"]},
    "invests":   {"type": "relation", "tokens": ["invest", "invests"]},
    "partners":  {"type": "relation", "tokens": ["partner", "partners"]},
    "produces":  {"type": "relation", "tokens": ["produce", "produces"]},
    "delays":    {"type": "relation", "tokens": ["delay", "delays"]},
    "battery":   {"type": "feature",  "tokens": ["battery", "batteries"]},
    "ev":        {"type": "feature",  "tokens": ["ev", "evs"]},
    "factory":   {"type": "feature",  "tokens": ["factory", "factories"]},
    "japan":     {"type": "market",   "tokens": ["japan"]},
    "china":     {"type": "market",   "tokens": ["china"]},
}


DOCUMENTS = [
    {"id": "d1", "text":
        "Toyota invests heavily in solid-state battery technology. "
        "Toyota partners with Panasonic on battery development in Japan."
    },
    {"id": "d2", "text":
        "Tesla produces batteries at its Gigafactory. "
        "Tesla expands its battery factory in North America."
    },
    {"id": "d3", "text":
        "Honda delays its electric vehicle production timeline. "
        "Honda partners with CATL for battery supply in China."
    },
]


def main(verbose: bool = True) -> dict:
    """Run the 5-step workflow. Returns a summary dict.

    Steps:
        1. Define the schema (anchors and types)
        2. Create a Cognition instance (ingests + trains embedder)
        3. Ingest documents
        4. Ask a question — get a belief mass
        5. Explore: root-cause analysis and a counterfactual

    If verbose is True, prints progress. Returns a dict of results
    so calling code can verify everything worked.
    """
    from cognition import Cognition, CognitionConfig
    from cognition.logic import HypergraphReasoner

    def log(msg: str, *, section: bool = False) -> None:
        if verbose:
            if section:
                print(f"\n{'─' * 60}")
                print(msg)
                print("─" * 60)
            else:
                print(f"  {msg}")

    t_total_start = time.perf_counter()

    with tempfile.TemporaryDirectory() as tmpdir:
        # ── 1. Define the schema ─────────────────────────────────
        log("Step 1: Define the schema", section=True)
        schema_path = os.path.join(tmpdir, "schema.json")
        with open(schema_path, "w") as f:
            json.dump(SCHEMA, f, indent=2)
        log(f"schema written to {schema_path}")
        log(f"schema has {len(SCHEMA)} anchors across 4 types")

        # ── 2. Create Cognition instance ─────────────────────────
        # On first use, this trains a small embedder on a synthetic
        # corpus generated from the schema (takes a few seconds).
        # On subsequent runs with the same schema, it reuses the cache.
        log("Step 2: Create Cognition instance", section=True)
        t0 = time.perf_counter()
        cog = Cognition(CognitionConfig(
            schema_path=schema_path,
            db_path=os.path.join(tmpdir, "cognition.db"),
            activation_threshold=0.2,
            min_confidence=0.02,
            top_k_per_role=3,
            use_trained_embedder=True,
            embedder_n_synth=400,   # small for a fast demo
            embedder_epochs=10,
            embedder_trunk_dim=64,
        ))
        t_init = time.perf_counter() - t0
        log(f"Cognition ready in {t_init:.2f}s")
        log(f"trained embedder attached: "
            f"sparse_dim={cog.embedder.sparse_dim}, "
            f"n_anchors={cog.embedder.n_anchors}")

        # ── 3. Ingest documents ──────────────────────────────────
        log("Step 3: Ingest documents", section=True)
        t0 = time.perf_counter()
        n_infons = 0
        for doc in DOCUMENTS:
            n_infons += cog.ingest([doc])
        cog.consolidate()
        t_ingest = time.perf_counter() - t0
        log(f"ingested {len(DOCUMENTS)} documents → "
            f"{n_infons} infons in {t_ingest:.2f}s")
        log(f"corpus stats: {cog.stats()}")

        # ── 4. Ask a question ────────────────────────────────────
        log("Step 4: Ask a question", section=True)
        reasoner = HypergraphReasoner(
            cog.store, cog.encoder, cog.schema,
            hidden_dim=64, n_layers=2,
        )
        # Attach the trained embedder so the GNN builder uses it.
        reasoner.builder.embedder = cog.embedder

        query = "Did Toyota invest in battery technology?"
        log(f"query: {query!r}")
        t0 = time.perf_counter()
        result = reasoner.reason(query)
        t_query = time.perf_counter() - t0

        m = result.mass
        log(f"verdict: {result.verdict}")
        log(f"belief mass:")
        log(f"   supports:  {m.supports:.3f}")
        log(f"   refutes:   {m.refutes:.3f}")
        log(f"   uncertain: {m.uncertain:.3f}")
        log(f"   θ (unknown): {m.theta:.3f}")
        log(f"query latency: {t_query*1000:.1f}ms")

        # ── 5. Explore: root-cause analysis ──────────────────────
        log("Step 5: Explore — which infons support the verdict?",
            section=True)
        # Refine first so we have CAUSES/CONTRADICTS edges
        reasoner.refine(verbose=False)
        # Pick a concrete infon to explain
        target_id = None
        for iid in list(reasoner.builder.build(feature_dim=64)
                        .infon_map.keys()):
            inf = cog.store.get_infon(iid)
            if inf and inf.subject == "toyota":
                target_id = iid
                break
        causes = []
        if target_id is not None:
            tinf = cog.store.get_infon(target_id)
            log(f"target: {tinf.subject}/{tinf.predicate}/{tinf.object}")
            causes = reasoner.root_cause(target_id, top_k=3, verbose=False)
            if causes:
                for c in causes:
                    label = (f"{c['subject']}/{c['predicate']}"
                             f"/{c['object']}")
                    log(f"  {label:40s}  score={c['score']:.3f}")
            else:
                log("(no causal ancestors — small corpus)")

        t_total = time.perf_counter() - t_total_start
        log(f"\ntotal runtime: {t_total:.2f}s", section=False)
        log("all steps completed", section=False)

        summary = {
            "n_infons": n_infons,
            "verdict": result.verdict,
            "mass_sum": m.supports + m.refutes + m.uncertain + m.theta,
            "query_latency_ms": t_query * 1000,
            "init_time_s": t_init,
            "ingest_time_s": t_ingest,
            "total_time_s": t_total,
            "n_root_causes": len(causes),
        }
        cog.close()
        return summary


if __name__ == "__main__":
    summary = main(verbose=True)
    print("\nsummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

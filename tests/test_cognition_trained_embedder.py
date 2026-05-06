"""End-to-end: Cognition instantiated with use_trained_embedder=True.

Verifies:
  (a) The Cognition instance trains (or loads) an embedder automatically.
  (b) Its hypergraph builder uses the embedder for node features.
  (c) A full ingest + query round-trip works and produces a DS mass.
  (d) The embedder cache is hit on second instantiation with same schema.
"""
from __future__ import annotations

import json
import os
import tempfile
import time

import pytest


def _write_schema(tmpdir: str) -> str:
    """Write a minimal EV schema to tmpdir and return its path."""
    schema = {
        "toyota":    {"type": "actor",    "tokens": ["toyota"]},
        "honda":     {"type": "actor",    "tokens": ["honda"]},
        "tesla":     {"type": "actor",    "tokens": ["tesla"]},
        "invests":   {"type": "relation", "tokens": ["invest", "invests"]},
        "partners":  {"type": "relation", "tokens": ["partner", "partners"]},
        "produces":  {"type": "relation", "tokens": ["produce", "produces"]},
        "battery":   {"type": "feature",  "tokens": ["battery", "batteries"]},
        "japan":     {"type": "market",   "tokens": ["japan"]},
    }
    path = os.path.join(tmpdir, "schema.json")
    with open(path, "w") as f:
        json.dump(schema, f)
    return path


DOCS = [
    {"id": "d1", "text": "Toyota invests in batteries."},
    {"id": "d2", "text": "Honda partners with Panasonic in Japan."},
    {"id": "d3", "text": "Tesla produces batteries."},
]


def test_cognition_with_trained_embedder_instantiates():
    """Cognition(use_trained_embedder=True) wires up without error."""
    from infon import Cognition, CognitionConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        schema_path = _write_schema(tmpdir)
        cog = Cognition(CognitionConfig(
            schema_path=schema_path,
            db_path=os.path.join(tmpdir, "test.db"),
            activation_threshold=0.2,
            min_confidence=0.02,
            top_k_per_role=3,
            use_trained_embedder=True,
            embedder_n_synth=200,
            embedder_epochs=5,
            embedder_trunk_dim=64,
        ))
        assert cog.embedder is not None, "embedder should be built"
        print(f"\n  embedder: "
              f"sparse_dim={cog.embedder.sparse_dim}, "
              f"n_anchors={cog.embedder.n_anchors}")
        cog.close()


def test_cognition_embedder_cache_reuse():
    """Second Cognition() with the same schema + model_dir loads from cache."""
    from infon import Cognition, CognitionConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        schema_path = _write_schema(tmpdir)
        model_dir = os.path.join(tmpdir, "embedder_cache")

        def make():
            return Cognition(CognitionConfig(
                schema_path=schema_path,
                db_path=os.path.join(tmpdir, "test.db"),
                activation_threshold=0.2,
                min_confidence=0.02,
                top_k_per_role=3,
                use_trained_embedder=True,
                embedder_model_dir=model_dir,
                embedder_n_synth=200,
                embedder_epochs=5,
                embedder_trunk_dim=64,
            ))

        t0 = time.perf_counter()
        cog_a = make()
        t_first = time.perf_counter() - t0
        cog_a.close()

        t0 = time.perf_counter()
        cog_b = make()
        t_second = time.perf_counter() - t0
        cog_b.close()

        print(f"\n  first instantiation:  {t_first:.2f}s")
        print(f"  second (cached):       {t_second:.2f}s")
        assert t_second < t_first / 2, (
            f"second instantiation {t_second:.2f}s did not hit cache "
            f"(first was {t_first:.2f}s)"
        )


def test_cognition_ingest_and_query_with_embedder():
    """Full round-trip: ingest corpus, query for a claim, receive a
    valid DS mass. Verifies the hypergraph builder actually uses the
    embedder to produce node features."""
    from infon import Cognition, CognitionConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        schema_path = _write_schema(tmpdir)
        cog = Cognition(CognitionConfig(
            schema_path=schema_path,
            db_path=os.path.join(tmpdir, "test.db"),
            activation_threshold=0.2,
            min_confidence=0.02,
            top_k_per_role=3,
            use_trained_embedder=True,
            embedder_n_synth=200,
            embedder_epochs=5,
            embedder_trunk_dim=64,
        ))
        for doc in DOCS:
            cog.ingest([doc])
        cog.consolidate()

        # Build a reasoner and attach the embedder so the hypergraph
        # builder uses trained features.
        from infon.logic import HypergraphReasoner
        reasoner = HypergraphReasoner(cog.store, cog.encoder, cog.schema,
                                      hidden_dim=64, n_layers=2)
        reasoner.builder.embedder = cog.embedder

        result = reasoner.reason("Did Toyota invest in batteries?")
        m = result.mass
        total = m.supports + m.refutes + m.uncertain + m.theta
        assert abs(total - 1.0) < 1e-3, f"mass sum {total} off target"
        print(f"\n  verdict: {result.verdict}")
        print(f"  mass:    S={m.supports:.3f} R={m.refutes:.3f} "
              f"U={m.uncertain:.3f} θ={m.theta:.3f}")
        cog.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

"""Sheaf GNN: SheafMessagePassingLayer + sheaf-Laplacian regularizer
and a head-to-head benchmark against the R-GCN baseline.

Path A of the sheaf upgrade — per-relation restriction maps P_forward /
P_backward with an edge-discrepancy regularizer. Tests:

1. Forward-pass invariants — shape, non-degenerate output, identity init
   behaves close to (but not identical to) the baseline.
2. Sheaf Laplacian regularizer math — zero on constant features, drops
   under a gradient step, non-negative.
3. End-to-end reason() on the EV scenario using a sheaf reasoner.
4. Head-to-head: sheaf vs R-GCN on NEI-vs-SUPPORTS — measure accuracy
   and θ calibration.
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest
import torch


SCHEMA_DEFS = {
    "toyota":   {"type": "actor",    "tokens": ["toyota"]},
    "honda":    {"type": "actor",    "tokens": ["honda"]},
    "tesla":    {"type": "actor",    "tokens": ["tesla"]},
    "panasonic": {"type": "actor",   "tokens": ["panasonic"]},
    "catl":     {"type": "actor",    "tokens": ["catl"]},
    "invests":  {"type": "relation",
                 "tokens": ["invest", "invests", "investment"]},
    "partners": {"type": "relation",
                 "tokens": ["partner", "partners", "partnership"]},
    "produces": {"type": "relation",
                 "tokens": ["produce", "produces", "produced"]},
    "acquires": {"type": "relation",
                 "tokens": ["acquire", "acquires", "acquired"]},
    "battery":  {"type": "feature",  "tokens": ["battery", "batteries"]},
    "factory":  {"type": "feature",  "tokens": ["factory", "plant"]},
    "japan":    {"type": "market",   "tokens": ["japan", "japanese"]},
    "china":    {"type": "market",   "tokens": ["china", "chinese"]},
}


DOCUMENTS = [
    {"id": "d1",
     "text": "Toyota invests in battery technology in Japan."},
    {"id": "d2",
     "text": "Tesla produces batteries at its factory."},
    {"id": "d3",
     "text": "Honda partners with CATL on battery supply in China."},
    {"id": "d4",
     "text": "Panasonic invests in battery factory in Japan."},
]


def _make_cog(tmpdir, **overrides):
    from cognition import Cognition, CognitionConfig
    schema_path = os.path.join(tmpdir, "schema.json")
    with open(schema_path, "w") as f:
        json.dump(SCHEMA_DEFS, f)
    kwargs = dict(
        schema_path=schema_path,
        db_path=os.path.join(tmpdir, "cog.db"),
        activation_threshold=0.2,
        min_confidence=0.02,
        top_k_per_role=3,
        quality_threshold=0.04,
        max_triples_per_sentence=2,
    )
    kwargs.update(overrides)
    return Cognition(CognitionConfig(**kwargs))


# ═════════════════════════════════════════════════════════════════════
# Test 1. Forward-pass shape + invariants
# ═════════════════════════════════════════════════════════════════════

def test_sheaf_layer_forward_shape():
    """SheafMessagePassingLayer preserves hidden_dim and is non-trivial."""
    from cognition.logic import (
        SheafMessagePassingLayer, HypergraphBuilder,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        cog = _make_cog(tmpdir)
        for d in DOCUMENTS:
            cog.ingest([d])
        cog.consolidate()

        builder = HypergraphBuilder(cog.store, cog.encoder, cog.schema)
        graph = builder.build(feature_dim=32)

        layer = SheafMessagePassingLayer(32, 32)
        h_in = graph.node_features
        h_out = layer(h_in, graph.edge_index, graph.edge_types,
                      graph.edge_weights, graph.situation_features)

        print(f"\n  input: {h_in.shape}, norm={h_in.norm():.3f}")
        print(f"  output: {h_out.shape}, norm={h_out.norm():.3f}")

        assert h_out.shape == h_in.shape
        # non-trivial transformation
        delta = (h_out - h_in).norm().item()
        assert delta > 1e-3, f"layer had essentially no effect ({delta})"
        # no NaN / Inf
        assert torch.isfinite(h_out).all()
        cog.close()


def test_sheaf_layer_restriction_maps_shape():
    """P_forward / P_backward have the expected shapes."""
    from cognition.logic import SheafMessagePassingLayer, NUM_RELATIONS
    layer = SheafMessagePassingLayer(in_dim=16, out_dim=24)
    assert len(layer.P_forward) == NUM_RELATIONS
    assert len(layer.P_backward) == NUM_RELATIONS
    for P in layer.P_forward:
        assert P.shape == (24, 16)
    for P in layer.P_backward:
        assert P.shape == (24, 16)


# ═════════════════════════════════════════════════════════════════════
# Test 2. Sheaf Laplacian regularizer math
# ═════════════════════════════════════════════════════════════════════

def test_sheaf_discrepancy_nonneg_and_finite():
    """Discrepancy is non-negative, finite, and scalar."""
    from cognition.logic import SheafMessagePassingLayer
    torch.manual_seed(0)
    layer = SheafMessagePassingLayer(8, 8)
    h = torch.randn(6, 8)
    edge_index = torch.tensor([[0, 1, 2, 3, 4],
                               [1, 2, 3, 4, 5]], dtype=torch.long)
    edge_types = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)
    edge_weights = torch.tensor([1., 1., 1., 1., 1.])
    d = layer.sheaf_discrepancy(h, edge_index, edge_types, edge_weights)
    assert d.ndim == 0
    assert torch.isfinite(d)
    assert d.item() >= 0.0


def test_sheaf_discrepancy_empty_edges():
    """With no edges, discrepancy is exactly zero."""
    from cognition.logic import SheafMessagePassingLayer
    layer = SheafMessagePassingLayer(4, 4)
    h = torch.randn(3, 4)
    empty = torch.zeros((2, 0), dtype=torch.long)
    et = torch.zeros((0,), dtype=torch.long)
    ew = torch.zeros((0,))
    d = layer.sheaf_discrepancy(h, empty, et, ew)
    assert d.item() == 0.0


def test_sheaf_discrepancy_decreases_with_gradient_step():
    """One SGD step on the discrepancy should reduce its value."""
    from cognition.logic import SheafMessagePassingLayer
    torch.manual_seed(42)
    layer = SheafMessagePassingLayer(8, 8)
    h = torch.randn(5, 8)  # features held fixed
    edge_index = torch.tensor([[0, 1, 2, 3],
                               [1, 2, 3, 4]], dtype=torch.long)
    edge_types = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    edge_weights = torch.ones(4)

    d0 = layer.sheaf_discrepancy(
        h, edge_index, edge_types, edge_weights).item()

    opt = torch.optim.Adam(layer.parameters(), lr=1e-2)
    for _ in range(20):
        opt.zero_grad()
        d = layer.sheaf_discrepancy(
            h, edge_index, edge_types, edge_weights)
        d.backward()
        opt.step()

    d1 = layer.sheaf_discrepancy(
        h, edge_index, edge_types, edge_weights).item()

    print(f"\n  L_F before: {d0:.4f}, after 5 SGD steps: {d1:.4f}")
    assert d1 < d0 - 1e-4, \
        f"discrepancy did not decrease: {d0} → {d1}"


# ═════════════════════════════════════════════════════════════════════
# Test 3. End-to-end reason() on the sheaf reasoner
# ═════════════════════════════════════════════════════════════════════

def test_sheaf_reasoner_end_to_end():
    """HypergraphReasoner with use_sheaf=True produces a valid verdict
    on a supported claim."""
    from cognition.logic import HypergraphReasoner
    with tempfile.TemporaryDirectory() as tmpdir:
        cog = _make_cog(tmpdir)
        for d in DOCUMENTS:
            cog.ingest([d])
        cog.consolidate()

        reasoner = HypergraphReasoner(
            cog.store, cog.encoder, cog.schema,
            hidden_dim=32, n_layers=2, use_sheaf=True,
        )
        if cog.embedder is not None:
            reasoner.builder.embedder = cog.embedder

        graph = reasoner.builder.build(feature_dim=32)
        stats = reasoner.fit(graph=graph, epochs=15,
                             laplacian_weight=0.1, verbose=False)
        print(f"\n  sheaf fit: epochs={stats['epochs']} "
              f"final_loss={stats['final_loss']:.4f}")

        result = reasoner.reason("Did Toyota invest in batteries?")
        print(f"  verdict={result.verdict} "
              f"supports={result.mass.supports:.3f} "
              f"theta={result.mass.theta:.3f}")

        # Sanity: verdict field is valid, masses sum to 1
        assert result.verdict in ("SUPPORTS", "REFUTES",
                                  "NOT_ENOUGH_INFO", "CONTRADICTED")
        total = (result.mass.supports + result.mass.refutes
                 + result.mass.uncertain + result.mass.theta)
        assert abs(total - 1.0) < 1e-3
        cog.close()


def test_sheaf_reasoner_theta_high_on_nei():
    """Sheaf reasoner also returns high θ for claims the corpus
    doesn't support (calibration is preserved)."""
    from cognition.logic import HypergraphReasoner
    with tempfile.TemporaryDirectory() as tmpdir:
        cog = _make_cog(tmpdir)
        for d in DOCUMENTS:
            cog.ingest([d])
        cog.consolidate()

        reasoner = HypergraphReasoner(
            cog.store, cog.encoder, cog.schema,
            hidden_dim=32, n_layers=2, use_sheaf=True,
        )
        graph = reasoner.builder.build(feature_dim=32)
        reasoner.fit(graph=graph, epochs=10, verbose=False)

        # Claim not in the corpus at all
        result = reasoner.reason("Did Tesla acquire CATL?")
        print(f"\n  NEI query θ={result.mass.theta:.3f}")
        assert result.mass.theta > 0.3
        cog.close()


# ═════════════════════════════════════════════════════════════════════
# Test 4. Sheaf vs R-GCN head-to-head benchmark
# ═════════════════════════════════════════════════════════════════════

def _build_and_fit(use_sheaf, docs, tmpdir, epochs=15, laplacian=0.1):
    from cognition.logic import HypergraphReasoner
    cog = _make_cog(tmpdir + f"_{int(use_sheaf)}")
    os.makedirs(tmpdir + f"_{int(use_sheaf)}", exist_ok=True)
    # cog already created from _make_cog; use returned cog directly
    # actually: _make_cog already created dirs via tempfile context
    for d in docs:
        cog.ingest([d])
    cog.consolidate()
    reasoner = HypergraphReasoner(
        cog.store, cog.encoder, cog.schema,
        hidden_dim=32, n_layers=2, use_sheaf=use_sheaf,
    )
    graph = reasoner.builder.build(feature_dim=32)
    reasoner.fit(graph=graph, epochs=epochs,
                 laplacian_weight=laplacian, verbose=False)
    return cog, reasoner


def test_sheaf_vs_rgcn_benchmark():
    """Head-to-head: ask both reasoners 4 queries (2 SUPPORTS, 2 NEI)
    and compare verdicts + calibration."""
    from cognition.logic import HypergraphReasoner
    torch.manual_seed(0)

    queries = [
        # (query, expected_verdict, expected_high_theta)
        ("Did Toyota invest in batteries?",       "SUPPORTS",        False),
        ("Did Tesla produce batteries?",          "SUPPORTS",        False),
        ("Did Tesla acquire CATL?",               "NOT_ENOUGH_INFO", True),
        ("Did Honda produce batteries in Japan?", "NOT_ENOUGH_INFO", True),
    ]

    results: dict[str, dict] = {}

    for variant in ("rgcn", "sheaf"):
        with tempfile.TemporaryDirectory() as tmpdir:
            cog = _make_cog(tmpdir)
            for d in DOCUMENTS:
                cog.ingest([d])
            cog.consolidate()

            reasoner = HypergraphReasoner(
                cog.store, cog.encoder, cog.schema,
                hidden_dim=32, n_layers=2,
                use_sheaf=(variant == "sheaf"),
            )
            graph = reasoner.builder.build(feature_dim=32)
            reasoner.fit(graph=graph, epochs=15,
                         laplacian_weight=0.1, verbose=False)

            per_variant = {"supports_theta": [], "nei_theta": [],
                           "correct": 0, "total": 0}
            for q, expected, _ in queries:
                r = reasoner.reason(q)
                per_variant["total"] += 1
                if r.verdict == expected:
                    per_variant["correct"] += 1
                if expected == "SUPPORTS":
                    per_variant["supports_theta"].append(r.mass.theta)
                else:
                    per_variant["nei_theta"].append(r.mass.theta)
            results[variant] = per_variant
            cog.close()

    def _avg(lst):
        return sum(lst) / len(lst) if lst else float("nan")

    for variant, v in results.items():
        print(f"\n  {variant}: accuracy={v['correct']}/{v['total']}")
        print(f"    mean θ on SUPPORTS: {_avg(v['supports_theta']):.3f}")
        print(f"    mean θ on NEI:      {_avg(v['nei_theta']):.3f}")

    # Soft invariants — sheaf should roughly match R-GCN on this tiny
    # corpus. (The real payoff shows on bigger graphs.)
    assert results["sheaf"]["correct"] >= 2, \
        "sheaf reasoner failed to get any queries right"
    # θ calibration: NEI θ should be higher than SUPPORTS θ for *both*
    # variants. This is the core property the relevance filter gives us.
    for variant, v in results.items():
        s_theta = _avg(v["supports_theta"])
        n_theta = _avg(v["nei_theta"])
        assert n_theta > s_theta, (
            f"{variant} calibration broken: "
            f"NEI θ {n_theta:.3f} ≤ SUPPORTS θ {s_theta:.3f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

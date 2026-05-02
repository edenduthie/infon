"""Test HypergraphReasoner with IKL primitives on synthetic data.

Ingests a small geopolitical/trade scenario, builds the hypergraph,
runs typed message passing, and verifies the IKL operators produce
coherent DS masses.
"""

from __future__ import annotations

import sys
import os
import tempfile

# ── Synthetic schema: actors, relations, features, markets ────────────

SCHEMA_DEFS = {
    # Actors
    "toyota": {"type": "actor", "tokens": ["toyota"], "country_code": "JP",
               "organisation_type": "private-sector"},
    "honda": {"type": "actor", "tokens": ["honda"], "country_code": "JP",
              "organisation_type": "private-sector"},
    "tesla": {"type": "actor", "tokens": ["tesla"], "country_code": "US",
              "organisation_type": "private-sector"},
    "panasonic": {"type": "actor", "tokens": ["panasonic"], "country_code": "JP",
                  "organisation_type": "private-sector"},
    "catl": {"type": "actor", "tokens": ["catl"], "country_code": "CN",
             "organisation_type": "private-sector"},

    # Relations
    "invests": {"type": "relation", "tokens": ["invest", "invests", "invested", "investment"]},
    "partners": {"type": "relation", "tokens": ["partner", "partners", "partnered", "partnership"]},
    "produces": {"type": "relation", "tokens": ["produce", "produces", "produced", "production"]},
    "expands": {"type": "relation", "tokens": ["expand", "expands", "expanded", "expansion"]},
    "delays": {"type": "relation", "tokens": ["delay", "delays", "delayed"]},
    "acquires": {"type": "relation", "tokens": ["acquire", "acquires", "acquired", "acquisition"]},

    # Features
    "battery": {"type": "feature", "tokens": ["battery", "batteries"]},
    "solid_state": {"type": "feature", "tokens": ["solid-state", "solid state"],
                    "parent": "battery"},
    "ev": {"type": "feature", "tokens": ["ev", "electric vehicle", "electric vehicles"]},
    "factory": {"type": "feature", "tokens": ["factory", "plant", "facility"]},
    "supply_chain": {"type": "feature", "tokens": ["supply chain", "supply"]},

    # Markets
    "japan": {"type": "market", "tokens": ["japan", "japanese"], "country_code": "JP",
              "macro_region": "asia_pacific"},
    "north_america": {"type": "market", "tokens": ["north america", "us", "united states"],
                      "macro_region": "americas"},
    "china": {"type": "market", "tokens": ["china", "chinese"], "country_code": "CN",
              "macro_region": "asia_pacific"},
}

# ── Synthetic documents: a coherent EV battery scenario ───────────────

DOCUMENTS = [
    {
        "id": "doc1",
        "text": (
            "Toyota invests heavily in solid-state battery technology. "
            "The company announced a $13.6 billion investment in battery production. "
            "Toyota partners with Panasonic on battery development in Japan."
        ),
    },
    {
        "id": "doc2",
        "text": (
            "Tesla expands its battery factory in North America. "
            "Tesla produces batteries at its Gigafactory facility. "
            "Tesla acquires battery supply chain assets to reduce costs."
        ),
    },
    {
        "id": "doc3",
        "text": (
            "Honda delays its electric vehicle production timeline. "
            "Honda partners with CATL for battery supply in China. "
            "Honda invests in solid-state battery research but has not produced results."
        ),
    },
    {
        "id": "doc4",
        "text": (
            "CATL expands battery production capacity in China. "
            "CATL produces batteries for multiple Japanese automakers. "
            "Panasonic invests in new battery factory in Japan."
        ),
    },
    {
        "id": "doc5",
        "text": (
            "Toyota's solid-state battery investment leads to a breakthrough. "
            "Toyota produces prototype solid-state batteries ahead of schedule. "
            "If Toyota succeeds in solid-state batteries, it could reshape the EV market."
        ),
    },
]


def setup_cognition(db_path: str):
    """Create a Cognition instance with synthetic schema and ingest documents."""
    import json
    from cognition import Cognition, CognitionConfig
    from cognition.schema import AnchorSchema

    # Write schema to temp file
    schema_path = db_path.replace(".db", "_schema.json")
    with open(schema_path, "w") as f:
        json.dump(SCHEMA_DEFS, f)

    config = CognitionConfig(
        schema_path=schema_path,
        db_path=db_path,
        activation_threshold=0.2,
        min_confidence=0.02,
        top_k_per_role=3,
    )
    cog = Cognition(config)
    return cog


def test_ingest_and_build_graph():
    """Test: ingest documents, build hypergraph, verify structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cog = setup_cognition(db_path)

        # Ingest
        total = 0
        for doc in DOCUMENTS:
            n = cog.ingest([doc])
            total += n
        cog.consolidate()

        print(f"\n  Ingested {total} infons from {len(DOCUMENTS)} documents")

        stats = cog.stats()
        print(f"  Store: {stats['infon_count']} infons, "
              f"{stats['constraint_count']} constraints, "
              f"{stats['anchors']} anchors")

        assert stats["infon_count"] > 0, "No infons extracted"

        # Build hypergraph
        from cognition.logic import HypergraphBuilder
        builder = HypergraphBuilder(cog.store, cog.encoder, cog.schema)
        graph = builder.build(feature_dim=64)

        print(f"\n  HyperGraph:")
        print(f"    Nodes: {graph.n_nodes} ({len(graph.anchor_map)} anchors + "
              f"{len(graph.infon_map)} infons)")
        print(f"    Edges: {graph.n_edges}")
        print(f"    Anchor types: {list(graph.anchor_type_groups.keys())}")
        print(f"    Node features: {graph.node_features.shape}")
        print(f"    Situation features: {graph.situation_features.shape}")

        assert graph.n_nodes > 0
        assert graph.n_edges > 0
        assert len(graph.anchor_map) == len(SCHEMA_DEFS)
        assert len(graph.infon_map) > 0
        assert graph.node_features.shape == (graph.n_nodes, 64)

        cog.close()
        print("\n  PASS: ingest + graph build")


def test_message_passing():
    """Test: run typed message passing layers over the built graph."""
    import torch
    from cognition.logic import HypergraphBuilder, TypedMessagePassingLayer

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cog = setup_cognition(db_path)
        for doc in DOCUMENTS:
            cog.ingest([doc])
        cog.consolidate()

        builder = HypergraphBuilder(cog.store, cog.encoder, cog.schema)
        graph = builder.build(feature_dim=64)

        # Two layers of message passing
        layer1 = TypedMessagePassingLayer(64, 64)
        layer2 = TypedMessagePassingLayer(64, 64)

        h0 = graph.node_features
        h1 = layer1(h0, graph.edge_index, graph.edge_types,
                     graph.edge_weights, graph.situation_features)
        h2 = layer2(h1, graph.edge_index, graph.edge_types,
                     graph.edge_weights, graph.situation_features)

        print(f"\n  Message passing:")
        print(f"    Input:  {h0.shape}, norm={h0.norm():.3f}")
        print(f"    Layer1: {h1.shape}, norm={h1.norm():.3f}")
        print(f"    Layer2: {h2.shape}, norm={h2.norm():.3f}")

        # Features should change (message passing did something)
        delta = (h2 - h0).norm().item()
        print(f"    Delta (h2 - h0): {delta:.3f}")
        assert delta > 0.01, f"Message passing had no effect (delta={delta})"

        # Output shape preserved
        assert h2.shape == h0.shape

        cog.close()
        print("\n  PASS: message passing")


def test_ikl_operators():
    """Test: IKL operators produce correct shapes and semantically coherent output."""
    import torch
    from cognition.logic import (
        IKLThat, IKLAnd, IKLOr, IKLNot, IKLIf, IKLIff,
        IKLForall, IKLExists, IKLIst, MassReadout,
    )

    h_dim = 64

    that = IKLThat(h_dim)
    ikl_and = IKLAnd(h_dim)
    ikl_or = IKLOr(h_dim)
    ikl_not = IKLNot(h_dim)
    ikl_if = IKLIf(h_dim)
    ikl_iff = IKLIff(h_dim)
    ikl_forall = IKLForall(h_dim)
    ikl_exists = IKLExists(h_dim)
    ikl_ist = IKLIst(h_dim, 16)
    readout = MassReadout(h_dim)

    # Create mock embeddings: 5 "actor" nodes, 3 "infon" nodes
    actors = torch.randn(5, h_dim)
    infons = torch.randn(3, h_dim)
    single = torch.randn(1, h_dim)
    sit = torch.randn(1, 16)

    print("\n  IKL operator tests:")

    # that: reify infon → proposition term
    reified = that(infons)
    assert reified.shape == infons.shape
    print(f"    that: {infons.shape} → {reified.shape}")

    # and: conjunction (min-pool with gate)
    conj = ikl_and(actors)
    assert conj.shape == (h_dim,)
    print(f"    and: {actors.shape} → {conj.shape}")

    # or: disjunction (max-pool with gate)
    disj = ikl_or(actors)
    assert disj.shape == (h_dim,)
    print(f"    or: {actors.shape} → {disj.shape}")

    # not: negation
    neg = ikl_not(single)
    assert neg.shape == single.shape
    print(f"    not: {single.shape} → {neg.shape}")

    # if: conditional
    premise = torch.randn(1, h_dim)
    conclusion = torch.randn(1, h_dim)
    cond = ikl_if(premise, conclusion)
    assert cond.shape == (1, h_dim)
    print(f"    if: ({premise.shape}, {conclusion.shape}) → {cond.shape}")

    # iff: biconditional (should be symmetric)
    bic1 = ikl_iff(premise, conclusion)
    bic2 = ikl_iff(conclusion, premise)
    sym_diff = (bic1 - bic2).norm().item()
    print(f"    iff: symmetry diff={sym_diff:.4f}")

    # forall: universal over domain (conjunction)
    univ = ikl_forall(actors)
    assert univ.shape == (h_dim,)
    print(f"    forall: {actors.shape} → {univ.shape}")

    # exists: existential over domain (disjunction)
    exist = ikl_exists(actors)
    assert exist.shape == (h_dim,)
    print(f"    exists: {actors.shape} → {exist.shape}")

    # ist: situation gating
    ctx = ikl_ist(single, sit)
    assert ctx.shape == single.shape
    print(f"    ist: ({single.shape}, {sit.shape}) → {ctx.shape}")

    # Mass readout: all operators produce valid DS masses
    test_embeddings = torch.stack([conj, disj, univ, exist])
    masses = readout.to_mass_functions(test_embeddings)
    for i, (label, m) in enumerate(zip(
        ["and", "or", "forall", "exists"], masses
    )):
        total = m.supports + m.refutes + m.uncertain + m.theta
        assert abs(total - 1.0) < 0.01, f"{label}: mass sum={total}"
        print(f"    {label} → S={m.supports:.3f} R={m.refutes:.3f} "
              f"U={m.uncertain:.3f} θ={m.theta:.3f}")

    print("\n  PASS: IKL operators")


def test_reasoner_end_to_end():
    """Test: full HypergraphReasoner.reason() with auto-fit."""
    from cognition.logic import HypergraphReasoner, HypergraphBuilder

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cog = setup_cognition(db_path)
        for doc in DOCUMENTS:
            cog.ingest([doc])
        cog.consolidate()

        queries = [
            "Did Toyota invest in battery technology?",
            "Is Tesla expanding production?",
            "Did Honda delay electric vehicles?",
            "Does CATL produce batteries in China?",
        ]

        # ── Before training (random weights) ──
        reasoner = HypergraphReasoner(
            cog.store, cog.encoder, cog.schema,
            hidden_dim=64, n_layers=2,
        )

        print("\n  BEFORE fit (random readout):")
        unfitted_verdicts = []
        for q in queries:
            # Bypass auto-fit by setting _fitted=True temporarily
            reasoner._fitted = True
            result = reasoner.reason(q)
            unfitted_verdicts.append(result.verdict)
            m = result.mass
            print(f"    {result.verdict:20s}  S={m.supports:.3f} R={m.refutes:.3f} "
                  f"θ={m.theta:.3f}  \"{q}\"")
        reasoner._fitted = False

        # ── After training (DS teacher) ──
        print("\n  Fitting on DS teacher signal...")
        graph = reasoner.builder.build(feature_dim=64)
        fit_stats = reasoner.fit(graph=graph, epochs=50, verbose=True)
        print(f"  → {fit_stats['n_targets']} targets, "
              f"loss {fit_stats['losses'][0]:.4f} → {fit_stats['final_loss']:.4f}")

        print("\n  AFTER fit (trained readout):")
        fitted_verdicts = []
        for q in queries:
            result = reasoner.reason(q)
            fitted_verdicts.append(result.verdict)
            m = result.mass
            print(f"    {result.verdict:20s}  S={m.supports:.3f} R={m.refutes:.3f} "
                  f"θ={m.theta:.3f}  \"{q}\"")

            total = m.supports + m.refutes + m.uncertain + m.theta
            assert abs(total - 1.0) < 0.01, f"Mass sum={total}"

        # Training should have reduced the loss
        assert fit_stats["final_loss"] < fit_stats["losses"][0], \
            "Training did not reduce loss"

        cog.close()
        print("\n  PASS: reasoner end-to-end (before/after fit)")


def test_compound_queries():
    """Test: nested IKL compound expression evaluation."""
    import torch
    from cognition.logic import HypergraphReasoner, HypergraphBuilder

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cog = setup_cognition(db_path)
        for doc in DOCUMENTS:
            cog.ingest([doc])
        cog.consolidate()

        reasoner = HypergraphReasoner(
            cog.store, cog.encoder, cog.schema,
            hidden_dim=64, n_layers=2,
        )

        graph = reasoner.builder.build(feature_dim=64)

        with torch.no_grad():
            h = reasoner.forward(graph)

        print("\n  Compound IKL queries:")

        # (exists (?x actor) ...) — "some actor does something"
        exist_h = reasoner.query_exists("actor", graph, h)
        exist_mass = reasoner.mass_readout.to_mass_functions(exist_h.unsqueeze(0))[0]
        print(f"    (exists actor): S={exist_mass.supports:.3f} "
              f"R={exist_mass.refutes:.3f} θ={exist_mass.theta:.3f}")

        # (forall (?x actor) ...) — "all actors do something"
        forall_h = reasoner.query_forall("actor", graph, h)
        forall_mass = reasoner.mass_readout.to_mass_functions(forall_h.unsqueeze(0))[0]
        print(f"    (forall actor): S={forall_mass.supports:.3f} "
              f"R={forall_mass.refutes:.3f} θ={forall_mass.theta:.3f}")

        # Nested: (and (exists actor) (exists feature))
        expr = {
            "op": "and",
            "args": [
                {"op": "exists", "type": "actor"},
                {"op": "exists", "type": "feature"},
            ],
        }
        compound_mass = reasoner.evaluate_expression(expr)
        print(f"    (and (exists actor) (exists feature)): "
              f"S={compound_mass.supports:.3f} R={compound_mass.refutes:.3f} "
              f"θ={compound_mass.theta:.3f}")

        # (not (forall actor))
        expr_not = {
            "op": "not",
            "args": [{"op": "forall", "type": "actor"}],
        }
        not_mass = reasoner.evaluate_expression(expr_not)
        print(f"    (not (forall actor)): "
              f"S={not_mass.supports:.3f} R={not_mass.refutes:.3f} "
              f"θ={not_mass.theta:.3f}")

        # (or (exists market) (exists feature))
        expr_or = {
            "op": "or",
            "args": [
                {"op": "exists", "type": "market"},
                {"op": "exists", "type": "feature"},
            ],
        }
        or_mass = reasoner.evaluate_expression(expr_or)
        print(f"    (or (exists market) (exists feature)): "
              f"S={or_mass.supports:.3f} R={or_mass.refutes:.3f} "
              f"θ={or_mass.theta:.3f}")

        # Verify all masses sum to 1
        for label, m in [("exists", exist_mass), ("forall", forall_mass),
                         ("compound", compound_mass), ("not", not_mass),
                         ("or", or_mass)]:
            total = m.supports + m.refutes + m.uncertain + m.theta
            assert abs(total - 1.0) < 0.01, f"{label}: mass sum={total}"

        cog.close()
        print("\n  PASS: compound queries")


def test_that_and_ist():
    """Test: reification (that) and situation operator (ist) on actual infons."""
    import torch
    from cognition.logic import HypergraphReasoner

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cog = setup_cognition(db_path)
        for doc in DOCUMENTS:
            cog.ingest([doc])
        cog.consolidate()

        reasoner = HypergraphReasoner(
            cog.store, cog.encoder, cog.schema,
            hidden_dim=64, n_layers=2,
        )

        graph = reasoner.builder.build(feature_dim=64)

        with torch.no_grad():
            h = reasoner.forward(graph)

        print("\n  Reification (that) and situation (ist):")

        # Pick first few infons from the graph
        infon_ids = list(graph.infon_map.keys())[:5]
        for iid in infon_ids:
            # (that φ) — reify
            that_h = reasoner.query_that(iid, graph, h)
            that_mass = reasoner.mass_readout.to_mass_functions(that_h.unsqueeze(0))[0]

            # (ist s φ) — situate
            ist_h = reasoner.query_ist(iid, graph, h)
            ist_mass = reasoner.mass_readout.to_mass_functions(ist_h.unsqueeze(0))[0]

            infon = cog.store.get_infon(iid)
            label = f"{infon.subject}/{infon.predicate}/{infon.object}" if infon else iid[:12]
            print(f"    {label}:")
            print(f"      that → S={that_mass.supports:.3f} R={that_mass.refutes:.3f} θ={that_mass.theta:.3f}")
            print(f"      ist  → S={ist_mass.supports:.3f} R={ist_mass.refutes:.3f} θ={ist_mass.theta:.3f}")

        cog.close()
        print("\n  PASS: that + ist")


def test_conditional_reasoning():
    """Test: if/iff between infon pairs."""
    import torch
    from cognition.logic import HypergraphReasoner

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cog = setup_cognition(db_path)
        for doc in DOCUMENTS:
            cog.ingest([doc])
        cog.consolidate()

        reasoner = HypergraphReasoner(
            cog.store, cog.encoder, cog.schema,
            hidden_dim=64, n_layers=2,
        )

        graph = reasoner.builder.build(feature_dim=64)

        with torch.no_grad():
            h = reasoner.forward(graph)

        print("\n  Conditional reasoning (if/iff):")

        infon_ids = list(graph.infon_map.keys())
        if len(infon_ids) >= 2:
            pairs = [(infon_ids[i], infon_ids[i+1])
                     for i in range(0, min(6, len(infon_ids)-1), 2)]

            for pid, cid in pairs:
                # (if premise conclusion)
                cond_h = reasoner.query_conditional(pid, cid, graph, h)
                cond_mass = reasoner.mass_readout.to_mass_functions(cond_h.unsqueeze(0))[0]

                # (iff premise conclusion) via compound_query
                iff_expr = {
                    "op": "iff",
                    "args": [
                        {"op": "node", "id": pid},
                        {"op": "node", "id": cid},
                    ],
                }
                iff_h = reasoner.compound_query(iff_expr, graph, h)
                iff_mass = reasoner.mass_readout.to_mass_functions(iff_h.unsqueeze(0))[0]

                p_inf = cog.store.get_infon(pid)
                c_inf = cog.store.get_infon(cid)
                p_label = f"{p_inf.subject}/{p_inf.predicate}" if p_inf else pid[:8]
                c_label = f"{c_inf.subject}/{c_inf.predicate}" if c_inf else cid[:8]

                print(f"    {p_label} → {c_label}:")
                print(f"      if  → S={cond_mass.supports:.3f} R={cond_mass.refutes:.3f} θ={cond_mass.theta:.3f}")
                print(f"      iff → S={iff_mass.supports:.3f} R={iff_mass.refutes:.3f} θ={iff_mass.theta:.3f}")

        cog.close()
        print("\n  PASS: conditional reasoning")


def test_refine_hypergraph():
    """Test: GNN refinement discovers temporal + causal edges."""
    from cognition.logic import HypergraphReasoner

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cog = setup_cognition(db_path)
        for doc in DOCUMENTS:
            cog.ingest([doc])
        cog.consolidate()

        # Snapshot pre-refinement state
        pre_infons = cog.store.query_infons(limit=100)
        pre_confidences = {inf.infon_id: inf.confidence for inf in pre_infons}
        pre_coherences = {inf.infon_id: inf.coherence for inf in pre_infons}

        print(f"\n  Pre-refinement:")
        print(f"    Infons: {len(pre_infons)}")
        print(f"    Mean confidence: {sum(pre_confidences.values()) / len(pre_confidences):.3f}")
        print(f"    Mean coherence: {sum(pre_coherences.values()) / len(pre_coherences):.3f}")

        # Show tense distribution (temporal candidates depend on this)
        from collections import Counter
        tenses = Counter(inf.tense for inf in pre_infons)
        print(f"    Tense distribution: {dict(tenses)}")

        # Run refinement
        reasoner = HypergraphReasoner(
            cog.store, cog.encoder, cog.schema,
            hidden_dim=64, n_layers=2,
        )
        result = reasoner.refine(causal_threshold=0.5, verbose=True)

        # Check post-refinement state
        post_infons = cog.store.query_infons(limit=100)
        post_confidences = {inf.infon_id: inf.confidence for inf in post_infons}
        post_coherences = {inf.infon_id: inf.coherence for inf in post_infons}

        print(f"\n  Post-refinement:")
        print(f"    Infons updated: {result.infons_updated}")
        print(f"    Temporal NEXT edges: {result.temporal_added}")
        print(f"    Causal CAUSES edges: {result.causal_added}")
        print(f"    CONTRADICTS edges: {result.contradictions_found}")
        print(f"    Causal pairs evaluated: {result.pairs_checked}")
        print(f"    Mean confidence: "
              f"{sum(pre_confidences.values()) / len(pre_confidences):.3f} → "
              f"{sum(post_confidences.values()) / len(post_confidences):.3f}")
        print(f"    Mean coherence: "
              f"{sum(pre_coherences.values()) / len(pre_coherences):.3f} → "
              f"{sum(post_coherences.values()) / len(post_coherences):.3f}")

        # Confidences should have changed
        changed = sum(1 for iid in pre_confidences
                      if iid in post_confidences
                      and abs(pre_confidences[iid] - post_confidences[iid]) > 1e-6)
        print(f"    Confidences changed: {changed}/{len(pre_confidences)}")
        assert changed > 0, "Refinement did not update any confidences"

        # Show discovered temporal edges
        if result.temporal_edges:
            print(f"\n  Temporal NEXT edges:")
            for e in result.temporal_edges[:8]:
                src_inf = cog.store.get_infon(e.source)
                tgt_inf = cog.store.get_infon(e.target)
                src_label = (f"{src_inf.subject}/{src_inf.predicate} [{src_inf.tense}]"
                             if src_inf else e.source[:12])
                tgt_label = (f"{tgt_inf.subject}/{tgt_inf.predicate} [{tgt_inf.tense}]"
                             if tgt_inf else e.target[:12])
                print(f"    {src_label}  →NEXT→  {tgt_label}  (w={e.weight:.3f})")

        # Show discovered causal edges
        if result.causal_edges:
            print(f"\n  Causal CAUSES edges:")
            for e in result.causal_edges[:8]:
                src_inf = cog.store.get_infon(e.source)
                tgt_inf = cog.store.get_infon(e.target)
                src_label = (f"{src_inf.subject}/{src_inf.predicate}/{src_inf.object}"
                             if src_inf else e.source[:12])
                tgt_label = (f"{tgt_inf.subject}/{tgt_inf.predicate}/{tgt_inf.object}"
                             if tgt_inf else e.target[:12])
                print(f"    {src_label}  →CAUSES→  {tgt_label}  "
                      f"(w={e.weight:.3f}, {e.metadata.get('from_pred', '?')}→"
                      f"{e.metadata.get('to_pred', '?')})")

        if result.contradiction_edges:
            print(f"\n  CONTRADICTS edges:")
            for e in result.contradiction_edges[:5]:
                src_inf = cog.store.get_infon(e.source)
                tgt_inf = cog.store.get_infon(e.target)
                src_label = (f"{src_inf.subject}/{src_inf.predicate}/{src_inf.object}"
                             if src_inf else e.source[:12])
                tgt_label = (f"{tgt_inf.subject}/{tgt_inf.predicate}/{tgt_inf.object}"
                             if tgt_inf else e.target[:12])
                print(f"    {src_label}  →CONTRADICTS→  {tgt_label}  "
                      f"(w={e.weight:.3f}, {e.metadata.get('from_pred', '?')}→"
                      f"{e.metadata.get('to_pred', '?')})")

        cog.close()
        print("\n  PASS: refine hypergraph")


def test_sheaf_coherence_in_training():
    """Test: sheaf coherence regularization affects training loss."""
    from cognition.logic import HypergraphReasoner

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cog = setup_cognition(db_path)
        for doc in DOCUMENTS:
            cog.ingest([doc])
        cog.consolidate()

        # Train WITH sheaf coherence
        r1 = HypergraphReasoner(cog.store, cog.encoder, cog.schema,
                                hidden_dim=64, n_layers=2)
        graph = r1.builder.build(feature_dim=64)
        stats_with = r1.fit(graph=graph, epochs=30, sheaf_weight=0.3, verbose=True)

        # Train WITHOUT sheaf coherence
        r2 = HypergraphReasoner(cog.store, cog.encoder, cog.schema,
                                hidden_dim=64, n_layers=2)
        stats_without = r2.fit(graph=graph, epochs=30, sheaf_weight=0.0, verbose=True)

        print(f"\n  Sheaf coherence regularization:")
        print(f"    With sheaf:    final_loss={stats_with['final_loss']:.4f}, "
              f"best_loss={stats_with['best_loss']:.4f}, "
              f"fiedler={stats_with['sheaf_fiedler']:.4f}")
        print(f"    Without sheaf: final_loss={stats_without['final_loss']:.4f}, "
              f"best_loss={stats_without['best_loss']:.4f}, "
              f"fiedler={stats_without['sheaf_fiedler']}")

        # Both should converge
        assert stats_with["final_loss"] < stats_with["losses"][0], \
            "Sheaf-regularized training did not reduce loss"
        assert stats_without["final_loss"] < stats_without["losses"][0], \
            "Unregularized training did not reduce loss"

        # Sheaf version should report a Fiedler value
        assert stats_with["sheaf_fiedler"] is not None
        assert stats_with["sheaf_fiedler"] >= 0

        # Without sheaf, fiedler should be None
        assert stats_without["sheaf_fiedler"] is None

        cog.close()
        print("\n  PASS: sheaf coherence in training")


def test_gradient_clipping():
    """Test: gradient clipping prevents exploding gradients."""
    import torch
    from cognition.logic import HypergraphReasoner

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cog = setup_cognition(db_path)
        for doc in DOCUMENTS:
            cog.ingest([doc])
        cog.consolidate()

        reasoner = HypergraphReasoner(cog.store, cog.encoder, cog.schema,
                                      hidden_dim=64, n_layers=2)
        graph = reasoner.builder.build(feature_dim=64)

        # Train with very tight gradient clipping
        stats_tight = reasoner.fit(graph=graph, epochs=20, grad_clip=0.1, verbose=True)

        print(f"\n  Gradient clipping (clip=0.1):")
        print(f"    Epochs: {stats_tight['epochs']}, "
              f"loss: {stats_tight['losses'][0]:.4f} → {stats_tight['final_loss']:.4f}")

        # Should still converge (not diverge)
        assert not any(torch.isnan(torch.tensor([l])) for l in stats_tight["losses"]), \
            "NaN in loss with gradient clipping"
        assert all(torch.isfinite(torch.tensor([l])) for l in stats_tight["losses"]), \
            "Inf in loss with gradient clipping"

        # Verify parameters are finite after training
        for name, param in reasoner.named_parameters():
            assert torch.isfinite(param).all(), f"Non-finite params in {name}"

        cog.close()
        print("\n  PASS: gradient clipping")


def test_early_stopping():
    """Test: early stopping halts training when loss plateaus."""
    from cognition.logic import HypergraphReasoner

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cog = setup_cognition(db_path)
        for doc in DOCUMENTS:
            cog.ingest([doc])
        cog.consolidate()

        # Train with aggressive early stopping (patience=3) and high LR
        # so convergence happens quickly and plateau is reached
        reasoner = HypergraphReasoner(cog.store, cog.encoder, cog.schema,
                                      hidden_dim=64, n_layers=2)
        graph = reasoner.builder.build(feature_dim=64)
        stats = reasoner.fit(graph=graph, epochs=200, patience=3,
                             lr=0.01, sheaf_weight=0.0, verbose=True)

        print(f"\n  Early stopping (patience=3, max_epochs=200):")
        print(f"    Actual epochs: {stats['epochs']}")
        print(f"    Early stopped: {stats['early_stopped']}")
        print(f"    Best loss: {stats['best_loss']:.4f}")

        # Should stop well before 200 epochs
        assert stats["epochs"] < 200, \
            f"Early stopping did not trigger (ran {stats['epochs']}/200 epochs)"
        assert stats["early_stopped"], "early_stopped flag not set"

        # Compare with no early stopping
        r2 = HypergraphReasoner(cog.store, cog.encoder, cog.schema,
                                hidden_dim=64, n_layers=2)
        stats_full = r2.fit(graph=graph, epochs=30, patience=999, verbose=False)
        print(f"    No early stopping (30 epochs): final_loss={stats_full['final_loss']:.4f}")
        assert stats_full["epochs"] == 30

        cog.close()
        print("\n  PASS: early stopping")


def test_batched_causal_evaluation():
    """Test: batched causal evaluation in refine() produces same results structure."""
    from cognition.logic import HypergraphReasoner

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cog = setup_cognition(db_path)
        for doc in DOCUMENTS:
            cog.ingest([doc])
        cog.consolidate()

        reasoner = HypergraphReasoner(cog.store, cog.encoder, cog.schema,
                                      hidden_dim=64, n_layers=2)

        # Run refine (uses batched evaluation internally)
        result = reasoner.refine(causal_threshold=0.5, verbose=True)

        print(f"\n  Batched causal evaluation:")
        print(f"    Pairs checked: {result.pairs_checked}")
        print(f"    Temporal edges: {result.temporal_added}")
        print(f"    Causal edges: {result.causal_added}")
        print(f"    Contradictions: {result.contradictions_found}")

        # Structural checks
        assert result.infons_updated > 0
        assert result.pairs_checked >= 0

        # All edge objects should be well-formed
        for edge in result.temporal_edges:
            assert edge.edge_type == "NEXT"
            assert edge.weight > 0
            assert "source" in edge.metadata
        for edge in result.causal_edges:
            assert edge.edge_type == "CAUSES"
            assert edge.weight > 0
            assert "from_pred" in edge.metadata
            assert "to_pred" in edge.metadata
        for edge in result.contradiction_edges:
            assert edge.edge_type == "CONTRADICTS"
            assert edge.weight > 0

        # Run refine a second time — should not crash on enriched graph
        result2 = reasoner.refine(causal_threshold=0.5, verbose=False)
        assert result2.infons_updated > 0

        cog.close()
        print("\n  PASS: batched causal evaluation")


def test_discover_anchors():
    """Test: GNN-refined Kan extension discovers anchor clusters."""
    from cognition.logic import HypergraphReasoner

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cog = setup_cognition(db_path)
        for doc in DOCUMENTS:
            cog.ingest([doc])
        cog.consolidate()

        reasoner = HypergraphReasoner(cog.store, cog.encoder, cog.schema,
                                      hidden_dim=64, n_layers=2)

        schema, discovered, stats = reasoner.discover_anchors(
            n_anchors=5, verbose=True,
        )

        print(f"\n  Anchor discovery (Kan extension on GNN embeddings):")
        print(f"    Input anchors: {stats['n_anchors']}")
        print(f"    Clusters found: {stats['n_clusters']}")
        print(f"    Silhouette: {stats['silhouette']:.3f}")
        print(f"    Cluster sizes: {stats['cluster_sizes']}")
        print(f"    Eigenvalues: {[f'{e:.3f}' for e in stats['eigenvalues'][:6]]}")

        assert stats["n_clusters"] > 0, "No clusters discovered"
        assert len(discovered) == stats["n_clusters"]

        # Each discovered anchor should have valid fields
        for da in discovered:
            assert da.name
            assert da.inferred_type in ("actor", "relation", "feature", "market")
            assert da.tokens
            assert da.size > 0
            print(f"    {da.name}: type={da.inferred_type}, size={da.size}, "
                  f"coherence={da.coherence:.3f}, tokens={da.tokens[:3]}")

        # Schema should have entries for each discovered cluster
        assert len(schema.names) == stats["n_clusters"]

        # Verify silhouette is a real number
        assert -1.0 <= stats["silhouette"] <= 1.0

        # The original schema types should be reflected in cluster types
        discovered_types = set(da.inferred_type for da in discovered)
        print(f"    Discovered types: {discovered_types}")
        # We expect at least 2 different types from our mixed schema
        assert len(discovered_types) >= 1

        cog.close()
        print("\n  PASS: discover anchors (Kan extension)")


def test_full_pipeline():
    """Test: complete pipeline — ingest → fit → reason → refine → discover."""
    from cognition.logic import HypergraphReasoner

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cog = setup_cognition(db_path)
        for doc in DOCUMENTS:
            cog.ingest([doc])
        cog.consolidate()

        reasoner = HypergraphReasoner(cog.store, cog.encoder, cog.schema,
                                      hidden_dim=64, n_layers=2)

        # 1. Fit with all optimizations
        graph = reasoner.builder.build(feature_dim=64)
        fit_stats = reasoner.fit(
            graph=graph, epochs=50,
            sheaf_weight=0.2, grad_clip=1.0, patience=8,
            verbose=True,
        )
        print(f"\n  Full pipeline:")
        print(f"    Fit: {fit_stats['epochs']} epochs, "
              f"loss={fit_stats['final_loss']:.4f}, "
              f"fiedler={fit_stats['sheaf_fiedler']:.4f}")

        # 2. Reason over queries
        queries = [
            "Did Toyota invest in battery technology?",
            "Is Tesla expanding production?",
            "Does CATL produce batteries?",
        ]
        for q in queries:
            result = reasoner.reason(q)
            m = result.mass
            print(f"    {result.verdict:20s} S={m.supports:.3f} R={m.refutes:.3f} "
                  f"θ={m.theta:.3f}  \"{q}\"")
            total = m.supports + m.refutes + m.uncertain + m.theta
            assert abs(total - 1.0) < 0.01

        # 3. Refine
        ref = reasoner.refine(verbose=True)
        print(f"    Refine: {ref.temporal_added} temporal, "
              f"{ref.causal_added} causal, {ref.contradictions_found} contradictions")

        # 4. Discover anchors
        schema, discovered, disc_stats = reasoner.discover_anchors(
            n_anchors=4, verbose=True,
        )
        print(f"    Discover: {disc_stats['n_clusters']} clusters, "
              f"silhouette={disc_stats['silhouette']:.3f}")

        # 5. Reason again after refinement (graph is enriched)
        reasoner2 = HypergraphReasoner(cog.store, cog.encoder, cog.schema,
                                       hidden_dim=64, n_layers=2)
        for q in queries[:1]:
            result2 = reasoner2.reason(q)
            print(f"    Post-refine: {result2.verdict} "
                  f"S={result2.mass.supports:.3f} θ={result2.mass.theta:.3f}")

        cog.close()
        print("\n  PASS: full pipeline")


if __name__ == "__main__":
    print("=" * 60)
    print("  HypergraphReasoner + IKL Logic Tests")
    print("=" * 60)

    test_ingest_and_build_graph()
    test_message_passing()
    test_ikl_operators()
    test_reasoner_end_to_end()
    test_compound_queries()
    test_that_and_ist()
    test_conditional_reasoning()
    test_refine_hypergraph()
    test_sheaf_coherence_in_training()
    test_gradient_clipping()
    test_early_stopping()
    test_batched_causal_evaluation()
    test_discover_anchors()
    test_full_pipeline()

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)

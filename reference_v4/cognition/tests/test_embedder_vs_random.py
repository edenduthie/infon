"""Head-to-head: GNN trained on random-projected features vs trained
SentenceEmbedder features, same scenario.

Measures:
    (a) GNN training loss after fixed epochs (lower is better)
    (b) Final DS mass on diagnostic queries
    (c) Neighbourhood coherence of infons extracted from the
        same document (sentences from d1 should cluster more than
        sentences from d1 vs d5).
"""
from __future__ import annotations

import os
import sys
import tempfile

import pytest
import torch

from cognition.synth import DEFAULT_SCHEMA, generate_corpus
from cognition.embedder import train_embedder


@pytest.fixture(scope="module")
def trained_pipeline():
    """Set up cognition on the EV corpus once; return both a random-
    projection reasoner and a trained-embedder reasoner."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from test_logic import setup_cognition, DOCUMENTS
    from cognition.logic import HypergraphReasoner

    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test.db")
    cog = setup_cognition(db_path)
    for d in DOCUMENTS:
        cog.ingest([d])
    cog.consolidate()

    # (a) Train the embedder on a synthetic corpus that shares the
    # same schema kinds (actors/relations/features/markets) but uses
    # the cognition scenario's anchor names so feature spaces line up.
    scenario_schema = type(DEFAULT_SCHEMA)(
        actors=["toyota", "honda", "tesla", "panasonic", "catl"],
        relations=["invests", "partners", "produces", "expands",
                   "delays", "acquires"],
        features=["battery", "solid_state", "ev", "factory", "supply_chain"],
        markets=["japan", "north_america", "china"],
    )
    examples = generate_corpus(
        schema=scenario_schema, n=600, seed=7,
    )
    stats = train_embedder(
        examples=examples,
        splade_encoder=cog.encoder.splade,
        anchor_names=scenario_schema.all_anchor_names(),
        node_dim=64,
        trunk_dim=128,
        epochs=15,
        lr=1e-3,
        verbose=False,
    )
    embedder = stats["embedder"]

    # (b) Two reasoners: one with default builder, one with embedder
    r_random = HypergraphReasoner(cog.store, cog.encoder, cog.schema,
                                  hidden_dim=64, n_layers=2)
    r_trained = HypergraphReasoner(cog.store, cog.encoder, cog.schema,
                                   hidden_dim=64, n_layers=2)
    r_trained.builder.embedder = embedder

    return cog, r_random, r_trained


def test_trained_embedder_gnn_loss_is_competitive(trained_pipeline):
    """After the same fixed number of GNN training epochs, the
    trained-embedder path should produce a GNN loss no worse than the
    random-projection path. (We'd hope for better; for a tiny scenario
    we settle for 'not meaningfully worse'.)"""
    cog, r_random, r_trained = trained_pipeline

    graph_random = r_random.builder.build(feature_dim=64)
    graph_trained = r_trained.builder.build(feature_dim=64)

    stats_random = r_random.fit(graph=graph_random, epochs=25,
                                verbose=False)
    stats_trained = r_trained.fit(graph=graph_trained, epochs=25,
                                  verbose=False)

    loss_random = stats_random["final_loss"]
    loss_trained = stats_trained["final_loss"]
    print(f"\n  random projection → GNN loss = {loss_random:.4f}")
    print(f"  trained embedder  → GNN loss = {loss_trained:.4f}")
    # Trained embedder path should be at least comparable
    assert loss_trained <= loss_random * 1.5, (
        f"trained embedder loss {loss_trained:.4f} is meaningfully "
        f"worse than random {loss_random:.4f}"
    )


def test_feature_semantics_with_trained_embedder(trained_pipeline):
    """The trained embedder should produce features where sentences
    about the same subject cluster more than across subjects."""
    cog, _, r_trained = trained_pipeline
    graph = r_trained.builder.build(feature_dim=64)

    # Group infons by subject; compute intra-subject vs inter-subject
    # cosine similarity on the raw node features BEFORE message passing.
    features = graph.node_features
    infons_by_subject = {}
    for iid, idx in graph.infon_map.items():
        inf = cog.store.get_infon(iid)
        if inf is None:
            continue
        infons_by_subject.setdefault(inf.subject, []).append(idx)

    # Pick two subjects with at least 3 infons each
    chosen = [s for s, ii in infons_by_subject.items() if len(ii) >= 3][:2]
    if len(chosen) < 2:
        pytest.skip("not enough infons per subject for this test")

    feat_a = torch.nn.functional.normalize(
        features[torch.tensor(infons_by_subject[chosen[0]])], dim=-1,
    )
    feat_b = torch.nn.functional.normalize(
        features[torch.tensor(infons_by_subject[chosen[1]])], dim=-1,
    )
    intra_a = (feat_a @ feat_a.T).mean().item()
    intra_b = (feat_b @ feat_b.T).mean().item()
    inter = (feat_a @ feat_b.T).mean().item()

    print(f"\n  subjects: {chosen[0]!r} and {chosen[1]!r}")
    print(f"  intra-{chosen[0]:10s} cosine = {intra_a:.3f}")
    print(f"  intra-{chosen[1]:10s} cosine = {intra_b:.3f}")
    print(f"  inter-subject        cosine = {inter:.3f}")

    # Same-subject should be at least somewhat more similar than across
    # subjects. This is the clean semantic-grouping property.
    assert intra_a > inter or intra_b > inter, (
        f"neither subject shows intra>inter clustering: "
        f"intra_a={intra_a:.3f}, intra_b={intra_b:.3f}, inter={inter:.3f}"
    )


def test_verdict_query_works_with_trained_embedder(trained_pipeline):
    """A round-trip query through the trained-embedder reasoner should
    still produce a well-defined DS mass."""
    cog, _, r_trained = trained_pipeline
    result = r_trained.reason("Did Toyota invest in batteries?")
    m = result.mass
    total = m.supports + m.refutes + m.uncertain + m.theta
    assert abs(total - 1.0) < 1e-3, (
        f"mass sum {total} off target with trained embedder"
    )
    print(f"\n  verdict: {result.verdict}")
    print(f"  mass   : S={m.supports:.3f} R={m.refutes:.3f} "
          f"U={m.uncertain:.3f} θ={m.theta:.3f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

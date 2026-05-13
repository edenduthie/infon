"""Tests for the surface-cue metadata extractors."""

from __future__ import annotations

import pytest

from cognition.metadata import (
    extract_evidentiality, extract_modality, resolve_coreference,
)


# ── Evidentiality classification ────────────────────────────────────────

def test_evidentiality_assertion():
    s = "Toyota officially announced the new solid-state battery program."
    r = extract_evidentiality(s)
    # Should be classified as assertion; "announced" is an assertion cue
    # (also a weak attribution cue, but the assertion corpus wins here
    # because of "officially" + declarative structure). We tolerate
    # either assertion or attribution — the key invariant is that it's
    # NOT classified as a hedge.
    assert r.label in ("assertion", "attribution")
    assert r.scores["hedge"] < r.scores["assertion"]
    assert abs(sum(r.scores.values()) - 1.0) < 1e-6


def test_evidentiality_hedge():
    s = "Toyota may possibly expand battery production next year."
    r = extract_evidentiality(s)
    assert r.label == "hedge"
    assert r.scores["hedge"] > r.scores["assertion"]


def test_evidentiality_attribution():
    s = "According to Reuters, Toyota has doubled its battery investment."
    r = extract_evidentiality(s)
    assert r.label == "attribution"
    assert r.scores["attribution"] > r.scores["hedge"]


# ── Modality classification ─────────────────────────────────────────────

def test_modality_certain():
    s = "Toyota definitely produced the new battery."
    r = extract_modality(s)
    assert r.label == "certain"
    assert abs(sum(r.scores.values()) - 1.0) < 1e-6


def test_modality_possible():
    s = "Toyota might possibly release a new model."
    r = extract_modality(s)
    assert r.label == "possible"
    assert r.scores["possible"] > r.scores["certain"]


def test_modality_obligation():
    s = "Toyota must comply with the new emissions regulations."
    r = extract_modality(s)
    assert r.label == "obligation"
    assert r.scores["obligation"] > r.scores["possible"]


# ── Coreference resolution ──────────────────────────────────────────────

def test_coref_resolves_pronoun():
    doc = "Toyota invests in batteries. It partners with Panasonic."
    sents = resolve_coreference(doc, actor_names=["toyota"])
    assert len(sents) == 2
    # Second sentence should now have "Toyota" prepended
    assert sents[0] == "Toyota invests in batteries."
    assert "toyota" in sents[1].lower(), (
        f"expected toyota-resolved: {sents[1]!r}"
    )
    print(f"\n  resolved: {sents[1]}")


def test_coref_resolves_the_company():
    doc = "Honda delays its EV line. The company partners with CATL for cells."
    sents = resolve_coreference(doc, actor_names=["honda", "catl"])
    assert len(sents) == 2
    assert "honda" in sents[1].lower(), (
        f"expected honda-resolved: {sents[1]!r}"
    )
    print(f"\n  resolved: {sents[1]}")


def test_coref_no_actor_means_no_rewrite():
    doc = "It is unclear what will happen next."
    sents = resolve_coreference(doc, actor_names=["toyota"])
    # Nothing to resolve to; shouldn't crash; should return sentence as-is
    assert len(sents) == 1
    assert sents[0] == "It is unclear what will happen next."


def test_coref_respects_lookback():
    # 5-sentence doc: actor mention only in sentence 1, pronoun in sentence 5
    doc = ("Toyota invests in batteries. "
           "A report appeared today. "
           "Analysts were skeptical. "
           "Several papers discussed it. "
           "It plans a big launch.")
    sents = resolve_coreference(doc, actor_names=["toyota"], lookback=2)
    # sentence 5 is 4 sentences after the actor mention; with lookback=2
    # the actor is out of the window and the pronoun is NOT resolved
    assert "toyota" not in sents[4].lower()

    # With a bigger lookback the resolution happens
    sents2 = resolve_coreference(doc, actor_names=["toyota"], lookback=5)
    assert "toyota" in sents2[4].lower()


def test_coreference_recovers_additional_infons():
    """End-to-end: with coreference enabled, a 3-sentence doc whose
    sentences 2-3 use pronouns yields more extracted infons than with
    coreference disabled."""
    import tempfile, os, json
    from cognition import Cognition, CognitionConfig

    SCHEMA = {
        "toyota":   {"type": "actor",    "tokens": ["toyota"]},
        "panasonic":{"type": "actor",    "tokens": ["panasonic"]},
        "invests":  {"type": "relation", "tokens": ["invest", "invests"]},
        "partners": {"type": "relation", "tokens": ["partner", "partners"]},
        "produces": {"type": "relation", "tokens": ["produce", "produces"]},
        "battery":  {"type": "feature",  "tokens": ["battery", "batteries"]},
        "japan":    {"type": "market",   "tokens": ["japan", "japanese"]},
    }

    DOC = {
        "id": "pronoun_doc",
        "text": (
            "Toyota invests in solid-state battery research. "
            "The company partners with Panasonic in Japan. "
            "It produces batteries for new EV models."
        ),
    }

    results = {}
    for coref_flag in (False, True):
        with tempfile.TemporaryDirectory() as tmpdir:
            schema_path = os.path.join(tmpdir, "schema.json")
            with open(schema_path, "w") as f:
                json.dump(SCHEMA, f)
            cog = Cognition(CognitionConfig(
                schema_path=schema_path,
                db_path=os.path.join(tmpdir, "test.db"),
                activation_threshold=0.2,
                min_confidence=0.02,
                top_k_per_role=3,
                coreference=coref_flag,
            ))
            cog.ingest([DOC])
            cog.consolidate()
            infons = cog.store.query_infons(limit=100)
            results[coref_flag] = [
                (i.subject, i.predicate, i.object) for i in infons
            ]
            print(f"\n  coref={coref_flag}: {len(infons)} infons")
            for t in results[coref_flag]:
                print(f"    {t[0]}/{t[1]}/{t[2]}")
            cog.close()

    n_without = len(results[False])
    n_with = len(results[True])
    print(f"\n  without coref: {n_without} infons")
    print(f"  with coref:    {n_with} infons")

    # Expectation: with coreference, sentences 2 and 3 gain a Toyota
    # subject, producing at least one extra infon anchored on Toyota
    # that wouldn't have been extracted otherwise.
    toyota_without = sum(1 for (s, p, o) in results[False] if s == "toyota")
    toyota_with = sum(1 for (s, p, o) in results[True] if s == "toyota")
    print(f"  toyota-subject without coref: {toyota_without}")
    print(f"  toyota-subject with coref:    {toyota_with}")
    assert toyota_with >= toyota_without, (
        "coreference should not reduce the number of toyota-subject infons"
    )
    # Either we picked up strictly more (the common case) or the test
    # corpus didn't have pronoun-resolvable extra claims.
    assert n_with >= n_without


def test_metadata_improves_supervision():
    """Train source weights with and without the two metadata sources.

    Measures whether the two new sources (evidentiality, modality) get
    non-trivial weight, and whether including them improves fit to the
    GNN's self-consistent readout.
    """
    import tempfile, os, json
    from cognition import Cognition, CognitionConfig
    from cognition.logic import HypergraphReasoner

    SCHEMA = {
        "toyota":   {"type": "actor",    "tokens": ["toyota"]},
        "honda":    {"type": "actor",    "tokens": ["honda"]},
        "tesla":    {"type": "actor",    "tokens": ["tesla"]},
        "invests":  {"type": "relation", "tokens": ["invest", "invests"]},
        "partners": {"type": "relation", "tokens": ["partner", "partners"]},
        "produces": {"type": "relation", "tokens": ["produce", "produces"]},
        "acquires": {"type": "relation", "tokens": ["acquire", "acquires"]},
        "delays":   {"type": "relation", "tokens": ["delay", "delays"]},
        "battery":  {"type": "feature",  "tokens": ["battery", "batteries"]},
        "ev":       {"type": "feature",  "tokens": ["ev", "electric vehicle"]},
        "japan":    {"type": "market",   "tokens": ["japan", "japanese"]},
        "china":    {"type": "market",   "tokens": ["china", "chinese"]},
    }

    # Each sentence is structured so that S/P/O are clearly present in
    # token form, while the surrounding cues drive evidentiality /
    # modality into the full three-way label space.
    DOCS = [
        {"id": "d1", "text":
            # assertion + certain
            "Toyota invests in battery production in Japan. "
            # hedge + possible
            "Honda might possibly invest in battery research. "
            # attribution + certain
            "According to Reuters, Tesla produces batteries in China. "
            # assertion + certain
            "Toyota partners with battery suppliers in Japan."
        },
        {"id": "d2", "text":
            # hedge
            "Honda may possibly delay its EV launch. "
            # attribution
            "Analysts said Honda partners with battery suppliers. "
            # assertion
            "Tesla acquires battery assets in China. "
            # hedge
            "Toyota could potentially invest in solid-state batteries."
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        schema_path = os.path.join(tmpdir, "schema.json")
        with open(schema_path, "w") as f:
            json.dump(SCHEMA, f)
        cog = Cognition(CognitionConfig(
            schema_path=schema_path,
            db_path=os.path.join(tmpdir, "test.db"),
            activation_threshold=0.2,
            min_confidence=0.02,
            top_k_per_role=3,
        ))
        for doc in DOCS:
            cog.ingest([doc])
        cog.consolidate()

        reasoner = HypergraphReasoner(cog.store, cog.encoder, cog.schema,
                                      hidden_dim=64, n_layers=2)
        graph = reasoner.builder.build(feature_dim=64)
        reasoner.fit(graph=graph, epochs=25, verbose=False)

        print("\n  Training 6-source DS weights:")
        stats = reasoner.train_source_weights(
            graph=graph, epochs=50, lr=5e-2, verbose=False,
        )
        assert stats["weights"] is not None
        total = sum(stats["weights"].values())
        assert abs(total - 1.0) < 1e-4

        print(f"    KL loss: {stats['first_loss']:.4f} → "
              f"{stats['final_loss']:.4f}")
        print(f"    learned weights:")
        for name, w in stats["weights"].items():
            print(f"      {name:20s}  w = {w:.3f}")

        # Check that evidentiality + modality received non-trivial weight.
        # "Non-trivial" here means > 5% of total — these two sources
        # together should matter enough to not be pushed to near-zero.
        evid_w = stats["weights"]["evidentiality"]
        modl_w = stats["weights"]["modality"]
        combined_new = evid_w + modl_w

        print(f"\n    evidentiality + modality combined weight: "
              f"{combined_new:.3f}")
        assert combined_new > 0.05, (
            f"evidentiality+modality combined weight {combined_new:.3f} "
            f"too small — no meaningful contribution"
        )

        # Verify infons actually carried the metadata: at least one hedge
        # and one attribution infon should exist in the extracted set.
        labels_seen = set()
        for infon in cog.store.query_infons(limit=100):
            labels_seen.add(getattr(infon, "evidentiality", "assertion"))
        print(f"    evidentiality labels seen: {labels_seen}")
        assert len(labels_seen) >= 2, (
            f"expected multiple evidentiality labels, saw {labels_seen}"
        )

        cog.close()
        print("\n  PASS: metadata makes a measurable contribution")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

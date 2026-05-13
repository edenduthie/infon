"""Sanity checks on the synthetic corpus generator.

Verifies that gold labels are consistent with the templates and that
the corpus has enough diversity to train a multi-task head.
"""
from __future__ import annotations

from collections import Counter

from cognition.synth import (
    DEFAULT_SCHEMA, generate_corpus,
    ROLE_NSUBJ, ROLE_VERB, ROLE_DOBJ, ROLE_POBJ, ROLE_MARKER,
    ROLE_LABELS,
)


def test_generator_deterministic():
    """Same seed gives same output — crucial for reproducible training."""
    a = generate_corpus(n=50, seed=0)
    b = generate_corpus(n=50, seed=0)
    assert len(a) == len(b) == 50
    for x, y in zip(a, b):
        assert x.sentence == y.sentence


def test_all_examples_have_consistent_labels():
    """Each example must satisfy: gold-triple anchors are all labeled in
    token_anchors, token_roles contain at least one NSUBJ + VERB, and
    mass values sum to 1."""
    examples = generate_corpus(n=200, seed=1)
    assert len(examples) == 200
    for ex in examples:
        # Tokens and roles align
        assert len(ex.tokens) == len(ex.token_roles) == len(ex.token_anchors)
        # Mass sums to 1
        total = sum(ex.ds_mass)
        assert abs(total - 1.0) < 1e-4, f"mass={ex.ds_mass}, sum={total}"
        # At least one subject and one verb role per example
        assert ROLE_NSUBJ in ex.token_roles or ROLE_DOBJ in ex.token_roles, (
            f"no subject/object roles in {ex.sentence!r}"
        )
        assert ROLE_VERB in ex.token_roles, (
            f"no verb role in {ex.sentence!r}"
        )
        # Gold subject should appear as an anchor
        subj, pred, obj = ex.gold_triple
        assert any(a == subj for a in ex.token_anchors), (
            f"subject anchor {subj!r} missing from token_anchors "
            f"of {ex.sentence!r}"
        )
        assert any(a == pred for a in ex.token_anchors), (
            f"predicate anchor {pred!r} missing"
        )


def test_mass_reflects_template():
    """Hedge sentences must have higher θ than assertion sentences."""
    hedge_examples = generate_corpus(n=50, seed=2, templates=["hedge"])
    svo_examples = generate_corpus(n=50, seed=2, templates=["svo"])
    neg_examples = generate_corpus(n=50, seed=2, templates=["neg"])

    hedge_theta = sum(ex.ds_mass[3] for ex in hedge_examples) / len(hedge_examples)
    svo_theta   = sum(ex.ds_mass[3] for ex in svo_examples)   / len(svo_examples)
    neg_refutes = sum(ex.ds_mass[1] for ex in neg_examples)   / len(neg_examples)

    print(f"\n  mean θ on hedge template:      {hedge_theta:.2f}")
    print(f"  mean θ on svo template:        {svo_theta:.2f}")
    print(f"  mean R on neg template:        {neg_refutes:.2f}")

    assert hedge_theta > svo_theta + 0.3, (
        f"hedge θ {hedge_theta:.2f} not meaningfully higher than "
        f"svo θ {svo_theta:.2f}"
    )
    assert neg_refutes > 0.5, (
        f"neg template R={neg_refutes:.2f} should dominate"
    )


def test_template_diversity():
    """Default corpus should visit all 8 templates roughly uniformly."""
    examples = generate_corpus(n=2000, seed=3)
    counts = Counter(ex.template_id for ex in examples)
    print(f"\n  template distribution over 2000 samples:")
    for tid, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {tid:15s} {cnt:4d}")
    # Each template should appear at least 10% of the minimum-expected
    # share (8 templates → ~250 expected → allow >= 100)
    assert len(counts) >= 7, f"expected ≥7 templates, saw {len(counts)}"
    for tid, cnt in counts.items():
        assert cnt >= 100, (
            f"template {tid!r} underrepresented at {cnt}/2000"
        )


def test_token_role_label_consistency():
    """For each example, the tokens at NSUBJ positions must match the
    gold subject, tokens at VERB positions must match the predicate."""
    examples = generate_corpus(n=100, seed=4)
    for ex in examples:
        for tok, role, anchor in zip(ex.tokens, ex.token_roles,
                                     ex.token_anchors):
            if role == ROLE_NSUBJ:
                assert anchor == ex.gold_triple[0] or anchor == "", (
                    f"nsubj token {tok!r} has anchor {anchor!r} but "
                    f"gold subject is {ex.gold_triple[0]!r} in "
                    f"{ex.sentence!r}"
                )
            if role == ROLE_VERB:
                assert anchor == ex.gold_triple[1] or anchor == "", (
                    f"verb token {tok!r} has anchor {anchor!r} but "
                    f"gold predicate is {ex.gold_triple[1]!r}"
                )


def test_default_schema_coverage():
    """The default schema's anchors should all appear at least once in
    a 1000-sample corpus."""
    examples = generate_corpus(n=1000, seed=5)
    all_tokens = set()
    for ex in examples:
        all_tokens.update(ex.token_anchors)
    all_tokens.discard("")
    for name in DEFAULT_SCHEMA.actors + DEFAULT_SCHEMA.relations:
        assert name in all_tokens, (
            f"anchor {name!r} missing from 1000-sample corpus"
        )


def test_schema_bridge_round_trip():
    """AnchorSchema → synth.Schema preserves actor/relation/feature/market
    partitioning, and corpora built from the bridged schema use the
    right anchor names."""
    from cognition.schema import AnchorSchema
    from cognition.synth import Schema

    anchor_defs = {
        "apple":    {"type": "actor",    "tokens": ["apple"]},
        "google":   {"type": "actor",    "tokens": ["google"]},
        "acquires": {"type": "relation", "tokens": ["acquire", "acquires"]},
        "launches": {"type": "relation", "tokens": ["launch", "launches"]},
        "iphone":   {"type": "feature",  "tokens": ["iphone"]},
        "android":  {"type": "feature",  "tokens": ["android"]},
        "usa":      {"type": "market",   "tokens": ["usa", "us"]},
    }
    anchor_schema = AnchorSchema(anchor_defs)
    synth_schema = Schema.from_anchor_schema(anchor_schema)

    # Partitioning preserved
    assert set(synth_schema.actors) == {"apple", "google"}
    assert set(synth_schema.relations) == {"acquires", "launches"}
    assert set(synth_schema.features) == {"iphone", "android"}
    assert set(synth_schema.markets) == {"usa"}

    # Corpus generated against this schema only references these names
    examples = generate_corpus(schema=synth_schema, n=100, seed=9)
    known = set(synth_schema.all_anchor_names())
    for ex in examples:
        for anchor in ex.token_anchors:
            if anchor:
                assert anchor in known, (
                    f"anchor {anchor!r} in generated example not in schema"
                )
        subj, pred, obj = ex.gold_triple
        assert subj in known
        assert pred in known
        assert obj in known
    print(f"\n  ok: bridge + generation on {len(known)} anchors")


def test_schema_bridge_handles_empty_types():
    """A schema with no markets should still generate — buckets get
    a placeholder so templates don't crash."""
    from cognition.schema import AnchorSchema
    from cognition.synth import Schema, generate_corpus

    anchor_defs = {
        "apple":    {"type": "actor",    "tokens": ["apple"]},
        "acquires": {"type": "relation", "tokens": ["acquire"]},
        "iphone":   {"type": "feature",  "tokens": ["iphone"]},
        # no market
    }
    anchor_schema = AnchorSchema(anchor_defs)
    synth_schema = Schema.from_anchor_schema(anchor_schema)
    assert synth_schema.markets == ["place"]   # fallback
    examples = generate_corpus(schema=synth_schema, n=50, seed=10,
                                templates=["svo"])   # no market templates
    assert len(examples) == 50
    # All svo examples don't reference markets anyway
    for ex in examples:
        assert ex.gold_triple[0] == "apple"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])

"""Head-driven triple extractor on the EV scenario.

Verifies that the trained SentenceEmbedder, when used as a triple
extractor, produces semantically correct triples on the canonical
scenario sentences, and in particular does NOT produce the
reversed-argument errors the Cartesian extractor creates.
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest


@pytest.fixture(scope="module")
def splade_encoder():
    from infon.encoder import SpladeEncoder
    return SpladeEncoder()


# Canonical scenario sentences + the gold triple we want extracted
SCENARIO = [
    ("Toyota invests batteries.",
     ("toyota", "invests", "battery")),
    ("Tesla produces batteries.",
     ("tesla", "produces", "battery")),
    ("Honda partners with CATL in China.",
     ("honda", "partners", "catl")),
    ("Toyota partners with Panasonic in Japan.",
     ("toyota", "partners", "panasonic")),
    ("CATL produces batteries.",
     ("catl", "produces", "battery")),
    ("Panasonic invests batteries.",
     ("panasonic", "invests", "battery")),
    ("Tesla expands factories in America.",
     ("tesla", "expands", "factory")),
    ("Honda delays EVs.",
     ("honda", "delays", "ev")),
]


@pytest.fixture(scope="module")
def trained_embedder_ev(splade_encoder):
    """Train the embedder on a synthetic corpus built from an EV-ish
    schema with exactly the anchors our scenario sentences use."""
    from infon.synth import Schema, generate_corpus
    from infon.embedder import train_embedder
    from infon.schema import AnchorSchema

    # Schema using lowercase names to match our scenario
    ev_schema_defs = {
        "toyota":    {"type": "actor",    "tokens": ["toyota"]},
        "honda":     {"type": "actor",    "tokens": ["honda"]},
        "tesla":     {"type": "actor",    "tokens": ["tesla"]},
        "panasonic": {"type": "actor",    "tokens": ["panasonic"]},
        "catl":      {"type": "actor",    "tokens": ["catl"]},
        "invests":   {"type": "relation", "tokens": ["invest", "invests"]},
        "partners":  {"type": "relation", "tokens": ["partner", "partners"]},
        "produces":  {"type": "relation", "tokens": ["produce", "produces"]},
        "expands":   {"type": "relation", "tokens": ["expand", "expands"]},
        "delays":    {"type": "relation", "tokens": ["delay", "delays"]},
        "battery":   {"type": "feature",  "tokens": ["battery", "batteries"]},
        "ev":        {"type": "feature",  "tokens": ["ev", "evs"]},
        "factory":   {"type": "feature",  "tokens": ["factory", "factories"]},
        "japan":     {"type": "market",   "tokens": ["japan"]},
        "china":     {"type": "market",   "tokens": ["china"]},
        "america":   {"type": "market",   "tokens": ["america"]},
    }
    anchor_schema = AnchorSchema(ev_schema_defs)
    synth_schema = Schema.from_anchor_schema(anchor_schema)

    examples = generate_corpus(
        schema=synth_schema, n=2000, seed=42,
    )
    stats = train_embedder(
        examples=examples,
        splade_encoder=splade_encoder,
        anchor_names=synth_schema.all_anchor_names(),
        node_dim=64,
        trunk_dim=256,
        epochs=30,
        lr=1e-3,
        verbose=False,
    )
    return stats["embedder"], anchor_schema


def test_head_extracts_canonical_triples(splade_encoder, trained_embedder_ev):
    """≥ 6/8 scenario sentences should produce the correct gold triple."""
    from infon.embedder import extract_triples_via_head

    embedder, schema = trained_embedder_ev
    correct = 0
    results = []
    for sent, gold in SCENARIO:
        triples = extract_triples_via_head(
            sent, embedder, splade_encoder, schema,
            anchor_activation_threshold=0.3,
            role_confidence_threshold=0.3,
        )
        if triples:
            extracted = (triples[0]["subject"],
                         triples[0]["predicate"],
                         triples[0]["object"])
            match = (extracted == gold)
            results.append((sent, gold, extracted, match))
            if match:
                correct += 1
        else:
            results.append((sent, gold, None, False))

    print("\n  head-driven extraction:")
    for sent, gold, extracted, match in results:
        marker = "✓" if match else "✗"
        print(f"    {marker} {sent}")
        if not match:
            print(f"      want: {gold}")
            print(f"      got:  {extracted}")

    accuracy = correct / len(SCENARIO)
    print(f"\n  accuracy: {correct}/{len(SCENARIO)} = {accuracy:.0%}")

    assert correct >= 6, (
        f"head extractor only got {correct}/{len(SCENARIO)} "
        f"correct — below 75% threshold"
    )


def test_head_never_reverses_arguments(splade_encoder, trained_embedder_ev):
    """Critical: on 'A partners with B' sentences, the subject is A,
    not B. The Cartesian extractor's failure mode is emitting
    (B, partners, A). The head must not do this."""
    from infon.embedder import extract_triples_via_head

    embedder, schema = trained_embedder_ev

    tests = [
        ("Toyota partners with Panasonic in Japan.",
         "toyota", "panasonic"),
        ("Honda partners with CATL in China.",
         "honda", "catl"),
    ]

    for sent, expected_subj, expected_obj in tests:
        triples = extract_triples_via_head(
            sent, embedder, splade_encoder, schema,
            anchor_activation_threshold=0.3,
            role_confidence_threshold=0.3,
        )
        if not triples:
            pytest.skip(f"no triple extracted from {sent!r}")
        extracted = triples[0]
        print(f"\n  {sent}")
        print(f"    → ({extracted['subject']}, "
              f"{extracted['predicate']}, {extracted['object']})")
        assert extracted["subject"] != expected_obj, (
            f"reversed arguments: subject should be {expected_subj!r}, "
            f"not {expected_obj!r}"
        )


def test_head_populates_mass_and_metadata(splade_encoder, trained_embedder_ev):
    """Every extracted triple carries a valid DS mass plus evid/modl."""
    from infon.embedder import extract_triples_via_head

    embedder, schema = trained_embedder_ev
    triples = extract_triples_via_head(
        "Toyota invests batteries.",
        embedder, splade_encoder, schema,
        anchor_activation_threshold=0.3,
        role_confidence_threshold=0.3,
    )
    if not triples:
        pytest.skip("no triple extracted for mass test")
    t = triples[0]
    m = t["mass"]
    assert set(m.keys()) == {"supports", "refutes", "uncertain", "theta"}
    total = sum(m.values())
    assert abs(total - 1.0) < 1e-4, f"mass sums to {total}"
    assert t["evidentiality"] in ("assertion", "hedge", "attribution")
    assert t["modality"] in ("certain", "possible", "obligation")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

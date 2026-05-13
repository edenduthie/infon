"""Quality filter: verify that the joint-score + role-type + per-
sentence cap reduces rubbish triples without dropping the
semantically-correct ones.

Three assertions:
    1. With a tighter quality_threshold, ingestion produces fewer infons.
    2. Canonical correct triples (toyota/invests/battery, etc.) still
       survive strict filtering.
    3. Hard role-type constraints reject obviously-wrong structures
       (subject.type != actor, object.type == relation).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest


# Reuse the EV scenario schema + docs
sys.path.insert(0, os.path.dirname(__file__))
from test_logic import setup_cognition, DOCUMENTS


def _count_infons(cog) -> int:
    return len(cog.store.query_infons(limit=500))


def _has_triple(cog, subj: str, pred: str, obj: str) -> bool:
    for inf in cog.store.query_infons(limit=500):
        if inf.subject == subj and inf.predicate == pred and inf.object == obj:
            return True
    return False


def _make_cognition(tmpdir, quality_threshold=0.05,
                     max_triples_per_sentence=3):
    """Fresh Cognition with the EV schema + quality-filter knobs."""
    from cognition import Cognition, CognitionConfig
    # Reuse the existing schema JSON from test_logic
    # (we need the anchor definitions; easier to write them fresh here)
    schema = {
        "toyota":    {"type": "actor",    "tokens": ["toyota"]},
        "honda":     {"type": "actor",    "tokens": ["honda"]},
        "tesla":     {"type": "actor",    "tokens": ["tesla"]},
        "panasonic": {"type": "actor",    "tokens": ["panasonic"]},
        "catl":      {"type": "actor",    "tokens": ["catl"]},
        "invests":   {"type": "relation", "tokens":
                       ["invest", "invests", "invested", "investment"]},
        "partners":  {"type": "relation", "tokens":
                       ["partner", "partners", "partnered", "partnership"]},
        "produces":  {"type": "relation", "tokens":
                       ["produce", "produces", "produced", "production"]},
        "expands":   {"type": "relation", "tokens":
                       ["expand", "expands", "expanded"]},
        "delays":    {"type": "relation", "tokens":
                       ["delay", "delays", "delayed"]},
        "acquires":  {"type": "relation", "tokens":
                       ["acquire", "acquires", "acquired"]},
        "battery":   {"type": "feature",  "tokens": ["battery", "batteries"]},
        "solid_state":{"type": "feature", "tokens":
                       ["solid-state", "solid state"]},
        "ev":        {"type": "feature",  "tokens":
                       ["ev", "electric vehicle", "electric vehicles"]},
        "factory":   {"type": "feature",  "tokens":
                       ["factory", "plant", "facility"]},
        "supply_chain":{"type": "feature","tokens": ["supply chain", "supply"]},
        "japan":     {"type": "market",   "tokens": ["japan", "japanese"]},
        "china":     {"type": "market",   "tokens": ["china", "chinese"]},
        "north_america":{"type": "market","tokens":
                       ["north america", "us", "united states"]},
    }
    schema_path = os.path.join(tmpdir, "schema.json")
    with open(schema_path, "w") as f:
        json.dump(schema, f)
    cog = Cognition(CognitionConfig(
        schema_path=schema_path,
        db_path=os.path.join(tmpdir, "test.db"),
        activation_threshold=0.2,
        min_confidence=0.02,
        top_k_per_role=3,
        quality_threshold=quality_threshold,
        max_triples_per_sentence=max_triples_per_sentence,
    ))
    return cog


def test_quality_filter_reduces_infons():
    """Tightening quality_threshold strictly reduces the number of
    extracted infons on the same corpus."""
    with tempfile.TemporaryDirectory() as tmpdir_lax:
        cog_lax = _make_cognition(tmpdir_lax,
                                  quality_threshold=0.02,
                                  max_triples_per_sentence=10)
        for doc in DOCUMENTS:
            cog_lax.ingest([doc])
        n_lax = _count_infons(cog_lax)
        cog_lax.close()

    with tempfile.TemporaryDirectory() as tmpdir_strict:
        cog_strict = _make_cognition(tmpdir_strict,
                                     quality_threshold=0.15,
                                     max_triples_per_sentence=2)
        for doc in DOCUMENTS:
            cog_strict.ingest([doc])
        n_strict = _count_infons(cog_strict)
        cog_strict.close()

    print(f"\n  lax    (q=0.02, cap=10): {n_lax} infons")
    print(f"  strict (q=0.15, cap=2):   {n_strict} infons")

    assert n_strict < n_lax, (
        f"strict filter produced {n_strict} infons, not less than "
        f"lax's {n_lax}"
    )


def test_core_correct_triples_survive_strict_filter():
    """Even with a strict filter, canonical EV-scenario triples
    should remain in the extracted corpus."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cog = _make_cognition(tmpdir,
                              quality_threshold=0.15,
                              max_triples_per_sentence=2)
        for doc in DOCUMENTS:
            cog.ingest([doc])

        # Core triples that the scenario clearly asserts
        expected = [
            ("toyota", "invests", "battery"),
            ("toyota", "invests", "solid_state"),
            ("tesla", "produces", "battery"),
            ("catl", "produces", "battery"),
        ]
        print(f"\n  strict-filter infons:")
        for inf in cog.store.query_infons(limit=200):
            print(f"    {inf.subject:10s} / {inf.predicate:10s} "
                  f"/ {inf.object:12s}  conf={inf.confidence:.3f}")
        print()

        # At least half of the expected triples should survive
        survived = sum(1 for t in expected if _has_triple(cog, *t))
        print(f"  expected triples that survived: "
              f"{survived}/{len(expected)}")
        assert survived >= 2, (
            f"only {survived} of {len(expected)} expected triples "
            f"survived the strict filter — threshold is too tight"
        )
        cog.close()


def test_role_type_hard_constraints():
    """No infon should have (a) subject whose type is not 'actor'
    (b) predicate whose type is not 'relation' (c) object whose type
    is 'relation'. These are structural invariants the filter
    enforces."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cog = _make_cognition(tmpdir,
                              quality_threshold=0.02,
                              max_triples_per_sentence=10)
        for doc in DOCUMENTS:
            cog.ingest([doc])

        infons = cog.store.query_infons(limit=500)
        assert len(infons) > 0
        for inf in infons:
            s_type = cog.schema.types.get(inf.subject, "")
            p_type = cog.schema.types.get(inf.predicate, "")
            o_type = cog.schema.types.get(inf.object, "")
            assert s_type == "actor", (
                f"subject {inf.subject!r} has type {s_type!r}, "
                f"expected 'actor' — role-type constraint violated"
            )
            assert p_type == "relation", (
                f"predicate {inf.predicate!r} has type {p_type!r}, "
                f"expected 'relation'"
            )
            assert o_type != "relation" and o_type != "", (
                f"object {inf.object!r} has type {o_type!r}, which is "
                f"invalid for an object role"
            )
        print(f"\n  all {len(infons)} infons respect role-type "
              f"constraints")
        cog.close()


def test_per_sentence_cap_bounds_triples():
    """max_triples_per_sentence=1 should produce at most 1 triple
    from any single sentence in the corpus."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cog = _make_cognition(tmpdir,
                              quality_threshold=0.02,
                              max_triples_per_sentence=1)
        for doc in DOCUMENTS:
            cog.ingest([doc])

        # Group by (doc_id, sent_id) and check the cap
        buckets: dict[tuple[str, str], int] = {}
        for inf in cog.store.query_infons(limit=500):
            key = (inf.doc_id, inf.sent_id)
            buckets[key] = buckets.get(key, 0) + 1
        print(f"\n  triples per sentence (cap=1):")
        for k, v in list(buckets.items())[:10]:
            print(f"    {k}: {v}")
        for key, n in buckets.items():
            assert n <= 1, (
                f"sentence {key} has {n} infons, cap was 1"
            )
        cog.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

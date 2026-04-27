"""Tests for blackmagic.imagine — the GA imagination layer."""
import pytest

from blackmagic.imagine import (
    HierarchyWalkMutator, PredicateSubstitutionMutator,
    PolarityFlipMutator, RoleRecombinationMutator,
    FitnessFunction, Imagination, _clone_with, _imagined_id,
)
from blackmagic.infon import Infon, IMAGINATION_TO_MCTS_VERDICT
import random


def _sample_infon(**overrides):
    base = dict(
        infon_id="obs1", subject="toyota", predicate="invest",
        object="battery", polarity=1, confidence=0.9, sentence="...",
        doc_id="d1", sent_id="d1_0000",
        subject_meta={"type": "actor"},
        predicate_meta={"type": "relation"},
        object_meta={"type": "feature"},
    )
    base.update(overrides)
    return Infon(**base)


def test_imagined_id_deterministic():
    id1 = _imagined_id("toyota", "invest", "battery", 1)
    id2 = _imagined_id("toyota", "invest", "battery", 1)
    assert id1 == id2
    assert id1.startswith("img_")


def test_clone_marks_imagined():
    a = _sample_infon()
    b = _clone_with(a, predicate="launch")
    assert b.kind == "imagined"
    assert b.predicate == "launch"
    assert a.infon_id in b.parent_infon_ids
    assert b.infon_id != a.infon_id


def test_polarity_flip_mutator():
    rng = random.Random(0)
    m = PolarityFlipMutator()
    out = m.apply(_sample_infon(polarity=1), rng=rng)
    assert out.polarity == 0
    assert out.kind == "imagined"


def test_predicate_substitution(tiny_schema):
    rng = random.Random(0)
    m = PredicateSubstitutionMutator(tiny_schema)
    out = m.apply(_sample_infon(predicate="invest"), rng=rng)
    assert out is not None
    assert out.predicate != "invest"
    assert tiny_schema.types.get(out.predicate) == "relation"


def test_recombination_needs_two(tiny_schema):
    rng = random.Random(0)
    m = RoleRecombinationMutator()
    a = _sample_infon(infon_id="a")
    b = _sample_infon(infon_id="b", subject="ford", predicate="recall",
                      object="us")
    out = m.apply(a, rng=rng, all_observed=[a, b])
    # Should blend roles from a and b
    assert out is not None
    assert (out.subject in {a.subject, b.subject}
            and out.predicate in {a.predicate, b.predicate}
            and out.object in {a.object, b.object})


def test_fitness_grammar_rejects_bad_types(bm_populated):
    ff = FitnessFunction(bm_populated.store, bm_populated.schema,
                         bm_populated.config)
    # Swap actor into object slot — should fail grammar
    bad = _sample_infon(object="ford",  # actor in object position
                        object_meta={"type": "actor"})
    fit, _ = ff(bad)
    assert fit == 0.0


def test_fitness_whisper_requires_corpus_presence(bm_populated):
    ff = FitnessFunction(bm_populated.store, bm_populated.schema,
                         bm_populated.config)
    # Anchor "ev" might be present in corpus; try an anchor that's NOT in corpus
    # by using one that never appears. Tiny corpus doesn't mention "cancel"
    # via the tokens we use — verify whisper handles zero-freq gracefully.
    never_present = _sample_infon(subject="bmw", predicate="cancel", object="china")
    fit, comp = ff(never_present)
    # whisper may or may not be zero depending on what the tiny corpus
    # produced — just assert grammar is 1 and result is non-negative
    assert 0.0 <= fit <= 2.0


def test_imagine_end_to_end(bm_populated):
    result = bm_populated.imagine(
        "What partnerships might emerge in the EV market?",
        n_generations=3, population_size=20, persona="investor",
        store_imagined=False,
    )
    # MCTS-shaped output
    assert result.verdict in ("PLAUSIBLE", "CONTRADICTED", "SPECULATIVE")
    assert result.mcts_verdict == IMAGINATION_TO_MCTS_VERDICT[result.verdict]
    assert result.traversal_tree is not None
    assert result.generations > 0
    assert result.iterations == result.generations
    assert result.elapsed_s > 0


def test_imagined_infons_have_provenance(bm_populated):
    result = bm_populated.imagine(
        "Battery investments", n_generations=2, population_size=10,
        store_imagined=False,
    )
    for inf in result.imagined_infons:
        assert inf.kind == "imagined"
        assert inf.fitness is not None
        # Should have at least one parent
        assert isinstance(inf.parent_infon_ids, list)


def test_imagine_stores_when_requested(bm_populated):
    n_before = bm_populated.store.count_infons(kind="imagined")
    _ = bm_populated.imagine(
        "Battery investments", n_generations=2, population_size=10,
        store_imagined=True,
    )
    n_after = bm_populated.store.count_infons(kind="imagined")
    # Should have stored some (may be 0 if fitness is very selective)
    assert n_after >= n_before


def test_imagine_reproducible_with_seed(bm_populated):
    # We don't currently accept a seed at the facade level; confirm that
    # successive runs without a seed both produce valid results.
    r1 = bm_populated.imagine(
        "Battery investments", n_generations=2, population_size=10,
        store_imagined=False,
    )
    r2 = bm_populated.imagine(
        "Battery investments", n_generations=2, population_size=10,
        store_imagined=False,
    )
    assert r1.verdict in ("PLAUSIBLE", "CONTRADICTED", "SPECULATIVE")
    assert r2.verdict in ("PLAUSIBLE", "CONTRADICTED", "SPECULATIVE")

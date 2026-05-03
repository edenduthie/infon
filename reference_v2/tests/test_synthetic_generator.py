"""Red-phase test for synthetic corpus generator determinism and oracle labels.

This test currently FAILS — that is the point. The
``reference_v2.synthetic.generator`` module does not exist yet, so
``ImportError`` is raised at import time.

Once the generator lands (task infon-8pa.2), all assertions here must
pass green:

1. ``Generator(seed=42)`` is constructable.
2. ``generate(...)`` is deterministic: two calls with the same seed return
   byte-equal scenario lists (same ids, same sentences, same labels).
3. Every scenario has ``planted_verdict``, ``planted_thinness``, and
   ``planted_hop_count`` populated (non-null).
4. ``planted_thinness`` for each scenario equals the actual count of
   supporting sentences produced in the corpus for that scenario's query.

Axes being validated here:
- H1 = compositional_depth (hop chains): drives ``planted_hop_count``
- H2 = evidence_redundancy (supporting sentences): drives ``planted_thinness``

Reference: openspec/epic-02 synthetic dataset spec
           tasks.md A.1 (red) → A.2 (green)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Module under test (does not exist yet — causes ImportError on collection)
# ---------------------------------------------------------------------------
from reference_v2.synthetic.generator import Generator  # type: ignore[import]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_PARAMS = dict(
    n_docs=10,
    evidence_redundancy=2,
    compositional_depth=2,
    contradiction_density=0.0,
    nei_fraction=0.0,
)


def _call_generate(gen: Generator) -> list:
    """Call ``gen.generate`` with the shared default parameters."""
    return gen.generate(**_DEFAULT_PARAMS)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_generator_is_deterministic():
    """Two Generator instances with the same seed must produce identical output.

    Phase A.1 (RED): expected failure is
        ImportError: No module named 'reference_v2.synthetic.generator'

    Phase A.2 (GREEN): the generator produces byte-equal scenario lists when
    constructed with the same seed, independent of call order or Python
    process state.
    """
    gen_a = Generator(seed=42)
    gen_b = Generator(seed=42)

    scenarios_a = _call_generate(gen_a)
    scenarios_b = _call_generate(gen_b)

    assert len(scenarios_a) == len(scenarios_b), (
        "deterministic generator must produce the same number of scenarios: "
        f"first call={len(scenarios_a)}, second call={len(scenarios_b)}"
    )

    for idx, (s_a, s_b) in enumerate(zip(scenarios_a, scenarios_b)):
        assert s_a == s_b, (
            f"scenario at index {idx} differs between two seeded runs:\n"
            f"  first  = {s_a!r}\n"
            f"  second = {s_b!r}"
        )


def test_oracle_labels_are_populated():
    """Every scenario must have planted_verdict, planted_thinness, and
    planted_hop_count set to a non-null value.

    These fields are the ground-truth oracle labels used to compute the
    ablation matrix for Epic 02; any null means the label is unusable.
    """
    gen = Generator(seed=42)
    scenarios = _call_generate(gen)

    assert scenarios, "generate() must return at least one scenario"

    for idx, scenario in enumerate(scenarios):
        assert scenario.planted_verdict is not None, (
            f"scenario[{idx}].planted_verdict must not be None"
        )
        assert scenario.planted_thinness is not None, (
            f"scenario[{idx}].planted_thinness must not be None"
        )
        assert scenario.planted_hop_count is not None, (
            f"scenario[{idx}].planted_hop_count must not be None"
        )


def test_planted_thinness_matches_corpus_count():
    """planted_thinness must equal the actual count of supporting sentences
    in the generated corpus for each scenario's query.

    This is the key oracle-accuracy invariant: the generator stamps the label
    itself, so the label must match what it actually planted in the corpus.
    Divergence here means the ablation matrix would be trained on wrong labels.
    """
    gen = Generator(seed=42)
    scenarios = _call_generate(gen)

    assert scenarios, "generate() must return at least one scenario"

    for idx, scenario in enumerate(scenarios):
        supporting_sentences = [
            sent
            for sent in scenario.corpus_sentences
            if sent.supports_query
        ]
        actual_count = len(supporting_sentences)
        assert scenario.planted_thinness == actual_count, (
            f"scenario[{idx}]: planted_thinness={scenario.planted_thinness!r} "
            f"does not match actual supporting sentence count={actual_count}. "
            f"Oracle label must reflect what was planted in the corpus."
        )


if __name__ == "__main__":
    # Manual invocation path for quick iteration; pytest is the primary runner.
    test_generator_is_deterministic()
    test_oracle_labels_are_populated()
    test_planted_thinness_matches_corpus_count()
    print("PASS: synthetic generator determinism and oracle labels")

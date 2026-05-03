"""Red-phase TDD tests for the configurable decisive_top_k fusion cap.

This file is intentionally written BEFORE A.5b lands the new
``CognitionConfig.decisive_top_k`` field (default 3) and the
``decisive_top_k=`` kwarg on ``HypergraphReasoner.reason()``. Tests 1,
2, and 4 MUST fail today; test 3 is a sanity contract that locks A.5b's
guarantee that ``decisive_top_k=1`` is equivalent to fusion ``rule="top1"``.

Spec:
- openspec/changes/epic-01-stabilize-theta/spec.md
  Requirement: Configurable Fusion Cap
- openspec/changes/epic-01-stabilize-theta/tasks.md A.5 (red) → A.5b (green)
- docs/publication/reproduction_audit.md §Path 2.4

Current state of the cap (pre-A.5b):
    reference_v2/src/cognition/logic.py:1085 reads
        decisive = [m for m, w in weighted_masses if m.theta < 0.95][:5]
    The literal ``[:5]`` is the cap. There is no ``decisive_top_k`` on
    CognitionConfig and ``reason()`` does not accept the kwarg.

No mocks; real reasoner, real masses, real EV scenario.
"""

from __future__ import annotations

import os
import tempfile

import pytest

# Match the corpus and schema used by tests/test_logic.py and the other
# epic-01 red tests so this test plugs into the same regression family.
from tests.test_logic import DOCUMENTS, setup_cognition


# Diagnostic query reused by the audit (Toyota probe — the EV corpus has
# ≥ 3 supporting infons such as "Toyota invests heavily in solid-state
# battery technology", "Toyota partners with Panasonic on battery
# development", "Toyota's solid-state battery investment leads to a
# breakthrough", which gives the fusion stage enough decisive masses to
# show the top-k cap effect).
_TOYOTA_QUERY = "Did Toyota invest in battery technology?"


def _build_and_ingest(db_path: str):
    """Construct a fresh Cognition + ingest the EV corpus."""
    cog = setup_cognition(db_path)
    for doc in DOCUMENTS:
        cog.ingest([doc])
    cog.consolidate()
    return cog


def _build_reasoner(cog):
    """Construct a fresh HypergraphReasoner over an already-ingested store.

    ``log_per_infon_masses=True`` is required by ``test_top_k_one_equals_top1_rule``
    which inspects ``result.per_infon_masses`` to drive the Path-B
    direct-DS-algebra comparison. After infon-6o3.36 the flag defaults to
    False, so tests that rely on the diagnostic output must opt in
    explicitly.
    """
    from cognition.logic import HypergraphReasoner

    return HypergraphReasoner(
        cog.store, cog.encoder, cog.schema,
        hidden_dim=64, n_layers=2,
        log_per_infon_masses=True,
    )


def _polarity(mass) -> str:
    """Return the argmax of {S, R, U, Θ} for a MassFunction."""
    parts = {
        "S": mass.supports,
        "R": mass.refutes,
        "U": mass.uncertain,
        "Theta": mass.theta,
    }
    return max(parts, key=parts.get)


# ── Test 1: default value of decisive_top_k on CognitionConfig ──────────


def test_decisive_top_k_default_value():
    """``CognitionConfig.decisive_top_k`` must default to 3.

    Phase A.5 (RED): the field does not exist on the dataclass yet, so
    ``cog.config.decisive_top_k`` raises ``AttributeError``. The audit
    motivates lowering the cap from the hardcoded 5 in ``logic.py:1085``
    to 3 (spec.md Requirement: Configurable Fusion Cap).

    Phase A.5b (GREEN): the field exists with default 3.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_default.db")
        cog = setup_cognition(db_path)
        try:
            assert hasattr(cog.config, "decisive_top_k"), (
                "CognitionConfig is missing the `decisive_top_k` field "
                "required by spec.md Requirement: Configurable Fusion Cap"
            )
            assert cog.config.decisive_top_k == 3, (
                f"CognitionConfig.decisive_top_k default must be 3 "
                f"(audit §Path 2.4 prescribes lowering from 5 → 3); "
                f"got {cog.config.decisive_top_k!r}"
            )
        finally:
            cog.close()


# ── Test 2: decisive_top_k kwarg caps fusion and preserves Θ ────────────


def test_decisive_top_k_caps_fusion():
    """``reason(query, decisive_top_k=1)`` must leave strictly more m(Θ)
    than ``reason(query, decisive_top_k=5)`` on a query with multiple
    confident agreeing supports.

    Phase A.5 (RED): ``HypergraphReasoner.reason()`` does not accept the
    ``decisive_top_k=`` kwarg today (signature is
    ``reason(query, max_infons, fit_epochs, verbose)``), so the first
    call raises ``TypeError: reason() got an unexpected keyword
    argument 'decisive_top_k'``.

    Phase A.5b (GREEN): with 5 confident agreeing masses, Dempster
    collapses m(Θ) → ≈ 0; capping at top-1 returns a mass with whatever
    Θ that single infon had (≥ 0.05 typically). The strict inequality
    therefore holds.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_cap.db")
        cog = _build_and_ingest(db_path)
        try:
            reasoner = _build_reasoner(cog)
            graph = reasoner.builder.build(feature_dim=64)
            reasoner.fit(graph=graph, epochs=30, seed=42)

            result_top1 = reasoner.reason(_TOYOTA_QUERY, decisive_top_k=1)
            result_top5 = reasoner.reason(_TOYOTA_QUERY, decisive_top_k=5)
        finally:
            cog.close()

    assert result_top1.mass.theta > result_top5.mass.theta, (
        f"capping fusion at top-1 must leave more m(Θ) than top-5 on a "
        f"query with multiple confident agreeing supports (audit §Path "
        f"2.4); got top1.theta={result_top1.mass.theta!r}, "
        f"top5.theta={result_top5.mass.theta!r}"
    )


# ── Test 3: decisive_top_k=1 ⇔ rule="top1" contract ─────────────────────


def test_top_k_one_equals_top1_rule():
    """``decisive_top_k=1`` must produce identical behaviour to
    ``combine_multiple(masses, rule="top1")`` for any fusion rule.

    This is the contract from spec.md Requirement: Configurable Fusion
    Cap acceptance criterion: "Setting ``decisive_top_k=1`` SHALL
    produce identical behaviour to ``rule='top1'`` for any fusion rule."

    Status (Phase A.5 / red): this test depends on the
    ``decisive_top_k=`` kwarg landing; until A.5b ships, the first
    ``reason(..., decisive_top_k=1)`` call raises ``TypeError`` and the
    test fails for the same kwarg-not-accepted reason as test 2.

    Phase A.5b (GREEN): the assertion below pins A.5b's contract — the
    fused mass returned from reason(top_k=1) and the mass produced by
    direct DS algebra with ``rule="top1"`` over the same per-infon
    masses must agree to 1e-6.
    """
    from cognition.dempster_shafer import combine_multiple
    from cognition.logic import MassFunction

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_top1_eq.db")
        cog = _build_and_ingest(db_path)
        try:
            reasoner = _build_reasoner(cog)
            graph = reasoner.builder.build(feature_dim=64)
            reasoner.fit(graph=graph, epochs=30, seed=42)

            # Path A: reason() with the new kwarg.
            result_via_kwarg = reasoner.reason(
                _TOYOTA_QUERY, decisive_top_k=1,
            )

            # Path B: direct DS algebra over the per-infon masses the
            # logger emitted on the SAME reason() call. Reusing the
            # logger's records guarantees we see exactly the masses
            # reason() would have fused, so the comparison is apples to
            # apples regardless of how many infons the graph yielded.
            log = result_via_kwarg.per_infon_masses
            assert log, (
                "per-infon mass log is empty for the Toyota query; the "
                "EV corpus must surface at least one supporting infon "
                "for this contract to be meaningful"
            )
            per_infon_masses = [
                MassFunction(
                    supports=record.mass[0],
                    refutes=record.mass[1],
                    uncertain=record.mass[2],
                    theta=record.mass[3],
                )
                for record in log
            ]
            result_via_rule = combine_multiple(
                per_infon_masses, rule="top1",
            )
        finally:
            cog.close()

    fused = result_via_kwarg.mass
    assert fused.supports == pytest.approx(
        result_via_rule.supports, abs=1e-6,
    ), (
        f"reason(decisive_top_k=1).supports={fused.supports} must equal "
        f"combine_multiple(rule='top1').supports={result_via_rule.supports} "
        f"per spec.md Requirement: Configurable Fusion Cap"
    )
    assert fused.refutes == pytest.approx(
        result_via_rule.refutes, abs=1e-6,
    )
    assert fused.uncertain == pytest.approx(
        result_via_rule.uncertain, abs=1e-6,
    )
    assert fused.theta == pytest.approx(
        result_via_rule.theta, abs=1e-6,
    )


# ── Test 4: polarity preserved across the cap sweep ─────────────────────


def test_polarity_preserved_across_top_k():
    """The verdict polarity (argmax of {S, R, U, Θ}) must not flip as
    ``decisive_top_k`` is swept across the *fusion* range {2, 3, 5}.

    Phase A.5 (RED): the kwarg is not accepted, so the first
    ``reason(..., decisive_top_k=2)`` call raises ``TypeError`` and the
    test fails immediately.

    Phase A.5b (GREEN): on the Toyota probe the EV corpus is firmly on
    the SUPPORTS side once at least two high-relevance contributors are
    fused; capping fusion in this range only changes the magnitude of
    the focal masses (and frees mass back to Θ), not which focal
    element dominates. Stage B's sweep across {2, 3, 5} relies on this
    invariant — if polarity flipped under the cap the cap would be
    unsafe.

    Note (A.5b finding): ``decisive_top_k=1`` is intentionally EXCLUDED
    from this invariant because spec.md Requirement: Configurable
    Fusion Cap pins ``decisive_top_k=1`` to ``rule="top1"`` semantics
    (smallest m(Θ) over the full per-infon pool, NOT highest-relevance).
    On a corpus where the most-decisive single mass happens to refute
    the query, ``k=1`` will legitimately flip polarity vs. ``k≥2``;
    that polarity-flip risk is exactly what Stage B's sweep over
    ``top_k ∈ {1, 2, 3, 5}`` will quantify (see
    ``openspec/changes/epic-01-stabilize-theta/tasks.md`` Task B.3 and
    audit §Path 2.4). The test 3 ``top_k=1 ⇔ rule='top1'`` contract
    above is the load-bearing assertion for ``k=1`` behaviour.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_polarity.db")
        cog = _build_and_ingest(db_path)
        try:
            reasoner = _build_reasoner(cog)
            graph = reasoner.builder.build(feature_dim=64)
            reasoner.fit(graph=graph, epochs=30, seed=42)

            polarities = {}
            for k in (2, 3, 5):
                result = reasoner.reason(_TOYOTA_QUERY, decisive_top_k=k)
                polarities[k] = _polarity(result.mass)
        finally:
            cog.close()

    distinct = set(polarities.values())
    assert len(distinct) == 1, (
        f"polarity must not flip across decisive_top_k ∈ {{2,3,5}} on "
        f"the Toyota probe; got per-k polarities {polarities!r}"
    )


if __name__ == "__main__":
    test_decisive_top_k_default_value()
    test_decisive_top_k_caps_fusion()
    test_top_k_one_equals_top1_rule()
    test_polarity_preserved_across_top_k()
    print("PASS: decisive_top_k cap tests (post-A.5b)")

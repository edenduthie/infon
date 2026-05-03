"""Red-phase TDD tests for alternative Dempster-Shafer fusion rules.

This file is intentionally written BEFORE the implementations of
``combine_yager``, ``combine_murphy``, ``combine_top1`` and the
``rule=`` dispatcher on ``combine_multiple`` exist. Tests 2, 3, 4,
and (most of) 5 MUST fail with ``AttributeError`` / ``TypeError``
until task A.4b lands.

Spec:
- openspec/changes/epic-01-stabilize-theta/spec.md
  Requirement: Alternative Fusion Rules
- design.md decision: "Add three fusion rules in addition to Dempster"

Per-rule semantics:
- dempster: conflict mass K is renormalized away (current behaviour).
- yager (Yager 1987): conflict mass is added to m(Θ) instead of being
  renormalized — captures "if sources disagree we are MORE ignorant".
- murphy (Murphy 2000): mean the masses, then Dempster-combine the
  average with itself n-1 times. Empirical workhorse that avoids
  Zadeh's counter-example.
- top1: trivial cautious floor — return the most-decisive (smallest
  m(Θ)) input mass without performing any fusion.

No mocks; real MassFunction objects, real DS algebra.
"""

from __future__ import annotations

import pytest

from cognition import dempster_shafer
from cognition.dempster_shafer import MassFunction, combine_dempster


# ── Helpers ─────────────────────────────────────────────────────────────


def _is_valid_mass(m: MassFunction, tol: float = 1e-6) -> bool:
    """Mass entries are non-negative and sum to 1 within ``tol``."""
    parts = (m.supports, m.refutes, m.uncertain, m.theta)
    if any(x < -tol for x in parts):
        return False
    total = sum(parts)
    return abs(total - 1.0) <= tol


def _high_confidence_supports() -> list[MassFunction]:
    """Three high-confidence agreeing masses: ~0.95 SUPPORTS, ~0.05 Θ."""
    return [
        MassFunction(supports=0.95, theta=0.05),
        MassFunction(supports=0.95, theta=0.05),
        MassFunction(supports=0.95, theta=0.05),
    ]


# ── Test 1: Dempster baseline collapses Θ (sanity for current code) ─────


def test_dempster_baseline_collapses_theta():
    """Three near-certain agreeing SUPPORTS masses → Dempster m(Θ) ~ 0.05^3.

    This exists to PASS today: it sanity-checks the existing baseline
    collapse behaviour that motivates the alternative rules. If this
    test ever fails, the regression is in ``combine_dempster`` itself.
    """
    masses = _high_confidence_supports()
    result = masses[0]
    for m in masses[1:]:
        result = combine_dempster(result, m)

    assert _is_valid_mass(result)
    # 0.05^3 = 1.25e-4; allow a generous ceiling.
    assert result.theta < 0.01, f"expected collapse, got m(Θ)={result.theta}"
    # And SUPPORTS should dominate.
    assert result.supports > 0.99


# ── Test 2: Yager preserves Θ (FAILS — combine_yager not implemented) ───


def test_yager_preserves_theta():
    """Yager should leave strictly more m(Θ) than Dempster on the same input.

    On three agreeing high-confidence supports, Yager's m(Θ) equals the
    product of the input m(Θ)s plus any conflict mass. On purely
    agreeing inputs there is zero conflict, so Yager m(Θ) = 0.05^3,
    which is identical to Dempster's m(Θ) — UNLESS the Yager
    implementation is correct: the rest of the agreement-mass
    accumulation gives the same renormalized SUPPORTS as Dempster only
    after the conflict-add-to-Θ rule. To make the comparison
    discriminating we use slightly disagreeing focal masses (a small
    REFUTES leak in one of the three) so there is non-zero conflict.
    """
    # Two high-confidence SUPPORTS, one with a small REFUTES leak:
    # creates non-zero conflict K so Yager and Dempster diverge.
    masses = [
        MassFunction(supports=0.90, theta=0.10),
        MassFunction(supports=0.85, refutes=0.05, theta=0.10),
        MassFunction(supports=0.90, theta=0.10),
    ]

    dempster_result = masses[0]
    for m in masses[1:]:
        dempster_result = combine_dempster(dempster_result, m)

    yager_result = dempster_shafer.combine_yager(masses[0], masses[1])
    yager_result = dempster_shafer.combine_yager(yager_result, masses[2])

    assert _is_valid_mass(yager_result)
    assert yager_result.theta > dempster_result.theta, (
        f"Yager m(Θ)={yager_result.theta} should exceed "
        f"Dempster m(Θ)={dempster_result.theta}"
    )


# ── Test 3: Murphy averaging produces a valid mass (FAILS) ──────────────


def test_murphy_averages():
    """Murphy 2000: average then n-1 Dempster combinations of the average."""
    masses = _high_confidence_supports()
    result = dempster_shafer.combine_murphy(masses)
    assert _is_valid_mass(result)
    # Three agreeing supports → Murphy should still favour SUPPORTS.
    assert result.supports > result.refutes
    assert result.supports > result.uncertain


# ── Test 4: top1 returns the most-decisive input ────────────────────────


def test_top1_returns_most_decisive():
    """top1 returns the single input mass with the smallest m(Θ)."""
    masses = [
        MassFunction(supports=0.40, theta=0.60),  # m(Θ)=0.60
        MassFunction(supports=0.80, theta=0.20),  # m(Θ)=0.20  ← most decisive
        MassFunction(supports=0.30, theta=0.70),  # m(Θ)=0.70
    ]
    result = dempster_shafer.combine_top1(masses)
    assert _is_valid_mass(result)
    # Should equal the second input (smallest theta).
    assert result.supports == pytest.approx(0.80)
    assert result.theta == pytest.approx(0.20)
    assert result.refutes == pytest.approx(0.0)
    assert result.uncertain == pytest.approx(0.0)


# ── Test 5: combine_multiple dispatches by rule ─────────────────────────


def test_dispatch_combine_multiple():
    """combine_multiple(masses, rule=...) dispatches across all four rules.

    With ``rule="dempster"`` (the default) this MUST preserve current
    behaviour — that is the regression guard for A.4b. The other three
    rules MUST fail today.
    """
    masses = _high_confidence_supports()

    # Default-rule call should match current behaviour (no kwarg).
    legacy = dempster_shafer.combine_multiple(masses)
    dispatched_dempster = dempster_shafer.combine_multiple(masses, rule="dempster")

    assert _is_valid_mass(legacy)
    assert _is_valid_mass(dispatched_dempster)
    assert dispatched_dempster.supports == pytest.approx(legacy.supports)
    assert dispatched_dempster.refutes == pytest.approx(legacy.refutes)
    assert dispatched_dempster.uncertain == pytest.approx(legacy.uncertain)
    assert dispatched_dempster.theta == pytest.approx(legacy.theta)

    # Non-default rules — each must produce a valid mass function.
    yager = dempster_shafer.combine_multiple(masses, rule="yager")
    murphy = dempster_shafer.combine_multiple(masses, rule="murphy")
    top1 = dempster_shafer.combine_multiple(masses, rule="top1")

    assert _is_valid_mass(yager)
    assert _is_valid_mass(murphy)
    assert _is_valid_mass(top1)


# ── Test 6: pathological inputs across all rules ────────────────────────


@pytest.mark.parametrize("rule", ["dempster", "yager", "murphy", "top1"])
def test_pathological_inputs(rule):
    """Each rule must produce a valid mass on pathological inputs.

    Cases:
    (i)  partial-conflict pair: m1=[0.9,0,0,0.1], m2=[0,0.9,0,0.1]
    (ii) single-mass list
    (iii) two identical masses

    Note on Dempster + total conflict: ``combine_dempster`` already
    returns the vacuous mass (theta=1.0) when normalization would
    divide by zero (see dempster_shafer.py lines 154-158). It does
    NOT raise. We therefore only require a valid mass function out
    on partial-conflict, single-mass, and identical inputs.
    """
    # (i) partial-conflict pair
    pair = [
        MassFunction(supports=0.9, theta=0.1),
        MassFunction(refutes=0.9, theta=0.1),
    ]
    result_i = dempster_shafer.combine_multiple(pair, rule=rule)
    assert _is_valid_mass(result_i), f"{rule} produced invalid mass on conflict pair"

    # (ii) single-mass list — should be a no-op or return the input unchanged
    single = [MassFunction(refutes=0.7, theta=0.3)]
    result_ii = dempster_shafer.combine_multiple(single, rule=rule)
    assert _is_valid_mass(result_ii), f"{rule} produced invalid mass on single input"
    # Single-mass invariant: most rules should preserve the input.
    # We assert mass-validity only here; per-rule semantics are tested above.

    # (iii) two identical masses
    twin = [
        MassFunction(supports=0.6, theta=0.4),
        MassFunction(supports=0.6, theta=0.4),
    ]
    result_iii = dempster_shafer.combine_multiple(twin, rule=rule)
    assert _is_valid_mass(result_iii), f"{rule} produced invalid mass on twin input"

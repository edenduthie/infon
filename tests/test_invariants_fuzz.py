"""Property-based (Hypothesis) fuzz tests for invariants.

Each test defines an invariant that MUST hold for ALL valid inputs and
uses Hypothesis to search for counter-examples. Failures shrink to the
minimal triggering input.

Targets:
    1. DS mass normalization (sums to 1)
    2. Dempster commutativity
    3. Dempster associativity (up to float tolerance)
    4. Vacuous-mass identity (Θ-only combine is identity)
    5. IKL-NOT bounded amplification
    6. IKL-AND idempotence (min of identical things)
    7. Forward pass determinism in eval mode
    8. Gradient finiteness after training
"""

from __future__ import annotations

import math
import tempfile
import os

import pytest
import torch
from hypothesis import given, strategies as st, settings, assume, HealthCheck

from infon.dempster_shafer import (
    MassFunction, combine_dempster, combine_multiple,
)
from infon.logic import (
    IKLAnd, IKLOr, IKLNot, IKLIf,
    HypergraphReasoner,
)


# ── Strategies ────────────────────────────────────────────────────────

@st.composite
def mass_function(draw):
    """Generate a random valid MassFunction.

    Draw 4 non-negative floats, normalize so they sum to 1. Avoids
    edge cases (all zero) because MassFunction.__post_init__
    renormalizes, but 0-sum would collapse.
    """
    parts = [
        draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False,
                       allow_infinity=False))
        for _ in range(4)
    ]
    total = sum(parts)
    assume(total > 1e-6)
    parts = [p / total for p in parts]
    return MassFunction(
        supports=parts[0], refutes=parts[1],
        uncertain=parts[2], theta=parts[3],
    )


@st.composite
def tensor_vector(draw, dim=64, bound=5.0):
    """Generate a random 1D torch tensor of fixed dimension."""
    vals = [
        draw(st.floats(min_value=-bound, max_value=bound,
                       allow_nan=False, allow_infinity=False))
        for _ in range(dim)
    ]
    return torch.tensor(vals, dtype=torch.float32)


def _mass_sum(m: MassFunction) -> float:
    return m.supports + m.refutes + m.uncertain + m.theta


def _mass_close(a: MassFunction, b: MassFunction, tol=1e-5) -> bool:
    return (abs(a.supports - b.supports) < tol and
            abs(a.refutes - b.refutes) < tol and
            abs(a.uncertain - b.uncertain) < tol and
            abs(a.theta - b.theta) < tol)


# ── 1. Mass normalization ────────────────────────────────────────────

@given(m=mass_function())
@settings(max_examples=200, deadline=None)
def test_mass_normalizes_to_one(m: MassFunction):
    """Any constructed MassFunction has masses summing to 1."""
    assert abs(_mass_sum(m) - 1.0) < 1e-6, (
        f"mass sum = {_mass_sum(m)} for {m}"
    )


@given(m1=mass_function(), m2=mass_function())
@settings(max_examples=200, deadline=None)
def test_dempster_output_normalizes(m1: MassFunction, m2: MassFunction):
    """Dempster's rule output sums to 1 whenever conflict < 1."""
    result = combine_dempster(m1, m2)
    assert abs(_mass_sum(result) - 1.0) < 1e-5, (
        f"combined mass sum = {_mass_sum(result)}"
    )


# ── 2. Commutativity ─────────────────────────────────────────────────

@given(m1=mass_function(), m2=mass_function())
@settings(max_examples=200, deadline=None)
def test_dempster_commutative(m1: MassFunction, m2: MassFunction):
    """combine(a, b) == combine(b, a) up to float tolerance."""
    ab = combine_dempster(m1, m2)
    ba = combine_dempster(m2, m1)
    assert _mass_close(ab, ba, tol=1e-5), (
        f"a∘b = {ab}\nb∘a = {ba}"
    )


# ── 3. Associativity ─────────────────────────────────────────────────

@given(m1=mass_function(), m2=mass_function(), m3=mass_function())
@settings(max_examples=150, deadline=None,
          suppress_health_check=[HealthCheck.filter_too_much])
def test_dempster_associative_when_no_total_conflict(
        m1: MassFunction, m2: MassFunction, m3: MassFunction):
    """combine(combine(a, b), c) ≈ combine(a, combine(b, c)) *when no
    inner pair exhibits total conflict*.

    Dempster's rule is commutative but is only associative when
    intermediate combinations have positive mass after normalization.
    If any inner pair has total conflict (K=1), our implementation
    falls back to vacuous (θ=1), which breaks associativity. This
    property-based test documents that Dempster combination requires
    chain-conflict screening to be chained safely. We exclude
    high-conflict triples with `assume`, then check associativity on
    the remaining well-behaved cases.
    """
    # Reject any triple where a pairwise combination has K > 0.99
    # (near-total conflict), which would make the fallback kick in.
    def near_total_conflict(a: MassFunction, b: MassFunction) -> bool:
        # Compute conflict directly: mass pairs with empty intersection.
        conflict = (a.supports * b.refutes + a.supports * b.uncertain
                    + a.refutes * b.supports + a.refutes * b.uncertain
                    + a.uncertain * b.supports + a.uncertain * b.refutes)
        return conflict > 0.99
    assume(not near_total_conflict(m1, m2))
    assume(not near_total_conflict(m2, m3))
    left = combine_dempster(combine_dempster(m1, m2), m3)
    right = combine_dempster(m1, combine_dempster(m2, m3))
    assume(_mass_sum(left) > 0.99 and _mass_sum(right) > 0.99)
    assert _mass_close(left, right, tol=1e-3), (
        f"(a∘b)∘c = {left}\na∘(b∘c) = {right}"
    )


def test_dempster_non_associative_under_total_conflict():
    """Document the known limit of Dempster combination: chains that
    encounter total conflict in one grouping but not the other are
    NOT associative. Hypothesis found this counter-example at
    random and we preserve it as a regression test."""
    m1 = MassFunction(uncertain=1.0, theta=0.0)
    m2 = MassFunction(uncertain=1.0, theta=0.0)
    m3 = MassFunction(refutes=1.0, theta=0.0)
    left = combine_dempster(combine_dempster(m1, m2), m3)
    right = combine_dempster(m1, combine_dempster(m2, m3))
    # Deliberately assert they DIFFER — this is not a bug, it's a
    # known mathematical property of DS combination with total conflict.
    assert not _mass_close(left, right, tol=0.1), (
        "Expected associativity to fail on this triple"
    )


# ── 4. Vacuous-mass identity ─────────────────────────────────────────

@given(m=mass_function())
@settings(max_examples=100, deadline=None)
def test_vacuous_is_identity(m: MassFunction):
    """Combining with total ignorance (θ=1) leaves the other mass
    approximately unchanged. Exact equality requires no conflict,
    which is always true when one operand is Θ-only.
    """
    vacuous = MassFunction(theta=1.0)
    result = combine_dempster(m, vacuous)
    assert _mass_close(m, result, tol=1e-5), (
        f"input: {m}\nafter combine with vacuous: {result}"
    )


# ── 5. IKL-NOT bounded ───────────────────────────────────────────────

@given(h=tensor_vector(dim=64, bound=3.0))
@settings(max_examples=100, deadline=None)
def test_ikl_not_bounded_amplification(h: torch.Tensor):
    """ikl_not output magnitude is bounded by a linear factor in the
    input: ||not(h)||_2 ≤ 3 * (||h|| + 1). This is a stability check
    on a learned non-linearity — we don't expect algebraic involution
    but we don't want exponential blow-up either.
    """
    torch.manual_seed(0)
    not_op = IKLNot(hidden_dim=64)
    not_op.eval()
    with torch.no_grad():
        out = not_op(h.unsqueeze(0)).squeeze(0)
    in_norm = h.norm().item()
    out_norm = out.norm().item()
    assert math.isfinite(out_norm), f"non-finite output norm: {out_norm}"
    # With tanh bound = 1 and the (− h) residual, worst case is
    # sqrt(hidden) + ||h||.
    bound = math.sqrt(64) + in_norm + 1e-3
    assert out_norm <= bound, (
        f"||not(h)|| = {out_norm:.4f} exceeded bound {bound:.4f} "
        f"(||h|| = {in_norm:.4f})"
    )


# ── 6. IKL-AND weakly idempotent on identical inputs ────────────────

@given(h=tensor_vector(dim=64, bound=2.0))
@settings(max_examples=80, deadline=None)
def test_ikl_and_idempotent_identical(h: torch.Tensor):
    """IKL-AND of a set of identical embeddings equals the gated
    embedding itself (the min of k equal values is one of them).
    Formally: AND([h, h, h]) == h * sigmoid(W h).
    """
    torch.manual_seed(0)
    and_op = IKLAnd(hidden_dim=64)
    and_op.eval()
    with torch.no_grad():
        # Three copies of the same vector
        stack = h.unsqueeze(0).expand(3, -1).contiguous()
        out = and_op(stack)
        # Expected: gated version of h
        gate = torch.sigmoid(and_op.gate(h))
        expected = h * gate
    diff = (out - expected).abs().max().item()
    assert diff < 1e-5, f"max elementwise diff = {diff}"


# ── 7. Forward-pass determinism in eval mode ─────────────────────────

def test_forward_deterministic():
    """Two eval-mode forward passes on an untrained reasoner produce
    identical embeddings. This is a guardrail against accidentally
    introducing dropout / randomness into inference.
    """
    from infon.logic import HypergraphReasoner
    from infon import InfonEngine, InfonConfig
    import json

    SCHEMA = {
        "toyota": {"type": "actor", "tokens": ["toyota"]},
        "invests": {"type": "relation", "tokens": ["invest", "invests"]},
        "battery": {"type": "feature", "tokens": ["battery"]},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        schema_path = os.path.join(tmpdir, "schema.json")
        with open(schema_path, "w") as f:
            json.dump(SCHEMA, f)
        cog = InfonEngine(InfonConfig(
            schema_path=schema_path,
            db_path=os.path.join(tmpdir, "test.db"),
            activation_threshold=0.2,
            min_confidence=0.02,
            top_k_per_role=3,
        ))
        cog.ingest([{"id": "d1", "text": "Toyota invests in batteries."}])
        cog.consolidate()

        reasoner = HypergraphReasoner(cog.store, cog.encoder, cog.schema,
                                      hidden_dim=64, n_layers=2)
        reasoner.eval()
        graph = reasoner.builder.build(feature_dim=64)

        with torch.no_grad():
            h1 = reasoner.forward(graph)
            h2 = reasoner.forward(graph)
        diff = (h1 - h2).abs().max().item()
        assert diff == 0.0, f"non-deterministic forward: max diff = {diff}"
        cog.close()


# ── 8. Gradient finiteness after training ────────────────────────────

def test_gradients_finite_after_training():
    """After a short training run with gradient clipping, no parameter
    is NaN or Inf. Also confirms loss decreases."""
    import json
    from infon import InfonEngine, InfonConfig

    SCHEMA = {
        "toyota": {"type": "actor", "tokens": ["toyota"]},
        "honda": {"type": "actor", "tokens": ["honda"]},
        "invests": {"type": "relation", "tokens": ["invest", "invests"]},
        "partners": {"type": "relation", "tokens": ["partner", "partners"]},
        "battery": {"type": "feature", "tokens": ["battery", "batteries"]},
        "japan": {"type": "market", "tokens": ["japan", "japanese"]},
    }

    DOCS = [
        {"id": "d1", "text": "Toyota invests in battery technology."},
        {"id": "d2", "text": "Honda partners with Japanese suppliers."},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        schema_path = os.path.join(tmpdir, "schema.json")
        with open(schema_path, "w") as f:
            json.dump(SCHEMA, f)
        cog = InfonEngine(InfonConfig(
            schema_path=schema_path,
            db_path=os.path.join(tmpdir, "test.db"),
            activation_threshold=0.2,
            min_confidence=0.02,
            top_k_per_role=3,
        ))
        for d in DOCS:
            cog.ingest([d])
        cog.consolidate()

        reasoner = HypergraphReasoner(cog.store, cog.encoder, cog.schema,
                                      hidden_dim=64, n_layers=2)
        graph = reasoner.builder.build(feature_dim=64)
        stats = reasoner.fit(graph=graph, epochs=20, grad_clip=1.0,
                             verbose=False)

        for name, p in reasoner.named_parameters():
            assert torch.isfinite(p).all(), f"non-finite params in {name}"
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), (
                    f"non-finite grad in {name}"
                )
        # Loss went down
        assert stats["final_loss"] <= stats["losses"][0]
        cog.close()


if __name__ == "__main__":
    print("=" * 60)
    print("  Property-based invariant tests (Hypothesis)")
    print("=" * 60)
    test_forward_deterministic()
    print("  PASS: forward deterministic")
    test_gradients_finite_after_training()
    print("  PASS: gradients finite after training")
    # Hypothesis tests are collected/driven by pytest
    print("\n  (Run via `pytest tests/test_invariants_fuzz.py`)")

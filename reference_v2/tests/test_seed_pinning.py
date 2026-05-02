"""Red-phase test for seed pinning in HypergraphReasoner.fit().

This test currently FAILS — that is the point. The released
``HypergraphReasoner.fit()`` does not accept a ``seed=`` kwarg, so the
first ``fit(seed=42)`` call raises ``TypeError``. Once A.2b lands the
parameter (pinning torch / numpy / random / CUDA RNGs from inside
``fit()``), the second-half assertions kick in and verify *bit-identical*
loss traces and final mass functions across two independent runs.

Reference: openspec/changes/epic-01-stabilize-theta/spec.md
           Requirement: Deterministic Reproduction
           tasks.md A.2 (red) → A.2b (green)
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import torch

# Match the corpus and schema used by tests/test_logic.py so this test is
# structurally compatible with the existing 14-test regression suite.
from tests.test_logic import DOCUMENTS, setup_cognition


# Diagnostic query reused by both runs; matches the audit's Toyota probe.
_TOYOTA_QUERY = "Did Toyota invest in battery technology?"


def _build_and_ingest(db_path: str):
    """Construct a fresh Cognition + ingest the EV corpus."""
    cog = setup_cognition(db_path)
    for doc in DOCUMENTS:
        cog.ingest([doc])
    cog.consolidate()
    return cog


def _build_reasoner(cog):
    """Construct a fresh HypergraphReasoner over an already-ingested store."""
    from cognition.logic import HypergraphReasoner

    return HypergraphReasoner(
        cog.store, cog.encoder, cog.schema,
        hidden_dim=64, n_layers=2,
    )


def _mass_array(mass) -> np.ndarray:
    """Pack a MassFunction into a 4-vector for bit-equal comparison."""
    return np.array(
        [mass.supports, mass.refutes, mass.uncertain, mass.theta],
        dtype=np.float64,
    )


def test_fit_accepts_seed_and_is_deterministic():
    """fit(seed=42) must (a) accept the kwarg and (b) be bit-reproducible.

    Phase A.2 (RED): expected failure mode is
        TypeError: fit() got an unexpected keyword argument 'seed'
    raised at the first ``fit(seed=42)`` call below.

    Phase A.2b (GREEN): once the seed parameter pins all RNGs, both halves
    of this test pass — the loss trace and final Toyota mass are
    bit-identical across two independent reasoner builds.
    """
    with tempfile.TemporaryDirectory() as tmpdir_a, \
            tempfile.TemporaryDirectory() as tmpdir_b:

        # ── First run ─────────────────────────────────────────────────
        cog_a = _build_and_ingest(os.path.join(tmpdir_a, "test_a.db"))
        try:
            reasoner_a = _build_reasoner(cog_a)
            graph_a = reasoner_a.builder.build(feature_dim=64)
            stats_a = reasoner_a.fit(graph=graph_a, epochs=30, seed=42)
            losses_a = np.asarray(stats_a["losses"], dtype=np.float64)
            result_a = reasoner_a.reason(_TOYOTA_QUERY)
            mass_a = _mass_array(result_a.mass)
        finally:
            cog_a.close()

        # ── Second run (fresh reasoner, same seed) ────────────────────
        cog_b = _build_and_ingest(os.path.join(tmpdir_b, "test_b.db"))
        try:
            reasoner_b = _build_reasoner(cog_b)
            graph_b = reasoner_b.builder.build(feature_dim=64)
            stats_b = reasoner_b.fit(graph=graph_b, epochs=30, seed=42)
            losses_b = np.asarray(stats_b["losses"], dtype=np.float64)
            result_b = reasoner_b.reason(_TOYOTA_QUERY)
            mass_b = _mass_array(result_b.mass)
        finally:
            cog_b.close()

        # ── Bit-identity assertions (per spec.md: bit-equal, not allclose) ──
        assert losses_a.shape == losses_b.shape, (
            f"seeded fits produced different loss-trace lengths: "
            f"a={losses_a.shape}, b={losses_b.shape}"
        )
        assert np.array_equal(losses_a, losses_b), (
            "seeded fits diverged on loss trace (bit-equal expected per "
            "spec.md Requirement: Deterministic Reproduction):\n"
            f"  a = {losses_a}\n"
            f"  b = {losses_b}\n"
            f"  max |a - b| = {np.abs(losses_a - losses_b).max()}"
        )
        assert np.array_equal(mass_a, mass_b), (
            "seeded fits diverged on Toyota mass (bit-equal expected per "
            "spec.md Requirement: Deterministic Reproduction):\n"
            f"  a = {mass_a}\n"
            f"  b = {mass_b}\n"
            f"  max |a - b| = {np.abs(mass_a - mass_b).max()}"
        )


if __name__ == "__main__":
    test_fit_accepts_seed_and_is_deterministic()
    print("PASS: seed pinning is deterministic")

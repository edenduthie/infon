"""TDD RED test for readout swap config field.

Verifies that ``CognitionConfig`` accepts a ``readout`` field with values in
``{"ds_4mass", "softmax_temperature", "dirichlet_edl"}``.

This file is intentionally RED: ``CognitionConfig`` does NOT have a ``readout``
field yet. The tests will fail with ``TypeError`` (unexpected keyword argument)
until the field is added in the GREEN phase.

Spec reference:
    openspec/changes/epic-02-synthetic-stress/spec.md
    Config.readout: Literal["ds_4mass", "softmax_temperature", "dirichlet_edl"]
    - ds_4mass            — current 4-element mass head (default)
    - softmax_temperature — 3-way softmax with post-hoc temperature scaling
    - dirichlet_edl       — evidential Dirichlet head per Sensoy et al. 2018

Each readout SHALL produce, on every query, a four-element confidence vector
[m(S), m(R), m(U), m(Theta)].

Task: infon-8pa.7 (Epic 02, A.4)
"""

from __future__ import annotations

import os
import tempfile

import pytest

from experiments.ev_corpus import DIAGNOSTIC_QUERIES
from tests.test_logic import DOCUMENTS, setup_cognition
from cognition.logic import HypergraphReasoner
from cognition.config import CognitionConfig

# The three valid readout identifiers per spec
_READOUT_VALUES = ("ds_4mass", "softmax_temperature", "dirichlet_edl")

# A single diagnostic query text (toyota) is enough to test 4-vec output
_QUERY_TEXT = DIAGNOSTIC_QUERIES["toyota"]


# ── Test 1: CognitionConfig accepts readout field ────────────────────────────

def test_readout_config_field_exists():
    """CognitionConfig must accept readout="ds_4mass" without TypeError.

    RED: Until the ``readout`` field is added to ``CognitionConfig`` this test
    will raise TypeError("__init__() got an unexpected keyword argument
    'readout'"), which causes the test to fail.
    """
    # This line raises TypeError until CognitionConfig.readout is defined.
    cfg = CognitionConfig(readout="ds_4mass")
    assert cfg.readout == "ds_4mass"


# ── Test 2: each readout produces a four-element mass vector ─────────────────

@pytest.mark.parametrize("readout", _READOUT_VALUES)
def test_readout_produces_4vec_per_readout(readout: str):
    """Fitting a reasoner with each readout type must yield a valid 4-vec.

    The returned ``result.mass`` must carry all four components:
    supports, refutes, uncertain, theta — and they must sum to 1.0.

    RED: Fails because (a) ``CognitionConfig`` has no ``readout`` field and
    (b) ``HypergraphReasoner`` does not route on it yet.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, f"readout_{readout}.db")

        # Setup cognition using the shared EV corpus helper (no readout yet
        # on CognitionConfig — this will raise TypeError before we even
        # reach the reasoner).
        cog = setup_cognition(db_path)
        for doc in DOCUMENTS:
            cog.ingest([doc])
        cog.consolidate()

        try:
            # Constructing CognitionConfig with readout will TypeError (RED).
            cfg = CognitionConfig(readout=readout)

            reasoner = HypergraphReasoner(
                cog.store, cog.encoder, cog.schema,
                hidden_dim=64, n_layers=2,
            )
            graph = reasoner.builder.build(feature_dim=64)
            reasoner.fit(graph=graph, epochs=5, seed=42)

            result = reasoner.reason(_QUERY_TEXT)
            m = result.mass

            # All four components must be present and non-negative.
            assert hasattr(m, "supports"), "mass missing 'supports'"
            assert hasattr(m, "refutes"), "mass missing 'refutes'"
            assert hasattr(m, "uncertain"), "mass missing 'uncertain'"
            assert hasattr(m, "theta"), "mass missing 'theta'"

            assert m.supports >= 0.0, f"[{readout}] supports < 0: {m.supports}"
            assert m.refutes >= 0.0, f"[{readout}] refutes < 0: {m.refutes}"
            assert m.uncertain >= 0.0, f"[{readout}] uncertain < 0: {m.uncertain}"
            assert m.theta >= 0.0, f"[{readout}] theta < 0: {m.theta}"

            total = m.supports + m.refutes + m.uncertain + m.theta
            assert abs(total - 1.0) < 0.01, (
                f"[{readout}] mass vector does not sum to 1.0; got {total:.6f} "
                f"(S={m.supports:.4f} R={m.refutes:.4f} "
                f"U={m.uncertain:.4f} Θ={m.theta:.4f})"
            )
        finally:
            cog.close()


# ── Test 3: softmax_temperature still produces a 4-vec ───────────────────────

def test_softmax_temperature_produces_4vec():
    """softmax_temperature readout must emit a 4-vec [S, R, U, Theta].

    Per spec, even though softmax_temperature is a 3-way head (S/R/U),
    it must still surface a 4-element confidence vector. m(U) and m(Theta)
    may both be zero by construction, but all four keys must exist and
    the vector must sum to 1.0.

    RED: Fails because CognitionConfig has no readout field yet.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "readout_softmax.db")
        cog = setup_cognition(db_path)
        for doc in DOCUMENTS:
            cog.ingest([doc])
        cog.consolidate()

        try:
            # TypeError until CognitionConfig gains readout field.
            cfg = CognitionConfig(readout="softmax_temperature")

            reasoner = HypergraphReasoner(
                cog.store, cog.encoder, cog.schema,
                hidden_dim=64, n_layers=2,
            )
            graph = reasoner.builder.build(feature_dim=64)
            reasoner.fit(graph=graph, epochs=5, seed=42)

            result = reasoner.reason(_QUERY_TEXT)
            m = result.mass

            # 4-vec completeness check (same as parametrized test above).
            assert hasattr(m, "supports"), "softmax_temperature: mass missing 'supports'"
            assert hasattr(m, "refutes"), "softmax_temperature: mass missing 'refutes'"
            assert hasattr(m, "uncertain"), "softmax_temperature: mass missing 'uncertain'"
            assert hasattr(m, "theta"), "softmax_temperature: mass missing 'theta'"

            total = m.supports + m.refutes + m.uncertain + m.theta
            assert abs(total - 1.0) < 0.01, (
                f"softmax_temperature mass does not sum to 1.0; got {total:.6f}"
            )
        finally:
            cog.close()

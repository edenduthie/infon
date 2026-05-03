"""Test gating of the per-infon mass logger on the ``log_per_infon_masses`` flag.

A.3b (commit ``f14580f``) added ``CognitionConfig.log_per_infon_masses`` and
``ReasoningResult.per_infon_masses`` but emitted records *unconditionally*.
The flag was preserved as the documented API contract surface and explicitly
flagged as a deferred fix in the Epic 01 Learnings.

infon-6o3.36 (this task) actually wires the gate: when the flag is False
(default), ``ReasoningResult.per_infon_masses`` is the empty list ``[]``;
when True, it is populated as before.

This is the new contract:

    * ``HypergraphReasoner(log_per_infon_masses=False)`` (the default)
      returns ``result.per_infon_masses == []`` even on inputs that would
      otherwise produce many contributors.
    * ``HypergraphReasoner(log_per_infon_masses=True)`` returns the
      pre-fusion ``PerInfonMassRecord`` log unchanged from A.3b.

The gate is fed from ``CognitionConfig.log_per_infon_masses`` by callers
that build a reasoner from a config (``experiments/run.py``,
``experiments/results/diagnostic/per_infon_analysis.py``); this test
exercises the construct-from-config path end-to-end as well.

Reference:
    openspec/changes/epic-01-stabilize-theta/spec.md
        Requirement: Per-Infon Mass Logging
    A.3b commit f14580f (the original Option A choice — emit unconditionally)
    infon-6o3.36 (this task — actually gate emission on the flag)
"""

from __future__ import annotations

import json
import os
import tempfile

from tests.test_logic import DOCUMENTS, SCHEMA_DEFS, setup_cognition


# Diagnostic query reused from ``test_per_infon_mass_logger.py`` — the EV
# corpus contains explicit Toyota/battery infons so the contributor list
# is non-empty when the flag is True.
_TOYOTA_QUERY = "Did Toyota invest in battery technology?"


def _build_cognition_with_flag(db_path: str, log_per_infon_masses: bool):
    """Construct a Cognition instance with a specific value of the flag.

    Mirrors ``tests/test_per_infon_mass_logger.py::_build_cognition_with_logger``
    but parameterized so we can exercise both flag values via the
    construct-from-config path.
    """
    from cognition import Cognition, CognitionConfig

    schema_path = db_path.replace(".db", "_schema.json")
    with open(schema_path, "w") as f:
        json.dump(SCHEMA_DEFS, f)

    config = CognitionConfig(
        schema_path=schema_path,
        db_path=db_path,
        activation_threshold=0.2,
        min_confidence=0.02,
        top_k_per_role=3,
        log_per_infon_masses=log_per_infon_masses,
    )
    return Cognition(config)


def _ingest(cog) -> None:
    for doc in DOCUMENTS:
        cog.ingest([doc])
    cog.consolidate()


def _reason_via_direct_constructor(log_per_infon_masses: bool):
    """Build a reasoner by passing the flag directly to ``HypergraphReasoner``.

    This is the path used by ``experiments/run.py`` — caller reads the flag
    out of the config and forwards it to the reasoner kwarg.
    """
    from cognition.logic import HypergraphReasoner

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "direct.db")
        # Flag value at the ``Cognition`` level is irrelevant for this path;
        # only the kwarg on ``HypergraphReasoner`` matters.
        cog = setup_cognition(db_path)
        try:
            _ingest(cog)
            reasoner = HypergraphReasoner(
                cog.store, cog.encoder, cog.schema,
                hidden_dim=64, n_layers=2,
                log_per_infon_masses=log_per_infon_masses,
            )
            graph = reasoner.builder.build(feature_dim=64)
            reasoner.fit(graph=graph, epochs=30, seed=42)
            return reasoner.reason(_TOYOTA_QUERY)
        finally:
            cog.close()


def _reason_via_config(log_per_infon_masses: bool):
    """Build a reasoner by reading the flag out of ``CognitionConfig``.

    Mirrors the production flow in ``experiments/run.py``: caller wires
    ``cog.config.log_per_infon_masses`` into the reasoner kwarg.
    """
    from cognition.logic import HypergraphReasoner

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "config.db")
        cog = _build_cognition_with_flag(db_path, log_per_infon_masses)
        try:
            _ingest(cog)
            # This is the threading we want to verify — caller reads the
            # flag out of ``cog.config`` and forwards to the reasoner.
            reasoner = HypergraphReasoner(
                cog.store, cog.encoder, cog.schema,
                hidden_dim=64, n_layers=2,
                log_per_infon_masses=cog.config.log_per_infon_masses,
            )
            graph = reasoner.builder.build(feature_dim=64)
            reasoner.fit(graph=graph, epochs=30, seed=42)
            return reasoner.reason(_TOYOTA_QUERY)
        finally:
            cog.close()


# ── Direct-constructor cases ────────────────────────────────────────────


def test_direct_constructor_flag_false_yields_empty_log():
    """``HypergraphReasoner(log_per_infon_masses=False)`` MUST emit ``[]``.

    The default behaviour (no flag set) is now an empty per-infon log.
    Production callers that don't opt in pay zero cost for the diagnostic.
    """
    result = _reason_via_direct_constructor(log_per_infon_masses=False)
    assert result.per_infon_masses == [], (
        f"Expected empty per_infon_masses with log_per_infon_masses=False; "
        f"got {len(result.per_infon_masses)} records. The flag is not "
        f"actually gating emission inside reason()."
    )


def test_direct_constructor_flag_true_yields_populated_log():
    """``HypergraphReasoner(log_per_infon_masses=True)`` MUST emit records.

    The Toyota query has multiple supporting infons in the EV corpus, so
    the log MUST be non-empty when the flag is True.
    """
    result = _reason_via_direct_constructor(log_per_infon_masses=True)
    assert len(result.per_infon_masses) >= 1, (
        f"Expected non-empty per_infon_masses with log_per_infon_masses=True; "
        f"got {len(result.per_infon_masses)} records. The Toyota query has "
        f"at least one supporting contributor in the EV corpus."
    )


# ── Construct-from-config cases ─────────────────────────────────────────


def test_config_flag_false_yields_empty_log():
    """``CognitionConfig(log_per_infon_masses=False)`` MUST emit ``[]``.

    The flag on the config dataclass MUST flow through to the reasoner
    when the caller wires it up via ``cog.config.log_per_infon_masses``.
    """
    result = _reason_via_config(log_per_infon_masses=False)
    assert result.per_infon_masses == [], (
        f"Expected empty per_infon_masses with "
        f"CognitionConfig(log_per_infon_masses=False); got "
        f"{len(result.per_infon_masses)} records."
    )


def test_config_flag_true_yields_populated_log():
    """``CognitionConfig(log_per_infon_masses=True)`` MUST emit records."""
    result = _reason_via_config(log_per_infon_masses=True)
    assert len(result.per_infon_masses) >= 1, (
        f"Expected non-empty per_infon_masses with "
        f"CognitionConfig(log_per_infon_masses=True); got "
        f"{len(result.per_infon_masses)} records."
    )


if __name__ == "__main__":
    test_direct_constructor_flag_false_yields_empty_log()
    test_direct_constructor_flag_true_yields_populated_log()
    test_config_flag_false_yields_empty_log()
    test_config_flag_true_yields_populated_log()
    print("PASS: per-infon mass logger gating")

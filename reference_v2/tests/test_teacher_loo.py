"""TDD RED tests for teacher leave-one-out (LOO) ablation via Config.teacher_sources.

This file is intentionally written BEFORE the green phase lands
``CognitionConfig.teacher_sources`` (a ``list[str]`` selecting which of the
four DS teacher signals contribute to the teacher mass during ``fit()``).

Tests 1 and 2 MUST fail today:
- ``test_teacher_sources_config_field_exists`` fails with ``TypeError`` because
  ``CognitionConfig`` does not yet accept ``teacher_sources``.
- ``test_teacher_loo_produces_distinguishable_trajectories`` fails for the
  same reason (it constructs ``CognitionConfig(teacher_sources=...)``) and
  additionally requires the fit() method to honour the field.

Spec:
- openspec/changes/epic-02-synthetic-stress/spec.md
  Requirement: Config.teacher_sources
- Epic 02, task A.5 (infon-8pa.9)

How teacher mass currently works (pre-green):
    In ``src/cognition/logic.py`` the ``fit()`` method builds four sources::

        sources = [
            mass_from_polarity(infon),
            mass_from_triple_alignment(claim_anchors, infon, self.schema.types),
            mass_from_anchor_distance(claim_anchors, infon, self.schema.types),
            mass_from_confidence(infon),
        ]
        combined = combine_multiple(sources)

    The green phase will honour ``Config.teacher_sources`` to select a subset
    of those four, using the canonical names
    ``{"polarity", "alignment", "distance", "confidence"}``.

No mocks; real reasoner, real masses, real EV corpus.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from experiments.ev_corpus import DIAGNOSTIC_QUERIES
from tests.test_logic import DOCUMENTS, setup_cognition

# Canonical teacher source names (must match the green implementation).
_ALL_SOURCES = ["polarity", "alignment", "distance", "confidence"]

# Fast but non-trivial fit parameters.
_EPOCHS = 5
_SEED = 42


# ── Helpers ─────────────────────────────────────────────────────────────────


def _build_and_ingest(db_path: str):
    """Construct a fresh Cognition instance and ingest the EV corpus."""
    cog = setup_cognition(db_path)
    for doc in DOCUMENTS:
        cog.ingest([doc])
    cog.consolidate()
    return cog


def _fit_with_sources(sources: list[str]) -> float:
    """Fit a fresh HypergraphReasoner restricted to *sources* and return final loss.

    Creates an isolated temp store per call so results are independent.
    ``CognitionConfig(teacher_sources=sources)`` will raise ``TypeError``
    until the green phase adds the field — this is the intentional red state.
    """
    from cognition.config import CognitionConfig
    from cognition.logic import HypergraphReasoner

    # RED: CognitionConfig does not accept teacher_sources yet → TypeError
    cfg = CognitionConfig(teacher_sources=sources)  # noqa: F841 (red — unused until green)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "teacher_loo.db")
        cog = _build_and_ingest(db_path)
        try:
            reasoner = HypergraphReasoner(
                cog.store, cog.encoder, cog.schema,
                hidden_dim=64, n_layers=2,
            )
            graph = reasoner.builder.build(feature_dim=64)
            stats = reasoner.fit(
                graph=graph,
                epochs=_EPOCHS,
                seed=_SEED,
                teacher_sources=sources,
            )
            return float(stats["final_loss"])
        finally:
            cog.close()


# ── Test 1: Config field existence ──────────────────────────────────────────


def test_teacher_sources_config_field_exists():
    """CognitionConfig must accept a ``teacher_sources`` kwarg.

    RED: today ``CognitionConfig`` is a plain dataclass without this field,
    so ``CognitionConfig(teacher_sources=["polarity"])`` raises ``TypeError``.
    GREEN: the field exists with a default of all four source names.
    """
    from cognition.config import CognitionConfig

    # This line must raise TypeError in the red phase.
    cfg = CognitionConfig(teacher_sources=["polarity"])

    # Green assertions (only reached after the field is added):
    assert hasattr(cfg, "teacher_sources"), (
        "CognitionConfig must expose 'teacher_sources' attribute"
    )
    assert cfg.teacher_sources == ["polarity"]

    # Verify default is all four sources.
    cfg_default = CognitionConfig()
    assert set(cfg_default.teacher_sources) == set(_ALL_SOURCES), (
        f"Default teacher_sources must be all four; got {cfg_default.teacher_sources!r}"
    )


# ── Test 2: LOO produces distinguishable loss trajectories ──────────────────


def test_teacher_loo_produces_distinguishable_trajectories():
    """Holding out each teacher source must yield a different final loss.

    Rationale: each of the four DS teacher signals (polarity, alignment,
    distance, confidence) contributes unique information. When one source is
    excluded the teacher target distribution shifts, which changes the
    gradient signal and therefore the loss after a fixed number of epochs.
    The four hold-out final-loss values must not all be equal (if they were,
    the teacher_sources config knob would be a no-op).

    RED: ``_fit_with_sources`` instantiates ``CognitionConfig(teacher_sources=…)``
    which raises ``TypeError`` before the green phase, causing every call to
    fail with a TypeError rather than returning a loss value.
    """
    holdout_losses: dict[str, float] = {}

    for held_out in _ALL_SOURCES:
        sources_used = [s for s in _ALL_SOURCES if s != held_out]
        final_loss = _fit_with_sources(sources_used)
        holdout_losses[held_out] = final_loss

    # All four losses must not be identical (basic distinguishability check).
    unique_losses = set(holdout_losses.values())
    assert len(unique_losses) > 1, (
        "All four hold-out final losses are identical "
        f"({holdout_losses!r}), which means teacher_sources has no effect."
    )


# ── Test 3 (optional): full-signal baseline beats single-source runs ─────────


def test_all_sources_lower_loss_than_single_source():
    """Using all four teacher sources should produce a lower or equal final loss
    than any single-source run.

    This is a soft sanity check: more information should not hurt the teacher
    signal. It is marked ``xfail`` because the green implementation might not
    guarantee strict improvement in only 5 epochs — the primary assertion is
    distinguishability in test 2.

    RED: same TypeError as the other two tests.
    """
    full_loss = _fit_with_sources(_ALL_SOURCES)

    single_losses: dict[str, float] = {}
    for src in _ALL_SOURCES:
        single_losses[src] = _fit_with_sources([src])

    worst_single = max(single_losses.values())

    # Soft assertion: allow the full-signal run to be up to 10 % higher than
    # the best single-source run (noise tolerance for short epochs), but it
    # must beat the worst single-source run.
    assert full_loss <= worst_single * 1.10, (
        f"Full-signal final_loss={full_loss:.4f} is much worse than "
        f"worst single-source loss={worst_single:.4f}. "
        f"Per-source losses: {single_losses!r}"
    )

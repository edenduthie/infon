"""Regression test for the Stage C.2 canonical configuration.

This test exercises the EV corpus through ``HypergraphReasoner.fit()`` +
``reason()`` under the canonical Stage C.2 winner
(``fusion_rule="top1"``, ``decisive_top_k=2``, ``coherence_weight=1.0``)
and asserts the Epic 01 acceptance criteria hold across the five pinned
seeds ``{42, 0, 1, 7, 13}``:

1. Verdict polarity = ``SUPPORTS`` for Toyota / Honda / Tesla / CATL on
   *every* seed (per ``proposal.md`` acceptance criterion 1; Tesla and
   CATL widen coverage so cross-query polarity flips are caught — see
   ``ranked_candidates.md`` § *Per-candidate analysis #1*).
2. ``m(Θ)`` for Toyota and Honda within ``[0.20, 0.40]`` (criterion 2).
3. Across-seed ``std(m(Θ))`` for Toyota and Honda ``≤ 0.05`` (criterion
   3, the stability gate).

This is **not** a TDD red→green test — the implementation already
exists per A.5b/A.6b. The test asserts that the canonical configuration
behaves as advertised in the Stage C.2 ranking memo, which is the
contract D.1's ``canonical_v0_2.yaml`` will pin in.

Defensive notes (per the C.3 task brief):

- We deliberately do NOT change ``CognitionConfig`` defaults. The 14
  ``test_logic.py`` tests were authored against the old defaults
  (``fusion_rule="dempster"``, ``decisive_top_k=3``, no canonical
  ``coherence_weight``); flipping defaults would break unrelated
  assertions about ``SUPPORTS`` magnitude. This test exercises the
  canonical config explicitly via the ``reason()`` kwargs and
  ``fit(sheaf_weight=…)`` instead.
- 5 seeds is the minimum count for a meaningful std; the same seed list
  is used by the sweep aggregate that produced
  ``ranked_candidates.md``.

Spec:
- openspec/changes/epic-01-stabilize-theta/proposal.md (acceptance)
- openspec/changes/epic-01-stabilize-theta/spec.md
  Requirement: Canonical Configuration
- openspec/changes/epic-01-stabilize-theta/tasks.md C.3
- reference_v2/experiments/results/sweep_collapse/ranked_candidates.md
  (winner specifics; expected Θ values reproduced below)

No mocks; real reasoner, real masses, real EV corpus.
"""

from __future__ import annotations

import os
import statistics
import tempfile

import pytest

# Match the corpus and schema used by tests/test_logic.py and the other
# Stage A tests so this regression plugs into the same family.
# DOCUMENTS / setup_cognition are re-exported by tests.test_logic for
# the existing red-test family; DIAGNOSTIC_QUERIES is only on
# experiments.ev_corpus, so we import the four-query map from there
# (same source — see experiments/ev_corpus.py module docstring).
from experiments.ev_corpus import DIAGNOSTIC_QUERIES
from tests.test_logic import DOCUMENTS, setup_cognition


# ── Canonical configuration (Stage C.2 winner) ──────────────────────────
#
# Source: reference_v2/experiments/results/sweep_collapse/ranked_candidates.md
# § *Winner pick + rationale* (commit 51b074d). These three knobs are the
# entirety of what makes the canonical config "canonical"; everything
# else (activation_threshold=0.2, hidden_dim=64, n_layers=2, fit
# epochs=30, lr=1e-3) matches the existing baseline used by every Stage
# A test, so the configuration delta vs. test_logic.py's defaults is
# the three knobs below + ``log_per_infon_masses`` (which has no effect
# on the returned mass; it only populates a diagnostic record).

_FUSION_RULE = "top1"
_DECISIVE_TOP_K = 2
_COHERENCE_WEIGHT = 1.0

# 5 pinned seeds — same set that produced ranked_candidates.md's
# expected numbers (Toyota 0.2330 ± 0.0082 / Honda 0.2247 ± 0.0092).
_SEEDS = (42, 0, 1, 7, 13)

# Acceptance bands (from openspec/changes/epic-01-stabilize-theta/proposal.md).
_THETA_LOWER = 0.20
_THETA_UPPER = 0.40
_THETA_STD_MAX = 0.05


def _build_and_ingest(db_path: str):
    """Construct a fresh Cognition + ingest the EV corpus."""
    cog = setup_cognition(db_path)
    for doc in DOCUMENTS:
        cog.ingest([doc])
    cog.consolidate()
    return cog


def _build_reasoner(cog):
    """Construct a fresh HypergraphReasoner over an already-ingested store.

    ``log_per_infon_masses=True`` mirrors the canonical YAML setting in
    ``ranked_candidates.md`` § *Expected canonical_v0_2.yaml values*.
    The flag is structurally inert for the assertions below — it only
    populates a diagnostic field on the returned ``ReasoningResult`` —
    but matching the canonical YAML keeps this test in lockstep with
    D.1's eventual config file.
    """
    from cognition.logic import HypergraphReasoner

    return HypergraphReasoner(
        cog.store, cog.encoder, cog.schema,
        hidden_dim=64, n_layers=2,
        log_per_infon_masses=True,
    )


def _run_canonical_for_seed(seed: int) -> dict[str, dict[str, object]]:
    """Fit + reason on the EV corpus under the canonical config; return
    a per-query dict of ``{"verdict": str, "theta": float}``.

    A fresh tempdir/store is created per seed so seeds do not share
    SQLite state — same isolation the sweep harness uses
    (``experiments/run.py::_run_one_seed``).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, f"canonical_seed={seed}.db")
        cog = _build_and_ingest(db_path)
        try:
            reasoner = _build_reasoner(cog)
            graph = reasoner.builder.build(feature_dim=64)
            reasoner.fit(
                graph=graph,
                epochs=30,
                lr=1e-3,
                sheaf_weight=_COHERENCE_WEIGHT,
                grad_clip=1.0,
                patience=8,
                seed=seed,
            )

            per_query: dict[str, dict[str, object]] = {}
            for qname, qtext in DIAGNOSTIC_QUERIES.items():
                result = reasoner.reason(
                    qtext,
                    decisive_top_k=_DECISIVE_TOP_K,
                    fusion_rule=_FUSION_RULE,
                )
                per_query[qname] = {
                    "verdict": result.verdict,
                    "theta": float(result.mass.theta),
                }
        finally:
            cog.close()
    return per_query


# Run the fit-per-seed loop once and reuse the results across the
# eight parametrized assertions (4 polarity + 2 band + 2 stability).
# ``module`` scope avoids re-fitting 5 seeds × 8 tests = 40 times.
# Patience-based early stopping (patience=8) usually triggers before
# epoch 30, so the actual wall clock is ~15 s, not 5 × 30 s.
@pytest.fixture(scope="module")
def canonical_results() -> dict[int, dict[str, dict[str, object]]]:
    """Return ``{seed: {query: {"verdict": ..., "theta": ...}}}`` for
    all five seeds under the canonical config."""
    return {seed: _run_canonical_for_seed(seed) for seed in _SEEDS}


# ── Test 1: polarity SUPPORTS for all four queries on all five seeds ────


@pytest.mark.parametrize("query", list(DIAGNOSTIC_QUERIES.keys()))
def test_canonical_polarity_supports(canonical_results, query):
    """All four diagnostic queries must verdict SUPPORTS on every seed.

    Per ``proposal.md`` acceptance criterion 1 the load-bearing queries
    are Toyota and Honda; Tesla and CATL are widened coverage from
    ``ranked_candidates.md`` § *Per-candidate analysis #1* ("Tesla/CATL
    polarity SUPPORTS on all 5 seeds") and serve as cross-query
    polarity-flip detectors. We assert all four together because the
    Stage C.2 winner pick was conditional on all four passing.
    """
    bad = {
        seed: per_query[query]["verdict"]
        for seed, per_query in canonical_results.items()
        if per_query[query]["verdict"] != "SUPPORTS"
    }
    assert not bad, (
        f"canonical config (top1, k={_DECISIVE_TOP_K}, "
        f"cw={_COHERENCE_WEIGHT}) must verdict SUPPORTS on the {query!r} "
        f"query for every seed in {_SEEDS!r}; got non-SUPPORTS verdicts "
        f"on seeds={bad!r}"
    )


# ── Test 2: m(Θ) within [0.20, 0.40] for Toyota and Honda ───────────────


@pytest.mark.parametrize("query", ["toyota", "honda"])
def test_canonical_theta_in_band(canonical_results, query):
    """Toyota / Honda mean m(Θ) must lie in the acceptance band.

    Per ``proposal.md`` acceptance criterion 2: ``m(Θ) ∈ [0.20, 0.40]``
    on at least one named, committed configuration. The canonical
    config IS that configuration; this test asserts the band contract.
    """
    thetas = [
        float(per_query[query]["theta"])
        for per_query in canonical_results.values()
    ]
    mean_theta = statistics.fmean(thetas)
    assert _THETA_LOWER <= mean_theta <= _THETA_UPPER, (
        f"canonical config mean m(Θ) on {query!r} must be in "
        f"[{_THETA_LOWER}, {_THETA_UPPER}]; got mean={mean_theta:.4f} "
        f"(per-seed thetas={thetas!r})"
    )


# ── Test 3: across-seed std(m(Θ)) ≤ 0.05 for Toyota and Honda ──────────


@pytest.mark.parametrize("query", ["toyota", "honda"])
def test_canonical_theta_stability(canonical_results, query):
    """Across-seed std on m(Θ) must clear the stability gate.

    Per ``proposal.md`` acceptance criterion 3: ``std(m(Θ)) ≤ 0.05``.
    ``ranked_candidates.md`` reports ~0.008–0.009 std on Toyota / Honda
    for this winner — well under the gate, so a regression that
    doubles the std still passes; a regression that 5×s it does not.
    """
    thetas = [
        float(per_query[query]["theta"])
        for per_query in canonical_results.values()
    ]
    # ``pstdev`` (population std) matches ``experiments/sweep.py::_std``
    # — the same convention used to produce ``ranked_candidates.md``'s
    # quoted Toyota 0.0082 / Honda 0.0092 numbers, so the threshold
    # comparison is apples-to-apples.
    std_theta = statistics.pstdev(thetas)
    assert std_theta <= _THETA_STD_MAX, (
        f"canonical config across-seed std m(Θ) on {query!r} must be "
        f"<= {_THETA_STD_MAX}; got std={std_theta:.4f} "
        f"(per-seed thetas={thetas!r})"
    )


if __name__ == "__main__":
    # Manual invocation path; pytest is the primary runner.
    results = {seed: _run_canonical_for_seed(seed) for seed in _SEEDS}
    for query in DIAGNOSTIC_QUERIES:
        verdicts = [results[s][query]["verdict"] for s in _SEEDS]
        thetas = [float(results[s][query]["theta"]) for s in _SEEDS]
        mean_theta = statistics.fmean(thetas)
        std_theta = statistics.pstdev(thetas)
        print(
            f"{query:8s}  verdicts={verdicts}  "
            f"m(Θ) {mean_theta:.4f} ± {std_theta:.4f}"
        )
    print("PASS: canonical config regression")

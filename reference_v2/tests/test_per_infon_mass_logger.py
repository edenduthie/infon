"""Red-phase test for the per-infon mass logger.

This test currently FAILS — that is the point.  The released
``CognitionConfig`` does not carry a ``log_per_infon_masses`` flag, and
``HypergraphReasoner.reason()`` returns a ``ReasoningResult`` whose
``per_infon_masses`` is a flat ``list[MassFunction]`` rather than the
structured per-infon record (``query_id``, ``infon_id``, mass vector,
``relevance_score``) required by the diagnostic spec.

Once A.3b lands the config flag and richer record structure, this test
will turn green.

Reference:
    openspec/changes/epic-01-stabilize-theta/spec.md
        Requirement: Per-Infon Mass Logging
    openspec/changes/epic-01-stabilize-theta/tasks.md A.3 (red) -> A.3b (green)
    docs/publication/reproduction_audit.md §Path 2.1
"""

from __future__ import annotations

import os
import tempfile

# Match the corpus / schema / setup helper used by tests/test_logic.py and
# tests/test_seed_pinning.py so this test plugs into the same regression.
from tests.test_logic import DOCUMENTS, SCHEMA_DEFS, setup_cognition


# Diagnostic query reused by the audit (Toyota probe — the corpus contains
# explicit supporting infons such as "Toyota invests heavily in solid-state
# battery technology", so the per-infon contributor list MUST be non-empty
# once the logger is wired through).
_TOYOTA_QUERY = "Did Toyota invest in battery technology?"


def _build_cognition_with_logger(db_path: str):
    """Construct a Cognition instance whose Config opts into the diagnostic.

    Mirrors ``tests/test_logic.py::setup_cognition`` but passes the new
    ``log_per_infon_masses=True`` flag.  At the red stage this raises
    ``TypeError`` because ``CognitionConfig`` does not yet accept the kwarg.
    """
    import json
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
        # NEW (red): A.3b will add this flag to CognitionConfig.  Until then,
        # this kwarg raises TypeError, which is the documented failure mode.
        log_per_infon_masses=True,
    )
    return Cognition(config)


def _build_and_ingest_with_logger(db_path: str):
    cog = _build_cognition_with_logger(db_path)
    for doc in DOCUMENTS:
        cog.ingest([doc])
    cog.consolidate()
    return cog


def _build_reasoner(cog):
    """Construct a HypergraphReasoner over an already-ingested store.

    ``log_per_infon_masses=True`` is required after infon-6o3.36 — the
    flag now actually gates emission inside ``reason()``, so this opt-in
    test must explicitly enable the diagnostic. (A.3b emitted records
    unconditionally; .36 made the flag actually do its job.)
    """
    from cognition.logic import HypergraphReasoner

    return HypergraphReasoner(
        cog.store, cog.encoder, cog.schema,
        hidden_dim=64, n_layers=2,
        log_per_infon_masses=True,
    )


def _record_field(record, name: str):
    """Pull ``name`` from a record whether it is a dataclass-like or dict.

    The spec doesn't fix the record's surface form (dict vs dataclass);
    A.3b chooses.  This helper accepts either so the assertions stay
    structural rather than tied to one implementation.
    """
    if isinstance(record, dict):
        if name not in record:
            raise AssertionError(
                f"per-infon record missing field {name!r}; got keys "
                f"{sorted(record.keys())}"
            )
        return record[name]
    if hasattr(record, name):
        return getattr(record, name)
    raise AssertionError(
        f"per-infon record (type={type(record).__name__}) is missing "
        f"the {name!r} field required by spec.md Requirement: "
        f"Per-Infon Mass Logging"
    )


def test_reason_emits_per_infon_mass_records():
    """``reason()`` must emit a structured per-infon mass log on the EV scenario.

    Phase A.3 (RED): the test fails at one of three points, in order of
    likelihood:

        1. ``CognitionConfig(..., log_per_infon_masses=True)`` raises
           ``TypeError: unexpected keyword argument 'log_per_infon_masses'``
           because the flag is not yet on the dataclass.
        2. The flag exists but ``ReasoningResult.per_infon_masses`` is a
           ``list[MassFunction]`` rather than a list of structured records;
           ``_record_field(record, "infon_id")`` raises ``AssertionError``.
        3. The flag is honoured but the record list is empty for the Toyota
           query; the "at least one contributor" assertion fails.

    Phase A.3b (GREEN): the diagnostic record carries
    ``{query_id, infon_id, mass: [m_S, m_R, m_U, m_Theta], relevance_score}``
    for every per-infon contributor; each mass is a 4-vector that is
    non-negative and sums to 1 within 1e-6.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_logger.db")
        cog = _build_and_ingest_with_logger(db_path)
        try:
            reasoner = _build_reasoner(cog)
            graph = reasoner.builder.build(feature_dim=64)
            reasoner.fit(graph=graph, epochs=30, seed=42)
            result = reasoner.reason(_TOYOTA_QUERY)
        finally:
            cog.close()

    # ── Field presence on the result ─────────────────────────────────
    # The exact attribute name is allowed to evolve in A.3b, but the
    # spec.md requirement names "per-infon masses" so we expect either
    # `per_infon_masses` (current placeholder) or `per_infon_mass_log`.
    log = None
    for candidate in ("per_infon_mass_log", "per_infon_masses"):
        if hasattr(result, candidate):
            log = getattr(result, candidate)
            break
    assert log is not None, (
        "ReasoningResult is missing both `per_infon_mass_log` and "
        "`per_infon_masses`; spec.md Requirement: Per-Infon Mass Logging "
        "demands at least one diagnostic field on the result"
    )

    # ── At least one contributor for the Toyota probe ────────────────
    # The EV corpus has multiple supporting infons for Toyota investing in
    # battery technology; an empty log here means the logger is not wired
    # through reason() correctly.
    assert len(log) >= 1, (
        f"per-infon mass log is empty for the Toyota query; expected at "
        f"least one supporting contributor from the EV corpus.  This means "
        f"the logger is not collecting per-infon masses inside reason()."
    )

    # ── Structural / numerical invariants on every record ───────────
    for i, record in enumerate(log):
        # NOTE: at the red stage, `record` is a bare MassFunction with no
        # `infon_id` / `query_id` / `relevance_score` — the next call
        # raises AssertionError, which is the expected red-phase failure.
        query_id = _record_field(record, "query_id")
        infon_id = _record_field(record, "infon_id")
        mass = _record_field(record, "mass")
        relevance_score = _record_field(record, "relevance_score")

        assert isinstance(query_id, str) and query_id, (
            f"record[{i}].query_id must be a non-empty string; got "
            f"{query_id!r}"
        )
        assert isinstance(infon_id, str) and infon_id, (
            f"record[{i}].infon_id must be a non-empty string; got "
            f"{infon_id!r}"
        )

        # mass must be a 4-vector (list / tuple / array) interpretable as
        # [m_S, m_R, m_U, m_Theta] per spec.md.
        try:
            mass_vec = list(mass)
        except TypeError as exc:
            raise AssertionError(
                f"record[{i}].mass must be iterable as a 4-vector; got "
                f"{type(mass).__name__}: {exc}"
            )
        assert len(mass_vec) == 4, (
            f"record[{i}].mass must have 4 entries [m_S, m_R, m_U, m_Theta]; "
            f"got len={len(mass_vec)}"
        )
        coords = [float(x) for x in mass_vec]
        assert all(c >= 0.0 for c in coords), (
            f"record[{i}].mass has a negative entry: {coords}"
        )
        total = sum(coords)
        assert abs(total - 1.0) <= 1e-6, (
            f"record[{i}].mass does not sum to 1 within 1e-6: "
            f"sum={total!r}, mass={coords}"
        )

        rs = float(relevance_score)
        assert rs >= 0.0, (
            f"record[{i}].relevance_score must be non-negative; got {rs!r}"
        )


if __name__ == "__main__":
    test_reason_emits_per_infon_mass_records()
    print("PASS: per-infon mass logger emits structured records")

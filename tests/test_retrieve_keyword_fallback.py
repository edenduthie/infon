"""Integration test for the keyword-fallback retrieval path.

The default code-mode schema (CODE_RELATION_ANCHORS) has only the eight
relation anchors and no actor anchors. When a user runs `infon init`
without a custom schema, AST-extracted infons end up with subject/object
strings that are symbol names from the source (e.g. ``InfonStore``,
``UserService``) — those strings are NOT in the schema.

When such a user types ``infon search "what calls InfonStore"``, the
SPLADE-only path finds zero candidates: the only schema anchor that
activates is ``calls`` (a relation), but ``store.query(subject="calls"
OR object="calls")`` returns nothing because the AST infons store
``"calls"`` in the predicate column, not the subject/object columns.

This test verifies that ``retrieve`` falls back to keyword matching in
this realistic case so users get useful results from day one — without
having to first run schema discovery to populate actor anchors.

No mocks — real DuckDB store, real Infon instances, real
``CODE_RELATION_ANCHORS`` schema.
"""

import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

from infon.grounding import ASTGrounding, Grounding
from infon.infon import ImportanceScore, Infon
from infon.retrieve import retrieve
from infon.schema import CODE_RELATION_ANCHORS, AnchorSchema
from infon.store import InfonStore


def _ast_infon(subject: str, predicate: str, obj: str, line: int = 1) -> Infon:
    """Build a real Infon mimicking what the AST extractor produces."""
    return Infon(
        id=str(uuid.uuid4()),
        subject=subject,
        predicate=predicate,
        object=obj,
        polarity=True,
        grounding=Grounding(
            root=ASTGrounding(
                grounding_type="ast",
                file_path=f"src/{subject}.py",
                line_number=line,
                node_type="call",
            )
        ),
        confidence=0.95,
        timestamp=datetime.now(UTC),
        importance=ImportanceScore(
            activation=0.8,
            coherence=0.7,
            specificity=0.9,
            novelty=0.5,
            reinforcement=0.6,
        ),
        kind="extracted",
        reinforcement_count=1,
    )


@pytest.fixture
def code_schema() -> AnchorSchema:
    """Default code-mode schema produced by `infon init`: relations only."""
    return AnchorSchema(
        version="0.1.0",
        language="code",
        anchors=CODE_RELATION_ANCHORS,
    )


@pytest.fixture
def ast_store(tmp_path: Path) -> InfonStore:
    """Real store populated with AST-style infons (symbol names as anchors)."""
    db_path = tmp_path / "kb.ddb"
    store = InfonStore(str(db_path))
    for infon in [
        _ast_infon("UserService", "calls", "InfonStore", line=42),
        _ast_infon("AuthHandler", "calls", "InfonStore", line=17),
        _ast_infon("InfonStore", "calls", "duckdb", line=8),
        _ast_infon("test_store", "imports", "InfonStore", line=1),
        _ast_infon("UserService", "imports", "AuthHandler", line=2),
        _ast_infon("Cache", "calls", "redis", line=10),
    ]:
        store.upsert(infon)
    yield store
    store.close()


def test_retrieve_natural_language_finds_ast_infons_via_fallback(
    ast_store: InfonStore, code_schema: AnchorSchema
) -> None:
    """`retrieve("what calls InfonStore", ...)` must return InfonStore call sites
    even though the schema has zero actor anchors and SPLADE alone can't map
    the symbol name to anything."""
    results = retrieve("what calls InfonStore", ast_store, code_schema, limit=10)

    assert results, "fallback must return at least one result"

    triples = [(r.infon.subject, r.infon.predicate, r.infon.object) for r in results]
    # The two callers of InfonStore must appear; ranked by relevance.
    assert ("UserService", "calls", "InfonStore") in triples
    assert ("AuthHandler", "calls", "InfonStore") in triples


def test_retrieve_unrelated_query_does_not_match_everything(
    ast_store: InfonStore, code_schema: AnchorSchema
) -> None:
    """The fallback should not collapse to a return-everything regex match —
    queries with no overlap should return [] (or at most very low-scored
    results), not the full store."""
    results = retrieve("xylophone marsupial", ast_store, code_schema, limit=10)
    # No overlap with any subject/predicate/object → empty result set.
    assert results == []


def test_retrieve_filters_short_and_stop_words(
    ast_store: InfonStore, code_schema: AnchorSchema
) -> None:
    """Stop words like "what", "is", "the" must NOT pull in unrelated infons.
    Without filtering, "what is" would match nothing (good) but "what calls"
    would match every "calls" infon (bad — too noisy)."""
    # "what" alone (a pure stop word) should yield nothing useful.
    results = retrieve("what", ast_store, code_schema, limit=10)
    assert results == []


def test_retrieve_keyword_fallback_orders_by_relevance(
    ast_store: InfonStore, code_schema: AnchorSchema
) -> None:
    """A query that names two distinct subjects should rank infons mentioning
    BOTH (or the more reinforced one) higher than ones mentioning only one."""
    results = retrieve("UserService calls", ast_store, code_schema, limit=10)
    assert results, "must return at least one result"
    # The top result must involve UserService (either as subject or object).
    top = results[0].infon
    assert "userservice" in (top.subject + top.object).lower()


def test_retrieve_keyword_bonus_boosts_query_term_matches_in_splade_path(
    ast_store: InfonStore, code_schema: AnchorSchema
) -> None:
    """When SPLADE activates a relation anchor (e.g. ``calls``), the predicate
    fan-out gathers EVERY infon with that predicate. Without a keyword bonus
    on top, "what calls InfonStore" returns calls infons unrelated to
    InfonStore at the same score as the InfonStore-mentioning ones — that's
    exactly the failure mode dog-fooding hit on the real repo.

    This test verifies that the keyword bonus pushes InfonStore-mentioning
    calls infons to the top of the ranking."""
    results = retrieve("what calls InfonStore", ast_store, code_schema, limit=10)
    assert results, "must return at least one result"

    # The top result must involve InfonStore as subject or object — without
    # the keyword bonus, the SPLADE-only path would rank Cache->redis or any
    # other "calls" infon equally high.
    top = results[0].infon
    triple_text = f"{top.subject} {top.object}".lower()
    assert "infonstore" in triple_text, (
        f"top result must mention InfonStore; got "
        f"{top.subject} -> {top.predicate} -> {top.object}"
    )

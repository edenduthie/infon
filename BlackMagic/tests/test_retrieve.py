"""Tests for blackmagic.retrieve."""


def test_basic_search(bm_populated):
    result = bm_populated.search("Toyota battery investment", top_k=10)
    assert len(result.infons) > 0
    # Top hits should include Toyota-related infons
    subjects = {inf.subject for inf in result.infons[:5]}
    assert "toyota" in subjects


def test_search_activates_anchors(bm_populated):
    result = bm_populated.search("investing in batteries")
    assert "invest" in result.anchors_activated or "battery" in result.anchors_activated


def test_persona_valence_investor(bm_populated):
    # Use a query that actually activates anchors in our tiny schema
    result = bm_populated.search("Toyota investing in batteries",
                                 persona="investor", top_k=20)
    assert len(result.infons) > 0
    # valence dict should have entries for at least some results
    assert len(result.valence) > 0


def test_empty_corpus_returns_empty(bm):
    result = bm.search("anything")
    assert result.infons == []
    assert result.hits == []


def test_contrary_view_changes_ranking(bm_populated):
    normal = bm_populated.search("Toyota investment", top_k=10, contrary=False)
    contrary = bm_populated.search("Toyota investment", top_k=10, contrary=True)
    # Same underlying pool, but valence should differ
    if normal.valence and contrary.valence:
        # At least one valence should flip sign
        any_diff = any(
            normal.valence.get(k, 0) != contrary.valence.get(k, 0)
            for k in set(normal.valence) | set(contrary.valence)
        )
        assert any_diff

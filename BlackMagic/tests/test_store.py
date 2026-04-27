"""Tests for blackmagic.store."""
from blackmagic.store import LocalStore
from blackmagic.infon import Infon, Edge, Constraint


def _make_infon(**overrides):
    base = dict(
        infon_id="t1", subject="toyota", predicate="invest", object="battery",
        sentence="Toyota invests in batteries.",
        doc_id="d1", sent_id="d1_0000", polarity=1, confidence=0.9,
        importance=0.5,
    )
    base.update(overrides)
    return Infon(**base)


def test_roundtrip_infon():
    s = LocalStore(":memory:"); s.init()
    inf = _make_infon(spans={"subject": {"text": "Toyota", "start": 0, "end": 6}})
    s.put_infon(inf)
    out = s.get_infon("t1")
    assert out.subject == "toyota"
    assert out.spans == {"subject": {"text": "Toyota", "start": 0, "end": 6}}
    s.close()


def test_query_filters():
    s = LocalStore(":memory:"); s.init()
    s.put_infons([
        _make_infon(infon_id="t1", subject="toyota"),
        _make_infon(infon_id="t2", subject="honda", importance=0.1),
        _make_infon(infon_id="t3", subject="toyota", predicate="launch",
                    object="ev"),
    ])
    out = s.query_infons(subject="toyota")
    assert len(out) == 2
    out = s.query_infons(subject="honda")
    assert len(out) == 1
    s.close()


def test_imagined_filter():
    s = LocalStore(":memory:"); s.init()
    s.put_infons([
        _make_infon(infon_id="obs1", kind="observed"),
        _make_infon(infon_id="img1", kind="imagined", fitness=0.7,
                    parent_infon_ids=["obs1"]),
    ])
    obs = s.query_infons()
    assert len(obs) == 1 and obs[0].kind == "observed"

    all_ = s.query_infons(include_imagined=True)
    assert len(all_) == 2

    assert s.count_infons() == 1
    assert s.count_infons(kind="imagined") == 1

    n = s.delete_imagined()
    assert n == 1
    assert s.count_infons(kind="imagined") == 0
    s.close()


def test_edges():
    s = LocalStore(":memory:"); s.init()
    s.put_edge(Edge(source="inf1", target="toyota",
                    edge_type="INITIATES", weight=0.9))
    s.put_edge(Edge(source="inf1", target="inf2",
                    edge_type="NEXT", weight=1.0,
                    metadata={"shared_anchor": "toyota"}))
    edges = s.get_edges(edge_type="NEXT")
    assert len(edges) == 1
    assert edges[0].metadata["shared_anchor"] == "toyota"
    s.close()


def test_constraints():
    s = LocalStore(":memory:"); s.init()
    c = Constraint(subject="toyota", predicate="invest", object="battery",
                   evidence=3, doc_count=2, strength=0.8, persistence=1,
                   score=0.9, infon_ids=["t1", "t2"])
    s.put_constraint(c)
    out = s.get_constraints()
    assert len(out) == 1
    assert out[0].strength == 0.8
    assert out[0].infon_ids == ["t1", "t2"]
    s.close()


def test_documents():
    s = LocalStore(":memory:"); s.init()
    s.put_document("d1", "Toyota invests.", timestamp="2026-03-01", n_infons=1)
    d = s.get_document("d1")
    assert d["text"] == "Toyota invests."
    assert d["timestamp"] == "2026-03-01"
    assert s.count_documents() == 1
    s.close()

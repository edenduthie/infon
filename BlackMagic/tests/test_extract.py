"""Tests for blackmagic.extract."""
import pytest

from blackmagic.extract import (
    split_sentences, _detect_polarity, _make_infon_id, extract_infons,
)


def test_split_sentences_english():
    text = "Toyota invests. Honda launches. Ford recalls."
    parts = split_sentences(text)
    assert len(parts) == 3


def test_split_sentences_preserves_single():
    assert split_sentences("Just one sentence.") == ["Just one sentence."]


def test_polarity_affirmed():
    assert _detect_polarity("Toyota is investing in batteries.") == 1


def test_polarity_negated():
    assert _detect_polarity("Toyota is not investing in batteries.") == 0


def test_infon_id_deterministic():
    id1 = _make_infon_id("d1", 0, "toyota", "invest", "battery")
    id2 = _make_infon_id("d1", 0, "toyota", "invest", "battery")
    assert id1 == id2


def test_extract_basic(bm_populated):
    # bm_populated already ingested tiny_corpus
    infons = bm_populated.store.query_infons(limit=200)
    assert len(infons) > 0
    triples = {inf.triple_key() for inf in infons}
    # Expect at least one Toyota/invest/battery-related triple
    toyota_infons = [i for i in infons if i.subject == "toyota"]
    assert len(toyota_infons) > 0


def test_extract_spans_offsets(bm_populated):
    infons = bm_populated.store.query_infons(subject="toyota", limit=20)
    inf = infons[0]
    # spans should map back into the sentence
    for role, span in inf.spans.items():
        assert inf.sentence[span["start"]:span["end"]].lower() == span["text"].lower()

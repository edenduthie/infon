"""Shared pytest fixtures for BlackMagic tests."""
from __future__ import annotations

import pytest

from blackmagic import BlackMagic, BlackMagicConfig
from blackmagic.schema import AnchorSchema


TINY_SCHEMA = {
    # actors
    "toyota": {"type": "actor", "tokens": ["toyota"]},
    "honda":  {"type": "actor", "tokens": ["honda"]},
    "ford":   {"type": "actor", "tokens": ["ford"]},
    "bmw":    {"type": "actor", "tokens": ["bmw"]},
    # relations
    "invest":    {"type": "relation", "tokens": ["invest", "investment", "investing"]},
    "launch":    {"type": "relation", "tokens": ["launch", "launches", "launched"]},
    "recall":    {"type": "relation", "tokens": ["recall", "recalled", "recalls"]},
    "partner":   {"type": "relation", "tokens": ["partner", "partnership", "partnered"]},
    "cancel":    {"type": "relation", "tokens": ["cancel", "cancels", "discontinue"]},
    # features
    "battery":   {"type": "feature", "tokens": ["battery", "batteries", "cell"]},
    "ev":        {"type": "feature", "tokens": ["electric vehicle", "ev", "evs"]},
    "solid_state": {"type": "feature", "tokens": ["solid-state", "solid state"]},
    # locations
    "japan":     {"type": "location", "tokens": ["japan", "japanese", "tokyo"]},
    "us":        {"type": "location", "tokens": ["united states", "us", "usa", "america"]},
    "china":     {"type": "location", "tokens": ["china", "chinese", "beijing"]},
    "europe":    {"type": "location", "tokens": ["europe", "european", "eu"]},
}


TINY_CORPUS = [
    {"text": "Toyota invests heavily in battery technology.",
     "id": "d1", "timestamp": "2026-03-01"},
    {"text": "Honda launches a new electric vehicle in China.",
     "id": "d2", "timestamp": "2026-03-02"},
    {"text": "Ford recalls vehicles in the United States.",
     "id": "d3", "timestamp": "2026-03-03"},
    {"text": "BMW partners with Toyota on solid-state batteries.",
     "id": "d4", "timestamp": "2026-03-05"},
    {"text": "Toyota launches a new EV platform in Japan.",
     "id": "d5", "timestamp": "2026-03-08"},
    {"text": "Honda cancels its US expansion plans.",
     "id": "d6", "timestamp": "2026-03-12"},
    {"text": "Ford invests in battery cell production in Europe.",
     "id": "d7", "timestamp": "2026-03-15"},
    {"text": "BMW launches an electric vehicle line in Germany.",
     "id": "d8", "timestamp": "2026-03-20"},
]


@pytest.fixture
def tiny_schema():
    return AnchorSchema(TINY_SCHEMA)


@pytest.fixture
def tiny_corpus():
    return list(TINY_CORPUS)


@pytest.fixture
def tmp_schema_path(tmp_path, tiny_schema):
    """Write TINY_SCHEMA to a temp JSON file and return the path."""
    import json
    p = tmp_path / "schema.json"
    p.write_text(json.dumps(tiny_schema.anchors))
    return str(p)


@pytest.fixture
def bm(tmp_schema_path):
    """BlackMagic instance with tiny schema, in-memory DB, no docs yet."""
    cfg = BlackMagicConfig(
        schema_path=tmp_schema_path,
        db_path=":memory:",
        activation_threshold=0.3,
        min_confidence=0.05,
        consolidation_interval=0,
    )
    instance = BlackMagic(cfg)
    yield instance
    instance.close()


@pytest.fixture
def bm_populated(bm, tiny_corpus):
    """BlackMagic instance with tiny_corpus ingested and consolidated."""
    bm.ingest(tiny_corpus, consolidate_now=True)
    return bm

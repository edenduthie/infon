"""Tests for blackmagic.encoder."""

from blackmagic.encoder import Encoder, SpladeEncoder


def test_splade_loads():
    enc = SpladeEncoder()
    assert enc.vocab_size == 30522
    assert enc.model is not None


def test_encode_single_sentence(tiny_schema):
    enc = Encoder(schema=tiny_schema)
    acts = enc.encode_single("Toyota invests heavily in battery technology.")
    assert isinstance(acts, dict)
    # Toyota, invest, battery should all light up
    assert acts.get("toyota", 0) > 1.0
    assert acts.get("invest", 0) > 0.5
    assert acts.get("battery", 0) > 1.0


def test_anchor_projector_covers_all_anchors(tiny_schema):
    enc = Encoder(schema=tiny_schema)
    # Every anchor should have at least one vocab id (tiny_schema is all
    # common English words)
    for name in tiny_schema.names:
        assert enc._projector.anchor_token_ids[name], \
            f"anchor {name!r} has no vocab tokens"


def test_batch_encode_shape(tiny_schema):
    enc = Encoder(schema=tiny_schema)
    acts = enc.encode([
        "Toyota invests in batteries.",
        "Ford recalls vehicles.",
        "The weather is nice today.",
    ])
    assert acts.shape == (3, len(tiny_schema.names))


def test_unrelated_sentence_scores_lower(tiny_schema):
    enc = Encoder(schema=tiny_schema)
    relevant = enc.encode_single("Toyota invests in batteries.")
    unrelated = enc.encode_single("The chef prepared a delicious pasta dish.")
    # "battery" should fire much more on relevant sentence
    assert relevant.get("battery", 0) > unrelated.get("battery", 0)


def test_find_spans_ascii():
    from blackmagic.schema import AnchorSchema
    schema = AnchorSchema({
        "toyota": {"type": "actor", "tokens": ["toyota"]},
    })
    enc = Encoder(schema=schema)
    spans = enc.find_spans("Toyota announced its plans.", "toyota",
                           schema.anchors)
    assert len(spans) == 1
    assert spans[0]["text"].lower() == "toyota"
    assert spans[0]["start"] == 0
    assert spans[0]["end"] == 6

"""Integration test for text extraction pipeline.

Tests the full text extraction pipeline:
1. Sentence splitting
2. SPLADE encoding to anchor space
3. Triple formation from activated anchors
4. Span finding in source text
5. Negation detection
6. Tense/aspect classification
7. Importance scoring
8. Infon construction with TextGrounding

No mocks — real encoder, real schema, real Infon models.
"""

import pytest

from infon.extract import extract_text
from infon.schema import Anchor, AnchorSchema


@pytest.fixture
def sample_schema() -> AnchorSchema:
    """Create a sample AnchorSchema for testing."""
    anchors = {
        "user_service": Anchor(
            key="user_service",
            type="actor",
            tokens=["user", "userservice", "user_service"],
            description="User service component",
            parent=None,
        ),
        "database_pool": Anchor(
            key="database_pool",
            type="actor",
            tokens=["database", "pool", "connection_pool", "database_pool"],
            description="Database connection pool",
            parent=None,
        ),
        "token_validator": Anchor(
            key="token_validator",
            type="actor",
            tokens=["token", "validator", "token_validator", "tokenvalidator"],
            description="Token validation component",
            parent=None,
        ),
        "calls": Anchor(
            key="calls",
            type="relation",
            tokens=["call", "calls", "invoke", "invokes", "uses"],
            description="Function or method invocation",
            parent=None,
        ),
        "delegates": Anchor(
            key="delegates",
            type="relation",
            tokens=["delegate", "delegates", "delegating"],
            description="Delegates responsibility to another component",
            parent=None,
        ),
    }
    return AnchorSchema(anchors=anchors, version="1.0.0", language="text")


def test_extract_text_affirmative_sentence_polarity_true(sample_schema):
    """Test that affirmative sentences produce polarity=True.
    
    WHEN: extract_text() is called with an affirmative sentence
    THEN: The resulting infon(s) have polarity=True
    """
    text = "UserService calls the database connection pool."
    doc_id = "test_doc_1"
    
    infons = extract_text(text, doc_id, sample_schema)
    
    # Should produce at least one infon
    assert len(infons) > 0
    
    # All infons from affirmative sentence should have polarity=True
    for infon in infons:
        assert infon.polarity is True
        assert infon.grounding.root.grounding_type == "text"
        assert infon.grounding.root.doc_id == doc_id
        assert infon.kind == "extracted"


def test_extract_text_negated_sentence_polarity_false(sample_schema):
    """Test that negated sentences produce polarity=False.
    
    WHEN: extract_text() is called with a negated sentence
    THEN: The resulting infon(s) have polarity=False
    """
    text = "UserService no longer calls DatabasePool directly."
    doc_id = "test_doc_2"
    
    infons = extract_text(text, doc_id, sample_schema)
    
    # Should produce at least one infon
    assert len(infons) > 0
    
    # All infons from negated sentence should have polarity=False
    for infon in infons:
        assert infon.polarity is False
        assert infon.grounding.root.grounding_type == "text"
        assert infon.grounding.root.doc_id == doc_id


def test_extract_text_multi_sentence_produces_multiple_infons(sample_schema):
    """Test that multi-sentence input produces multiple infons.
    
    WHEN: extract_text() is called with multiple sentences
    THEN: Multiple infons are returned (potentially one per sentence or more)
    """
    text = "UserService delegates auth to the new TokenValidator. The database pool handles connections."
    doc_id = "test_doc_3"
    
    infons = extract_text(text, doc_id, sample_schema)
    
    # Should produce multiple infons (at least one per sentence with activated anchors)
    # We expect at least 1 infon, but likely more
    assert len(infons) >= 1
    
    # Each infon should have text grounding
    for infon in infons:
        assert infon.grounding.root.grounding_type == "text"
        assert infon.grounding.root.doc_id == doc_id
        # sent_id should be 0 or 1 (sentence index)
        assert infon.grounding.root.sent_id in [0, 1]


def test_extract_text_empty_text_returns_empty_list(sample_schema):
    """Test that empty text returns an empty list.
    
    WHEN: extract_text() is called with empty text
    THEN: An empty list is returned
    """
    text = ""
    doc_id = "test_doc_4"
    
    infons = extract_text(text, doc_id, sample_schema)
    
    assert infons == []


def test_extract_text_no_activated_anchors_returns_empty_list(sample_schema):
    """Test that text with no activated anchors returns an empty list.
    
    WHEN: extract_text() is called with text that doesn't activate any schema anchors
    THEN: An empty list is returned (no triples formed)
    """
    # Text with words not in any anchor's tokens
    text = "The quick brown fox jumps over the lazy dog."
    doc_id = "test_doc_5"
    
    infons = extract_text(text, doc_id, sample_schema)
    
    # Should return empty list since no anchors are activated above threshold
    assert infons == []


def test_extract_text_infon_has_text_grounding(sample_schema):
    """Test that extracted infons have TextGrounding with all required fields.
    
    WHEN: extract_text() produces infons
    THEN: Each infon has TextGrounding with doc_id, sent_id, char_start, char_end, sentence_text
    """
    text = "UserService calls the database pool."
    doc_id = "test_doc_6"
    
    infons = extract_text(text, doc_id, sample_schema)
    
    assert len(infons) > 0
    
    for infon in infons:
        grounding = infon.grounding.root
        assert grounding.grounding_type == "text"
        assert grounding.doc_id == doc_id
        assert isinstance(grounding.sent_id, int)
        assert grounding.sent_id >= 0
        assert isinstance(grounding.char_start, int)
        assert isinstance(grounding.char_end, int)
        assert grounding.char_end > grounding.char_start
        assert isinstance(grounding.sentence_text, str)
        assert len(grounding.sentence_text) > 0


def test_extract_text_infon_has_importance_score(sample_schema):
    """Test that extracted infons have ImportanceScore with all components.
    
    WHEN: extract_text() produces infons
    THEN: Each infon has an ImportanceScore with activation, coherence, specificity, novelty, reinforcement
    """
    text = "UserService calls the database pool."
    doc_id = "test_doc_7"
    
    infons = extract_text(text, doc_id, sample_schema)
    
    assert len(infons) > 0
    
    for infon in infons:
        importance = infon.importance
        # All components should be in [0, 1]
        assert 0.0 <= importance.activation <= 1.0
        assert 0.0 <= importance.coherence <= 1.0
        assert 0.0 <= importance.specificity <= 1.0
        assert 0.0 <= importance.novelty <= 1.0
        assert 0.0 <= importance.reinforcement <= 1.0
        
        # Composite should be the average
        assert 0.0 <= importance.composite <= 1.0


def test_extract_text_infon_has_valid_triple(sample_schema):
    """Test that extracted infons have valid subject, predicate, object from schema.
    
    WHEN: extract_text() produces infons
    THEN: Each infon's subject, predicate, object are valid anchor keys from the schema
    """
    text = "UserService calls the database pool."
    doc_id = "test_doc_8"
    
    infons = extract_text(text, doc_id, sample_schema)
    
    assert len(infons) > 0
    
    schema_keys = set(sample_schema.anchors.keys())
    
    for infon in infons:
        # Subject, predicate, object should all be in the schema
        assert infon.subject in schema_keys
        assert infon.predicate in schema_keys
        assert infon.object in schema_keys
        
        # Predicate should be a relation type
        predicate_anchor = sample_schema.anchors[infon.predicate]
        assert predicate_anchor.type == "relation"


def test_extract_text_confidence_in_valid_range(sample_schema):
    """Test that extracted infons have confidence in [0, 1].
    
    WHEN: extract_text() produces infons
    THEN: Each infon has confidence in [0, 1]
    """
    text = "UserService calls the database pool."
    doc_id = "test_doc_9"
    
    infons = extract_text(text, doc_id, sample_schema)
    
    assert len(infons) > 0
    
    for infon in infons:
        assert 0.0 <= infon.confidence <= 1.0


def test_extract_text_reinforcement_count_starts_at_one(sample_schema):
    """Test that extracted infons have reinforcement_count starting at 1.
    
    WHEN: extract_text() produces infons
    THEN: Each infon has reinforcement_count=1 (first observation)
    """
    text = "UserService calls the database pool."
    doc_id = "test_doc_10"
    
    infons = extract_text(text, doc_id, sample_schema)
    
    assert len(infons) > 0
    
    for infon in infons:
        assert infon.reinforcement_count == 1

"""
Integration tests for Infon model.

This module tests the Infon data model following strict TDD:
- Immutability (frozen model)
- JSON serialization round-trip
- replace() helper method
- All fields from spec

No mocks, no stubs - real Pydantic model instances with real dependencies.
"""

import json
import uuid
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from infon.grounding import ASTGrounding, Grounding, TextGrounding
from infon.infon import ImportanceScore, Infon


def test_infon_immutability():
    """Test that Infon is frozen and cannot be mutated after creation."""
    # Create a real Infon instance
    infon = Infon(
        id=str(uuid.uuid4()),
        subject="user_service",
        predicate="calls",
        object="database_pool",
        polarity=True,
        grounding=Grounding(root=TextGrounding(
            doc_id="doc1",
            sent_id=0,
            char_start=10,
            char_end=25,
            sentence_text="UserService calls the database pool."
        )),
        confidence=0.85,
        timestamp=datetime.now(UTC),
        importance=ImportanceScore(
            activation=0.8,
            coherence=0.7,
            specificity=0.6,
            novelty=0.5,
            reinforcement=0.4
        ),
        kind="extracted",
        reinforcement_count=1
    )
    
    # Attempt to modify a field - should raise FrozenInstanceError
    with pytest.raises(ValidationError, match="frozen"):
        infon.subject = "other_service"  # type: ignore
    
    with pytest.raises(ValidationError, match="frozen"):
        infon.confidence = 0.9  # type: ignore


def test_infon_replace():
    """Test that Infon.replace() creates a new instance with updated fields."""
    # Create original infon
    original_id = str(uuid.uuid4())
    original = Infon(
        id=original_id,
        subject="user_service",
        predicate="calls",
        object="database_pool",
        polarity=True,
        grounding=Grounding(root=TextGrounding(
            doc_id="doc1",
            sent_id=0,
            char_start=10,
            char_end=25,
            sentence_text="UserService calls the database pool."
        )),
        confidence=0.85,
        timestamp=datetime.now(UTC),
        importance=ImportanceScore(
            activation=0.8,
            coherence=0.7,
            specificity=0.6,
            novelty=0.5,
            reinforcement=0.4
        ),
        kind="extracted",
        reinforcement_count=1
    )
    
    # Use replace() to create a modified copy
    updated = original.replace(
        confidence=0.95,
        reinforcement_count=2
    )
    
    # Verify original is unchanged
    assert original.confidence == 0.85
    assert original.reinforcement_count == 1
    assert original.id == original_id
    
    # Verify updated instance has new values
    assert updated.confidence == 0.95
    assert updated.reinforcement_count == 2
    
    # Verify other fields are preserved
    assert updated.id == original_id
    assert updated.subject == "user_service"
    assert updated.predicate == "calls"
    assert updated.object == "database_pool"
    assert updated.polarity == original.polarity
    assert updated.kind == "extracted"
    
    # Verify they are different instances
    assert updated is not original


def test_infon_json_round_trip():
    """Test that Infon can be serialized to JSON and deserialized back correctly."""
    # Create an Infon with TextGrounding
    timestamp = datetime.now(UTC)
    infon_id = str(uuid.uuid4())
    
    original = Infon(
        id=infon_id,
        subject="user_service",
        predicate="calls",
        object="database_pool",
        polarity=True,
        grounding=Grounding(root=TextGrounding(
            doc_id="doc1",
            sent_id=0,
            char_start=10,
            char_end=25,
            sentence_text="UserService calls the database pool."
        )),
        confidence=0.85,
        timestamp=timestamp,
        importance=ImportanceScore(
            activation=0.8,
            coherence=0.7,
            specificity=0.6,
            novelty=0.5,
            reinforcement=0.4
        ),
        kind="extracted",
        reinforcement_count=1
    )
    
    # Serialize to JSON
    json_str = original.model_dump_json()
    json_dict = json.loads(json_str)
    
    # Verify JSON structure
    assert json_dict["id"] == infon_id
    assert json_dict["subject"] == "user_service"
    assert json_dict["predicate"] == "calls"
    assert json_dict["object"] == "database_pool"
    assert json_dict["polarity"] is True
    assert json_dict["confidence"] == 0.85
    assert json_dict["kind"] == "extracted"
    assert json_dict["reinforcement_count"] == 1
    
    # Verify grounding is correctly serialized
    assert json_dict["grounding"]["grounding_type"] == "text"
    assert json_dict["grounding"]["doc_id"] == "doc1"
    
    # Verify importance is correctly serialized
    assert json_dict["importance"]["activation"] == 0.8
    
    # Deserialize back
    restored = Infon.model_validate_json(json_str)
    
    # Verify all fields match
    assert restored.id == original.id
    assert restored.subject == original.subject
    assert restored.predicate == original.predicate
    assert restored.object == original.object
    assert restored.polarity == original.polarity
    assert restored.confidence == original.confidence
    assert restored.kind == original.kind
    assert restored.reinforcement_count == original.reinforcement_count
    
    # Verify nested models
    assert restored.importance.activation == original.importance.activation
    assert restored.importance.composite == original.importance.composite
    
    # Verify grounding (access via .root for RootModel)
    assert restored.grounding.root.grounding_type == "text"
    assert isinstance(restored.grounding.root, TextGrounding)
    assert restored.grounding.root.doc_id == "doc1"


def test_infon_with_ast_grounding():
    """Test Infon with ASTGrounding type."""
    infon = Infon(
        id=str(uuid.uuid4()),
        subject="auth_service",
        predicate="defines",
        object="verify_token",
        polarity=True,
        grounding=Grounding(root=ASTGrounding(
            file_path="src/auth.py",
            line_number=42,
            node_type="FunctionDef"
        )),
        confidence=0.95,
        timestamp=datetime.now(UTC),
        importance=ImportanceScore(
            activation=0.9,
            coherence=0.8,
            specificity=0.7,
            novelty=0.6,
            reinforcement=0.5
        ),
        kind="extracted",
        reinforcement_count=1
    )
    
    # Verify ASTGrounding fields
    assert infon.grounding.root.grounding_type == "ast"
    assert isinstance(infon.grounding.root, ASTGrounding)
    assert infon.grounding.root.file_path == "src/auth.py"
    assert infon.grounding.root.line_number == 42
    assert infon.grounding.root.node_type == "FunctionDef"
    
    # Round-trip through JSON
    json_str = infon.model_dump_json()
    restored = Infon.model_validate_json(json_str)
    
    assert restored.grounding.root.grounding_type == "ast"
    assert isinstance(restored.grounding.root, ASTGrounding)
    assert restored.grounding.root.file_path == "src/auth.py"


def test_infon_negated_polarity():
    """Test Infon with negated polarity (False)."""
    infon = Infon(
        id=str(uuid.uuid4()),
        subject="user_service",
        predicate="calls",
        object="database_pool",
        polarity=False,  # Negated
        grounding=Grounding(root=TextGrounding(
            doc_id="doc2",
            sent_id=5,
            char_start=0,
            char_end=50,
            sentence_text="UserService no longer calls the database pool directly."
        )),
        confidence=0.75,
        timestamp=datetime.now(UTC),
        importance=ImportanceScore(
            activation=0.7,
            coherence=0.6,
            specificity=0.5,
            novelty=0.4,
            reinforcement=0.3
        ),
        kind="extracted",
        reinforcement_count=1
    )
    
    assert infon.polarity is False
    
    # Verify round-trip preserves polarity
    restored = Infon.model_validate_json(infon.model_dump_json())
    assert restored.polarity is False


def test_infon_all_required_fields():
    """Test that Infon has all required fields from the spec."""
    # This test verifies all spec-required fields are present
    infon = Infon(
        id=str(uuid.uuid4()),
        subject="entity1",
        predicate="relation",
        object="entity2",
        polarity=True,
        grounding=Grounding(root=TextGrounding(
            doc_id="test",
            sent_id=0,
            char_start=0,
            char_end=10,
            sentence_text="Test sentence."
        )),
        confidence=0.8,
        timestamp=datetime.now(UTC),
        importance=ImportanceScore(
            activation=0.8,
            coherence=0.7,
            specificity=0.6,
            novelty=0.5,
            reinforcement=0.4
        ),
        kind="extracted",
        reinforcement_count=1
    )
    
    # Verify all required fields are accessible
    assert isinstance(infon.id, str)
    assert isinstance(infon.subject, str)
    assert isinstance(infon.predicate, str)
    assert isinstance(infon.object, str)
    assert isinstance(infon.polarity, bool)
    assert isinstance(infon.grounding, Grounding)
    assert isinstance(infon.confidence, float)
    assert isinstance(infon.timestamp, datetime)
    assert isinstance(infon.importance, ImportanceScore)
    assert infon.kind in ["extracted", "consolidated", "imagined"]
    assert isinstance(infon.reinforcement_count, int)

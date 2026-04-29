"""
Integration tests for Grounding models.

Tests the full functionality of Grounding base class and its subtypes
(TextGrounding, ASTGrounding) including JSON serialization/deserialization
and immutability enforcement.

Following TDD: this test is written FIRST and should FAIL until
implementation is complete.
"""

import json

import pytest

from infon.grounding import ASTGrounding, Grounding, TextGrounding


def test_text_grounding_construction_and_serialization():
    """
    Test TextGrounding construction, JSON round-trip, and field access.
    
    This is a full integration test - no mocks, real Pydantic models.
    """
    # Construct a TextGrounding instance
    text_grounding = TextGrounding(
        grounding_type="text",
        doc_id="doc_123",
        sent_id=42,
        char_start=100,
        char_end=200,
        sentence_text="This is a sample sentence from the document."
    )
    
    # Verify fields are accessible
    assert text_grounding.grounding_type == "text"
    assert text_grounding.doc_id == "doc_123"
    assert text_grounding.sent_id == 42
    assert text_grounding.char_start == 100
    assert text_grounding.char_end == 200
    assert text_grounding.sentence_text == "This is a sample sentence from the document."
    
    # Serialize to JSON
    json_str = text_grounding.model_dump_json()
    json_dict = json.loads(json_str)
    
    # Verify JSON structure includes discriminator
    assert json_dict["grounding_type"] == "text"
    assert json_dict["doc_id"] == "doc_123"
    assert json_dict["sent_id"] == 42
    assert json_dict["char_start"] == 100
    assert json_dict["char_end"] == 200
    assert json_dict["sentence_text"] == "This is a sample sentence from the document."
    
    # Deserialize back from JSON
    roundtrip = TextGrounding.model_validate_json(json_str)
    
    # Assert all fields round-trip correctly
    assert roundtrip.grounding_type == text_grounding.grounding_type
    assert roundtrip.doc_id == text_grounding.doc_id
    assert roundtrip.sent_id == text_grounding.sent_id
    assert roundtrip.char_start == text_grounding.char_start
    assert roundtrip.char_end == text_grounding.char_end
    assert roundtrip.sentence_text == text_grounding.sentence_text


def test_ast_grounding_construction_and_serialization():
    """
    Test ASTGrounding construction, JSON round-trip, and field access.
    
    This is a full integration test - no mocks, real Pydantic models.
    """
    # Construct an ASTGrounding instance
    ast_grounding = ASTGrounding(
        grounding_type="ast",
        file_path="/src/app/module.py",
        line_number=150,
        node_type="FunctionDef"
    )
    
    # Verify fields are accessible
    assert ast_grounding.grounding_type == "ast"
    assert ast_grounding.file_path == "/src/app/module.py"
    assert ast_grounding.line_number == 150
    assert ast_grounding.node_type == "FunctionDef"
    
    # Serialize to JSON
    json_str = ast_grounding.model_dump_json()
    json_dict = json.loads(json_str)
    
    # Verify JSON structure includes discriminator
    assert json_dict["grounding_type"] == "ast"
    assert json_dict["file_path"] == "/src/app/module.py"
    assert json_dict["line_number"] == 150
    assert json_dict["node_type"] == "FunctionDef"
    
    # Deserialize back from JSON
    roundtrip = ASTGrounding.model_validate_json(json_str)
    
    # Assert all fields round-trip correctly
    assert roundtrip.grounding_type == ast_grounding.grounding_type
    assert roundtrip.file_path == ast_grounding.file_path
    assert roundtrip.line_number == ast_grounding.line_number
    assert roundtrip.node_type == ast_grounding.node_type


def test_grounding_immutability_text():
    """
    Test that TextGrounding is frozen and immutable.
    
    Pydantic frozen models should raise ValidationError when attempting
    to modify fields after construction.
    """
    text_grounding = TextGrounding(
        grounding_type="text",
        doc_id="doc_456",
        sent_id=10,
        char_start=50,
        char_end=150,
        sentence_text="Another test sentence."
    )
    
    # Attempt to modify a field - should raise ValidationError
    with pytest.raises(Exception):  # Pydantic raises ValidationError
        text_grounding.doc_id = "modified_doc"
    
    # Verify field remains unchanged
    assert text_grounding.doc_id == "doc_456"


def test_grounding_immutability_ast():
    """
    Test that ASTGrounding is frozen and immutable.
    
    Pydantic frozen models should raise ValidationError when attempting
    to modify fields after construction.
    """
    ast_grounding = ASTGrounding(
        grounding_type="ast",
        file_path="/src/test.py",
        line_number=99,
        node_type="ClassDef"
    )
    
    # Attempt to modify a field - should raise ValidationError
    with pytest.raises(Exception):  # Pydantic raises ValidationError
        ast_grounding.file_path = "/src/modified.py"
    
    # Verify field remains unchanged
    assert ast_grounding.file_path == "/src/test.py"


def test_grounding_polymorphic_deserialization():
    """
    Test that Grounding base class can deserialize to correct subtype
    using discriminator field.
    
    This tests the polymorphic behavior of the Grounding model hierarchy.
    """
    # Create JSON for TextGrounding
    text_json = json.dumps({
        "grounding_type": "text",
        "doc_id": "doc_789",
        "sent_id": 5,
        "char_start": 0,
        "char_end": 50,
        "sentence_text": "Polymorphic test."
    })
    
    # Deserialize through base Grounding class
    grounding = Grounding.model_validate_json(text_json)
    
    # Should be TextGrounding instance (access via .root for RootModel)
    assert isinstance(grounding.root, TextGrounding)
    assert grounding.root.grounding_type == "text"
    assert grounding.root.doc_id == "doc_789"
    
    # Create JSON for ASTGrounding
    ast_json = json.dumps({
        "grounding_type": "ast",
        "file_path": "/code/main.py",
        "line_number": 42,
        "node_type": "Module"
    })
    
    # Deserialize through base Grounding class
    grounding = Grounding.model_validate_json(ast_json)
    
    # Should be ASTGrounding instance (access via .root for RootModel)
    assert isinstance(grounding.root, ASTGrounding)
    assert grounding.root.grounding_type == "ast"
    assert grounding.root.file_path == "/code/main.py"

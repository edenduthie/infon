"""
Integration tests for ImportanceScore model.

Tests the full functionality of ImportanceScore including construction,
composite property calculation (weighted average), JSON serialization/
deserialization, and immutability enforcement.

Following TDD: this test is written FIRST and should FAIL until
implementation is complete.
"""

import json

import pytest

from infon.infon import ImportanceScore


def test_importance_score_construction_and_fields():
    """
    Test ImportanceScore construction and field access.
    
    This is a full integration test - no mocks, real Pydantic model.
    """
    # Construct an ImportanceScore instance
    score = ImportanceScore(
        activation=0.8,
        coherence=0.6,
        specificity=0.9,
        novelty=0.4,
        reinforcement=0.7
    )
    
    # Verify all fields are accessible
    assert score.activation == 0.8
    assert score.coherence == 0.6
    assert score.specificity == 0.9
    assert score.novelty == 0.4
    assert score.reinforcement == 0.7


def test_importance_score_composite_property():
    """
    Test that composite property returns the weighted average of all components.
    
    The composite is the average of activation, coherence, specificity, 
    novelty, and reinforcement.
    """
    # Create score with known values
    score = ImportanceScore(
        activation=0.8,
        coherence=0.6,
        specificity=0.9,
        novelty=0.4,
        reinforcement=0.7
    )
    
    # Expected composite: (0.8 + 0.6 + 0.9 + 0.4 + 0.7) / 5 = 3.4 / 5 = 0.68
    expected_composite = (0.8 + 0.6 + 0.9 + 0.4 + 0.7) / 5.0
    assert score.composite == pytest.approx(expected_composite, abs=1e-6)
    
    # Test with different values
    score2 = ImportanceScore(
        activation=1.0,
        coherence=1.0,
        specificity=1.0,
        novelty=1.0,
        reinforcement=1.0
    )
    assert score2.composite == pytest.approx(1.0, abs=1e-6)
    
    # Test with zeros
    score3 = ImportanceScore(
        activation=0.0,
        coherence=0.0,
        specificity=0.0,
        novelty=0.0,
        reinforcement=0.0
    )
    assert score3.composite == pytest.approx(0.0, abs=1e-6)


def test_importance_score_json_serialization():
    """
    Test ImportanceScore JSON serialization and deserialization.
    
    Verifies that the model can be serialized to JSON and deserialized
    back with all fields intact.
    """
    # Construct an ImportanceScore
    score = ImportanceScore(
        activation=0.75,
        coherence=0.85,
        specificity=0.65,
        novelty=0.55,
        reinforcement=0.95
    )
    
    # Serialize to JSON
    json_str = score.model_dump_json()
    json_dict = json.loads(json_str)
    
    # Verify JSON structure
    assert json_dict["activation"] == 0.75
    assert json_dict["coherence"] == 0.85
    assert json_dict["specificity"] == 0.65
    assert json_dict["novelty"] == 0.55
    assert json_dict["reinforcement"] == 0.95
    
    # Deserialize back from JSON
    roundtrip = ImportanceScore.model_validate_json(json_str)
    
    # Assert all fields round-trip correctly
    assert roundtrip.activation == score.activation
    assert roundtrip.coherence == score.coherence
    assert roundtrip.specificity == score.specificity
    assert roundtrip.novelty == score.novelty
    assert roundtrip.reinforcement == score.reinforcement
    assert roundtrip.composite == score.composite


def test_importance_score_immutability():
    """
    Test that ImportanceScore is frozen and immutable.
    
    Pydantic frozen models should raise ValidationError when attempting
    to modify fields after construction.
    """
    score = ImportanceScore(
        activation=0.5,
        coherence=0.6,
        specificity=0.7,
        novelty=0.8,
        reinforcement=0.9
    )
    
    # Attempt to modify a field - should raise an exception
    with pytest.raises(Exception):  # Pydantic raises ValidationError
        score.activation = 0.99
    
    # Verify field remains unchanged
    assert score.activation == 0.5
    
    # Try modifying another field
    with pytest.raises(Exception):
        score.composite = 0.99
    
    # Composite should still return the calculated value
    expected_composite = (0.5 + 0.6 + 0.7 + 0.8 + 0.9) / 5.0
    assert score.composite == pytest.approx(expected_composite, abs=1e-6)


def test_importance_score_field_validation():
    """
    Test that ImportanceScore validates field ranges [0, 1].
    
    All fields should be floats in the range [0, 1].
    """
    # Valid boundary values
    score_min = ImportanceScore(
        activation=0.0,
        coherence=0.0,
        specificity=0.0,
        novelty=0.0,
        reinforcement=0.0
    )
    assert score_min.activation == 0.0
    
    score_max = ImportanceScore(
        activation=1.0,
        coherence=1.0,
        specificity=1.0,
        novelty=1.0,
        reinforcement=1.0
    )
    assert score_max.activation == 1.0
    
    # Test out-of-range values should raise validation error
    with pytest.raises(Exception):  # Pydantic ValidationError
        ImportanceScore(
            activation=1.5,  # Out of range
            coherence=0.5,
            specificity=0.5,
            novelty=0.5,
            reinforcement=0.5
        )
    
    with pytest.raises(Exception):  # Pydantic ValidationError
        ImportanceScore(
            activation=0.5,
            coherence=-0.1,  # Out of range
            specificity=0.5,
            novelty=0.5,
            reinforcement=0.5
        )

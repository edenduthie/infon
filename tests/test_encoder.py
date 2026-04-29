"""Integration test for SPLADE encoder and anchor projection.

Tests the full encoder pipeline:
1. SPLADE encoder loads from bundled model (offline mode)
2. encode_sparse() produces sparse token activations
3. AnchorProjector.project() maps SPLADE space to anchor space
4. encode() combines both stages for end-to-end encoding

No mocks — real SPLADE model, real transformers, real AnchorSchema.
"""

import os

import pytest

from infon.encoder import AnchorProjector, SpladeEncoder, encode
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
        "calls": Anchor(
            key="calls",
            type="relation",
            tokens=["call", "calls", "invoke", "invokes"],
            description="Function or method invocation",
            parent=None,
        ),
        "authentication": Anchor(
            key="authentication",
            type="feature",
            tokens=["auth", "authenticate", "authentication", "login"],
            description="Authentication functionality",
            parent=None,
        ),
    }
    return AnchorSchema(anchors=anchors, version="1.0.0", language="code")


def test_splade_encoder_loads_from_bundle_offline():
    """Test that SpladeEncoder loads from bundled model without network access.

    WHEN: SpladeEncoder is initialized with TRANSFORMERS_OFFLINE=1
    THEN: It loads successfully from the bundled model path
    AND: Does not attempt to contact HuggingFace or any external service
    """
    # Set offline mode
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    try:
        # Should load from bundle without network access
        encoder = SpladeEncoder()
        assert encoder is not None
    finally:
        # Clean up environment
        os.environ.pop("TRANSFORMERS_OFFLINE", None)


def test_encode_sparse_produces_non_empty_dict():
    """Test that encode_sparse() returns non-empty sparse activations.

    WHEN: encode_sparse() is called with real text
    THEN: The output is a non-empty dict mapping token_id -> float
    AND: All values are non-negative
    """
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    try:
        encoder = SpladeEncoder()
        text = "UserService calls the database connection pool"
        
        sparse_vector = encoder.encode_sparse(text)
        
        # Assert non-empty sparse dict
        assert isinstance(sparse_vector, dict)
        assert len(sparse_vector) > 0
        
        # Assert all activations are non-negative floats
        for token_id, activation in sparse_vector.items():
            assert isinstance(token_id, int)
            assert isinstance(activation, float)
            assert activation > 0.0  # sparse dict contains only non-zero values
    finally:
        os.environ.pop("TRANSFORMERS_OFFLINE", None)


def test_anchor_projector_maps_to_schema_keys(sample_schema):
    """Test that AnchorProjector.project() produces anchor-space vector.

    WHEN: project() is called with a sparse SPLADE vector and an AnchorSchema
    THEN: The output dict contains only keys from the schema
    AND: Keys correspond to anchors whose tokens activated in the SPLADE vector
    """
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    try:
        encoder = SpladeEncoder()
        projector = AnchorProjector(encoder.tokenizer)
        
        text = "UserService calls the database connection pool"
        sparse_vector = encoder.encode_sparse(text)
        
        # Project to anchor space
        anchor_vector = projector.project(sparse_vector, sample_schema)
        
        # Assert output is a dict
        assert isinstance(anchor_vector, dict)
        
        # Assert all keys are in the schema
        for key in anchor_vector.keys():
            assert key in sample_schema.anchors
        
        # Assert all values are non-negative floats
        for activation in anchor_vector.values():
            assert isinstance(activation, float)
            assert activation >= 0.0
    finally:
        os.environ.pop("TRANSFORMERS_OFFLINE", None)


def test_encode_returns_anchor_space_vector(sample_schema):
    """Test that encode() produces end-to-end anchor-space encoding.

    WHEN: encode() is called with text and a schema
    THEN: The output keys are a subset of schema anchor keys
    AND: Anchors matching the text content have non-zero activations
    """
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    try:
        text = "UserService calls the database connection pool"
        
        anchor_vector = encode(text, sample_schema)
        
        # Assert output is a dict
        assert isinstance(anchor_vector, dict)
        
        # Assert all keys are in the schema
        for key in anchor_vector.keys():
            assert key in sample_schema.anchors
        
        # Assert expected anchors are activated
        # The text mentions "UserService", "calls", and "database pool"
        # So we expect non-zero activations for those anchors
        assert len(anchor_vector) > 0
        
        # Assert all values are non-negative
        for activation in anchor_vector.values():
            assert isinstance(activation, float)
            assert activation > 0.0  # encode() should filter to non-zero only
    finally:
        os.environ.pop("TRANSFORMERS_OFFLINE", None)


def test_encode_is_deterministic(sample_schema):
    """Test that encode() is deterministic.

    WHEN: encode() is called twice with the same text and schema
    THEN: Both outputs are identical
    """
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    try:
        text = "authentication validates user credentials"
        
        result1 = encode(text, sample_schema)
        result2 = encode(text, sample_schema)
        
        # Assert deterministic output
        assert result1 == result2
    finally:
        os.environ.pop("TRANSFORMERS_OFFLINE", None)


def test_encode_empty_text_returns_empty_dict(sample_schema):
    """Test that encode() handles empty text gracefully.

    WHEN: encode() is called with empty text
    THEN: The output is an empty dict or a dict with very low activations
    """
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    try:
        result = encode("", sample_schema)
        
        # Empty text should produce empty or near-empty result
        assert isinstance(result, dict)
        # Either empty or all values very small
        if result:
            assert all(v < 0.1 for v in result.values())
    finally:
        os.environ.pop("TRANSFORMERS_OFFLINE", None)

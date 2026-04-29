"""Integration tests for schema auto-discovery via spectral clustering.

Tests the SchemaDiscovery class that derives an AnchorSchema from a corpus
using spectral clustering on the SPLADE co-activation matrix.

All tests use real SpladeEncoder, real tree-sitter parsers, and real fixtures.
No mocks or stubs.
"""

import tempfile
import warnings
from pathlib import Path

from infon.discovery import SchemaDiscovery
from infon.schema import CODE_RELATION_ANCHORS


def test_discover_code_mode_has_eight_builtin_anchors():
    """Test that schema discovery in code mode produces the eight built-in relation anchors.
    
    This test runs SchemaDiscovery.discover() on tests/fixtures/ with mode='code',
    asserts the schema contains all eight built-in relation anchors, verifies JSON
    round-trip, and checks for small corpus warning.
    """
    # Arrange: Get path to fixtures directory
    fixtures_dir = Path(__file__).parent / "fixtures"
    assert fixtures_dir.exists(), f"Fixtures directory not found: {fixtures_dir}"
    
    # Act: Run schema discovery in code mode
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        discovery = SchemaDiscovery()
        schema = discovery.discover(str(fixtures_dir), mode="code")
        
        # Assert: Small corpus warning was emitted
        warning_messages = [str(warning.message) for warning in w]
        assert any(
            "fewer than 50 files" in msg.lower() for msg in warning_messages
        ), f"Expected small corpus warning, got: {warning_messages}"
    
    # Assert: Schema contains all eight built-in relation anchors
    assert schema.language == "code"
    assert schema.version == "auto-1.0"
    
    for key in CODE_RELATION_ANCHORS.keys():
        assert key in schema.anchors, f"Missing built-in relation anchor: {key}"
        anchor = schema.anchors[key]
        assert anchor.key == key
        assert anchor.type == "relation"
    
    # Assert: All eight relation anchors are present
    relation_keys = {a.key for a in schema.relations}
    for key in ["calls", "imports", "inherits", "mutates", "defines", "returns", "raises", "decorates"]:
        assert key in relation_keys, f"Missing relation anchor: {key}"
    
    # Assert: Schema has some actor anchors derived from the corpus
    assert len(schema.actors) > 0, "Expected at least one actor anchor from corpus"
    
    # Assert: JSON round-trip works
    with tempfile.TemporaryDirectory() as tmpdir:
        schema_path = Path(tmpdir) / "schema.json"
        schema.to_json(schema_path)
        
        # Verify file was written
        assert schema_path.exists()
        
        # Load it back
        from infon.schema import AnchorSchema
        loaded_schema = AnchorSchema.from_json(schema_path)
        
        # Verify all eight anchors are still present
        for key in CODE_RELATION_ANCHORS.keys():
            assert key in loaded_schema.anchors
            assert loaded_schema.anchors[key].type == "relation"
        
        # Verify actor anchors were preserved
        assert len(loaded_schema.actors) == len(schema.actors)
        
        # Verify version and language
        assert loaded_schema.version == "auto-1.0"
        assert loaded_schema.language == "code"


def test_discover_validates_mode():
    """Test that discover() raises on invalid mode."""
    discovery = SchemaDiscovery()
    fixtures_dir = Path(__file__).parent / "fixtures"
    
    try:
        schema = discovery.discover(str(fixtures_dir), mode="invalid_mode")
        assert False, "Expected ValueError for invalid mode"
    except ValueError as e:
        assert "mode" in str(e).lower()


def test_discover_empty_corpus_warning():
    """Test that discovery on an empty directory still completes and warns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            discovery = SchemaDiscovery()
            schema = discovery.discover(tmpdir, mode="code")
            
            # Should still produce valid schema with eight anchors
            assert len(schema.anchors) >= 8
            for key in CODE_RELATION_ANCHORS.keys():
                assert key in schema.anchors
            
            # Should have emitted warning
            warning_messages = [str(warning.message) for warning in w]
            assert any("fewer than 50 files" in msg.lower() for msg in warning_messages)

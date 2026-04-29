"""Integration tests for Anchor and AnchorSchema models.

Tests exercise:
- Anchor creation and serialization
- AnchorSchema creation with anchors dict
- JSON serialization (to_json) and deserialization (from_json)
- Hierarchy traversal (ancestors, descendants)
- CODE_RELATION_ANCHORS presence in code-mode schemas
- Round-trip JSON fidelity
- Property accessors (actors, relations, features, markets, locations)

No mocks. Real file I/O to temp directories.
"""

import json
import tempfile
from pathlib import Path

import pytest

from infon.schema import CODE_RELATION_ANCHORS, Anchor, AnchorSchema


class TestAnchor:
    """Tests for the Anchor model."""

    def test_anchor_creation(self):
        """Test that Anchor instances can be created with all required fields."""
        anchor = Anchor(
            key="user_service",
            type="actor",
            tokens=["user", "userservice", "user_service"],
            description="User authentication and management service",
            parent=None,
        )
        assert anchor.key == "user_service"
        assert anchor.type == "actor"
        assert anchor.tokens == ["user", "userservice", "user_service"]
        assert anchor.description == "User authentication and management service"
        assert anchor.parent is None

    def test_anchor_with_parent(self):
        """Test that Anchor can have a parent link."""
        child = Anchor(
            key="postgresql",
            type="actor",
            tokens=["postgres", "postgresql", "pg"],
            description="PostgreSQL database",
            parent="database_layer",
        )
        assert child.parent == "database_layer"

    def test_anchor_immutability(self):
        """Test that Anchor is frozen (immutable)."""
        anchor = Anchor(
            key="test_anchor",
            type="relation",
            tokens=["test"],
            description="Test anchor",
            parent=None,
        )
        with pytest.raises(Exception):  # Pydantic raises ValidationError or AttributeError
            anchor.key = "modified"  # type: ignore

    def test_anchor_json_serialization(self):
        """Test that Anchor can round-trip through JSON."""
        original = Anchor(
            key="database_pool",
            type="feature",
            tokens=["pool", "connection_pool", "database_pool"],
            description="Database connection pooling",
            parent="database_layer",
        )
        json_str = original.model_dump_json()
        data = json.loads(json_str)
        restored = Anchor(**data)
        assert restored.key == original.key
        assert restored.type == original.type
        assert restored.tokens == original.tokens
        assert restored.description == original.description
        assert restored.parent == original.parent


class TestAnchorSchema:
    """Tests for the AnchorSchema model."""

    def test_schema_creation(self):
        """Test that AnchorSchema can be created with anchors dict."""
        anchors = {
            "user_service": Anchor(
                key="user_service",
                type="actor",
                tokens=["user", "userservice"],
                description="User service",
                parent=None,
            ),
            "calls": Anchor(
                key="calls",
                type="relation",
                tokens=["call", "calls", "invoke"],
                description="Function call relation",
                parent=None,
            ),
        }
        schema = AnchorSchema(anchors=anchors, version="1.0.0", language="code")
        assert len(schema.anchors) == 2
        assert schema.version == "1.0.0"
        assert schema.language == "code"

    def test_property_accessors(self):
        """Test that property accessors filter anchors by type."""
        anchors = {
            "user_service": Anchor(
                key="user_service",
                type="actor",
                tokens=["user"],
                description="User service",
                parent=None,
            ),
            "auth_service": Anchor(
                key="auth_service",
                type="actor",
                tokens=["auth"],
                description="Auth service",
                parent=None,
            ),
            "calls": Anchor(
                key="calls",
                type="relation",
                tokens=["call"],
                description="Calls relation",
                parent=None,
            ),
            "database_feature": Anchor(
                key="database_feature",
                type="feature",
                tokens=["database"],
                description="Database feature",
                parent=None,
            ),
            "us_market": Anchor(
                key="us_market",
                type="market",
                tokens=["us", "usa"],
                description="US market",
                parent=None,
            ),
            "san_francisco": Anchor(
                key="san_francisco",
                type="location",
                tokens=["sf", "san_francisco"],
                description="San Francisco",
                parent=None,
            ),
        }
        schema = AnchorSchema(anchors=anchors, version="1.0.0", language="text")
        
        assert len(schema.actors) == 2
        assert all(a.type == "actor" for a in schema.actors)
        
        assert len(schema.relations) == 1
        assert all(a.type == "relation" for a in schema.relations)
        
        assert len(schema.features) == 1
        assert all(a.type == "feature" for a in schema.features)
        
        assert len(schema.markets) == 1
        assert all(a.type == "market" for a in schema.markets)
        
        assert len(schema.locations) == 1
        assert all(a.type == "location" for a in schema.locations)

    def test_code_relation_anchors_constant(self):
        """Test that CODE_RELATION_ANCHORS contains the eight built-in anchors."""
        assert len(CODE_RELATION_ANCHORS) == 8
        expected_keys = {
            "calls",
            "imports",
            "inherits",
            "mutates",
            "defines",
            "returns",
            "raises",
            "decorates",
        }
        assert set(CODE_RELATION_ANCHORS.keys()) == expected_keys
        # Verify all are relation-type
        assert all(a.type == "relation" for a in CODE_RELATION_ANCHORS.values())

    def test_ancestors_traversal(self):
        """Test that ancestors() walks parent links correctly."""
        # Build hierarchy: root -> middle -> leaf
        anchors = {
            "root": Anchor(
                key="root",
                type="actor",
                tokens=["root"],
                description="Root",
                parent=None,
            ),
            "middle": Anchor(
                key="middle",
                type="actor",
                tokens=["middle"],
                description="Middle",
                parent="root",
            ),
            "leaf": Anchor(
                key="leaf",
                type="actor",
                tokens=["leaf"],
                description="Leaf",
                parent="middle",
            ),
        }
        schema = AnchorSchema(anchors=anchors, version="1.0.0", language="text")
        
        # Leaf should have middle and root as ancestors
        ancestors = schema.ancestors("leaf")
        assert ancestors == ["middle", "root"]
        
        # Middle should have root as ancestor
        ancestors = schema.ancestors("middle")
        assert ancestors == ["root"]
        
        # Root should have no ancestors
        ancestors = schema.ancestors("root")
        assert ancestors == []

    def test_descendants_traversal(self):
        """Test that descendants() finds all child anchors."""
        # Build hierarchy:
        #   database_layer -> postgresql
        #   database_layer -> redis
        #   database_layer -> dynamodb
        anchors = {
            "database_layer": Anchor(
                key="database_layer",
                type="actor",
                tokens=["database"],
                description="Database layer",
                parent=None,
            ),
            "postgresql": Anchor(
                key="postgresql",
                type="actor",
                tokens=["postgres"],
                description="PostgreSQL",
                parent="database_layer",
            ),
            "redis": Anchor(
                key="redis",
                type="actor",
                tokens=["redis"],
                description="Redis",
                parent="database_layer",
            ),
            "dynamodb": Anchor(
                key="dynamodb",
                type="actor",
                tokens=["dynamo"],
                description="DynamoDB",
                parent="database_layer",
            ),
        }
        schema = AnchorSchema(anchors=anchors, version="1.0.0", language="text")
        
        descendants = schema.descendants("database_layer")
        assert set(descendants) == {"postgresql", "redis", "dynamodb"}

    def test_descendants_multi_level(self):
        """Test that descendants() walks multiple levels."""
        # Build hierarchy:
        #   root -> child1 -> grandchild1
        #        -> child2
        anchors = {
            "root": Anchor(
                key="root",
                type="actor",
                tokens=["root"],
                description="Root",
                parent=None,
            ),
            "child1": Anchor(
                key="child1",
                type="actor",
                tokens=["child1"],
                description="Child 1",
                parent="root",
            ),
            "child2": Anchor(
                key="child2",
                type="actor",
                tokens=["child2"],
                description="Child 2",
                parent="root",
            ),
            "grandchild1": Anchor(
                key="grandchild1",
                type="actor",
                tokens=["grandchild1"],
                description="Grandchild 1",
                parent="child1",
            ),
        }
        schema = AnchorSchema(anchors=anchors, version="1.0.0", language="text")
        
        # Root should have all descendants
        descendants = schema.descendants("root")
        assert set(descendants) == {"child1", "child2", "grandchild1"}
        
        # Child1 should have grandchild1
        descendants = schema.descendants("child1")
        assert descendants == ["grandchild1"]


class TestAnchorSchemaJSON:
    """Tests for JSON serialization and file I/O."""

    def test_to_json_and_from_json_round_trip(self):
        """Test that schema can be written to JSON file and reloaded identically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            schema_path = Path(tmpdir) / "schema.json"
            
            # Create schema with parent-child relationships (use text mode to avoid automatic built-ins)
            anchors = {
                "database_layer": Anchor(
                    key="database_layer",
                    type="actor",
                    tokens=["database", "db"],
                    description="Database abstraction layer",
                    parent=None,
                ),
                "postgresql": Anchor(
                    key="postgresql",
                    type="actor",
                    tokens=["postgres", "postgresql", "pg"],
                    description="PostgreSQL database",
                    parent="database_layer",
                ),
                "calls": Anchor(
                    key="calls",
                    type="relation",
                    tokens=["call", "calls", "invoke"],
                    description="Function call relation",
                    parent=None,
                ),
            }
            original = AnchorSchema(anchors=anchors, version="1.0.0", language="text")
            
            # Write to JSON
            original.to_json(schema_path)
            
            # Verify file exists and is valid JSON
            assert schema_path.exists()
            with open(schema_path) as f:
                data = json.load(f)
            assert "anchors" in data
            assert "version" in data
            assert "language" in data
            
            # Reload from JSON
            restored = AnchorSchema.from_json(schema_path)
            
            # Verify all fields match
            assert restored.version == original.version
            assert restored.language == original.language
            assert len(restored.anchors) == len(original.anchors)
            
            for key, anchor in original.anchors.items():
                assert key in restored.anchors
                restored_anchor = restored.anchors[key]
                assert restored_anchor.key == anchor.key
                assert restored_anchor.type == anchor.type
                assert restored_anchor.tokens == anchor.tokens
                assert restored_anchor.description == anchor.description
                assert restored_anchor.parent == anchor.parent

    def test_from_json_with_code_mode_includes_builtins(self):
        """Test that loading a code-mode schema includes CODE_RELATION_ANCHORS."""
        with tempfile.TemporaryDirectory() as tmpdir:
            schema_path = Path(tmpdir) / "schema.json"
            
            # Create code-mode schema with just one custom anchor
            anchors = {
                "user_service": Anchor(
                    key="user_service",
                    type="actor",
                    tokens=["user"],
                    description="User service",
                    parent=None,
                ),
            }
            original = AnchorSchema(anchors=anchors, version="1.0.0", language="code")
            original.to_json(schema_path)
            
            # Reload
            restored = AnchorSchema.from_json(schema_path)
            
            # Verify CODE_RELATION_ANCHORS are present
            for key in CODE_RELATION_ANCHORS.keys():
                assert key in restored.anchors, f"Missing built-in anchor: {key}"
                assert restored.anchors[key].type == "relation"

    def test_from_json_missing_file_raises_error(self):
        """Test that loading from non-existent file raises SchemaLoadError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_path = Path(tmpdir) / "nonexistent.json"
            with pytest.raises(Exception):  # Should raise SchemaLoadError
                AnchorSchema.from_json(missing_path)

    def test_from_json_malformed_json_raises_error(self):
        """Test that loading malformed JSON raises SchemaLoadError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            schema_path = Path(tmpdir) / "malformed.json"
            schema_path.write_text("{ this is not valid json }")
            
            with pytest.raises(Exception):  # Should raise SchemaLoadError
                AnchorSchema.from_json(schema_path)

    def test_locations_property(self):
        """Test that locations() method returns location-type anchors."""
        anchors = {
            "san_francisco": Anchor(
                key="san_francisco",
                type="location",
                tokens=["sf", "san_francisco"],
                description="San Francisco",
                parent=None,
            ),
            "new_york": Anchor(
                key="new_york",
                type="location",
                tokens=["ny", "nyc"],
                description="New York",
                parent=None,
            ),
            "user_service": Anchor(
                key="user_service",
                type="actor",
                tokens=["user"],
                description="User service",
                parent=None,
            ),
        }
        schema = AnchorSchema(anchors=anchors, version="1.0.0", language="text")
        
        locations = schema.locations
        assert len(locations) == 2
        assert all(loc.type == "location" for loc in locations)
        location_keys = {loc.key for loc in locations}
        assert location_keys == {"san_francisco", "new_york"}

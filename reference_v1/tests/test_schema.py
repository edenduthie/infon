"""Tests for blackmagic.schema."""
from blackmagic.schema import AnchorSchema


def test_load_from_dict(tiny_schema):
    assert "toyota" in tiny_schema.anchors
    assert tiny_schema.types["toyota"] == "actor"
    assert "invest" in tiny_schema.names


def test_load_from_file(tmp_schema_path):
    s = AnchorSchema.from_file(tmp_schema_path)
    assert len(s.names) > 10
    assert "battery" in s.anchors


def test_types_grouping(tiny_schema):
    assert "toyota" in tiny_schema.by_type["actor"]
    assert "invest" in tiny_schema.by_type["relation"]
    assert "battery" in tiny_schema.by_type["feature"]


def test_empty_schema():
    s = AnchorSchema({})
    assert s.names == []
    assert s.types == {}


def test_hierarchy_traversal():
    defs = {
        "north_america": {"type": "location", "tokens": ["north america"]},
        "us":            {"type": "location", "tokens": ["us"],
                          "parent": "north_america"},
        "canada":        {"type": "location", "tokens": ["canada"],
                          "parent": "north_america"},
    }
    s = AnchorSchema(defs)
    assert s.get_parent("us") == "north_america"
    assert "us" in s.get_children("north_america")
    assert "canada" in s.get_children("north_america")
    assert "north_america" in s.get_ancestors("us")
    descendants = s.get_descendants("north_america")
    assert "us" in descendants and "canada" in descendants


def test_role_mapping(tiny_schema):
    assert tiny_schema.role_for_type("actor") == "subject"
    assert tiny_schema.role_for_type("relation") == "predicate"
    assert tiny_schema.role_for_type("location") == "object"
    assert tiny_schema.role_for_type("feature") == "object"

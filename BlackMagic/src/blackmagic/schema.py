"""Anchor schema: typed vocabulary with hierarchy for situation semantics.

Each anchor has a type, tokens (for text matching), and optional hierarchy
metadata (organisation_type, country_code, level, macro_region, etc.).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

# Hierarchy fields per anchor type (beyond type/tokens)
HIERARCHY_FIELDS = {
    "actor":    {"organisation_type", "country_code", "canonical_name", "parent"},
    "relation": {"level", "domain", "parent"},
    "feature":  {"level", "parent"},
    "market":   {"level", "macro_region", "country_code", "parent"},
    "location": {"level", "macro_region", "country_code", "parent"},
    "temporal": {"precision", "date_iso", "parent"},
    "variant":  {"parent"},
}


class AnchorSchema:
    """Typed anchor vocabulary with hierarchy traversal.

    anchor_defs: {
        "toyota": {"type": "actor", "tokens": ["toyota"],
                    "organisation_type": "private-sector", "country_code": "JP"},
        "invest": {"type": "relation", "tokens": ["invest", "investment"]},
        ...
    }
    """

    _HIERARCHY_KEYS = {
        "level", "macro_region", "country_code", "precision", "date_iso",
        "organisation_type", "canonical_name", "domain", "parent",
    }

    def __init__(self, anchor_defs: dict):
        self.anchors = anchor_defs
        self.names = list(anchor_defs.keys())
        self.types = {name: info["type"] for name, info in anchor_defs.items()}
        self.anchor_types = sorted(set(self.types.values()))

        self.by_type = defaultdict(list)
        for name, atype in self.types.items():
            self.by_type[atype].append(name)

        # Parent/child index
        self._children = defaultdict(list)
        self._parent = {}
        for name, info in anchor_defs.items():
            parent = info.get("parent")
            if parent:
                self._parent[name] = parent
                self._children[parent].append(name)

    @classmethod
    def from_file(cls, path: str | Path) -> AnchorSchema:
        with open(path) as f:
            return cls(json.load(f))

    def get_hierarchy(self, name: str) -> dict:
        info = self.anchors.get(name, {})
        return {k: v for k, v in info.items()
                if k in self._HIERARCHY_KEYS and v is not None}

    def get_parent(self, name: str) -> str | None:
        return self._parent.get(name)

    def get_children(self, name: str) -> list[str]:
        return self._children.get(name, [])

    def get_ancestors(self, name: str) -> list[str]:
        ancestors = []
        current = name
        seen = set()
        while current in self._parent and current not in seen:
            seen.add(current)
            current = self._parent[current]
            ancestors.append(current)
        return ancestors

    def get_descendants(self, name: str) -> list[str]:
        descendants = []
        queue = list(self._children.get(name, []))
        seen = set()
        while queue:
            child = queue.pop(0)
            if child not in seen:
                seen.add(child)
                descendants.append(child)
                queue.extend(self._children.get(child, []))
        return descendants

    def get_anchors_by_field(self, field: str, value) -> list[str]:
        return [n for n, info in self.anchors.items() if info.get(field) == value]

    def role_for_type(self, atype: str) -> str:
        """Map anchor type to triple role."""
        if atype in ("actor",):
            return "subject"
        if atype in ("relation",):
            return "predicate"
        return "object"

    def subject_types(self) -> list[str]:
        return ["actor"]

    def predicate_types(self) -> list[str]:
        return ["relation"]

    def object_types(self) -> list[str]:
        return [t for t in self.anchor_types
                if t not in ("actor", "relation")]

    def save(self, path: str | Path):
        with open(path, "w") as f:
            json.dump(self.anchors, f, indent=2)

"""Anchor and AnchorSchema models for typed vocabulary definition.

The AnchorSchema defines the typed vocabulary of the knowledge base — the set of
named anchors (concepts) and their types. Each anchor has a key, type, tokens
for SPLADE projection, description, and optional parent for hierarchy.

For language=code, the schema always includes the eight built-in code relation
anchors: calls, imports, inherits, mutates, defines, returns, raises, decorates.
"""

import json
from pathlib import Path

from pydantic import BaseModel


class SchemaLoadError(Exception):
    """Raised when schema loading fails (missing file, malformed JSON, validation error)."""

    pass


class Anchor(BaseModel, frozen=True):
    """A single anchor (concept) in the knowledge base vocabulary.

    Attributes:
        key: Unique string identifier (snake_case)
        type: One of actor, relation, feature, market, location
        tokens: List of vocabulary strings that map to this anchor (for SPLADE projector)
        description: Human-readable one-line description
        parent: Optional parent anchor key (enables hierarchy)
    """

    key: str
    type: str  # actor | relation | feature | market | location
    tokens: list[str]
    description: str
    parent: str | None = None


class AnchorSchema(BaseModel, frozen=True):
    """Schema defining the typed vocabulary of the knowledge base.

    Attributes:
        anchors: Dict mapping anchor key to Anchor instance
        version: Semver string
        language: 'text' or 'code'
    """

    anchors: dict[str, Anchor]
    version: str
    language: str  # text | code

    @property
    def actors(self) -> list[Anchor]:
        """Return list of actor-type anchors."""
        return [a for a in self.anchors.values() if a.type == "actor"]

    @property
    def relations(self) -> list[Anchor]:
        """Return list of relation-type anchors."""
        return [a for a in self.anchors.values() if a.type == "relation"]

    @property
    def features(self) -> list[Anchor]:
        """Return list of feature-type anchors."""
        return [a for a in self.anchors.values() if a.type == "feature"]

    @property
    def markets(self) -> list[Anchor]:
        """Return list of market-type anchors."""
        return [a for a in self.anchors.values() if a.type == "market"]

    @property
    def locations(self) -> list[Anchor]:
        """Return list of location-type anchors."""
        return [a for a in self.anchors.values() if a.type == "location"]

    def ancestors(self, key: str) -> list[str]:
        """Return list of ancestor anchor keys by walking parent links.

        Args:
            key: The anchor key to find ancestors for

        Returns:
            List of ancestor keys from immediate parent to root, in order
        """
        result: list[str] = []
        current = self.anchors.get(key)
        if not current:
            return result

        while current and current.parent:
            result.append(current.parent)
            current = self.anchors.get(current.parent)

        return result

    def descendants(self, key: str) -> list[str]:
        """Return list of descendant anchor keys.

        Recursively finds all anchors that have this key as an ancestor.

        Args:
            key: The anchor key to find descendants for

        Returns:
            List of descendant keys (all levels)
        """
        result: list[str] = []

        def _collect_children(parent_key: str) -> None:
            for anchor_key, anchor in self.anchors.items():
                if anchor.parent == parent_key:
                    result.append(anchor_key)
                    _collect_children(anchor_key)

        _collect_children(key)
        return result

    def to_json(self, path: Path | str) -> None:
        """Write schema to JSON file.

        Args:
            path: Path to write JSON file to
        """
        path = Path(path)
        # Convert to dict for JSON serialization
        data = {
            "version": self.version,
            "language": self.language,
            "anchors": {
                key: anchor.model_dump() for key, anchor in self.anchors.items()
            },
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def from_json(cls, path: Path | str) -> "AnchorSchema":
        """Load schema from JSON file.

        For language=code, automatically merges CODE_RELATION_ANCHORS into the
        loaded schema if they are not already present.

        Args:
            path: Path to JSON file

        Returns:
            Loaded AnchorSchema instance

        Raises:
            SchemaLoadError: If file is missing, malformed, or validation fails
        """
        path = Path(path)

        try:
            if not path.exists():
                raise SchemaLoadError(f"Schema file not found: {path}")

            text = path.read_text()
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise SchemaLoadError(f"Malformed JSON in schema file: {e}") from e
        except OSError as e:
            raise SchemaLoadError(f"Failed to read schema file: {e}") from e

        try:
            version = data["version"]
            language = data["language"]
            anchors_data = data["anchors"]

            # Deserialize anchors
            anchors = {key: Anchor(**anchor_data) for key, anchor_data in anchors_data.items()}

            # For code mode, ensure CODE_RELATION_ANCHORS are present
            if language == "code":
                for key, anchor in CODE_RELATION_ANCHORS.items():
                    if key not in anchors:
                        anchors[key] = anchor

            return cls(anchors=anchors, version=version, language=language)
        except (KeyError, TypeError, ValueError) as e:
            raise SchemaLoadError(f"Schema validation failed: {e}") from e


# The eight built-in code relation anchors
# These MUST be present in all code-mode schemas and must not be removed or overridden
CODE_RELATION_ANCHORS: dict[str, Anchor] = {
    "calls": Anchor(
        key="calls",
        type="relation",
        tokens=["call", "calls", "invoke", "invokes", "execute"],
        description="Function or method invocation",
        parent=None,
    ),
    "imports": Anchor(
        key="imports",
        type="relation",
        tokens=["import", "imports", "require", "include"],
        description="Module or dependency import",
        parent=None,
    ),
    "inherits": Anchor(
        key="inherits",
        type="relation",
        tokens=["inherit", "inherits", "extends", "subclass"],
        description="Class inheritance relationship",
        parent=None,
    ),
    "mutates": Anchor(
        key="mutates",
        type="relation",
        tokens=["mutate", "mutates", "modify", "modifies", "update", "updates"],
        description="State mutation or modification",
        parent=None,
    ),
    "defines": Anchor(
        key="defines",
        type="relation",
        tokens=["define", "defines", "declare", "declares"],
        description="Symbol definition or declaration",
        parent=None,
    ),
    "returns": Anchor(
        key="returns",
        type="relation",
        tokens=["return", "returns", "yield", "yields"],
        description="Return value or yielded value",
        parent=None,
    ),
    "raises": Anchor(
        key="raises",
        type="relation",
        tokens=["raise", "raises", "throw", "throws", "error", "exception"],
        description="Exception or error raising",
        parent=None,
    ),
    "decorates": Anchor(
        key="decorates",
        type="relation",
        tokens=["decorate", "decorates", "annotate", "annotates"],
        description="Decorator or annotation application",
        parent=None,
    ),
}

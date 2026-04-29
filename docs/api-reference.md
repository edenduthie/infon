# API Reference

This page documents the Python API for programmatic usage of `infon`. Use these classes and functions to build custom integrations, automation scripts, or tools on top of the infon knowledge base.

---

## Core Classes

### Infon

Represents a single information unit (typed, grounded triple).

**Module**: `infon.infon`

**Definition**:

```python
from dataclasses import dataclass
from datetime import datetime
from infon.grounding import Grounding
from infon.infon import ImportanceScore

@dataclass
class Infon:
    id: str                          # Unique identifier (UUID)
    subject: str                     # Subject anchor
    predicate: str                   # Predicate anchor (relation)
    object: str                      # Object anchor
    polarity: bool                   # True (+) or False (-)
    grounding: Grounding             # Source location
    confidence: float                # 0.0 to 1.0
    timestamp: datetime              # Creation time
    importance: ImportanceScore      # Multi-dimensional importance
    kind: str                        # "extracted", "observed", "inferred"
    reinforcement_count: int         # Number of reinforcements
```

**Example**:

```python
from datetime import UTC, datetime
from infon.infon import Infon, ImportanceScore
from infon.grounding import Grounding, ASTGrounding

infon = Infon(
    id="550e8400-e29b-41d4-a716-446655440000",
    subject="process_data",
    predicate="calls",
    object="validate_input",
    polarity=True,
    grounding=Grounding(
        root=ASTGrounding(
            grounding_type="ast",
            file_path="src/services/data.py",
            line_number=15,
            node_type="call_expression"
        )
    ),
    confidence=1.0,
    timestamp=datetime.now(UTC),
    importance=ImportanceScore(
        activation=0.8,
        coherence=0.7,
        specificity=0.8,
        novelty=0.5,
        reinforcement=0.5
    ),
    kind="extracted",
    reinforcement_count=1
)
```

---

### Grounding

Represents the source location of an infon.

**Module**: `infon.grounding`

**Types**:

```python
from pydantic import BaseModel

class ASTGrounding(BaseModel):
    grounding_type: str = "ast"
    file_path: str          # Path to source file
    line_number: int        # Line number (1-indexed)
    node_type: str          # AST node type

class TextGrounding(BaseModel):
    grounding_type: str = "text"
    doc_id: str             # Document identifier
    sent_id: int            # Sentence index
    char_start: int         # Character start offset
    char_end: int           # Character end offset
    sentence_text: str      # Full sentence text

class Grounding(BaseModel):
    root: ASTGrounding | TextGrounding
```

**Example (AST)**:

```python
from infon.grounding import Grounding, ASTGrounding

grounding = Grounding(
    root=ASTGrounding(
        grounding_type="ast",
        file_path="src/services/user.py",
        line_number=42,
        node_type="call_expression"
    )
)
```

**Example (Text)**:

```python
from infon.grounding import Grounding, TextGrounding

grounding = Grounding(
    root=TextGrounding(
        grounding_type="text",
        doc_id="session-123",
        sent_id=0,
        char_start=0,
        char_end=50,
        sentence_text="UserService has circular dependency with AuthService"
    )
)
```

---

### AnchorSchema

Represents the vocabulary of concepts (anchors) used in infons.

**Module**: `infon.schema`

**Definition**:

```python
from pydantic import BaseModel

class Anchor(BaseModel):
    name: str                    # Canonical token or phrase
    type: str                    # "entity", "relation", "property", "constraint", "event"
    frequency: float = 0.0       # Proportion of units containing this anchor
    aliases: list[str] = []      # Synonyms or variations

class AnchorSchema(BaseModel):
    version: str                 # Schema version
    language: str                # "code", "text"
    anchors: dict[str, Anchor]   # Mapping from name to Anchor
```

**Example**:

```python
from infon.schema import AnchorSchema, Anchor

schema = AnchorSchema(
    version="1.0",
    language="code",
    anchors={
        "calls": Anchor(name="calls", type="relation"),
        "imports": Anchor(name="imports", type="relation"),
        "user": Anchor(name="user", type="entity", frequency=0.15)
    }
)
```

**Load from JSON**:

```python
from pathlib import Path
from infon.schema import AnchorSchema

schema_path = Path(".infon/schema.json")
schema = AnchorSchema.model_validate_json(schema_path.read_text())
```

**Built-in Code Relations**:

```python
from infon.schema import CODE_RELATION_ANCHORS

# Dictionary of eight code relation anchors
# Keys: "calls", "imports", "inherits", "defines", "returns", "raises", "decorates", "mutates"
```

---

### InfonStore

DuckDB-backed persistent storage for infons.

**Module**: `infon.store`

**Definition**:

```python
class InfonStore:
    def __init__(self, db_path: str | Path)
    def insert(self, infons: list[Infon]) -> None
    def upsert(self, infon: Infon) -> None
    def query(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
        limit: int = 100
    ) -> list[Infon]
    def stats(self) -> StoreStats
    def close(self) -> None
```

**Example (Insert)**:

```python
from infon.store import InfonStore
from infon.infon import Infon
from infon.grounding import Grounding, ASTGrounding

store = InfonStore(".infon/kb.ddb")

infons = [
    Infon(
        subject="process_data",
        predicate="calls",
        object="validate_input",
        polarity=True,
        grounding=Grounding(root=ASTGrounding(
            grounding_type="ast",
            file_path="src/data.py",
            line_number=15,
            node_type="call"
        )),
        confidence=1.0,
        # ... other fields
    )
]

store.insert(infons)
store.close()
```

**Example (Query)**:

```python
from infon.store import InfonStore

with InfonStore(".infon/kb.ddb") as store:
    # Query by subject
    results = store.query(subject="process_data")
    
    # Query by predicate
    calls = store.query(predicate="calls", limit=20)
    
    # Query by subject and predicate
    user_calls = store.query(subject="UserService", predicate="calls")
    
    for infon in results:
        print(f"{infon.subject} {infon.predicate} {infon.object}")
```

**Example (Upsert)**:

```python
from infon.store import InfonStore

with InfonStore(".infon/kb.ddb") as store:
    # Upsert merges with existing infon if (subject, predicate, object, polarity) match
    store.upsert(infon)
```

**Example (Stats)**:

```python
from infon.store import InfonStore

with InfonStore(".infon/kb.ddb") as store:
    stats = store.stats()
    print(f"Infons: {stats.infon_count}")
    print(f"Edges: {stats.edge_count}")
    print(f"Top anchors: {stats.top_anchors[:5]}")
```

---

## Extraction Functions

### encode

Encode text to anchor space using SPLADE.

**Module**: `infon.encoder`

**Signature**:

```python
def encode(text: str, schema: AnchorSchema) -> dict[str, float]:
    """
    Encode text to anchor activations.
    
    Args:
        text: Natural language text
        schema: AnchorSchema defining the anchor vocabulary
        
    Returns:
        Dictionary mapping anchor names to activation scores (0.0 to 1.0)
    """
```

**Example**:

```python
from infon.encoder import encode
from infon.schema import AnchorSchema

schema = AnchorSchema.model_validate_json(Path(".infon/schema.json").read_text())

anchor_scores = encode("UserService validates email addresses", schema)
# Returns: {"UserService": 0.85, "validates": 0.70, "email": 0.90, ...}
```

---

### extract_text

Extract infons from natural language text.

**Module**: `infon.extract`

**Signature**:

```python
def extract_text(text: str, doc_id: str, schema: AnchorSchema) -> list[Infon]:
    """
    Extract infons from natural language text.
    
    Args:
        text: The text to extract from
        doc_id: Document identifier for grounding
        schema: AnchorSchema for anchor vocabulary
        
    Returns:
        List of extracted Infons with TextGrounding
    """
```

**Example**:

```python
from infon.extract import extract_text
from infon.schema import AnchorSchema

schema = AnchorSchema.model_validate_json(Path(".infon/schema.json").read_text())

infons = extract_text(
    text="UserService validates email addresses using regex patterns.",
    doc_id="observation-123",
    schema=schema
)

for infon in infons:
    print(f"{infon.subject} {infon.predicate} {infon.object}")
# Output: UserService validates email
#         UserService uses regex_pattern
```

---

### ingest_repo

Ingest a repository (extract AST infons from source files).

**Module**: `infon.ast.ingest`

**Signature**:

```python
def ingest_repo(
    repo_path: Path,
    store: InfonStore,
    schema: AnchorSchema
) -> list[Infon]:
    """
    Ingest all source files from a repository.
    
    Args:
        repo_path: Path to repository root
        store: InfonStore to insert infons into
        schema: AnchorSchema for anchors
        
    Returns:
        List of extracted Infons
    """
```

**Example**:

```python
from pathlib import Path
from infon.ast.ingest import ingest_repo
from infon.store import InfonStore
from infon.schema import AnchorSchema, CODE_RELATION_ANCHORS

schema = AnchorSchema(
    version="1.0",
    language="code",
    anchors=CODE_RELATION_ANCHORS
)

with InfonStore(".infon/kb.ddb") as store:
    infons = ingest_repo(Path.cwd(), store, schema)
    print(f"Extracted {len(infons)} infons")
```

---

### consolidate

Consolidate infons (merge duplicates, build NEXT chains, apply importance decay).

**Module**: `infon.consolidate`

**Signature**:

```python
def consolidate(store: InfonStore, schema: AnchorSchema) -> None:
    """
    Consolidate the knowledge base.
    
    Performs:
    - Duplicate merging (increment reinforcement_count)
    - NEXT-edge temporal chain building
    - Importance decay
    - Constraint aggregation
    
    Args:
        store: InfonStore to consolidate
        schema: AnchorSchema for anchors
    """
```

**Example**:

```python
from infon.consolidate import consolidate
from infon.store import InfonStore
from infon.schema import AnchorSchema

schema = AnchorSchema.model_validate_json(Path(".infon/schema.json").read_text())

with InfonStore(".infon/kb.ddb") as store:
    consolidate(store, schema)
    print("Consolidation complete")
```

---

## AST Extraction

### ExtractorRegistry

Registry for mapping file extensions to AST extractors.

**Module**: `infon.ast.registry`

**Definition**:

```python
class ExtractorRegistry:
    def __init__(self, schema: AnchorSchema)
    def get_extractor(self, file_path: Path) -> BaseASTExtractor | None
    def has_extractor(self, file_path: Path) -> bool
    def supported_extensions(self) -> list[str]
    def register(self, extension: str, extractor_class: type[BaseASTExtractor]) -> None
```

**Example**:

```python
from pathlib import Path
from infon.ast.registry import ExtractorRegistry
from infon.schema import AnchorSchema, CODE_RELATION_ANCHORS

schema = AnchorSchema(version="1.0", language="code", anchors=CODE_RELATION_ANCHORS)
registry = ExtractorRegistry(schema)

# Check if extractor exists for file
if registry.has_extractor(Path("src/main.py")):
    extractor = registry.get_extractor(Path("src/main.py"))
    infons = extractor.extract(Path("src/main.py"))
    print(f"Extracted {len(infons)} infons")

# List supported extensions
print(registry.supported_extensions())
# Output: ['.py', '.ts', '.tsx', '.js', '.jsx']
```

---

### BaseASTExtractor

Abstract base class for language-specific AST extractors.

**Module**: `infon.ast.base`

**Definition**:

```python
from abc import ABC, abstractmethod

class BaseASTExtractor(ABC):
    def __init__(self, schema: AnchorSchema)
    
    @abstractmethod
    def extract(self, file_path: Path) -> list[Infon]:
        """Extract infons from a source file."""
        pass
```

**Example (Custom Extractor)**:

```python
from pathlib import Path
from infon.ast.base import BaseASTExtractor
from infon.infon import Infon
from infon.schema import AnchorSchema

class MyCustomExtractor(BaseASTExtractor):
    def extract(self, file_path: Path) -> list[Infon]:
        infons = []
        # Parse file and extract infons
        # ...
        return infons

# Register with ExtractorRegistry
from infon.ast.registry import ExtractorRegistry

schema = AnchorSchema(...)
registry = ExtractorRegistry(schema)
registry.register(".rs", MyCustomExtractor)
```

---

## Schema Discovery

### SchemaDiscovery

Auto-discover anchor schema from a corpus using spectral clustering.

**Module**: `infon.discovery`

**Definition**:

```python
class SchemaDiscovery:
    def __init__(
        self,
        n_clusters: int = 50,
        top_tokens: int = 2000,
        min_activation: float = 0.1
    )
    
    def discover(self, corpus_path: str, mode: str = "code") -> AnchorSchema:
        """
        Discover AnchorSchema from a corpus.
        
        Args:
            corpus_path: Path to directory containing corpus files
            mode: "code" or "text"
            
        Returns:
            AnchorSchema with auto-discovered anchors
        """
```

**Example**:

```python
from infon.discovery import SchemaDiscovery

discovery = SchemaDiscovery(
    n_clusters=50,
    top_tokens=2000,
    min_activation=0.1
)

schema = discovery.discover(corpus_path=".", mode="code")

# Save schema
from pathlib import Path
Path(".infon/schema.json").write_text(schema.model_dump_json(indent=2))
```

---

## Retrieval

### retrieve

Retrieve infons matching a query with semantic ranking.

**Module**: `infon.retrieve`

**Signature**:

```python
def retrieve(
    query: str,
    store: InfonStore,
    schema: AnchorSchema,
    limit: int = 10,
    persona: str | None = None
) -> list[tuple[Infon, float]]:
    """
    Retrieve infons matching a query.
    
    Args:
        query: Natural language query
        store: InfonStore to search
        schema: AnchorSchema for encoding
        limit: Maximum results
        persona: Optional persona ("investor", "engineer", etc.)
        
    Returns:
        List of (Infon, score) tuples sorted by score descending
    """
```

**Example**:

```python
from infon.retrieve import retrieve
from infon.store import InfonStore
from infon.schema import AnchorSchema

schema = AnchorSchema.model_validate_json(Path(".infon/schema.json").read_text())

with InfonStore(".infon/kb.ddb") as store:
    results = retrieve(
        query="what calls process_data",
        store=store,
        schema=schema,
        limit=10,
        persona="engineer"
    )
    
    for infon, score in results:
        print(f"[{score:.2f}] {infon.subject} {infon.predicate} {infon.object}")
        print(f"  {infon.grounding.root.file_path}:{infon.grounding.root.line_number}")
```

---

## Next Steps

- [CLI Reference](cli.md) — command-line interface
- [MCP Server](mcp.md) — integrate with Claude Code
- [Contributing](contributing.md) — add new features or extractors

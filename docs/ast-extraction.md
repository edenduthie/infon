# AST Extraction

`infon` extracts code relationships from source files using tree-sitter parsers. This page explains how AST extraction works, which languages are supported, and how to add support for new languages.

---

## Overview

AST (Abstract Syntax Tree) extraction converts source code into typed, grounded infons representing code relationships:

- `calls`: Function/method calls
- `imports`: Module/package imports
- `inherits`: Class inheritance
- `defines`: Function/class definitions
- `returns`: Return statements
- `raises`: Exception raising
- `decorates`: Decorator applications
- `mutates`: Attribute mutations (e.g., `self.x = y`)

Each extracted infon is **grounded** to a source location:

```python
ASTGrounding(
    file_path="src/services/user.py",
    line_number=42,
    node_type="call_expression"
)
```

This allows queries to trace facts back to the exact file and line number.

---

## Supported Languages

`infon` currently supports:

### Python (.py)

Extracted relations:

| Relation | Example | Infon |
|----------|---------|-------|
| `imports` | `import numpy` | `<<imports, "module", "numpy"; +>>` |
| `imports` | `from flask import Flask` | `<<imports, "module", "Flask"; +>>` |
| `calls` | `process_data(x)` | `<<calls, "caller", "process_data"; +>>` |
| `inherits` | `class A(B):` | `<<inherits, "A", "B"; +>>` |
| `defines` | `def foo():` | `<<defines, "module", "foo"; +>>` |
| `defines` | `class Bar:` | `<<defines, "module", "Bar"; +>>` |
| `returns` | `return x` | `<<returns, "foo", "x"; +>>` |
| `raises` | `raise ValueError` | `<<raises, "foo", "ValueError"; +>>` |
| `decorates` | `@app.route("/")` | `<<decorates, "app.route", "handler"; +>>` |
| `mutates` | `self.x = y` | `<<mutates, "self", "x"; +>>` |

Uses `tree-sitter-python`.

### TypeScript / JavaScript (.ts, .tsx, .js, .jsx)

Extracted relations:

| Relation | Example | Infon |
|----------|---------|-------|
| `imports` | `import { foo } from 'bar'` | `<<imports, "module", "foo"; +>>` |
| `imports` | `import * as fs from 'fs'` | `<<imports, "module", "fs"; +>>` |
| `calls` | `processData(x)` | `<<calls, "caller", "processData"; +>>` |
| `inherits` | `class A extends B` | `<<inherits, "A", "B"; +>>` |
| `defines` | `function foo() {}` | `<<defines, "module", "foo"; +>>` |
| `defines` | `class Bar {}` | `<<defines, "module", "Bar"; +>>` |
| `returns` | `return x` | `<<returns, "foo", "x"; +>>` |
| `raises` | `throw new Error()` | `<<raises, "foo", "Error"; +>>` |
| `decorates` | `@Component()` | `<<decorates, "Component", "MyClass"; +>>` |
| `mutates` | `this.x = y` | `<<mutates, "this", "x"; +>>` |

Uses `tree-sitter-javascript`.

---

## Extraction Workflow

### 1. File Discovery

`infon init` or `infon ingest` scans the current directory for source files:

```python
# Python files
.py

# TypeScript/JavaScript files
.ts, .tsx, .js, .jsx
```

Excludes:

- `node_modules/`
- `.venv/`, `venv/`, `.env/`
- `__pycache__/`
- `.git/`
- `dist/`, `build/`, `out/`

### 2. Parsing

Each file is parsed with tree-sitter to produce an AST:

```python
from tree_sitter import Language, Parser
import tree_sitter_python as tspython

language = Language(tspython.language())
parser = Parser(language)
tree = parser.parse(source_code)
```

### 3. Walking the AST

The extractor walks the AST and identifies relevant nodes:

```python
def _walk_tree(node, file_path, source_code, infons):
    if node.type == "import_statement":
        extract_import(node, file_path, source_code, infons)
    elif node.type == "call":
        extract_call(node, file_path, source_code, infons)
    elif node.type == "class_definition":
        extract_class_def(node, file_path, source_code, infons)
    # ... etc.
    
    for child in node.children:
        _walk_tree(child, file_path, source_code, infons)
```

### 4. Creating Infons

For each relevant node, an infon is created:

```python
infon = Infon(
    subject="process_data",
    predicate="calls",
    object="validate_input",
    polarity=True,
    grounding=ASTGrounding(
        file_path="src/services/data.py",
        line_number=15,
        node_type="call_expression"
    ),
    confidence=1.0
)
```

### 5. Insertion

Infons are inserted into the DuckDB store:

```python
store.insert(infons)
```

---

## Extraction Examples

### Python Import

**Code:**

```python
# src/services/user.py, line 3
from flask import Flask, request
```

**Extracted Infons:**

```python
<<imports, "user", "Flask"; +>>
  grounding: src/services/user.py:3 (import_from_statement)
  
<<imports, "user", "request"; +>>
  grounding: src/services/user.py:3 (import_from_statement)
```

---

### Python Function Call

**Code:**

```python
# src/services/data.py, line 15
def process_data(x):
    return validate_input(x)
```

**Extracted Infons:**

```python
<<calls, "process_data", "validate_input"; +>>
  grounding: src/services/data.py:16 (call)
```

---

### Python Class Inheritance

**Code:**

```python
# src/models/user.py, line 5
class UserService(BaseService):
    pass
```

**Extracted Infons:**

```python
<<inherits, "UserService", "BaseService"; +>>
  grounding: src/models/user.py:5 (class_definition)
  
<<defines, "user", "UserService"; +>>
  grounding: src/models/user.py:5 (class_definition)
```

---

### TypeScript Import

**Code:**

```typescript
// src/utils.ts, line 1
import { validateEmail } from './validators';
```

**Extracted Infons:**

```python
<<imports, "utils", "validateEmail"; +>>
  grounding: src/utils.ts:1 (import_statement)
```

---

### TypeScript Method Call

**Code:**

```typescript
// src/services/user.ts, line 10
class UserService {
  createUser(email: string) {
    return this.validateEmail(email);
  }
}
```

**Extracted Infons:**

```python
<<calls, "createUser", "validateEmail"; +>>
  grounding: src/services/user.ts:12 (call_expression)
```

---

## Mapping Table

Comprehensive mapping of AST node types to extracted relations.

### Python

| AST Node Type | Relation | Subject | Predicate | Object |
|---------------|----------|---------|-----------|--------|
| `import_statement` | imports | module | imports | imported module |
| `import_from_statement` | imports | module | imports | imported symbol |
| `call` | calls | caller | calls | callee |
| `class_definition` | defines | module | defines | class name |
| `class_definition` (with bases) | inherits | class name | inherits | base class |
| `function_definition` | defines | module | defines | function name |
| `return_statement` | returns | function | returns | returned value |
| `raise_statement` | raises | function | raises | exception type |
| `decorated_definition` | decorates | decorator | decorates | decorated symbol |
| `assignment` (to attribute) | mutates | object | mutates | attribute |

### TypeScript / JavaScript

| AST Node Type | Relation | Subject | Predicate | Object |
|---------------|----------|---------|-----------|--------|
| `import_statement` | imports | module | imports | imported symbol |
| `call_expression` | calls | caller | calls | callee |
| `class_declaration` | defines | module | defines | class name |
| `class_declaration` (extends) | inherits | class name | inherits | superclass |
| `function_declaration` | defines | module | defines | function name |
| `return_statement` | returns | function | returns | returned value |
| `throw_statement` | raises | function | raises | exception type |
| `decorator` | decorates | decorator | decorates | decorated symbol |
| `assignment_expression` (to property) | mutates | object | mutates | property |

---

## Adding Support for New Languages

To add support for a new language (e.g., Rust, Go, Java):

### 1. Install tree-sitter Grammar

Install the tree-sitter grammar for the language:

```bash
pip install tree-sitter-rust  # or tree-sitter-go, tree-sitter-java, etc.
```

### 2. Create Extractor Class

Create a new file `src/infon/ast/rust_extractor.py`:

```python
from pathlib import Path
import tree_sitter_rust as tsrust
from tree_sitter import Language, Parser
from infon.ast.base import BaseASTExtractor
from infon.infon import Infon
from infon.schema import AnchorSchema


class RustASTExtractor(BaseASTExtractor):
    """Extracts code relations from Rust source files."""
    
    def __init__(self, schema: AnchorSchema):
        super().__init__(schema)
        self.language = Language(tsrust.language())
        self.parser = Parser(self.language)
    
    def extract(self, file_path: Path) -> list[Infon]:
        infons = []
        
        try:
            source_code = file_path.read_bytes()
            tree = self.parser.parse(source_code)
            root_node = tree.root_node
            
            # Walk the tree and extract relations
            self._walk_tree(root_node, file_path, source_code, infons)
        except Exception as e:
            print(f"Warning: Failed to extract from {file_path}: {e}")
        
        return infons
    
    def _walk_tree(self, node, file_path, source_code, infons):
        # Extract use statements (imports)
        if node.type == "use_declaration":
            self._extract_use(node, file_path, source_code, infons)
        # Extract function calls
        elif node.type == "call_expression":
            self._extract_call(node, file_path, source_code, infons)
        # ... etc.
        
        for child in node.children:
            self._walk_tree(child, file_path, source_code, infons)
    
    def _extract_use(self, node, file_path, source_code, infons):
        # Extract use foo::bar;
        line_number = node.start_point[0] + 1
        # Parse the use path and create infon
        # ...
```

### 3. Register Extractor

Add the extractor to `src/infon/ast/registry.py`:

```python
from infon.ast.rust_extractor import RustASTExtractor

class ExtractorRegistry:
    def _register_defaults(self):
        # ... existing registrations ...
        
        # Rust
        self._extractors[".rs"] = RustASTExtractor
```

### 4. Test

Write integration tests in `tests/ast/test_rust_extractor.py`:

```python
from pathlib import Path
from infon.ast.rust_extractor import RustASTExtractor
from infon.schema import AnchorSchema, CODE_RELATION_ANCHORS


def test_rust_use_extraction(tmp_path):
    # Create test file
    rust_file = tmp_path / "test.rs"
    rust_file.write_text("use std::collections::HashMap;")
    
    # Extract
    schema = AnchorSchema(version="0.1.0", language="code", anchors=CODE_RELATION_ANCHORS)
    extractor = RustASTExtractor(schema)
    infons = extractor.extract(rust_file)
    
    # Assert
    assert len(infons) == 1
    assert infons[0].predicate == "imports"
    assert infons[0].object == "HashMap"
```

Run tests:

```bash
pytest tests/ast/test_rust_extractor.py -v
```

### 5. Update Documentation

Add Rust to the supported languages table in this page.

---

## Troubleshooting

### Extractor returns empty list

- Verify tree-sitter grammar is installed: `pip list | grep tree-sitter`
- Check file encoding (tree-sitter expects UTF-8)
- Enable debug logging to see parsing errors

### Missing relations

- Check AST node types: use tree-sitter's inspect tool
- Add new node type handlers to `_walk_tree()`
- Ensure node type names match the grammar's spec

### Duplicate infons

- Consolidation merges duplicates automatically
- Check reinforcement_count field to see how many times a fact was observed

---

## Next Steps

- [Schema Discovery](schema-discovery.md) — how anchors are auto-discovered
- [API Reference](api-reference.md) — Python API for AST extraction
- [Contributing](contributing.md) — add new language support

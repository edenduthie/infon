# Contributing

Thank you for your interest in contributing to `infon`! This page explains how to set up your development environment, run tests, and submit contributions.

---

## Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/edenduthie/infon.git
cd infon
```

### 2. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

### 3. Install Dependencies

Install the package in editable mode with development dependencies:

```bash
pip install -e ".[dev]"
```

This installs:

- Core dependencies (DuckDB, tree-sitter, transformers, etc.)
- Development tools (pytest, ruff, mypy, mkdocs)

### 4. Verify Installation

```bash
# Run tests
pytest -v

# Run linters
ruff check src/
mypy src/infon/

# Check infon CLI
infon --version
```

---

## Development Guidelines

### Test-Driven Development (TDD)

All development **must** follow strict TDD:

1. **Write the test first** — before any implementation code
2. **Verify the test fails (red)** — run the test and confirm it fails
3. **Write minimal implementation** — write only the code needed to make the test pass
4. **Verify the test passes (green)** — run the test and confirm it passes
5. **Refactor if needed** — improve code quality while keeping tests green
6. **Repeat** — iterate for each piece of functionality

**Example Workflow**:

```bash
# 1. Write a failing test
cat > tests/test_new_feature.py << EOF
def test_new_feature():
    result = new_feature()
    assert result == "expected"
EOF

# 2. Run and verify it fails
pytest tests/test_new_feature.py -v
# FAILED (NameError: name 'new_feature' is not defined)

# 3. Implement the feature
cat > src/infon/new_feature.py << EOF
def new_feature():
    return "expected"
EOF

# 4. Run and verify it passes
pytest tests/test_new_feature.py -v
# PASSED
```

---

### No Mocks Policy

This is a **hard rule**. All tests must use **real dependencies**:

- ✅ **Use real DuckDB databases** (temporary files in tests)
- ✅ **Use real file systems** (temporary directories via `tmp_path` fixture)
- ✅ **Use real SPLADE models** (bundled with transformers)
- ✅ **Use real tree-sitter parsers**

- ❌ **No `mock`, `MagicMock`, `stub`, or `patch`**
- ❌ **No isolated unit tests** that test functions in isolation
- ❌ **No fake implementations**

**Why?**

Testing against mocks is testing the mock, not the system. We want to verify that the **entire system works end-to-end**, not that a function behaves correctly against a fake.

**Example (Good Test)**:

```python
def test_store_insert_and_query(tmp_path):
    # Use real DuckDB database
    db_path = tmp_path / "test.ddb"
    store = InfonStore(db_path)
    
    # Insert real infon
    infon = Infon(
        subject="test",
        predicate="calls",
        object="foo",
        polarity=True,
        grounding=Grounding(root=ASTGrounding(...)),
        confidence=1.0,
        # ... full infon
    )
    store.insert([infon])
    
    # Query and verify
    results = store.query(subject="test")
    assert len(results) == 1
    assert results[0].object == "foo"
    
    store.close()
```

**Example (Bad Test)**:

```python
# ❌ DO NOT DO THIS
from unittest.mock import MagicMock

def test_store_insert_mock():
    # Mock the database connection
    mock_conn = MagicMock()
    store = InfonStore(mock_conn)
    
    # ... test against mock ...
```

---

### Code Quality

Before submitting a PR, ensure all quality checks pass:

```bash
# Run tests
pytest tests/ -v

# Check code style
ruff check src/

# Format code
ruff format src/

# Type checking
mypy src/infon/

# Test coverage (optional)
pytest tests/ --cov=src/infon --cov-report=html
```

**Minimum Requirements**:

- All tests pass
- No linter errors
- No type errors
- Test coverage ≥ 80% for new code

---

## Testing

### Running Tests

```bash
# Run all tests
pytest -v

# Run specific test file
pytest tests/test_store.py -v

# Run specific test function
pytest tests/test_store.py::test_insert -v

# Run with verbose output
pytest -vv

# Run with coverage
pytest --cov=src/infon --cov-report=html
```

### Test Structure

Tests are organized by module:

```
tests/
├── test_infon.py          # Infon data model tests
├── test_grounding.py      # Grounding tests
├── test_schema.py         # AnchorSchema tests
├── test_store.py          # InfonStore tests
├── test_encoder.py        # SPLADE encoder tests
├── test_extract.py        # Text extraction tests
├── test_consolidate.py    # Consolidation tests
├── test_retrieve.py       # Retrieval tests
├── test_discovery.py      # Schema discovery tests
├── ast/
│   ├── test_python_extractor.py
│   ├── test_typescript_extractor.py
│   └── test_registry.py
└── mcp/
    └── test_server.py
```

### Writing Tests

Follow these guidelines when writing tests:

**1. Use fixtures for setup and teardown**:

```python
import pytest
from pathlib import Path

@pytest.fixture
def temp_db(tmp_path):
    db_path = tmp_path / "test.ddb"
    store = InfonStore(db_path)
    yield store
    store.close()

def test_insert(temp_db):
    temp_db.insert([infon])
    # ...
```

**2. Provision real resources**:

```python
def test_extract_python(tmp_path):
    # Create real Python file
    py_file = tmp_path / "test.py"
    py_file.write_text("def foo(): pass")
    
    # Extract with real tree-sitter parser
    extractor = PythonASTExtractor(schema)
    infons = extractor.extract(py_file)
    
    # Verify
    assert len(infons) > 0
```

**3. Test end-to-end flows**:

```python
def test_full_ingestion_workflow(tmp_path):
    # Create repository
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text("import numpy")
    
    # Initialize store
    db_path = tmp_path / "kb.ddb"
    schema = AnchorSchema(version="1.0", language="code", anchors=CODE_RELATION_ANCHORS)
    
    with InfonStore(db_path) as store:
        # Ingest repository
        infons = ingest_repo(repo, store, schema)
        
        # Verify extraction
        assert len(infons) > 0
        
        # Verify storage
        results = store.query(predicate="imports")
        assert len(results) > 0
        assert results[0].object == "numpy"
```

---

## Adding New Features

### Adding a New AST Extractor

To add support for a new language (e.g., Rust):

**1. Install tree-sitter grammar**:

```bash
pip install tree-sitter-rust
```

**2. Create extractor class**:

```python
# src/infon/ast/rust_extractor.py
from infon.ast.base import BaseASTExtractor
import tree_sitter_rust as tsrust

class RustASTExtractor(BaseASTExtractor):
    def __init__(self, schema):
        super().__init__(schema)
        self.language = Language(tsrust.language())
        self.parser = Parser(self.language)
    
    def extract(self, file_path):
        # Implementation
        pass
```

**3. Register extractor**:

```python
# src/infon/ast/registry.py
from infon.ast.rust_extractor import RustASTExtractor

class ExtractorRegistry:
    def _register_defaults(self):
        # ... existing ...
        self._extractors[".rs"] = RustASTExtractor
```

**4. Write tests** (TDD):

```python
# tests/ast/test_rust_extractor.py
def test_rust_use_statement(tmp_path):
    rust_file = tmp_path / "test.rs"
    rust_file.write_text("use std::collections::HashMap;")
    
    schema = AnchorSchema(version="1.0", language="code", anchors=CODE_RELATION_ANCHORS)
    extractor = RustASTExtractor(schema)
    infons = extractor.extract(rust_file)
    
    assert len(infons) == 1
    assert infons[0].predicate == "imports"
    assert infons[0].object == "HashMap"
```

**5. Update documentation**:

Add Rust to the supported languages table in `docs/ast-extraction.md`.

---

### Adding a New CLI Command

**1. Write test first**:

```python
# tests/test_cli.py
from click.testing import CliRunner
from infon.cli import cli

def test_new_command():
    runner = CliRunner()
    result = runner.invoke(cli, ['new-command', '--option', 'value'])
    assert result.exit_code == 0
    assert "expected output" in result.output
```

**2. Implement command**:

```python
# src/infon/cli.py
@cli.command()
@click.option('--option', help='Description')
def new_command(option):
    """Command description."""
    click.echo(f"Running new command with {option}")
```

**3. Update documentation**:

Add command to `docs/cli.md`.

---

## Pull Request Process

### 1. Create a Branch

```bash
git checkout -b feature/my-new-feature
```

### 2. Commit Changes

Follow conventional commit format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types**:

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Build process or tooling changes

**Examples**:

```bash
git commit -m "feat(ast): add Rust AST extractor"
git commit -m "fix(store): handle concurrent writes correctly"
git commit -m "docs(cli): document new command"
```

### 3. Run Quality Checks

```bash
pytest -v
ruff check src/
ruff format src/
mypy src/infon/
```

### 4. Update CHANGELOG.md

Add your changes to the `[Unreleased]` section:

```markdown
## [Unreleased]

### Added
- Rust AST extractor support

### Fixed
- Concurrent write handling in InfonStore
```

### 5. Push and Create PR

```bash
git push origin feature/my-new-feature
```

Open a pull request on GitHub with:

- Clear description of changes
- Reference to related issues (if any)
- Checklist confirming:
  - [ ] All tests pass
  - [ ] Code follows TDD approach
  - [ ] No mocks used
  - [ ] Documentation updated
  - [ ] CHANGELOG.md updated

### 6. Request Review

Tag maintainers for review. Address any feedback and update your branch.

---

## Code Style

### Python Style

- Follow PEP 8 (enforced by ruff)
- Line length: 100 characters
- Use type hints for all function signatures
- Use docstrings for all public functions and classes

**Example**:

```python
def extract_text(text: str, doc_id: str, schema: AnchorSchema) -> list[Infon]:
    """Extract infons from natural language text.
    
    Args:
        text: The text to extract from
        doc_id: Document identifier for grounding
        schema: AnchorSchema for anchor vocabulary
        
    Returns:
        List of extracted Infons with TextGrounding
    """
    # Implementation
    pass
```

### Import Order

1. Standard library
2. Third-party packages
3. Local imports

Enforced by ruff's `I` rule:

```python
import json
from pathlib import Path

import duckdb
from pydantic import BaseModel

from infon.infon import Infon
from infon.schema import AnchorSchema
```

---

## Documentation

### Writing Documentation

Documentation is written in Markdown and built with MkDocs Material.

**Local preview**:

```bash
mkdocs serve
# Open http://127.0.0.1:8000
```

**Build docs**:

```bash
mkdocs build --strict
```

**Add new page**:

1. Create Markdown file in `docs/`
2. Add to `mkdocs.yml` navigation
3. Preview locally
4. Commit

---

## Questions?

- Open an issue on GitHub for bugs or feature requests
- Join the discussion in GitHub Discussions
- Email the maintainers: eduthie@gmail.com

---

## Code of Conduct

Be respectful, constructive, and collaborative. We welcome contributions from developers of all experience levels.

---

## License

By contributing to infon, you agree that your contributions will be licensed under the Apache 2.0 License.

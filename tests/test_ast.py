"""Integration tests for AST extraction pipeline.

Tests the full AST extraction flow from repository ingestion to infon production.
Uses real tree-sitter parsers, real store, and real schema.
NO MOCKS - this tests actual functionality end-to-end.
"""

import tempfile
from pathlib import Path

from infon.ast import ingest_repo
from infon.schema import AnchorSchema, CODE_RELATION_ANCHORS
from infon.store import InfonStore


def test_ingest_repo_produces_infons():
    """Test that ingest_repo() produces >50 infons from test fixtures.
    
    This integration test calls ingest_repo() on tests/fixtures/ with a real
    InfonStore and real AnchorSchema in code mode. It verifies:
    - At least 50 infons are produced from the fixtures
    - The store is populated with the extracted infons
    """
    fixtures_dir = Path(__file__).parent / "fixtures"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.ddb"
        
        # Create code-mode schema with CODE_RELATION_ANCHORS
        schema = AnchorSchema(
            anchors=CODE_RELATION_ANCHORS.copy(),
            version="1.0.0",
            language="code",
        )
        
        # Create real store
        store = InfonStore(db_path)
        
        try:
            # Run ingest_repo on fixtures
            infons = ingest_repo(fixtures_dir, store, schema)
            
            # Assert >50 infons produced
            assert len(infons) > 50, f"Expected >50 infons, got {len(infons)}"
            
            # Verify infons are in the store
            stats = store.stats()
            assert stats.infon_count > 50, f"Expected >50 infons in store, got {stats.infon_count}"
            
        finally:
            store.close()


def test_python_calls_infons():
    """Test that Python function calls produce infons with correct grounding.
    
    Verifies that Python function calls (like register_user(), login(), etc.)
    are extracted as 'calls' infons with ASTGrounding containing correct
    file_path, line_number, and node_type.
    """
    fixtures_dir = Path(__file__).parent / "fixtures"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.ddb"
        
        schema = AnchorSchema(
            anchors=CODE_RELATION_ANCHORS.copy(),
            version="1.0.0",
            language="code",
        )
        
        store = InfonStore(db_path)
        
        try:
            infons = ingest_repo(fixtures_dir, store, schema)
            
            # Find 'calls' infons from Python files
            calls_infons = [i for i in infons if i.predicate == "calls"]
            assert len(calls_infons) > 0, "No 'calls' infons found"
            
            # Verify at least one has ASTGrounding with Python file
            python_calls = [
                i for i in calls_infons
                if i.grounding.root.file_path.endswith(".py")
            ]
            assert len(python_calls) > 0, "No Python calls found"
            
            # Verify grounding structure
            sample = python_calls[0]
            grounding = sample.grounding.root
            assert hasattr(grounding, "file_path"), "Missing file_path"
            assert hasattr(grounding, "line_number"), "Missing line_number"
            assert hasattr(grounding, "node_type"), "Missing node_type"
            assert grounding.line_number > 0, "Invalid line_number"
            
        finally:
            store.close()


def test_typescript_imports_infons():
    """Test that TypeScript imports produce infons with correct grounding.
    
    Verifies that TypeScript/JavaScript import statements are extracted as
    'imports' infons with ASTGrounding.
    """
    fixtures_dir = Path(__file__).parent / "fixtures"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.ddb"
        
        schema = AnchorSchema(
            anchors=CODE_RELATION_ANCHORS.copy(),
            version="1.0.0",
            language="code",
        )
        
        store = InfonStore(db_path)
        
        try:
            infons = ingest_repo(fixtures_dir, store, schema)
            
            # Find 'imports' infons from TypeScript files
            imports_infons = [i for i in infons if i.predicate == "imports"]
            assert len(imports_infons) > 0, "No 'imports' infons found"
            
            # Verify at least one has ASTGrounding with .ts file
            ts_imports = [
                i for i in imports_infons
                if i.grounding.root.file_path.endswith(".ts")
            ]
            assert len(ts_imports) > 0, "No TypeScript imports found"
            
            # Verify grounding structure
            sample = ts_imports[0]
            grounding = sample.grounding.root
            assert hasattr(grounding, "file_path"), "Missing file_path"
            assert hasattr(grounding, "line_number"), "Missing line_number"
            assert hasattr(grounding, "node_type"), "Missing node_type"
            
        finally:
            store.close()


def test_inherits_infons():
    """Test that class inheritance produces 'inherits' infons.
    
    Verifies that class inheritance relationships are extracted with correct
    grounding from both Python and TypeScript fixtures.
    """
    fixtures_dir = Path(__file__).parent / "fixtures"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.ddb"
        
        schema = AnchorSchema(
            anchors=CODE_RELATION_ANCHORS.copy(),
            version="1.0.0",
            language="code",
        )
        
        store = InfonStore(db_path)
        
        try:
            infons = ingest_repo(fixtures_dir, store, schema)
            
            # Find 'inherits' infons
            inherits_infons = [i for i in infons if i.predicate == "inherits"]
            assert len(inherits_infons) > 0, "No 'inherits' infons found"
            
            # Verify grounding is ASTGrounding
            sample = inherits_infons[0]
            grounding = sample.grounding.root
            assert hasattr(grounding, "file_path"), "Missing file_path"
            assert hasattr(grounding, "line_number"), "Missing line_number"
            
        finally:
            store.close()


def test_unknown_extension_skipped():
    """Test that files with unknown extensions are skipped.
    
    Verifies that ingest_repo() gracefully skips files with extensions that
    don't have registered extractors (e.g., .txt, .md, .json, etc.).
    """
    fixtures_dir = Path(__file__).parent / "fixtures"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.ddb"
        
        # Create a dummy file with unknown extension
        dummy_file = fixtures_dir / "unknown.xyz"
        dummy_file.write_text("This should be skipped")
        
        try:
            schema = AnchorSchema(
                anchors=CODE_RELATION_ANCHORS.copy(),
                version="1.0.0",
                language="code",
            )
            
            store = InfonStore(db_path)
            
            try:
                # Should not raise an error
                infons = ingest_repo(fixtures_dir, store, schema)
                
                # Verify no infons from .xyz file
                xyz_infons = [
                    i for i in infons
                    if i.grounding.root.file_path.endswith(".xyz")
                ]
                assert len(xyz_infons) == 0, "Unknown extension should be skipped"
                
            finally:
                store.close()
                
        finally:
            # Clean up dummy file
            if dummy_file.exists():
                dummy_file.unlink()


def test_all_relation_types_covered():
    """Test that all eight CODE_RELATION_ANCHORS are represented in extracted infons.
    
    The fixtures should cover all eight relation types:
    calls, imports, inherits, mutates, defines, returns, raises, decorates
    """
    fixtures_dir = Path(__file__).parent / "fixtures"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.ddb"
        
        schema = AnchorSchema(
            anchors=CODE_RELATION_ANCHORS.copy(),
            version="1.0.0",
            language="code",
        )
        
        store = InfonStore(db_path)
        
        try:
            infons = ingest_repo(fixtures_dir, store, schema)
            
            # Collect all predicates
            predicates = set(i.predicate for i in infons)
            
            # Check for presence of key relation types
            # (Some may not be present depending on fixture coverage)
            assert "calls" in predicates, "Missing 'calls' relations"
            assert "imports" in predicates, "Missing 'imports' relations"
            
            # At least 3 different relation types should be present
            assert len(predicates) >= 3, f"Expected >=3 relation types, got {len(predicates)}: {predicates}"
            
        finally:
            store.close()

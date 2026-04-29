"""
Integration tests for DuckDB InfonStore.

This module tests the InfonStore following strict TDD:
- All four tables (infons, edges, constraints, documents) with indexes
- upsert() with reinforcement merge logic
- get(), query(), add_edge(), get_edges(), upsert_constraint(), upsert_document(), stats()
- Concurrent write detection (ConcurrentWriteError)
- Context manager protocol (__enter__, __exit__)

No mocks, no stubs - real DuckDB database in temp directory with real Infon instances.
"""

import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

from infon.grounding import ASTGrounding, Grounding, TextGrounding
from infon.infon import ImportanceScore, Infon


def test_store_creates_all_tables_with_indexes():
    """Test that InfonStore creates all four tables with correct indexes on initialization."""
    from infon.store import InfonStore
    
    # Create a temporary directory for the DuckDB file
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.ddb"
        
        # Open the store - should create all tables
        store = InfonStore(db_path)
        
        # Verify all four tables exist
        conn = store._conn
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = {row[0] for row in tables}
        
        assert "infons" in table_names
        assert "edges" in table_names
        assert "constraints" in table_names
        assert "documents" in table_names
        
        # Verify infons table has the correct indexes
        # In DuckDB, we can verify indexes exist by querying information_schema
        indexes = conn.execute("""
            SELECT index_name 
            FROM duckdb_indexes() 
            WHERE table_name = 'infons'
        """).fetchall()
        index_names = [row[0] for row in indexes]
        
        # Should have indexes on: (subject, predicate, object), subject, predicate, doc_id, timestamp, kind
        assert any('subject' in name for name in index_names)
        assert any('predicate' in name for name in index_names)
        assert any('doc_id' in name for name in index_names)
        assert any('timestamp' in name for name in index_names)
        assert any('kind' in name for name in index_names)
        
        # Verify edges table has index on (from_infon_id, edge_type)
        indexes = conn.execute("""
            SELECT index_name 
            FROM duckdb_indexes() 
            WHERE table_name = 'edges'
        """).fetchall()
        index_names = [row[0] for row in indexes]
        
        assert any('from' in name for name in index_names)
        
        # Verify constraints table has index on (subject, predicate, object)
        indexes = conn.execute("""
            SELECT index_name 
            FROM duckdb_indexes() 
            WHERE table_name = 'constraints'
        """).fetchall()
        index_names = [row[0] for row in indexes]
        
        assert any('triple' in name or 'subject' in name for name in index_names)
        
        # Verify documents table has index on path
        indexes = conn.execute("""
            SELECT index_name 
            FROM duckdb_indexes() 
            WHERE table_name = 'documents'
        """).fetchall()
        index_names = [row[0] for row in indexes]
        
        assert any('path' in name for name in index_names)
        
        store.close()


def test_upsert_and_reinforcement_merge():
    """Test that upsert merges infons with same triple+polarity and increments reinforcement_count."""
    from infon.store import InfonStore
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.ddb"
        store = InfonStore(db_path)
        
        # Create first infon
        infon1 = Infon(
            id=str(uuid.uuid4()),
            subject="user_service",
            predicate="calls",
            object="database_pool",
            polarity=True,
            grounding=Grounding(root=TextGrounding(
                doc_id="doc1",
                sent_id=0,
                char_start=10,
                char_end=25,
                sentence_text="UserService calls the database pool."
            )),
            confidence=0.8,
            timestamp=datetime.now(UTC),
            importance=ImportanceScore(
                activation=0.8,
                coherence=0.7,
                specificity=0.6,
                novelty=0.5,
                reinforcement=0.4
            ),
            kind="extracted",
            reinforcement_count=1
        )
        
        # Upsert first infon
        result1 = store.upsert(infon1)
        assert result1.reinforcement_count == 1
        
        # Create second infon with same triple but different doc_id and confidence
        infon2 = Infon(
            id=str(uuid.uuid4()),
            subject="user_service",
            predicate="calls",
            object="database_pool",
            polarity=True,
            grounding=Grounding(root=TextGrounding(
                doc_id="doc2",
                sent_id=1,
                char_start=5,
                char_end=20,
                sentence_text="The user service calls database pool."
            )),
            confidence=0.9,
            timestamp=datetime.now(UTC),
            importance=ImportanceScore(
                activation=0.8,
                coherence=0.7,
                specificity=0.6,
                novelty=0.5,
                reinforcement=0.4
            ),
            kind="extracted",
            reinforcement_count=1
        )
        
        # Upsert second infon - should merge with first
        result2 = store.upsert(infon2)
        
        # Verify reinforcement count incremented
        assert result2.reinforcement_count == 2
        
        # Verify confidence is averaged
        expected_confidence = (0.8 + 0.9) / 2
        assert abs(result2.confidence - expected_confidence) < 0.01
        
        # Verify only one row exists in database for this triple
        rows = store.query(subject="user_service", predicate="calls", object="database_pool")
        assert len(rows) == 1
        assert rows[0].reinforcement_count == 2
        
        store.close()


def test_get_by_id():
    """Test that get() retrieves an infon by UUID."""
    from infon.store import InfonStore
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.ddb"
        store = InfonStore(db_path)
        
        # Create and upsert an infon
        infon_id = str(uuid.uuid4())
        infon = Infon(
            id=infon_id,
            subject="user_service",
            predicate="calls",
            object="auth_service",
            polarity=True,
            grounding=Grounding(root=ASTGrounding(
                file_path="src/main.py",
                line_number=42,
                node_type="FunctionCall"
            )),
            confidence=0.95,
            timestamp=datetime.now(UTC),
            importance=ImportanceScore(
                activation=0.8,
                coherence=0.7,
                specificity=0.6,
                novelty=0.5,
                reinforcement=0.4
            ),
            kind="extracted",
            reinforcement_count=1
        )
        
        store.upsert(infon)
        
        # Retrieve by ID
        retrieved = store.get(infon_id)
        
        assert retrieved is not None
        assert retrieved.id == infon_id
        assert retrieved.subject == "user_service"
        assert retrieved.predicate == "calls"
        assert retrieved.object == "auth_service"
        
        # Test get with non-existent ID returns None
        non_existent = store.get(str(uuid.uuid4()))
        assert non_existent is None
        
        store.close()


def test_query_with_filters():
    """Test that query() filters infons by subject/predicate/object and min_confidence."""
    from infon.store import InfonStore
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.ddb"
        store = InfonStore(db_path)
        
        # Create multiple infons with different triples and confidences
        infons = [
            Infon(
                id=str(uuid.uuid4()),
                subject="user_service",
                predicate="calls",
                object="database_pool",
                polarity=True,
                grounding=Grounding(root=ASTGrounding(
                    file_path="src/user.py",
                    line_number=10,
                    node_type="FunctionCall"
                )),
                confidence=0.9,
                timestamp=datetime.now(UTC),
                importance=ImportanceScore(
                    activation=0.8, coherence=0.7, specificity=0.6,
                    novelty=0.5, reinforcement=0.4
                ),
                kind="extracted",
                reinforcement_count=1
            ),
            Infon(
                id=str(uuid.uuid4()),
                subject="user_service",
                predicate="calls",
                object="auth_service",
                polarity=True,
                grounding=Grounding(root=ASTGrounding(
                    file_path="src/user.py",
                    line_number=20,
                    node_type="FunctionCall"
                )),
                confidence=0.7,
                timestamp=datetime.now(UTC),
                importance=ImportanceScore(
                    activation=0.8, coherence=0.7, specificity=0.6,
                    novelty=0.5, reinforcement=0.4
                ),
                kind="extracted",
                reinforcement_count=1
            ),
            Infon(
                id=str(uuid.uuid4()),
                subject="auth_service",
                predicate="calls",
                object="database_pool",
                polarity=True,
                grounding=Grounding(root=ASTGrounding(
                    file_path="src/auth.py",
                    line_number=30,
                    node_type="FunctionCall"
                )),
                confidence=0.85,
                timestamp=datetime.now(UTC),
                importance=ImportanceScore(
                    activation=0.8, coherence=0.7, specificity=0.6,
                    novelty=0.5, reinforcement=0.4
                ),
                kind="extracted",
                reinforcement_count=1
            ),
        ]
        
        for infon in infons:
            store.upsert(infon)
        
        # Query by subject only
        results = store.query(subject="user_service")
        assert len(results) == 2
        assert all(r.subject == "user_service" for r in results)
        
        # Query by predicate only
        results = store.query(predicate="calls")
        assert len(results) == 3
        
        # Query by object only
        results = store.query(object="database_pool")
        assert len(results) == 2
        assert all(r.object == "database_pool" for r in results)
        
        # Query by subject and object
        results = store.query(subject="user_service", object="auth_service")
        assert len(results) == 1
        assert results[0].subject == "user_service"
        assert results[0].object == "auth_service"
        
        # Query with min_confidence filter
        results = store.query(min_confidence=0.85)
        assert len(results) == 2
        assert all(r.confidence >= 0.85 for r in results)
        
        # Query with limit
        results = store.query(limit=2)
        assert len(results) == 2
        
        store.close()


def test_add_edge_and_get_edges():
    """Test that add_edge() creates edges and get_edges() retrieves them."""
    from infon.store import InfonStore
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.ddb"
        store = InfonStore(db_path)
        
        # Create two infons
        infon1_id = str(uuid.uuid4())
        infon1 = Infon(
            id=infon1_id,
            subject="user_service",
            predicate="calls",
            object="database_pool",
            polarity=True,
            grounding=Grounding(root=ASTGrounding(
                file_path="src/user.py",
                line_number=10,
                node_type="FunctionCall"
            )),
            confidence=0.9,
            timestamp=datetime.now(UTC),
            importance=ImportanceScore(
                activation=0.8, coherence=0.7, specificity=0.6,
                novelty=0.5, reinforcement=0.4
            ),
            kind="extracted",
            reinforcement_count=1
        )
        
        infon2_id = str(uuid.uuid4())
        infon2 = Infon(
            id=infon2_id,
            subject="auth_service",
            predicate="calls",
            object="database_pool",
            polarity=True,
            grounding=Grounding(root=ASTGrounding(
                file_path="src/auth.py",
                line_number=20,
                node_type="FunctionCall"
            )),
            confidence=0.85,
            timestamp=datetime.now(UTC),
            importance=ImportanceScore(
                activation=0.8, coherence=0.7, specificity=0.6,
                novelty=0.5, reinforcement=0.4
            ),
            kind="extracted",
            reinforcement_count=1
        )
        
        store.upsert(infon1)
        store.upsert(infon2)
        
        # Add edges between infons
        store.add_edge(infon1_id, infon2_id, edge_type="NEXT", weight=0.9)
        store.add_edge(infon1_id, infon2_id, edge_type="SIMILAR", weight=0.7)
        
        # Get all edges from infon1
        edges = store.get_edges(infon1_id)
        assert len(edges) == 2
        
        # Get edges filtered by type
        next_edges = store.get_edges(infon1_id, edge_type="NEXT")
        assert len(next_edges) == 1
        assert next_edges[0]["edge_type"] == "NEXT"
        assert next_edges[0]["to_infon_id"] == infon2_id
        assert abs(next_edges[0]["weight"] - 0.9) < 0.01
        
        similar_edges = store.get_edges(infon1_id, edge_type="SIMILAR")
        assert len(similar_edges) == 1
        assert similar_edges[0]["edge_type"] == "SIMILAR"
        
        store.close()


def test_upsert_constraint():
    """Test that upsert_constraint() creates or updates constraints."""
    from infon.store import InfonStore
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.ddb"
        store = InfonStore(db_path)
        
        # Upsert a constraint
        store.upsert_constraint(
            subject="user_service",
            predicate="calls",
            object="database_pool",
            evidence_count=5,
            strength=0.85,
            persistence=0.9
        )
        
        # Verify constraint was created
        conn = store._conn
        rows = conn.execute(
            "SELECT * FROM constraints WHERE subject = ? AND predicate = ? AND object = ?",
            ["user_service", "calls", "database_pool"]
        ).fetchall()
        
        assert len(rows) == 1
        assert rows[0][1] == "user_service"  # subject
        assert rows[0][2] == "calls"  # predicate
        assert rows[0][3] == "database_pool"  # object
        assert rows[0][4] == 5  # evidence_count
        assert abs(rows[0][5] - 0.85) < 0.01  # strength
        assert abs(rows[0][6] - 0.9) < 0.01  # persistence
        
        # Upsert again with updated values
        store.upsert_constraint(
            subject="user_service",
            predicate="calls",
            object="database_pool",
            evidence_count=10,
            strength=0.95,
            persistence=0.95
        )
        
        # Verify constraint was updated, not duplicated
        rows = conn.execute(
            "SELECT * FROM constraints WHERE subject = ? AND predicate = ? AND object = ?",
            ["user_service", "calls", "database_pool"]
        ).fetchall()
        
        assert len(rows) == 1
        assert rows[0][4] == 10  # evidence_count updated
        assert abs(rows[0][5] - 0.95) < 0.01  # strength updated
        
        store.close()


def test_upsert_document():
    """Test that upsert_document() creates or updates document records."""
    from infon.store import InfonStore
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.ddb"
        store = InfonStore(db_path)
        
        # Upsert a document
        store.upsert_document(
            doc_id="doc1",
            path="src/user.py",
            kind="code",
            token_count=250
        )
        
        # Verify document was created
        conn = store._conn
        rows = conn.execute(
            "SELECT * FROM documents WHERE id = ?",
            ["doc1"]
        ).fetchall()
        
        assert len(rows) == 1
        assert rows[0][0] == "doc1"
        assert rows[0][1] == "src/user.py"
        assert rows[0][2] == "code"
        assert rows[0][4] == 250  # token_count
        
        # Upsert again with updated values
        store.upsert_document(
            doc_id="doc1",
            path="src/user.py",
            kind="code",
            token_count=300
        )
        
        # Verify document was updated
        rows = conn.execute(
            "SELECT * FROM documents WHERE id = ?",
            ["doc1"]
        ).fetchall()
        
        assert len(rows) == 1
        assert rows[0][4] == 300  # token_count updated
        
        store.close()


def test_stats():
    """Test that stats() returns correct counts and top anchors."""
    from infon.store import InfonStore
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.ddb"
        store = InfonStore(db_path)
        
        # Create multiple infons
        for i in range(10):
            infon = Infon(
                id=str(uuid.uuid4()),
                subject="user_service",
                predicate="calls",
                object=f"service_{i}",
                polarity=True,
                grounding=Grounding(root=ASTGrounding(
                    file_path=f"src/file_{i}.py",
                    line_number=10,
                    node_type="FunctionCall"
                )),
                confidence=0.9,
                timestamp=datetime.now(UTC),
                importance=ImportanceScore(
                    activation=0.8, coherence=0.7, specificity=0.6,
                    novelty=0.5, reinforcement=0.4
                ),
                kind="extracted",
                reinforcement_count=1
            )
            store.upsert(infon)
        
        # Add some edges
        infons_list = store.query(limit=5)
        if len(infons_list) >= 2:
            store.add_edge(infons_list[0].id, infons_list[1].id, "NEXT", 0.9)
            store.add_edge(infons_list[1].id, infons_list[2].id, "NEXT", 0.8)
        
        # Add constraints
        store.upsert_constraint("user_service", "calls", "database_pool", 5, 0.9, 0.95)
        
        # Add documents
        store.upsert_document("doc1", "src/main.py", "code", 100)
        store.upsert_document("doc2", "src/auth.py", "code", 150)
        
        # Get stats
        stats = store.stats()
        
        assert stats["infon_count"] == 10
        assert stats["edge_count"] == 2
        assert stats["constraint_count"] == 1
        assert stats["document_count"] == 2
        assert "top_anchors" in stats
        assert len(stats["top_anchors"]) > 0
        
        store.close()


def test_concurrent_write_detection():
    """Test that opening a second writer raises ConcurrentWriteError."""
    from infon.store import ConcurrentWriteError, InfonStore
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.ddb"
        
        # Open first writer
        store1 = InfonStore(db_path)
        
        # Attempt to open second writer - should raise
        with pytest.raises(ConcurrentWriteError):
            store2 = InfonStore(db_path)
        
        # Close first writer
        store1.close()
        
        # Now second writer should succeed
        store2 = InfonStore(db_path)
        store2.close()


def test_context_manager():
    """Test that InfonStore works as a context manager with automatic close."""
    from infon.store import InfonStore
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.ddb"
        
        # Use as context manager
        with InfonStore(db_path) as store:
            # Create an infon
            infon = Infon(
                id=str(uuid.uuid4()),
                subject="user_service",
                predicate="calls",
                object="database_pool",
                polarity=True,
                grounding=Grounding(root=ASTGrounding(
                    file_path="src/user.py",
                    line_number=10,
                    node_type="FunctionCall"
                )),
                confidence=0.9,
                timestamp=datetime.now(UTC),
                importance=ImportanceScore(
                    activation=0.8, coherence=0.7, specificity=0.6,
                    novelty=0.5, reinforcement=0.4
                ),
                kind="extracted",
                reinforcement_count=1
            )
            
            store.upsert(infon)
            
            # Verify it was stored
            results = store.query(subject="user_service")
            assert len(results) == 1
        
        # Context manager should have closed the connection
        # Opening a new store should work (no concurrent write error)
        with InfonStore(db_path) as store2:
            # Verify data persisted
            results = store2.query(subject="user_service")
            assert len(results) == 1

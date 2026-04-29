"""
Integration tests for consolidation pipeline.

This module tests the consolidate() function following strict TDD:
- NEXT edges created between chronologically ordered infons sharing anchors
- Idempotency (calling consolidate() twice produces same result)
- Constraint aggregation with evidence_count, strength, persistence
- Importance decay for old infons (>7 days)

No mocks, no stubs - real DuckDB database in temp directory with real Infon instances.
"""

import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from infon.grounding import ASTGrounding, Grounding, TextGrounding
from infon.infon import Infon, ImportanceScore
from infon.schema import AnchorSchema
from infon.store import InfonStore


def test_consolidate_creates_next_edges():
    """Test that consolidate() creates NEXT edges between chronologically ordered infons."""
    from infon.consolidate import consolidate

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.ddb"
        store = InfonStore(db_path)

        # Create three infons sharing the same subject anchor "user_service"
        # with timestamps T1 < T2 < T3
        now = datetime.now(timezone.utc)
        t1 = now - timedelta(hours=2)
        t2 = now - timedelta(hours=1)
        t3 = now

        infon1 = Infon(
            id=str(uuid.uuid4()),
            subject="user_service",
            predicate="calls",
            object="auth_service",
            polarity=True,
            grounding=Grounding(
                root=ASTGrounding(
                    file_path="main.py",
                    line_number=10,
                    node_type="call_expression",
                )
            ),
            confidence=0.9,
            timestamp=t1,
            importance=ImportanceScore(
                activation=0.8,
                coherence=0.7,
                specificity=0.6,
                novelty=0.5,
                reinforcement=0.4,
            ),
            kind="extracted",
            reinforcement_count=1,
        )

        infon2 = Infon(
            id=str(uuid.uuid4()),
            subject="user_service",
            predicate="imports",
            object="database",
            polarity=True,
            grounding=Grounding(
                root=ASTGrounding(
                    file_path="main.py",
                    line_number=5,
                    node_type="import_statement",
                )
            ),
            confidence=0.85,
            timestamp=t2,
            importance=ImportanceScore(
                activation=0.75,
                coherence=0.65,
                specificity=0.55,
                novelty=0.45,
                reinforcement=0.35,
            ),
            kind="extracted",
            reinforcement_count=1,
        )

        infon3 = Infon(
            id=str(uuid.uuid4()),
            subject="user_service",
            predicate="implements",
            object="user_interface",
            polarity=True,
            grounding=Grounding(
                root=ASTGrounding(
                    file_path="main.py",
                    line_number=20,
                    node_type="class_definition",
                )
            ),
            confidence=0.95,
            timestamp=t3,
            importance=ImportanceScore(
                activation=0.9,
                coherence=0.8,
                specificity=0.7,
                novelty=0.6,
                reinforcement=0.5,
            ),
            kind="extracted",
            reinforcement_count=1,
        )

        # Insert all three infons
        store.upsert(infon1)
        store.upsert(infon2)
        store.upsert(infon3)

        # Create a minimal schema (needed for consolidate signature)
        schema = AnchorSchema(anchors={}, version="1.0.0", language="code")

        # Call consolidate
        consolidate(store, schema)

        # Verify NEXT edges: T1→T2 and T2→T3 for the "user_service" anchor
        edges_from_infon1 = store.get_edges(infon1.id, edge_type="NEXT")
        assert len(edges_from_infon1) == 1
        assert edges_from_infon1[0]["to_infon_id"] == infon2.id
        assert edges_from_infon1[0]["edge_type"] == "NEXT"
        # Weight should be 1 / (1 + days_between)
        # T1 to T2 is 1 hour = ~0.042 days
        # Weight ≈ 1 / (1 + 0.042) ≈ 0.96
        assert edges_from_infon1[0]["weight"] > 0.9

        edges_from_infon2 = store.get_edges(infon2.id, edge_type="NEXT")
        assert len(edges_from_infon2) == 1
        assert edges_from_infon2[0]["to_infon_id"] == infon3.id
        assert edges_from_infon2[0]["edge_type"] == "NEXT"

        # Verify no direct T1→T3 edge
        all_edges_from_infon1 = store.get_edges(infon1.id)
        assert all(edge["to_infon_id"] != infon3.id for edge in all_edges_from_infon1)

        # Verify infon3 has no outgoing NEXT edges
        edges_from_infon3 = store.get_edges(infon3.id, edge_type="NEXT")
        assert len(edges_from_infon3) == 0

        store.close()


def test_consolidate_is_idempotent():
    """Test that calling consolidate() twice produces the same result."""
    from infon.consolidate import consolidate

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.ddb"
        store = InfonStore(db_path)

        # Create two infons sharing an anchor
        now = datetime.now(timezone.utc)

        infon1 = Infon(
            id=str(uuid.uuid4()),
            subject="api_service",
            predicate="calls",
            object="database",
            polarity=True,
            grounding=Grounding(
                root=ASTGrounding(
                    file_path="api.py", line_number=15, node_type="call_expression"
                )
            ),
            confidence=0.9,
            timestamp=now - timedelta(hours=1),
            importance=ImportanceScore(
                activation=0.8,
                coherence=0.7,
                specificity=0.6,
                novelty=0.5,
                reinforcement=0.4,
            ),
            kind="extracted",
            reinforcement_count=1,
        )

        infon2 = Infon(
            id=str(uuid.uuid4()),
            subject="api_service",
            predicate="imports",
            object="config",
            polarity=True,
            grounding=Grounding(
                root=ASTGrounding(
                    file_path="api.py", line_number=1, node_type="import_statement"
                )
            ),
            confidence=0.85,
            timestamp=now,
            importance=ImportanceScore(
                activation=0.75,
                coherence=0.65,
                specificity=0.55,
                novelty=0.45,
                reinforcement=0.35,
            ),
            kind="extracted",
            reinforcement_count=1,
        )

        store.upsert(infon1)
        store.upsert(infon2)

        schema = AnchorSchema(anchors={}, version="1.0.0", language="code")

        # Call consolidate() first time
        consolidate(store, schema)
        stats1 = store.stats()
        edge_count_1 = stats1["edge_count"]
        constraint_count_1 = stats1["constraint_count"]

        # Call consolidate() second time
        consolidate(store, schema)
        stats2 = store.stats()
        edge_count_2 = stats2["edge_count"]
        constraint_count_2 = stats2["constraint_count"]

        # Assert counts are identical
        assert edge_count_1 == edge_count_2
        assert constraint_count_1 == constraint_count_2

        store.close()


def test_consolidate_aggregates_constraints():
    """Test that consolidate() aggregates constraints with correct evidence_count, strength, persistence."""
    from infon.consolidate import consolidate

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.ddb"
        store = InfonStore(db_path)

        now = datetime.now(timezone.utc)

        # Create two infons with the same (subject, predicate, object) triple
        # The second one will be merged via upsert, increasing reinforcement_count
        infon1 = Infon(
            id=str(uuid.uuid4()),
            subject="user_service",
            predicate="calls",
            object="auth_service",
            polarity=True,
            grounding=Grounding(
                root=ASTGrounding(
                    file_path="main.py", line_number=10, node_type="call_expression"
                )
            ),
            confidence=0.9,
            timestamp=now - timedelta(hours=2),
            importance=ImportanceScore(
                activation=0.8,
                coherence=0.7,
                specificity=0.6,
                novelty=0.5,
                reinforcement=0.4,
            ),
            kind="extracted",
            reinforcement_count=1,
        )

        # Same triple as infon1 - should merge
        infon2 = Infon(
            id=str(uuid.uuid4()),
            subject="user_service",
            predicate="calls",
            object="auth_service",
            polarity=True,
            grounding=Grounding(
                root=ASTGrounding(
                    file_path="main.py", line_number=15, node_type="call_expression"
                )
            ),
            confidence=0.95,
            timestamp=now - timedelta(hours=1),
            importance=ImportanceScore(
                activation=0.85,
                coherence=0.75,
                specificity=0.65,
                novelty=0.55,
                reinforcement=0.45,
            ),
            kind="extracted",
            reinforcement_count=1,
        )

        # Insert and merge
        stored_infon1 = store.upsert(infon1)
        stored_infon2 = store.upsert(infon2)  # This should merge with infon1

        # Verify merge happened
        assert stored_infon2.id == stored_infon1.id
        assert stored_infon2.reinforcement_count == 2

        schema = AnchorSchema(anchors={}, version="1.0.0", language="code")

        # Call consolidate
        consolidate(store, schema)

        # Query constraints table directly
        conn = store._conn
        constraints = conn.execute(
            """
            SELECT subject, predicate, object, evidence_count, strength, persistence
            FROM constraints
            WHERE subject = 'user_service' AND predicate = 'calls' AND object = 'auth_service'
            """
        ).fetchall()

        assert len(constraints) == 1
        constraint = constraints[0]

        # evidence_count should equal reinforcement_count
        assert constraint[3] == 2

        # strength should be average confidence
        # After merge: (0.9 * 1 + 0.95) / 2 = 0.925
        assert abs(constraint[4] - 0.925) < 0.01

        # persistence should be 1 - decay_factor^evidence_count
        # Default decay_factor = 0.95
        # persistence = 1 - 0.95^2 = 1 - 0.9025 = 0.0975
        expected_persistence = 1 - (0.95**2)
        assert abs(constraint[5] - expected_persistence) < 0.01

        store.close()


def test_consolidate_applies_importance_decay():
    """Test that consolidate() applies exponential decay to old infons (>7 days)."""
    from infon.consolidate import consolidate

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.ddb"
        store = InfonStore(db_path)

        now = datetime.now(timezone.utc)

        # Create an infon that's 10 days old
        old_timestamp = now - timedelta(days=10)

        old_infon = Infon(
            id=str(uuid.uuid4()),
            subject="old_service",
            predicate="calls",
            object="database",
            polarity=True,
            grounding=Grounding(
                root=ASTGrounding(
                    file_path="old.py", line_number=10, node_type="call_expression"
                )
            ),
            confidence=0.9,
            timestamp=old_timestamp,
            importance=ImportanceScore(
                activation=0.8,
                coherence=0.7,
                specificity=0.6,
                novelty=0.5,
                reinforcement=0.8,  # Starting value for decay test
            ),
            kind="extracted",
            reinforcement_count=1,
        )

        # Create a recent infon (< 7 days) - should NOT decay
        recent_infon = Infon(
            id=str(uuid.uuid4()),
            subject="new_service",
            predicate="calls",
            object="api",
            polarity=True,
            grounding=Grounding(
                root=ASTGrounding(
                    file_path="new.py", line_number=5, node_type="call_expression"
                )
            ),
            confidence=0.85,
            timestamp=now - timedelta(days=3),
            importance=ImportanceScore(
                activation=0.75,
                coherence=0.65,
                specificity=0.55,
                novelty=0.45,
                reinforcement=0.7,  # Should remain unchanged
            ),
            kind="extracted",
            reinforcement_count=1,
        )

        store.upsert(old_infon)
        store.upsert(recent_infon)

        schema = AnchorSchema(anchors={}, version="1.0.0", language="code")

        # Call consolidate
        consolidate(store, schema)

        # Retrieve the infons and check decay
        updated_old_infon = store.get(old_infon.id)
        updated_recent_infon = store.get(recent_infon.id)

        # Old infon (10 days): days_since = 10, decay_factor = 0.95
        # new_reinforcement = 0.8 * 0.95^10
        expected_old_reinforcement = 0.8 * (0.95**10)
        assert (
            abs(updated_old_infon.importance.reinforcement - expected_old_reinforcement)
            < 0.01
        )

        # Recent infon (3 days < 7 days): should NOT decay
        assert updated_recent_infon.importance.reinforcement == 0.7

        store.close()

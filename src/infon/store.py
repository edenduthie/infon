"""
DuckDB-backed storage for the Infon knowledge base.

This module implements the InfonStore class which provides persistent storage
for infons, edges, constraints, and documents using DuckDB.

The store maintains four tables:
- infons: Core knowledge triples with grounding and metadata
- edges: Relationships between infons (NEXT, CONTRADICTS, SUPPORTS, SIMILAR)
- constraints: Aggregated patterns from multiple infon observations
- documents: Tracking of ingested documents for incremental processing

The store uses WAL mode for write-ahead logging and detects concurrent writers
to prevent database corruption.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

from infon.grounding import ASTGrounding, Grounding, TextGrounding
from infon.infon import Infon, ImportanceScore


class ConcurrentWriteError(Exception):
    """Raised when attempting to open a store that is already open for writing."""

    pass


class StoreStats:
    """Statistics about the contents of the InfonStore."""

    def __init__(
        self,
        infon_count: int,
        edge_count: int,
        constraint_count: int,
        document_count: int,
        top_anchors: list[tuple[str, int]],
    ):
        self.infon_count = infon_count
        self.edge_count = edge_count
        self.constraint_count = constraint_count
        self.document_count = document_count
        self.top_anchors = top_anchors

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access for testing."""
        if key == "infon_count":
            return self.infon_count
        elif key == "edge_count":
            return self.edge_count
        elif key == "constraint_count":
            return self.constraint_count
        elif key == "document_count":
            return self.document_count
        elif key == "top_anchors":
            return self.top_anchors
        else:
            raise KeyError(key)
    
    def __contains__(self, key: str) -> bool:
        """Support 'in' operator for testing."""
        return key in ["infon_count", "edge_count", "constraint_count", "document_count", "top_anchors"]


class InfonStore:
    """
    DuckDB-backed storage for Infons and related data.

    The InfonStore manages four tables:
    - infons: Atomic information units (triples with grounding)
    - edges: Relationships between infons
    - constraints: Aggregated patterns from observations
    - documents: Tracking of ingested documents

    All write operations are transactional. The store uses WAL mode and
    detects concurrent writers to prevent corruption.

    Example:
        >>> with InfonStore("kb.ddb") as store:
        ...     store.upsert(infon)
        ...     results = store.query(subject="user_service")
    """

    def __init__(self, db_path: str | Path):
        """
        Initialize the InfonStore.

        Args:
            db_path: Path to the DuckDB database file

        Raises:
            ConcurrentWriteError: If another writer has the database open
        """
        self.db_path = Path(db_path)
        self._conn = None
        self._lock_path = self.db_path.with_suffix('.lock')

        # Create parent directory if needed
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Check for concurrent writer using lock file
        if self._lock_path.exists():
            raise ConcurrentWriteError(
                f"Database is already open by another writer: {self.db_path}"
            )

        # Create lock file
        try:
            self._lock_path.write_text(str(datetime.now(timezone.utc)))
        except Exception as e:
            raise ConcurrentWriteError(
                f"Failed to create lock file: {self._lock_path}"
            ) from e

        # Open connection
        try:
            self._conn = duckdb.connect(str(self.db_path))
        except Exception as e:
            # Clean up lock file on connection failure
            self._lock_path.unlink(missing_ok=True)
            raise

        # Enable WAL mode for write-ahead logging
        # DuckDB uses checkpoint_threshold instead of wal_autocheckpoint
        # Setting to 16MB for automatic checkpoints
        self._conn.execute("PRAGMA checkpoint_threshold='16MB'")

        # Create tables and indexes
        self._create_tables()

    def _create_tables(self) -> None:
        """Create all four tables with indexes if they don't exist."""

        # Create infons table
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS infons (
                id VARCHAR PRIMARY KEY,
                subject VARCHAR NOT NULL,
                predicate VARCHAR NOT NULL,
                object VARCHAR NOT NULL,
                polarity BOOLEAN NOT NULL,
                grounding_type VARCHAR NOT NULL,
                grounding_json JSON NOT NULL,
                confidence FLOAT NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                importance_json JSON NOT NULL,
                kind VARCHAR NOT NULL,
                reinforcement_count INTEGER NOT NULL,
                doc_id VARCHAR,
                created_at TIMESTAMPTZ NOT NULL
            )
        """)

        # Create indexes on infons table
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_infons_subject ON infons(subject)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_infons_predicate ON infons(predicate)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_infons_object ON infons(object)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_infons_triple ON infons(subject, predicate, object)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_infons_doc_id ON infons(doc_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_infons_timestamp ON infons(timestamp)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_infons_kind ON infons(kind)"
        )

        # Create edges table
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                id VARCHAR PRIMARY KEY,
                from_infon_id VARCHAR NOT NULL,
                to_infon_id VARCHAR NOT NULL,
                edge_type VARCHAR NOT NULL,
                weight FLOAT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL
            )
        """)

        # Create indexes on edges table
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(from_infon_id, edge_type)"
        )

        # Create constraints table
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS constraints (
                id VARCHAR PRIMARY KEY,
                subject VARCHAR NOT NULL,
                predicate VARCHAR NOT NULL,
                object VARCHAR NOT NULL,
                evidence_count INTEGER NOT NULL,
                strength FLOAT NOT NULL,
                persistence FLOAT NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL
            )
        """)

        # Create indexes on constraints table
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_constraints_triple ON constraints(subject, predicate, object)"
        )

        # Create documents table
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id VARCHAR PRIMARY KEY,
                path VARCHAR NOT NULL,
                kind VARCHAR NOT NULL,
                ingested_at TIMESTAMPTZ NOT NULL,
                token_count INTEGER NOT NULL
            )
        """)

        # Create index on documents table
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_path ON documents(path)"
        )

    def upsert(self, infon: Infon) -> Infon:
        """
        Insert or merge an infon.

        If an infon with the same (subject, predicate, object, polarity) already
        exists, merge by incrementing reinforcement_count and averaging confidence.
        Otherwise, insert as new.

        Args:
            infon: The Infon to upsert

        Returns:
            The upserted Infon with updated reinforcement_count and confidence
        """
        # Extract doc_id from grounding if available
        doc_id = None
        if isinstance(infon.grounding.root, TextGrounding):
            doc_id = infon.grounding.root.doc_id
        elif isinstance(infon.grounding.root, ASTGrounding):
            doc_id = infon.grounding.root.file_path

        # Check if infon with same triple+polarity already exists
        existing = self._conn.execute(
            """
            SELECT id, confidence, reinforcement_count
            FROM infons
            WHERE subject = ? AND predicate = ? AND object = ? AND polarity = ?
            """,
            [infon.subject, infon.predicate, infon.object, infon.polarity],
        ).fetchone()

        if existing:
            # Merge: increment reinforcement_count and average confidence
            existing_id, existing_confidence, existing_reinforcement = existing

            new_reinforcement = existing_reinforcement + 1
            new_confidence = (
                existing_confidence * existing_reinforcement + infon.confidence
            ) / new_reinforcement

            # Update the existing row
            self._conn.execute(
                """
                UPDATE infons
                SET confidence = ?,
                    reinforcement_count = ?,
                    timestamp = ?
                WHERE id = ?
                """,
                [new_confidence, new_reinforcement, infon.timestamp, existing_id],
            )

            # Retrieve the updated infon
            return self.get(existing_id)
        else:
            # Insert new infon
            grounding_json = json.dumps(infon.grounding.model_dump())
            importance_json = json.dumps(infon.importance.model_dump())

            self._conn.execute(
                """
                INSERT INTO infons (
                    id, subject, predicate, object, polarity,
                    grounding_type, grounding_json,
                    confidence, timestamp, importance_json, kind,
                    reinforcement_count, doc_id, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    infon.id,
                    infon.subject,
                    infon.predicate,
                    infon.object,
                    infon.polarity,
                    infon.grounding.root.grounding_type,
                    grounding_json,
                    infon.confidence,
                    infon.timestamp,
                    importance_json,
                    infon.kind,
                    infon.reinforcement_count,
                    doc_id,
                    datetime.now(timezone.utc),
                ],
            )

            return infon

    def get(self, infon_id: str) -> Infon | None:
        """
        Retrieve an infon by ID.

        Args:
            infon_id: UUID of the infon

        Returns:
            The Infon if found, None otherwise
        """
        row = self._conn.execute(
            """
            SELECT id, subject, predicate, object, polarity,
                   grounding_type, grounding_json,
                   confidence, timestamp, importance_json, kind,
                   reinforcement_count
            FROM infons
            WHERE id = ?
            """,
            [infon_id],
        ).fetchone()

        if row is None:
            return None

        return self._row_to_infon(row)

    def query(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
        min_confidence: float = 0.0,
        limit: int = 100,
    ) -> list[Infon]:
        """
        Query infons with optional filters.

        Args:
            subject: Filter by subject anchor key (optional)
            predicate: Filter by predicate anchor key (optional)
            object: Filter by object anchor key (optional)
            min_confidence: Minimum confidence threshold (default 0.0)
            limit: Maximum number of results (default 100)

        Returns:
            List of matching Infons
        """
        # Build WHERE clause dynamically
        conditions = ["confidence >= ?"]
        params: list[Any] = [min_confidence]

        if subject is not None:
            conditions.append("subject = ?")
            params.append(subject)

        if predicate is not None:
            conditions.append("predicate = ?")
            params.append(predicate)

        if object is not None:
            conditions.append("object = ?")
            params.append(object)

        where_clause = " AND ".join(conditions)
        params.append(limit)

        rows = self._conn.execute(
            f"""
            SELECT id, subject, predicate, object, polarity,
                   grounding_type, grounding_json,
                   confidence, timestamp, importance_json, kind,
                   reinforcement_count
            FROM infons
            WHERE {where_clause}
            LIMIT ?
            """,
            params,
        ).fetchall()

        return [self._row_to_infon(row) for row in rows]

    def add_edge(
        self, from_infon_id: str, to_infon_id: str, edge_type: str, weight: float
    ) -> None:
        """
        Add an edge between two infons.

        Args:
            from_infon_id: Source infon UUID
            to_infon_id: Target infon UUID
            edge_type: One of NEXT, CONTRADICTS, SUPPORTS, SIMILAR
            weight: Edge weight (typically in [0, 1])
        """
        edge_id = str(uuid.uuid4())
        self._conn.execute(
            """
            INSERT INTO edges (id, from_infon_id, to_infon_id, edge_type, weight, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                edge_id,
                from_infon_id,
                to_infon_id,
                edge_type,
                weight,
                datetime.now(timezone.utc),
            ],
        )

    def get_edges(
        self, infon_id: str, edge_type: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Retrieve edges from an infon.

        Args:
            infon_id: Source infon UUID
            edge_type: Optional filter by edge type

        Returns:
            List of edge dicts with keys: id, to_infon_id, edge_type, weight, created_at
        """
        if edge_type is None:
            rows = self._conn.execute(
                """
                SELECT id, to_infon_id, edge_type, weight, created_at
                FROM edges
                WHERE from_infon_id = ?
                """,
                [infon_id],
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT id, to_infon_id, edge_type, weight, created_at
                FROM edges
                WHERE from_infon_id = ? AND edge_type = ?
                """,
                [infon_id, edge_type],
            ).fetchall()

        return [
            {
                "id": row[0],
                "to_infon_id": row[1],
                "edge_type": row[2],
                "weight": row[3],
                "created_at": row[4],
            }
            for row in rows
        ]

    def upsert_constraint(
        self,
        subject: str,
        predicate: str,
        object: str,
        evidence_count: int,
        strength: float,
        persistence: float,
    ) -> None:
        """
        Upsert a constraint (aggregated pattern).

        Args:
            subject: Subject anchor key
            predicate: Predicate anchor key
            object: Object anchor key
            evidence_count: Number of supporting observations
            strength: Average confidence of supporting infons
            persistence: Persistence score (1 - decay^evidence_count)
        """
        # Check if constraint exists
        existing = self._conn.execute(
            """
            SELECT id FROM constraints
            WHERE subject = ? AND predicate = ? AND object = ?
            """,
            [subject, predicate, object],
        ).fetchone()

        if existing:
            # Update existing constraint
            self._conn.execute(
                """
                UPDATE constraints
                SET evidence_count = ?,
                    strength = ?,
                    persistence = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                [
                    evidence_count,
                    strength,
                    persistence,
                    datetime.now(timezone.utc),
                    existing[0],
                ],
            )
        else:
            # Insert new constraint
            constraint_id = str(uuid.uuid4())
            self._conn.execute(
                """
                INSERT INTO constraints (
                    id, subject, predicate, object,
                    evidence_count, strength, persistence, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    constraint_id,
                    subject,
                    predicate,
                    object,
                    evidence_count,
                    strength,
                    persistence,
                    datetime.now(timezone.utc),
                ],
            )

    def upsert_document(
        self, doc_id: str, path: str, kind: str, token_count: int
    ) -> None:
        """
        Upsert a document record.

        Args:
            doc_id: Unique document identifier
            path: File path or document path
            kind: Document kind (e.g., "code", "text")
            token_count: Number of tokens in document
        """
        # Check if document exists
        existing = self._conn.execute(
            """
            SELECT id FROM documents WHERE id = ?
            """,
            [doc_id],
        ).fetchone()

        if existing:
            # Update existing document
            self._conn.execute(
                """
                UPDATE documents
                SET path = ?,
                    kind = ?,
                    token_count = ?,
                    ingested_at = ?
                WHERE id = ?
                """,
                [path, kind, token_count, datetime.now(timezone.utc), doc_id],
            )
        else:
            # Insert new document
            self._conn.execute(
                """
                INSERT INTO documents (id, path, kind, ingested_at, token_count)
                VALUES (?, ?, ?, ?, ?)
                """,
                [doc_id, path, kind, datetime.now(timezone.utc), token_count],
            )

    def stats(self) -> StoreStats:
        """
        Get statistics about the store.

        Returns:
            StoreStats with counts and top anchors
        """
        # Get counts
        infon_count = self._conn.execute("SELECT COUNT(*) FROM infons").fetchone()[0]
        edge_count = self._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        constraint_count = self._conn.execute(
            "SELECT COUNT(*) FROM constraints"
        ).fetchone()[0]
        document_count = self._conn.execute(
            "SELECT COUNT(*) FROM documents"
        ).fetchone()[0]

        # Get top-10 most referenced anchors
        # Count anchor occurrences as subject or object
        top_anchors_rows = self._conn.execute(
            """
            WITH anchor_counts AS (
                SELECT subject AS anchor FROM infons
                UNION ALL
                SELECT object AS anchor FROM infons
            )
            SELECT anchor, COUNT(*) as count
            FROM anchor_counts
            GROUP BY anchor
            ORDER BY count DESC
            LIMIT 10
            """
        ).fetchall()

        top_anchors = [(row[0], row[1]) for row in top_anchors_rows]

        return StoreStats(
            infon_count=infon_count,
            edge_count=edge_count,
            constraint_count=constraint_count,
            document_count=document_count,
            top_anchors=top_anchors,
        )

    def close(self) -> None:
        """Close the database connection and release the lock."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        
        # Remove lock file
        if hasattr(self, '_lock_path'):
            self._lock_path.unlink(missing_ok=True)

    def _row_to_infon(self, row: tuple) -> Infon:
        """
        Convert a database row to an Infon instance.

        Args:
            row: Database row tuple

        Returns:
            Reconstructed Infon instance
        """
        (
            id,
            subject,
            predicate,
            object,
            polarity,
            grounding_type,
            grounding_json,
            confidence,
            timestamp,
            importance_json,
            kind,
            reinforcement_count,
        ) = row

        # Deserialize grounding
        # The Grounding RootModel unwraps the data in model_dump(), so grounding_json
        # contains the inner TextGrounding/ASTGrounding data directly
        grounding_data = json.loads(grounding_json)
        if grounding_type == "text":
            grounding = Grounding(root=TextGrounding(**grounding_data))
        elif grounding_type == "ast":
            grounding = Grounding(root=ASTGrounding(**grounding_data))
        else:
            raise ValueError(f"Unknown grounding type: {grounding_type}")

        # Deserialize importance
        importance_data = json.loads(importance_json)
        importance = ImportanceScore(**importance_data)

        return Infon(
            id=id,
            subject=subject,
            predicate=predicate,
            object=object,
            polarity=polarity,
            grounding=grounding,
            confidence=confidence,
            timestamp=timestamp,
            importance=importance,
            kind=kind,
            reinforcement_count=reinforcement_count,
        )

    def __enter__(self) -> "InfonStore":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager and close connection."""
        self.close()

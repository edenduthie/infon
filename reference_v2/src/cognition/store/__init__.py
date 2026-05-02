"""Store backends: protocol + local (SQLite) and cloud (DynamoDB+S3).

All stores implement the StoreBackend protocol — code above this layer
never knows which backend is active.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..infon import Infon, Edge, Constraint


@runtime_checkable
class StoreBackend(Protocol):
    """Protocol that all store backends implement."""

    def init(self) -> None:
        """Create tables/indexes if they don't exist."""
        ...

    # ── Infons ──────────────────────────────────────────────────────────

    def put_infon(self, infon: Infon) -> None:
        """Insert or update a single infon."""
        ...

    def put_infons(self, infons: list[Infon]) -> None:
        """Batch insert/update infons."""
        ...

    def get_infon(self, infon_id: str) -> Infon | None:
        """Retrieve one infon by ID."""
        ...

    def query_infons(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
        doc_id: str | None = None,
        min_importance: float = 0.0,
        limit: int = 100,
    ) -> list[Infon]:
        """Query infons by anchor roles, document, or importance."""
        ...

    def get_infons_for_anchor(self, anchor: str, role: str | None = None,
                               limit: int = 100) -> list[Infon]:
        """Get all infons where anchor appears in any/specified role."""
        ...

    def count_infons(self) -> int:
        """Total number of infons in the store."""
        ...

    # ── Edges ───────────────────────────────────────────────────────────

    def put_edge(self, edge: Edge) -> None:
        """Insert or update an edge."""
        ...

    def put_edges(self, edges: list[Edge]) -> None:
        """Batch insert/update edges."""
        ...

    def get_edges(self, source: str | None = None, target: str | None = None,
                  edge_type: str | None = None, limit: int = 100) -> list[Edge]:
        """Query edges by source, target, or type."""
        ...

    def get_next_chain(self, infon_id: str, anchor: str,
                       limit: int = 50) -> list[Edge]:
        """Get the NEXT chain for an infon through a shared anchor."""
        ...

    # ── Constraints ─────────────────────────────────────────────────────

    def put_constraint(self, constraint: Constraint) -> None:
        """Insert or update a constraint."""
        ...

    def get_constraints(self, subject: str | None = None,
                        predicate: str | None = None,
                        object: str | None = None,
                        min_score: float = 0.0,
                        limit: int = 100) -> list[Constraint]:
        """Query constraints."""
        ...

    # ── Maintenance ─────────────────────────────────────────────────────

    def prune(self, threshold: float) -> int:
        """Soft-delete infons below importance threshold. Returns count pruned."""
        ...

    def vacuum(self) -> None:
        """Reclaim storage from pruned records."""
        ...

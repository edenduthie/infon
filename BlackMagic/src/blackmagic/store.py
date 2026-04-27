"""LocalStore: single-file SQLite backend for BlackMagic.

Tables:
  - infons     : extracted triples with full metadata + imagination columns
  - edges      : typed directed edges (NEXT, INITIATES, ASSERTS, TARGETS, ...)
  - constraints: cross-document aggregations of triples
  - documents  : original doc text, for snippet retrieval

The `kind` column on infons distinguishes "observed" (extracted from source
text) from "imagined" (proposed by the GA imagination layer). Most queries
default to observed only; imagination consumers opt in explicitly.
"""

from __future__ import annotations

import json
import sqlite3

from .infon import Infon, Edge, Constraint


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS infons (
    infon_id    TEXT PRIMARY KEY,
    subject     TEXT NOT NULL,
    predicate   TEXT NOT NULL,
    object      TEXT NOT NULL,
    polarity    INTEGER DEFAULT 1,
    direction   TEXT DEFAULT 'forward',
    confidence  REAL DEFAULT 0.0,
    sentence    TEXT,
    doc_id      TEXT,
    sent_id     TEXT,
    spans       TEXT,
    support     TEXT,
    subject_meta  TEXT,
    predicate_meta TEXT,
    object_meta   TEXT,
    locations   TEXT,
    timestamp   TEXT,
    precision   TEXT DEFAULT 'unknown',
    temporal_refs TEXT,
    tense       TEXT DEFAULT 'unknown',
    aspect      TEXT DEFAULT 'unknown',
    activation  REAL DEFAULT 0.0,
    coherence   REAL DEFAULT 0.0,
    specificity REAL DEFAULT 0.0,
    novelty     REAL DEFAULT 1.0,
    importance  REAL DEFAULT 0.0,
    metrics     TEXT,
    reinforcement_count INTEGER DEFAULT 0,
    last_reinforced TEXT,
    decay_rate  REAL DEFAULT 0.01,
    pruned      INTEGER DEFAULT 0,
    kind        TEXT DEFAULT 'observed',
    parent_infon_ids TEXT,
    fitness     REAL
);

CREATE INDEX IF NOT EXISTS idx_infon_subject ON infons(subject);
CREATE INDEX IF NOT EXISTS idx_infon_predicate ON infons(predicate);
CREATE INDEX IF NOT EXISTS idx_infon_object ON infons(object);
CREATE INDEX IF NOT EXISTS idx_infon_doc ON infons(doc_id);
CREATE INDEX IF NOT EXISTS idx_infon_timestamp ON infons(timestamp);
CREATE INDEX IF NOT EXISTS idx_infon_importance ON infons(importance);
CREATE INDEX IF NOT EXISTS idx_infon_kind ON infons(kind);
CREATE INDEX IF NOT EXISTS idx_infon_triple ON infons(subject, predicate, object);

CREATE TABLE IF NOT EXISTS edges (
    source      TEXT NOT NULL,
    target      TEXT NOT NULL,
    edge_type   TEXT NOT NULL,
    weight      REAL DEFAULT 1.0,
    metadata    TEXT,
    PRIMARY KEY (source, target, edge_type)
);

CREATE INDEX IF NOT EXISTS idx_edge_source ON edges(source);
CREATE INDEX IF NOT EXISTS idx_edge_target ON edges(target);
CREATE INDEX IF NOT EXISTS idx_edge_type ON edges(edge_type);

CREATE TABLE IF NOT EXISTS constraints (
    subject     TEXT NOT NULL,
    predicate   TEXT NOT NULL,
    object      TEXT NOT NULL,
    evidence    INTEGER DEFAULT 0,
    doc_count   INTEGER DEFAULT 0,
    strength    REAL DEFAULT 0.0,
    persistence INTEGER DEFAULT 0,
    score       REAL DEFAULT 0.0,
    infon_ids   TEXT,
    PRIMARY KEY (subject, predicate, object)
);

CREATE INDEX IF NOT EXISTS idx_constraint_score ON constraints(score);

CREATE TABLE IF NOT EXISTS documents (
    doc_id    TEXT PRIMARY KEY,
    text      TEXT NOT NULL,
    timestamp TEXT,
    n_infons  INTEGER DEFAULT 0,
    ingested_at REAL DEFAULT (unixepoch())
);
"""

# JSON-serialized fields on the Infon dataclass
_JSON_FIELDS = {
    "spans", "support", "subject_meta", "predicate_meta", "object_meta",
    "locations", "temporal_refs", "metrics", "parent_infon_ids",
}

_INFON_COLS = [
    "infon_id", "subject", "predicate", "object", "polarity", "direction",
    "confidence", "sentence", "doc_id", "sent_id", "spans", "support",
    "subject_meta", "predicate_meta", "object_meta", "locations",
    "timestamp", "precision", "temporal_refs", "tense", "aspect",
    "metrics",
    "activation", "coherence", "specificity", "novelty", "importance",
    "reinforcement_count", "last_reinforced", "decay_rate",
    "kind", "parent_infon_ids", "fitness",
]


def _infon_to_row(infon: Infon) -> tuple:
    d = infon.to_dict()
    # metrics isn't a defined Infon field in BlackMagic; coerce to None.
    d.setdefault("metrics", None)
    vals = []
    for col in _INFON_COLS:
        v = d.get(col)
        if col in _JSON_FIELDS and v is not None:
            v = json.dumps(v)
        vals.append(v)
    return tuple(vals)


def _row_to_infon(row: sqlite3.Row) -> Infon:
    d = dict(row)
    for fld in _JSON_FIELDS:
        if d.get(fld) and isinstance(d[fld], str):
            try:
                d[fld] = json.loads(d[fld])
            except (json.JSONDecodeError, TypeError):
                d[fld] = None
    d.pop("pruned", None)
    d.pop("metrics", None)  # not on BlackMagic Infon
    return Infon.from_dict(d)


class LocalStore:
    """SQLite-backed store for BlackMagic."""

    def __init__(self, db_path: str = "blackmagic.db"):
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        return self._conn

    def init(self) -> None:
        self.conn.executescript(_SCHEMA_SQL)
        # Migration: add imagination columns if the DB pre-dates them
        for col, ddl in [
            ("kind",             "ALTER TABLE infons ADD COLUMN kind TEXT DEFAULT 'observed'"),
            ("parent_infon_ids", "ALTER TABLE infons ADD COLUMN parent_infon_ids TEXT"),
            ("fitness",          "ALTER TABLE infons ADD COLUMN fitness REAL"),
        ]:
            try:
                self.conn.execute(f"SELECT {col} FROM infons LIMIT 1")
            except sqlite3.OperationalError:
                self.conn.execute(ddl)
                self.conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── Infons ──────────────────────────────────────────────────────────

    def put_infon(self, infon: Infon) -> None:
        placeholders = ", ".join(["?"] * len(_INFON_COLS))
        cols = ", ".join(_INFON_COLS)
        sql = f"INSERT OR REPLACE INTO infons ({cols}) VALUES ({placeholders})"
        self.conn.execute(sql, _infon_to_row(infon))
        self.conn.commit()

    def put_infons(self, infons: list[Infon]) -> None:
        if not infons:
            return
        placeholders = ", ".join(["?"] * len(_INFON_COLS))
        cols = ", ".join(_INFON_COLS)
        sql = f"INSERT OR REPLACE INTO infons ({cols}) VALUES ({placeholders})"
        self.conn.executemany(sql, [_infon_to_row(inf) for inf in infons])
        self.conn.commit()

    def get_infon(self, infon_id: str) -> Infon | None:
        row = self.conn.execute(
            "SELECT * FROM infons WHERE infon_id = ? AND pruned = 0",
            (infon_id,),
        ).fetchone()
        return _row_to_infon(row) if row else None

    def query_infons(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
        doc_id: str | None = None,
        min_importance: float = 0.0,
        kind: str | None = "observed",
        include_imagined: bool = False,
        limit: int = 100,
    ) -> list[Infon]:
        """Retrieve infons, optionally filtered by role or kind.

        Default: observed only. Pass `include_imagined=True` to include both,
        or `kind="imagined"` to get only imagined infons.
        """
        clauses = ["pruned = 0"]
        params: list = []
        if subject:
            clauses.append("subject = ?")
            params.append(subject)
        if predicate:
            clauses.append("predicate = ?")
            params.append(predicate)
        if object:
            clauses.append("object = ?")
            params.append(object)
        if doc_id:
            clauses.append("doc_id = ?")
            params.append(doc_id)
        if min_importance > 0:
            clauses.append("importance >= ?")
            params.append(min_importance)
        if not include_imagined and kind is not None:
            clauses.append("kind = ?")
            params.append(kind)

        where = " AND ".join(clauses)
        sql = f"SELECT * FROM infons WHERE {where} ORDER BY importance DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        return [_row_to_infon(r) for r in rows]

    def get_infons_for_anchor(self, anchor: str, role: str | None = None,
                              include_imagined: bool = False,
                              limit: int = 100) -> list[Infon]:
        kind_clause = "" if include_imagined else " AND kind = 'observed'"
        if role == "subject":
            sql = f"SELECT * FROM infons WHERE subject = ? AND pruned = 0{kind_clause} ORDER BY importance DESC LIMIT ?"
        elif role == "predicate":
            sql = f"SELECT * FROM infons WHERE predicate = ? AND pruned = 0{kind_clause} ORDER BY importance DESC LIMIT ?"
        elif role == "object":
            sql = f"SELECT * FROM infons WHERE object = ? AND pruned = 0{kind_clause} ORDER BY importance DESC LIMIT ?"
        else:
            sql = (
                f"SELECT * FROM infons WHERE "
                f"(subject = ? OR predicate = ? OR object = ?) "
                f"AND pruned = 0{kind_clause} ORDER BY importance DESC LIMIT ?"
            )
            rows = self.conn.execute(
                sql, (anchor, anchor, anchor, limit)
            ).fetchall()
            return [_row_to_infon(r) for r in rows]

        rows = self.conn.execute(sql, (anchor, limit)).fetchall()
        return [_row_to_infon(r) for r in rows]

    def count_infons(self, kind: str | None = "observed") -> int:
        if kind is None:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM infons WHERE pruned = 0"
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM infons WHERE pruned = 0 AND kind = ?",
                (kind,),
            ).fetchone()
        return row[0]

    def delete_imagined(self) -> int:
        """Remove all imagined infons. Returns number deleted."""
        cur = self.conn.execute(
            "DELETE FROM infons WHERE kind = 'imagined'"
        )
        self.conn.commit()
        return cur.rowcount

    # ── Edges ───────────────────────────────────────────────────────────

    def put_edge(self, edge: Edge) -> None:
        meta = json.dumps(edge.metadata) if edge.metadata else None
        self.conn.execute(
            "INSERT OR REPLACE INTO edges (source, target, edge_type, weight, metadata) "
            "VALUES (?, ?, ?, ?, ?)",
            (edge.source, edge.target, edge.edge_type, edge.weight, meta),
        )
        self.conn.commit()

    def put_edges(self, edges: list[Edge]) -> None:
        if not edges:
            return
        rows = []
        for e in edges:
            meta = json.dumps(e.metadata) if e.metadata else None
            rows.append((e.source, e.target, e.edge_type, e.weight, meta))
        self.conn.executemany(
            "INSERT OR REPLACE INTO edges (source, target, edge_type, weight, metadata) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        self.conn.commit()

    def get_edges(self, source: str | None = None, target: str | None = None,
                  edge_type: str | None = None, limit: int = 100) -> list[Edge]:
        clauses = []
        params: list = []
        if source:
            clauses.append("source = ?")
            params.append(source)
        if target:
            clauses.append("target = ?")
            params.append(target)
        if edge_type:
            clauses.append("edge_type = ?")
            params.append(edge_type)

        where = " AND ".join(clauses) if clauses else "1=1"
        sql = f"SELECT * FROM edges WHERE {where} LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()

        result = []
        for r in rows:
            meta = json.loads(r["metadata"]) if r["metadata"] else {}
            result.append(Edge(
                source=r["source"], target=r["target"],
                edge_type=r["edge_type"], weight=r["weight"],
                metadata=meta,
            ))
        return result

    def get_next_chain(self, infon_id: str, anchor: str,
                       limit: int = 50) -> list[Edge]:
        """Walk the NEXT chain forward from an infon through a shared anchor."""
        chain = []
        current = infon_id
        seen = set()
        while len(chain) < limit and current not in seen:
            seen.add(current)
            row = self.conn.execute(
                "SELECT * FROM edges WHERE source = ? AND edge_type = 'NEXT' "
                "AND metadata LIKE ? LIMIT 1",
                (current, f'%"{anchor}"%'),
            ).fetchone()
            if not row:
                break
            meta = json.loads(row["metadata"]) if row["metadata"] else {}
            edge = Edge(source=row["source"], target=row["target"],
                        edge_type="NEXT", weight=row["weight"], metadata=meta)
            chain.append(edge)
            current = row["target"]
        return chain

    # ── Constraints ─────────────────────────────────────────────────────

    def put_constraint(self, constraint: Constraint) -> None:
        ids_json = json.dumps(constraint.infon_ids) if constraint.infon_ids else "[]"
        self.conn.execute(
            "INSERT OR REPLACE INTO constraints "
            "(subject, predicate, object, evidence, doc_count, strength, "
            "persistence, score, infon_ids) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (constraint.subject, constraint.predicate, constraint.object,
             constraint.evidence, constraint.doc_count, constraint.strength,
             constraint.persistence, constraint.score, ids_json),
        )
        self.conn.commit()

    def get_constraints(self, subject: str | None = None,
                        predicate: str | None = None,
                        object: str | None = None,
                        min_score: float = 0.0,
                        limit: int = 100) -> list[Constraint]:
        clauses: list[str] = []
        params: list = []
        if subject:
            clauses.append("subject = ?")
            params.append(subject)
        if predicate:
            clauses.append("predicate = ?")
            params.append(predicate)
        if object:
            clauses.append("object = ?")
            params.append(object)
        if min_score > 0:
            clauses.append("score >= ?")
            params.append(min_score)

        where = " AND ".join(clauses) if clauses else "1=1"
        sql = f"SELECT * FROM constraints WHERE {where} ORDER BY score DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()

        result = []
        for r in rows:
            ids = json.loads(r["infon_ids"]) if r["infon_ids"] else []
            result.append(Constraint(
                subject=r["subject"], predicate=r["predicate"],
                object=r["object"], evidence=r["evidence"],
                doc_count=r["doc_count"], strength=r["strength"],
                persistence=r["persistence"], score=r["score"],
                infon_ids=ids,
            ))
        return result

    # ── Documents ───────────────────────────────────────────────────────

    def put_document(self, doc_id: str, text: str,
                     timestamp: str | None = None,
                     n_infons: int = 0) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO documents "
            "(doc_id, text, timestamp, n_infons, ingested_at) "
            "VALUES (?, ?, ?, ?, unixepoch())",
            (doc_id, text, timestamp, n_infons),
        )
        self.conn.commit()

    def get_document(self, doc_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return dict(row) if row else None

    def count_documents(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()
        return row[0]

    # ── Maintenance ─────────────────────────────────────────────────────

    def prune(self, threshold: float) -> int:
        cursor = self.conn.execute(
            "UPDATE infons SET pruned = 1 WHERE importance < ? AND pruned = 0",
            (threshold,),
        )
        self.conn.commit()
        return cursor.rowcount

    def vacuum(self) -> None:
        self.conn.execute("VACUUM")

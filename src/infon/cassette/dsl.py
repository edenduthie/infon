"""Composable query DSL over the cassette store.

Three primitive layers, all indexed (no hydration required to answer):

  Grammar   — who/what: subject / predicate / object / polarity / anchor
  Timeline  — when:      before / after / between
  Logic     — how they combine: AND (chain), OR (run_any),
                                NOT (contradicting = polarity-flipped triple)

Every `Query` produces `list[Hit]` — same type the retrieval layer already
consumes. The manifest-level anchor/time pruner still fires because we
route each query to the narrowest index kind that answers it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .index import (
    Manifest, Hit, _tbl_to_hits, _scan,
    _prune_by_anchors, _prune_by_time, _all_paths,
)


# ═══════════════════════════════════════════════════════════════════════
# QUERY  (immutable; builder methods return new instances)
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Query:
    """A composable filter over the cassette store.

    Pinned roles (subject/predicate/object) are AND-combined and drive the
    manifest pruner. Anchor-only filters use the `by_anchor` index so you
    can find a name in any role.
    """
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    polarity: int | None = None            # 1=affirmed, 0=negated
    min_confidence: float = 0.0
    anchor_any: tuple[str, ...] = ()        # match any of these in any role
    t_start: str | None = None
    t_end: str | None = None
    # Role-level anchor SETS, populated by expand_hierarchy(). When
    # non-empty they widen the role filter from a singleton to "any of".
    # Kept separate from the singleton fields so existing callers are
    # unaffected and the pruner can still short-circuit on non-expanded
    # queries.
    subject_set: tuple[str, ...] = ()
    predicate_set: tuple[str, ...] = ()
    object_set: tuple[str, ...] = ()

    # ── GRAMMAR ─────────────────────────────────────────────────────────
    def where(self, *, subject: str | None = None,
              predicate: str | None = None,
              object: str | None = None) -> "Query":
        """Pin triple roles. Multiple where() calls AND together."""
        return replace(self,
                       subject=subject or self.subject,
                       predicate=predicate or self.predicate,
                       object=object or self.object)

    def mentioning(self, *anchors: str) -> "Query":
        """Role-free anchor match — finds the anchor in any of S/P/O."""
        return replace(self, anchor_any=self.anchor_any + anchors)

    def affirmed(self) -> "Query":
        return replace(self, polarity=1)

    def negated(self) -> "Query":
        return replace(self, polarity=0)

    def min_conf(self, c: float) -> "Query":
        return replace(self, min_confidence=c)

    # ── TIMELINE ────────────────────────────────────────────────────────
    def between(self, t_start: str, t_end: str) -> "Query":
        return replace(self, t_start=t_start, t_end=t_end)

    def before(self, t: str) -> "Query":
        return replace(self, t_end=t)

    def after(self, t: str) -> "Query":
        return replace(self, t_start=t)

    # ── HIERARCHY EXPANSION ─────────────────────────────────────────────
    def expand_hierarchy(self, schema) -> "Query":
        """Walk schema descendants for each pinned role.

        A query pinned to a parent anchor (e.g. subject='chip_maker')
        becomes a query matching ANY of the parent plus its descendants
        (tsmc, intel, samsung, ...). The singleton fields stay populated
        so `describe()` still reads naturally; execution walks the set.

        Invariants:
          • Non-pinned roles are untouched.
          • Already-set expansion sets are overwritten (re-expanding is
            idempotent; calling twice doesn't double up).
          • Schema with no hierarchy (flat) returns a Query with
            singleton-equivalent sets — harmless, the executor unifies
            the two paths.
        """
        def expand(name: str | None) -> tuple[str, ...]:
            if name is None:
                return ()
            # Include the anchor itself plus all descendants (transitive).
            descendants = tuple(schema.get_descendants(name))
            return (name,) + descendants

        return replace(
            self,
            subject_set=expand(self.subject),
            predicate_set=expand(self.predicate),
            object_set=expand(self.object),
        )

    # ── LOGIC: NOT (polarity flip on a pinned triple) ───────────────────
    def contradicting(self) -> "Query":
        """Flip polarity. 'X invests Y' → 'X does NOT invest Y'.
        Only meaningful when the triple roles are pinned; use to find
        refuters of a specific claim."""
        new_pol = 0 if (self.polarity is None or self.polarity == 1) else 1
        return replace(self, polarity=new_pol)

    # ── EXECUTION ───────────────────────────────────────────────────────
    def run(self, manifest: Manifest) -> list[Hit]:
        return _execute(manifest, self)

    # ── REPR ────────────────────────────────────────────────────────────
    def describe(self) -> str:
        parts = []
        for k in ("subject", "predicate", "object"):
            v = getattr(self, k)
            if v: parts.append(f"{k[0]}={v}")
        if self.polarity is not None: parts.append(f"pol={self.polarity}")
        if self.min_confidence: parts.append(f"conf≥{self.min_confidence:.2f}")
        if self.anchor_any: parts.append(f"mentions={list(self.anchor_any)}")
        if self.t_start or self.t_end:
            parts.append(f"t∈[{self.t_start or '-∞'},{self.t_end or '+∞'}]")
        return "Query(" + ", ".join(parts) + ")"


# ═══════════════════════════════════════════════════════════════════════
# LOGIC: OR, AND (AND is chaining, OR needs a helper)
# ═══════════════════════════════════════════════════════════════════════

def run_any(manifest: Manifest, queries: Iterable[Query]) -> list[Hit]:
    """OR: union of hits across queries, deduped by infon_id."""
    seen: dict[str, Hit] = {}
    for q in queries:
        for h in q.run(manifest):
            seen.setdefault(h.loc.infon_id, h)
    return list(seen.values())


def run_all(manifest: Manifest, queries: Iterable[Query]) -> list[Hit]:
    """AND across independent queries (rare — usually just chain on one).
    Kept for the case where you have two disjoint grammar constraints
    that can't both be expressed in a single Query (e.g. anchor_any ∩
    triple-pinned)."""
    sets: list[set[str]] = []
    by_id: dict[str, Hit] = {}
    for q in queries:
        hits = q.run(manifest)
        sets.append({h.loc.infon_id for h in hits})
        for h in hits:
            by_id.setdefault(h.loc.infon_id, h)
    if not sets:
        return []
    common = sets[0]
    for s in sets[1:]:
        common &= s
    return [by_id[i] for i in common]


# ═══════════════════════════════════════════════════════════════════════
# EXECUTION PLANNER
# ═══════════════════════════════════════════════════════════════════════
# Pick the narrowest index + use manifest pruning. The key idea: we don't
# always have a pinned triple role — so three paths:
#
#   1. Pinned triple (at least one of s/p/o) → by_triple + anchor prune.
#   2. anchor_any only → by_anchor + anchor prune (OR across anchors).
#   3. Nothing pinned, just time → by_time + time prune.
#
# All three can be further narrowed by polarity / confidence / time.


def _execute(m: Manifest, q: Query) -> list[Hit]:
    has_triple = bool(q.subject or q.predicate or q.object)

    # Role filter sets: prefer the expanded set when populated, else
    # fall back to the singleton. "None" means role is not pinned.
    def role_set(singleton: str | None,
                 expanded: tuple[str, ...]) -> set[str] | None:
        if expanded:
            return set(expanded)
        if singleton:
            return {singleton}
        return None

    subj_set = role_set(q.subject, q.subject_set)
    pred_set = role_set(q.predicate, q.predicate_set)
    obj_set  = role_set(q.object,  q.object_set)

    if has_triple:
        filters = []
        # "in" filters are supported by pyarrow; use them when the set
        # is >1, else keep the "=" form (smaller filter object).
        def eq_or_in(col: str, values: set[str]):
            if len(values) == 1:
                filters.append((col, "=", next(iter(values))))
            else:
                filters.append((col, "in", list(values)))

        if subj_set: eq_or_in("subject", subj_set)
        if pred_set: eq_or_in("predicate", pred_set)
        if obj_set:  eq_or_in("object",   obj_set)
        if q.polarity is not None:
            filters.append(("polarity", "=", q.polarity))
        if q.min_confidence > 0:
            filters.append(("confidence", ">=", q.min_confidence))
        if q.t_start: filters.append(("timestamp", ">=", q.t_start))
        if q.t_end:   filters.append(("timestamp", "<=", q.t_end))

        # Pruner: pass the full role set so a cassette is kept if ANY of
        # the expanded anchors appears in it. This preserves the 10-300x
        # shard-pruning speedup we verified earlier.
        if subj_set:
            paths = _prune_by_anchors(m, "by_triple", subj_set, ("subjects",))
        elif pred_set:
            paths = _prune_by_anchors(m, "by_triple", pred_set, ("predicates",))
        else:
            paths = _prune_by_anchors(m, "by_triple", obj_set, ("objects",))
        if q.t_start or q.t_end:
            paths = _intersect_time(m, paths, q)
        hits = _tbl_to_hits(_scan(paths, filters or None))

    elif q.anchor_any:
        # by_anchor shard → dedupe across roles
        anchors = set(q.anchor_any)
        paths = _prune_by_anchors(m, "by_anchor", anchors,
                                  ("subjects", "predicates", "objects"))
        filters = [("anchor", "in", list(anchors))]
        if q.polarity is not None:  # polarity not in by_anchor → post-filter
            pass
        if q.t_start: filters.append(("timestamp", ">=", q.t_start))
        if q.t_end:   filters.append(("timestamp", "<=", q.t_end))
        if q.min_confidence > 0:
            filters.append(("confidence", ">=", q.min_confidence))
        if q.t_start or q.t_end:
            paths = _intersect_time(m, paths, q)
        hits = _tbl_to_hits(_scan(paths, filters))
        # dedupe — one infon can match multiple roles
        seen: dict[str, Hit] = {}
        for h in hits:
            seen.setdefault(h.loc.infon_id, h)
        hits = list(seen.values())
        # polarity post-filter
        if q.polarity is not None:
            hits = [h for h in hits if h.loc.polarity == q.polarity]

    elif q.t_start or q.t_end:
        paths = _prune_by_time(m, "by_time",
                               q.t_start or "", q.t_end or "9999")
        filters = []
        if q.t_start: filters.append(("timestamp", ">=", q.t_start))
        if q.t_end:   filters.append(("timestamp", "<=", q.t_end))
        hits = _tbl_to_hits(_scan(paths, filters))
        # timeline shard has no polarity/confidence — post-filter
        if q.polarity is not None:
            hits = [h for h in hits if h.loc.polarity == q.polarity]
        if q.min_confidence > 0:
            hits = [h for h in hits if h.loc.confidence >= q.min_confidence]
    else:
        # Fully open query → scan by_triple (there's no way to prune).
        hits = _tbl_to_hits(_scan(_all_paths(m, "by_triple"), None))

    return hits


def _intersect_time(m: Manifest, paths: list[str], q: Query) -> list[str]:
    """Keep only shards whose cassette's time range overlaps [q.t_start, q.t_end]."""
    kept = set()
    for c in m.cassettes:
        t_min, t_max = c.get("t_min"), c.get("t_max")
        if t_min is None or t_max is None:
            continue
        if q.t_end and t_min > q.t_end: continue
        if q.t_start and t_max < q.t_start: continue
        for p in c.get("index_paths", {}).values():
            kept.add(p)
    return [p for p in paths if p in kept]


# ═══════════════════════════════════════════════════════════════════════
# TIMELINE HELPERS (aggregate pushdown — never hydrate)
# ═══════════════════════════════════════════════════════════════════════

def first_seen(m: Manifest, anchor: str) -> str | None:
    """Earliest timestamp mentioning `anchor`. Pure Parquet aggregation."""
    return _extreme_time(m, anchor, mode="min")


def last_seen(m: Manifest, anchor: str) -> str | None:
    return _extreme_time(m, anchor, mode="max")


def _extreme_time(m: Manifest, anchor: str, mode: str) -> str | None:
    paths = _prune_by_anchors(m, "by_anchor", {anchor},
                              ("subjects", "predicates", "objects"))
    if not paths:
        return None
    best = None
    for p in paths:
        if "://" in p:
            import fsspec
            fs, rel = fsspec.core.url_to_fs(p)
            tbl = pq.read_table(rel, filesystem=fs,
                                 columns=["anchor", "timestamp"],
                                 filters=[("anchor", "=", anchor)])
        else:
            tbl = pq.read_table(p, columns=["anchor", "timestamp"],
                                 filters=[("anchor", "=", anchor)])
        if tbl.num_rows == 0:
            continue
        col = tbl.column("timestamp")
        v = pc.min(col).as_py() if mode == "min" else pc.max(col).as_py()
        if v is None or v == "":
            continue
        if best is None or (v < best if mode == "min" else v > best):
            best = v
    return best


def timeline(m: Manifest, anchor: str) -> list[tuple[str, Hit]]:
    """All hits for `anchor`, sorted by timestamp. Timeline view of an entity."""
    from .index import query_anchor
    hits = query_anchor(m, anchor)
    ts_hits = [(h.loc.timestamp or "", h) for h in hits]
    ts_hits.sort(key=lambda x: x[0])
    return ts_hits


def count_by(m: Manifest, q: Query, groupby: str = "predicate") -> dict[str, int]:
    """Aggregate: how many hits per groupby value. No hydration.
    Reads the by_triple index rows of a filtered query and counts."""
    hits = q.run(m)
    out: dict[str, int] = {}
    for h in hits:
        key = getattr(h.loc, groupby, "") or "<none>"
        out[key] = out.get(key, 0) + 1
    return out


# ═══════════════════════════════════════════════════════════════════════
# TRAJECTORY  (query-time NEXT — see BUILD NOTES at end of file)
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class NextEdge:
    """Consecutive-in-time pair of infons for a shared anchor.

    NEXT isn't materialized; it's derived at query time by sorting an
    anchor's hits by timestamp. This means delta ingests never need to
    rewrite edge files — the edges emerge whenever you ask for them.
    """
    from_infon_id: str
    to_infon_id: str
    anchor: str
    from_timestamp: str
    to_timestamp: str
    gap_days: int | None  # may be None if timestamps aren't ISO dates


def trajectory_hits(m: Manifest, anchor: str,
                    role: str | None = None) -> list[Hit]:
    """All hits for `anchor`, sorted by timestamp ascending.

    This is the ordered sequence on which NEXT edges are defined: each
    consecutive pair is a NEXT edge. Returns hits (not Infons) so the
    caller can decide whether to hydrate — for building chains, the
    index-level data is usually enough.
    """
    from .index import query_anchor
    hits = query_anchor(m, anchor, role=role)
    return sorted(hits, key=lambda h: (h.loc.timestamp or ""))


def next_edges(m: Manifest, anchor: str,
               role: str | None = None) -> list[NextEdge]:
    """Consecutive-in-time edges for an anchor's hits. Pure index read."""
    hits = trajectory_hits(m, anchor, role=role)
    out: list[NextEdge] = []
    for a, b in zip(hits, hits[1:]):
        gap = _days_between(a.loc.timestamp or "", b.loc.timestamp or "")
        out.append(NextEdge(
            from_infon_id=a.loc.infon_id,
            to_infon_id=b.loc.infon_id,
            anchor=anchor,
            from_timestamp=a.loc.timestamp or "",
            to_timestamp=b.loc.timestamp or "",
            gap_days=gap,
        ))
    return out


def _days_between(a: str, b: str) -> int | None:
    try:
        import datetime as _dt
        da = _dt.date.fromisoformat(a[:10])
        db = _dt.date.fromisoformat(b[:10])
        return (db - da).days
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════
# CONSTRAINT  (corpus-level reinforcement, no hydration)
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Constraint:
    """Aggregated view of all infons matching a specific (S, P, O) triple.

    Consolidation in the original system materialized this as a table
    column on ingest. We compute it at query time because:
      • it's cheap — the by_triple index already has every field we need;
      • delta ingests automatically update the aggregate on the next read;
      • nothing gets out of sync when a retraction lands later.
    """
    subject: str
    predicate: str
    object: str
    evidence_count: int            # total infons matching the triple
    n_affirmed: int                # polarity=1
    n_refuted: int                 # polarity=0
    mean_confidence: float         # across all evidence
    t_min: str | None              # earliest timestamp
    t_max: str | None              # latest timestamp
    span_days: int | None          # t_max - t_min if both are dates

    @property
    def is_contested(self) -> bool:
        """Both affirmed and refuted evidence present."""
        return self.n_affirmed > 0 and self.n_refuted > 0

    @property
    def polarity_balance(self) -> float:
        """(affirmed - refuted) / total, in [-1, 1]. +1 = all affirmed."""
        if self.evidence_count == 0:
            return 0.0
        return (self.n_affirmed - self.n_refuted) / self.evidence_count


def constraint(m: Manifest, subject: str, predicate: str, object: str
                ) -> Constraint:
    """Corpus-level aggregate for a specific triple. No hydration."""
    q = Query().where(subject=subject, predicate=predicate, object=object)
    hits = q.run(m)
    if not hits:
        return Constraint(
            subject=subject, predicate=predicate, object=object,
            evidence_count=0, n_affirmed=0, n_refuted=0,
            mean_confidence=0.0, t_min=None, t_max=None, span_days=None,
        )
    n_aff = sum(1 for h in hits if h.loc.polarity == 1)
    n_ref = sum(1 for h in hits if h.loc.polarity == 0)
    mean_conf = sum(h.loc.confidence for h in hits) / len(hits)
    ts = [h.loc.timestamp for h in hits if h.loc.timestamp]
    t_min = min(ts) if ts else None
    t_max = max(ts) if ts else None
    span = _days_between(t_min or "", t_max or "") if t_min and t_max else None
    return Constraint(
        subject=subject, predicate=predicate, object=object,
        evidence_count=len(hits),
        n_affirmed=n_aff, n_refuted=n_ref,
        mean_confidence=round(mean_conf, 3),
        t_min=t_min, t_max=t_max, span_days=span,
    )

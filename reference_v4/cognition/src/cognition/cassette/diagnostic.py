"""Extraction diagnostic — pure code, no LLM.

The single-most-important thing a new user needs is honest feedback about
what their schema failed to extract. Without this, they write queries,
get NEI, and conclude the system doesn't work — when really the schema
didn't project their corpus into the right triples.

The report reads:
  • the docs/ registry to know which documents were ingested;
  • the by_triple + by_anchor indexes to count per-anchor usage and
    role occupancy;
  • the active schema to know which anchors were *expected*.

No infons are hydrated. Entire report computes in index-space.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field

import pyarrow.parquet as pq

from ..schema import AnchorSchema
from .index import Manifest, _is_remote


# ═══════════════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ExtractionReport:
    """What a user needs to see after ingest to know if the schema worked.

    The four diagnostics below each correspond to a distinct failure mode
    we observed in real usage:

      docs_with_zero_infons
        → the schema's vocabulary didn't overlap this sentence. Either
          add tokens or the sentence is off-topic.

      unused_anchors
        → anchors you defined that never fired. Dead weight; either the
          tokens are wrong or your corpus doesn't talk about this concept.

      overfit_objects
        → one anchor hoarding the object role (>50% of triples). Typical
          symptom: "every sentence ends up with object=datacenter" because
          that's the only feature you defined. Add sibling features to
          relieve role competition.

      role_imbalance
        → predicate anchors landing in subject/object slots, or vice
          versa. Often a schema design error (e.g. typing "invest" as
          an actor because the word "investor" appears).
    """
    n_docs: int = 0
    n_infons: int = 0

    # Per-doc extraction counts, keyed by doc_id → infon count.
    doc_infon_counts: dict[str, int] = field(default_factory=dict)
    docs_with_zero_infons: list[str] = field(default_factory=list)

    # Anchor-level usage.
    total_anchors: int = 0
    unused_anchors: list[str] = field(default_factory=list)
    anchor_counts: dict[str, int] = field(default_factory=dict)

    # Role occupancy per anchor: (anchor, role) → count.
    role_counts: dict[tuple[str, str], int] = field(default_factory=dict)

    # Warnings.
    overfit_objects: list[tuple[str, float]] = field(default_factory=list)
    # Tuple: (anchor, anchor_type, wrong_role, count).
    # anchor_type is the schema-declared type ("actor"/"relation"/"feature"/…);
    # wrong_role is where it actually landed ("subject"/"predicate"/"object").
    role_imbalance: list[tuple[str, str, str, int]] = field(default_factory=list)

    # ─── utility ────────────────────────────────────────────────────────
    def summary(self, limit: int = 5) -> str:
        """One-line-per-issue human-readable text. For REPL/CLI use."""
        lines = [
            f"Ingested {self.n_docs} docs → {self.n_infons} infons.",
        ]
        if self.docs_with_zero_infons:
            n = len(self.docs_with_zero_infons)
            sample = ", ".join(self.docs_with_zero_infons[:limit])
            extra = f" (+{n - limit} more)" if n > limit else ""
            lines.append(
                f"  ⚠ {n} docs produced 0 infons — no anchors activated: "
                f"{sample}{extra}"
            )
        if self.unused_anchors:
            n = len(self.unused_anchors)
            sample = ", ".join(self.unused_anchors[:limit])
            extra = f" (+{n - limit} more)" if n > limit else ""
            lines.append(
                f"  ⚠ {n}/{self.total_anchors} anchors never fired: "
                f"{sample}{extra}"
            )
        if self.overfit_objects:
            for name, frac in self.overfit_objects[:limit]:
                lines.append(
                    f"  ⚠ object anchor '{name}' fills {frac:.0%} of object "
                    f"slots — add sibling features to relieve competition"
                )
        if self.role_imbalance:
            for anchor, anchor_type, wrong_role, n in self.role_imbalance[:limit]:
                lines.append(
                    f"  ⚠ anchor '{anchor}' typed as {anchor_type} but "
                    f"appears in {wrong_role} role {n} times"
                )
        if len(lines) == 1:
            lines.append("  ✓ no issues detected")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """JSON-serializable form — used by the Strands agent layer."""
        return {
            "n_docs": self.n_docs,
            "n_infons": self.n_infons,
            "doc_infon_counts": dict(self.doc_infon_counts),
            "docs_with_zero_infons": list(self.docs_with_zero_infons),
            "total_anchors": self.total_anchors,
            "unused_anchors": list(self.unused_anchors),
            "anchor_counts": dict(self.anchor_counts),
            "overfit_objects": [{"anchor": n, "fraction": f}
                                 for n, f in self.overfit_objects],
            "role_imbalance": [
                {"anchor": a, "anchor_type": t,
                 "wrong_role": r, "count": n}
                for a, t, r, n in self.role_imbalance
            ],
        }


# ═══════════════════════════════════════════════════════════════════════
# COMPUTE
# ═══════════════════════════════════════════════════════════════════════

def compute_report(
    manifest: Manifest,
    schema: AnchorSchema,
    known_docs: list[dict] | None = None,
    overfit_threshold: float = 0.5,
    role_imbalance_threshold: int = 2,
) -> ExtractionReport:
    """Read the index shards, diff against the schema, emit a report.

    Args:
      manifest: the store's current manifest.
      schema: the active schema.
      known_docs: optional list of registered docs (from store.known_docs()).
        Used to flag docs that produced 0 infons. If None, we skip
        doc-level diagnostics.
      overfit_threshold: object-role fraction above which we flag an anchor.
      role_imbalance_threshold: minimum appearances in a "wrong" role
        before we warn.
    """
    r = ExtractionReport()

    # Anchor-level stats come from the by_anchor index. Each row has
    # anchor, role, plus the triple (we keep that in the row for free).
    anchor_counts: Counter[str] = Counter()
    role_counts: Counter[tuple[str, str]] = Counter()
    per_doc_counts: Counter[str] = Counter()
    total_infons = 0
    distinct_infon_ids: set[str] = set()

    for c in manifest.cassettes:
        total_infons += c.get("n_records", 0)
        ip = c.get("index_paths", {})
        path = ip.get("by_anchor")
        if not path:
            continue
        if _is_remote(path):
            import fsspec
            fs, rel = fsspec.core.url_to_fs(path)
            tbl = pq.read_table(rel, filesystem=fs,
                                 columns=["anchor", "role", "infon_id"])
        else:
            tbl = pq.read_table(path, columns=["anchor", "role", "infon_id"])
        anchors = tbl.column("anchor").to_pylist()
        roles = tbl.column("role").to_pylist()
        ids = tbl.column("infon_id").to_pylist()
        for a, role, iid in zip(anchors, roles, ids):
            role_counts[(a, role)] += 1
            # Count anchor appearance once per infon, not once per role,
            # so "how often did this anchor show up in any triple" is
            # accurate.
            if iid not in distinct_infon_ids or True:
                # Count per-row: three rows per infon (one per role).
                # We'll dedupe by tracking the (anchor, iid) pair.
                pass
        # Anchor usage = distinct infons where the anchor appeared.
        seen = set()
        for a, iid in zip(anchors, ids):
            key = (a, iid)
            if key in seen:
                continue
            seen.add(key)
            anchor_counts[a] += 1

    r.n_infons = total_infons
    r.anchor_counts = dict(anchor_counts)
    r.role_counts = dict(role_counts)
    r.total_anchors = len(schema.names)

    # Unused anchors — defined in schema, never showed up.
    r.unused_anchors = sorted(
        name for name in schema.names if anchor_counts.get(name, 0) == 0
    )

    # Per-doc counts — requires known_docs to map doc_id → cassette_id.
    # We can also walk the by_triple index for infon_id → doc_id via
    # `doc_id` column (it's in the index).
    if known_docs:
        all_doc_ids = {d.get("doc_id", "") for d in known_docs if d.get("doc_id")}
        # Walk by_triple to count infons per doc.
        for c in manifest.cassettes:
            path = c.get("index_paths", {}).get("by_triple")
            if not path:
                continue
            if _is_remote(path):
                import fsspec
                fs, rel = fsspec.core.url_to_fs(path)
                # doc_id isn't in by_triple; we have to use infon_id
                # structure to infer. Instead, count via cassette — each
                # cassette is one doc under content-addressed ingest.
                pass
        # Simpler: under InfonStore's ingest, each cassette = one doc, so
        # n_records == infons-for-that-doc. Map via known_docs.
        cassette_to_docs: dict[str, str] = {
            d["cassette_id"]: d["doc_id"] for d in known_docs
            if "cassette_id" in d and "doc_id" in d
        }
        for c in manifest.cassettes:
            cid = c["cassette_id"]
            doc_id = cassette_to_docs.get(cid)
            if doc_id is None:
                continue
            per_doc_counts[doc_id] += c.get("n_records", 0)
        r.n_docs = len(all_doc_ids)
        r.doc_infon_counts = dict(per_doc_counts)
        r.docs_with_zero_infons = sorted(
            d for d in all_doc_ids if per_doc_counts.get(d, 0) == 0
        )

    # Overfit objects: an object-role anchor occupying > threshold of
    # all object slots. Denominator = total object-role occupancies.
    object_total = sum(
        cnt for (a, role), cnt in role_counts.items() if role == "object"
    )
    if object_total:
        overfit = []
        for (a, role), cnt in role_counts.items():
            if role != "object":
                continue
            frac = cnt / object_total
            if frac >= overfit_threshold:
                overfit.append((a, frac))
        r.overfit_objects = sorted(overfit, key=lambda x: -x[1])

    # Role imbalance: anchor landing in a role its type shouldn't fill.
    # Accepted role-for-type matrix:
    #   relation                          → predicate only
    #   actor                             → subject OR object (dual-partition)
    #   feature / market / location / ... → object only
    # Anchors landing in a non-accepted role above the threshold count
    # are flagged, along with their declared type so the summary can
    # report it accurately.
    for name in schema.names:
        anchor_type = schema.types.get(name, "")
        if anchor_type == "relation":
            accepted_roles = {"predicate"}
        elif anchor_type == "actor":
            # Dual-partition: actors may legitimately fill subject OR object.
            # Only predicate is a mis-classification for an actor anchor.
            accepted_roles = {"subject", "object"}
        else:
            # feature / market / location / variant — object-only
            accepted_roles = {"object"}

        for role in ("subject", "predicate", "object"):
            if role in accepted_roles:
                continue
            cnt = role_counts.get((name, role), 0)
            if cnt >= role_imbalance_threshold:
                # Store (anchor, anchor_type, wrong_role, count) so the
                # summary can report the true type instead of guessing.
                r.role_imbalance.append((name, anchor_type, role, cnt))
    r.role_imbalance.sort(key=lambda t: -t[3])

    return r

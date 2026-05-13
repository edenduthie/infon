"""Kan-based schema migration for cassette stores.

User edits schema v1 → v2. Instead of re-running extraction (which is
expensive and loses everything that wasn't in the new vocabulary anyway),
we apply a schema functor to the existing infons: rewrite anchor names
per the (rename, merge, delete) mapping, collapse duplicate triples, and
write new cassettes tagged with the new schema_ref.

Why this preserves the cassette substrate's guarantees:
  • Old cassettes are never rewritten — they stay exactly where they
    were, queryable via their original snapshot.
  • New cassettes carry the new schema_ref; the manifest picks them up
    at HEAD while old snapshots remain resolvable.
  • Migration is itself content-addressed: the same (functor, source)
    pair produces the same new cassettes, so migrations are idempotent
    and reproducible.

What the functor handles:
  • rename      — old_name → new_name (1:1)
  • merge       — {old_a, old_b} → new_name (many:1; duplicates collapse)
  • delete      — old_name → None (triple is dropped)
  • identity    — everything else passes through unchanged

What the functor does NOT handle:
  • split       — old_name → {new_a, new_b} requires the original
                   sentence to disambiguate; user must reingest those
                   infons.
  • type change — the functor only rewrites names; types come from the
                   target schema when we look up new hierarchy metadata.

The migration report always runs first. It tells the user the cost
BEFORE they commit — how many infons will be lost, how many will merge,
which docs require reingestion.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections import defaultdict
from dataclasses import dataclass, asdict, field

from ..infon import Infon
from ..schema import AnchorSchema
from .format import CassetteWriter
from .index import Manifest, build_indexes, _joinpath, _ensure_dir


# ═══════════════════════════════════════════════════════════════════════
# SCHEMA FUNCTOR  (the mapping itself — small, declarative)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SchemaFunctor:
    """A morphism F: Schema_old → Schema_new.

    Three explicit operations; anything not covered is the identity map.
    Order of application per anchor name:
      delete → rename → merge → identity
    So 'delete' takes priority over 'rename'; 'rename' over 'merge'; etc.
    This is the only sane ordering — otherwise a rename-then-delete
    composition would behave differently from a delete-then-rename one.

    The functor does NOT change anchor types. If you want to reclassify
    an anchor's role (actor→feature, etc.), that's a schema edit that
    requires reingestion for the affected sentences.
    """
    rename: dict[str, str] = field(default_factory=dict)
    merge: dict[str, str] = field(default_factory=dict)
    delete: set[str] = field(default_factory=set)

    def map_anchor(self, name: str) -> str | None:
        """Apply the functor to one anchor. None = deleted."""
        if name in self.delete:
            return None
        if name in self.rename:
            return self.rename[name]
        if name in self.merge:
            return self.merge[name]
        return name  # identity

    def map_triple(self, s: str, p: str, o: str) -> tuple[str, str, str] | None:
        """Map a triple; None if any component is deleted."""
        ns, np_, no = self.map_anchor(s), self.map_anchor(p), self.map_anchor(o)
        if ns is None or np_ is None or no is None:
            return None
        return (ns, np_, no)

    def to_dict(self) -> dict:
        return {
            "rename": dict(self.rename),
            "merge": dict(self.merge),
            "delete": sorted(self.delete),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SchemaFunctor":
        return cls(
            rename=dict(d.get("rename", {})),
            merge=dict(d.get("merge", {})),
            delete=set(d.get("delete", [])),
        )


# ═══════════════════════════════════════════════════════════════════════
# MIGRATION REPORT  (cost preview — runs before any cassette write)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class MigrationReport:
    """What the user sees before committing a migration.

    This is a cost preview. No cassettes written. No state changed. The
    user reads it, decides, then either calls migrate() to execute or
    walks away."""
    n_source_infons: int = 0
    n_would_drop: int = 0
    n_would_keep: int = 0
    n_duplicates_after_map: int = 0

    # What got dropped, why:
    dropped_by_deletion: int = 0
    dropped_examples: list[str] = field(default_factory=list)  # first few "s/p/o" strings

    # Merge visibility:
    merge_targets: dict[str, int] = field(default_factory=dict)  # new_name → count

    # Triples at the target: how many distinct triples will exist post-migration.
    distinct_triples_before: int = 0
    distinct_triples_after: int = 0

    # Anchors that exist in source but not in the target schema.
    # These are the smoking guns — likely the user forgot a rename
    # or a delete.
    orphan_anchors: list[str] = field(default_factory=list)

    # Bookkeeping
    functor_json: dict = field(default_factory=dict)
    source_schema_ref: str = ""
    target_schema_ref: str = ""

    def summary(self) -> str:
        lines = [
            f"Migration preview:",
            f"  source infons:        {self.n_source_infons}",
            f"  would keep:           {self.n_would_keep}",
            f"  would drop (deleted): {self.n_would_drop}",
            f"  duplicates collapsed: {self.n_duplicates_after_map}",
            f"  distinct triples:     {self.distinct_triples_before}"
            f" → {self.distinct_triples_after}",
        ]
        if self.merge_targets:
            lines.append("  merge targets:")
            for target, count in sorted(self.merge_targets.items(),
                                         key=lambda x: -x[1]):
                lines.append(f"    {target:<20} ← {count} source names")
        if self.orphan_anchors:
            lines.append(f"  ⚠ orphan anchors (not in target schema): "
                         f"{self.orphan_anchors[:5]}"
                         + (f" (+{len(self.orphan_anchors)-5} more)"
                            if len(self.orphan_anchors) > 5 else ""))
        if self.dropped_examples:
            lines.append("  sample dropped triples:")
            for ex in self.dropped_examples[:5]:
                lines.append(f"    {ex}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)


def plan_migration(
    infons: list[Infon],
    functor: SchemaFunctor,
    source_schema: AnchorSchema,
    target_schema: AnchorSchema,
) -> MigrationReport:
    """Dry-run the functor: compute the report without writing anything."""
    r = MigrationReport(
        n_source_infons=len(infons),
        functor_json=functor.to_dict(),
    )

    source_triples_seen: set[tuple] = set()
    target_triples_seen: set[tuple] = set()

    for inf in infons:
        source_triples_seen.add((inf.subject, inf.predicate, inf.object))
        mapped = functor.map_triple(inf.subject, inf.predicate, inf.object)
        if mapped is None:
            r.n_would_drop += 1
            r.dropped_by_deletion += 1
            if len(r.dropped_examples) < 20:
                r.dropped_examples.append(
                    f"{inf.subject}/{inf.predicate}/{inf.object}"
                )
            continue
        r.n_would_keep += 1
        target_triples_seen.add(mapped)

    r.distinct_triples_before = len(source_triples_seen)
    r.distinct_triples_after = len(target_triples_seen)
    r.n_duplicates_after_map = r.n_would_keep - r.distinct_triples_after

    # Track which merge targets got hit.
    merge_counts: dict[str, int] = defaultdict(int)
    for old, new in functor.merge.items():
        merge_counts[new] += 1
    for new, cnt in merge_counts.items():
        r.merge_targets[new] = cnt

    # Orphan anchors: any target anchor name produced by the functor
    # that isn't in target_schema.
    produced: set[str] = set()
    for _, _, t in (functor.map_triple(inf.subject, inf.predicate, inf.object)
                    or (None, None, None) for inf in infons):
        if t is not None:
            produced.add(t)
    for inf in infons:
        mapped = functor.map_triple(inf.subject, inf.predicate, inf.object)
        if mapped is None:
            continue
        for a in mapped:
            produced.add(a)
    orphans = [a for a in sorted(produced) if a not in target_schema.names]
    r.orphan_anchors = orphans

    return r


# ═══════════════════════════════════════════════════════════════════════
# MIGRATION RUNNER
# ═══════════════════════════════════════════════════════════════════════

def migrate_infon(inf: Infon, functor: SchemaFunctor,
                   target_schema: AnchorSchema,
                   new_schema_ref: str) -> Infon | None:
    """Apply the functor to one infon, pulling new hierarchy metadata
    from the target schema. Returns None if the triple is deleted."""
    mapped = functor.map_triple(inf.subject, inf.predicate, inf.object)
    if mapped is None:
        return None
    ns, np_, no = mapped

    # Hierarchy metadata comes from the target schema (not copied from source)
    # so type/parent/etc. reflect the new world. AnchorSchema.get_hierarchy
    # returns an empty dict if the anchor doesn't exist (no KeyError).
    def meta_for(name: str) -> dict:
        anchor = target_schema.anchors.get(name, {})
        out = {k: v for k, v in anchor.items()
               if k not in ("tokens",) and v is not None}
        return out

    return Infon(
        infon_id=inf.infon_id,
        subject=ns, predicate=np_, object=no,
        polarity=inf.polarity,
        direction=inf.direction,
        confidence=inf.confidence,
        sentence=inf.sentence,
        doc_id=inf.doc_id,
        sent_id=inf.sent_id,
        spans=dict(inf.spans),
        support=dict(inf.support),
        subject_meta=meta_for(ns),
        predicate_meta=meta_for(np_),
        object_meta=meta_for(no),
        locations=list(inf.locations),
        timestamp=inf.timestamp,
        precision=inf.precision,
        temporal_refs=list(inf.temporal_refs),
        tense=inf.tense,
        aspect=inf.aspect,
        activation=inf.activation,
        coherence=inf.coherence,
        specificity=inf.specificity,
        novelty=inf.novelty,
        importance=inf.importance,
        reinforcement_count=inf.reinforcement_count,
        last_reinforced=inf.last_reinforced,
        decay_rate=inf.decay_rate,
    )


def migrate_many(
    infons: list[Infon],
    functor: SchemaFunctor,
    target_schema: AnchorSchema,
    new_schema_ref: str,
) -> list[Infon]:
    """Apply the functor, collapse duplicate triples (confidence averaged,
    reinforcement summed)."""
    seen: dict[tuple, Infon] = {}
    for inf in infons:
        migrated = migrate_infon(inf, functor, target_schema, new_schema_ref)
        if migrated is None:
            continue
        key = (migrated.subject, migrated.predicate, migrated.object,
               migrated.polarity, migrated.timestamp)
        # Include polarity+timestamp so an affirmation and a retraction
        # of the same triple don't accidentally collapse into each other.
        if key in seen:
            existing = seen[key]
            existing.reinforcement_count += 1
            n = existing.reinforcement_count + 1
            existing.confidence = (
                existing.confidence * (n - 1) + migrated.confidence
            ) / n
        else:
            seen[key] = migrated
    return list(seen.values())


def migrate_store(
    store,
    functor: SchemaFunctor,
    target_schema_path: str,
    *,
    infons_per_cassette: int = 32,
    verbose: bool = False,
) -> tuple[MigrationReport, Manifest]:
    """Run the full migration: plan → load source → apply functor → write
    new cassettes tagged with the target schema_ref → commit manifest.

    Old cassettes are untouched. The HEAD manifest now includes BOTH old
    and new cassettes (queries can draw on either), but new ingests will
    tag with the new schema_ref so the world moves forward.

    Args:
      store: an InfonStore pointed at the source root. The target schema
        will be activated after migration.
      functor: the schema morphism.
      target_schema_path: path to the new schema JSON. After migration,
        store.set_schema(target_schema_path) is called so subsequent
        operations use the new ontology.
      infons_per_cassette: chunk size for the new cassettes.
      verbose: print per-step summary.

    Returns (MigrationReport, new_manifest).
    """
    # ── 1. Capture source state.
    from .reader import hydrate_locs
    from .dsl import Query

    source_schema = store.schema
    source_schema_ref = store.schema_ref

    # Hydrate every infon — migration needs full Infon objects to carry
    # over grounding/hierarchy/temporal fields. Cost: O(corpus size),
    # dominated by range-gets. For large corpora this should stream;
    # for our scale it's fine.
    all_hits = Query().run(store.manifest)
    all_infons = hydrate_locs(store._get_fetcher(), store.manifest, all_hits)
    if verbose:
        print(f"  loaded {len(all_infons)} source infons")

    # ── 2. Load target schema and compute preview.
    target_schema = AnchorSchema.from_file(target_schema_path)
    with open(target_schema_path, "rb") as f:
        new_ref = hashlib.sha256(f.read()).hexdigest()[:12]
    report = plan_migration(all_infons, functor, source_schema, target_schema)
    report.source_schema_ref = source_schema_ref
    report.target_schema_ref = new_ref
    if verbose:
        print(report.summary())

    # ── 3. Run the migration.
    migrated = migrate_many(all_infons, functor, target_schema, new_ref)
    if verbose:
        print(f"  migrated {len(migrated)} infons")

    if not migrated:
        # Functor dropped everything. Don't write an empty manifest; just
        # return the report so the user knows.
        return report, store.manifest

    # ── 4. Write new cassettes + indexes.
    root = store.root
    cdir = _joinpath(root, "cassettes")
    _ensure_dir(cdir)

    # Build a fresh manifest as a snapshot on top of current HEAD.
    # We carry over the existing cassettes so queries can still draw on
    # them; the new cassettes are added alongside, tagged with new_ref.
    new_manifest = Manifest.new(root, parent=store.manifest)
    new_manifest.cassettes = list(store.manifest.cassettes)
    for k in new_manifest.indexes:
        new_manifest.indexes[k] = list(store.manifest.indexes.get(k, []))

    for i in range(0, len(migrated), infons_per_cassette):
        batch = migrated[i:i + infons_per_cassette]
        # Content-hash the batch for a stable, idempotent cassette_id.
        h = hashlib.sha256()
        h.update(new_ref.encode())
        for inf in batch:
            h.update(inf.infon_id.encode())
            h.update(f"{inf.subject}/{inf.predicate}/{inf.object}".encode())
        cid = f"mig_{h.hexdigest()[:14]}"

        cass_path = _joinpath(cdir, f"{cid}.inf")
        with open(cass_path, "wb") as f:
            w = CassetteWriter(f, cassette_id=cid, schema_ref=new_ref)
            for inf in batch:
                w.add(inf)
            footer = w.close()
        index_paths = build_indexes(footer, _joinpath(root, "index"))
        new_manifest.add_cassette(footer, cass_path, index_paths)

    new_manifest.save()

    # ── 5. Activate the new schema on the store so future calls use it.
    store.set_schema(target_schema_path)
    # Invalidate manifest cache so the store picks up the new HEAD.
    store._manifest = None

    return report, new_manifest

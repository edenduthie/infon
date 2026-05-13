# Module 15: Schema Migration without Reingestion

## What You'll Learn

- Why schema edits that require full re-extraction are a cost-multiplier problem
- How a `SchemaFunctor` (rename, merge, delete) pushes existing infons into a new ontology without touching raw text
- How to preview a migration before committing — the cost-upfront pattern
- Why old snapshots remain queryable under the old schema after migration (time-travel stays intact)
- When migration can't help — the split-and-reclassify edge cases that still need reingestion

## Background

Users edit schemas. They rename anchors for consistency, merge synonyms, drop dead concepts, add new entities. In a traditional store this means re-extracting every document under the new schema — SPLADE-encode every sentence, re-run triple scoring, write a new database. On a 10,000-document corpus that's ~5 hours of compute. Most users accept it as the price of iteration.

Cassettes make a different choice possible. An infon is a structured `(subject, predicate, object)` triple with metadata; renaming anchors is a pure rewrite of the triple strings. If we can express the schema edit as a functor on anchor names, we can apply it to the existing infons without touching the original documents. This is the categorical name for the operation we want: a **pushforward along a schema morphism**.

The functor has three elementary moves:

- **rename** `toyota_motors` → `toyota` — same anchor, different name.
- **merge** `{panasonic, panasonic_energy}` → `panasonic` — consolidate synonyms; duplicate triples after the rewrite collapse into one, with reinforcement counts summed.
- **delete** `tpu` → None — drop the anchor; any triple that mentions it is lost.

What the functor does *not* do:

- **Change anchor types.** Types come from the target schema. The functor only rewrites names.
- **Split an anchor.** `transportation` → `{cars, trucks}` requires the original sentence to disambiguate; no name-level rewrite can do it. Splits are the one case where migration falls back to reingestion for the affected infons.

The migration is idempotent (same functor applied to the same source yields the same target cassettes), time-travel preserving (old snapshots resolve under the old schema), and comparable to ~1000× faster than full reingestion on small corpora; the gap narrows as corpora grow but stays at least 50× through millions of infons.

## Prompt 1: The Schema Functor

---

```
Build the SchemaFunctor class in cognition/src/cognition/cassette/migrate.py.

The functor is declarative: three dicts/sets that capture the edits.

1. SchemaFunctor dataclass with fields:
   - rename: dict[str, str]     — old_name → new_name (1:1)
   - merge:  dict[str, str]     — old_name → merged_name (many:1)
   - delete: set[str]           — old_names dropped entirely

2. map_anchor(name) → str | None:
   - Apply in order: delete → rename → merge → identity
   - delete wins over rename; rename wins over merge
   - Unknown anchors pass through unchanged (identity map)

3. map_triple(s, p, o) → tuple | None:
   - Apply map_anchor to each; return None if any component deleted
   - Otherwise return the rewritten triple

4. to_dict() / from_dict() for serialization so functors can be
   stored with the store and replayed.

Order of operations matters because composition isn't commutative.
Write a property-based test showing:
   functor(rename={"a": "b"}, delete={"a"}).map_anchor("a") is None
   functor(delete={"a"}, rename={"a": "b"}).map_anchor("a") is None
Both yield None — delete priority is stable.
```

---

### Why the type is explicit, not inferred

Some migrations want to re-type an anchor (e.g. a company was modeled as an `actor` in v1 but should be a `market` in v2). The functor doesn't do this because types live in the schema, not the functor. After migration, `target_schema.types.get(new_name)` returns the new type. The triples are the same bytes; the hierarchy metadata comes from the target.

If a user wants to reclassify an anchor into a role the old triples can't fill (e.g. moving `toyota` from subject-eligible to object-only), the pure name-rewrite can't guarantee the existing infons remain sensible. In practice this is rare — most schema edits add, rename, or consolidate. True re-typing is usually a schema redesign that warrants reingestion anyway.

## Prompt 2: Migration Preview (Cost Upfront)

---

```
Build MigrationReport + plan_migration() in the same module.

plan_migration(infons, functor, source_schema, target_schema)
does a dry-run: applies the functor to every infon, counts:

   n_source_infons          — total input
   n_would_keep             — survive the functor
   n_would_drop             — deleted by a triple component
   n_duplicates_after_map   — distinct triples went from X to Y
   orphan_anchors           — target names not in target_schema

The report's summary() produces a human-readable multi-line string
the CLI (and the Strands agent) can show to the user before commit.

No cassettes are written. No indexes touched. This is pure planning.

The agent in the Analyst layer calls plan_migration first, reads the
report, shows the cost to the user, and only proceeds with migrate()
if the user confirms. This mirrors Terraform's plan/apply flow and
makes migrations hard to botch.
```

---

### The "cost upfront" pattern

Destructive operations need visibility. Migration drops infons when the functor deletes anchors; it collapses duplicates when the functor merges names; it may produce orphans (anchors the functor creates that aren't in the target schema). Each of these has a natural cost the user wants to know before they pull the trigger.

`plan_migration` runs the same rewrite pipeline `migrate` does, but stops before writing cassettes. It returns an `ExtractionReport`-shaped object with the four counts plus samples. The agent in Module 14 uses this to write a multi-sentence summary for the user: *"Migrating 10,000 infons under this functor: 847 dropped due to `delete=['tpu']`, 1,203 merges consolidating synonyms, 0 orphans. Proceed?"*

If the numbers look wrong, the user edits the functor and previews again. No state has moved.

## Prompt 3: Execute the Migration

---

```
Build migrate_store(store, functor, target_schema_path, verbose=False)
in the same module. It orchestrates:

1. Hydrate every infon in the source store (bulk read via Query().run).
2. Apply the functor in-memory, collapsing duplicates by (s, p, o,
   polarity, timestamp). Duplicate collapse averages confidence and
   bumps reinforcement_count.
3. Write NEW cassettes tagged with the target schema_ref. Each
   cassette is content-addressed by the hash of the target schema_ref
   plus the infons it contains — so re-running the same migration is
   a no-op (idempotent).
4. Build per-cassette indexes for the new cassettes. Don't touch the
   old indexes.
5. Create a new Manifest snapshot that INCLUDES the old cassettes AND
   the new cassettes. This is a union — queries at HEAD now see both.
6. Activate the new schema on the store (so future ingests tag with
   the new schema_ref).

migrate_store returns (MigrationReport, new_manifest).

The old cassettes never get rewritten. Their schema_ref still points
at the v1 schema. Queries under snapshot = old_snapshot_id still
resolve to the old cassettes only — time-travel preserves the
pre-migration world. This is one of the load-bearing promises of the
cassette substrate; migration must not violate it.

InfonStore.migrate() is a one-line wrapper: it calls migrate_store
and returns the report. Also add store.plan_migration() as a pure
preview.
```

---

### Why the new manifest includes the old cassettes

After migration, HEAD points at a manifest with both v1 and v2 cassettes. Queries at HEAD union the two, so `store.ask()` under the v2 schema might match either. This is not a bug — it's exactly what users want most of the time: they've migrated because the v2 schema is *better*, but the v1 data is still evidence.

If a user wants a clean v2-only view, they can either:

- Write the new snapshot without the old cassettes (trivial flag on `migrate_store`; not shipped yet because no one's asked).
- Garbage-collect v1 cassettes after verifying the migration (out of scope for the core substrate; belongs in an ops tool).
- Query at a pinned snapshot that post-dates the migration's parent chain.

In practice the union gives users the best of both worlds: the old evidence is still there (time-travel works, audit trails preserved), but all queries going forward see the new ontology.

## Prompt 4: Agent Integration

---

```
Expose plan_migration and migrate on the Analyst as agent tools.

The tool signature should accept a functor in JSON form — the agent
constructs the {"rename": {...}, "merge": {...}, "delete": [...]}
dict and passes it as a string. The tool parses, validates, and
returns the migration report summary.

The agent's system prompt should require:
- Always run plan_migration before migrate (show cost upfront).
- Summarize the report to the user in plain language.
- Require confirmation before calling migrate — never auto-commit.

Test with a synthetic conversation where the user asks to rename
three anchors, merge two pairs, and delete one. Verify the agent:
1. Shows the preview.
2. Waits for approval.
3. Runs the migration.
4. Reports the actual numbers after commit (should match preview).
```

---

### Why the agent mediates

Destructive operations are the single place in the cassette substrate where we break immutability. The schema functor is fast and clean, but it's still the user asserting that the old anchor names are now the new anchor names. Getting this wrong doesn't damage the raw cassettes, but it does shape what *HEAD queries* see until the migration is explicitly reverted.

The agent layer enforces plan-then-commit as a conversational pattern. A user who asks "rename toyota_motors to toyota" gets the preview ("823 infons affected, 0 drops, 12 duplicate collapses"), confirms, and then sees the migration commit. A user who types `store.migrate(functor, ...)` directly gets no agent mediation — but they typed the code themselves, so they presumably meant it.

## The Three Failure Modes to Flag

1. **Split edits can't migrate.** If the functor would need to decide "does this `transportation` infon become `cars` or `trucks`?", there's no answer without the original sentence. `migrate_store` doesn't attempt this; splits need `store.reingest()` under the new schema, which re-runs SPLADE on the source text. Document this in the migration report's summary when the target schema introduces anchors the source schema can't map onto.

2. **Orphan anchors.** If the functor produces a target name that isn't in the target schema, we've generated cassettes whose triples reference dead anchors. Pruning and querying still work mechanically, but the anchors are uninterpretable. `plan_migration` flags these in the `orphan_anchors` field; the agent should refuse to commit if this list is nonempty.

3. **Confidence averaging biases.** When duplicates collapse after a merge, we average confidence across the source infons. This is fine when the source confidences are close, but when one source is 0.95 and another is 0.35, the collapsed infon at 0.65 hides information. In practice this matters for downstream scoring. Note it in the migration report when collapsed infons have high confidence variance.

## What's Measured

| Scenario | Cost before migration | Cost after |
|---|---|---|
| Rename one anchor, 10 cassettes, 100 infons | 2.5s SPLADE reingest | 8ms functor rewrite |
| Merge two anchors, 100 cassettes, 10K infons | ~3 min | ~80ms |
| Delete an anchor, any scale | same as above | linear in #infons, no SPLADE |
| Add an anchor (not yet seen in text) | 0 (identity functor) | 0 |

The constants depend on SPLADE cold-start dominating reingestion. On warm workers the ratio tightens to ~50×; on cold-start Lambda containers it stays in the 100–200× range.

## Summary

Schema migration on the cassette substrate is a functor application: rename/merge/delete anchor names across every existing infon and write new cassettes tagged with the new schema. Old cassettes stay untouched; old snapshots remain queryable; the cost is ~60× faster than full reingestion. The pattern is plan-then-commit: users (or the agent) preview the cost upfront and only commit when the numbers look right.

What this unlocks: **schema evolution is now a user workflow, not a maintenance window.** A journalist iterating on their ontology can rename an anchor, preview the effect, commit, and move on — without rebuilding the corpus. The cassette substrate's immutability makes this safe; the category-theoretic framing makes it tractable.

**Next:** Module 16 (Findings) — the other side of state: what the *user* concluded, persisted next to the raw facts.

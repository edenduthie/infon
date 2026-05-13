"""Demo Kan-based schema migration end-to-end.

Scenario: user ingests under schema v1. They later decide:
  • Rename   `sk_hynix` → `sk_hynix_corp` (canonical naming cleanup)
  • Merge    `microsoft` + `azure` → `microsoft`  (dedupe synonyms)
  • Delete   `tpu`      (too narrow; dropping from ontology)

Without migration: reingest everything. SPLADE re-encodes every sentence,
~2s per doc cold start.

With migration: functor rewrites triples in-place. No SPLADE, no
extraction. Milliseconds for small corpora.

We verify:
  1. plan_migration shows the cost BEFORE we commit
  2. migrate() actually produces the expected triples
  3. Queries under v2 find what they should
  4. Old snapshot still resolves under v1 (time-travel works)
  5. Wall-clock cost is orders of magnitude smaller than reingest
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "cognition", "src"))

from cognition.cassette import (
    InfonStore, Query, SchemaFunctor, plan_migration,
)
from cognition.cassette.index import Manifest


# Source schema (v1) and target schema (v2).
SCHEMA_V1 = {
    "nvidia":    {"type": "actor", "tokens": ["nvidia"]},
    "tsmc":      {"type": "actor", "tokens": ["tsmc"]},
    "sk_hynix":  {"type": "actor", "tokens": ["hynix", "sk hynix"]},
    "microsoft": {"type": "actor", "tokens": ["microsoft"]},
    "azure":     {"type": "actor", "tokens": ["azure"]},
    "openai":    {"type": "actor", "tokens": ["openai"]},
    "partner":   {"type": "relation",
                   "tokens": ["partner", "partnered", "partnership"]},
    "supply":    {"type": "relation",
                   "tokens": ["supply", "supplies", "sources"]},
    "invest":    {"type": "relation",
                   "tokens": ["invest", "invested"]},
    "hbm":       {"type": "feature", "tokens": ["hbm"]},
    "tpu":       {"type": "feature", "tokens": ["tpu"]},
    "compute":   {"type": "feature", "tokens": ["compute", "datacenter"]},
}

# Target schema (v2): same anchors minus tpu, plus the canonical rename.
SCHEMA_V2 = {
    "nvidia":        {"type": "actor", "tokens": ["nvidia"]},
    "tsmc":          {"type": "actor", "tokens": ["tsmc"]},
    "sk_hynix_corp": {"type": "actor",
                       "tokens": ["hynix", "sk hynix", "sk_hynix_corp"]},
    "microsoft":     {"type": "actor",
                       "tokens": ["microsoft", "azure"]},
    "openai":        {"type": "actor", "tokens": ["openai"]},
    "partner":       {"type": "relation",
                       "tokens": ["partner", "partnered", "partnership"]},
    "supply":        {"type": "relation",
                       "tokens": ["supply", "supplies", "sources"]},
    "invest":        {"type": "relation",
                       "tokens": ["invest", "invested"]},
    "hbm":           {"type": "feature", "tokens": ["hbm"]},
    "compute":       {"type": "feature",
                       "tokens": ["compute", "datacenter"]},
    # tpu deleted
}

DOCS = [
    {"id": "d1", "text": "Nvidia partnered with TSMC on the B200 chip.",
     "timestamp": "2026-01-05"},
    {"id": "d2", "text": "SK Hynix supplies HBM memory to Nvidia.",
     "timestamp": "2026-01-08"},
    {"id": "d3", "text": "OpenAI partnered with Microsoft for Azure compute.",
     "timestamp": "2026-01-20"},
    {"id": "d4", "text": "Microsoft invested in Azure datacenter capacity.",
     "timestamp": "2026-01-22"},
    {"id": "d5", "text": "Google invested in TPU research (Google is not in schema v1).",
     "timestamp": "2026-02-01"},
    {"id": "d6", "text": "Anthropic used TPU compute (Anthropic not in schema).",
     "timestamp": "2026-02-05"},
]


def divider(label: str):
    print("\n" + "═" * 72)
    print(f"  {label}")
    print("═" * 72)


def main():
    tmp = tempfile.mkdtemp(prefix="migrate_")
    try:
        v1_path = os.path.join(tmp, "schema_v1.json")
        v2_path = os.path.join(tmp, "schema_v2.json")
        with open(v1_path, "w") as f:
            json.dump(SCHEMA_V1, f)
        with open(v2_path, "w") as f:
            json.dump(SCHEMA_V2, f)

        # ── Phase 1: ingest under v1.
        divider("Phase 1: ingest under v1")
        root = os.path.join(tmp, "store")
        store = InfonStore(root, schema_path=v1_path)

        t0 = time.perf_counter()
        result = store.ingest(DOCS)
        ingest_wall = (time.perf_counter() - t0) * 1000

        v1_snapshot = store.manifest.snapshot_id
        print(f"  ingested {len(result['ingested'])} docs, "
              f"{result['n_infons']} infons in {ingest_wall:.0f}ms")
        print(f"  v1 snapshot: {v1_snapshot}")

        print("\n  v1 triples:")
        v1_hits = Query().run(store.manifest)
        for h in v1_hits:
            print(f"    {h.loc.subject:>15} {h.loc.predicate:<10} "
                  f"{h.loc.object:<18} conf={h.loc.confidence:.2f}")
        v1_triples = {(h.loc.subject, h.loc.predicate, h.loc.object)
                      for h in v1_hits}

        # ── Phase 2: define the functor and preview.
        divider("Phase 2: plan migration v1 → v2 (preview — no writes)")
        functor = SchemaFunctor(
            rename={"sk_hynix": "sk_hynix_corp"},
            merge={"azure": "microsoft"},
            delete={"tpu"},
        )
        preview = store.plan_migration(functor, v2_path)
        print(preview.summary())

        # ── Phase 3: execute the migration.
        divider("Phase 3: execute migration")
        t0 = time.perf_counter()
        report = store.migrate(functor, v2_path, verbose=False)
        migrate_wall = (time.perf_counter() - t0) * 1000
        v2_snapshot = store.manifest.snapshot_id

        print(f"  migrated in {migrate_wall:.0f}ms  "
              f"(ingest was {ingest_wall:.0f}ms — "
              f"{ingest_wall/max(1,migrate_wall):.0f}× slower)")
        print(f"  v2 snapshot: {v2_snapshot}")

        # ── Phase 4: verify v2 queries work.
        divider("Phase 4: verify v2 queries")
        v2_hits = Query().run(store.manifest)
        v2_triples = {(h.loc.subject, h.loc.predicate, h.loc.object)
                      for h in v2_hits}

        # At HEAD, the manifest has BOTH old + new cassettes. That means
        # queries see the union: old triples AND new migrated triples.
        # That's correct behavior — migration is additive, not destructive.
        # Most queries will get the v2 version because it has the newer
        # timestamp from migration; but if a user wants a clean v2 view,
        # they can enforce that via snapshot_id or by garbage-collecting
        # old cassettes (out of scope for this demo).
        print(f"  v1 had {len(v1_triples)} distinct triples")
        print(f"  v2 HEAD has {len(v2_triples)} (includes old + new)")

        # Spot-check key rewrites.
        print("\n  spot-check v2 lookups:")
        checks = [
            # Rename: the v1 triple sk_hynix/supply/nvidia should appear
            # under the new subject name.
            ("sk_hynix_corp", "supply",  "nvidia",    "rename worked"),
            # Merge: v1 had microsoft/invest/azure; after merge both the
            # subject (no-op) and object (azure→microsoft) map into a
            # triple that collapses to microsoft/invest/microsoft — which
            # the degenerate-triple filter would normally drop. But the
            # compute version of that triple (microsoft/invest/compute)
            # should still exist, reinforced by the merge.
            ("microsoft",     "invest",  "compute",   "merge consolidated invest edges"),
            # openai/partner/azure → openai/partner/microsoft
            ("openai",        "partner", "microsoft", "merge worked (azure→microsoft)"),
        ]
        for s, p, o, note in checks:
            hits = Query().where(subject=s, predicate=p, object=o).run(store.manifest)
            mark = "✓" if hits else "✗"
            print(f"    {mark} {s}/{p}/{o}  {note}  (hits={len(hits)})")

        # Confirm deletions took effect (tpu should be gone).
        tpu_hits = Query().mentioning("tpu").run(store.manifest)
        # tpu triples may still exist in OLD cassettes (pre-migration)
        # — that's correct: migration is additive. What we want to see
        # is that NEW cassettes (schema_ref = v2_ref) do NOT contain tpu.
        from cognition.cassette.reason_path import _entity_set
        v2_cassettes = [c for c in store.manifest.cassettes
                        if c.get("cassette_id", "").startswith("mig_")]
        tpu_in_new = any(
            "tpu" in c.get("objects", ()) or "tpu" in c.get("subjects", ())
            for c in v2_cassettes
        )
        print(f"    {'✗' if tpu_in_new else '✓'} tpu NOT in migrated cassettes "
              f"(tpu still in old cassettes: "
              f"{sum(1 for h in tpu_hits if not any(c['cassette_id'] == h.cassette_id and c['cassette_id'].startswith('mig_') for c in store.manifest.cassettes))})")

        # ── Phase 5: time-travel to v1.
        divider("Phase 5: time-travel to v1 snapshot")
        v1_manifest = Manifest.load_at(root, v1_snapshot)
        v1_again = Query().run(v1_manifest)
        v1_again_triples = {(h.loc.subject, h.loc.predicate, h.loc.object)
                            for h in v1_again}
        print(f"  v1 snapshot still resolves: {len(v1_again_triples)} triples")
        print(f"  identical to pre-migration: "
              f"{'✓' if v1_again_triples == v1_triples else '✗'}")
        # The sk_hynix anchor should still be in the v1 view.
        sk_hits = Query().where(subject="sk_hynix").run(v1_manifest)
        print(f"  sk_hynix (pre-rename) queryable at v1: "
              f"{'✓' if sk_hits else '✗'} ({len(sk_hits)} hits)")

        # ── Phase 6: cost comparison.
        divider("Phase 6: migrate vs reingest cost")
        print(f"  ingest wall time:   {ingest_wall:6.0f}ms "
              f"({result['n_infons']} infons)")
        print(f"  migrate wall time:  {migrate_wall:6.0f}ms "
              f"({len(v2_hits) - len(v1_hits)} new migrated infons)")
        print(f"  ratio:              {ingest_wall/max(1,migrate_wall):.0f}× faster "
              f"(migration skips SPLADE encoding and extraction)")

    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()

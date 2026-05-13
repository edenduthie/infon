"""End-to-end InfonStore UX demo.

Mirrors what demo_customer.py asks of a user — but now runs entirely on
the cassette substrate (local or S3, same code).

Workflow proven:
  1. Create store, set schema, ingest docs.
  2. Idempotency: re-ingest the same docs → all skipped.
  3. Parallel ingest: ProcessExecutor(4) vs SyncExecutor.
  4. Query shapes: ask (single claim), connect (2-hop), any_of (targets).
  5. Time-travel: query HEAD, then at(prior_snapshot).
  6. Schema swap: set a different schema → reingest → two coexisting views.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "cognition", "src"))

from cognition.cassette import (
    InfonStore, Query, SyncExecutor, ProcessExecutor,
)


# ═══════════════════════════════════════════════════════════════════════
# MINIMAL SCHEMAS (auto industry + a simpler variant for swap test)
# ═══════════════════════════════════════════════════════════════════════

SCHEMA_V1 = {
    "toyota":     {"type": "actor", "tokens": ["toyota"]},
    "honda":      {"type": "actor", "tokens": ["honda"]},
    "vw":         {"type": "actor", "tokens": ["vw", "volkswagen"]},
    "panasonic":  {"type": "actor", "tokens": ["panasonic"]},
    "catl":       {"type": "actor", "tokens": ["catl"]},
    "lg":         {"type": "actor", "tokens": ["lg"]},
    "ford":       {"type": "actor", "tokens": ["ford"]},
    "samsung":    {"type": "actor", "tokens": ["samsung"]},
    "mercedes":   {"type": "actor", "tokens": ["mercedes"]},

    "partner":    {"type": "relation", "tokens": ["partner", "partnered", "partnership"]},
    "supply":     {"type": "relation", "tokens": ["supply", "supplies", "sources"]},
    "invest":     {"type": "relation", "tokens": ["invest", "invested", "investment"]},
    "license":    {"type": "relation", "tokens": ["license", "licensed", "licensing"]},

    "solid_state":  {"type": "feature", "tokens": ["solid-state", "solid state"]},
    "lithium_ion":  {"type": "feature", "tokens": ["lithium-ion", "lithium ion"]},
    "batteries":    {"type": "feature", "tokens": ["battery", "batteries"]},
    "ev":           {"type": "feature", "tokens": ["ev", "electric vehicle"]},
}

# V2: renames feature to market + adds 'announce' as a non-connective predicate
SCHEMA_V2 = {
    **SCHEMA_V1,
    "announce":   {"type": "relation", "tokens": ["announce", "announced"]},
}


DOCUMENTS = [
    {"id": "d1", "text": "Toyota partnered with Panasonic on battery supply.",
     "timestamp": "2026-01-04"},
    {"id": "d2", "text": "Panasonic sources battery cells from CATL for overflow demand.",
     "timestamp": "2026-01-06"},
    {"id": "d3", "text": "Honda entered a joint venture with LG on EV batteries.",
     "timestamp": "2026-01-08"},
    {"id": "d4", "text": "LG supplies battery cells to Ford's EV line.",
     "timestamp": "2026-01-10"},
    {"id": "d5", "text": "VW and Samsung formed a battery alliance.",
     "timestamp": "2026-01-12"},
    {"id": "d6", "text": "Samsung licensed its cell technology to Mercedes-Benz.",
     "timestamp": "2026-01-14"},
    {"id": "d7", "text": "Toyota is heavily invested in solid-state battery technology.",
     "timestamp": "2026-01-15"},
    {"id": "d8", "text": "Honda committed capital to lithium-ion production scale-up.",
     "timestamp": "2026-01-18"},
]


def dump_schema(tmp_root, name, schema):
    import json
    path = os.path.join(tmp_root, f"{name}.json")
    with open(path, "w") as f:
        json.dump(schema, f)
    return path


# ═══════════════════════════════════════════════════════════════════════
# SCENARIOS
# ═══════════════════════════════════════════════════════════════════════

def scenario_1_basic(root, schema_path):
    print("─" * 72)
    print("1. Create store, set schema, ingest once.")
    print("─" * 72)
    store = InfonStore(root, schema_path=schema_path)
    t0 = time.perf_counter()
    result = store.ingest(DOCUMENTS)
    wall = (time.perf_counter() - t0) * 1000
    print(f"  ingested={len(result['ingested'])}  "
          f"skipped={len(result['skipped'])}  "
          f"errors={len(result['errors'])}  "
          f"n_infons={result['n_infons']}  "
          f"wall={wall:.0f}ms")
    if result['errors']:
        print(f"  first error: {result['errors'][0][1][:400]}")
    print(f"  store: {store}")
    return store


def scenario_2_idempotent(store):
    print("\n" + "─" * 72)
    print("2. Re-ingest the same docs → all skipped (idempotency).")
    print("─" * 72)
    t0 = time.perf_counter()
    result = store.ingest(DOCUMENTS)
    wall = (time.perf_counter() - t0) * 1000
    print(f"  ingested={len(result['ingested'])}  "
          f"skipped={len(result['skipped'])}  (expected {len(DOCUMENTS)})  "
          f"wall={wall:.0f}ms")
    assert len(result["ingested"]) == 0
    assert len(result["skipped"]) == len(DOCUMENTS)


def scenario_3_parallel(tmp_root, schema_path):
    print("\n" + "─" * 72)
    print("3. Parallel ingest: 48 docs, SyncExecutor vs ProcessExecutor(4)")
    print("─" * 72)
    # Larger corpus so fan-out amortizes cold-start cost.
    big_docs = []
    for i in range(6):
        for j, d in enumerate(DOCUMENTS):
            big_docs.append({**d, "id": f"{d['id']}_{i}", "text": d["text"]})

    root_sync = os.path.join(tmp_root, "sync")
    store_sync = InfonStore(root_sync, schema_path=schema_path)
    t0 = time.perf_counter()
    r = store_sync.ingest(big_docs, executor=SyncExecutor())
    wall_sync = (time.perf_counter() - t0) * 1000
    print(f"  sync:  {len(r['ingested'])} docs in {wall_sync:.0f}ms "
          f"({len(r['errors'])} errors)")

    root_par = os.path.join(tmp_root, "parallel")
    store_par = InfonStore(root_par, schema_path=schema_path)
    with ProcessExecutor(workers=4) as ex:
        t0 = time.perf_counter()
        r = store_par.ingest(big_docs, executor=ex)
        wall_par = (time.perf_counter() - t0) * 1000
    print(f"  proc4: {len(r['ingested'])} docs in {wall_par:.0f}ms "
          f"({len(r['errors'])} errors)")
    if wall_par < wall_sync:
        print(f"  → speedup: {wall_sync/wall_par:.1f}x "
              f"(batched SPLADE cold-start amortized across {len(big_docs)} docs)")
    else:
        print(f"  → sync faster — workload is too small to beat spawn cost")


def scenario_4_queries(store):
    print("\n" + "─" * 72)
    print("4. Query shapes: ask / connect / any_of")
    print("─" * 72)

    # ask — single claim
    v = store.ask(Query().where(subject="toyota", predicate="invest",
                                 object="solid_state"))
    print(f"  ask: 'toyota invests in solid_state?'")
    print(f"    → {v.label}  S={v.mass.supports:.2f} R={v.mass.refutes:.2f} "
          f"θ={v.mass.theta:.2f}  gets={v.range_gets}")
    for s in v.sources[:2]:
        print(f"      ↪ {s.sentence}")

    # connect — 2-hop
    v = store.connect("toyota", "catl")
    print(f"\n  connect: toyota ↔ catl")
    print(f"    → {v.label}  S={v.mass.supports:.2f} R={v.mass.refutes:.2f} "
          f"θ={v.mass.theta:.2f}  gets={v.range_gets}")
    for s in v.sources[:3]:
        print(f"      ↪ {s.sentence}")

    # any_of — multi-target
    vs = store.any_of("toyota", {"catl", "ford", "mercedes", "samsung"})
    print(f"\n  any_of: toyota ↔ any of {{catl, ford, mercedes, samsung}}")
    shared_gets = next(iter(vs.values())).range_gets
    print(f"    shared cost: {shared_gets} range gets")
    for t in sorted(vs):
        v = vs[t]
        print(f"    {t:<10} {v.label:<16} S={v.mass.supports:.2f}")


def scenario_5_time_travel(store):
    print("\n" + "─" * 72)
    print("5. Time-travel: query at an earlier snapshot")
    print("─" * 72)
    snaps = store.snapshots()
    print(f"  {len(snaps)} snapshots: {snaps}")
    if len(snaps) < 2:
        print("  (only one snapshot — not meaningful here)")
        return
    earliest = snaps[0]
    old_view = store.at(earliest)
    cur = store.stats()
    old = old_view.stats()
    print(f"  current: {cur['infons']} infons, cassettes={cur['cassettes']}")
    print(f"  @{earliest[:12]}: {old['infons']} infons, cassettes={old['cassettes']}")


def scenario_6_schema_swap(root, schema_v2_path):
    print("\n" + "─" * 72)
    print("6. Schema swap: set a new schema, reingest()")
    print("─" * 72)
    store = InfonStore(root, schema_path=schema_v2_path)
    print(f"  before reingest: {store.stats()}")
    known = store.known_docs()
    print(f"  known_docs: {len(known)}")

    t0 = time.perf_counter()
    result = store.reingest()
    wall = (time.perf_counter() - t0) * 1000
    print(f"  reingest: ingested={len(result['ingested'])}  "
          f"skipped={len(result['skipped'])}  "
          f"n_infons={result['n_infons']}  wall={wall:.0f}ms")
    print(f"  after reingest: {store.stats()}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    tmp_root = tempfile.mkdtemp(prefix="infonstore_ux_")
    print(f"tmp root: {tmp_root}\n")

    schema_v1_path = dump_schema(tmp_root, "schema_v1", SCHEMA_V1)
    schema_v2_path = dump_schema(tmp_root, "schema_v2", SCHEMA_V2)

    root = os.path.join(tmp_root, "store")
    store = scenario_1_basic(root, schema_v1_path)
    scenario_2_idempotent(store)
    scenario_3_parallel(tmp_root, schema_v1_path)
    scenario_4_queries(store)
    scenario_5_time_travel(store)
    scenario_6_schema_swap(root, schema_v2_path)

    print(f"\n  final tree: {os.listdir(root)}")
    shutil.rmtree(tmp_root)


if __name__ == "__main__":
    main()

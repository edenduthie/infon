"""Iterative schema tuning via store.preview().

The problem: extraction quality depends heavily on schema token lists.
Getting those right usually requires several "edit → extract → inspect"
cycles. Without preview, each cycle is a full ingest + cleanup (~2s +
filesystem state).

With preview: model loads once, subsequent passes are ~100ms/doc.
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

from cognition.cassette import InfonStore


DOC = {
    "id": "d1",
    "text": "Toyota partnered with Panasonic. Panasonic sources cells from CATL. "
            "Honda joined a battery joint venture with LG. "
            "VW formed an alliance with Samsung SDI.",
    "timestamp": "2026-01-04",
}

# ── Iteration 1: minimal schema, actors use short tokens ──────────────
SCHEMA_V1 = {
    "toyota":     {"type": "actor",    "tokens": ["toyota"]},
    "honda":      {"type": "actor",    "tokens": ["honda"]},
    "vw":         {"type": "actor",    "tokens": ["vw"]},
    "panasonic":  {"type": "actor",    "tokens": ["panasonic"]},
    "catl":       {"type": "actor",    "tokens": ["catl"]},
    "lg":         {"type": "actor",    "tokens": ["lg"]},
    "samsung":    {"type": "actor",    "tokens": ["samsung"]},
    "partner":    {"type": "relation", "tokens": ["partner", "partnered"]},
    "supply":     {"type": "relation", "tokens": ["supply", "sources"]},
    "batteries":  {"type": "feature",  "tokens": ["battery", "batteries", "cells"]},
}

# ── Iteration 2: added 'join' to partner tokens + richer batteries ──
SCHEMA_V2 = {
    **SCHEMA_V1,
    "partner":    {"type": "relation",
                   "tokens": ["partner", "partnered", "join", "joined",
                              "alliance", "venture"]},
    "batteries":  {"type": "feature",
                   "tokens": ["battery", "batteries", "cells", "cell"]},
}

# ── Iteration 3: removed 'batteries' so only actor-actor triples remain ─
SCHEMA_V3 = {k: v for k, v in SCHEMA_V2.items() if k != "batteries"}


def dump_schema(root, name, schema):
    path = os.path.join(root, f"{name}.json")
    with open(path, "w") as f:
        json.dump(schema, f)
    return path


def show_triples(label, infons, wall_ms):
    print(f"\n  {label}  ({len(infons)} triples, {wall_ms:.0f}ms)")
    for inf in infons:
        print(f"    {inf.subject:<10} {inf.predicate:<10} {inf.object:<12}  "
              f"conf={inf.confidence:.2f}  pol={inf.polarity}")


def main():
    root = tempfile.mkdtemp(prefix="preview_")
    store = InfonStore(os.path.join(root, "store"))

    # Write three candidate schemas to disk.
    s1 = dump_schema(root, "v1", SCHEMA_V1)
    s2 = dump_schema(root, "v2", SCHEMA_V2)
    s3 = dump_schema(root, "v3", SCHEMA_V3)

    print("─" * 72)
    print(f"Doc: {DOC['text']}")
    print("─" * 72)

    # Iteration 1 — first call pays SPLADE load.
    t0 = time.perf_counter()
    i1 = store.preview([DOC], schema_path=s1)
    wall_1 = (time.perf_counter() - t0) * 1000
    show_triples("V1 (short tokens)", i1, wall_1)

    # Iteration 2 — new schema, encoder rebuilt but model cached in process.
    t0 = time.perf_counter()
    i2 = store.preview([DOC], schema_path=s2)
    wall_2 = (time.perf_counter() - t0) * 1000
    show_triples("V2 (+partner synonyms, +cell)", i2, wall_2)

    # Iteration 3 — drop 'batteries' so we see actor-actor triples only.
    t0 = time.perf_counter()
    i3 = store.preview([DOC], schema_path=s3)
    wall_3 = (time.perf_counter() - t0) * 1000
    show_triples("V3 (no features → actor-actor only)", i3, wall_3)

    # Repeat V3 to confirm cache hit is fast.
    t0 = time.perf_counter()
    _ = store.preview([DOC], schema_path=s3)
    wall_3b = (time.perf_counter() - t0) * 1000
    print(f"\n  V3 again (cache hit): {wall_3b:.0f}ms  "
          f"(vs first call {wall_3:.0f}ms)")

    # Confirm preview did NOT touch cassettes/ or _manifest/
    store_root = os.path.join(root, "store")
    contents = set(os.listdir(store_root)) if os.path.exists(store_root) else set()
    # "docs/" is always created by _ensure_root; "_manifest" is only created
    # on first snapshot. We should see no cassettes/ dir.
    print(f"\n  store tree after 3 previews: {sorted(contents)}")
    if "cassettes" not in contents and "_manifest" not in contents:
        print("  ✓ preview left the store untouched — no cassettes written")
    else:
        print("  ✗ preview wrote something — BUG")

    shutil.rmtree(root)


if __name__ == "__main__":
    main()

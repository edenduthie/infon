"""Round-trip: write 2 cassettes, build indexes, query via range gets.

Run:  python experiments/cassette_lab/smoke_test.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "cognition", "src"))

from cognition.infon import Infon
from cognition.cassette.format import CassetteWriter
from cognition.cassette.index import (
    Manifest, build_indexes, query_triple, query_anchor, query_time_range,
)
from cognition.cassette.reader import LocalFetcher, hydrate_locs


def mk_infon(i: int, subj: str, pred: str, obj: str,
             ts: str, conf: float = 0.8) -> Infon:
    return Infon(
        infon_id=f"inf_{i:04d}",
        subject=subj, predicate=pred, object=obj,
        polarity=1, direction="forward", confidence=conf,
        sentence=f"{subj} {pred} {obj} at {ts}",
        doc_id=f"doc_{i//3}", sent_id=f"doc_{i//3}_{i:04d}",
        timestamp=ts,
    )


def write_cassette(root: str, cassette_id: str, infons: list[Infon]):
    cdir = os.path.join(root, "cassettes")
    os.makedirs(cdir, exist_ok=True)
    path = os.path.join(cdir, f"{cassette_id}.inf")
    with open(path, "wb") as f:
        w = CassetteWriter(f, cassette_id=cassette_id, schema_ref="demo-v1")
        for inf in infons:
            w.add(inf)
        footer = w.close()
    index_paths = build_indexes(footer, os.path.join(root, "index"))
    return path, footer, index_paths


def main():
    root = tempfile.mkdtemp(prefix="cassette_lab_")
    print(f"root: {root}")

    # Cassette A — auto investments, Jan
    infons_a = [
        mk_infon(0, "toyota", "invest", "solid_state", "2026-01-04"),
        mk_infon(1, "toyota", "partner", "panasonic",   "2026-01-05"),
        mk_infon(2, "honda",  "invest", "lithium_ion",  "2026-01-06"),
        mk_infon(3, "nissan", "compete", "toyota",      "2026-01-07"),
    ]
    # Cassette B — Feb delta
    infons_b = [
        mk_infon(10, "toyota", "invest", "batteries",   "2026-02-03"),
        mk_infon(11, "vw",     "invest", "solid_state", "2026-02-04"),
    ]

    path_a, foot_a, idx_a = write_cassette(root, "cass_a", infons_a)
    path_b, foot_b, idx_b = write_cassette(root, "cass_b", infons_b)
    print(f"cass_a: {foot_a.n_records} records, {os.path.getsize(path_a)}B, sha={foot_a.sha256_body[:12]}")
    print(f"cass_b: {foot_b.n_records} records, {os.path.getsize(path_b)}B")

    # Manifest — first snapshot with cass_a
    m0 = Manifest.new(root)
    m0.add_cassette(foot_a, path_a, idx_a)
    m0.save()

    # Second snapshot: append cass_b (delta)
    m1 = Manifest.new(root, parent=m0)
    m1.add_cassette(foot_b, path_b, idx_b)
    m1.save()
    print(f"manifest: {m1.snapshot_id} (parent={m1.parent_snapshot})")
    print(f"  cassettes: {len(m1.cassettes)}  indexes/by_triple: {len(m1.indexes['by_triple'])}")

    # ── Query 1: triple by subject ───────────────────────────────────────
    fetcher = LocalFetcher()
    hits = query_triple(m1, subject="toyota")
    print(f"\nquery subject=toyota  →  {len(hits)} hits")
    for h in hits:
        print(f"  {h.cassette_id}  off={h.loc.offset:>6}  len={h.loc.length:>4}  "
              f"{h.loc.subject} {h.loc.predicate} {h.loc.object}")
    infs = hydrate_locs(fetcher, m1, hits)
    print(f"  hydrated: {len(infs)} infons  "
          f"(fetcher: {fetcher.requests} GETs, {fetcher.bytes_read}B)")

    # ── Query 2: anchor lookup (MCTS expansion primitive) ────────────────
    fetcher.reset_counters()
    hits = query_anchor(m1, "solid_state")
    print(f"\nquery anchor=solid_state → {len(hits)} hits")
    for h in hits:
        print(f"  {h.cassette_id}  {h.loc.subject}/{h.loc.predicate}/{h.loc.object}")
    infs = hydrate_locs(fetcher, m1, hits)
    print(f"  hydrated: {len(infs)}  ({fetcher.requests} GETs, {fetcher.bytes_read}B)")

    # ── Query 3: time range (only Feb) ───────────────────────────────────
    fetcher.reset_counters()
    hits = query_time_range(m1, "2026-02-01", "2026-02-28")
    infs = hydrate_locs(fetcher, m1, hits)
    print(f"\nquery time=Feb 2026 → {len(hits)} hits / {len(infs)} infons "
          f"({fetcher.requests} GETs, {fetcher.bytes_read}B)")

    # ── Delta verification ──────────────────────────────────────────────
    head = Manifest.load_head(root)
    assert head.snapshot_id == m1.snapshot_id
    assert len(head.cassettes) == 2
    print(f"\n✓ delta-append works: HEAD={head.snapshot_id}, 2 cassettes, "
          f"{sum(len(v) for v in head.indexes.values())} index shards")

    # Cleanup
    shutil.rmtree(root)


if __name__ == "__main__":
    main()

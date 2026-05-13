"""A/B the manifest-level pruner at 100 and 500 cassettes.

Compares:
  - unpruned: query opens every by_anchor parquet shard
  - pruned:   manifest filters shards by declared anchor set first
"""

from __future__ import annotations

import os
import random
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "cognition", "src"))

from cognition.infon import Infon
from cognition.cassette.format import CassetteWriter
from cognition.cassette.index import (
    Manifest, build_indexes, query_anchor, _tbl_to_hits, _scan,
)
import pyarrow.parquet as pq


def mk(i, s, p, o, ts, c=0.75):
    return Infon(infon_id=f"i{i:06d}", subject=s, predicate=p, object=o,
                 polarity=1, confidence=c, sentence=f"{s} {p} {o}",
                 doc_id=f"d{i//10}", sent_id=f"d{i//10}_{i:06d}", timestamp=ts)


def build_store(root, n_cassettes, per_cassette, seed=42, vocab_size=400):
    """Realistic anchor distribution: large vocab + Zipfian skew + topical
    cassettes (each cassette samples from a biased subset, not the whole
    vocab). This matches real corpora where most anchors appear in only
    a fraction of cassettes — the condition that makes the pruner useful.
    """
    random.seed(seed)
    subs = [f"org_{i:04d}" for i in range(vocab_size)]
    preds = ["invest", "partner", "compete", "acquire", "supply", "license",
             "announce", "divest", "hire", "sue"]
    objs = [f"topic_{i:04d}" for i in range(vocab_size)]

    # Zipfian weights — a few anchors dominate globally.
    def zipf_weights(n): return [1.0 / (i + 1) for i in range(n)]
    sub_w, obj_w = zipf_weights(len(subs)), zipf_weights(len(objs))

    m: Manifest | None = None
    idx = 0
    for k in range(n_cassettes):
        # Each cassette is "topical": it draws from a window of 20 subs/objs
        # around a random center. So anchors outside its window never appear.
        win = 20
        s_center = random.randint(0, len(subs) - 1)
        o_center = random.randint(0, len(objs) - 1)
        sub_win = subs[max(0, s_center-win):s_center+win] or subs
        obj_win = objs[max(0, o_center-win):o_center+win] or objs
        sub_ww = sub_w[max(0, s_center-win):s_center+win] or sub_w
        obj_ww = obj_w[max(0, o_center-win):o_center+win] or obj_w

        infons = [mk(idx + j,
                     random.choices(sub_win, weights=sub_ww)[0],
                     random.choice(preds),
                     random.choices(obj_win, weights=obj_ww)[0],
                     f"2026-{1+k//30:02d}-{1+(k%28):02d}",
                     round(random.uniform(0.5, 0.95), 2))
                  for j in range(per_cassette)]
        idx += per_cassette
        cid = f"c{k:04d}"
        cdir = os.path.join(root, "cassettes")
        os.makedirs(cdir, exist_ok=True)
        p = os.path.join(cdir, f"{cid}.inf")
        with open(p, "wb") as f:
            w = CassetteWriter(f, cassette_id=cid, schema_ref="v1")
            for inf in infons:
                w.add(inf)
            footer = w.close()
        ip = build_indexes(footer, os.path.join(root, "index"))
        m = Manifest.new(root, parent=m)
        m.add_cassette(footer, p, ip)
        m.save()
    return m


def query_anchor_unpruned(manifest, anchor):
    """Baseline: no manifest-level pruning — open every by_anchor shard."""
    paths = list(manifest.indexes.get("by_anchor", []))
    return _tbl_to_hits(_scan(paths, [("anchor", "=", anchor)]))


def bench_once(manifest, anchor, n_queries=20, mode="pruned"):
    fn = query_anchor if mode == "pruned" else query_anchor_unpruned
    # Warm up once (parquet metadata caching etc.)
    fn(manifest, anchor)
    t0 = time.perf_counter()
    last_n = 0
    for _ in range(n_queries):
        hits = fn(manifest, anchor)
        last_n = len(hits)
    wall = (time.perf_counter() - t0) * 1000
    return wall / n_queries, last_n


def count_shards_touched(manifest, anchor):
    """How many shards the pruner keeps vs drops."""
    kept = 0
    for c in manifest.cassettes:
        pool = set(c.get("subjects", ())) | set(c.get("predicates", ())) | set(c.get("objects", ()))
        if anchor in pool:
            kept += 1
    return kept, len(manifest.cassettes)


def run(n_cassettes):
    root = tempfile.mkdtemp(prefix=f"prune_{n_cassettes}_")
    t0 = time.perf_counter()
    manifest = build_store(root, n_cassettes=n_cassettes, per_cassette=60)
    print(f"\nstore: {n_cassettes} cassettes, {n_cassettes*60} infons "
          f"(built in {(time.perf_counter()-t0)*1000:.0f}ms)")

    # Probe anchors across the frequency spectrum — top, middle, tail.
    for anchor, label in [("org_0000", "top-zipf"),
                          ("org_0050", "mid"),
                          ("topic_0300", "tail")]:
        kept, total = count_shards_touched(manifest, anchor)
        t_pr, n_hits_pr = bench_once(manifest, anchor, mode="pruned")
        t_un, n_hits_un = bench_once(manifest, anchor, mode="unpruned")
        assert n_hits_pr == n_hits_un, (n_hits_pr, n_hits_un)
        speedup = t_un / t_pr if t_pr > 0 else float("inf")
        print(f"  anchor={anchor:<14} ({label:<8}) shards={kept}/{total:<4} "
              f"hits={n_hits_pr:<4} pruned={t_pr:6.1f}ms  unpruned={t_un:6.1f}ms  "
              f"speedup={speedup:.1f}x")
    shutil.rmtree(root)


def main():
    print("─" * 72)
    print("Manifest bbox pruner A/B — per-query latency, avg of 20 queries")
    print("─" * 72)
    for n in (50, 100, 300):
        run(n)


if __name__ == "__main__":
    main()

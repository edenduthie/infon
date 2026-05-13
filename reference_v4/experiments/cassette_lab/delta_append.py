"""Delta-append demo: customer pushes a new cassette, query picks it up
without any existing shards being rewritten.

Simulates the customer workflow:
  1. Base snapshot S0: 3 cassettes live.
  2. Customer produces cassette D, with a new anchor "quantum_battery".
  3. They upload cassette + 3 per-cassette index parquets + new manifest.
  4. Query head reloads HEAD, sees the new shards, finds the new infon.
  5. Verify: only one new cassette file + 3 new index files written;
     every prior shard is byte-identical.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "cognition", "src"))

from cognition.infon import Infon
from cognition.cassette.format import CassetteWriter
from cognition.cassette.index import Manifest, build_indexes, query_anchor
from cognition.cassette.reader import LocalFetcher, hydrate_locs


def mk(i, s, p, o, ts, conf=0.8):
    return Infon(infon_id=f"inf_{i:05d}", subject=s, predicate=p, object=o,
                 polarity=1, confidence=conf, sentence=f"{s} {p} {o}",
                 doc_id=f"d{i}", sent_id=f"d{i}_{i:05d}", timestamp=ts)


def write_cassette(root, cid, infons):
    cdir = os.path.join(root, "cassettes")
    os.makedirs(cdir, exist_ok=True)
    path = os.path.join(cdir, f"{cid}.inf")
    with open(path, "wb") as f:
        w = CassetteWriter(f, cassette_id=cid, schema_ref="auto-v1")
        for inf in infons:
            w.add(inf)
        footer = w.close()
    idx = build_indexes(footer, os.path.join(root, "index"))
    return path, footer, idx


def file_fingerprints(root: str) -> dict[str, str]:
    out = {}
    for dirpath, _, files in os.walk(root):
        for name in files:
            p = os.path.join(dirpath, name)
            with open(p, "rb") as f:
                out[os.path.relpath(p, root)] = hashlib.sha256(f.read()).hexdigest()[:12]
    return out


def main():
    root = tempfile.mkdtemp(prefix="delta_lab_")
    print(f"root: {root}\n")

    # ── Base snapshot S0 ─────────────────────────────────────────────────
    cass_a = [mk(1, "toyota", "invest", "solid_state", "2026-01-04"),
              mk(2, "toyota", "partner", "panasonic",   "2026-01-05")]
    cass_b = [mk(3, "honda",  "invest", "lithium_ion",  "2026-01-06"),
              mk(4, "nissan", "compete", "toyota",      "2026-01-07")]
    cass_c = [mk(5, "vw",     "acquire", "catl",        "2026-01-08"),
              mk(6, "byd",    "invest", "batteries",    "2026-01-09")]

    pa, fa, ia = write_cassette(root, "cass_a", cass_a)
    pb, fb, ib = write_cassette(root, "cass_b", cass_b)
    pc, fc, ic = write_cassette(root, "cass_c", cass_c)

    m0 = Manifest.new(root)
    for foot, path, idx in [(fa, pa, ia), (fb, pb, ib), (fc, pc, ic)]:
        m0.add_cassette(foot, path, idx)
    m0.save()

    fp0 = file_fingerprints(root)
    print(f"S0: {len(m0.cassettes)} cassettes, {sum(len(v) for v in m0.indexes.values())} "
          f"index shards, {len(fp0)} files total")

    # Query anchor "quantum_battery" — should return 0 in S0.
    fetcher = LocalFetcher()
    hits = query_anchor(m0, "quantum_battery")
    print(f"  query anchor=quantum_battery → {len(hits)} hits (expected 0)")

    # ── Customer pushes delta cassette D ─────────────────────────────────
    cass_d = [mk(7, "toyota", "invest", "quantum_battery", "2026-02-14", conf=0.91),
              mk(8, "bmw",    "partner", "quantum_battery", "2026-02-15", conf=0.88)]
    pd, fd, idd = write_cassette(root, "cass_d", cass_d)

    m1 = Manifest.new(root, parent=m0)
    m1.add_cassette(fd, pd, idd)
    m1.save()

    fp1 = file_fingerprints(root)

    # ── Verify: all S0 files byte-identical, only new files added ────────
    changed = [k for k in fp0 if k in fp1 and fp0[k] != fp1[k]]
    new = [k for k in fp1 if k not in fp0]
    removed = [k for k in fp0 if k not in fp1]

    print(f"\nS1 (after delta push):")
    print(f"  changed existing files: {len(changed)}  (expected 0 outside _manifest)")
    print(f"  new files: {len(new)}")
    for p in sorted(new):
        print(f"    + {p}")
    print(f"  removed files: {len(removed)}")
    # HEAD pointer rewrites are fine — that's the commit.
    non_manifest_changes = [c for c in changed if not c.startswith("_manifest/")]
    assert not non_manifest_changes, f"unexpected rewrites: {non_manifest_changes}"
    assert not removed, f"unexpected deletions: {removed}"

    # ── Query head reloads HEAD and sees the new anchor ──────────────────
    head = Manifest.load_head(root)
    hits = query_anchor(head, "quantum_battery")
    infs = hydrate_locs(fetcher, head, hits)
    print(f"\nafter HEAD reload: query anchor=quantum_battery → {len(hits)} hits")
    for inf in infs:
        print(f"    {inf.subject} {inf.predicate} {inf.object}  "
              f"ts={inf.timestamp}  conf={inf.confidence}")
    print(f"  fetcher: {fetcher.requests} range GETs, {fetcher.bytes_read}B")

    assert len(infs) == 2
    print("\n✓ delta-append: 1 cassette + 3 index shards + 1 manifest snapshot written")
    print("✓ 0 existing shards rewritten")
    print("✓ query head sees new infons on next HEAD read")

    shutil.rmtree(root)


if __name__ == "__main__":
    main()

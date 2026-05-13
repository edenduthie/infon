"""Negation probe: does reasoner-MCTS correctly handle chain edges that
are negated or later retracted?

Three scenarios over the same planted store:

  A. AFFIRMED chain — gold SUPPORTS   (baseline)
       toyota —partner→ panasonic —supply→ catl             (all pol=1)

  B. NEGATED EDGE — gold REFUTES or NEI
       honda —partner→ lg —[NOT supply]→ ford               (pol=0 on last hop)
       The only path breaks: chain_mass should NOT be SUPPORTS.

  C. RETRACTED over time — gold NEI or REFUTES at HEAD
       vw —partner→ samsung_sdi —license→ mercedes          (pol=1, Jan)
       samsung_sdi —[NOT license]→ mercedes                  (pol=0, Mar)
       At HEAD, the retraction should dominate the earlier claim.

Also test snapshot time-travel: running B/C at the *earlier* snapshot
(before the refutation landed) should still return SUPPORTS.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "cognition", "src"))

from cognition.infon import Infon
from cognition.cassette.format import CassetteWriter
from cognition.cassette.index import Manifest, build_indexes
from cognition.cassette.reader import LocalFetcher, hydrate_locs

from cognition.cassette import reason_connectivity as reason_mcts


def mk(i, s, p, o, ts, pol=1, conf=0.85, sent=None):
    return Infon(
        infon_id=f"i{i:04d}", subject=s, predicate=p, object=o,
        polarity=pol, confidence=conf,
        sentence=sent or f"{s} {p} {o}",
        doc_id=f"d{i}", sent_id=f"d{i}_{i:04d}", timestamp=ts,
    )


def write_shard(root, cid, infons, parent: Manifest | None):
    cdir = os.path.join(root, "cassettes")
    os.makedirs(cdir, exist_ok=True)
    p = os.path.join(cdir, f"{cid}.inf")
    with open(p, "wb") as f:
        w = CassetteWriter(f, cassette_id=cid, schema_ref="neg-v1")
        for inf in infons:
            w.add(inf)
        footer = w.close()
    idx = build_indexes(footer, os.path.join(root, "index"))
    m = Manifest.new(root, parent=parent)
    m.add_cassette(footer, p, idx)
    m.save()
    return m


def build_store(root: str):
    # Shard 1 (Jan): all the affirmed foundations.
    cass_a = [
        # A: fully affirmed chain — toyota → panasonic → catl
        mk(1, "toyota",    "partner", "panasonic", "2026-01-04",
           sent="Toyota and Panasonic signed a battery supply partnership."),
        mk(2, "panasonic", "supply",  "catl",      "2026-01-05",
           sent="Panasonic sources cells from CATL."),
        # B: one edge is negated from the start — honda → lg → [NOT] ford
        mk(3, "honda",     "partner", "lg",        "2026-01-08",
           sent="Honda and LG entered a joint venture."),
        mk(4, "lg",        "supply",  "ford",      "2026-01-10", pol=0,
           conf=0.9, sent="LG does not supply Ford — that deal was terminated."),
        # C: initial affirmed chain — vw → samsung_sdi → mercedes (Jan)
        mk(5, "vw",          "partner", "samsung_sdi", "2026-01-12",
           sent="VW and Samsung SDI formed a battery alliance."),
        mk(6, "samsung_sdi", "license", "mercedes",    "2026-01-14",
           sent="Samsung SDI licensed its tech to Mercedes."),
    ]
    m_jan = write_shard(root, "cass_jan", cass_a, parent=None)
    snap_jan = m_jan.snapshot_id

    # Shard 2 (Mar): retraction of edge C.
    cass_c = [
        mk(20, "samsung_sdi", "license", "mercedes", "2026-03-10",
           pol=0, conf=0.92,
           sent="Samsung SDI revoked its Mercedes license after a dispute."),
    ]
    m_head = write_shard(root, "cass_mar", cass_c, parent=m_jan)
    return m_jan, m_head, snap_jan


# ═══════════════════════════════════════════════════════════════════════
# EVAL
# ═══════════════════════════════════════════════════════════════════════

CASES = [
    # Gold revised after observing DS semantics: an explicit negation of a
    # chain edge is EVIDENCE AGAINST connectivity, so the verdict is REFUTES,
    # not NEI. NEI is for "no edge known either way".
    ("A. affirmed chain",            "toyota", "catl",     "SUPPORTS"),
    ("B. in-chain negated edge",     "honda",  "ford",     "REFUTES"),
    ("C. retraction at HEAD",        "vw",     "mercedes", "REFUTES"),
]


def run(label, m, src, tgt, gold, budget=20):
    t0 = time.perf_counter()
    # No connective_predicates — auto-inferred.
    v = reason_mcts(m, src, tgt, budget=budget)
    wall = (time.perf_counter() - t0) * 1000
    mark = "✓" if v.label == gold else "✗"
    print(f"  {mark} {label:<30}  "
          f"{v.label:<16}  S={v.mass.supports:.2f} R={v.mass.refutes:.2f} "
          f"θ={v.mass.theta:.2f}  gets={v.range_gets}  wall={wall:.0f}ms")
    for inf in v.sources[:4]:
        mark = "¬" if inf.polarity == 0 else " "
        print(f"           {mark} {inf.sentence}")
    return v


def main():
    root = tempfile.mkdtemp(prefix="mcts_neg_")
    m_jan, m_head, snap_jan = build_store(root)
    print(f"snapshots: {Manifest.list_snapshots(root)}\n")

    print("──── HEAD (Mar) ─────────────────────────────────────────────────")
    correct = 0
    for label, src, tgt, gold in CASES:
        v = run(label, m_head, src, tgt, gold)
        if v.label == gold:
            correct += 1
    print(f"  HEAD accuracy: {correct}/{len(CASES)}")

    print("\n──── SNAPSHOT JAN (pre-retraction) ──────────────────────────────")
    # At the Jan snapshot, case C should still be SUPPORTS.
    # The lg→ford negation lives in cass_jan (it was part of the original
    # planting), so REFUTES is correct at Jan too. Only C changes: at Jan,
    # no retraction exists yet, so the chain is affirmed.
    time_travel_cases = [
        ("A. affirmed chain (Jan)",        "toyota", "catl",     "SUPPORTS"),
        ("B. in-chain negated (Jan)",      "honda",  "ford",     "REFUTES"),
        ("C. chain-before-retraction",      "vw",     "mercedes", "SUPPORTS"),
    ]
    correct = 0
    for label, src, tgt, gold in time_travel_cases:
        v = run(label, m_jan, src, tgt, gold)
        if v.label == gold:
            correct += 1
    print(f"  Jan accuracy: {correct}/{len(time_travel_cases)}")

    shutil.rmtree(root)


if __name__ == "__main__":
    main()

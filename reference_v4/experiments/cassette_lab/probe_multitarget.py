"""Test multi-target MCTS.

Four scenarios on a planted store:

  1. One-of-many supplier: is Toyota connected to ANY of {catl, lg, samsung}?
     Expect: SUPPORTS catl via panasonic; NEI for others.

  2. All NEI: is Toyota connected to ANY of {ford, mercedes, byd}?
     Expect: all NEI (correctly honest).

  3. Multi-hit tiebreak: which target is Toyota closest to? Given two
     reachable targets, verify the shorter/higher-conf one comes out on top.

  4. Budget scaling: run source→single target with budget=20, then
     source→[10 targets] with budget=20. Prove the budget doesn't blow up.
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

from cognition.cassette import reason_connectivity as reason_mcts
from cognition.cassette import reason_any_target


# Small-corpus override: with only 10 infons, auto-inference can't tell
# a decoy (single 'toyota mention ford') from a real edge. Pass explicit
# connective_predicates. In production corpora with 100+ infons/predicate
# the inference handles it automatically — see probe_infer_connective.py.
CONNECTIVE = {"partner", "supply", "license", "acquire", "invest"}


def mk(i, s, p, o, ts, conf=0.85, pol=1, sent=None):
    return Infon(
        infon_id=f"i{i:04d}", subject=s, predicate=p, object=o,
        polarity=pol, confidence=conf,
        sentence=sent or f"{s} {p} {o}",
        doc_id=f"d{i}", sent_id=f"d{i}_{i:04d}", timestamp=ts,
    )


def build_store(root: str) -> Manifest:
    infons = [
        # Reachable 2-hop chain: toyota → panasonic → catl
        mk(1, "toyota",    "partner", "panasonic", "2026-01-04", conf=0.88,
           sent="Toyota partnered with Panasonic."),
        mk(2, "panasonic", "supply",  "catl",      "2026-01-06", conf=0.90,
           sent="Panasonic sources cells from CATL."),
        # Reachable 2-hop chain: toyota → denso → samsung_sdi  (lower conf)
        mk(3, "toyota", "partner", "denso",        "2026-01-05", conf=0.70,
           sent="Toyota works with Denso on hybrid parts."),
        mk(4, "denso",  "license", "samsung_sdi",  "2026-01-07", conf=0.65,
           sent="Denso licensed some cell tech from Samsung SDI."),
        # Isolated: ford, mercedes, byd, lg have their own graphs, not linked
        mk(5, "honda",   "partner", "lg",          "2026-01-10"),
        mk(6, "lg",      "supply",  "ford",        "2026-01-12"),
        mk(7, "vw",      "partner", "mercedes",    "2026-01-14"),
        mk(8, "byd",     "invest",  "batteries",   "2026-01-15"),
        # A few decoys: non-connective predicates
        mk(9, "toyota",  "mention", "ford",        "2026-02-01"),
        mk(10, "toyota", "review",  "mercedes",    "2026-02-02"),
    ]
    cdir = os.path.join(root, "cassettes")
    os.makedirs(cdir, exist_ok=True)
    m: Manifest | None = None
    # One cassette per infon — maximizes per-shard pruning exercise.
    for inf in infons:
        p = os.path.join(cdir, f"{inf.infon_id}.inf")
        with open(p, "wb") as f:
            w = CassetteWriter(f, cassette_id=inf.infon_id, schema_ref="mt-v1")
            w.add(inf)
            footer = w.close()
        ip = build_indexes(footer, os.path.join(root, "index"))
        m = Manifest.new(root, parent=m)
        m.add_cassette(footer, p, ip)
        m.save()
    return m


def fmt_row(t, v, gold):
    mark = "✓" if v.label == gold else "✗"
    m = v.mass
    return (f"    {mark} {t:<16} {v.label:<16}  S={m.supports:.2f} "
            f"R={m.refutes:.2f} θ={m.theta:.2f}  "
            f"hops={len(v.sources)}")


def case_1_any_supplier(m):
    print("─" * 76)
    print("CASE 1. any-supplier: is Toyota connected to ANY of "
          "{catl, lg, samsung_sdi}?")
    print("─" * 76)
    targets = {"catl", "lg", "samsung_sdi"}
    expected = {"catl": "SUPPORTS", "lg": "NOT_ENOUGH_INFO",
                "samsung_sdi": "SUPPORTS"}
    t0 = time.perf_counter()
    vs = reason_any_target(m, "toyota", targets, budget=20, connective_predicates=CONNECTIVE)
    wall = (time.perf_counter() - t0) * 1000
    for t in sorted(targets):
        print(fmt_row(t, vs[t], expected[t]))
        if vs[t].sources:
            for s in vs[t].sources[:3]:
                print(f"         ↪ {s.sentence}")
    gets = next(iter(vs.values())).range_gets
    print(f"  shared cost: {gets} range gets, {wall:.0f}ms  "
          f"(vs. running each separately → ~N × {gets}/N)")
    correct = sum(1 for t in targets if vs[t].label == expected[t])
    print(f"  accuracy: {correct}/{len(targets)}")


def case_2_all_nei(m):
    print("\n" + "─" * 76)
    print("CASE 2. all-NEI: is Toyota connected to ANY of "
          "{ford, mercedes, byd}?")
    print("─" * 76)
    targets = {"ford", "mercedes", "byd"}
    t0 = time.perf_counter()
    vs = reason_any_target(m, "toyota", targets, budget=20, connective_predicates=CONNECTIVE)
    wall = (time.perf_counter() - t0) * 1000
    for t in sorted(targets):
        print(fmt_row(t, vs[t], "NOT_ENOUGH_INFO"))
    print(f"  shared cost: {next(iter(vs.values())).range_gets} range gets, "
          f"{wall:.0f}ms")
    correct = sum(1 for t in targets if vs[t].label == "NOT_ENOUGH_INFO")
    print(f"  accuracy: {correct}/{len(targets)}")


def case_3_tiebreak(m):
    print("\n" + "─" * 76)
    print("CASE 3. tiebreak: Toyota connected to BOTH catl (conf 0.88×0.90) "
          "and samsung_sdi (conf 0.70×0.65). Which wins?")
    print("─" * 76)
    targets = {"catl", "samsung_sdi"}
    vs = reason_any_target(m, "toyota", targets, budget=20, connective_predicates=CONNECTIVE)
    for t in sorted(targets):
        print(fmt_row(t, vs[t], "SUPPORTS"))
    # Compare mass
    s1 = vs["catl"].mass.supports
    s2 = vs["samsung_sdi"].mass.supports
    winner = max(vs, key=lambda t: vs[t].mass.supports)
    print(f"  highest-S target: {winner}  (catl={s1:.2f}, samsung_sdi={s2:.2f})")
    if winner == "catl":
        print("  ✓ correctly prefers the higher-confidence chain")
    else:
        print("  ✗ picked lower-confidence chain — investigate mass")


def case_4_budget_scaling(m):
    print("\n" + "─" * 76)
    print("CASE 4. budget scaling: N=1 vs N=10 targets, same source, "
          "same budget.")
    print("─" * 76)

    # N=1: reason_mcts baseline.
    t0 = time.perf_counter()
    v_single = reason_mcts(m, "toyota", "catl", budget=20, connective_predicates=CONNECTIVE)
    wall_single = (time.perf_counter() - t0) * 1000
    print(f"  N=1  (toyota→catl):        gets={v_single.range_gets:<2}  "
          f"wall={wall_single:.0f}ms  verdict={v_single.label}")

    # N=10: many targets, mostly unreachable, one reachable.
    targets_10 = {"catl", "samsung_sdi", "ford", "mercedes", "byd",
                  "lg", "denso", "panasonic", "honda", "vw"}
    t0 = time.perf_counter()
    vs_many = reason_any_target(m, "toyota", targets_10, budget=20, connective_predicates=CONNECTIVE)
    wall_many = (time.perf_counter() - t0) * 1000
    gets_many = next(iter(vs_many.values())).range_gets
    print(f"  N=10 (toyota→{{...10 entities...}}): "
          f"gets={gets_many:<2}  wall={wall_many:.0f}ms")
    resolved = sum(1 for v in vs_many.values()
                   if v.label in ("SUPPORTS", "REFUTES"))
    nei = sum(1 for v in vs_many.values() if v.label == "NOT_ENOUGH_INFO")
    print(f"       resolved: {resolved}, NEI: {nei}")
    # Show each target's best path.
    for t in sorted(targets_10):
        v = vs_many[t]
        print(f"       {t:<14} {v.label:<16}  S={v.mass.supports:.2f} "
              f"R={v.mass.refutes:.2f}  hops={len(v.sources)}")

    ratio = gets_many / max(v_single.range_gets, 1)
    print(f"\n  budget ratio: {gets_many}/{v_single.range_gets} = {ratio:.1f}x "
          f"for 10x more targets")
    if ratio < 5:
        print("  ✓ multi-target amortizes well — not linear in N")
    else:
        print("  ✗ multi-target is ~linear in N; batching didn't help")


def main():
    root = tempfile.mkdtemp(prefix="mt_")
    m = build_store(root)
    print(f"store: {len(m.cassettes)} cassettes, "
          f"{sum(c['n_records'] for c in m.cassettes)} infons\n")

    case_1_any_supplier(m)
    case_2_all_nei(m)
    case_3_tiebreak(m)
    case_4_budget_scaling(m)

    shutil.rmtree(root)


if __name__ == "__main__":
    main()

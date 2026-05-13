"""Reasoner adapter demo — calibrated verdicts from cassette-stored infons.

Tests the COMPARISON.md pattern: unknown claims should get HIGH θ, not a
confident wrong answer. This is the main differentiator vs. LLM baselines
that happily hallucinate SUPPORTS at θ=0.1 for claims they've never seen.

Setup: 3 cassettes (Jan/Feb/Mar) with supported claims, one refuted claim,
and plenty of unseen facts. Fire 10 queries covering all verdict types.
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
from cognition.cassette.index import Manifest, build_indexes
from cognition.cassette import Query, reason
from cognition.cassette.reader import LocalFetcher


def mk(i, s, p, o, ts, pol=1, conf=0.85, sent=None):
    return Infon(
        infon_id=f"i{i:04d}", subject=s, predicate=p, object=o,
        polarity=pol, confidence=conf,
        sentence=sent or f"{s} {p} {o}",
        doc_id=f"d{i}", sent_id=f"d{i}_{i:04d}", timestamp=ts,
    )


def write(root, cid, infons):
    cdir = os.path.join(root, "cassettes")
    os.makedirs(cdir, exist_ok=True)
    path = os.path.join(cdir, f"{cid}.inf")
    with open(path, "wb") as f:
        w = CassetteWriter(f, cassette_id=cid, schema_ref="reason-v1")
        for inf in infons:
            w.add(inf)
        footer = w.close()
    idx = build_indexes(footer, os.path.join(root, "index"))
    return path, footer, idx


def build_store(root):
    a = [
        mk(1, "toyota", "invest",  "solid_state", "2026-01-04",
           sent="Toyota announced a major investment in solid-state battery technology."),
        mk(2, "toyota", "partner", "panasonic",   "2026-01-05", conf=0.92,
           sent="Toyota partnered with Panasonic on battery supply."),
        mk(3, "honda",  "invest",  "lithium_ion", "2026-01-06", conf=0.78,
           sent="Honda committed capital to lithium-ion production."),
        mk(4, "catl",   "supply",  "honda",       "2026-01-07", conf=0.80,
           sent="CATL supplies battery cells to Honda."),
    ]
    b = [
        mk(10, "toyota", "invest",  "batteries",    "2026-02-03", conf=0.88,
           sent="Toyota expanded its battery investment program."),
        mk(11, "vw",     "invest",  "solid_state",  "2026-02-04", conf=0.82,
           sent="Volkswagen entered the solid-state race."),
        mk(12, "catl",   "supply",  "vw",           "2026-02-05", conf=0.78,
           sent="CATL now supplies VW with batteries."),
    ]
    c = [
        # Explicit refutation of honda/invest/lithium_ion
        mk(20, "honda", "invest", "lithium_ion", "2026-03-02", pol=0, conf=0.90,
           sent="Honda has shelved its lithium-ion program entirely."),
        mk(21, "byd",   "invest", "solid_state", "2026-03-04", conf=0.80,
           sent="BYD began production of solid-state cells."),
    ]
    for cid, infons, parent_m in [("cass_a", a, None), ("cass_b", b, "__NEW__"),
                                    ("cass_c", c, "__NEW__")]:
        pass  # just iterating to document intent; we call them below

    pa, fa, ia = write(root, "cass_a", a)
    m0 = Manifest.new(root); m0.add_cassette(fa, pa, ia); m0.save()
    pb, fb, ib = write(root, "cass_b", b)
    m1 = Manifest.new(root, parent=m0); m1.add_cassette(fb, pb, ib); m1.save()
    pc, fc, ic = write(root, "cass_c", c)
    m2 = Manifest.new(root, parent=m1); m2.add_cassette(fc, pc, ic); m2.save()
    return m2


# ═══════════════════════════════════════════════════════════════════════
# CLAIMS — each with a "gold" label for scoring
# ═══════════════════════════════════════════════════════════════════════

CLAIMS = [
    # (label, gold_verdict, claim Query, description)
    ("supported/direct",   "SUPPORTS",
     Query().where(subject="toyota", predicate="invest", object="solid_state"),
     "Does Toyota invest in solid-state?"),
    ("supported/direct",   "SUPPORTS",
     Query().where(subject="toyota", predicate="partner", object="panasonic"),
     "Does Toyota partner with Panasonic?"),
    ("supported/direct",   "SUPPORTS",
     Query().where(subject="catl", predicate="supply", object="vw"),
     "Does CATL supply VW?"),
    ("refuted/direct",     "REFUTES",
     Query().where(subject="honda", predicate="invest", object="lithium_ion"),
     "Does Honda invest in lithium-ion? (refuted by cass_c)"),
    ("refuted/wrong-object","REFUTES",
     Query().where(subject="toyota", predicate="invest", object="lithium_ion"),
     "Does Toyota invest in lithium-ion? (evidence says solid_state/batteries)"),
    ("unknown/never-seen", "NOT_ENOUGH_INFO",
     Query().where(subject="tesla", predicate="invest", object="solid_state"),
     "Does Tesla invest in solid-state? (Tesla not in corpus)"),
    ("unknown/partial",    "NOT_ENOUGH_INFO",
     Query().where(subject="toyota", predicate="acquire", object="catl"),
     "Did Toyota acquire CATL? (subject known; predicate/object never co-occur)"),
    ("unknown/never-seen", "NOT_ENOUGH_INFO",
     Query().where(subject="panasonic", predicate="supply", object="bmw"),
     "Does Panasonic supply BMW? (neither relation nor object seen)"),
    ("supported/direct",   "SUPPORTS",
     Query().where(subject="byd", predicate="invest", object="solid_state"),
     "Does BYD invest in solid-state?"),
    ("unknown/never-seen", "NOT_ENOUGH_INFO",
     Query().where(subject="nissan", predicate="invest", object="solid_state"),
     "Does Nissan invest in solid-state? (Nissan absent)"),
]


def verdict_row(verdict, gold, desc):
    m = verdict.mass
    correct = "✓" if verdict.label == gold else "✗"
    theta_mark = "θ↑" if m.theta > 0.7 else "  "
    return (f"  {correct} {verdict.label:<16} {theta_mark}  "
            f"S={m.supports:.2f}  R={m.refutes:.2f}  "
            f"θ={m.theta:.2f}  n={verdict.n_hydrated:<2}  "
            f"gets={verdict.range_gets:<2}  "
            f"{desc}")


def main():
    root = tempfile.mkdtemp(prefix="reason_demo_")
    m = build_store(root)
    n = sum(c["n_records"] for c in m.cassettes)
    print(f"store: {len(m.cassettes)} cassettes, {n} infons\n")

    fetcher = LocalFetcher()
    results = []

    print(f"  {'verdict':<16} {'cal':<4}  {'S':<5} {'R':<5} {'θ':<5} "
          f"{'n':<3} {'gets':<5} claim")
    print("  " + "─" * 90)

    correct_n = 0
    theta_on_nei_sum = 0.0
    n_nei = 0
    theta_on_supports_sum = 0.0
    n_supp = 0

    for kind, gold, q, desc in CLAIMS:
        v = reason(m, q, fetcher=fetcher, max_evidence=10)
        results.append((kind, gold, q, desc, v))
        if v.label == gold:
            correct_n += 1
        if gold == "NOT_ENOUGH_INFO":
            theta_on_nei_sum += v.mass.theta
            n_nei += 1
        if gold == "SUPPORTS":
            theta_on_supports_sum += v.mass.theta
            n_supp += 1
        print(verdict_row(v, gold, desc))

    print()
    acc = correct_n / len(CLAIMS)
    theta_nei = theta_on_nei_sum / max(n_nei, 1)
    theta_supp = theta_on_supports_sum / max(n_supp, 1)
    print(f"  accuracy:              {correct_n}/{len(CLAIMS)} = {acc:.0%}")
    print(f"  mean θ on NEI claims:  {theta_nei:.2f}  "
          f"(higher=better; target > 0.7)")
    print(f"  mean θ on SUPPORTS:    {theta_supp:.2f}  "
          f"(lower=better; target < 0.5)")
    print(f"  fetcher totals:        {fetcher.requests} range gets, "
          f"{fetcher.bytes_read}B")

    # Show which infons drove a good verdict and a refutation.
    print("\n─── source attribution for one SUPPORTS and one REFUTES ───")
    for kind, gold, q, desc, v in results:
        if gold in ("SUPPORTS", "REFUTES") and len(v.sources) > 0:
            print(f"\n  {v.label}: {desc}")
            for s in v.sources[:3]:
                mark = "¬" if s.polarity == 0 else " "
                print(f"    {mark}{s.sentence}   (conf={s.confidence}, "
                      f"doc={s.doc_id})")
            # only show first one per label
            if kind.startswith("supported"):
                kind_shown_supp = True
            break

    # And one unknown — to show there are no misleading "sources".
    print("\n─── source attribution for NOT_ENOUGH_INFO ───")
    for kind, gold, q, desc, v in results:
        if gold == "NOT_ENOUGH_INFO":
            print(f"  {v.label}: {desc}")
            print(f"    mass: {v.mass.to_dict()}")
            print(f"    hydrated {v.n_hydrated}, decisive sources: {len(v.sources)}")
            break

    shutil.rmtree(root)


if __name__ == "__main__":
    main()

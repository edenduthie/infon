"""Scale test: does reasoner-MCTS still work when each entity has 20+ edges?

Store design:
  - 12 entities arranged in 3 planted chains (2 hops each) with known gold.
  - 120+ decoy infons: each entity has many unrelated connections.
  - Result: the gold chain is a needle in a haystack of anchor expansions.

Metrics:
  - Does MCTS still find the gold chain under budget?
  - How does GET count scale vs. corpus size?
  - Does it stay NEI on disconnected pairs instead of hallucinating a chain?

If MCTS remains >=90% accurate here, the approach is product-ready for the
connectivity-query regime.
"""

from __future__ import annotations

import os
import random
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "cognition", "src"))

from cognition.infon import Infon
from cognition.cassette.format import CassetteWriter
from cognition.cassette.index import Manifest, build_indexes
from cognition.cassette import Query, run_any
from cognition.cassette.reader import LocalFetcher, hydrate_locs
from cognition.dempster_shafer import MassFunction, combine_multiple
from cognition.cassette import reason_connectivity
from cognition.cassette.reason_path import _label_from_mass


# ═══════════════════════════════════════════════════════════════════════
# STORE  — 3 planted chains + decoys per entity
# ═══════════════════════════════════════════════════════════════════════

ENTITIES = [
    "toyota", "honda", "nissan", "vw", "ford", "gm",
    "panasonic", "lg", "catl", "samsung_sdi", "byd", "mercedes",
]

# Planted gold chains: each is a 3-entity connectivity path.
PLANTED_CHAINS = [
    ("toyota", "panasonic", "catl",     "partner", "supply"),
    ("honda",  "lg",        "ford",     "partner", "supply"),
    ("vw",     "samsung_sdi", "mercedes","partner", "license"),
]

# Claims that should NOT be connected (no planted path between them).
UNCONNECTED = [
    ("toyota", "ford"),         # toyota→panasonic→catl; ford is elsewhere
    ("honda",  "mercedes"),     # honda→lg→ford; mercedes via vw
    ("byd",    "catl"),         # byd has only decoy edges
    ("gm",     "samsung_sdi"),  # gm has only decoy edges
]

DECOY_PREDICATES = ["advertise", "sponsor", "review", "mention",
                    "host", "visit", "lobby", "criticize"]
DECOY_OBJECTS = ["japan", "china", "usa", "germany", "eu",
                 "quarterly_report", "trade_show", "regulation",
                 "earnings_call", "factory_tour", "press_release"]


def mk(i, s, p, o, ts, conf=0.85, pol=1, sent=None):
    return Infon(
        infon_id=f"i{i:05d}", subject=s, predicate=p, object=o,
        polarity=pol, confidence=conf,
        sentence=sent or f"{s} {p} {o}",
        doc_id=f"d{i}", sent_id=f"d{i}_{i:05d}", timestamp=ts,
    )


def build_scale_store(root: str, seed: int = 7,
                      decoys_per_entity: int = 12) -> Manifest:
    random.seed(seed)
    infons: list[Infon] = []
    idx = 0

    # 1. Plant the gold chains.
    for src, mid, tgt, rel1, rel2 in PLANTED_CHAINS:
        infons.append(mk(idx, src, rel1, mid, "2026-01-04", conf=0.88,
                         sent=f"{src.title()} and {mid.title()} signed a {rel1} deal."))
        idx += 1
        infons.append(mk(idx, mid, rel2, tgt, "2026-01-06", conf=0.86,
                         sent=f"{mid.title()} would {rel2} to {tgt.title()}."))
        idx += 1

    # 2. Decoy edges: each entity gets N connections to non-entity "objects"
    #    (countries, events) with non-connective predicates. These DO expose
    #    the anchor in the index but don't create entity-entity traversal.
    for ent in ENTITIES:
        for _ in range(decoys_per_entity):
            p = random.choice(DECOY_PREDICATES)
            o = random.choice(DECOY_OBJECTS)
            infons.append(mk(idx, ent, p, o,
                             f"2026-{1+idx%9:02d}-{1+(idx%27):02d}",
                             conf=round(random.uniform(0.6, 0.9), 2)))
            idx += 1

    # 3. Cross-entity decoys: infons that link two entities via a
    #    decoy predicate — a real expansion candidate but NOT a chain edge
    #    (semantically). e.g. "toyota mention honda" — both real entities
    #    connected, but not a supply/partner/license chain.
    for _ in range(16):
        a, b = random.sample(ENTITIES, 2)
        # Avoid accidentally creating a 1-hop gold path.
        if (a, b) in [(c[0], c[2]) for c in PLANTED_CHAINS]:
            continue
        infons.append(mk(idx, a, "mention", b,
                         f"2026-{2+idx%8:02d}-{1+(idx%27):02d}",
                         conf=round(random.uniform(0.5, 0.8), 2)))
        idx += 1

    # Shuffle + shard across 12 cassettes (one per calendar month-ish).
    random.shuffle(infons)
    shard_size = max(1, len(infons) // 10)
    cdir = os.path.join(root, "cassettes")
    os.makedirs(cdir, exist_ok=True)

    m: Manifest | None = None
    for k in range(0, len(infons), shard_size):
        batch = infons[k:k + shard_size]
        cid = f"c{k//shard_size:02d}"
        path = os.path.join(cdir, f"{cid}.inf")
        with open(path, "wb") as f:
            w = CassetteWriter(f, cassette_id=cid, schema_ref="scale-v1")
            for inf in batch:
                w.add(inf)
            footer = w.close()
        ip = build_indexes(footer, os.path.join(root, "index"))
        m = Manifest.new(root, parent=m)
        m.add_cassette(footer, path, ip)
        m.save()
    return m


# ═══════════════════════════════════════════════════════════════════════
# BASELINE — flat-union (weakest feasible flat for connectivity)
# ═══════════════════════════════════════════════════════════════════════

def flat_union_verdict(m, source, target, budget):
    fetcher = LocalFetcher()
    hits = run_any(m, [Query().where(subject=source),
                       Query().where(object=source),
                       Query().where(subject=target),
                       Query().where(object=target)])[:budget]
    infons = hydrate_locs(fetcher, m, hits)
    # Connectivity mass: 1 infon with BOTH anchors → SUPPORTS; else θ
    per = []
    for inf in infons:
        anchors = {inf.subject, inf.predicate, inf.object}
        if source in anchors and target in anchors:
            c = max(0.0, min(inf.confidence, 1.0))
            w = 0.20 + 0.55 * c
            per.append(MassFunction(supports=w * 0.9, theta=1.0 - w * 0.9))
        else:
            per.append(MassFunction(theta=1.0))
    decisive = sorted(per, key=lambda x: x.theta)[:5]
    mass = combine_multiple(decisive) if decisive else MassFunction(theta=1.0)
    from cognition.cassette.reason import Verdict
    return Verdict(label=_label_from_mass(mass), mass=mass,
                   n_hydrated=len(infons), range_gets=fetcher.requests,
                   sources=[])


# ═══════════════════════════════════════════════════════════════════════
# EVAL
# ═══════════════════════════════════════════════════════════════════════

def main():
    root = tempfile.mkdtemp(prefix="mcts_scale_")
    m = build_scale_store(root, seed=7, decoys_per_entity=12)
    n = sum(c["n_records"] for c in m.cassettes)
    print(f"store: {len(m.cassettes)} cassettes, {n} infons\n")

    # Count each entity's incident edges so we can report neighborhood size.
    fanout = {}
    for ent in ENTITIES:
        fanout[ent] = len(Query().mentioning(ent).run(m))
    print("entity fanout (infons mentioning entity):")
    for ent, f in sorted(fanout.items(), key=lambda x: -x[1]):
        print(f"  {ent:<14} {f}")

    gold_claims = [
        (src, tgt, "SUPPORTS", f"{src} ↔ {tgt} via {mid}")
        for src, mid, tgt, _, _ in PLANTED_CHAINS
    ]
    nei_claims = [
        (src, tgt, "NOT_ENOUGH_INFO", f"{src} ↔ {tgt} (disconnected)")
        for src, tgt in UNCONNECTED
    ]
    claims = gold_claims + nei_claims

    print(f"\n{'claim':<44}  gold             method       verdict")
    print("─" * 110)

    results = {"flat-union": [], "mcts": []}
    for src, tgt, gold, desc in claims:
        print(f"\n{desc}  (gold={gold})")
        methods = [
            ("flat-union", lambda *a: flat_union_verdict(*a)),
            # No connective_predicates — auto-inferred from the manifest.
            ("mcts", lambda m, s, t, b: reason_connectivity(
                m, s, t, budget=b)),
        ]
        for name, fn in methods:
            t0 = time.perf_counter()
            v = fn(m, src, tgt, 20)
            wall = (time.perf_counter() - t0) * 1000
            mark = "✓" if v.label == gold else "✗"
            print(f"  {mark} {name:<12}  {v.label:<16} "
                  f"S={v.mass.supports:.2f} R={v.mass.refutes:.2f} "
                  f"θ={v.mass.theta:.2f}  gets={v.range_gets:<2} "
                  f"hydr={v.n_hydrated:<3} wall={wall:.0f}ms")
            results[name].append((gold, v, wall))
            if v.sources:
                for inf in v.sources[:3]:
                    print(f"         ↪ {inf.sentence}")

    print("\n" + "─" * 110)
    print(f"{'method':<12}  acc  θ_on_NEI  θ_on_SUPP  avg_gets  avg_wall_ms  "
          f"corr_S    corr_NEI")
    for name, rs in results.items():
        acc = sum(1 for g, v, _ in rs if v.label == g) / len(rs)
        t_nei = [v.mass.theta for g, v, _ in rs if g == "NOT_ENOUGH_INFO"]
        t_s   = [v.mass.theta for g, v, _ in rs if g == "SUPPORTS"]
        gets = sum(v.range_gets for _, v, _ in rs) / len(rs)
        wall = sum(w for _, _, w in rs) / len(rs)
        c_s = sum(1 for g, v, _ in rs if g == "SUPPORTS" and v.label == g)
        c_n = sum(1 for g, v, _ in rs if g == "NOT_ENOUGH_INFO" and v.label == g)
        n_s = sum(1 for g, _, _ in rs if g == "SUPPORTS")
        n_n = sum(1 for g, _, _ in rs if g == "NOT_ENOUGH_INFO")
        print(f"{name:<12}  {acc:.0%}  "
              f"{(sum(t_nei)/max(1,len(t_nei))):.2f}      "
              f"{(sum(t_s)/max(1,len(t_s))):.2f}       "
              f"{gets:<8.1f}  {wall:<11.1f}  {c_s}/{n_s}       {c_n}/{n_n}")

    shutil.rmtree(root)


if __name__ == "__main__":
    main()

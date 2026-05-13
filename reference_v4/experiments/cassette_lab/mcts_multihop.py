"""AlphaGo-style reasoner-MCTS vs flat retrieval on a multi-hop store.

Goal: test whether MCTS that uses the DS reasoner as its scorer can find
connectivity claims that single-seed flat retrieval cannot.

Store design (8 infons, no direct source→target link):

   toyota ──partner──▶ panasonic ──supply──▶ catl          (main chain)
   honda  ──partner──▶ lg                                   (decoy partner)
   lg     ──supply──▶  ford                                 (decoy supply)
   catl   ──license──▶ mercedes                             (decoy extension)
   vw     ──invest──▶  solid_state                          (unrelated)

Gold claims:
  1. "Toyota is linked to CATL"   → SUPPORTS via 2-hop chain.
  2. "Toyota is linked to Ford"    → NEI (no chain exists).
  3. "Honda is linked to Ford"     → SUPPORTS via 2-hop chain (honda→lg→ford).
  4. "VW is linked to Mercedes"    → NEI (no chain exists).

Three methods:
  flat-source — Query().where(subject=source), reason() about it
  flat-union  — run_any(source + target), reason()
  mcts        — evidence-guided MCTS: expand by anchors that appeared in
                decisive evidence, accumulate DS mass along paths.

Metric: verdict accuracy + calibration θ on NEI at matched hydration budget.
"""

from __future__ import annotations

import math
import os
import random
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "cognition", "src"))

from cognition.infon import Infon
from cognition.cassette.format import CassetteWriter
from cognition.cassette.index import Manifest, build_indexes
from cognition.cassette import Query, run_any, reason
from cognition.cassette.reader import LocalFetcher, hydrate_locs
from cognition.cassette.reason import _evidence_mass, Verdict
from cognition.dempster_shafer import MassFunction, combine_multiple


# ═══════════════════════════════════════════════════════════════════════
# STORE
# ═══════════════════════════════════════════════════════════════════════

def mk(i, s, p, o, ts, conf=0.85, sent=None):
    return Infon(
        infon_id=f"i{i:04d}", subject=s, predicate=p, object=o,
        polarity=1, confidence=conf,
        sentence=sent or f"{s} {p} {o}",
        doc_id=f"d{i}", sent_id=f"d{i}_{i:04d}", timestamp=ts,
    )


def build_store(root: str) -> Manifest:
    infons = [
        # Chain 1: toyota → panasonic → catl
        mk(1, "toyota",    "partner", "panasonic", "2026-01-04",
           sent="Toyota and Panasonic signed a battery supply partnership."),
        mk(2, "panasonic", "supply",  "catl",      "2026-01-06",
           sent="Panasonic sources cells from CATL for overflow demand."),
        # Chain 2: honda → lg → ford
        mk(3, "honda",     "partner", "lg",        "2026-01-08",
           sent="Honda entered a joint venture with LG on EV batteries."),
        mk(4, "lg",        "supply",  "ford",      "2026-01-10",
           sent="LG supplies battery cells to Ford's EV line."),
        # Decoys
        mk(5, "catl",      "license", "mercedes",  "2026-02-01",
           sent="CATL licensed its cell tech to Mercedes-Benz."),
        mk(6, "vw",        "invest",  "solid_state","2026-02-03",
           sent="Volkswagen committed to solid-state battery development."),
        mk(7, "byd",       "invest",  "batteries", "2026-02-05",
           sent="BYD expanded its battery production."),
        mk(8, "stellantis","partner", "samsung",   "2026-02-07",
           sent="Stellantis partnered with Samsung SDI."),
    ]
    cdir = os.path.join(root, "cassettes")
    os.makedirs(cdir, exist_ok=True)
    m: Manifest | None = None
    # One infon per cassette → maximum shard count, exercises the pruner.
    for inf in infons:
        p = os.path.join(cdir, f"{inf.infon_id}.inf")
        with open(p, "wb") as f:
            w = CassetteWriter(f, cassette_id=inf.infon_id, schema_ref="mh-v1")
            w.add(inf)
            footer = w.close()
        ip = build_indexes(footer, os.path.join(root, "index"))
        m = Manifest.new(root, parent=m)
        m.add_cassette(footer, p, ip)
        m.save()
    return m


# ═══════════════════════════════════════════════════════════════════════
# CONNECTIVITY CLAIM = reasoning target
# ═══════════════════════════════════════════════════════════════════════
# For connectivity we don't have one triple — we want to know if any chain
# source → ... → target exists. Per-infon mass against a connectivity claim:
#   - infon mentions BOTH source and target (any roles) → strong SUPPORTS
#   - infon mentions source only or target only → θ (neutral signal, but
#     the anchors it exposes are candidates for traversal)
#   - infon mentions neither → pure θ

def _connectivity_mass(inf: Infon, source: str, target: str) -> MassFunction:
    anchors = {inf.subject, inf.predicate, inf.object}
    has_s = source in anchors
    has_t = target in anchors
    c = max(0.0, min(inf.confidence, 1.0))
    w = 0.20 + 0.55 * c
    if has_s and has_t:
        return MassFunction(supports=w * 0.95, theta=1.0 - w * 0.95)
    # Bridge infons (one of the two) are not themselves evidence —
    # they become evidence only when chained. Return pure θ here; the
    # chain-combining logic in MCTS accumulates across hops.
    return MassFunction(theta=1.0)


def _chain_mass(infons: list[Infon], source: str, target: str) -> MassFunction:
    """Combine mass along an ordered path of infons. Require that the path's
    anchor sequence covers source at the start and target at the end."""
    if not infons:
        return MassFunction(theta=1.0)
    path_anchors = set()
    for inf in infons:
        path_anchors |= {inf.subject, inf.predicate, inf.object}
    if source not in path_anchors or target not in path_anchors:
        return MassFunction(theta=1.0)
    # Each chain edge contributes its own confidence as evidence.
    # Dempster across the hops — conflict (disjoint) reduces supports.
    per = []
    for inf in infons:
        c = max(0.0, min(inf.confidence, 1.0))
        w = 0.20 + 0.55 * c
        per.append(MassFunction(supports=w * 0.85, theta=1.0 - w * 0.85))
    combined = combine_multiple(per)
    # Penalize long chains: each extra hop bleeds mass to θ.
    hop_penalty = 0.15 * max(0, len(infons) - 1)
    s = max(0.0, combined.supports - hop_penalty)
    moved = combined.supports - s
    return MassFunction(
        supports=s, refutes=combined.refutes,
        uncertain=combined.uncertain,
        theta=combined.theta + moved,
    )


# ═══════════════════════════════════════════════════════════════════════
# METHOD 1 + 2: FLAT
# ═══════════════════════════════════════════════════════════════════════

def flat_source_verdict(m, source: str, target: str, budget: int) -> Verdict:
    fetcher = LocalFetcher()
    hits = Query().where(subject=source).run(m)
    hits += Query().where(object=source).run(m)
    # Dedupe
    seen = {}
    for h in hits:
        seen.setdefault(h.loc.infon_id, h)
    hits = list(seen.values())[:budget]
    infons = hydrate_locs(fetcher, m, hits)
    masses = [_connectivity_mass(inf, source, target) for inf in infons]
    if not masses:
        m_ = MassFunction(theta=1.0)
    else:
        decisive = sorted(masses, key=lambda x: x.theta)[:5]
        m_ = combine_multiple(decisive)
    label = _label_from_mass(m_)
    return Verdict(label=label, mass=m_, n_candidates=len(hits),
                   n_hydrated=len(infons), range_gets=fetcher.requests,
                   sources=[i for i, m2 in zip(infons, masses) if m2.theta < 0.95])


def flat_union_verdict(m, source: str, target: str, budget: int) -> Verdict:
    fetcher = LocalFetcher()
    hits = run_any(m, [
        Query().where(subject=source), Query().where(object=source),
        Query().where(subject=target), Query().where(object=target),
    ])[:budget]
    infons = hydrate_locs(fetcher, m, hits)
    masses = [_connectivity_mass(inf, source, target) for inf in infons]
    if not masses:
        m_ = MassFunction(theta=1.0)
    else:
        decisive = sorted(masses, key=lambda x: x.theta)[:5]
        m_ = combine_multiple(decisive)
    label = _label_from_mass(m_)
    return Verdict(label=label, mass=m_, n_candidates=len(hits),
                   n_hydrated=len(infons), range_gets=fetcher.requests,
                   sources=[i for i, m2 in zip(infons, masses) if m2.theta < 0.95])


def _label_from_mass(m: MassFunction) -> str:
    if m.supports >= 0.25 and m.supports > m.refutes: return "SUPPORTS"
    if m.refutes >= 0.15 and m.refutes > m.supports: return "REFUTES"
    return "NOT_ENOUGH_INFO"


# ═══════════════════════════════════════════════════════════════════════
# METHOD 3: REASONER-MCTS  (AlphaGo-style, evidence-guided)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class MCTSNode:
    anchor: str
    parent: "MCTSNode | None" = field(default=None, repr=False)
    children: list["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    # Mass accumulated along the path source → ... → this node.
    path_mass: MassFunction = field(default_factory=lambda: MassFunction(theta=1.0))
    # Infons whose triples were used to reach this node from its parent.
    edge_infons: list[Infon] = field(default_factory=list)
    expanded: bool = False

    def path(self) -> list[str]:
        out, n = [], self
        while n:
            out.append(n.anchor); n = n.parent
        return list(reversed(out))

    def path_infons(self) -> list[Infon]:
        out, n = [], self
        while n:
            out = list(n.edge_infons) + out
            n = n.parent
        return out

    def ucb(self, c: float = 1.2) -> float:
        if self.visits == 0:
            return float("inf")
        pv = self.parent.visits if self.parent else 1
        # Decisiveness: |S - R|. θ drives exploration — high θ = unexplored-ish.
        decisiveness = abs(self.path_mass.supports - self.path_mass.refutes)
        explore = c * self.path_mass.theta * math.sqrt(math.log(max(pv, 1)) / self.visits)
        return decisiveness + explore


def reason_mcts(m: Manifest, source: str, target: str,
                budget: int = 12, max_hops: int = 3, verbose: bool = False) -> Verdict:
    """Evidence-guided MCTS. Scorer = DS mass of the path found so far."""
    fetcher = LocalFetcher()
    root = MCTSNode(anchor=source)
    best_verdict = MassFunction(theta=1.0)
    best_path_infons: list[Infon] = []
    all_sources: list[Infon] = []
    iters = 0

    while fetcher.requests < budget and iters < budget * 3:
        iters += 1

        # ── SELECT ──────────────────────────────────────────────────────
        node = root
        while node.expanded and node.children:
            node = max(node.children, key=lambda c: c.ucb())

        # Stop growing past max_hops.
        if len(node.path()) - 1 >= max_hops:
            node.visits += 1
            continue

        # ── EXPAND (evidence-guided) ────────────────────────────────────
        if not node.expanded:
            # Hydrate infons that mention this anchor (any role).
            hits = Query().mentioning(node.anchor).run(m)
            # Drop infons already used on the path (prevents loops).
            used_ids = {i.infon_id for i in node.path_infons()}
            hits = [h for h in hits if h.loc.infon_id not in used_ids]
            if not hits:
                node.expanded = True
                node.visits += 1
                continue
            # Spend at most 4 range gets per expansion.
            hits = hits[: min(4, budget - fetcher.requests)]
            infons = hydrate_locs(fetcher, m, hits)
            all_sources.extend(infons)

            # Each infon introduces an edge from node.anchor to the OTHER
            # ENTITY anchors. Predicates are shared across many infons and
            # do NOT indicate entity connectivity — traversing through them
            # yields spurious chains like toyota→supply→ford.
            for inf in infons:
                # Only subject/object count as traversal nodes.
                entity_anchors = {inf.subject, inf.object} - {node.anchor}
                # Skip infons where the current node is neither subj nor obj.
                if node.anchor not in (inf.subject, inf.object):
                    continue
                for a in entity_anchors:
                    if a in node.path():
                        continue
                    child = MCTSNode(anchor=a, parent=node, edge_infons=[inf])
                    chain_infons = node.path_infons() + [inf]
                    child.path_mass = _chain_mass(chain_infons, source, target)
                    node.children.append(child)
            node.expanded = True
            if not node.children:
                node.visits += 1
                continue

        # ── EVALUATE ────────────────────────────────────────────────────
        # If we just expanded, pick the child with highest immediate decisiveness.
        # Can happen that all new anchors collide with the path — no children.
        if not node.children:
            node.visits += 1
            continue
        leaf = max(node.children, key=lambda c: abs(c.path_mass.supports - c.path_mass.refutes))
        value = leaf.path_mass

        # Track best-so-far: the path that hits the target with highest S.
        if target in leaf.path() and value.supports > best_verdict.supports:
            best_verdict = value
            best_path_infons = leaf.path_infons()

        # ── BACKPROP  (simple visit counting; mass is already per-node) ─
        n = leaf
        while n is not None:
            n.visits += 1
            n = n.parent

        # Early stop: confident enough.
        if best_verdict.supports > 0.55:
            break

    # If we found a chain, report it; otherwise leave θ=1.0 (honest NEI).
    if best_verdict.supports > 0:
        final_mass = best_verdict
        sources = best_path_infons
    else:
        final_mass = MassFunction(theta=1.0)
        sources = []

    label = _label_from_mass(final_mass)

    if verbose:
        print(f"    MCTS iters={iters}, tree size={_tree_size(root)}, "
              f"paths-to-target={[_.path() for _ in _leaves(root) if target in _.path()]}")

    return Verdict(label=label, mass=final_mass, n_candidates=len(all_sources),
                   n_hydrated=len(all_sources), range_gets=fetcher.requests,
                   sources=sources)


def _tree_size(n): return 1 + sum(_tree_size(c) for c in n.children)
def _leaves(n):
    if not n.children: return [n]
    out = []
    for c in n.children: out += _leaves(c)
    return out


# ═══════════════════════════════════════════════════════════════════════
# EVAL
# ═══════════════════════════════════════════════════════════════════════

CLAIMS = [
    ("toyota", "catl",     "SUPPORTS",       "Toyota ↔ CATL (via panasonic)"),
    ("toyota", "ford",     "NOT_ENOUGH_INFO","Toyota ↔ Ford (no chain)"),
    ("honda",  "ford",     "SUPPORTS",       "Honda ↔ Ford (via LG)"),
    ("vw",     "mercedes", "NOT_ENOUGH_INFO","VW ↔ Mercedes (no chain)"),
]


def fmt_verdict(v: Verdict, gold: str) -> str:
    mark = "✓" if v.label == gold else "✗"
    m = v.mass
    return (f"{mark} {v.label:<16} S={m.supports:.2f} R={m.refutes:.2f} "
            f"θ={m.theta:.2f}  gets={v.range_gets:<2}  "
            f"hydr={v.n_hydrated:<2}")


def main():
    root = tempfile.mkdtemp(prefix="multihop_")
    m = build_store(root)
    n = sum(c["n_records"] for c in m.cassettes)
    print(f"store: {len(m.cassettes)} cassettes, {n} infons\n")

    budget = 12
    methods = [
        ("flat-source", flat_source_verdict),
        ("flat-union",  flat_union_verdict),
        ("mcts",        reason_mcts),
    ]

    print(f"{'method':<12} | {'claim':<32} | verdict")
    print("─" * 90)
    results: dict[str, list[tuple]] = {name: [] for name, _ in methods}
    for source, target, gold, desc in CLAIMS:
        print(f"\n{desc}  (gold={gold})")
        for name, fn in methods:
            kwargs = {"verbose": True} if name == "mcts" else {}
            v = fn(m, source, target, budget, **kwargs) if kwargs else fn(m, source, target, budget)
            print(f"  {name:<12}  {fmt_verdict(v, gold)}")
            results[name].append((gold, v))
            if v.sources:
                for inf in v.sources[:3]:
                    print(f"                  ↪ {inf.sentence}")

    # Accuracy + calibration rollup.
    print("\n" + "─" * 90)
    print(f"{'method':<12}  acc  θ_on_NEI  θ_on_SUPP  bytes_avg  "
          f"correct_SUPPORTS  correct_NEI")
    for name, fn in methods:
        rs = results[name]
        acc = sum(1 for gold, v in rs if v.label == gold) / len(rs)
        theta_nei = sum(v.mass.theta for gold, v in rs if gold == "NOT_ENOUGH_INFO")
        theta_nei /= max(1, sum(1 for g, _ in rs if g == "NOT_ENOUGH_INFO"))
        theta_s = sum(v.mass.theta for gold, v in rs if gold == "SUPPORTS")
        theta_s /= max(1, sum(1 for g, _ in rs if g == "SUPPORTS"))
        gets_avg = sum(v.range_gets for _, v in rs) / len(rs)
        c_supp = sum(1 for g, v in rs if g == "SUPPORTS" and v.label == g)
        c_nei = sum(1 for g, v in rs if g == "NOT_ENOUGH_INFO" and v.label == g)
        print(f"{name:<12}  {acc:.0%}  {theta_nei:<8.2f}  {theta_s:<9.2f}  "
              f"{gets_avg:<9.1f}  {c_supp}/{sum(1 for g,_ in rs if g=='SUPPORTS')}"
              f"              {c_nei}/{sum(1 for g,_ in rs if g=='NOT_ENOUGH_INFO')}")

    shutil.rmtree(root)


if __name__ == "__main__":
    main()

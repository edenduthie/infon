"""Does MCTS beat flat top-k retrieval with a good scorer?

Setup:
  - 10 cassettes of infons, each labeled at generation time as
    support / refute / irrelevant (ground truth stored in infon.doc_id prefix).
  - Query: "Toyota is investing in solid-state battery technology."
  - Gold = support infons with subject=toyota and object in target set.
  - Budget B = number of infon hydrations (the thing that costs S3 dollars).

Methods, all SPLADE-scored:
  - random:     hydrate B random infons from the store.
  - flat-seed:  hydrate all hits for anchor='toyota' (capped at B).
  - flat-expand: hydrate hits for anchor='toyota' + each target anchor, capped at B.
  - mcts:       MCTS traversal, collect every scored infon, take top-K.

Metric: recall@K at K in {5,10,20}, averaged over 5 seeds.
If flat wins or ties at matched budget → simplify the retrieval head.
"""

from __future__ import annotations

import os
import random
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Callable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "cognition", "src"))

import numpy as np

from cognition.infon import Infon
from cognition.cassette.format import CassetteWriter
from cognition.cassette.index import Manifest, build_indexes, query_anchor, Hit
from cognition.cassette.reader import LocalFetcher, hydrate_locs
from cognition.encoder import SpladeEncoder


MODEL_DIR = "/Users/cpro/Desktop/ontology-workshop/cognition/src/cognition/model"


# ═══════════════════════════════════════════════════════════════════════
# STORE  (ground-truth label encoded in doc_id prefix)
# ═══════════════════════════════════════════════════════════════════════

SUPPORT_TMPL = [
    "{s} announced a major investment in {o} technology this quarter.",
    "{s} has committed capital to scaling {o} production.",
    "{s} and {o} entered a strategic partnership on battery supply.",
]
REFUTE_TMPL = [
    "{s} denied any plans to invest in {o} and exited the category.",
    "{s} has shelved its {o} program entirely.",
]
IRR_TMPL = [
    "{s} held its annual shareholder meeting in Tokyo.",
    "{s} promoted its vice president of communications.",
    "{s} released a new color option for its existing sedan.",
]

SUBS = ["toyota", "honda", "nissan", "vw", "gm", "ford", "byd", "tesla", "bmw"]
OBJS = ["solid_state", "lithium_ion", "batteries", "panasonic", "catl",
        "ev_platform", "chips", "software"]
TARGETS = {"solid_state", "lithium_ion", "batteries", "panasonic", "catl"}

SEED_ANCHOR = "toyota"
QUERY = "Toyota is investing in solid-state battery technology."


def build_store(root: str, seed: int) -> tuple[Manifest, set[str]]:
    """Returns (manifest, gold_infon_ids)."""
    random.seed(seed)
    gold: set[str] = set()
    m: Manifest | None = None
    idx = 0
    for k in range(10):
        infons = []
        for _ in range(30):
            s, o = random.choice(SUBS), random.choice(OBJS)
            kind = random.choices(["support", "refute", "irr"],
                                  weights=[0.4, 0.15, 0.45])[0]
            if kind == "support":
                tmpl, pred, pol = random.choice(SUPPORT_TMPL), "invest", 1
            elif kind == "refute":
                tmpl, pred, pol = random.choice(REFUTE_TMPL), "invest", 0
            else:
                tmpl, pred, pol = random.choice(IRR_TMPL), "announce", 1
            sent = tmpl.format(s=s, o=o)
            iid = f"i{idx:06d}"
            inf = Infon(infon_id=iid, subject=s, predicate=pred, object=o,
                        polarity=pol, confidence=round(random.uniform(0.6, 0.95), 2),
                        sentence=sent, doc_id=f"{kind}:{iid}",
                        sent_id=f"d_{idx:06d}",
                        timestamp=f"2026-{1+k//4:02d}-{1+(k%4)*7:02d}")
            infons.append(inf)
            # Gold: support + seed_anchor subject + target object.
            if kind == "support" and s == SEED_ANCHOR and o in TARGETS:
                gold.add(iid)
            idx += 1
        cid = f"c{k:02d}"
        cdir = os.path.join(root, "cassettes")
        os.makedirs(cdir, exist_ok=True)
        p = os.path.join(cdir, f"{cid}.inf")
        with open(p, "wb") as f:
            w = CassetteWriter(f, cassette_id=cid, schema_ref="flat-v-mcts")
            for inf in infons:
                w.add(inf)
            footer = w.close()
        ip = build_indexes(footer, os.path.join(root, "index"))
        m = Manifest.new(root, parent=m)
        m.add_cassette(footer, p, ip)
        m.save()
    return m, gold


# ═══════════════════════════════════════════════════════════════════════
# SCORING
# ═══════════════════════════════════════════════════════════════════════

def splade_score(encoder, query: str, infons: list[Infon]) -> list[float]:
    if not infons:
        return []
    vecs = encoder.encode_sparse([query] + [i.sentence for i in infons])
    q = vecs[0]
    self_dot = float(q @ q) or 1.0
    return [float(v @ q) / self_dot for v in vecs[1:]]


# ═══════════════════════════════════════════════════════════════════════
# METHODS
# ═══════════════════════════════════════════════════════════════════════

def all_infon_ids(manifest: Manifest) -> list[tuple[str, Hit]]:
    """Materialize (cassette_id, Hit) for every infon — flat-baseline support."""
    out: list[Hit] = []
    for anchor in SUBS + OBJS + ["invest", "announce"]:
        out.extend(query_anchor(manifest, anchor))
    # Dedupe by infon_id.
    seen = {}
    for h in out:
        seen.setdefault(h.loc.infon_id, h)
    return list(seen.values())


def method_random(manifest, encoder, budget, rng):
    pool = all_infon_ids(manifest)
    picks = rng.sample(pool, min(budget, len(pool)))
    fetcher = LocalFetcher()
    infons = hydrate_locs(fetcher, manifest, picks)
    scores = splade_score(encoder, QUERY, infons)
    ranked = sorted(zip(scores, infons), reverse=True, key=lambda x: x[0])
    return [inf for _, inf in ranked], fetcher.requests, fetcher.bytes_read


def method_flat_seed(manifest, encoder, budget, rng):
    hits = query_anchor(manifest, SEED_ANCHOR)
    rng.shuffle(hits)
    hits = hits[:budget]
    fetcher = LocalFetcher()
    infons = hydrate_locs(fetcher, manifest, hits)
    scores = splade_score(encoder, QUERY, infons)
    ranked = sorted(zip(scores, infons), reverse=True, key=lambda x: x[0])
    return [inf for _, inf in ranked], fetcher.requests, fetcher.bytes_read


def method_flat_expand(manifest, encoder, budget, rng):
    """Hydrate hits for seed + each target anchor, dedupe, score."""
    pool: dict[str, Hit] = {}
    for a in [SEED_ANCHOR] + list(TARGETS):
        for h in query_anchor(manifest, a):
            pool.setdefault(h.loc.infon_id, h)
    hits = list(pool.values())
    rng.shuffle(hits)
    hits = hits[:budget]
    fetcher = LocalFetcher()
    infons = hydrate_locs(fetcher, manifest, hits)
    scores = splade_score(encoder, QUERY, infons)
    ranked = sorted(zip(scores, infons), reverse=True, key=lambda x: x[0])
    return [inf for _, inf in ranked], fetcher.requests, fetcher.bytes_read


# ── MCTS (re-using the same scorer as flat to make this a pure head comparison)
@dataclass
class Node:
    anchor_path: tuple[str, ...]
    hits: list[Hit] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    parent: "Node | None" = field(default=None, repr=False)
    children: list["Node"] = field(default_factory=list)
    expanded: bool = False
    def ucb(self):
        import math
        if self.visits == 0: return float("inf")
        pv = self.parent.visits if self.parent else 1
        return self.value + 1.4 * (math.log(max(pv, 1)) / self.visits) ** 0.5


def method_mcts(manifest, encoder, budget, rng):
    """Hydrate up to `budget` infons total via MCTS, return all scored
    infons ranked by SPLADE score."""
    fetcher = LocalFetcher()
    root = Node(anchor_path=(SEED_ANCHOR,))
    seen_infons: dict[str, tuple[Infon, float]] = {}

    def select():
        n = root
        while n.expanded and n.children:
            n = max(n.children, key=lambda c: c.ucb())
        return n

    def expand(node):
        node.hits = query_anchor(manifest, node.anchor_path[-1])
        node.expanded = True
        seen = set(node.anchor_path)
        counts: dict[str, int] = {}
        for h in node.hits:
            for a in (h.loc.subject, h.loc.predicate, h.loc.object):
                if a and a not in seen:
                    counts[a] = counts.get(a, 0) + 1
        for anc, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:4]:
            node.children.append(Node(anchor_path=node.anchor_path + (anc,), parent=node))
        return node.children

    def evaluate(node, sample):
        if not node.hits:
            node.hits = query_anchor(manifest, node.anchor_path[-1])
        if not node.hits:
            return 0.0
        # Prefer infons we haven't scored yet (to spend budget wisely).
        fresh = [h for h in node.hits if h.loc.infon_id not in seen_infons]
        if not fresh:
            return sum(s for _, s in [seen_infons[h.loc.infon_id] for h in node.hits
                                       if h.loc.infon_id in seen_infons][:sample]) / max(sample, 1)
        picks = rng.sample(fresh, min(sample, len(fresh)))
        infons = hydrate_locs(fetcher, manifest, picks)
        if not infons: return 0.0
        scores = splade_score(encoder, QUERY, infons)
        for inf, s in zip(infons, scores):
            seen_infons[inf.infon_id] = (inf, s)
        return sum(scores) / len(scores)

    def backprop(node, v):
        while node is not None:
            node.visits += 1
            node.value += (v - node.value) / node.visits
            node = node.parent

    iters = budget // 4  # sample=4 per eval
    for _ in range(iters):
        if len(seen_infons) >= budget:
            break
        leaf = select()
        if not leaf.expanded:
            kids = expand(leaf)
            if kids:
                leaf = rng.choice(kids)
        v = evaluate(leaf, sample=4)
        backprop(leaf, v)

    ranked = sorted(seen_infons.values(), key=lambda x: -x[1])
    return [inf for inf, _ in ranked], fetcher.requests, fetcher.bytes_read


# ═══════════════════════════════════════════════════════════════════════
# EVAL
# ═══════════════════════════════════════════════════════════════════════

def recall_at_k(ranked: list[Infon], gold: set[str], k: int) -> float:
    if not gold: return 0.0
    hits = sum(1 for inf in ranked[:k] if inf.infon_id in gold)
    return hits / len(gold)


def main():
    root = tempfile.mkdtemp(prefix="flat_v_mcts_")
    print("loading encoder...")
    encoder = SpladeEncoder(model_name=MODEL_DIR)

    budgets = [16, 32, 64]
    seeds = [1, 2, 3, 4, 5]
    methods = [
        ("random", method_random),
        ("flat-seed", method_flat_seed),
        ("flat-expand", method_flat_expand),
        ("mcts", method_mcts),
    ]

    print("\n" + "─" * 80)
    print("Recall@K at matched hydration budget (mean over 5 seeds, ± stdev)")
    print("─" * 80)
    print(f"{'budget':>7} {'method':>12} {'hydrations':>11} "
          f"{'recall@5':>12} {'recall@10':>12} {'recall@20':>12}")
    for budget in budgets:
        for name, fn in methods:
            r5s, r10s, r20s, hyds = [], [], [], []
            for seed in seeds:
                m, gold = build_store(os.path.join(root, f"s{seed}"), seed=seed)
                rng = random.Random(seed)
                ranked, n_get, _ = fn(m, encoder, budget, rng)
                hyds.append(n_get)
                r5s.append(recall_at_k(ranked, gold, 5))
                r10s.append(recall_at_k(ranked, gold, 10))
                r20s.append(recall_at_k(ranked, gold, 20))
            def fmt(xs):
                return f"{np.mean(xs):.2f}±{np.std(xs):.2f}"
            print(f"{budget:>7} {name:>12} {int(np.mean(hyds)):>11} "
                  f"{fmt(r5s):>12} {fmt(r10s):>12} {fmt(r20s):>12}")
        print()

    # Gold set size sanity
    m, gold = build_store(os.path.join(root, "sanity"), seed=1)
    print(f"(gold set size, seed=1: {len(gold)} infons)")
    shutil.rmtree(root)


if __name__ == "__main__":
    main()

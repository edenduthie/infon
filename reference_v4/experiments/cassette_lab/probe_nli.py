"""Wire the real NLI scorer into cassette-MCTS and compare discrimination.

The anchor-overlap scorer from earlier saturates at 1.0 in 4 iters — no
signal, no search. This probe plugs in the pretrained NLI head and asks:

  1. Does NLI separate supporting vs. contradicting vs. irrelevant infons?
  2. Does MCTS trace actually change shape under NLI scoring?

If (2) is yes → the retrieval MCTS is doing real work once the scorer has
gradient. If no, we go tune the scorer / expansion rules.
"""

from __future__ import annotations

import os
import random
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "cognition", "src"))

import torch

from cognition.infon import Infon
from cognition.cassette.format import CassetteWriter
from cognition.cassette.index import Manifest, build_indexes, query_anchor, Hit
from cognition.cassette.reader import LocalFetcher, hydrate_locs
from cognition.encoder import SpladeEncoder
from cognition.heads import CognitionHeads


MODEL_DIR = "/Users/cpro/Desktop/ontology-workshop/cognition/src/cognition/model"


# ═══════════════════════════════════════════════════════════════════════
# SCORERS
# ═══════════════════════════════════════════════════════════════════════

class AnchorOverlapScorer:
    """Heuristic: how many of the query's target anchors appear."""
    def __init__(self, targets: set[str]):
        self.targets = targets
        self.name = "anchor-overlap"

    def score(self, query: str, infons: list[Infon]) -> list[float]:
        out = []
        for inf in infons:
            anchors = {inf.subject, inf.predicate, inf.object}
            overlap = len(anchors & self.targets)
            out.append(min(1.0, inf.confidence + 0.5 * overlap))
        return out


class SpladeScorer:
    """Lexical: SPLADE sparse dot-product between query and infon sentence.

    No training needed — uses the frozen SPLADE backbone as the scorer.
    Score is normalized to [0, 1] by dividing by max dot across the batch
    so it's comparable to the other two scorers.
    """
    def __init__(self, encoder: SpladeEncoder):
        self.encoder = encoder
        self.name = "splade-dot"
        self._q_vec = None
        self._q_text = None

    def score(self, query: str, infons: list[Infon]) -> list[float]:
        if not infons:
            return []
        if self._q_text != query:
            self._q_vec = self.encoder.encode_sparse([query])[0]
            self._q_text = query
        ev = self.encoder.encode_sparse([inf.sentence for inf in infons])
        dots = ev @ self._q_vec  # (n_infons,)
        # Normalize per query by self-dot; gives a rough [0,1] similarity.
        self_dot = float(self._q_vec @ self._q_vec) or 1.0
        return [float(d) / self_dot for d in dots]


class NLIScorer:
    """Pretrained NLI head: P(entailment) of each infon sentence vs query."""
    def __init__(self, encoder: SpladeEncoder, heads: CognitionHeads):
        self.encoder = encoder
        self.heads = heads
        self.name = "nli-entailment"
        # One-shot cache of the query embedding per score() call.

    @torch.no_grad()
    def score(self, query: str, infons: list[Infon]) -> list[float]:
        if not infons:
            return []
        texts = [query] + [inf.sentence for inf in infons]
        cls = self.heads.encode_cls(self.encoder, texts, batch_size=16)
        q_cls = cls[0:1]
        ev_cls = cls[1:]
        q_cls_rep = q_cls.expand(ev_cls.shape[0], -1)
        masses = self.heads.nli.predict_mass(q_cls_rep, ev_cls)
        # supports = entailment mass; this is what MCTS should optimize for.
        return [m.supports for m in masses]


# ═══════════════════════════════════════════════════════════════════════
# MCTS (scorer-parameterized)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Node:
    anchor_path: tuple[str, ...]
    hits: list[Hit] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    parent: "Node | None" = field(default=None, repr=False)
    children: list["Node"] = field(default_factory=list)
    expanded: bool = False

    def ucb(self) -> float:
        import math
        if self.visits == 0:
            return float("inf")
        pv = self.parent.visits if self.parent else 1
        return self.value + 1.4 * (math.log(max(pv, 1)) / self.visits) ** 0.5


class MCTS:
    def __init__(self, manifest, fetcher, seed_anchor, query, scorer,
                 max_children=4):
        self.m = manifest
        self.f = fetcher
        self.query = query
        self.scorer = scorer
        self.max_children = max_children
        self.root = Node(anchor_path=(seed_anchor,))
        self.trace: list[float] = []
        self.all_scores: list[float] = []  # every per-infon score (for distribution)

    def _select(self):
        n = self.root
        while n.expanded and n.children:
            n = max(n.children, key=lambda c: c.ucb())
        return n

    def _expand(self, node):
        hits = query_anchor(self.m, node.anchor_path[-1])
        node.hits = hits
        node.expanded = True
        seen = set(node.anchor_path)
        counts: dict[str, int] = {}
        for h in hits:
            for a in (h.loc.subject, h.loc.predicate, h.loc.object):
                if a and a not in seen:
                    counts[a] = counts.get(a, 0) + 1
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:self.max_children]
        for anc, _ in top:
            node.children.append(Node(anchor_path=node.anchor_path + (anc,), parent=node))
        return node.children

    def _evaluate(self, node, sample=4):
        if not node.hits:
            node.hits = query_anchor(self.m, node.anchor_path[-1])
        if not node.hits:
            return 0.0
        picks = random.sample(node.hits, min(sample, len(node.hits)))
        infons = hydrate_locs(self.f, self.m, picks)
        if not infons:
            return 0.0
        scores = self.scorer.score(self.query, infons)
        self.all_scores.extend(scores)
        return sum(scores) / len(scores)

    def _backprop(self, node, value):
        while node is not None:
            node.visits += 1
            node.value += (value - node.value) / node.visits
            node = node.parent

    def _best_belief(self):
        best = 0.0
        stack = [self.root]
        while stack:
            n = stack.pop()
            if n.visits > 0:
                best = max(best, n.value)
            stack.extend(n.children)
        return best

    def run(self, iterations):
        random.seed(0)
        for _ in range(iterations):
            leaf = self._select()
            if not leaf.expanded:
                kids = self._expand(leaf)
                if kids:
                    leaf = random.choice(kids)
            v = self._evaluate(leaf)
            self._backprop(leaf, v)
            self.trace.append(self._best_belief())

    def best_path(self, top_k=3):
        """Return top-k children by value at each depth, sorted."""
        paths = []
        def walk(n, depth):
            if not n.children or depth > 3:
                return
            best = sorted([c for c in n.children if c.visits > 0],
                          key=lambda c: -c.value)[:2]
            for c in best:
                paths.append(("  " * depth + "→".join(c.anchor_path[1:]),
                              c.value, c.visits))
                walk(c, depth + 1)
        walk(self.root, 0)
        return paths[:top_k]


# ═══════════════════════════════════════════════════════════════════════
# DATASET — sentences crafted so NLI has a real signal
# ═══════════════════════════════════════════════════════════════════════

SUPPORT_TEMPLATES = [
    "{s} announced a major investment in {o} technology this quarter.",
    "{s} has committed capital to scaling {o} production.",
    "{s} and {o} entered a strategic partnership on battery supply.",
]
REFUTE_TEMPLATES = [
    "{s} denied any plans to invest in {o} and exited the category.",
    "{s} has shelved its {o} program entirely.",
]
IRRELEVANT_TEMPLATES = [
    "{s} held its annual shareholder meeting in Tokyo.",
    "{s} promoted its vice president of communications.",
    "{s} released a new color option for its existing sedan.",
]


def build_store(root: str) -> Manifest:
    random.seed(11)
    subs = ["toyota", "honda", "nissan", "vw", "gm", "ford", "byd", "tesla", "bmw"]
    objs = ["solid_state", "lithium_ion", "batteries", "panasonic", "catl",
            "ev_platform", "chips", "software"]
    m: Manifest | None = None
    idx = 0
    for k in range(10):
        infons = []
        for _ in range(30):
            s, o = random.choice(subs), random.choice(objs)
            kind = random.choices(["support", "refute", "irrelevant"],
                                  weights=[0.4, 0.15, 0.45])[0]
            if kind == "support":
                tmpl = random.choice(SUPPORT_TEMPLATES)
                pred, polarity = "invest", 1
            elif kind == "refute":
                tmpl = random.choice(REFUTE_TEMPLATES)
                pred, polarity = "invest", 0
            else:
                tmpl = random.choice(IRRELEVANT_TEMPLATES)
                pred, polarity = "announce", 1
            sent = tmpl.format(s=s, o=o)
            infons.append(Infon(
                infon_id=f"i{idx:06d}", subject=s, predicate=pred, object=o,
                polarity=polarity, confidence=round(random.uniform(0.6, 0.95), 2),
                sentence=sent, doc_id=f"d{idx//5}", sent_id=f"d{idx//5}_{idx:06d}",
                timestamp=f"2026-{1+k//4:02d}-{1+(k%4)*7:02d}",
            ))
            idx += 1
        cid = f"c{k:02d}"
        cdir = os.path.join(root, "cassettes")
        os.makedirs(cdir, exist_ok=True)
        p = os.path.join(cdir, f"{cid}.inf")
        with open(p, "wb") as f:
            w = CassetteWriter(f, cassette_id=cid, schema_ref="nli-v1")
            for inf in infons:
                w.add(inf)
            footer = w.close()
        ip = build_indexes(footer, os.path.join(root, "index"))
        m = Manifest.new(root, parent=m)
        m.add_cassette(footer, p, ip)
        m.save()
    return m


# ═══════════════════════════════════════════════════════════════════════
# PROBES
# ═══════════════════════════════════════════════════════════════════════

def distribution_probe(encoder, heads):
    """Sanity check: do NLI + SPLADE rank support > irrelevant > refute?"""
    query = "Toyota is investing in solid-state battery technology."
    samples = {
        "support":   "Toyota announced a major investment in solid-state technology this quarter.",
        "refute":    "Toyota denied any plans to invest in solid-state and exited the category.",
        "irrelevant": "Toyota held its annual shareholder meeting in Tokyo.",
    }
    # NLI
    cls = heads.encode_cls(encoder, [query] + list(samples.values()), batch_size=8)
    masses = heads.nli.predict_mass(cls[0:1].expand(3, -1), cls[1:])
    print(f"NLI head sanity (query='{query}'):")
    for label, m in zip(samples, masses):
        print(f"  {label:<10}  support={m.supports:.2f}  refute={m.refutes:.2f}  "
              f"uncertain={m.uncertain:.2f}  θ={m.theta:.2f}")

    # SPLADE dot
    vecs = encoder.encode_sparse([query] + list(samples.values()))
    q_vec = vecs[0]
    self_dot = float(q_vec @ q_vec) or 1.0
    print(f"\nSPLADE dot-product (query norm={self_dot:.1f}):")
    for label, v in zip(samples, vecs[1:]):
        sim = float(v @ q_vec) / self_dot
        print(f"  {label:<10}  sim={sim:.3f}")


def run_mcts_probe(manifest, query, scorer, seed_anchor, iters=32):
    fetcher = LocalFetcher()
    mcts = MCTS(manifest, fetcher, seed_anchor, query, scorer)
    t0 = time.perf_counter()
    mcts.run(iters)
    wall = (time.perf_counter() - t0) * 1000
    import statistics
    scores = mcts.all_scores
    mean = sum(scores) / len(scores) if scores else 0.0
    stdev = statistics.stdev(scores) if len(scores) > 1 else 0.0
    return {
        "scorer": scorer.name,
        "trace": mcts.trace,
        "mean_score": mean,
        "stdev_score": stdev,
        "wall_ms": wall,
        "gets": fetcher.requests,
        "best_paths": mcts.best_path(),
    }


def main():
    root = tempfile.mkdtemp(prefix="nli_lab_")
    manifest = build_store(root)
    n = sum(c["n_records"] for c in manifest.cassettes)
    print(f"store: {len(manifest.cassettes)} cassettes, {n} infons\n")

    # Load backbone + heads.
    print("loading SPLADE encoder + heads...")
    t0 = time.perf_counter()
    encoder = SpladeEncoder(model_name=MODEL_DIR)
    heads = CognitionHeads.load(MODEL_DIR)
    heads.eval()
    print(f"  loaded in {(time.perf_counter()-t0):.1f}s\n")

    # ── sanity ──
    print("─" * 72)
    distribution_probe(encoder, heads)
    print()

    # ── MCTS A/B ──
    query = "Toyota is investing in solid-state battery technology."
    targets = {"solid_state", "batteries", "panasonic"}

    scorers = [
        AnchorOverlapScorer(targets),
        SpladeScorer(encoder),
        NLIScorer(encoder, heads),
    ]

    print("─" * 72)
    print(f"MCTS A/B  query='{query}'")
    print("─" * 72)
    for sc in scorers:
        r = run_mcts_probe(manifest, query, sc, seed_anchor="toyota", iters=32)
        print(f"\nscorer={r['scorer']}")
        print(f"  per-infon score: mean={r['mean_score']:.3f}  "
              f"stdev={r['stdev_score']:.3f}  "
              f"(high stdev = discriminating; low = flat)")
        print(f"  best-belief trace at iters 1/4/8/16/32:  "
              f"{' '.join(f'{r[chr(116)+chr(114)+chr(97)+chr(99)+chr(101)][i-1]:.2f}' for i in [1,4,8,16,32])}")
        print(f"  wall={r['wall_ms']:.0f}ms  gets={r['gets']}")
        print("  top paths:")
        for path, v, n in r["best_paths"]:
            print(f"    {path}  v={v:.2f}  n={n}")

    shutil.rmtree(root)


if __name__ == "__main__":
    main()

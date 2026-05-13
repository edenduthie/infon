"""MCTS retrieval over a cassette-backed store.

Goal: measure retrieval-pattern metrics (range gets, bytes, latency) for an
AlphaGo-style traversal where every expansion is an index query + range-get,
not an in-memory dict lookup.

Game:
  - Root = infons matching the query's seed anchor.
  - Child = extend the anchor-path by one hop (follow another anchor in any
    hit in the current cluster).
  - UCB1 balances belief (mean hit confidence) vs exploration.
  - At evaluation we hydrate the child's infons via range-get and score them.

This is not the final retrieval — it's a lab to watch how many GETs MCTS
burns for a given budget, and how much of that is footer/index scan vs.
actual infon bytes. Swap LocalFetcher for an s3fs-backed fetcher and the
numbers become real S3 costs.
"""

from __future__ import annotations

import math
import os
import random
import sys
import time
import tempfile
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "cognition", "src"))

from cognition.infon import Infon
from cognition.cassette.format import CassetteWriter
from cognition.cassette.index import (
    Manifest, build_indexes, query_anchor, Hit,
)
from cognition.cassette.reader import LocalFetcher, hydrate_locs


# ═══════════════════════════════════════════════════════════════════════
# MCTS NODE
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Node:
    anchor_path: tuple[str, ...]      # anchors traversed to get here
    hits: list[Hit] = field(default_factory=list)        # cluster at this node
    visits: int = 0
    value: float = 0.0                 # running belief score (0..1)
    parent: "Node | None" = field(default=None, repr=False)
    children: list["Node"] = field(default_factory=list)
    expanded: bool = False

    @property
    def key(self) -> str:
        return "→".join(self.anchor_path) or "<root>"

    def ucb(self, c: float = 1.4) -> float:
        if self.visits == 0:
            return float("inf")
        parent_visits = self.parent.visits if self.parent else 1
        return self.value + c * math.sqrt(math.log(parent_visits) / self.visits)


# ═══════════════════════════════════════════════════════════════════════
# MCTS ENGINE
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class MCTSStats:
    iterations: int = 0
    index_queries: int = 0
    range_gets: int = 0
    bytes_read: int = 0
    wall_ms: float = 0.0
    frontier: list[str] = field(default_factory=list)


class CassetteMCTS:
    def __init__(self, manifest: Manifest, fetcher: LocalFetcher,
                 seed_anchor: str, target_anchors: set[str] | None = None,
                 max_children: int = 4):
        self.manifest = manifest
        self.fetcher = fetcher
        self.seed = seed_anchor
        self.targets = target_anchors or set()
        self.max_children = max_children
        self.stats = MCTSStats()
        self.root = Node(anchor_path=(seed_anchor,))

    # ── selection ───────────────────────────────────────────────────────
    def _select(self) -> Node:
        node = self.root
        while node.expanded and node.children:
            node = max(node.children, key=lambda c: c.ucb())
        return node

    # ── expansion: one index query per expansion ────────────────────────
    def _expand(self, node: Node) -> list[Node]:
        anchor = node.anchor_path[-1]
        self.stats.index_queries += 1
        hits = query_anchor(self.manifest, anchor)
        node.hits = hits
        node.expanded = True

        # Candidate next anchors = all anchors appearing in hit triples,
        # excluding ones already in the path (avoid trivial loops).
        seen = set(node.anchor_path)
        counts: dict[str, int] = {}
        for h in hits:
            for a in (h.loc.subject, h.loc.predicate, h.loc.object):
                if a and a not in seen:
                    counts[a] = counts.get(a, 0) + 1
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:self.max_children]

        children = []
        for anchor_next, _ in top:
            child = Node(anchor_path=node.anchor_path + (anchor_next,),
                         parent=node)
            node.children.append(child)
            children.append(child)
        return children

    # ── evaluation: hydrate a sample, score vs. target set ──────────────
    def _evaluate(self, node: Node, sample: int = 3) -> float:
        if not node.hits:
            self.stats.index_queries += 1
            node.hits = query_anchor(self.manifest, node.anchor_path[-1])
        if not node.hits:
            return 0.0
        picks = random.sample(node.hits, min(sample, len(node.hits)))
        before_r, before_b = self.fetcher.requests, self.fetcher.bytes_read
        infons = hydrate_locs(self.fetcher, self.manifest, picks)
        self.stats.range_gets += self.fetcher.requests - before_r
        self.stats.bytes_read += self.fetcher.bytes_read - before_b

        score = 0.0
        for inf in infons:
            anchors = {inf.subject, inf.predicate, inf.object}
            overlap = len(anchors & self.targets) if self.targets else 0
            bonus = 0.5 * overlap
            score += min(1.0, inf.confidence + bonus)
        return score / len(infons)

    # ── backprop ────────────────────────────────────────────────────────
    def _backprop(self, node: Node, value: float):
        while node is not None:
            node.visits += 1
            # running mean
            node.value += (value - node.value) / node.visits
            node = node.parent

    # ── main loop ───────────────────────────────────────────────────────
    def run(self, iterations: int = 16) -> MCTSStats:
        t0 = time.perf_counter()
        for _ in range(iterations):
            self.stats.iterations += 1
            leaf = self._select()
            if not leaf.expanded:
                children = self._expand(leaf)
                if children:
                    leaf = random.choice(children)
            value = self._evaluate(leaf)
            self._backprop(leaf, value)
        self.stats.wall_ms = (time.perf_counter() - t0) * 1000

        # Capture top frontier for inspection.
        def walk(n: Node, depth: int = 0):
            if depth > 4 or not n.children:
                return
            best = sorted(n.children, key=lambda c: -c.value)[:2]
            for c in best:
                self.stats.frontier.append(
                    f"  {'  '*depth}{c.key}  v={c.value:.2f} n={c.visits}"
                )
                walk(c, depth + 1)
        walk(self.root)
        return self.stats


# ═══════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════

def mk_infon(idx: int, s: str, p: str, o: str, ts: str, conf=0.75) -> Infon:
    return Infon(infon_id=f"inf_{idx:05d}", subject=s, predicate=p, object=o,
                 polarity=1, direction="forward", confidence=conf,
                 sentence=f"{s} {p} {o}", doc_id=f"d{idx//5}",
                 sent_id=f"d{idx//5}_{idx:05d}", timestamp=ts)


def build_demo_store(root: str) -> Manifest:
    """Ten small cassettes — one per week — over auto-industry infons."""
    random.seed(7)
    subjects = ["toyota", "honda", "nissan", "vw", "gm", "ford", "byd", "tesla"]
    predicates = ["invest", "partner", "compete", "acquire", "supply"]
    objects = ["solid_state", "lithium_ion", "batteries", "panasonic",
               "catl", "ev_platform", "chips", "software"]

    manifest: Manifest | None = None
    idx = 0
    for week in range(10):
        infons = []
        for _ in range(60):
            infons.append(mk_infon(
                idx,
                random.choice(subjects),
                random.choice(predicates),
                random.choice(objects),
                f"2026-{1+week//4:02d}-{1+(week%4)*7:02d}",
                conf=round(random.uniform(0.5, 0.95), 2),
            ))
            idx += 1
        cid = f"week_{week:02d}"
        cdir = os.path.join(root, "cassettes")
        os.makedirs(cdir, exist_ok=True)
        path = os.path.join(cdir, f"{cid}.inf")
        with open(path, "wb") as f:
            w = CassetteWriter(f, cassette_id=cid, schema_ref="auto-v1")
            for inf in infons:
                w.add(inf)
            footer = w.close()
        index_paths = build_indexes(footer, os.path.join(root, "index"))
        manifest = Manifest.new(root, parent=manifest)
        manifest.add_cassette(footer, path, index_paths)
        manifest.save()
    return manifest


def main():
    root = tempfile.mkdtemp(prefix="mcts_lab_")
    t0 = time.perf_counter()
    manifest = build_demo_store(root)
    t_build = (time.perf_counter() - t0) * 1000
    total_bytes = sum(
        os.path.getsize(os.path.join(root, "cassettes", f))
        for f in os.listdir(os.path.join(root, "cassettes"))
    )
    idx_bytes = 0
    for kind in ("by_triple", "by_time", "by_anchor"):
        d = os.path.join(root, "index", kind)
        idx_bytes += sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d))
    n_infons = sum(c["n_records"] for c in manifest.cassettes)
    print(f"built: {len(manifest.cassettes)} cassettes, {n_infons} infons, "
          f"{total_bytes}B data + {idx_bytes}B index  ({t_build:.0f}ms)")
    print(f"index/data ratio: {idx_bytes / total_bytes:.2f}x\n")

    for budget in (8, 16, 32):
        fetcher = LocalFetcher()
        mcts = CassetteMCTS(
            manifest, fetcher,
            seed_anchor="toyota",
            target_anchors={"solid_state", "batteries", "panasonic"},
            max_children=4,
        )
        stats = mcts.run(iterations=budget)
        print(f"budget={budget:>3}  "
              f"iters={stats.iterations}  "
              f"index_q={stats.index_queries}  "
              f"gets={stats.range_gets:>3}  "
              f"bytes={stats.bytes_read:>6}  "
              f"wall={stats.wall_ms:5.0f}ms")
        print(f"  fraction of corpus read: {stats.bytes_read/total_bytes:.1%}")
        for line in stats.frontier[:5]:
            print(line)
        print()

    import shutil
    shutil.rmtree(root)


if __name__ == "__main__":
    main()

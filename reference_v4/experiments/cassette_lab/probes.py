"""Three probes to figure out what to build next.

Q1. Latency-bound or byte-bound?
    Inject simulated S3 RTT into the fetcher. If wall time ≈ requests × RTT,
    the bottleneck is request count, not bytes. That means (a) coalescing is
    worth building and (b) NLI scoring quality is roughly free.

Q2. Does request coalescing actually help?
    Group adjacent offsets in the same cassette into one range GET. Compare
    to naive one-GET-per-hit. Decision threshold: >2x drop → build it.

Q3. At 10x scale (100 cassettes / ~6k infons), does MCTS belief saturate?
    Plot belief-of-best-child vs. iteration. Plateau = retrieval is good
    enough, quality of scoring matters more. Still climbing = retrieval
    pattern still matters.
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

from cognition.infon import Infon
from cognition.cassette.format import CassetteWriter, _U32
from cognition.cassette.index import Manifest, build_indexes, query_anchor, Hit
from cognition.cassette.reader import LocalFetcher, hydrate_locs

import gzip, json
from cognition.infon import Infon as _Infon


# ═══════════════════════════════════════════════════════════════════════
# FETCHER VARIANTS
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SimFetcher:
    """LocalFetcher + per-request latency (simulated S3 RTT).

    Also tracks total sleep time separately so we can see whether the wall
    clock is dominated by latency or by actual work (decode/decompress).
    """
    rtt_ms: float = 0.0
    requests: int = 0
    bytes_read: int = 0
    sleep_ms: float = 0.0

    def fetch(self, path: str, offset: int, length: int) -> bytes:
        self.requests += 1
        if self.rtt_ms:
            time.sleep(self.rtt_ms / 1000)
            self.sleep_ms += self.rtt_ms
        with open(path, "rb") as f:
            f.seek(offset)
            buf = f.read(length)
        self.bytes_read += len(buf)
        return buf

    def size(self, path: str) -> int:
        return os.path.getsize(path)

    def reset(self):
        self.requests = 0
        self.bytes_read = 0
        self.sleep_ms = 0.0


def hydrate_coalesced(fetcher: SimFetcher, manifest: Manifest,
                      hits: list[Hit], gap_threshold: int = 4096) -> list[Infon]:
    """Merge adjacent hits (same cassette, offsets within gap_threshold bytes
    of each other) into a single range GET, then split locally."""
    by_cassette: dict[str, list[Hit]] = {}
    for h in hits:
        by_cassette.setdefault(h.cassette_id, []).append(h)

    infons: list[Infon] = []
    for cid, hs in by_cassette.items():
        path = manifest.cassette_path(cid)
        hs.sort(key=lambda h: h.loc.offset)

        # Build runs of coalesced hits.
        runs: list[list[Hit]] = []
        for h in hs:
            if runs:
                last = runs[-1][-1]
                gap = h.loc.offset - (last.loc.offset + last.loc.length)
                if 0 <= gap <= gap_threshold:
                    runs[-1].append(h)
                    continue
            runs.append([h])

        for run in runs:
            start = run[0].loc.offset
            end = run[-1].loc.offset + run[-1].loc.length
            buf = fetcher.fetch(path, start, end - start)
            for h in run:
                local = h.loc.offset - start
                frame_len = _U32.unpack(buf[local:local + _U32.size])[0]
                body = gzip.decompress(
                    buf[local + _U32.size: local + _U32.size + frame_len]
                )
                infons.append(_Infon.from_dict(json.loads(body.decode())))
    return infons


# ═══════════════════════════════════════════════════════════════════════
# MCTS  (instrumented — records per-iteration belief trace)
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
        if self.visits == 0:
            return float("inf")
        import math
        pv = self.parent.visits if self.parent else 1
        return self.value + 1.4 * (math.log(max(pv, 1)) / self.visits) ** 0.5


class MCTS:
    def __init__(self, manifest, fetcher, seed, targets, hydrate_fn,
                 max_children=4):
        self.m = manifest
        self.f = fetcher
        self.hydrate = hydrate_fn
        self.targets = targets
        self.max_children = max_children
        self.root = Node(anchor_path=(seed,))
        self.trace: list[float] = []  # best-leaf belief after each iter
        self.index_queries = 0

    def _select(self):
        n = self.root
        while n.expanded and n.children:
            n = max(n.children, key=lambda c: c.ucb())
        return n

    def _expand(self, node):
        self.index_queries += 1
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
            c = Node(anchor_path=node.anchor_path + (anc,), parent=node)
            node.children.append(c)
        return node.children

    def _evaluate(self, node, sample=3):
        if not node.hits:
            self.index_queries += 1
            node.hits = query_anchor(self.m, node.anchor_path[-1])
        if not node.hits:
            return 0.0
        picks = random.sample(node.hits, min(sample, len(node.hits)))
        infons = self.hydrate(self.f, self.m, picks)
        if not infons:
            return 0.0
        score = 0.0
        for inf in infons:
            anchors = {inf.subject, inf.predicate, inf.object}
            overlap = len(anchors & self.targets)
            score += min(1.0, inf.confidence + 0.5 * overlap)
        return score / len(infons)

    def _backprop(self, node, value):
        while node is not None:
            node.visits += 1
            node.value += (value - node.value) / node.visits
            node = node.parent

    def _best_belief(self) -> float:
        best = 0.0
        stack = [self.root]
        while stack:
            n = stack.pop()
            if n.visits > 0:
                best = max(best, n.value)
            stack.extend(n.children)
        return best

    def run(self, iterations: int):
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


# ═══════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════

def mk(i, s, p, o, ts, c=0.75):
    return Infon(infon_id=f"i{i:06d}", subject=s, predicate=p, object=o,
                 polarity=1, confidence=c, sentence=f"{s} {p} {o}",
                 doc_id=f"d{i//10}", sent_id=f"d{i//10}_{i:06d}", timestamp=ts)


def build_store(root: str, n_cassettes: int, per_cassette: int) -> Manifest:
    random.seed(42)
    subs = ["toyota", "honda", "nissan", "vw", "gm", "ford", "byd", "tesla",
            "bmw", "hyundai", "kia", "stellantis"]
    preds = ["invest", "partner", "compete", "acquire", "supply", "license"]
    objs = ["solid_state", "lithium_ion", "batteries", "panasonic", "catl",
            "ev_platform", "chips", "software", "motors", "charging_net"]
    m: Manifest | None = None
    idx = 0
    for k in range(n_cassettes):
        infons = [mk(idx + j,
                     random.choice(subs), random.choice(preds), random.choice(objs),
                     f"2026-{1+k//30:02d}-{1+(k%28):02d}",
                     round(random.uniform(0.5, 0.95), 2))
                  for j in range(per_cassette)]
        idx += per_cassette
        cid = f"c{k:03d}"
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


# ═══════════════════════════════════════════════════════════════════════
# PROBES
# ═══════════════════════════════════════════════════════════════════════

def probe_latency(manifest, targets):
    print("─" * 72)
    print("Q1. Latency-bound or byte-bound?")
    print("─" * 72)
    print(f"{'rtt_ms':>7} {'iters':>6} {'gets':>5} {'bytes':>7} "
          f"{'wall_ms':>8} {'sleep_ms':>9} {'work_ms':>8} {'wall/(rtt·gets)':>16}")
    for rtt in (0, 5, 30):
        for it in (16, 64):
            f = SimFetcher(rtt_ms=rtt)
            mcts = MCTS(manifest, f, "toyota", targets, hydrate_locs)
            t0 = time.perf_counter()
            mcts.run(it)
            wall = (time.perf_counter() - t0) * 1000
            work = wall - f.sleep_ms
            ratio = wall / max(rtt * f.requests, 1) if rtt else float("nan")
            ratio_s = f"{ratio:.2f}" if rtt else "n/a"
            print(f"{rtt:>7} {it:>6} {f.requests:>5} {f.bytes_read:>7} "
                  f"{wall:>8.0f} {f.sleep_ms:>9.0f} {work:>8.0f} {ratio_s:>16}")
    print()
    print("Read: if wall/(rtt·gets) ≈ 1.0, wall time = latency × requests →")
    print("      workload is latency-bound → coalescing wins.\n")


def probe_coalescing(manifest, targets):
    print("─" * 72)
    print("Q2. Coalescing payoff (at 30ms simulated RTT)")
    print("─" * 72)
    print(f"{'mode':>12} {'iters':>6} {'gets':>5} {'bytes':>7} {'wall_ms':>8}")
    for it in (16, 64):
        for mode, hyd in (("naive", hydrate_locs),
                          ("coalesced", hydrate_coalesced)):
            f = SimFetcher(rtt_ms=30)
            mcts = MCTS(manifest, f, "toyota", targets, hyd)
            t0 = time.perf_counter()
            mcts.run(it)
            wall = (time.perf_counter() - t0) * 1000
            print(f"{mode:>12} {it:>6} {f.requests:>5} {f.bytes_read:>7} {wall:>8.0f}")
    print()
    print("Read: if coalesced cuts gets by >2x at same bytes, build it.\n")


def probe_saturation(manifest, targets, n_iters=128):
    print("─" * 72)
    print(f"Q3. Belief saturation at scale ({len(manifest.cassettes)} cassettes)")
    print("─" * 72)
    f = SimFetcher(rtt_ms=0)
    mcts = MCTS(manifest, f, "toyota", targets, hydrate_locs)
    mcts.run(n_iters)
    print(f"total: {mcts.index_queries} index queries, "
          f"{f.requests} range gets, {f.bytes_read}B")
    # Print trace as sparse checkpoints + crude ASCII sparkline.
    checkpoints = [1, 4, 8, 16, 32, 64, 128]
    print(f"\n{'iter':>5} {'best_belief':>12}")
    for k in checkpoints:
        if k <= len(mcts.trace):
            print(f"{k:>5} {mcts.trace[k-1]:>12.3f}")

    # Ratio: belief gain in second half vs first half — if < 0.1, saturated.
    half = len(mcts.trace) // 2
    gain_h1 = max(mcts.trace[:half]) - mcts.trace[0]
    gain_h2 = max(mcts.trace[half:]) - max(mcts.trace[:half])
    print(f"\nH1 gain (iters 1..{half}):     {gain_h1:+.3f}")
    print(f"H2 gain (iters {half+1}..{len(mcts.trace)}): {gain_h2:+.3f}")
    if gain_h2 < 0.02:
        print("→ saturated: retrieval plateaus; scoring quality is next bottleneck.")
    else:
        print("→ still climbing: retrieval pattern still matters; coalesce/expand.")
    print()


def main():
    root = tempfile.mkdtemp(prefix="probe_")
    t0 = time.perf_counter()
    manifest = build_store(root, n_cassettes=100, per_cassette=60)
    t_build = (time.perf_counter() - t0) * 1000
    n = sum(c["n_records"] for c in manifest.cassettes)
    cass_bytes = sum(os.path.getsize(os.path.join(root, "cassettes", f))
                     for f in os.listdir(os.path.join(root, "cassettes")))
    idx_bytes = 0
    for k in ("by_triple", "by_time", "by_anchor"):
        d = os.path.join(root, "index", k)
        idx_bytes += sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d))
    print(f"store: {len(manifest.cassettes)} cassettes, {n} infons, "
          f"{cass_bytes/1024:.0f}KB data + {idx_bytes/1024:.0f}KB index, "
          f"ratio={idx_bytes/cass_bytes:.2f}x, built in {t_build:.0f}ms\n")

    targets = {"solid_state", "batteries", "panasonic", "catl"}

    probe_latency(manifest, targets)
    probe_coalescing(manifest, targets)
    probe_saturation(manifest, targets, n_iters=128)

    shutil.rmtree(root)


if __name__ == "__main__":
    main()

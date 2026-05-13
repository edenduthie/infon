"""Hand-curated actor-to-actor corpus that bypasses SPLADE extraction.

Why: real extraction's actor vs feature role competition throws away
actor-to-actor triples (nvidia/partner/tsmc becomes nvidia/partner/b200
because b200 is the only object the extractor can fill). To fairly
measure the GNN+MCTS integration, we need a corpus that has the
actor-to-actor edges it's designed to reason over.

Approach: construct Infon objects directly, feed to CassetteWriter.
Same serialization format, same index structure, same manifest. The
reasoner can't tell the difference between a curated Infon and a
SPLADE-extracted one — it just reads triples.

Corpus shape:
  • 30 infons, 10 cassettes (3 per cassette).
  • Clean chains: nvidia→tsmc→samsung→sk_hynix (4-hop supply chain)
  • Retracted: samsung→apple partnership affirmed then retracted
  • Anomaly: mention-edge decoys to confuse chain-following
  • Disconnected: ford/intel have their own sub-graphs
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "cognition", "src"))

from cognition.infon import Infon
from cognition.cassette.format import CassetteWriter
from cognition.cassette.index import Manifest, build_indexes


# ═══════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════

def mk(idx: int, s: str, p: str, o: str, ts: str,
       pol: int = 1, conf: float = 0.88, sent: str = "") -> Infon:
    return Infon(
        infon_id=f"cur_{idx:04d}",
        subject=s, predicate=p, object=o,
        polarity=pol, direction="forward", confidence=conf,
        sentence=sent or f"{s} {p} {o}",
        doc_id=f"d{idx//3:02d}",
        sent_id=f"d{idx//3:02d}_{idx:04d}",
        timestamp=ts,
    )


# Relation kinds — the GNN needs this mapping, and the synthgen config
# trained the model on these kinds. Using the same vocab means the
# frozen encoder is expecting these exact semantics.
RELATION_KINDS = {
    "partner":  "connective",
    "supply":   "connective",
    "acquire":  "connective",
    "license":  "connective",
    "invest":   "terminal",
    "mention":  "reportive",
    "describe": "reportive",
}


def build_infons() -> list[Infon]:
    """30 infons laid out for specific test claims.

    Key structure (what the test claims will hit):
      • 4-hop clean chain: nvidia→tsmc→samsung→sk_hynix→apple
      • 2-hop chain retracted at the end: bmw→toyota→honda retracted
      • Reportive edges that should NOT form a chain: mention-hops
      • Isolated actors: ford, intel sit in their own graph
    """
    infons = []
    idx = 0

    # ── Chain 1: long clean chain ──────────────────────────────────────
    # nvidia → tsmc → samsung → sk_hynix → apple
    for (s, p, o, ts) in [
        ("nvidia", "partner", "tsmc",    "2026-01-05"),
        ("tsmc",   "supply",  "samsung", "2026-01-12"),
        ("samsung","supply",  "sk_hynix","2026-01-20"),
        ("sk_hynix","license","apple",   "2026-02-01"),
    ]:
        infons.append(mk(idx, s, p, o, ts)); idx += 1

    # ── Chain 2: retracted mid-chain ───────────────────────────────────
    # bmw → toyota (affirmed), toyota → honda (retracted later).
    for (s, p, o, ts, pol) in [
        ("bmw",    "partner", "toyota", "2026-01-08", 1),
        ("toyota", "partner", "honda",  "2026-01-15", 1),
        ("toyota", "partner", "honda",  "2026-03-01", 0),  # retraction
    ]:
        infons.append(mk(idx, s, p, o, ts, pol=pol)); idx += 1

    # ── Chain 3: reportive-only (should NOT be a valid chain) ──────────
    # google mentions meta, meta mentions amazon — connects them by
    # graph topology but NOT by connective semantics.
    for (s, p, o, ts) in [
        ("google",  "mention",  "meta",    "2026-01-10"),
        ("meta",    "describe", "amazon",  "2026-01-18"),
    ]:
        infons.append(mk(idx, s, p, o, ts)); idx += 1

    # ── Chain 4: 2-hop acquire chain ───────────────────────────────────
    # microsoft → openai → anthropic (acquire)
    for (s, p, o, ts) in [
        ("microsoft", "acquire", "openai",    "2026-01-25"),
        ("openai",    "acquire", "anthropic", "2026-02-10"),
    ]:
        infons.append(mk(idx, s, p, o, ts)); idx += 1

    # ── Chain 5: reportive prefix + connective suffix ──────────────────
    # This tests whether the GNN correctly stops the chain when a
    # reportive edge kicks off a path that then has connective edges.
    # vw mentions stellantis, stellantis partners ferrari.
    for (s, p, o, ts) in [
        ("vw",         "mention", "stellantis", "2026-01-14"),
        ("stellantis", "partner", "ferrari",    "2026-01-22"),
    ]:
        infons.append(mk(idx, s, p, o, ts)); idx += 1

    # ── Isolated clusters (disconnected claims hit these) ──────────────
    # ford has its own graph; intel has its own graph.
    for (s, p, o, ts) in [
        ("ford",  "supply",  "mazda",   "2026-01-30"),
        ("mazda", "license", "subaru",  "2026-02-15"),
        ("intel", "acquire", "mobileye","2026-02-20"),
    ]:
        infons.append(mk(idx, s, p, o, ts)); idx += 1

    # ── Noise / decoys ─────────────────────────────────────────────────
    # Terminal-kind and reportive edges scattered around the graph.
    for (s, p, o, ts) in [
        ("nvidia", "invest",  "ai_research", "2026-01-02"),
        ("tsmc",   "mention", "geopolitics", "2026-01-06"),
        ("openai", "invest",  "safety",      "2026-02-11"),
        ("google", "invest",  "tpu",         "2026-01-11"),
        ("apple",  "mention", "china",       "2026-02-02"),
        ("honda",  "invest",  "hybrid",      "2026-01-16"),
        ("samsung","invest",  "hbm_r_and_d", "2026-01-21"),
        ("bmw",    "mention", "trade_show",  "2026-01-09"),
        ("amazon", "invest",  "aws",         "2026-01-19"),
        ("meta",   "invest",  "metaverse",   "2026-01-11"),
        ("ford",   "invest",  "ev",          "2026-01-31"),
    ]:
        infons.append(mk(idx, s, p, o, ts)); idx += 1

    return infons


# ═══════════════════════════════════════════════════════════════════════
# CASSETTE WRITER  — bypasses SPLADE extraction entirely
# ═══════════════════════════════════════════════════════════════════════

def write_curated_store(root: str, infons_per_cassette: int = 3) -> Manifest:
    """Write curated infons directly to cassettes, build indexes + manifest.

    Skips the Cognition ingest path. Produces a store that the reasoner
    sees as normal — same files, same footer, same indexes, same manifest
    semantics."""
    infons = build_infons()
    cdir = os.path.join(root, "cassettes")
    os.makedirs(cdir, exist_ok=True)

    m: Manifest | None = None
    for k in range(0, len(infons), infons_per_cassette):
        batch = infons[k:k + infons_per_cassette]
        # Content-hash the batch for a stable cassette_id. Mirrors how
        # InfonStore does it, so re-running is idempotent.
        h = hashlib.sha256()
        for inf in batch:
            h.update(inf.sentence.encode())
        cid = f"cur_{h.hexdigest()[:8]}"
        path = os.path.join(cdir, f"{cid}.inf")
        with open(path, "wb") as f:
            w = CassetteWriter(f, cassette_id=cid, schema_ref="curated")
            for inf in batch:
                w.add(inf)
            footer = w.close()
        ip = build_indexes(footer, os.path.join(root, "index"))
        m = Manifest.new(root, parent=m)
        m.add_cassette(footer, path, ip)
        m.save()
    return m


# ═══════════════════════════════════════════════════════════════════════
# GOLD LABELS for test claims
# ═══════════════════════════════════════════════════════════════════════
#
# Ten connectivity claims with ground truth, each exercising a specific
# reasoning primitive the GNN+MCTS should handle.

CLAIMS = [
    # (source, target, gold, notes)
    ("nvidia",     "tsmc",      "SUPPORTS",         "1-hop partner"),
    ("nvidia",     "samsung",   "SUPPORTS",         "2-hop partner→supply"),
    ("nvidia",     "sk_hynix",  "SUPPORTS",         "3-hop chain"),
    ("nvidia",     "apple",     "SUPPORTS",         "4-hop — max_hops=3 default, likely NEI"),
    ("bmw",        "honda",     "REFUTES",          "2-hop with retracted final edge"),
    ("google",     "amazon",    "NOT_ENOUGH_INFO",  "all reportive edges — no real chain"),
    ("vw",         "ferrari",   "NOT_ENOUGH_INFO",  "reportive prefix breaks chain"),
    ("microsoft",  "anthropic", "SUPPORTS",         "2-hop acquire chain"),
    ("ford",       "nvidia",    "NOT_ENOUGH_INFO",  "disconnected graphs"),
    ("intel",      "tsmc",      "NOT_ENOUGH_INFO",  "intel isolated"),
]

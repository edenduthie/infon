"""Verify synthgen fidelity: does symbolic chain_mass agree with labels?

If the symbolic reasoner's verdict on a generated chain disagrees with
the synthgen ground-truth label, one of two things is true:
  • Our hand-coded chain_mass rule is wrong (and we'd learn it here).
  • The generator produces graphs the rule can't handle (and we should
    fix the generator before training a GNN on biased data).

This probe also prints distributional stats so we can eyeball whether
the config produces a balanced training set.
"""

from __future__ import annotations

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "cognition", "src"))

from cognition.cassette.synthgen import (
    SynthGenConfig, RelationSpec, SynthEdge, SynthGraph,
    generate, summarize, _default_config,
)
from cognition.cassette.reason_path import chain_mass, _label_from_mass
from cognition.infon import Infon


def synthedge_to_infon(e: SynthEdge) -> Infon:
    """Thin adapter so we can feed SynthEdges into chain_mass unchanged."""
    return Infon(
        infon_id=f"syn_{id(e):x}",
        subject=e.subject, predicate=e.predicate, object=e.object,
        polarity=e.polarity, confidence=e.confidence,
        sentence=f"{e.subject} {e.predicate} {e.object}",
        timestamp=f"2026-01-{1 + (e.t % 28):02d}",
    )


def chain_from_synth(g: SynthGraph) -> list[list[Infon]]:
    """Extract the ORDERED path edges from source to target, grouping
    same-triple infons (so a retraction lands on the same edge).

    Mirrors what the cassette reasoner's MCTS does when it walks a path:
    find each hop's triple, collect all its affirmations+retractions.
    """
    # Build adjacency: src → [(obj, edges)]
    adj: dict[str, list[tuple[str, list[Infon]]]] = {}
    triples: dict[tuple, list[Infon]] = {}
    for e in g.edges:
        t = (e.subject, e.predicate, e.object)
        triples.setdefault(t, []).append(synthedge_to_infon(e))
    for (s, p, o), infons in triples.items():
        adj.setdefault(s, []).append((o, infons))

    # BFS for the shortest path from source to target, preferring
    # connective-polarity-1 edges first (the "default" chain).
    if g.chain_source == g.chain_target:
        return []
    from collections import deque
    queue = deque([(g.chain_source, [])])
    seen = {g.chain_source}
    while queue:
        node, path = queue.popleft()
        for nxt, edge_infons in adj.get(node, []):
            new_path = path + [edge_infons]
            if nxt == g.chain_target:
                return new_path
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, new_path))
    return []  # no path — should yield NEI


def verdict_from_symbolic(g: SynthGraph) -> str:
    path_edges = chain_from_synth(g)
    m = chain_mass(path_edges, g.chain_source, g.chain_target)
    return _label_from_mass(m)


def main():
    print("═" * 72)
    print("  Synthgen fidelity probe")
    print("═" * 72)

    config = _default_config()
    config.n_samples = 400
    config.seed = 7

    graphs = generate(config)
    stats = summarize(graphs)

    print("\n── distribution ──")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n── symbolic vs. gold ──")
    agree = 0
    confusion: dict[tuple, int] = Counter()
    per_kind_agree: dict[str, list[int]] = {}  # kind → [agree_count, total]
    for g in graphs:
        predicted = verdict_from_symbolic(g)
        gold = g.chain_verdict
        confusion[(gold, predicted)] += 1
        ok = (predicted == gold)
        if ok:
            agree += 1
        stats_list = per_kind_agree.setdefault(g.kind, [0, 0])
        stats_list[1] += 1
        if ok:
            stats_list[0] += 1

    total = len(graphs)
    print(f"  overall agreement: {agree}/{total} = {agree/total:.1%}")

    print(f"\n  by kind:")
    for kind, (a, t) in sorted(per_kind_agree.items()):
        print(f"    {kind:<14} {a}/{t} = {a/t:.0%}")

    print(f"\n  confusion (gold → predicted):")
    verdicts = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
    for gold in verdicts:
        row = [confusion.get((gold, pred), 0) for pred in verdicts]
        print(f"    {gold:<16} → " + "  ".join(
            f"{pred[:3]}:{n:>3}" for pred, n in zip(verdicts, row)
        ))

    # Show a few disagreements so we can eyeball them.
    print(f"\n── first 5 disagreements ──")
    shown = 0
    for g in graphs:
        pred = verdict_from_symbolic(g)
        if pred == g.chain_verdict:
            continue
        print(f"\n  kind={g.kind}  gold={g.chain_verdict}  "
              f"symbolic={pred}  source={g.chain_source} → target={g.chain_target}")
        for e in g.edges[:8]:
            mark = "¬" if e.polarity == 0 else " "
            print(f"    t={e.t:>3}  {mark}{e.subject}/{e.predicate}/{e.object} "
                  f"(conf={e.confidence})")
        shown += 1
        if shown >= 5:
            break


if __name__ == "__main__":
    main()

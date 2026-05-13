"""Multi-target reasoner-MCTS.

Question pattern: "Is source connected to ANY of {t1, t2, ...}?" or "Which
of {t1, t2, ...} is source closest to?"

Key insight: one tree walk outward from source serves all targets. Each
range-get contributes to the search for every target simultaneously. A
path from source to node N attributes to N automatically — if N ∈ targets,
we capture the best mass for that target.

Budget model:
  - Single target: ~2 gets to resolve a reachable chain.
  - N targets: still one tree walk. Reachable ones resolve as the path
    passes through them. Unreachable ones require the tree to expand
    until UCB exhausts search space → θ stays 1.0 (honest NEI).
  - Budget is NOT O(N); it's closer to max over targets.

Returns dict[target → Verdict] so each target gets its own calibrated
mass, label, and source path.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "cognition", "src"))

from cognition.infon import Infon
from cognition.cassette import Query
from cognition.cassette.index import Manifest
from cognition.cassette.reason import Verdict
from cognition.cassette.reader import LocalFetcher, hydrate_locs
from cognition.dempster_shafer import MassFunction, combine_multiple

from mcts_core import (
    MCTSNode, _edge_mass, chain_mass, _entity_set,
    _label_from_mass, _tree_size, _leaves,
)


def reason_any_target(
    m: Manifest, source: str, targets: set[str],
    budget: int = 20, max_hops: int = 3,
    max_children_per_expand: int = 6,
    connective_predicates: set[str] | None = None,
    verbose: bool = False,
) -> dict[str, Verdict]:
    """One tree walk from `source`; returns a Verdict per target.

    Selection uses the MAX over targets of |S - R| so we prefer expanding
    nodes that might resolve any still-uncertain target.
    """
    fetcher = LocalFetcher()
    entities = _entity_set(m) | {source} | set(targets)
    root = MCTSNode(anchor=source)

    # Per-target best (mass, path_infons).
    best: dict[str, tuple[MassFunction, list[Infon]]] = {
        t: (MassFunction(theta=1.0), []) for t in targets
    }
    all_sources: list[Infon] = []
    iters = 0

    def node_score(n: MCTSNode) -> float:
        """For selection: a node is promising if ANY target on its path is
        well-resolved, OR if its θ is high (still room to reduce ignorance
        by going deeper)."""
        on_path = set(n.path()) & targets
        if on_path:
            # Best |S-R| across targets this path already resolves.
            decisive = max(
                abs(best[t][0].supports - best[t][0].refutes)
                for t in on_path
            )
        else:
            decisive = abs(n.path_mass.supports - n.path_mass.refutes)
        if n.visits == 0:
            return float("inf")
        pv = n.parent.visits if n.parent else 1
        # θ proxy: if we've confidently resolved a target on this path,
        # keep-θ-low. Otherwise use the path_mass's θ.
        theta = n.path_mass.theta
        explore = 1.2 * theta * math.sqrt(math.log(max(pv, 1)) / n.visits)
        return decisive + explore

    def all_targets_resolved() -> bool:
        for t in targets:
            mass, _ = best[t]
            if mass.supports < 0.4 and mass.refutes < 0.4:
                return False  # still uncertain
        return True

    while fetcher.requests < budget and iters < budget * 4:
        iters += 1

        # SELECT
        node = root
        while node.expanded and node.children:
            node = max(node.children, key=node_score)

        if len(node.path()) - 1 >= max_hops:
            node.visits += 1
            continue

        # EXPAND
        if not node.expanded:
            hits = Query().mentioning(node.anchor).run(m)
            if connective_predicates is not None:
                hits = [h for h in hits if h.loc.predicate in connective_predicates]
            used = {i.infon_id for i in node.path_infons()}
            hits = [h for h in hits if h.loc.infon_id not in used]
            remaining = budget - fetcher.requests
            hits = hits[: min(max_children_per_expand, remaining)]
            if not hits:
                node.expanded = True
                node.visits += 1
                continue
            infons = hydrate_locs(fetcher, m, hits)
            all_sources.extend(infons)

            edges_by_triple: dict[tuple, list[Infon]] = {}
            for inf in infons:
                if node.anchor not in (inf.subject, inf.object):
                    continue
                if connective_predicates is not None and \
                   inf.predicate not in connective_predicates:
                    continue
                triple = (inf.subject, inf.predicate, inf.object)
                edges_by_triple.setdefault(triple, []).append(inf)

            for triple, edge_infons in edges_by_triple.items():
                entity_others = {triple[0], triple[2]} - {node.anchor}
                entity_others &= entities
                for a in entity_others:
                    if a in node.path():
                        continue
                    child = MCTSNode(anchor=a, parent=node,
                                     edge_infons=edge_infons)
                    # The child's path_mass is the chain to a generic target:
                    # use a sentinel = child.anchor so chain_mass doesn't
                    # require path_anchors to include a specific target.
                    child.path_mass = chain_mass(
                        node.path_edges() + [edge_infons], source, a)
                    node.children.append(child)
            node.expanded = True
            if not node.children:
                node.visits += 1
                continue

        # EVALUATE
        if not node.children:
            node.visits += 1
            continue
        leaf = max(node.children,
                   key=lambda c: abs(c.path_mass.supports - c.path_mass.refutes))

        # Attribute leaf to every target on its path (not just endpoint).
        # A leaf at 'panasonic' on the way to 'catl' doesn't update catl yet;
        # but a leaf at 'catl' DOES, and also any intermediate anchors that
        # happen to be targets.
        leaf_path = set(leaf.path())
        for t in leaf_path & targets:
            # Recompute mass for this specific target based on the prefix
            # of the path that ends at t.
            prefix_edges = []
            for i, edge in enumerate(leaf.path_edges()):
                prefix_edges.append(edge)
                # Check if after adding edge `i`, the path now contains t.
                reached = {source}
                for e in prefix_edges:
                    for inf in e:
                        reached.add(inf.subject); reached.add(inf.object)
                if t in reached:
                    mass_t = chain_mass(prefix_edges, source, t)
                    curr_mass, _ = best[t]
                    if abs(mass_t.supports - mass_t.refutes) > \
                       abs(curr_mass.supports - curr_mass.refutes):
                        best[t] = (mass_t, [x for e in prefix_edges for x in e])
                    break  # stop at first prefix reaching t

        # BACKPROP
        n = leaf
        while n is not None:
            n.visits += 1
            n = n.parent

        if all_targets_resolved():
            break

    # Assemble Verdict per target.
    out: dict[str, Verdict] = {}
    total_gets = fetcher.requests
    for t in targets:
        mass, sources = best[t]
        label = _label_from_mass(mass)
        out[t] = Verdict(
            label=label, mass=mass,
            n_candidates=len(all_sources),
            n_hydrated=len(all_sources),
            range_gets=total_gets,
            sources=sources,
        )

    if verbose:
        paths = {t: leaf.path() for leaf in _leaves(root)
                 for t in set(leaf.path()) & targets}
        print(f"    [multi] iters={iters} tree={_tree_size(root)} "
              f"gets={total_gets} paths-by-target={paths}")
    return out

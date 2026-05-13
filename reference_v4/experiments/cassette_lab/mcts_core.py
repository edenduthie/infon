"""Shared reasoner-MCTS core. Isolated from test harnesses so scale/negation
probes can reuse it.

Public API:
  reason_mcts(manifest, source, target, budget, max_hops=3, verbose=False) → Verdict
  chain_mass(infons, source, target) → MassFunction  (respects polarity)
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


def _edge_mass(edge_infons: list[Infon]) -> MassFunction:
    """Combine all infons that describe the same (S,P,O) edge across time.

    A later negation of an earlier affirmation should cancel (or reverse)
    the edge. Dempster's rule does this naturally: affirm (S=0.7, θ=0.3)
    ⊕ refute (R=0.7, θ=0.3) → S≈0.21, R≈0.21, θ≈0.09, conflict≈0.49,
    which normalizes to S≈R — honestly unknown, not confidently connected.
    """
    if not edge_infons:
        return MassFunction(theta=1.0)
    per = []
    for inf in edge_infons:
        c = max(0.0, min(inf.confidence, 1.0))
        w = 0.20 + 0.55 * c
        if inf.polarity == 0:
            per.append(MassFunction(refutes=w * 0.85, theta=1.0 - w * 0.85))
        else:
            per.append(MassFunction(supports=w * 0.85, theta=1.0 - w * 0.85))
    return combine_multiple(per)


def _label_from_mass(m: MassFunction,
                     supports_threshold: float = 0.25,
                     refutes_threshold: float = 0.15) -> str:
    if m.supports >= supports_threshold and m.supports > m.refutes:
        return "SUPPORTS"
    if m.refutes >= refutes_threshold and m.refutes > m.supports:
        return "REFUTES"
    return "NOT_ENOUGH_INFO"


def chain_mass(path_edges: list[list[Infon]], source: str, target: str) -> MassFunction:
    """Combine mass along a path whose edges may each have MULTIPLE
    corroborating/refuting infons.

    A chain is a CONJUNCTION — the whole path holds only if every edge
    holds. So:
      S(chain) = min over edges (weakest link)
      R(chain) = max over edges (any refuted edge kills the chain)
      θ(chain) = 1 - S - R

    Dempster's additive combine is wrong here: it would amplify multiple
    independent affirmations into higher S, but an additional affirming
    edge does not strengthen a chain — it can only maintain or weaken it.
    """
    if not path_edges:
        return MassFunction(theta=1.0)
    all_infons = [i for edge in path_edges for i in edge]
    path_anchors: set[str] = set()
    for inf in all_infons:
        path_anchors |= {inf.subject, inf.predicate, inf.object}
    if source not in path_anchors or target not in path_anchors:
        return MassFunction(theta=1.0)

    per_edge = [_edge_mass(edge) for edge in path_edges]
    chain_s = min(m.supports for m in per_edge)
    chain_r = max(m.refutes for m in per_edge)

    # Long-chain penalty: bleed S to θ as hops grow.
    hop_penalty = 0.10 * max(0, len(path_edges) - 1)
    chain_s = max(0.0, chain_s - hop_penalty)

    chain_t = max(0.0, 1.0 - chain_s - chain_r)
    return MassFunction(supports=chain_s, refutes=chain_r, theta=chain_t)


@dataclass
class MCTSNode:
    anchor: str
    parent: "MCTSNode | None" = field(default=None, repr=False)
    children: list["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    path_mass: MassFunction = field(default_factory=lambda: MassFunction(theta=1.0))
    # edge_infons = ALL infons for the (parent.anchor, predicate, anchor)
    # edge — includes affirmations and refutations across cassettes.
    edge_infons: list[Infon] = field(default_factory=list)
    expanded: bool = False

    def path(self) -> list[str]:
        out, n = [], self
        while n:
            out.append(n.anchor); n = n.parent
        return list(reversed(out))

    def path_edges(self) -> list[list[Infon]]:
        """Edges on the path from root to this node. Each edge is the list
        of all infons that describe the same triple (may span shards)."""
        out, n = [], self
        while n.parent is not None:
            out.append(list(n.edge_infons))
            n = n.parent
        return list(reversed(out))

    def path_infons(self) -> list[Infon]:
        """Flat list — used only for loop avoidance (can this infon already
        be on the path)."""
        return [i for edge in self.path_edges() for i in edge]

    def ucb(self, c: float = 1.2) -> float:
        if self.visits == 0:
            return float("inf")
        pv = self.parent.visits if self.parent else 1
        decisiveness = abs(self.path_mass.supports - self.path_mass.refutes)
        explore = c * self.path_mass.theta * math.sqrt(math.log(max(pv, 1)) / self.visits)
        return decisiveness + explore


def _entity_set(m: Manifest) -> set[str]:
    """Anchors that have appeared as a subject in any cassette. An entity
    is something that ACTS — not just something that gets acted on. This
    excludes terminal attributes like country codes or event names that
    appear only as objects in decoy edges, which would otherwise let MCTS
    build fake chains via graph-coincidental overlap."""
    out: set[str] = set()
    for c in m.cassettes:
        out |= set(c.get("subjects", ()))
    return out


def reason_mcts(m: Manifest, source: str, target: str,
                budget: int = 12, max_hops: int = 3,
                max_children_per_expand: int = 6,
                connective_predicates: set[str] | None = None,
                verbose: bool = False) -> Verdict:
    """AlphaGo-style MCTS with the DS reasoner as scorer.

    connective_predicates: optional allow-list of predicates that count as
      a real chain edge. If None, every predicate is connective (legacy
      behavior). In production this comes from the schema — relations
      like 'partner', 'supply', 'acquire', 'license' are connective;
      'mention', 'review', 'host' are not.
    """
    fetcher = LocalFetcher()
    # Entity set: anchors that appear as a subject anywhere, PLUS the
    # query's source/target (the caller is asserting these are entities,
    # and they may only appear as objects — e.g., terminal nodes of a
    # supply chain never 'act' on anything downstream).
    entities = _entity_set(m) | {source, target}
    root = MCTSNode(anchor=source)
    best_mass = MassFunction(theta=1.0)
    best_path_infons: list[Infon] = []
    all_sources: list[Infon] = []
    iters = 0

    while fetcher.requests < budget and iters < budget * 4:
        iters += 1

        # SELECT
        node = root
        while node.expanded and node.children:
            node = max(node.children, key=lambda c: c.ucb())

        if len(node.path()) - 1 >= max_hops:
            node.visits += 1
            continue

        # EXPAND
        if not node.expanded:
            # Index-level predicate filter: only hydrate edges where the
            # predicate is connective. The index carries predicate already,
            # so this is a cheap Parquet filter — we avoid hydrating the
            # dozens of decoy "mention/advertise/sponsor" edges.
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

            # Group hydrated infons by (subject, predicate, object) triple.
            # Each distinct triple = one edge; all its infons collapse into
            # a single edge mass via Dempster's rule, so retractions cancel
            # affirmations automatically.
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
                # Candidate next anchor: the OTHER entity in this triple.
                entity_others = {triple[0], triple[2]} - {node.anchor}
                entity_others &= entities
                for a in entity_others:
                    if a in node.path():
                        continue
                    child = MCTSNode(anchor=a, parent=node,
                                     edge_infons=edge_infons)
                    child.path_mass = chain_mass(
                        node.path_edges() + [edge_infons], source, target)
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
        value = leaf.path_mass

        # Track best path-to-target by decisiveness (supports AND refutes both count).
        if target in leaf.path():
            current_best = abs(best_mass.supports - best_mass.refutes)
            leaf_best = abs(value.supports - value.refutes)
            if leaf_best > current_best:
                best_mass = value
                best_path_infons = leaf.path_infons()

        # BACKPROP
        n = leaf
        while n is not None:
            n.visits += 1
            n = n.parent

        # Early stop on confidence (either direction).
        if best_mass.supports > 0.55 or best_mass.refutes > 0.45:
            break

    if best_mass.supports > 0 or best_mass.refutes > 0:
        final_mass = best_mass
        sources = best_path_infons
    else:
        final_mass = MassFunction(theta=1.0)
        sources = []

    if verbose:
        paths = [_.path() for _ in _leaves(root) if target in _.path()]
        all_paths = [_.path() for _ in _leaves(root)]
        print(f"    [mcts] iters={iters} tree={_tree_size(root)} gets={fetcher.requests} "
              f"paths-to-target={paths[:4]} all_leaves={all_paths[:6]}")

    return Verdict(
        label=_label_from_mass(final_mass),
        mass=final_mass,
        n_candidates=len(all_sources),
        n_hydrated=len(all_sources),
        range_gets=fetcher.requests,
        sources=sources,
    )


def _tree_size(n): return 1 + sum(_tree_size(c) for c in n.children)


def _leaves(n):
    if not n.children: return [n]
    out = []
    for c in n.children: out += _leaves(c)
    return out

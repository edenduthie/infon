"""Connectivity reasoning over cassette-stored infons — AlphaGo-style MCTS.

Use when a claim is about whether two entities are connected through a
chain of relations (multi-hop). For single-triple claims the flat-seed
`reason()` is faster and simpler; use this when no single infon contains
both endpoints.

Public API:
  reason_connectivity(manifest, source, target, *, budget, ...)
      → Verdict with S/R/θ and the path infons that drove it.

  reason_any_target(manifest, source, targets: set, *, budget, ...)
      → dict[target → Verdict], all resolved in a single tree walk.

Key design choices (see BUILD NOTES at the bottom of this file):
  1. Group hydrated infons by (S,P,O) triple per edge — later retractions
     cancel earlier affirmations via Dempster's rule.
  2. Chain mass = min(S), max(R) — a chain is a conjunction; a refuted
     hop kills it. Don't use additive combine across hops.
  3. Index-level predicate filter — only hydrate edges whose predicate
     is "connective" (partner, supply, acquire, license, invest). Skips
     decoy mention/sponsor/advertise edges without hydrating them.
  4. Entity anchors only for traversal — an entity is one that has appeared
     as a subject somewhere, plus the query endpoints. Excludes terminal
     attribute objects (countries, events) that fake chain overlap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..infon import Infon
from ..dempster_shafer import MassFunction, combine_multiple
from .dsl import Query
from .index import Manifest
from .reader import LocalFetcher, RangeFetcher, hydrate_locs
from .reason import Verdict


# ═══════════════════════════════════════════════════════════════════════
# EDGE / CHAIN MASS
# ═══════════════════════════════════════════════════════════════════════

def _edge_mass(edge_infons: list[Infon]) -> MassFunction:
    """Combine all infons that describe the same (S, P, O) edge.

    A later negation of an earlier affirmation should cancel or reverse
    the edge. Dempster's rule does this via conflict normalization:
    affirm (S=0.7, θ=0.3) ⊕ refute (R=0.7, θ=0.3) → S ≈ R ≈ 0.2,
    which is honestly uncertain — neither confidently supported nor
    confidently refuted.
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


def chain_mass(path_edges: list[list[Infon]],
               source: str, target: str) -> MassFunction:
    """Combine mass along a path whose edges may each have multiple infons.

    A chain is a CONJUNCTION — the whole path holds only if every edge
    holds. So:
      S(chain) = min over edges (weakest link)
      R(chain) = max over edges (any refuted edge breaks it)
      θ(chain) = 1 - S - R

    Dempster's additive combine across edges is wrong here: it amplifies
    S as edges are added, but an extra affirmed edge doesn't make a chain
    shorter — if anything it introduces more room for failure. The min/max
    semantics match the conjunctive meaning of "A and B and C all hold".
    """
    if not path_edges:
        return MassFunction(theta=1.0)

    path_anchors: set[str] = set()
    for edge in path_edges:
        for inf in edge:
            path_anchors |= {inf.subject, inf.predicate, inf.object}
    if source not in path_anchors or target not in path_anchors:
        return MassFunction(theta=1.0)

    per_edge = [_edge_mass(edge) for edge in path_edges]
    chain_s = min(m.supports for m in per_edge)
    chain_r = max(m.refutes for m in per_edge)

    # Longer chains bleed S to θ.
    hop_penalty = 0.10 * max(0, len(path_edges) - 1)
    chain_s = max(0.0, chain_s - hop_penalty)
    chain_t = max(0.0, 1.0 - chain_s - chain_r)
    return MassFunction(supports=chain_s, refutes=chain_r, theta=chain_t)


# ═══════════════════════════════════════════════════════════════════════
# MCTS NODE
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class _Node:
    anchor: str
    parent: "_Node | None" = field(default=None, repr=False)
    children: list["_Node"] = field(default_factory=list)
    visits: int = 0
    path_mass: MassFunction = field(default_factory=lambda: MassFunction(theta=1.0))
    # All infons for the triple (parent.anchor, predicate, anchor) or its
    # inverse — affirmed and refuted across shards.
    edge_infons: list[Infon] = field(default_factory=list)
    expanded: bool = False

    def path(self) -> list[str]:
        out, n = [], self
        while n is not None:
            out.append(n.anchor); n = n.parent
        return list(reversed(out))

    def path_edges(self) -> list[list[Infon]]:
        out, n = [], self
        while n.parent is not None:
            out.append(list(n.edge_infons)); n = n.parent
        return list(reversed(out))

    def path_infons(self) -> list[Infon]:
        return [i for edge in self.path_edges() for i in edge]


def _entity_set(m: Manifest) -> set[str]:
    """Anchors that have appeared as a subject. An entity ACTS — this
    excludes terminal attribute objects that would otherwise fake chain
    connectivity via graph-coincidental overlap."""
    out: set[str] = set()
    for c in m.cassettes:
        out |= set(c.get("subjects", ()))
    return out


def infer_connective_predicates(
    m: Manifest,
    *,
    min_entity_object_ratio: float = 0.5,
    min_occurrences: int = 1,
) -> set[str]:
    """Auto-derive which predicates count as real connectivity edges.

    Heuristic: a predicate is 'connective' if most of its object positions
    are filled by entities (anchors that ACT somewhere) rather than
    terminal attributes (countries, events, reports).

        toyota supply catl          → catl acts elsewhere → connective
        toyota mention japan        → japan never acts    → non-connective
        toyota announce q4_report   → q4_report never acts → non-connective

    This is inferred from the corpus itself via the cassette manifests —
    no schema annotation required, and the inference improves as more
    cassettes land.

    Args:
      min_entity_object_ratio: predicate qualifies if at least this
        fraction of its distinct objects are entities. Default 0.5
        (majority-entity). Raise to 1.0 for a strict regime where ANY
        non-entity object disqualifies the predicate — useful if you
        know your corpus has clean actor-to-actor edges. Lower it when
        extraction produces mixed actor/feature objects (the common
        case post-extraction-fix: 'partner' legitimately has both
        actor-objects like tsmc AND feature-objects like b200).
      min_occurrences: predicate must appear in at least this many distinct
        (subject, object) combinations before we trust the ratio. Default 1
        accepts rare-but-clean predicates; bump to 2+ if you want to suppress
        one-off connections until they're corroborated.

    Note on corpus size: the heuristic works well with >~10 occurrences per
    predicate. Below that, single-use non-connective predicates with entity
    objects (e.g. "toyota mention honda") can slip through. For small
    stores, pass an explicit `connective_predicates` set; for production
    corpora with hundreds of infons per predicate, the inference converges
    on the right answer automatically as more cassettes land.

    Returns: set of predicate anchor names.
    """
    subjects = _entity_set(m)  # anchors that ACT

    # Gather distinct (subject, object) pairs per predicate from the
    # by_triple parquet shards — stays in index land, no hydration.
    import pyarrow.parquet as pq
    pred_subjects: dict[str, set[str]] = {}
    pred_objects: dict[str, set[str]] = {}
    # How often each non-subject anchor appears as an object
    # (distinct subjects that point to it).
    object_inbound: dict[str, set[str]] = {}
    for c in m.cassettes:
        path = c.get("index_paths", {}).get("by_triple")
        if not path:
            continue
        if "://" in path:
            import fsspec
            fs, rel = fsspec.core.url_to_fs(path)
            tbl = pq.read_table(rel, filesystem=fs,
                                 columns=["predicate", "subject", "object"])
        else:
            tbl = pq.read_table(path,
                                 columns=["predicate", "subject", "object"])
        for p, s, o in zip(tbl.column("predicate").to_pylist(),
                            tbl.column("subject").to_pylist(),
                            tbl.column("object").to_pylist()):
            if p and s and o:
                pred_subjects.setdefault(p, set()).add(s)
                pred_objects.setdefault(p, set()).add(o)
                if s in subjects:  # only count entity-sourced edges
                    object_inbound.setdefault(o, set()).add(s)

    # Infer terminal-entities: object-only anchors that receive actions
    # from a SMALL NUMBER of entities (1-2). A true terminal entity like
    # CATL receives actions from a handful of specific entities that
    # supply/license/etc. to it. A decoy object like 'eu' or
    # 'earnings_call' receives actions from MANY entities (because every
    # entity has a decoy edge pointing at it).
    #
    # Threshold: anchor is a terminal entity if its distinct-entity-sources
    # count is ≤ n_entities * 0.3 AND ≥ 1. This captures specific-purpose
    # terminal nodes without pulling in universal decoy attributes.
    n_entities = max(1, len(subjects))
    inbound_cap = max(2, int(n_entities * 0.3))
    terminal_entities: set[str] = set()
    for o, srcs in object_inbound.items():
        if o in subjects:
            continue
        if 1 <= len(srcs) <= inbound_cap:
            terminal_entities.add(o)

    widened = subjects | terminal_entities

    connective: set[str] = set()
    for p, objs in pred_objects.items():
        if len(objs) < min_occurrences:
            continue
        ratio = sum(1 for o in objs if o in widened) / len(objs)
        if ratio >= min_entity_object_ratio:
            connective.add(p)
    return connective


def _label_from_mass(m: MassFunction,
                     supports_threshold: float = 0.25,
                     refutes_threshold: float = 0.15) -> str:
    if m.supports >= supports_threshold and m.supports > m.refutes:
        return "SUPPORTS"
    if m.refutes >= refutes_threshold and m.refutes > m.supports:
        return "REFUTES"
    return "NOT_ENOUGH_INFO"


def _expand(node: _Node, manifest: Manifest, fetcher: RangeFetcher,
            source: str, target: str,
            entities: set[str],
            connective_predicates: set[str] | None,
            max_children_per_expand: int,
            remaining_budget: int,
            all_sources: list[Infon]) -> None:
    """Expand one node in-place. Adds child nodes for each reachable entity."""
    hits = Query().mentioning(node.anchor).run(manifest)
    if connective_predicates is not None:
        hits = [h for h in hits if h.loc.predicate in connective_predicates]
    used = {i.infon_id for i in node.path_infons()}
    hits = [h for h in hits if h.loc.infon_id not in used]
    hits = hits[: min(max_children_per_expand, remaining_budget)]
    node.expanded = True
    if not hits:
        return
    infons = hydrate_locs(fetcher, manifest, hits)
    all_sources.extend(infons)

    # Group by (S, P, O) triple so later retractions cancel affirmations.
    edges_by_triple: dict[tuple[str, str, str], list[Infon]] = {}
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
            child = _Node(anchor=a, parent=node, edge_infons=edge_infons)
            child.path_mass = chain_mass(
                node.path_edges() + [edge_infons], source, a)
            node.children.append(child)


# ═══════════════════════════════════════════════════════════════════════
# SINGLE-TARGET
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
# GNN INTEGRATION  (optional, lazy-loaded)
# ═══════════════════════════════════════════════════════════════════════
#
# The GNN is a terminal scorer: it sees the chain MCTS discovered and
# emits a verdict. It does NOT guide MCTS expansion (that would be the
# next-level integration, per earlier design notes). Keeping it terminal
# means the symbolic path stays interpretable — callers always get the
# MCTS-chosen sources, even when the GNN's verdict overrides.
#
# Model location: <manifest.root>/_model/gnn.pt
# Absent? use_gnn becomes a no-op.

_GNN_CACHE: dict[str, object] = {}   # root → loaded model


def _load_gnn(root: str):
    """Lazy-load the trained GNN from <root>/_model/gnn.pt.

    Caches per-root so a batch of queries pays the torch.load cost once.
    Returns None if no model is present — callers must handle this."""
    if root in _GNN_CACHE:
        return _GNN_CACHE[root]
    import os
    model_path = os.path.join(root, "_model", "gnn.pt")
    if not os.path.exists(model_path):
        _GNN_CACHE[root] = None
        return None
    try:
        import torch
        from .gnn_encoder import SheafHypergraphEncoder
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        model = SheafHypergraphEncoder(
            hidden_dim=ckpt.get("hidden_dim", 64),
            n_layers=ckpt.get("n_layers", 3),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        _GNN_CACHE[root] = model
        return model
    except Exception:
        _GNN_CACHE[root] = None
        return None


def _infer_relation_kinds_for_gnn(manifest: Manifest,
                                    connective: set[str]) -> dict[str, str]:
    """Map each predicate seen in the manifest to its GNN kind.

    connective predicates are already known; remaining predicates we
    classify as reportive (the GNN's "doesn't propagate" semantics).
    A real deployment would persist this alongside the model; for the
    probe we reconstruct it here."""
    kinds: dict[str, str] = {}
    seen: set[str] = set()
    for c in manifest.cassettes:
        for p in c.get("predicates", []):
            seen.add(p)
    for p in seen:
        if p in connective:
            kinds[p] = "connective"
        else:
            # Default: reportive — GNN dampens messages through this.
            # Terminal vs reportive doesn't matter much here because
            # both are non-propagating; pick reportive as the conservative
            # choice.
            kinds[p] = "reportive"
    return kinds


def _score_with_gnn(manifest: Manifest,
                    path_infons: list,
                    source: str, target: str,
                    *,
                    relation_kinds: dict[str, str] | None,
                    verbose: bool = False) -> MassFunction | None:
    """Encode the chain for the GNN and return a MassFunction over the
    three-class softmax. Returns None if no model is loadable."""
    model = _load_gnn(manifest.root)
    if model is None:
        if verbose:
            print(f"    [gnn] no model at {manifest.root}/_model/gnn.pt")
        return None
    if not path_infons:
        # MCTS found nothing — trivial NEI prior, blend won't change it.
        return MassFunction(theta=1.0)

    # Group path_infons back into edges by (s, p, o) triple. Each edge may
    # have multiple infons (affirmation + later retraction of the same
    # triple).
    from collections import OrderedDict
    edges: "OrderedDict[tuple, list]" = OrderedDict()
    for inf in path_infons:
        k = (inf.subject, inf.predicate, inf.object)
        edges.setdefault(k, []).append(inf)
    path_edges = list(edges.values())

    if relation_kinds is None:
        connective = infer_connective_predicates(manifest)
        relation_kinds = _infer_relation_kinds_for_gnn(manifest, connective)

    try:
        import torch
        from .gnn_encoder import (
            HypergraphBatch, encode_edge, EDGE_FEATURE_DIM, IDX_TO_VERDICT,
        )
    except ImportError:
        return None

    # Build the per-edge feature rows (mirrors encode_chain_from_synth).
    rows = []
    prev_obj = None
    max_len = 12
    for edge_infons in path_edges:
        for inf in sorted(edge_infons, key=lambda i: i.timestamp or ""):
            kind = relation_kinds.get(inf.predicate, "reportive")
            rows.append(encode_edge(
                kind=kind,
                polarity=inf.polarity,
                confidence=inf.confidence,
                gap_days=0,
                is_last=False,
                touches_source=(inf.subject == source or inf.object == source),
                touches_target=(inf.subject == target or inf.object == target),
                connects_prev=(prev_obj is None or inf.subject == prev_obj),
            ))
            prev_obj = inf.object
    if not rows:
        return MassFunction(theta=1.0)

    # Flip the last row's is_last bit to 1. is_last is at position -4
    # (touches_source, touches_target, connects_prev are the trailing 3).
    # Cleaner: reconstruct the last row.
    pad = [0.0] * EDGE_FEATURE_DIM
    rows = rows[-max_len:]
    mask = [1.0] * len(rows)
    while len(rows) < max_len:
        rows.insert(0, pad)
        mask.insert(0, 0.0)

    batch = HypergraphBatch(
        edges=torch.tensor([rows], dtype=torch.float32),
        edge_mask=torch.tensor([mask], dtype=torch.float32),
        verdicts=torch.tensor([0], dtype=torch.long),
    )
    with torch.no_grad():
        out = model(batch)
        probs = torch.softmax(out["logits"], dim=-1)[0].tolist()
    # probs: [SUPPORTS, REFUTES, NOT_ENOUGH_INFO]. Map to MassFunction.
    # NEI goes to θ; SUPPORTS/REFUTES go to their slots. No "uncertain"
    # class from the GNN (its head is 3-way S/R/N).
    return MassFunction(
        supports=probs[0],
        refutes=probs[1],
        theta=probs[2],
    )


def _blend_masses(symbolic: MassFunction, gnn: MassFunction,
                   gnn_weight: float = 0.5) -> MassFunction:
    """Weighted average of two mass functions on the same chain.

    Dempster's rule would be wrong here because the two "sources" aren't
    independent — the GNN is looking at the same path the symbolic method
    walked. A weighted average mirrors how AlphaZero blends policy and
    value networks: λ·v_policy + (1-λ)·v_mcts."""
    w = max(0.0, min(1.0, gnn_weight))
    return MassFunction(
        supports=(1 - w) * symbolic.supports + w * gnn.supports,
        refutes=(1 - w) * symbolic.refutes + w * gnn.refutes,
        uncertain=(1 - w) * symbolic.uncertain + w * getattr(gnn, "uncertain", 0.0),
        theta=(1 - w) * symbolic.theta + w * gnn.theta,
    )


def reason_connectivity(
    manifest: Manifest,
    source: str,
    target: str,
    *,
    budget: int = 20,
    max_hops: int = 3,
    max_children_per_expand: int = 6,
    connective_predicates: set[str] | None = None,
    fetcher: RangeFetcher | None = None,
    verbose: bool = False,
    use_gnn: bool = False,
    gnn_weight: float = 0.5,
    relation_kinds: dict[str, str] | None = None,
) -> Verdict:
    """Answer 'is source connected to target?' via evidence-guided MCTS.

    Args:
      source, target: entity anchors.
      budget: max range gets (hydrations) to spend searching.
      max_hops: cap path length.
      connective_predicates: allow-list of edge types that count as real
        connectivity. If None (the default), auto-inferred from the corpus
        via `infer_connective_predicates(manifest)` — predicates whose
        objects are mostly other entities. Pass an explicit set to override.
      fetcher: optional custom RangeFetcher (S3, LocalFetcher for local).

    Returns: Verdict with label, DS mass (S/R/θ), and the infons along
    the best path to target.
    """
    if connective_predicates is None:
        connective_predicates = infer_connective_predicates(manifest)
    fetcher = fetcher or LocalFetcher()
    before_gets = getattr(fetcher, "requests", 0)
    entities = _entity_set(manifest) | {source, target}
    root = _Node(anchor=source)
    best_mass = MassFunction(theta=1.0)
    best_path_infons: list[Infon] = []
    all_sources: list[Infon] = []
    iters = 0

    while (getattr(fetcher, "requests", 0) - before_gets) < budget and \
          iters < budget * 4:
        iters += 1

        # SELECT: descend by UCB; decisiveness = |S-R|, exploration = θ*sqrt.
        node = root
        while node.expanded and node.children:
            def ucb(n: _Node) -> float:
                if n.visits == 0:
                    return float("inf")
                pv = n.parent.visits if n.parent else 1
                decisive = abs(n.path_mass.supports - n.path_mass.refutes)
                explore = 1.2 * n.path_mass.theta * \
                          math.sqrt(math.log(max(pv, 1)) / n.visits)
                return decisive + explore
            node = max(node.children, key=ucb)

        if len(node.path()) - 1 >= max_hops:
            node.visits += 1
            continue

        # EXPAND
        if not node.expanded:
            remaining = budget - (getattr(fetcher, "requests", 0) - before_gets)
            _expand(node, manifest, fetcher, source, target, entities,
                    connective_predicates, max_children_per_expand,
                    remaining, all_sources)
            if not node.children:
                node.visits += 1
                continue

        # EVALUATE: pick most decisive child; track best path-to-target.
        if not node.children:
            node.visits += 1
            continue
        leaf = max(node.children,
                   key=lambda c: abs(c.path_mass.supports - c.path_mass.refutes))
        value = leaf.path_mass
        if target in leaf.path():
            best_dec = abs(best_mass.supports - best_mass.refutes)
            leaf_dec = abs(value.supports - value.refutes)
            if leaf_dec > best_dec:
                best_mass = value
                best_path_infons = leaf.path_infons()

        # BACKPROP
        n = leaf
        while n is not None:
            n.visits += 1
            n = n.parent

        # Early stop if we've resolved the claim.
        if best_mass.supports > 0.55 or best_mass.refutes > 0.45:
            break

    # If nothing found, explicit NEI with θ=1.0 — not a confident zero.
    if best_mass.supports == 0.0 and best_mass.refutes == 0.0:
        final_mass = MassFunction(theta=1.0)
        sources: list[Infon] = []
    else:
        final_mass = best_mass
        sources = best_path_infons

    # ── GNN terminal scorer ────────────────────────────────────────────
    # Re-scores the MCTS-discovered chain with the trained sheaf GNN.
    # Combined with the symbolic mass via weighted average, NOT Dempster
    # (the two aren't independent — they're looking at the same chain).
    # Falls back to symbolic-only if no model is present or GNN errors.
    gnn_mass: MassFunction | None = None
    if use_gnn:
        gnn_mass = _score_with_gnn(
            manifest, sources, source, target,
            relation_kinds=relation_kinds,
            verbose=verbose,
        )
        if gnn_mass is not None:
            final_mass = _blend_masses(final_mass, gnn_mass, gnn_weight)

    if verbose:
        print(f"    [mcts] iters={iters} gets={getattr(fetcher, 'requests', 0) - before_gets} "
              f"best={final_mass.to_dict()}")
        if gnn_mass is not None:
            print(f"    [gnn] mass={gnn_mass.to_dict()}  "
                  f"blended@{gnn_weight}")

    return Verdict(
        label=_label_from_mass(final_mass),
        mass=final_mass,
        n_candidates=len(all_sources),
        n_hydrated=len(all_sources),
        range_gets=getattr(fetcher, "requests", 0) - before_gets,
        sources=sources,
    )


# ═══════════════════════════════════════════════════════════════════════
# MULTI-TARGET
# ═══════════════════════════════════════════════════════════════════════

def reason_any_target(
    manifest: Manifest,
    source: str,
    targets: set[str],
    *,
    budget: int = 20,
    max_hops: int = 3,
    max_children_per_expand: int = 6,
    connective_predicates: set[str] | None = None,
    fetcher: RangeFetcher | None = None,
    verbose: bool = False,
) -> dict[str, Verdict]:
    """Resolve connectivity from `source` to every target in one tree walk.

    Cost is ≈ constant in |targets| — each range-get contributes to the
    search for every target. Reachable targets resolve as paths cross them;
    unreachable ones remain NEI after the tree exhausts or budget runs out.

    Returns dict[target → Verdict], one per target.
    """
    if connective_predicates is None:
        connective_predicates = infer_connective_predicates(manifest)
    fetcher = fetcher or LocalFetcher()
    before_gets = getattr(fetcher, "requests", 0)
    entities = _entity_set(manifest) | {source} | set(targets)
    root = _Node(anchor=source)

    # Per-target best (mass, path_infons). Updated whenever any leaf's
    # path passes through a target.
    best: dict[str, tuple[MassFunction, list[Infon]]] = {
        t: (MassFunction(theta=1.0), []) for t in targets
    }
    all_sources: list[Infon] = []
    iters = 0

    def node_score(n: _Node) -> float:
        """Prefer nodes whose path already confidently resolves SOMETHING
        but whose descendants could still resolve an uncertain target.
        Uses max decisiveness over any target on the path."""
        if n.visits == 0:
            return float("inf")
        on_path = set(n.path()) & targets
        if on_path:
            decisive = max(
                abs(best[t][0].supports - best[t][0].refutes)
                for t in on_path
            )
        else:
            decisive = abs(n.path_mass.supports - n.path_mass.refutes)
        pv = n.parent.visits if n.parent else 1
        theta = n.path_mass.theta
        explore = 1.2 * theta * math.sqrt(math.log(max(pv, 1)) / n.visits)
        return decisive + explore

    def all_resolved() -> bool:
        for _, (mass, _) in best.items():
            if mass.supports < 0.4 and mass.refutes < 0.4:
                return False
        return True

    while (getattr(fetcher, "requests", 0) - before_gets) < budget and \
          iters < budget * 4:
        iters += 1

        node = root
        while node.expanded and node.children:
            node = max(node.children, key=node_score)

        if len(node.path()) - 1 >= max_hops:
            node.visits += 1
            continue

        if not node.expanded:
            remaining = budget - (getattr(fetcher, "requests", 0) - before_gets)
            # Target arg here is cosmetic — chain_mass recomputes per child.
            _expand(node, manifest, fetcher, source, next(iter(targets)),
                    entities, connective_predicates,
                    max_children_per_expand, remaining, all_sources)
            if not node.children:
                node.visits += 1
                continue

        if not node.children:
            node.visits += 1
            continue
        leaf = max(node.children,
                   key=lambda c: abs(c.path_mass.supports - c.path_mass.refutes))

        # Attribute leaf's prefix to every target it crosses.
        leaf_targets = set(leaf.path()) & targets
        for t in leaf_targets:
            prefix_edges: list[list[Infon]] = []
            for edge in leaf.path_edges():
                prefix_edges.append(edge)
                reached = {source}
                for e in prefix_edges:
                    for inf in e:
                        reached.add(inf.subject); reached.add(inf.object)
                if t in reached:
                    mass_t = chain_mass(prefix_edges, source, t)
                    curr_mass, _ = best[t]
                    if abs(mass_t.supports - mass_t.refutes) > \
                       abs(curr_mass.supports - curr_mass.refutes):
                        best[t] = (mass_t,
                                   [i for e in prefix_edges for i in e])
                    break

        n = leaf
        while n is not None:
            n.visits += 1
            n = n.parent

        if all_resolved():
            break

    total_gets = getattr(fetcher, "requests", 0) - before_gets
    out: dict[str, Verdict] = {}
    for t in targets:
        mass, sources = best[t]
        out[t] = Verdict(
            label=_label_from_mass(mass),
            mass=mass,
            n_candidates=len(all_sources),
            n_hydrated=len(all_sources),
            range_gets=total_gets,
            sources=sources,
        )

    if verbose:
        resolved = sum(1 for t in targets
                       if out[t].label in ("SUPPORTS", "REFUTES"))
        print(f"    [multi] iters={iters} gets={total_gets} "
              f"resolved={resolved}/{len(targets)}")
    return out


# ═══════════════════════════════════════════════════════════════════════
# BUILD NOTES  (why the non-obvious choices above are needed)
# ═══════════════════════════════════════════════════════════════════════
#
# 1. Edge-level triple grouping: without grouping by (S,P,O), a later
#    negation of the same triple becomes a sibling path instead of
#    cancelling the original. Tested via the retraction-at-HEAD case —
#    time-travel to the pre-retraction snapshot correctly returns
#    SUPPORTS.
#
# 2. min/max over chain mass (not Dempster): Dempster combine across edges
#    was amplifying multiple affirmed edges into higher S, treating
#    independent hops as corroborating evidence. Semantically wrong —
#    a chain is a conjunction. min/max gives a 2-hop S=0.48 vs 1-hop
#    S=0.58, which correctly prefers shorter paths for the same endpoint.
#
# 3. connective_predicates filter at the index: without it, scale tests
#    hallucinated chains through non-connective edges (toyota→mention→honda
#    →mention→ford). The filter is schema knowledge — which relation types
#    indicate real connectivity — and is cheap because predicate is a
#    pushdown column in by_anchor.parquet.
#
# 4. Entity-set filter for traversal candidates: without it, decoy objects
#    (countries, events) that appear in many unrelated infons become
#    traversal hubs. Restricting to "subjects seen anywhere" + the query's
#    endpoints excludes these while still allowing terminal entities
#    (like CATL in a supply chain) to be reached.

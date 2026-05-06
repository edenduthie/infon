"""Geometric deep learning + IKL logical primitives over the infon hypergraph.

Builds a heterogeneous graph from the store's anchors, infons, and typed edges,
runs RGCN-style message passing where aggregation functions implement IKL
(IKRIS Knowledge Language) connectives, and outputs refined Dempster-Shafer
mass functions compatible with the existing belief framework.

Three layers:

1. **HypergraphBuilder** — constructs node features (from SPLADE projections)
   and typed adjacency from the store.

2. **TypedMessagePassing** — per-relation-type weight matrices with IKL
   aggregators: that (reification), and/or/not (lattice ops on masses),
   if/iff (asymmetric/symmetric attention), forall/exists (typed domain
   quantification), ist (situation-contextualized gating).

3. **HypergraphReasoner** — end-to-end: store → graph → message passing →
   refined MassFunction outputs, pluggable into GraphMCTS or verify_claim.

IKL reference: the IKRIS Knowledge Language extends KIF with:
  - (that φ)         reify a proposition as a term
  - (and φ ψ ...)    conjunction
  - (or φ ψ ...)     disjunction
  - (not φ)          negation
  - (if φ ψ)         material conditional
  - (iff φ ψ)        biconditional
  - (forall (?x T) φ) universal over type T
  - (exists (?x T) φ) existential over type T
  - (ist s φ)        φ holds in situation/context s

References
----------
Hayes, P., & Menzel, C. (2006). "IKL Guide."
    IKRIS Knowledge Language specification.
Schlichtkrull, M. et al. (2018). "Modeling Relational Data with Graph
    Convolutional Networks." ESWC 2018. arXiv:1703.06103 (R-GCN baseline).
Shafer, G. (1976). "A Mathematical Theory of Evidence." Princeton U. Press
    (Dempster-Shafer mass functions).
Barwise, J., & Perry, J. (1983). "Situations and Attitudes." MIT Press
    (the `ist` situation operator).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from collections import defaultdict, Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .atom import Infon, Edge
from .dempster_shafer import MassFunction, combine_dempster, combine_multiple


# ═══════════════════════════════════════════════════════════════════════
# 1. HYPERGRAPH CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════

RELATION_TYPES = [
    "INITIATES",    # subject → infon
    "ASSERTS",      # infon → predicate
    "TARGETS",      # infon → object
    "NEXT",         # infon → infon (temporal)
    "ENTAILS",      # infon → infon (logical)
    "LOCATED_AT",   # infon → location
    "CONTRADICTS",  # infon → infon (semantic contradiction)
    "CAUSES",       # infon → infon (causal: earlier action → later outcome)
]
REL_TO_IDX = {r: i for i, r in enumerate(RELATION_TYPES)}
NUM_RELATIONS = len(RELATION_TYPES)

NODE_TYPES = ["anchor", "infon"]


@dataclass
class HyperGraph:
    """A typed heterogeneous graph built from the store."""
    node_ids: list[str]
    node_types: list[str]                    # "anchor" or "infon"
    node_features: torch.Tensor              # (n_nodes, feat_dim)
    edge_index: torch.Tensor                 # (2, n_edges) — source, target indices
    edge_types: torch.Tensor                 # (n_edges,) — relation type indices
    edge_weights: torch.Tensor               # (n_edges,)

    # Metadata for IKL quantification
    anchor_type_groups: dict[str, list[int]] # anchor_type → [node indices]
    infon_indices: list[int]                 # node indices that are infons
    infon_map: dict[str, int]                # infon_id → node index
    anchor_map: dict[str, int]               # anchor_name → node index

    # Situation grounding for ist()
    situation_features: torch.Tensor | None  # (n_infons, sit_dim) temporal+spatial

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def n_edges(self) -> int:
        return self.edge_index.shape[1] if self.edge_index.numel() > 0 else 0

    def to(self, device: str | torch.device) -> HyperGraph:
        self.node_features = self.node_features.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_types = self.edge_types.to(device)
        self.edge_weights = self.edge_weights.to(device)
        if self.situation_features is not None:
            self.situation_features = self.situation_features.to(device)
        return self


class HypergraphBuilder:
    """Build a HyperGraph from the store + encoder.

    Optionally consumes a trained SentenceEmbedder (from `cognition.embedder`)
    that replaces the default seeded-random-projection feature reduction.
    When a trained embedder is attached, infon features are the embedder's
    64-d node output, which carries task-aware structure (role, anchor,
    template clustering) rather than just random-projected SPLADE.
    """

    def __init__(self, store, encoder, schema, embedder=None):
        self.store = store
        self.encoder = encoder
        self.schema = schema
        # Optional trained sentence embedder (see cognition.embedder).
        # When set, it replaces _reduce_sparse for infon features.
        self.embedder = embedder

    def build(self, infons: list[Infon] | None = None,
              edges: list[Edge] | None = None,
              max_infons: int = 500,
              feature_dim: int = 64) -> HyperGraph:
        """Construct the heterogeneous hypergraph.

        Nodes: anchors (from schema) + infons (from store).
        Edges: spoke edges (INITIATES/ASSERTS/TARGETS) + NEXT/ENTAILS.
        Features: SPLADE-projected anchor activations, zero-padded to feature_dim.
        """
        if infons is None:
            infons = self.store.query_infons(limit=max_infons)
        if edges is None:
            edges = self.store.get_edges(limit=max_infons * 5)

        anchor_names = list(self.schema.names)
        n_anchors = len(anchor_names)

        # Node registries
        node_ids: list[str] = []
        node_types: list[str] = []
        anchor_map: dict[str, int] = {}
        infon_map: dict[str, int] = {}

        # Register anchors
        for name in anchor_names:
            anchor_map[name] = len(node_ids)
            node_ids.append(f"anchor:{name}")
            node_types.append("anchor")

        # Register infons
        for inf in infons:
            if inf.infon_id not in infon_map:
                infon_map[inf.infon_id] = len(node_ids)
                node_ids.append(f"infon:{inf.infon_id[:12]}")
                node_types.append("infon")

        n_nodes = len(node_ids)
        infon_indices = [infon_map[inf.infon_id] for inf in infons
                         if inf.infon_id in infon_map]

        # Build node features
        node_feat = torch.zeros(n_nodes, feature_dim)

        # Anchor features: encode anchor names through SPLADE, project to feature_dim
        if anchor_names:
            anchor_texts = [self.schema.anchors[n].get("tokens", [n])[0]
                            for n in anchor_names]
            sparse = self.encoder.encode_sparse(anchor_texts)
            # PCA-style dimensionality reduction: take top-k activations
            anchor_feat = self._reduce_sparse(sparse, feature_dim)
            node_feat[:n_anchors] = anchor_feat

        # Infon features: encode sentences, reduce.
        # If a trained SentenceEmbedder is attached, use its node head
        # to produce semantically-aware 64-d features; otherwise fall
        # back to the seeded random projection.
        infon_sentences = []
        infon_node_indices = []
        for inf in infons:
            if inf.sentence and inf.infon_id in infon_map:
                infon_sentences.append(inf.sentence)
                infon_node_indices.append(infon_map[inf.infon_id])
        if infon_sentences:
            sparse = self.encoder.encode_sparse(infon_sentences)
            if self.embedder is not None:
                sparse_t = torch.from_numpy(sparse).float()
                infon_feat = self.embedder.node_embedding(sparse_t)
                # Pad or crop to feature_dim if mismatched
                d = infon_feat.shape[-1]
                if d < feature_dim:
                    pad = torch.zeros(infon_feat.shape[0],
                                      feature_dim - d)
                    infon_feat = torch.cat([infon_feat, pad], dim=-1)
                elif d > feature_dim:
                    infon_feat = infon_feat[:, :feature_dim]
            else:
                infon_feat = self._reduce_sparse(sparse, feature_dim)
            for i, node_idx in enumerate(infon_node_indices):
                node_feat[node_idx] = infon_feat[i]

        # Build edges
        src_list, tgt_list, type_list, weight_list = [], [], [], []

        # Spoke edges from infons
        for inf in infons:
            inf_idx = infon_map.get(inf.infon_id)
            if inf_idx is None:
                continue

            # INITIATES: subject → infon
            s_idx = anchor_map.get(inf.subject)
            if s_idx is not None:
                src_list.append(s_idx)
                tgt_list.append(inf_idx)
                type_list.append(REL_TO_IDX["INITIATES"])
                weight_list.append(inf.confidence)

            # ASSERTS: infon → predicate
            p_idx = anchor_map.get(inf.predicate)
            if p_idx is not None:
                src_list.append(inf_idx)
                tgt_list.append(p_idx)
                type_list.append(REL_TO_IDX["ASSERTS"])
                weight_list.append(inf.confidence)

            # TARGETS: infon → object
            o_idx = anchor_map.get(inf.object)
            if o_idx is not None:
                src_list.append(inf_idx)
                tgt_list.append(o_idx)
                type_list.append(REL_TO_IDX["TARGETS"])
                weight_list.append(inf.confidence)

        # Store edges (NEXT, ENTAILS, LOCATED_AT)
        for edge in edges:
            rel_idx = REL_TO_IDX.get(edge.edge_type)
            if rel_idx is None:
                continue
            s = infon_map.get(edge.source, anchor_map.get(edge.source))
            t = infon_map.get(edge.target, anchor_map.get(edge.target))
            if s is not None and t is not None:
                src_list.append(s)
                tgt_list.append(t)
                type_list.append(rel_idx)
                weight_list.append(edge.weight)

        if src_list:
            edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long)
            edge_type_tensor = torch.tensor(type_list, dtype=torch.long)
            edge_weight_tensor = torch.tensor(weight_list, dtype=torch.float32)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_type_tensor = torch.zeros(0, dtype=torch.long)
            edge_weight_tensor = torch.zeros(0, dtype=torch.float32)

        # Anchor type groups for forall/exists quantification
        anchor_type_groups: dict[str, list[int]] = defaultdict(list)
        for name in anchor_names:
            atype = self.schema.types.get(name, "feature")
            anchor_type_groups[atype].append(anchor_map[name])

        # Situation features for ist() — temporal + spatial encoding
        situation_features = self._build_situation_features(
            infons, infon_map, feature_dim=16,
        )

        return HyperGraph(
            node_ids=node_ids,
            node_types=node_types,
            node_features=node_feat,
            edge_index=edge_index,
            edge_types=edge_type_tensor,
            edge_weights=edge_weight_tensor,
            anchor_type_groups=dict(anchor_type_groups),
            infon_indices=infon_indices,
            infon_map=infon_map,
            anchor_map=anchor_map,
            situation_features=situation_features,
        )

    def _reduce_sparse(self, sparse_matrix: np.ndarray,
                       target_dim: int) -> torch.Tensor:
        """Reduce sparse SPLADE vectors to dense features via top-k + projection."""
        n, vocab = sparse_matrix.shape
        if vocab <= target_dim:
            out = np.zeros((n, target_dim), dtype=np.float32)
            out[:, :vocab] = sparse_matrix
            return torch.from_numpy(out)

        # Random stable projection (seeded for reproducibility)
        rng = np.random.RandomState(42)
        proj = rng.randn(vocab, target_dim).astype(np.float32)
        proj /= np.sqrt(vocab)
        reduced = sparse_matrix.astype(np.float32) @ proj
        # L2 normalize
        norms = np.linalg.norm(reduced, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        reduced /= norms
        return torch.from_numpy(reduced)

    def _build_situation_features(self, infons: list[Infon],
                                  infon_map: dict[str, int],
                                  feature_dim: int = 16) -> torch.Tensor:
        """Encode temporal + spatial grounding for ist() operator."""
        n_nodes = max(infon_map.values()) + 1 if infon_map else 0
        sit = torch.zeros(n_nodes, feature_dim)

        TENSE_MAP = {"past": 0, "present": 1, "future": 2,
                      "conditional": 3, "present_continuous": 4, "unknown": 5}
        PRECISION_MAP = {"year": 0, "half": 1, "quarter": 2,
                          "month": 3, "unknown": 4}

        for inf in infons:
            idx = infon_map.get(inf.infon_id)
            if idx is None:
                continue

            # Tense one-hot (6 dims)
            t = TENSE_MAP.get(inf.tense, 5)
            if t < 6:
                sit[idx, t] = 1.0

            # Precision one-hot (5 dims, offset 6)
            p = PRECISION_MAP.get(inf.precision, 4)
            if p < 5:
                sit[idx, 6 + p] = 1.0

            # Polarity (1 dim, offset 11)
            sit[idx, 11] = float(inf.polarity)

            # Confidence (1 dim, offset 12)
            sit[idx, 12] = inf.confidence

            # Has location (1 dim, offset 13)
            sit[idx, 13] = 1.0 if inf.locations else 0.0

            # Has temporal ref (1 dim, offset 14)
            sit[idx, 14] = 1.0 if inf.temporal_refs else 0.0

            # Importance (1 dim, offset 15)
            sit[idx, 15] = inf.importance

        return sit


# ═══════════════════════════════════════════════════════════════════════
# 2. IKL PRIMITIVES AS AGGREGATION OPERATORS
# ═══════════════════════════════════════════════════════════════════════

class IKLThat(nn.Module):
    """(that φ) — reification: project an infon embedding into the anchor
    feature space so it can participate as a first-class term.

    Takes an infon node's hidden state and produces a "proposition embedding"
    that can be referenced by other infons (enables higher-order statements
    like "Toyota believes that batteries will improve").
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.reify = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

    def forward(self, infon_h: torch.Tensor) -> torch.Tensor:
        """(n_infons, hidden) → (n_infons, hidden) reified proposition embeddings."""
        return self.reify(infon_h)


class IKLAnd(nn.Module):
    """(and φ ψ ...) — conjunction: min-pooling over neighbor masses.

    In DS terms, conjunction of independent evidence narrows belief.
    Geometrically, we take element-wise minimum (t-norm) of neighbor
    embeddings, which preserves only features present in ALL neighbors.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, neighbor_h: torch.Tensor,
                weights: torch.Tensor | None = None) -> torch.Tensor:
        """(n_neighbors, hidden) → (hidden,) conjunctive aggregation."""
        if neighbor_h.shape[0] == 0:
            return torch.zeros(neighbor_h.shape[-1], device=neighbor_h.device)
        gated = torch.sigmoid(self.gate(neighbor_h))
        conjunct = neighbor_h * gated
        if weights is not None:
            weights = weights.unsqueeze(-1)
            conjunct = conjunct * weights
        return conjunct.min(dim=0).values


class IKLOr(nn.Module):
    """(or φ ψ ...) — disjunction: max-pooling over neighbor masses.

    Disjunction widens belief — any one piece of evidence suffices.
    Element-wise max (t-conorm) retains features present in ANY neighbor.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, neighbor_h: torch.Tensor,
                weights: torch.Tensor | None = None) -> torch.Tensor:
        """(n_neighbors, hidden) → (hidden,) disjunctive aggregation."""
        if neighbor_h.shape[0] == 0:
            return torch.zeros(neighbor_h.shape[-1], device=neighbor_h.device)
        gated = torch.sigmoid(self.gate(neighbor_h))
        disjunct = neighbor_h * gated
        if weights is not None:
            disjunct = disjunct * weights.unsqueeze(-1)
        return disjunct.max(dim=0).values


class IKLNot(nn.Module):
    """(not φ) — negation: learned embedding inversion + DS mass swap.

    Flips the geometric direction of the embedding (learned, not just
    negation) and swaps supports↔refutes in the mass readout.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.negate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """(*, hidden) → (*, hidden) negated embedding."""
        return self.negate(h) - h


class IKLIf(nn.Module):
    """(if φ ψ) — material conditional: asymmetric attention.

    Models "if premise then conclusion" via cross-attention where the
    premise gates what information flows to the conclusion. Implements
    the IKL conditional as: the conclusion's embedding is modulated by
    how much the premise "permits" it.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W_premise = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_conclusion = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = math.sqrt(hidden_dim)

    def forward(self, premise_h: torch.Tensor,
                conclusion_h: torch.Tensor) -> torch.Tensor:
        """(batch, hidden) × (batch, hidden) → (batch, hidden)."""
        attn = (self.W_premise(premise_h) * self.W_conclusion(conclusion_h)).sum(-1, keepdim=True)
        gate = torch.sigmoid(attn / self.scale)
        return conclusion_h * gate + premise_h * (1.0 - gate)


class IKLIff(nn.Module):
    """(iff φ ψ) — biconditional: symmetric exchange.

    Both directions must hold: computes bidirectional gating so that
    the result is strong only when both embeddings are mutually consistent.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fwd = IKLIf(hidden_dim)
        self.bwd = IKLIf(hidden_dim)

    def forward(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        """(batch, hidden) × (batch, hidden) → (batch, hidden)."""
        fwd = self.fwd(h1, h2)
        bwd = self.bwd(h2, h1)
        return (fwd + bwd) * 0.5


class IKLForall(nn.Module):
    """(forall (?x T) φ) — universal quantification over typed domain.

    Aggregates over ALL anchors of type T using conjunction (IKLAnd).
    "For all actors x, φ(x) holds" means the minimum signal across actors.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.conjunction = IKLAnd(hidden_dim)

    def forward(self, domain_h: torch.Tensor,
                weights: torch.Tensor | None = None) -> torch.Tensor:
        """(n_domain, hidden) → (hidden,) universal over the domain."""
        return self.conjunction(domain_h, weights)


class IKLExists(nn.Module):
    """(exists (?x T) φ) — existential quantification over typed domain.

    Aggregates over anchors of type T using disjunction (IKLOr).
    "There exists an actor x such that φ(x)" means max signal across actors.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.disjunction = IKLOr(hidden_dim)

    def forward(self, domain_h: torch.Tensor,
                weights: torch.Tensor | None = None) -> torch.Tensor:
        """(n_domain, hidden) → (hidden,) existential over the domain."""
        return self.disjunction(domain_h, weights)


class IKLIst(nn.Module):
    """(ist s φ) — situation operator: contextualize φ in situation s.

    Gates the proposition embedding by situation features (temporal,
    spatial, polarity grounding) so the same logical content has
    different effective embeddings in different situations.
    """

    def __init__(self, hidden_dim: int, situation_dim: int = 16):
        super().__init__()
        self.sit_proj = nn.Sequential(
            nn.Linear(situation_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, h: torch.Tensor,
                sit: torch.Tensor) -> torch.Tensor:
        """(batch, hidden) × (batch, sit_dim) → (batch, hidden)."""
        gate = self.sit_proj(sit)
        return h * gate


# ═══════════════════════════════════════════════════════════════════════
# 3. TYPED MESSAGE PASSING LAYER (RGCN + IKL AGGREGATORS)
# ═══════════════════════════════════════════════════════════════════════

class TypedMessagePassingLayer(nn.Module):
    """One layer of RGCN-style message passing with IKL aggregation.

    Each relation type r has its own weight matrix W_r. Messages from
    neighbors of type r are transformed by W_r, then aggregated using
    the IKL operator appropriate to the edge semantics:

    - INITIATES/ASSERTS/TARGETS (spoke edges): IKLAnd — the triple's
      components must jointly support the infon (conjunction)
    - NEXT: IKLIf — temporal precedence is conditional
    - ENTAILS: IKLIf — logical implication is conditional
    - LOCATED_AT: IKLIst — spatial grounding contextualizes

    Self-loop via identity preserves the node's own features.
    """

    def __init__(self, in_dim: int, out_dim: int,
                 n_relations: int = NUM_RELATIONS,
                 situation_dim: int = 16):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Per-relation-type weight matrices
        self.W_rel = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias=False)
            for _ in range(n_relations)
        ])

        # Self-loop
        self.W_self = nn.Linear(in_dim, out_dim, bias=False)

        # IKL aggregators
        self.ikl_and = IKLAnd(out_dim)
        self.ikl_or = IKLOr(out_dim)
        self.ikl_if = IKLIf(out_dim)
        self.ikl_ist = IKLIst(out_dim, situation_dim)

        # Layer norm + activation
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor,
                edge_types: torch.Tensor, edge_weights: torch.Tensor,
                situation_features: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            h: (n_nodes, in_dim) node features
            edge_index: (2, n_edges) source, target
            edge_types: (n_edges,) relation type indices
            edge_weights: (n_edges,) confidence weights
            situation_features: (n_nodes, sit_dim) for ist gating

        Returns:
            h_new: (n_nodes, out_dim) updated features
        """
        n = h.shape[0]
        out = self.W_self(h)  # self-loop baseline

        if edge_index.numel() == 0:
            return self.norm(F.relu(out))

        src, tgt = edge_index[0], edge_index[1]

        # Group edges by target node
        # For each target, collect {relation_type: [(source_idx, weight), ...]}
        target_messages: dict[int, dict[int, list[tuple[int, float]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for e in range(edge_index.shape[1]):
            s, t, r = int(src[e]), int(tgt[e]), int(edge_types[e])
            w = float(edge_weights[e])
            target_messages[t][r].append((s, w))

        # Compute messages and aggregate per target node
        for t_idx, rel_groups in target_messages.items():
            aggregated = []
            for r, src_list in rel_groups.items():
                if not src_list:
                    continue
                src_indices = [s for s, _ in src_list]
                weights = torch.tensor([w for _, w in src_list],
                                       device=h.device, dtype=torch.float32)

                # Transform source features through relation-specific W
                src_h = self.W_rel[r](h[src_indices])

                # Aggregate using IKL operator based on relation semantics
                if r in (REL_TO_IDX["INITIATES"], REL_TO_IDX["ASSERTS"],
                         REL_TO_IDX["TARGETS"]):
                    # Spoke edges: conjunction — triple components must agree
                    msg = self.ikl_and(src_h, weights)
                elif r == REL_TO_IDX["NEXT"]:
                    # Temporal: conditional — past gates present
                    tgt_h = self.W_rel[r](h[t_idx].unsqueeze(0))
                    premise = src_h.mean(dim=0, keepdim=True)
                    msg = self.ikl_if(premise, tgt_h).squeeze(0)
                elif r == REL_TO_IDX["ENTAILS"]:
                    # Logical implication: conditional
                    tgt_h = self.W_rel[r](h[t_idx].unsqueeze(0))
                    premise = src_h.mean(dim=0, keepdim=True)
                    msg = self.ikl_if(premise, tgt_h).squeeze(0)
                elif r == REL_TO_IDX["LOCATED_AT"]:
                    # Spatial: situation gating
                    msg = src_h.mean(dim=0)
                    if situation_features is not None and t_idx < situation_features.shape[0]:
                        sit = situation_features[t_idx].unsqueeze(0)
                        msg = self.ikl_ist(msg.unsqueeze(0), sit).squeeze(0)
                else:
                    msg = src_h.mean(dim=0)

                aggregated.append(msg)

            if aggregated:
                # Combine messages from different relation types via sum
                combined = torch.stack(aggregated).sum(dim=0)
                out[t_idx] = out[t_idx] + combined

        return self.norm(F.relu(out))


class SheafMessagePassingLayer(nn.Module):
    """Sheaf-theoretic message passing with per-relation restriction maps.

    Generalizes TypedMessagePassingLayer by replacing each relation's single
    weight matrix W_r with a pair of restriction maps (P_forward_r,
    P_backward_r). Each edge (s -r-> t) carries an "edge stalk" into which
    both endpoints project: the source via P_forward_r, the target via
    P_backward_r. When the two projections agree at the stalk, the sheaf
    admits a consistent global section there — i.e. the relation's view of
    the endpoints is coherent.

    Why this helps vs plain R-GCN:
    - A single W_r forces the same linear view for everyone listening on
      relation r. Forward/backward decoupling lets an edge apply one view
      when the source "speaks" and a different one when the target "reads".
    - The sheaf Laplacian (edge-wise disagreement) is an *unsupervised*
      regularizer: minimizing it pushes embeddings toward structurally
      coherent node features without any label signal.

    References
    ----------
    Bodnar, C., Di Giovanni, F., Chamberlain, B. P., Liò, P., & Bronstein,
        M. M. (2022). "Neural Sheaf Diffusion: A Topological Perspective
        on Heterophily and Oversmoothing in GNNs." NeurIPS 2022.
        arXiv:2202.04579
    Hansen, J., & Ghrist, R. (2019). "Toward a spectral theory of cellular
        sheaves." Journal of Applied and Computational Topology, 3(4).
    """

    def __init__(self, in_dim: int, out_dim: int,
                 n_relations: int = NUM_RELATIONS,
                 situation_dim: int = 16):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_relations = n_relations

        # Initialize restriction maps near identity so the sheaf layer
        # starts out approximately equal to R-GCN with W_r = I. Tiny
        # Gaussian noise breaks the forward/backward symmetry.
        def _init_restriction() -> nn.Parameter:
            mat = torch.zeros(out_dim, in_dim)
            m = min(out_dim, in_dim)
            mat[:m, :m] = torch.eye(m)
            return nn.Parameter(mat + 0.01 * torch.randn(out_dim, in_dim))

        self.P_forward = nn.ParameterList(
            [_init_restriction() for _ in range(n_relations)]
        )
        self.P_backward = nn.ParameterList(
            [_init_restriction() for _ in range(n_relations)]
        )

        self.W_self = nn.Linear(in_dim, out_dim, bias=False)

        # IKL aggregators — same semantics as the R-GCN layer
        self.ikl_and = IKLAnd(out_dim)
        self.ikl_or = IKLOr(out_dim)
        self.ikl_if = IKLIf(out_dim)
        self.ikl_ist = IKLIst(out_dim, situation_dim)

        self.norm = nn.LayerNorm(out_dim)

    def _restrict_forward(self, r: int, h_src: torch.Tensor) -> torch.Tensor:
        return h_src @ self.P_forward[r].T

    def _restrict_backward(self, r: int, h_tgt: torch.Tensor) -> torch.Tensor:
        return h_tgt @ self.P_backward[r].T

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor,
                edge_types: torch.Tensor, edge_weights: torch.Tensor,
                situation_features: torch.Tensor | None = None) -> torch.Tensor:
        n = h.shape[0]
        out = self.W_self(h)

        if edge_index.numel() == 0:
            return self.norm(F.relu(out))

        src, tgt = edge_index[0], edge_index[1]

        target_messages: dict[int, dict[int, list[tuple[int, float]]]] = \
            defaultdict(lambda: defaultdict(list))
        for e in range(edge_index.shape[1]):
            s, t, r = int(src[e]), int(tgt[e]), int(edge_types[e])
            w = float(edge_weights[e])
            target_messages[t][r].append((s, w))

        for t_idx, rel_groups in target_messages.items():
            aggregated = []
            for r, src_list in rel_groups.items():
                if not src_list:
                    continue
                src_indices = [s for s, _ in src_list]
                weights = torch.tensor([w for _, w in src_list],
                                       device=h.device, dtype=torch.float32)

                # Project source through the FORWARD restriction map.
                src_h = self._restrict_forward(r, h[src_indices])

                if r in (REL_TO_IDX["INITIATES"], REL_TO_IDX["ASSERTS"],
                         REL_TO_IDX["TARGETS"]):
                    msg = self.ikl_and(src_h, weights)
                elif r == REL_TO_IDX["NEXT"]:
                    tgt_h = self._restrict_backward(r, h[t_idx].unsqueeze(0))
                    premise = src_h.mean(dim=0, keepdim=True)
                    msg = self.ikl_if(premise, tgt_h).squeeze(0)
                elif r == REL_TO_IDX["ENTAILS"]:
                    tgt_h = self._restrict_backward(r, h[t_idx].unsqueeze(0))
                    premise = src_h.mean(dim=0, keepdim=True)
                    msg = self.ikl_if(premise, tgt_h).squeeze(0)
                elif r == REL_TO_IDX["LOCATED_AT"]:
                    msg = src_h.mean(dim=0)
                    if (situation_features is not None
                            and t_idx < situation_features.shape[0]):
                        sit = situation_features[t_idx].unsqueeze(0)
                        msg = self.ikl_ist(msg.unsqueeze(0), sit).squeeze(0)
                else:
                    msg = src_h.mean(dim=0)

                aggregated.append(msg)

            if aggregated:
                combined = torch.stack(aggregated).sum(dim=0)
                out[t_idx] = out[t_idx] + combined

        return self.norm(F.relu(out))

    def sheaf_discrepancy(self, h: torch.Tensor,
                          edge_index: torch.Tensor,
                          edge_types: torch.Tensor,
                          edge_weights: torch.Tensor | None = None,
                          ) -> torch.Tensor:
        """Sheaf-Laplacian edge-discrepancy penalty.

        For every edge (s -r-> t):
            d_e = || P_forward[r] @ h[s] - P_backward[r] @ h[t] ||²

        The weighted mean of d_e. Zero means every edge's two endpoints
        project to the *same* point at the edge stalk — a perfectly
        coherent global section. Used as an unsupervised regularizer.
        """
        if edge_index.numel() == 0:
            return torch.zeros((), device=h.device)

        src, tgt = edge_index[0], edge_index[1]
        diffs: list[torch.Tensor] = []
        weights_list: list[float] = []
        for e in range(edge_index.shape[1]):
            s = int(src[e]); t = int(tgt[e]); r = int(edge_types[e])
            proj_s = h[s] @ self.P_forward[r].T
            proj_t = h[t] @ self.P_backward[r].T
            diffs.append((proj_s - proj_t).pow(2).sum())
            if edge_weights is not None:
                weights_list.append(float(edge_weights[e]))
            else:
                weights_list.append(1.0)

        diffs_t = torch.stack(diffs)
        w_t = torch.tensor(weights_list, device=h.device, dtype=torch.float32)
        w_t = w_t / (w_t.sum() + 1e-8)
        return (diffs_t * w_t).sum()


# ═══════════════════════════════════════════════════════════════════════
# 4. MASS READOUT — EMBEDDINGS → DS MASS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

class MassReadout(nn.Module):
    """Map node embeddings to Dempster-Shafer mass functions.

    Output: (supports, refutes, uncertain, theta) summing to 1.
    Uses softmax over 4 logits, with a temperature parameter that
    controls how peaked vs uniform (ignorant) the masses are.
    """

    def __init__(self, hidden_dim: int, temperature: float = 1.0):
        super().__init__()
        self.readout = nn.Linear(hidden_dim, 4)
        self.temperature = temperature

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """(batch, hidden) → (batch, 4) mass values."""
        logits = self.readout(h) / self.temperature
        return F.softmax(logits, dim=-1)

    def to_mass_functions(self, h: torch.Tensor) -> list[MassFunction]:
        """Convert embeddings to MassFunction objects."""
        masses_tensor = self.forward(h)
        masses = []
        for row in masses_tensor.detach().cpu().numpy():
            masses.append(MassFunction(
                supports=float(row[0]),
                refutes=float(row[1]),
                uncertain=float(row[2]),
                theta=float(row[3]),
            ))
        return masses


class NextAnchorHead(nn.Module):
    """Supervised head that predicts the next anchor distribution for an infon.

    Given a post-message-passing infon embedding, outputs a probability
    distribution over all schema anchors. Trained on NEXT-edge pairs
    where the target for the earlier infon is a two-hot distribution
    over the predicate and object anchors of the later infon.

    A simple linear projection suffices: message passing has already
    done the heavy lifting of encoding structural context into the
    embedding.
    """

    def __init__(self, hidden_dim: int, num_anchors: int):
        super().__init__()
        self.num_anchors = num_anchors
        self.proj = nn.Linear(hidden_dim, num_anchors)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """(batch, hidden) → (batch, num_anchors) probability distribution."""
        return F.softmax(self.proj(h), dim=-1)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        """(batch, hidden) → (batch, num_anchors) unnormalized logits."""
        return self.proj(h)


class SubgraphPool(nn.Module):
    """Pool a set of node embeddings into a single subgraph embedding.

    Modes:
        mean      — element-wise mean of member embeddings
        sum       — element-wise sum (preserves size)
        max       — element-wise max (any strong signal wins)
        attention — learned attention weights over members
    """

    def __init__(self, hidden_dim: int, mode: str = "mean"):
        super().__init__()
        self.mode = mode
        self.hidden_dim = hidden_dim
        if mode == "attention":
            self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor, indices: list[int] | torch.Tensor
                ) -> torch.Tensor:
        """(n_nodes, hidden) + index list → (hidden,) pooled embedding."""
        if isinstance(indices, list):
            indices = torch.tensor(indices, dtype=torch.long, device=h.device)
        if indices.numel() == 0:
            return torch.zeros(self.hidden_dim, device=h.device)
        sub = h[indices]
        if self.mode == "mean":
            return sub.mean(dim=0)
        if self.mode == "sum":
            return sub.sum(dim=0)
        if self.mode == "max":
            return sub.max(dim=0).values
        if self.mode == "attention":
            w = F.softmax(self.attn(sub), dim=0)
            return (w * sub).sum(dim=0)
        raise ValueError(f"Unknown pool mode: {self.mode}")


class TimeToEventHead(nn.Module):
    """Predict time-to-next-event via a Weibull distribution.

    Outputs two scalars per input embedding: log-scale (log λ) and
    log-shape (log k) of a Weibull distribution. The expected time to
    event is λ · Γ(1 + 1/k); we expose it via predict_mean_time.

    Training uses Weibull negative-log-likelihood on observed intervals
    from NEXT-edge pairs where both endpoints have parseable timestamps.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, 2)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """(batch, hidden) → (batch, 2) = (log_scale, log_shape)."""
        return self.proj(h)

    def nll(self, h: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        """Weibull NLL. delta_t shape (batch,), positive floats.

        -log f(t) = -log k + (1-k) log t + k log λ - k (t/λ)^k
        We optimize the NLL written in terms of (log_scale, log_shape).
        """
        out = self.forward(h)
        log_scale = out[:, 0]
        log_shape = out[:, 1].clamp(min=-2.0, max=2.0)
        k = log_shape.exp()
        lam = log_scale.exp()
        t_safe = delta_t.clamp(min=1e-6)
        nll = -(torch.log(k) + (k - 1.0) * torch.log(t_safe) -
                k * log_scale - (t_safe / lam).pow(k).clamp(max=50.0))
        return nll.mean()

    def predict_mean_time(self, h: torch.Tensor) -> torch.Tensor:
        """Expected time under the predicted Weibull. (batch,) positive."""
        out = self.forward(h)
        log_scale = out[:, 0]
        log_shape = out[:, 1].clamp(min=-2.0, max=2.0)
        k = log_shape.exp()
        lam = log_scale.exp()
        # E[T] = λ · Γ(1 + 1/k). Use lgamma for stability.
        return lam * torch.exp(torch.lgamma(1.0 + 1.0 / k.clamp(min=0.1)))


class RiskRankingHead(nn.Module):
    """Per-node scalar risk score.

    A single linear projection → sigmoid. Trained with a pairwise
    margin-ranking loss: high-risk nodes should receive higher scores
    than low-risk nodes. Positives are infons with negative polarity
    or involved in CONTRADICTS edges; negatives are randomly-sampled
    other infons.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """(batch, hidden) → (batch,) risk score in [0, 1]."""
        return torch.sigmoid(self.proj(h)).squeeze(-1)

    def score(self, h: torch.Tensor) -> torch.Tensor:
        """Alias for forward."""
        return self.forward(h)


class AnomalyLocalizationHead(nn.Module):
    """Self-supervised anomaly scoring via bottleneck reconstruction.

    Trains an autoencoder on infon embeddings; at inference time the
    reconstruction error is the anomaly score. No labels needed.
    """

    def __init__(self, hidden_dim: int, bottleneck_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Linear(bottleneck_dim, hidden_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """(batch, hidden) → (batch, hidden) reconstruction."""
        return self.decoder(self.encoder(h))

    def anomaly_score(self, h: torch.Tensor) -> torch.Tensor:
        """(batch, hidden) → (batch,) normalized reconstruction error."""
        recon = self.forward(h)
        err = (h - recon).pow(2).sum(dim=-1)
        return err


class LearnedDempsterWeights(nn.Module):
    """Per-source trust weights over the four DS evidence sources.

    Each of the four heuristic sources (polarity, triple alignment,
    anchor distance, confidence) gets a learned scalar logit. A
    softmax produces non-negative weights summing to 1, which are
    used to interpolate the four mass functions before Dempster
    combination.

    Trained against the GNN's own self-consistent mass readout: a
    source is 'trusted' insofar as its per-infon mass agrees with
    the GNN's final belief after message passing.
    """

    def __init__(self, num_sources: int = 4):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_sources))

    def weights(self) -> torch.Tensor:
        return F.softmax(self.logits, dim=0)

    def forward(self) -> torch.Tensor:
        return self.weights()


class RoleTypeHead(nn.Module):
    """Masked-role type prediction.

    Given the embeddings of two known anchors in a (subject, predicate,
    object) triple, predict the type of the third (masked) role. The
    set of types is learned from the seed schema — the number of
    output classes is configurable at construction time.

    Input is a 2H-dim concatenation of the two known anchor embeddings
    plus a one-hot mask indicator telling the head which role is
    missing.
    """

    def __init__(self, hidden_dim: int, num_types: int):
        super().__init__()
        # Input = [h_role_A, h_role_B, mask_one_hot(3)]
        self.proj = nn.Sequential(
            nn.Linear(2 * hidden_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_types),
        )

    def forward(self, h_a: torch.Tensor, h_b: torch.Tensor,
                mask_onehot: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h_a, h_b, mask_onehot], dim=-1)
        return self.proj(x)


class TemporalSuccessorHead(nn.Module):
    """Asymmetric bilinear scoring head for temporal precedence.

    Given embeddings for two infons, outputs a scalar score for
    "i precedes j". An asymmetric matrix W makes score(i, j) differ
    from score(j, i), which is the property we need.

    Supervision is self-supervised from document order: sentences
    appearing earlier in the same document produce positive pairs,
    reversed-order pairs are negatives. No hand-coded tense rules are
    used during training.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Parameter(
            0.01 * torch.randn(hidden_dim, hidden_dim)
            + torch.eye(hidden_dim) * 0.5
        )

    def forward(self, h_i: torch.Tensor, h_j: torch.Tensor) -> torch.Tensor:
        """(batch, hidden) x (batch, hidden) -> (batch,) in [0, 1]."""
        # Direction-aware: score(i, j) = sigmoid(h_i^T W h_j)
        Wj = h_j @ self.W.T
        logits = (h_i * Wj).sum(dim=-1)
        return torch.sigmoid(logits)

    def logits(self, h_i: torch.Tensor, h_j: torch.Tensor) -> torch.Tensor:
        Wj = h_j @ self.W.T
        return (h_i * Wj).sum(dim=-1)


class RecommenderHead(nn.Module):
    """Bilinear scoring head: score(u, i) = u^T W i.

    Trained with Bayesian Personalized Ranking (BPR) loss on observed
    (user, interaction, item) triples extracted from the graph.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.eye(hidden_dim) +
                              0.01 * torch.randn(hidden_dim, hidden_dim))

    def forward(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        """(batch, hidden) × (batch, hidden) → (batch,) scalar scores."""
        Wi = i @ self.W.T
        return (u * Wi).sum(dim=-1)

    def score_matrix(self, users: torch.Tensor,
                     items: torch.Tensor) -> torch.Tensor:
        """(n_users, hidden) × (n_items, hidden) → (n_users, n_items)."""
        return users @ self.W @ items.T


class DiversityHead(nn.Module):
    """Pairwise-cosine-similarity penalty on a set of items.

    Higher output = more redundant recommendations. Used as a negative
    regularizer during training so the recommender learns to spread
    its top-k picks across the item space.
    """

    def __init__(self):
        super().__init__()

    def forward(self, items: torch.Tensor) -> torch.Tensor:
        """(k, hidden) → scalar mean off-diagonal cosine similarity."""
        if items.shape[0] < 2:
            return torch.zeros(1, device=items.device).squeeze()
        norm = F.normalize(items, dim=-1)
        sim = norm @ norm.T
        k = items.shape[0]
        off_diag = (sim.sum() - torch.diagonal(sim).sum()) / (k * (k - 1))
        return off_diag


# ═══════════════════════════════════════════════════════════════════════
# 5. HYPERGRAPH REASONER — END-TO-END
# ═══════════════════════════════════════════════════════════════════════

class HypergraphReasoner(nn.Module):
    """End-to-end geometric reasoning over the infon hypergraph.

    Pipeline:
        store → HypergraphBuilder → HyperGraph
        → TypedMessagePassing (N layers, IKL aggregators)
        → MassReadout → MassFunction per node
        → IKL compound queries (forall, exists, that, if, ist)

    Compatible with existing DS framework: output masses feed into
    combine_dempster, verify_claim, or GraphMCTS backprop.

    Usage:
        reasoner = HypergraphReasoner(store, encoder, schema)
        result = reasoner.reason("Did Toyota invest in batteries?")
        print(result.verdict, result.mass)
    """

    def __init__(self, store, encoder, schema,
                 hidden_dim: int = 64,
                 n_layers: int = 2,
                 situation_dim: int = 16,
                 use_sheaf: bool = False):
        super().__init__()
        self.store = store
        self.encoder = encoder
        self.schema = schema
        self.hidden_dim = hidden_dim
        self.builder = HypergraphBuilder(store, encoder, schema)
        self._fitted = False
        self.use_sheaf = use_sheaf

        # Message passing layers — R-GCN by default, sheaf if opted in.
        layer_cls = SheafMessagePassingLayer if use_sheaf \
            else TypedMessagePassingLayer
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            in_d = hidden_dim
            self.layers.append(
                layer_cls(in_d, hidden_dim, situation_dim=situation_dim)
            )

        # Mass readout
        self.mass_readout = MassReadout(hidden_dim)

        # IKL operators for compound queries
        self.ikl_that = IKLThat(hidden_dim)
        self.ikl_not = IKLNot(hidden_dim)
        self.ikl_if = IKLIf(hidden_dim)
        self.ikl_iff = IKLIff(hidden_dim)
        self.ikl_forall = IKLForall(hidden_dim)
        self.ikl_exists = IKLExists(hidden_dim)
        self.ikl_ist = IKLIst(hidden_dim, situation_dim)
        self.ikl_and = IKLAnd(hidden_dim)
        self.ikl_or = IKLOr(hidden_dim)

        # Dedicated next-anchor head. Built eagerly if the schema has
        # any anchors; otherwise deferred until first training call.
        anchor_names = list(self.schema.names)
        if anchor_names:
            self.next_head = NextAnchorHead(hidden_dim, len(anchor_names))
            self._next_head_anchors = anchor_names
        else:
            self.next_head = None
            self._next_head_anchors = []
        self._next_head_trained = False

        # Additional question-specific heads. All lazy-init; constructors
        # are cheap so we build them eagerly and only require training
        # when the caller asks for predictions.
        self.time_to_event_head = TimeToEventHead(hidden_dim)
        self._time_head_trained = False
        self.risk_head = RiskRankingHead(hidden_dim)
        self._risk_head_trained = False
        self.anomaly_head = AnomalyLocalizationHead(hidden_dim)
        self._anomaly_head_trained = False
        self.recommender_head = RecommenderHead(hidden_dim)
        self._recommender_trained = False
        self.diversity_head = DiversityHead()
        self.temporal_head = TemporalSuccessorHead(hidden_dim)
        self._temporal_head_trained = False

        # Role typing head — lazily sized to the schema's anchor type count
        self._role_types = sorted(set(self.schema.types.values())) \
            if self.schema.types else []
        if self._role_types:
            self.role_head = RoleTypeHead(hidden_dim, len(self._role_types))
        else:
            self.role_head = None
        self._role_head_trained = False

        # Learned per-source DS weights (uniform by default).
        self.ds_source_weights = LearnedDempsterWeights(num_sources=6)
        self._ds_weights_trained = False

        # Pooling module shared across subgraph / graph level calls.
        self.subgraph_pool = SubgraphPool(hidden_dim, mode="mean")

    def forward(self, graph: HyperGraph) -> torch.Tensor:
        """Run message passing, return refined node embeddings."""
        h = graph.node_features
        for layer in self.layers:
            h = layer(h, graph.edge_index, graph.edge_types,
                      graph.edge_weights, graph.situation_features)
        return h

    def fit(self, graph: HyperGraph | None = None,
            max_infons: int = 500,
            epochs: int = 30,
            lr: float = 1e-3,
            sheaf_weight: float = 0.2,
            laplacian_weight: float = 0.1,
            grad_clip: float = 1.0,
            patience: int = 8,
            verbose: bool = False) -> dict:
        """Transductive training: fit message passing + readout on this graph.

        Uses the existing DS heuristic masses (polarity, triple alignment,
        anchor distance, confidence) as teacher targets. The GNN learns to
        reproduce those masses after message passing — but with the benefit
        of structural context from neighbor propagation.

        Sheaf coherence regularization: penalizes the GNN for assigning
        high confidence (low theta) to infons whose anchors don't genuinely
        co-occur across the corpus. This prevents overconfidence on
        structurally weak triples.

        Training features:
        - Gradient clipping to prevent exploding gradients in message passing
        - Patience-based early stopping on loss plateau
        - Sheaf coherence as regularization term

        Returns training stats dict.
        """
        from .dempster_shafer import (
            mass_from_polarity, mass_from_triple_alignment,
            mass_from_anchor_distance, mass_from_confidence,
            mass_from_evidentiality, mass_from_modality,
        )
        from .category import SheafCoherence

        if graph is None:
            graph = self.builder.build(max_infons=max_infons,
                                       feature_dim=self.hidden_dim)

        # Collect teacher targets and infon metadata for all infon nodes
        infon_ids = list(graph.infon_map.keys())
        infon_node_indices = []
        teacher_masses = []
        infon_objects = []

        for iid in infon_ids:
            infon = self.store.get_infon(iid)
            if infon is None:
                continue
            idx = graph.infon_map[iid]

            claim_anchors = {}
            for role in [infon.subject, infon.predicate, infon.object]:
                claim_anchors[role] = infon.confidence

            sources = [
                mass_from_polarity(infon),
                mass_from_triple_alignment(claim_anchors, infon, self.schema.types),
                mass_from_anchor_distance(claim_anchors, infon, self.schema.types),
                mass_from_confidence(infon),
                mass_from_evidentiality(infon),
                mass_from_modality(infon),
            ]
            combined = combine_multiple(sources)
            teacher_masses.append(combined)
            infon_node_indices.append(idx)
            infon_objects.append(infon)

        if not teacher_masses:
            return {"epochs": 0, "loss": float("nan"), "n_targets": 0}

        # Build teacher tensor: (n_infons, 4)
        teacher = torch.tensor([
            [m.supports, m.refutes, m.uncertain, m.theta]
            for m in teacher_masses
        ], dtype=torch.float32)
        indices = torch.tensor(infon_node_indices, dtype=torch.long)

        # ── Sheaf coherence: compute per-infon structural scores ──────
        sheaf = SheafCoherence(list(self.schema.names))

        # Build activation matrix from infon sentences
        sentences = [inf.sentence for inf in infon_objects if inf.sentence]
        if sentences and sheaf_weight > 0:
            act_matrix = self.encoder.encode(sentences)
            sheaf.observe(act_matrix)
            sheaf.fit()
            sheaf_scores = torch.tensor(
                [sheaf.score_infon(inf) for inf in infon_objects],
                dtype=torch.float32,
            )
        else:
            sheaf_scores = torch.ones(len(infon_objects), dtype=torch.float32)

        # Train
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        best_loss = float("inf")
        patience_counter = 0
        losses = []
        actual_epochs = 0

        for epoch in range(epochs):
            actual_epochs = epoch + 1
            optimizer.zero_grad()
            h = self.forward(graph)
            predicted = self.mass_readout(h[indices])  # (n_infons, 4)

            # KL divergence loss (teacher signal)
            kl_loss = F.kl_div(
                predicted.log().clamp(min=-20),
                teacher,
                reduction="batchmean",
                log_target=False,
            )

            # Sheaf coherence regularization:
            # For infons with low sheaf coherence, penalize high confidence
            # (low theta). confidence_mass = 1 - theta = predicted[:, 0:3].sum()
            if sheaf_weight > 0:
                confidence_mass = 1.0 - predicted[:, 3]  # 1 - theta
                # Low coherence → high penalty for being confident
                # coherence_penalty = confidence * (1 - sheaf_score)
                sheaf_penalty = (confidence_mass * (1.0 - sheaf_scores)).mean()
                loss = kl_loss + sheaf_weight * sheaf_penalty
            else:
                loss = kl_loss

            # Sheaf-Laplacian regularizer on the node embeddings. Every
            # edge (s -r-> t) should have agreeing restriction-map
            # projections at its stalk. Computed *after* message passing
            # (same h fed to the mass readout) so it reflects the final
            # geometry the reasoner uses.
            lap_val = None
            if self.use_sheaf and laplacian_weight > 0 and self.layers:
                last_layer = self.layers[-1]
                lap_val = last_layer.sheaf_discrepancy(
                    h, graph.edge_index, graph.edge_types,
                    graph.edge_weights,
                )
                loss = loss + laplacian_weight * lap_val

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
            optimizer.step()

            loss_val = loss.item()
            losses.append(loss_val)

            if loss_val < best_loss - 1e-5:
                best_loss = loss_val
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch + 1) % 5 == 0:
                extra = f" sheaf={sheaf_penalty.item():.4f}" if sheaf_weight > 0 else ""
                if lap_val is not None:
                    extra += f" L_F={lap_val.item():.4f}"
                print(f"    Epoch {epoch+1}/{epochs}: loss={loss_val:.4f}{extra}")

            if patience_counter >= patience and epoch >= 10:
                if verbose:
                    print(f"    Early stopping at epoch {epoch+1} "
                          f"(no improvement for {patience} epochs)")
                break

        self.eval()
        self._fitted = True
        return {
            "epochs": actual_epochs,
            "final_loss": losses[-1] if losses else float("nan"),
            "best_loss": best_loss,
            "n_targets": len(teacher_masses),
            "losses": losses,
            "early_stopped": patience_counter >= patience,
            "sheaf_fiedler": sheaf.fiedler_value if sheaf_weight > 0 else None,
        }

    def reason(self, query: str,
               max_infons: int = 500,
               fit_epochs: int = 30,
               verbose: bool = False) -> ReasoningResult:
        """Full reasoning pipeline: query → graph → fit → message passing → verdict.

        Auto-fits on first call (transductive: trains on the graph it will
        reason over, using DS heuristic masses as teacher signal).
        """
        # Encode query
        query_activations = self.encoder.encode_single(query)

        # Build graph from store
        graph = self.builder.build(max_infons=max_infons,
                                   feature_dim=self.hidden_dim)

        if verbose:
            print(f"  Graph: {graph.n_nodes} nodes, {graph.n_edges} edges")
            print(f"  Anchors: {len(graph.anchor_map)}, "
                  f"Infons: {len(graph.infon_map)}")

        if graph.n_nodes == 0:
            return ReasoningResult(
                query=query, verdict="NOT ENOUGH INFO",
                mass=MassFunction(theta=1.0),
            )

        # Auto-fit on first call
        if not self._fitted:
            fit_stats = self.fit(graph=graph, epochs=fit_epochs, verbose=verbose)
            if verbose:
                print(f"  Fit: {fit_stats['n_targets']} targets, "
                      f"loss {fit_stats['best_loss']:.4f} → {fit_stats['final_loss']:.4f}")

        # Message passing
        with torch.no_grad():
            h = self.forward(graph)

        # Find relevant infon nodes via query anchor overlap.
        # Critical calibration point: we want to answer SUPPORTS/REFUTES
        # only when we have infons whose TRIPLE matches the query's
        # intended triple, not just infons that share a single anchor
        # with the query text.
        #
        # We partition query_activations by type to infer the asked
        # triple, then score an infon by role-wise overlap:
        #   subject_overlap   = activation of infon.subject under
        #                       an actor-type query anchor
        #   predicate_overlap = activation of infon.predicate under
        #                       a relation-type query anchor
        #   object_overlap    = activation of infon.object under
        #                       any other-type query anchor
        #
        # Relevance = product × (bonus if all three overlap).
        types = self.schema.types
        relevant_indices = []
        relevant_weights = []
        for inf_id, node_idx in graph.infon_map.items():
            infon = self.store.get_infon(inf_id)
            if infon is None:
                continue
            subj_s = query_activations.get(infon.subject, 0.0)
            pred_s = query_activations.get(infon.predicate, 0.0)
            obj_s = query_activations.get(infon.object, 0.0)
            # Role-matched relevance: each role must have some
            # activation, AND the product is the combined score.
            min_role = min(subj_s, pred_s, obj_s)
            max_role = max(subj_s, pred_s, obj_s)
            # Drop infons where any role has zero query-activation
            if min_role <= 0.0:
                continue
            # All three roles need at least weak match
            if min_role < 0.05:
                continue
            # Geometric mean score
            rel = (subj_s * pred_s * obj_s) ** (1 / 3)
            relevant_indices.append(node_idx)
            relevant_weights.append(rel)

        # If no infon strongly matches, return high-θ NEI. This is the
        # key calibration fix: the corpus doesn't speak to this claim.
        if not relevant_indices:
            return ReasoningResult(
                query=query, verdict="NOT ENOUGH INFO",
                mass=MassFunction(theta=1.0),
                n_nodes=graph.n_nodes, n_edges=graph.n_edges,
            )

        # If the strongest relevance is still weak (< 0.2), the
        # claim isn't well-supported. Return a high-θ mass rather
        # than combining sources and concentrating confidence.
        if max(relevant_weights) < 0.2:
            # Blend: 40% of the top mass, 60% θ
            relevant_h = h[relevant_indices]
            masses = self.mass_readout.to_mass_functions(relevant_h)
            top_mass = masses[0]
            blended = MassFunction(
                supports=top_mass.supports * 0.4,
                refutes=top_mass.refutes * 0.4,
                uncertain=top_mass.uncertain * 0.4,
                theta=top_mass.theta * 0.4 + 0.6,
            )
            return ReasoningResult(
                query=query, verdict="NOT ENOUGH INFO",
                mass=blended,
                per_infon_masses=masses,
                n_nodes=graph.n_nodes, n_edges=graph.n_edges,
                n_relevant=len(relevant_indices),
            )

        # Read out masses for relevant infons
        relevant_h = h[relevant_indices]
        masses = self.mass_readout.to_mass_functions(relevant_h)

        if verbose:
            print(f"  Relevant infons: {len(masses)}")
            for i, m in enumerate(masses[:5]):
                print(f"    [{i}] S={m.supports:.3f} R={m.refutes:.3f} "
                      f"U={m.uncertain:.3f} θ={m.theta:.3f}")

        # Weight by relevance and combine top-k
        weighted_masses = sorted(
            zip(masses, relevant_weights),
            key=lambda x: x[1], reverse=True,
        )[:10]
        decisive = [m for m, w in weighted_masses if m.theta < 0.95][:5]
        combined = combine_multiple(decisive) if decisive else MassFunction(theta=1.0)

        # Verdict via pignistic transform
        total_focal = combined.supports + combined.refutes + combined.uncertain
        if total_focal > 0:
            pig_s = combined.supports + combined.theta * (combined.supports / total_focal)
            pig_r = combined.refutes + combined.theta * (combined.refutes / total_focal)
        else:
            pig_s = combined.theta / 3
            pig_r = combined.theta / 3

        if pig_r > 0.15 and pig_r > pig_s:
            verdict = "REFUTES"
        elif pig_s > 0.25 and pig_s > pig_r:
            verdict = "SUPPORTS"
        else:
            verdict = "NOT ENOUGH INFO"

        return ReasoningResult(
            query=query,
            verdict=verdict,
            mass=combined,
            per_infon_masses=masses,
            n_nodes=graph.n_nodes,
            n_edges=graph.n_edges,
            n_relevant=len(relevant_indices),
        )

    # ── IKL COMPOUND QUERY INTERFACE ──────────────────────────────────

    def query_forall(self, anchor_type: str, graph: HyperGraph,
                     h: torch.Tensor) -> torch.Tensor:
        """(forall (?x T) φ) — universal quantification over anchor type T."""
        indices = graph.anchor_type_groups.get(anchor_type, [])
        if not indices:
            return torch.zeros(self.hidden_dim)
        domain_h = h[indices]
        return self.ikl_forall(domain_h)

    def query_exists(self, anchor_type: str, graph: HyperGraph,
                     h: torch.Tensor) -> torch.Tensor:
        """(exists (?x T) φ) — existential over anchor type T."""
        indices = graph.anchor_type_groups.get(anchor_type, [])
        if not indices:
            return torch.zeros(self.hidden_dim)
        domain_h = h[indices]
        return self.ikl_exists(domain_h)

    def query_that(self, infon_id: str, graph: HyperGraph,
                   h: torch.Tensor) -> torch.Tensor:
        """(that φ) — reify infon as a first-class proposition term."""
        idx = graph.infon_map.get(infon_id)
        if idx is None:
            return torch.zeros(self.hidden_dim)
        return self.ikl_that(h[idx].unsqueeze(0)).squeeze(0)

    def query_conditional(self, premise_id: str, conclusion_id: str,
                          graph: HyperGraph, h: torch.Tensor) -> torch.Tensor:
        """(if φ ψ) — does premise support conclusion?"""
        p_idx = graph.infon_map.get(premise_id)
        c_idx = graph.infon_map.get(conclusion_id)
        if p_idx is None or c_idx is None:
            return torch.zeros(self.hidden_dim)
        return self.ikl_if(
            h[p_idx].unsqueeze(0),
            h[c_idx].unsqueeze(0),
        ).squeeze(0)

    def query_ist(self, infon_id: str, graph: HyperGraph,
                  h: torch.Tensor) -> torch.Tensor:
        """(ist s φ) — contextualize infon in its situation."""
        idx = graph.infon_map.get(infon_id)
        if idx is None:
            return torch.zeros(self.hidden_dim)
        if graph.situation_features is None or idx >= graph.situation_features.shape[0]:
            return h[idx]
        return self.ikl_ist(
            h[idx].unsqueeze(0),
            graph.situation_features[idx].unsqueeze(0),
        ).squeeze(0)

    def compound_query(self, expr: dict, graph: HyperGraph,
                       h: torch.Tensor) -> torch.Tensor:
        """Evaluate a nested IKL expression.

        Expression format (S-expression as dict):
            {"op": "and", "args": [expr1, expr2, ...]}
            {"op": "or", "args": [expr1, expr2, ...]}
            {"op": "not", "args": [expr]}
            {"op": "if", "args": [premise_expr, conclusion_expr]}
            {"op": "iff", "args": [expr1, expr2]}
            {"op": "that", "infon_id": "..."}
            {"op": "forall", "type": "actor", "body": expr}
            {"op": "exists", "type": "actor", "body": expr}
            {"op": "ist", "infon_id": "..."}
            {"op": "node", "id": "..."}  # leaf: anchor or infon reference
        """
        op = expr.get("op", "node")

        if op == "node":
            node_id = expr["id"]
            idx = graph.infon_map.get(node_id, graph.anchor_map.get(node_id))
            if idx is None:
                return torch.zeros(self.hidden_dim, device=h.device)
            return h[idx]

        if op == "that":
            return self.query_that(expr["infon_id"], graph, h)

        if op == "ist":
            return self.query_ist(expr["infon_id"], graph, h)

        if op == "not":
            inner = self.compound_query(expr["args"][0], graph, h)
            return self.ikl_not(inner.unsqueeze(0)).squeeze(0)

        if op == "and":
            parts = torch.stack([
                self.compound_query(a, graph, h) for a in expr["args"]
            ])
            return self.ikl_and(parts)

        if op == "or":
            parts = torch.stack([
                self.compound_query(a, graph, h) for a in expr["args"]
            ])
            return self.ikl_or(parts)

        if op == "if":
            premise = self.compound_query(expr["args"][0], graph, h)
            conclusion = self.compound_query(expr["args"][1], graph, h)
            return self.ikl_if(
                premise.unsqueeze(0), conclusion.unsqueeze(0)
            ).squeeze(0)

        if op == "iff":
            h1 = self.compound_query(expr["args"][0], graph, h)
            h2 = self.compound_query(expr["args"][1], graph, h)
            return self.ikl_iff(
                h1.unsqueeze(0), h2.unsqueeze(0)
            ).squeeze(0)

        if op == "forall":
            return self.query_forall(expr["type"], graph, h)

        if op == "exists":
            return self.query_exists(expr["type"], graph, h)

        raise ValueError(f"Unknown IKL operator: {op}")

    def refine(self, graph: HyperGraph | None = None,
               max_infons: int = 500,
               causal_threshold: float = 0.6,
               verbose: bool = False) -> RefinementResult:
        """Refine the hypergraph: temporal and causal edges only.

        Focused refinement that writes back to the store:

        1. **Confidence/coherence update** — blend GNN mass with original
           extraction confidence for all infons.

        2. **Temporal edges** — for infon pairs sharing a subject (same
           actor), if one is past-tense and another present/future, add
           a directed NEXT edge (past → present/future). Uses tense
           ordering from the infon metadata, no model evaluation needed.

        3. **Causal edges** — for infon pairs sharing a subject where
           predicates differ, score the (if earlier later) conditional
           via the IKL operator. High supports → CAUSES edge.
           High refutes → CONTRADICTS edge. Only checks temporally
           ordered pairs, so the number of evaluations is small.

        Must be called after fit(). Auto-fits if needed.
        """
        if graph is None:
            graph = self.builder.build(max_infons=max_infons,
                                       feature_dim=self.hidden_dim)

        if not self._fitted:
            self.fit(graph=graph, verbose=verbose)

        with torch.no_grad():
            h = self.forward(graph)

        infon_ids = list(graph.infon_map.keys())
        infon_indices = [graph.infon_map[iid] for iid in infon_ids]

        # ── 1. Update infon confidence and coherence ──────────────────

        if infon_indices:
            infon_h = h[infon_indices]
            masses = self.mass_readout.to_mass_functions(infon_h)
        else:
            masses = []

        updated_infons = []
        for iid, mass in zip(infon_ids, masses):
            infon = self.store.get_infon(iid)
            if infon is None:
                continue
            old_conf = infon.confidence
            infon.confidence = 0.6 * mass.supports + 0.4 * old_conf
            infon.coherence = max(infon.coherence, 1.0 - mass.theta)
            updated_infons.append(infon)

        if updated_infons:
            self.store.put_infons(updated_infons)

        # ── 2. Temporal + causal edges ────────────────────────────────

        TENSE_ORDER = {"past": 0, "present": 1, "present_continuous": 1,
                       "future": 2, "conditional": 2, "unknown": -1}

        # Group infons by subject (actor) — temporal/causal relations
        # connect actions by the same entity
        subject_groups: dict[str, list[tuple[str, Infon]]] = defaultdict(list)
        infon_cache: dict[str, Infon] = {}
        for iid in infon_ids:
            infon = self.store.get_infon(iid)
            if infon is None:
                continue
            infon_cache[iid] = infon
            subject_groups[infon.subject].append((iid, infon))

        new_temporal = []
        new_causal = []
        contradictions = []
        causal_candidates = []
        pairs_checked = 0

        for subject, group in subject_groups.items():
            if len(group) < 2:
                continue

            # Sort by tense order within each actor's group
            ordered = sorted(group, key=lambda x: TENSE_ORDER.get(x[1].tense, -1))

            for i, (earlier_id, earlier) in enumerate(ordered):
                earlier_tense = TENSE_ORDER.get(earlier.tense, -1)
                if earlier_tense < 0:
                    continue

                for later_id, later in ordered[i+1:]:
                    later_tense = TENSE_ORDER.get(later.tense, -1)
                    if later_tense < 0 or later_tense < earlier_tense:
                        continue
                    if earlier_id == later_id:
                        continue

                    # Temporal edge: same actor, earlier tense → later tense
                    if later_tense > earlier_tense:
                        new_temporal.append(Edge(
                            source=earlier_id, target=later_id,
                            edge_type="NEXT",
                            weight=min(earlier.confidence, later.confidence),
                            metadata={"source": "gnn_refine",
                                      "anchor": subject,
                                      "relation": "temporal"},
                        ))

                    # Collect causal candidates for batched evaluation
                    if earlier.predicate != later.predicate:
                        p_idx = graph.infon_map.get(earlier_id)
                        c_idx = graph.infon_map.get(later_id)
                        if p_idx is not None and c_idx is not None:
                            causal_candidates.append(
                                (earlier_id, later_id, p_idx, c_idx,
                                 subject, earlier.predicate, later.predicate)
                            )

        # ── 3. Batched causal evaluation ──────────────────────────────
        if causal_candidates:
            premise_indices = [c[2] for c in causal_candidates]
            conclusion_indices = [c[3] for c in causal_candidates]
            pairs_checked = len(causal_candidates)

            premise_h = h[premise_indices]       # (batch, hidden)
            conclusion_h = h[conclusion_indices]  # (batch, hidden)
            cond_h = self.ikl_if(premise_h, conclusion_h)  # (batch, hidden)
            cond_masses = self.mass_readout.to_mass_functions(cond_h)

            for (earlier_id, later_id, _, _, subject,
                 from_pred, to_pred), cond_mass in zip(causal_candidates, cond_masses):
                if cond_mass.supports > causal_threshold:
                    new_causal.append(Edge(
                        source=earlier_id, target=later_id,
                        edge_type="CAUSES",
                        weight=cond_mass.supports,
                        metadata={
                            "source": "gnn_refine",
                            "anchor": subject,
                            "from_pred": from_pred,
                            "to_pred": to_pred,
                        },
                    ))

                if cond_mass.refutes > causal_threshold:
                    contradictions.append(Edge(
                        source=earlier_id, target=later_id,
                        edge_type="CONTRADICTS",
                        weight=cond_mass.refutes,
                        metadata={
                            "source": "gnn_refine",
                            "anchor": subject,
                            "from_pred": from_pred,
                            "to_pred": to_pred,
                        },
                    ))

        # Write edges to store
        all_new_edges = new_temporal + new_causal + contradictions
        if all_new_edges:
            self.store.put_edges(all_new_edges)

        if verbose:
            print(f"  Refinement:")
            print(f"    Updated {len(updated_infons)} infon confidences")
            print(f"    Temporal NEXT edges: {len(new_temporal)}")
            print(f"    Causal CAUSES edges: {len(new_causal)}")
            print(f"    CONTRADICTS edges: {len(contradictions)}")
            print(f"    Causal pairs evaluated: {pairs_checked}")

        return RefinementResult(
            infons_updated=len(updated_infons),
            temporal_added=len(new_temporal),
            causal_added=len(new_causal),
            contradictions_found=len(contradictions),
            pairs_checked=pairs_checked,
            temporal_edges=new_temporal,
            causal_edges=new_causal,
            contradiction_edges=contradictions,
        )

    def predict_next_anchors(self, subject: str,
                             graph: HyperGraph | None = None,
                             k: int = 5,
                             max_infons: int = 500,
                             verbose: bool = False) -> list[dict]:
        """Predict which anchors are likely to appear next for a given subject.

        Uses the trained IF–THEN aggregator as a next-step predictor.
        Picks the most recent infon for `subject` (latest in tense order),
        then batch-evaluates IF–THEN with every other infon as a candidate
        conclusion. Aggregates m(SUPPORTS) by the predicate and object
        anchors of the top-ranked conclusions, producing a ranked list of
        anchors that the network thinks are likely continuations.

        Args:
            subject: subject anchor name (e.g., "toyota")
            graph: optional pre-built HyperGraph
            k: number of anchors to return
            max_infons: cap when building a fresh graph
            verbose: print intermediate scores

        Returns:
            list of dicts with keys {anchor, score, support_mass, theta}.
            Ranked descending by score. May be empty if the subject has
            no infons or the graph is empty.
        """
        if graph is None:
            graph = self.builder.build(max_infons=max_infons,
                                       feature_dim=self.hidden_dim)
        if not self._fitted:
            self.fit(graph=graph, verbose=verbose)

        TENSE_ORDER = {"past": 0, "present": 1, "present_continuous": 1,
                       "future": 2, "conditional": 2, "unknown": -1}

        # Find most recent infon for this subject
        subject_infons = []
        for iid in graph.infon_map:
            inf = self.store.get_infon(iid)
            if inf is None or inf.subject != subject:
                continue
            subject_infons.append((iid, inf))
        if not subject_infons:
            if verbose:
                print(f"  no infons found with subject={subject!r}")
            return []

        # Latest-tense infon is the premise
        subject_infons.sort(key=lambda x: TENSE_ORDER.get(x[1].tense, -1),
                            reverse=True)
        premise_id, premise_infon = subject_infons[0]
        premise_idx = graph.infon_map[premise_id]

        # Candidate conclusions: all other infons in the graph
        candidate_ids = []
        candidate_indices = []
        candidate_infons = []
        for iid, idx in graph.infon_map.items():
            if iid == premise_id:
                continue
            inf = self.store.get_infon(iid)
            if inf is None:
                continue
            candidate_ids.append(iid)
            candidate_indices.append(idx)
            candidate_infons.append(inf)

        if not candidate_ids:
            return []

        # Batch IF–THEN evaluation: premise broadcast against all conclusions
        with torch.no_grad():
            h = self.forward(graph)
            premise_h = h[premise_idx].unsqueeze(0).expand(len(candidate_indices), -1)
            conclusion_h = h[candidate_indices]
            cond_h = self.ikl_if(premise_h, conclusion_h)
            cond_masses = self.mass_readout.to_mass_functions(cond_h)

        # Aggregate m(SUPPORTS) by predicate and object anchor
        anchor_scores: dict[str, float] = defaultdict(float)
        anchor_theta: dict[str, list[float]] = defaultdict(list)
        anchor_count: dict[str, int] = defaultdict(int)

        for inf, mass in zip(candidate_infons, cond_masses):
            for anchor_name in (inf.predicate, inf.object):
                if not anchor_name:
                    continue
                anchor_scores[anchor_name] += mass.supports
                anchor_theta[anchor_name].append(mass.theta)
                anchor_count[anchor_name] += 1

        # Normalize by count (mean support per anchor mention)
        ranked = []
        for anchor, total in anchor_scores.items():
            count = anchor_count[anchor]
            mean_support = total / count if count > 0 else 0.0
            mean_theta = (sum(anchor_theta[anchor]) / count) if count > 0 else 1.0
            ranked.append({
                "anchor": anchor,
                "score": mean_support,
                "support_mass": mean_support,
                "theta": mean_theta,
                "evidence_count": count,
            })

        ranked.sort(key=lambda x: x["score"], reverse=True)
        top_k = ranked[:k]

        if verbose:
            print(f"  premise: {premise_infon.subject}/"
                  f"{premise_infon.predicate}/{premise_infon.object} "
                  f"[{premise_infon.tense}]")
            print(f"  evaluated {len(candidate_ids)} candidate conclusions")
            print(f"  top {len(top_k)} predicted next anchors:")
            for r in top_k:
                print(f"    {r['anchor']:20s}  score={r['score']:.3f}  "
                      f"θ={r['theta']:.3f}  (n={r['evidence_count']})")

        return top_k

    def train_next_head(self, graph: HyperGraph | None = None,
                        epochs: int = 50,
                        lr: float = 1e-2,
                        max_infons: int = 500,
                        verbose: bool = False) -> dict:
        """Train the dedicated next-anchor head on NEXT-edge supervision.

        Targets are built from NEXT edges already in the store: for each
        (earlier, later) pair, the target distribution for earlier is a
        two-hot vector over later.predicate and later.object, split
        evenly. The head is a single Linear + softmax on the
        post-message-passing infon embedding.

        Cross-entropy against the target distribution. Message-passing
        layers are frozen during head training so that only the head's
        linear weights move; the GNN's belief readout is unaffected.

        Returns training stats dict.
        """
        if graph is None:
            graph = self.builder.build(max_infons=max_infons,
                                       feature_dim=self.hidden_dim)
        if not self._fitted:
            self.fit(graph=graph, verbose=verbose)

        if self.next_head is None or not self._next_head_anchors:
            if verbose:
                print("  no schema anchors — cannot train next head")
            return {"epochs": 0, "final_loss": float("nan"),
                    "n_targets": 0, "losses": []}

        anchor_to_idx = {name: i for i, name in enumerate(self._next_head_anchors)}
        num_anchors = len(self._next_head_anchors)

        # Collect NEXT-edge supervision pairs
        next_edges = self.store.get_edges(edge_type="NEXT", limit=max_infons * 5)
        pairs = []
        for edge in next_edges:
            earlier = self.store.get_infon(edge.source)
            later = self.store.get_infon(edge.target)
            if earlier is None or later is None:
                continue
            e_idx = graph.infon_map.get(edge.source)
            if e_idx is None:
                continue
            # Target is two-hot over later.predicate and later.object
            target = torch.zeros(num_anchors, dtype=torch.float32)
            hits = 0
            if later.predicate in anchor_to_idx:
                target[anchor_to_idx[later.predicate]] += 1.0
                hits += 1
            if later.object in anchor_to_idx:
                target[anchor_to_idx[later.object]] += 1.0
                hits += 1
            if hits == 0:
                continue
            target /= hits
            pairs.append((e_idx, target))

        if not pairs:
            if verbose:
                print("  no NEXT-edge supervision — train_next_head skipped")
            return {"epochs": 0, "final_loss": float("nan"),
                    "n_targets": 0, "losses": []}

        src_indices = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        targets = torch.stack([p[1] for p in pairs])

        # Freeze GNN layers; train only the next head
        for p in self.parameters():
            p.requires_grad = False
        for p in self.next_head.parameters():
            p.requires_grad = True

        optimizer = torch.optim.Adam(self.next_head.parameters(), lr=lr)
        losses = []

        self.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            h = self.forward(graph)
            src_h = h[src_indices]
            logits = self.next_head.logits(src_h)
            log_probs = F.log_softmax(logits, dim=-1)
            # Cross-entropy against soft two-hot target
            loss = -(targets * log_probs).sum(dim=-1).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if verbose and (epoch + 1) % 10 == 0:
                print(f"    next-head epoch {epoch+1}/{epochs}: "
                      f"loss={loss.item():.4f}")

        # Re-enable gradients on the full reasoner (so subsequent .fit still works)
        for p in self.parameters():
            p.requires_grad = True

        self.eval()
        self._next_head_trained = True
        return {
            "epochs": epochs,
            "final_loss": losses[-1],
            "first_loss": losses[0],
            "n_targets": len(pairs),
            "losses": losses,
        }

    def predict_next_with_head(self, infon_id: str,
                               graph: HyperGraph | None = None,
                               k: int = 5,
                               max_infons: int = 500,
                               verbose: bool = False) -> list[dict]:
        """Predict next anchors for a specific infon using the trained head.

        Runs the GNN forward pass to get the infon's embedding, then
        projects through the next-anchor head to get a probability
        distribution over all schema anchors. Returns top-k.

        Args:
            infon_id: id of the infon to predict continuations for
            graph: optional pre-built HyperGraph
            k: number of anchors to return
            max_infons: cap when building a fresh graph
            verbose: print intermediate scores
        """
        if graph is None:
            graph = self.builder.build(max_infons=max_infons,
                                       feature_dim=self.hidden_dim)
        if not self._fitted:
            self.fit(graph=graph, verbose=verbose)
        if self.next_head is None or not self._next_head_trained:
            if verbose:
                print("  next head not trained — call train_next_head() first")
            return []

        idx = graph.infon_map.get(infon_id)
        if idx is None:
            return []

        with torch.no_grad():
            h = self.forward(graph)
            probs = self.next_head(h[idx].unsqueeze(0)).squeeze(0)

        probs_np = probs.cpu().numpy()
        ranked_idx = probs_np.argsort()[::-1][:k]
        results = []
        for ai in ranked_idx:
            results.append({
                "anchor": self._next_head_anchors[int(ai)],
                "probability": float(probs_np[int(ai)]),
            })

        if verbose:
            infon = self.store.get_infon(infon_id)
            if infon:
                print(f"  query: {infon.subject}/{infon.predicate}/"
                      f"{infon.object} [{infon.tense}]")
            print(f"  top {len(results)} next anchors (head):")
            for r in results:
                print(f"    {r['anchor']:20s}  p={r['probability']:.3f}")

        return results

    def discover_anchors(self, graph: HyperGraph | None = None,
                         n_anchors: int = 10,
                         max_infons: int = 500,
                         verbose: bool = False) -> tuple:
        """Discover new anchor types via left Kan extension on GNN embeddings.

        After message passing, anchor node embeddings encode structural
        context from their neighborhood (which infons they participate in,
        how those infons connect to other anchors). Spectral clustering on
        these enriched embeddings finds "natural" anchor categories that
        the original schema may have missed.

        Returns (AnchorSchema, list[DiscoveredAnchor], dict) where dict has
        cluster assignments and silhouette scores.
        """
        from .category import SchemaDiscovery, DiscoveredAnchor

        if graph is None:
            graph = self.builder.build(max_infons=max_infons,
                                       feature_dim=self.hidden_dim)

        if not self._fitted:
            self.fit(graph=graph, verbose=verbose)

        with torch.no_grad():
            h = self.forward(graph)

        anchor_names = list(graph.anchor_map.keys())
        anchor_indices = [graph.anchor_map[n] for n in anchor_names]

        if not anchor_indices:
            from .schema import AnchorSchema
            return AnchorSchema({}), [], {"n_anchors": 0}

        anchor_h = h[anchor_indices].cpu().numpy()  # (n_anchors, hidden_dim)

        # Build affinity matrix from GNN embeddings (cosine similarity)
        norms = np.linalg.norm(anchor_h, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        anchor_norm = anchor_h / norms
        affinity = anchor_norm @ anchor_norm.T
        affinity = np.maximum(affinity, 0)
        np.fill_diagonal(affinity, 0)

        n = len(anchor_names)
        k = min(n_anchors, n - 1)
        if k < 2:
            from .schema import AnchorSchema
            return AnchorSchema({}), [], {"n_anchors": n, "k": k}

        # Spectral clustering on the GNN affinity
        degree = affinity.sum(axis=1)
        degree_safe = np.where(degree > 0, degree, 1.0)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degree_safe))
        L_norm = np.eye(n) - D_inv_sqrt @ affinity @ D_inv_sqrt

        eigenvalues, eigenvectors = np.linalg.eigh(L_norm)
        features = eigenvectors[:, 1:k+1]

        # K-means
        labels = self._kmeans(features, k)

        # Build discovered anchors from clusters
        discovered = []
        cluster_map = defaultdict(list)
        for i, label in enumerate(labels):
            cluster_map[int(label)].append(i)

        for cluster_id, members in sorted(cluster_map.items()):
            member_names = [anchor_names[i] for i in members]
            member_types = [self.schema.types.get(n, "feature") for n in member_names]

            # Cluster centroid in GNN embedding space
            centroid = anchor_h[members].mean(axis=0)

            # Internal coherence: mean pairwise cosine sim within cluster
            if len(members) > 1:
                cluster_aff = affinity[np.ix_(members, members)]
                coherence = float(cluster_aff.mean())
            else:
                coherence = 0.0

            # Most common type in cluster
            type_counts = Counter(member_types)
            dominant_type = type_counts.most_common(1)[0][0]

            # Tokens from member anchors
            tokens = []
            for nm in member_names[:5]:
                anchor_def = self.schema.anchors.get(nm, {})
                toks = anchor_def.get("tokens", [nm])
                tokens.extend(toks[:2])

            # Name: most central member (highest degree in sub-affinity)
            sub_degree = affinity[np.ix_(members, members)].sum(axis=1) if len(members) > 1 else [0]
            best_idx = int(np.argmax(sub_degree))
            name = f"cluster_{member_names[best_idx]}"

            from .category import DiscoveredAnchor
            discovered.append(DiscoveredAnchor(
                name=name,
                inferred_type=dominant_type,
                tokens=tokens[:5],
                centroid_indices=members,
                size=len(members),
                mean_activation=float(np.linalg.norm(centroid)),
                coherence=coherence,
            ))

        # Build schema from discovered clusters
        anchor_defs = {}
        for da in discovered:
            anchor_defs[da.name] = {
                "type": da.inferred_type,
                "tokens": da.tokens,
            }

        from .schema import AnchorSchema
        schema = AnchorSchema(anchor_defs)

        # Silhouette-like score: (inter - intra) / max(inter, intra)
        intra_scores = []
        inter_scores = []
        for i in range(n):
            ci = int(labels[i])
            same = [j for j in range(n) if int(labels[j]) == ci and j != i]
            diff = [j for j in range(n) if int(labels[j]) != ci]
            if same:
                intra_scores.append(float(np.mean([affinity[i, j] for j in same])))
            if diff:
                inter_scores.append(float(np.mean([affinity[i, j] for j in diff])))

        mean_intra = np.mean(intra_scores) if intra_scores else 0.0
        mean_inter = np.mean(inter_scores) if inter_scores else 0.0
        denom = max(mean_intra, mean_inter, 1e-8)
        silhouette = float((mean_intra - mean_inter) / denom)

        stats = {
            "n_anchors": n,
            "n_clusters": len(discovered),
            "k": k,
            "silhouette": silhouette,
            "mean_intra_sim": float(mean_intra),
            "mean_inter_sim": float(mean_inter),
            "cluster_sizes": [da.size for da in discovered],
            "eigenvalues": eigenvalues[:k+1].tolist(),
        }

        if verbose:
            print(f"  Anchor discovery ({n} anchors → {len(discovered)} clusters):")
            print(f"    Silhouette: {silhouette:.3f}")
            for da in discovered:
                members = [anchor_names[i] for i in da.centroid_indices]
                print(f"    {da.name} ({da.inferred_type}, size={da.size}, "
                      f"coherence={da.coherence:.3f}): {members}")

        return schema, discovered, stats

    @staticmethod
    def _kmeans(features: np.ndarray, k: int, max_iter: int = 50) -> np.ndarray:
        """Simple k-means clustering."""
        n = features.shape[0]
        if n == 0 or k == 0:
            return np.zeros(0, dtype=np.int32)

        rng = np.random.RandomState(42)
        centroids = [features[rng.randint(n)]]
        for _ in range(k - 1):
            dists = np.array([
                min(np.sum((f - c) ** 2) for c in centroids)
                for f in features
            ])
            dists_safe = dists / (dists.sum() + 1e-10)
            next_idx = rng.choice(n, p=dists_safe)
            centroids.append(features[next_idx])
        centroids = np.array(centroids)

        labels = np.zeros(n, dtype=np.int32)
        for _ in range(max_iter):
            dists = np.array([
                np.sum((features - centroids[j]) ** 2, axis=1)
                for j in range(k)
            ]).T
            new_labels = dists.argmin(axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for j in range(k):
                members = features[labels == j]
                if len(members) > 0:
                    centroids[j] = members.mean(axis=0)
        return labels

    # ── SUBGRAPH / GRAPH-LEVEL CLASSIFICATION ─────────────────────────

    def subgraph_classify(self, indices: list[int],
                          head: nn.Module,
                          graph: HyperGraph | None = None,
                          pool_mode: str = "mean",
                          max_infons: int = 500) -> torch.Tensor:
        """Apply a head to a pooled subgraph embedding.

        Args:
            indices: node indices to pool over. For graph-level
                classification pass graph.infon_indices.
            head: any nn.Module that maps (batch, hidden) → output.
            pool_mode: mean / sum / max / attention
            graph: optional pre-built HyperGraph
            max_infons: cap when rebuilding

        Returns:
            head output on the pooled embedding, with a leading batch
            dim of 1.
        """
        if graph is None:
            graph = self.builder.build(max_infons=max_infons,
                                       feature_dim=self.hidden_dim)
        if not self._fitted:
            self.fit(graph=graph)

        # Swap pool mode if different from default
        if pool_mode != self.subgraph_pool.mode:
            pool = SubgraphPool(self.hidden_dim, mode=pool_mode)
        else:
            pool = self.subgraph_pool

        with torch.no_grad():
            h = self.forward(graph)
            pooled = pool(h, indices).unsqueeze(0)
            out = head(pooled)
        return out

    # ── TIME-TO-EVENT HEAD ────────────────────────────────────────────

    def train_time_to_event_head(self, graph: HyperGraph | None = None,
                                 epochs: int = 80,
                                 lr: float = 1e-2,
                                 max_infons: int = 500,
                                 verbose: bool = False) -> dict:
        """Train the Weibull time-to-event head on NEXT-edge intervals.

        Supervision comes from NEXT edges where both endpoints have
        parseable ISO-format timestamps. Target is the positive time
        delta (in days) between the two infons. GNN layers are frozen.
        """
        from datetime import datetime

        if graph is None:
            graph = self.builder.build(max_infons=max_infons,
                                       feature_dim=self.hidden_dim)
        if not self._fitted:
            self.fit(graph=graph, verbose=verbose)

        next_edges = self.store.get_edges(edge_type="NEXT",
                                          limit=max_infons * 5)

        def parse_ts(s):
            if not s:
                return None
            for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
                try:
                    return datetime.strptime(s.split("T")[0], "%Y-%m-%d")
                except Exception:
                    continue
            return None

        pairs = []
        for edge in next_edges:
            earlier = self.store.get_infon(edge.source)
            later = self.store.get_infon(edge.target)
            if earlier is None or later is None:
                continue
            t_e = parse_ts(getattr(earlier, "timestamp", None))
            t_l = parse_ts(getattr(later, "timestamp", None))
            if t_e is None or t_l is None:
                continue
            delta = (t_l - t_e).days
            if delta <= 0:
                continue
            e_idx = graph.infon_map.get(edge.source)
            if e_idx is None:
                continue
            pairs.append((e_idx, float(delta)))

        # Fallback: if no timestamped pairs, synthesize 1.0-day intervals
        # for every NEXT edge so the head has *something* to train on.
        if not pairs:
            for edge in next_edges:
                e_idx = graph.infon_map.get(edge.source)
                if e_idx is not None:
                    pairs.append((e_idx, 1.0))

        if not pairs:
            if verbose:
                print("  no NEXT supervision for time-to-event head")
            return {"epochs": 0, "final_loss": float("nan"),
                    "n_targets": 0, "losses": []}

        src_indices = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        deltas = torch.tensor([p[1] for p in pairs], dtype=torch.float32)

        # Freeze everything except the time head
        for p in self.parameters():
            p.requires_grad = False
        for p in self.time_to_event_head.parameters():
            p.requires_grad = True

        optimizer = torch.optim.Adam(self.time_to_event_head.parameters(), lr=lr)
        losses = []
        self.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            h = self.forward(graph)
            src_h = h[src_indices]
            loss = self.time_to_event_head.nll(src_h, deltas)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if verbose and (epoch + 1) % 20 == 0:
                print(f"    time-head epoch {epoch+1}/{epochs}: "
                      f"loss={loss.item():.4f}")

        for p in self.parameters():
            p.requires_grad = True
        self.eval()
        self._time_head_trained = True

        return {
            "epochs": epochs,
            "final_loss": losses[-1],
            "first_loss": losses[0],
            "n_targets": len(pairs),
            "losses": losses,
        }

    def predict_time_to_event(self, infon_id: str,
                              graph: HyperGraph | None = None,
                              max_infons: int = 500) -> dict:
        """Return expected time-to-next-event (days) + Weibull params."""
        if graph is None:
            graph = self.builder.build(max_infons=max_infons,
                                       feature_dim=self.hidden_dim)
        idx = graph.infon_map.get(infon_id)
        if idx is None:
            return {"expected_days": None, "log_scale": None,
                    "log_shape": None, "trained": self._time_head_trained}

        with torch.no_grad():
            h = self.forward(graph)
            out = self.time_to_event_head(h[idx].unsqueeze(0))
            mean_t = self.time_to_event_head.predict_mean_time(
                h[idx].unsqueeze(0)
            )
        return {
            "expected_days": float(mean_t.item()),
            "log_scale": float(out[0, 0].item()),
            "log_shape": float(out[0, 1].item()),
            "trained": self._time_head_trained,
        }

    # ── RISK RANKING HEAD ─────────────────────────────────────────────

    def train_risk_head(self, graph: HyperGraph | None = None,
                        epochs: int = 60,
                        lr: float = 1e-2,
                        max_infons: int = 500,
                        verbose: bool = False) -> dict:
        """Train the risk head via pairwise ranking on heuristic labels.

        Positives: infons with polarity < 0 OR participating in a
        CONTRADICTS edge (as either endpoint). Negatives: random other
        infons. Loss: margin ranking loss pushing positives above
        negatives by margin 0.3.
        """
        import random

        if graph is None:
            graph = self.builder.build(max_infons=max_infons,
                                       feature_dim=self.hidden_dim)
        if not self._fitted:
            self.fit(graph=graph, verbose=verbose)

        # Collect positives
        pos_ids = set()
        for iid in graph.infon_map:
            inf = self.store.get_infon(iid)
            if inf is None:
                continue
            if getattr(inf, "polarity", 1) < 0:
                pos_ids.add(iid)
        for edge in self.store.get_edges(edge_type="CONTRADICTS",
                                         limit=max_infons * 5):
            pos_ids.add(edge.source)
            pos_ids.add(edge.target)

        all_ids = list(graph.infon_map.keys())
        neg_ids = [i for i in all_ids if i not in pos_ids]

        if not pos_ids or not neg_ids:
            if verbose:
                print("  insufficient positive/negative examples for risk head")
            return {"epochs": 0, "final_loss": float("nan"),
                    "n_positive": len(pos_ids), "n_negative": len(neg_ids),
                    "losses": []}

        pos_idx_list = [graph.infon_map[i] for i in pos_ids
                        if i in graph.infon_map]
        neg_idx_list = [graph.infon_map[i] for i in neg_ids
                        if i in graph.infon_map]

        # Build pairs: each positive vs a random sampling of negatives
        rng = random.Random(42)
        pairs = []
        for _ in range(epochs):
            # regenerate each "epoch": one pair per positive
            epoch_pairs = []
            for p in pos_idx_list:
                n = rng.choice(neg_idx_list)
                epoch_pairs.append((p, n))
            pairs.append(epoch_pairs)

        # Freeze all but the risk head
        for p in self.parameters():
            p.requires_grad = False
        for p in self.risk_head.parameters():
            p.requires_grad = True

        optimizer = torch.optim.Adam(self.risk_head.parameters(), lr=lr)
        losses = []
        self.train()
        for epoch, ep_pairs in enumerate(pairs):
            optimizer.zero_grad()
            h = self.forward(graph)
            pos_h = h[torch.tensor([p for p, _ in ep_pairs], dtype=torch.long)]
            neg_h = h[torch.tensor([n for _, n in ep_pairs], dtype=torch.long)]
            pos_score = self.risk_head(pos_h)
            neg_score = self.risk_head(neg_h)
            # Margin ranking: want pos > neg by >= margin
            margin = 0.3
            loss = F.relu(margin - (pos_score - neg_score)).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if verbose and (epoch + 1) % 15 == 0:
                print(f"    risk-head epoch {epoch+1}/{epochs}: "
                      f"loss={loss.item():.4f}")

        for p in self.parameters():
            p.requires_grad = True
        self.eval()
        self._risk_head_trained = True

        return {
            "epochs": epochs,
            "final_loss": losses[-1],
            "first_loss": losses[0],
            "n_positive": len(pos_idx_list),
            "n_negative": len(neg_idx_list),
            "losses": losses,
        }

    def rank_risk(self, graph: HyperGraph | None = None,
                  group_by: str = "subject",
                  top_k: int = 10,
                  max_infons: int = 500,
                  verbose: bool = False) -> list[dict]:
        """Rank actors/entities by aggregated risk score.

        Args:
            group_by: which role to aggregate over — 'subject', 'object',
                      or 'infon' (returns per-infon scores instead).
            top_k: number of entries to return
        """
        if graph is None:
            graph = self.builder.build(max_infons=max_infons,
                                       feature_dim=self.hidden_dim)
        if not self._fitted:
            self.fit(graph=graph)

        with torch.no_grad():
            h = self.forward(graph)

        if group_by == "infon":
            scores = []
            for iid, idx in graph.infon_map.items():
                score = float(self.risk_head(h[idx].unsqueeze(0)).item())
                inf = self.store.get_infon(iid)
                scores.append({"infon_id": iid, "risk": score,
                               "subject": inf.subject if inf else None,
                               "predicate": inf.predicate if inf else None,
                               "object": inf.object if inf else None})
            scores.sort(key=lambda x: x["risk"], reverse=True)
            return scores[:top_k]

        # Aggregate by subject/object anchor
        from collections import defaultdict
        agg_scores = defaultdict(list)
        for iid, idx in graph.infon_map.items():
            inf = self.store.get_infon(iid)
            if inf is None:
                continue
            entity = getattr(inf, group_by, None)
            if not entity:
                continue
            score = float(self.risk_head(h[idx].unsqueeze(0)).item())
            agg_scores[entity].append(score)

        ranked = []
        for entity, vals in agg_scores.items():
            ranked.append({
                "entity": entity,
                "mean_risk": sum(vals) / len(vals),
                "max_risk": max(vals),
                "n_infons": len(vals),
            })
        ranked.sort(key=lambda x: x["mean_risk"], reverse=True)
        top = ranked[:top_k]

        if verbose:
            print(f"  top {len(top)} entities by mean risk "
                  f"(group_by={group_by}):")
            for r in top:
                print(f"    {r['entity']:20s}  mean={r['mean_risk']:.3f}  "
                      f"max={r['max_risk']:.3f}  (n={r['n_infons']})")

        return top

    # ── ANOMALY LOCALIZATION HEAD ─────────────────────────────────────

    def train_anomaly_head(self, graph: HyperGraph | None = None,
                           epochs: int = 100,
                           lr: float = 1e-2,
                           max_infons: int = 500,
                           verbose: bool = False) -> dict:
        """Self-supervised autoencoder on infon embeddings.

        No labels required: the head learns to reconstruct the (frozen)
        GNN embeddings of all infons. At inference time, infons whose
        embeddings reconstruct poorly are flagged as anomalous.
        """
        if graph is None:
            graph = self.builder.build(max_infons=max_infons,
                                       feature_dim=self.hidden_dim)
        if not self._fitted:
            self.fit(graph=graph, verbose=verbose)

        indices = list(graph.infon_map.values())
        if not indices:
            return {"epochs": 0, "final_loss": float("nan"), "n_targets": 0,
                    "losses": []}
        idx_tensor = torch.tensor(indices, dtype=torch.long)

        for p in self.parameters():
            p.requires_grad = False
        for p in self.anomaly_head.parameters():
            p.requires_grad = True

        optimizer = torch.optim.Adam(self.anomaly_head.parameters(), lr=lr)
        losses = []
        self.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            with torch.no_grad():
                h = self.forward(graph)
            infon_h = h[idx_tensor]
            recon = self.anomaly_head(infon_h)
            loss = F.mse_loss(recon, infon_h)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if verbose and (epoch + 1) % 25 == 0:
                print(f"    anomaly-head epoch {epoch+1}/{epochs}: "
                      f"loss={loss.item():.4f}")

        for p in self.parameters():
            p.requires_grad = True
        self.eval()
        self._anomaly_head_trained = True

        return {
            "epochs": epochs,
            "final_loss": losses[-1],
            "first_loss": losses[0],
            "n_targets": len(indices),
            "losses": losses,
        }

    def score_anomalies(self, graph: HyperGraph | None = None,
                        top_k: int = 10,
                        max_infons: int = 500,
                        verbose: bool = False) -> list[dict]:
        """Return the top-k most anomalous infons by reconstruction error."""
        if graph is None:
            graph = self.builder.build(max_infons=max_infons,
                                       feature_dim=self.hidden_dim)
        if not self._fitted:
            self.fit(graph=graph)

        with torch.no_grad():
            h = self.forward(graph)

        scores = []
        for iid, idx in graph.infon_map.items():
            infon_h = h[idx].unsqueeze(0)
            score = float(self.anomaly_head.anomaly_score(infon_h).item())
            inf = self.store.get_infon(iid)
            scores.append({
                "infon_id": iid,
                "anomaly_score": score,
                "subject": inf.subject if inf else None,
                "predicate": inf.predicate if inf else None,
                "object": inf.object if inf else None,
            })
        scores.sort(key=lambda x: x["anomaly_score"], reverse=True)
        top = scores[:top_k]

        if verbose:
            print(f"  top {len(top)} anomalous infons:")
            for r in top:
                label = f"{r['subject']}/{r['predicate']}/{r['object']}"
                print(f"    {label:40s}  score={r['anomaly_score']:.4f}")

        return top

    # ── COUNTERFACTUAL WRAPPER ────────────────────────────────────────

    def counterfactual(self, target_infon_id: str,
                       intervention: dict,
                       max_infons: int = 500,
                       verbose: bool = False) -> dict:
        """Run a counterfactual simulation by perturbing the graph.

        Args:
            target_infon_id: the infon whose belief we want to diff.
            intervention: one of
                {"remove_infon": infon_id}
                {"remove_edge": (src_id, tgt_id, edge_type)}
                {"swap_anchor": {"role": "subject|predicate|object",
                                 "old": name, "new": name}}
                {"zero_node": node_id}  # zero the feature vector

        Returns:
            {baseline_mass, counterfactual_mass, delta} where delta is
            the element-wise difference of (S, R, U, θ).
        """
        # Baseline on unmodified graph
        baseline_graph = self.builder.build(max_infons=max_infons,
                                            feature_dim=self.hidden_dim)
        if not self._fitted:
            self.fit(graph=baseline_graph, verbose=verbose)

        base_idx = baseline_graph.infon_map.get(target_infon_id)
        if base_idx is None:
            return {"error": "target_infon_id not found in graph"}

        with torch.no_grad():
            base_h = self.forward(baseline_graph)
            base_mass = self.mass_readout.to_mass_functions(
                base_h[base_idx].unsqueeze(0)
            )[0]

        # Construct a perturbed HyperGraph
        g = baseline_graph
        new_infon_map = dict(g.infon_map)
        new_anchor_map = dict(g.anchor_map)
        node_features = g.node_features.clone()
        edge_index = g.edge_index.clone()
        edge_types = g.edge_types.clone()
        edge_weights = g.edge_weights.clone()

        keep_mask = torch.ones(edge_index.shape[1], dtype=torch.bool)

        if "remove_infon" in intervention:
            rid = intervention["remove_infon"]
            ridx = new_infon_map.get(rid)
            if ridx is not None:
                # Zero its features and mask out its edges
                node_features[ridx] = 0
                keep_mask &= ~((edge_index[0] == ridx) | (edge_index[1] == ridx))

        if "zero_node" in intervention:
            nid = intervention["zero_node"]
            nidx = new_infon_map.get(nid, new_anchor_map.get(nid))
            if nidx is not None:
                node_features[nidx] = 0

        if "remove_edge" in intervention:
            src_id, tgt_id, etype = intervention["remove_edge"]
            sidx = new_infon_map.get(src_id, new_anchor_map.get(src_id))
            tidx = new_infon_map.get(tgt_id, new_anchor_map.get(tgt_id))
            r = REL_TO_IDX.get(etype)
            if sidx is not None and tidx is not None and r is not None:
                keep_mask &= ~(
                    (edge_index[0] == sidx) &
                    (edge_index[1] == tidx) &
                    (edge_types == r)
                )

        if "do_anchor" in intervention:
            # Replace one anchor's feature vector with another's —
            # "imagine it had been the replacement instead".
            src_name, replacement_name = intervention["do_anchor"]
            sidx = new_anchor_map.get(src_name)
            ridx = new_anchor_map.get(replacement_name)
            if sidx is not None and ridx is not None:
                node_features[sidx] = g.node_features[ridx].clone()

        edge_index = edge_index[:, keep_mask]
        edge_types = edge_types[keep_mask]
        edge_weights = edge_weights[keep_mask]

        cf_graph = HyperGraph(
            node_ids=g.node_ids,
            node_types=g.node_types,
            node_features=node_features,
            edge_index=edge_index,
            edge_types=edge_types,
            edge_weights=edge_weights,
            anchor_type_groups=g.anchor_type_groups,
            infon_indices=g.infon_indices,
            infon_map=new_infon_map,
            anchor_map=new_anchor_map,
            situation_features=g.situation_features,
        )

        with torch.no_grad():
            cf_h = self.forward(cf_graph)
            cf_mass = self.mass_readout.to_mass_functions(
                cf_h[base_idx].unsqueeze(0)
            )[0]

        delta = {
            "supports": cf_mass.supports - base_mass.supports,
            "refutes": cf_mass.refutes - base_mass.refutes,
            "uncertain": cf_mass.uncertain - base_mass.uncertain,
            "theta": cf_mass.theta - base_mass.theta,
        }

        if verbose:
            print(f"  counterfactual for {target_infon_id[:12]}:")
            print(f"    intervention: {intervention}")
            print(f"    baseline:       S={base_mass.supports:.3f} "
                  f"R={base_mass.refutes:.3f} θ={base_mass.theta:.3f}")
            print(f"    counterfactual: S={cf_mass.supports:.3f} "
                  f"R={cf_mass.refutes:.3f} θ={cf_mass.theta:.3f}")
            print(f"    delta:          ΔS={delta['supports']:+.3f} "
                  f"ΔR={delta['refutes']:+.3f} Δθ={delta['theta']:+.3f}")

        return {
            "baseline_mass": base_mass,
            "counterfactual_mass": cf_mass,
            "delta": delta,
        }

    # ── ATTRIBUTION VIA INTEGRATED GRADIENTS ──────────────────────────

    def attribute(self, target_infon_id: str,
                  target_class: str = "supports",
                  steps: int = 32,
                  top_k: int = 10,
                  max_infons: int = 500,
                  verbose: bool = False) -> list[dict]:
        """Integrated-gradients attribution of the target's belief to all
        other infon nodes.

        Args:
            target_infon_id: which infon's belief we're explaining.
            target_class: 'supports' | 'refutes' | 'uncertain' | 'theta'
            steps: number of integration steps (baseline = zero features)
            top_k: how many attributing nodes to return

        Returns:
            ranked list of {infon_id, score, subject, predicate, object}
        """
        CLASS_IDX = {"supports": 0, "refutes": 1, "uncertain": 2, "theta": 3}
        cls = CLASS_IDX.get(target_class)
        if cls is None:
            raise ValueError(f"target_class must be one of {list(CLASS_IDX)}")

        graph = self.builder.build(max_infons=max_infons,
                                   feature_dim=self.hidden_dim)
        if not self._fitted:
            self.fit(graph=graph, verbose=verbose)

        target_idx = graph.infon_map.get(target_infon_id)
        if target_idx is None:
            return []

        # Integrated gradients: interpolate feature tensor from 0 → actual
        baseline = torch.zeros_like(graph.node_features)
        actual = graph.node_features
        total_grad = torch.zeros_like(actual)

        self.train()  # need grad, but we don't optimize
        for step in range(1, steps + 1):
            alpha = step / steps
            interp = (baseline + alpha * (actual - baseline)).clone().detach()
            interp.requires_grad_(True)

            g_stepped = HyperGraph(
                node_ids=graph.node_ids,
                node_types=graph.node_types,
                node_features=interp,
                edge_index=graph.edge_index,
                edge_types=graph.edge_types,
                edge_weights=graph.edge_weights,
                anchor_type_groups=graph.anchor_type_groups,
                infon_indices=graph.infon_indices,
                infon_map=graph.infon_map,
                anchor_map=graph.anchor_map,
                situation_features=graph.situation_features,
            )
            h = self.forward(g_stepped)
            logits = self.mass_readout(h[target_idx].unsqueeze(0))
            score = logits[0, cls]
            grad = torch.autograd.grad(score, interp, retain_graph=False)[0]
            total_grad += grad

        self.eval()
        # IG attribution per node = (x - baseline) * avg_grad
        attr_per_node = ((actual - baseline) * (total_grad / steps)).sum(dim=-1)
        attr_np = attr_per_node.detach().cpu().numpy()

        results = []
        for iid, idx in graph.infon_map.items():
            if iid == target_infon_id:
                continue
            inf = self.store.get_infon(iid)
            results.append({
                "infon_id": iid,
                "score": float(attr_np[idx]),
                "subject": inf.subject if inf else None,
                "predicate": inf.predicate if inf else None,
                "object": inf.object if inf else None,
            })
        results.sort(key=lambda x: abs(x["score"]), reverse=True)
        top = results[:top_k]

        if verbose:
            inf = self.store.get_infon(target_infon_id)
            if inf:
                print(f"  target: {inf.subject}/{inf.predicate}/"
                      f"{inf.object}  class={target_class}")
            print(f"  top {len(top)} attributing infons (|IG| score):")
            for r in top:
                label = f"{r['subject']}/{r['predicate']}/{r['object']}"
                print(f"    {label:40s}  IG={r['score']:+.4f}")

        return top

    # ── CAUSAL VIEW & ROOT-CAUSE ANALYSIS ─────────────────────────────

    def causal_view(self, graph: HyperGraph | None = None,
                    edge_types: tuple = ("CAUSES", "NEXT"),
                    max_infons: int = 500) -> "CausalView":
        """Build a DAG view over infon-to-infon causal / temporal edges.

        Collects all edges of the given types, breaks cycles by
        preferring higher-confidence edges (removes the lowest-weight
        edge on any back-path discovered during topological sort), and
        exposes ancestor / descendant / path queries.
        """
        if graph is None:
            graph = self.builder.build(max_infons=max_infons,
                                       feature_dim=self.hidden_dim)

        # Collect edges of interest
        adj: dict[str, list[tuple[str, float, str]]] = defaultdict(list)
        radj: dict[str, list[tuple[str, float, str]]] = defaultdict(list)
        edge_list = []
        for etype in edge_types:
            for edge in self.store.get_edges(edge_type=etype,
                                             limit=max_infons * 10):
                if (edge.source in graph.infon_map and
                        edge.target in graph.infon_map):
                    edge_list.append((edge.source, edge.target,
                                      edge.weight, etype))

        # Break cycles: Kahn's algorithm on a working copy; if we ever
        # run out of zero-in-degree nodes before emptying, drop the
        # lowest-weight edge participating in a remaining cycle.
        # Simpler heuristic that suffices for our graph sizes: sort
        # edges by weight descending and add greedily, skipping any
        # edge that would create a cycle via DFS.
        edge_list.sort(key=lambda e: -e[2])
        accepted = []
        for src, tgt, w, etype in edge_list:
            # Would adding src→tgt create a cycle? i.e. is src
            # reachable from tgt in the accepted graph?
            if self._reachable(adj, tgt, src):
                continue
            adj[src].append((tgt, w, etype))
            radj[tgt].append((src, w, etype))
            accepted.append((src, tgt, w, etype))

        return CausalView(
            adj=dict(adj),
            reverse_adj=dict(radj),
            edges=accepted,
            infon_ids=list(graph.infon_map.keys()),
        )

    @staticmethod
    def _reachable(adj: dict, src: str, tgt: str) -> bool:
        """Is tgt reachable from src in the current adj dict?"""
        if src == tgt:
            return True
        visited = {src}
        stack = [src]
        while stack:
            node = stack.pop()
            for nb, _, _ in adj.get(node, []):
                if nb == tgt:
                    return True
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        return False

    def root_cause(self, target_infon_id: str,
                   graph: HyperGraph | None = None,
                   top_k: int = 5,
                   max_infons: int = 500,
                   attribution_steps: int = 16,
                   verbose: bool = False) -> list[dict]:
        """Rank likely root causes of a target infon.

        Combines three signals per candidate ancestor:
            - integrated-gradient attribution to the target's SUPPORTS
              mass (explanatory power);
            - anomaly score (is this ancestor itself unusual?);
            - inverse path distance to the target (closer = stronger).

        Score = |IG| × anomaly × (1 / distance). Returns top-k
        ancestors with full diagnostic info.

        Requires the anomaly head to be trained; trains it on demand
        if it isn't yet.
        """
        if graph is None:
            graph = self.builder.build(max_infons=max_infons,
                                       feature_dim=self.hidden_dim)
        if not self._fitted:
            self.fit(graph=graph, verbose=verbose)
        if not self._anomaly_head_trained:
            self.train_anomaly_head(graph=graph, verbose=False)

        # 1. Build causal view and find ancestors with distances
        cview = self.causal_view(graph=graph)
        ancestors = cview.ancestors_with_distance(target_infon_id)
        if not ancestors:
            if verbose:
                print(f"  no causal ancestors found for {target_infon_id[:12]}")
            return []

        # 2. Compute IG attribution for SUPPORTS
        attr = self.attribute(target_infon_id, target_class="supports",
                              steps=attribution_steps,
                              top_k=len(graph.infon_map),
                              verbose=False)
        attr_by_id = {a["infon_id"]: a["score"] for a in attr}

        # 3. Compute anomaly scores for the ancestor set
        with torch.no_grad():
            h = self.forward(graph)

        ranked = []
        for anc_id, distance in ancestors.items():
            idx = graph.infon_map.get(anc_id)
            if idx is None:
                continue
            infon_h = h[idx].unsqueeze(0)
            anom = float(self.anomaly_head.anomaly_score(infon_h).item())
            ig = attr_by_id.get(anc_id, 0.0)
            dist_factor = 1.0 / (1.0 + distance)
            score = abs(ig) * anom * dist_factor
            inf = self.store.get_infon(anc_id)
            ranked.append({
                "infon_id": anc_id,
                "score": score,
                "ig": ig,
                "anomaly": anom,
                "distance": distance,
                "subject": inf.subject if inf else None,
                "predicate": inf.predicate if inf else None,
                "object": inf.object if inf else None,
            })
        ranked.sort(key=lambda x: x["score"], reverse=True)
        top = ranked[:top_k]

        if verbose:
            inf = self.store.get_infon(target_infon_id)
            if inf:
                print(f"  target: {inf.subject}/{inf.predicate}/{inf.object}")
            print(f"  {len(ancestors)} causal ancestors; top {len(top)}:")
            for r in top:
                label = f"{r['subject']}/{r['predicate']}/{r['object']}"
                print(f"    {label:40s}  score={r['score']:.4f}  "
                      f"IG={r['ig']:+.3f}  anom={r['anomaly']:.3f}  "
                      f"d={r['distance']}")

        return top

    # ── REFUTE WITH PLACEBO INTERVENTIONS ─────────────────────────────

    def refute(self, target_infon_id: str,
               intervention: dict,
               n_trials: int = 10,
               method: str = "placebo",
               max_infons: int = 500,
               verbose: bool = False) -> dict:
        """Compare the observed intervention effect against random placebos.

        method='placebo': run n_trials random interventions of the same
        structural type (remove a random infon / zero a random anchor /
        do_anchor with a random replacement) and measure their effect
        on the target. Return the observed effect magnitude and the
        p-value-like fraction of placebos whose magnitude matched or
        exceeded the observed.
        """
        import random

        # Observed effect
        observed = self.counterfactual(target_infon_id, intervention,
                                       max_infons=max_infons, verbose=False)
        if "error" in observed:
            return observed

        obs_mag = sum(abs(v) for v in observed["delta"].values())

        # Build placebo interventions of matching structure
        graph = self.builder.build(max_infons=max_infons,
                                   feature_dim=self.hidden_dim)
        rng = random.Random(42)
        placebo_effects = []

        if "remove_infon" in intervention:
            all_ids = [i for i in graph.infon_map if i != target_infon_id
                       and i != intervention["remove_infon"]]
            rng.shuffle(all_ids)
            for random_id in all_ids[:n_trials]:
                r = self.counterfactual(target_infon_id,
                                        {"remove_infon": random_id},
                                        verbose=False)
                mag = sum(abs(v) for v in r["delta"].values())
                placebo_effects.append(mag)

        elif "do_anchor" in intervention:
            src, _ = intervention["do_anchor"]
            all_anchors = [a for a in graph.anchor_map if a != src]
            rng.shuffle(all_anchors)
            for random_anchor in all_anchors[:n_trials]:
                r = self.counterfactual(target_infon_id,
                                        {"do_anchor": (src, random_anchor)},
                                        verbose=False)
                mag = sum(abs(v) for v in r["delta"].values())
                placebo_effects.append(mag)

        elif "zero_node" in intervention:
            all_nodes = [n for n in graph.anchor_map
                         if n != intervention["zero_node"]]
            rng.shuffle(all_nodes)
            for random_node in all_nodes[:n_trials]:
                r = self.counterfactual(target_infon_id,
                                        {"zero_node": random_node},
                                        verbose=False)
                mag = sum(abs(v) for v in r["delta"].values())
                placebo_effects.append(mag)
        else:
            return {"error": f"refute not implemented for "
                             f"{list(intervention.keys())}"}

        # p-value-like: fraction of placebos whose effect was ≥ observed
        if placebo_effects:
            n_extreme = sum(1 for x in placebo_effects if x >= obs_mag)
            p_like = n_extreme / len(placebo_effects)
            placebo_mean = sum(placebo_effects) / len(placebo_effects)
            placebo_max = max(placebo_effects)
        else:
            p_like = 1.0
            placebo_mean = 0.0
            placebo_max = 0.0

        result = {
            "observed_magnitude": obs_mag,
            "observed_delta": observed["delta"],
            "placebo_effects": placebo_effects,
            "placebo_mean": placebo_mean,
            "placebo_max": placebo_max,
            "p_value_like": p_like,
            "n_trials": len(placebo_effects),
        }

        if verbose:
            print(f"  refute ({method}, n={len(placebo_effects)}):")
            print(f"    observed  |Δ| = {obs_mag:.4f}")
            print(f"    placebo mean  = {placebo_mean:.4f}")
            print(f"    placebo max   = {placebo_max:.4f}")
            print(f"    p_value_like  = {p_like:.3f}  "
                  f"({'significant' if p_like < 0.1 else 'not significant'})")

        return result

    # ── RECOMMENDER ───────────────────────────────────────────────────

    def train_recommender_head(self, graph: HyperGraph | None = None,
                               user_type: str = "actor",
                               item_types: tuple = ("feature", "actor"),
                               interaction_anchors: tuple = (
                                   "adopts", "prefers", "avoids",
                                   "selects", "rejects"),
                               epochs: int = 60,
                               lr: float = 1e-2,
                               diversity_weight: float = 0.1,
                               max_infons: int = 500,
                               verbose: bool = False) -> dict:
        """Train the recommender head via BPR on extracted interactions.

        Walks infons whose predicate is one of `interaction_anchors` and
        treats them as (user, item, polarity) tuples. Positives are
        polarity > 0; hard negatives are polarity < 0 on the same pair
        of anchor types. Soft negatives are random non-interacted items.

        Loss = BPR + diversity_weight · DiversityHead on the top-5 scored
        items per user.
        """
        import random

        if graph is None:
            graph = self.builder.build(max_infons=max_infons,
                                       feature_dim=self.hidden_dim)
        if not self._fitted:
            self.fit(graph=graph, verbose=verbose)

        # Collect anchor indices by type
        user_anchor_ids = [n for n in graph.anchor_map
                           if self.schema.types.get(n) == user_type]
        item_anchor_ids = [n for n in graph.anchor_map
                           if self.schema.types.get(n) in item_types]
        if not user_anchor_ids or not item_anchor_ids:
            if verbose:
                print(f"  no users or items found for user_type={user_type}")
            return {"epochs": 0, "n_positive": 0, "n_hard_negative": 0,
                    "losses": [], "final_loss": float("nan")}

        item_name_to_idx = {n: graph.anchor_map[n] for n in item_anchor_ids}
        user_name_to_idx = {n: graph.anchor_map[n] for n in user_anchor_ids}

        # Walk interaction infons
        interaction_set = set(interaction_anchors)
        positives: list[tuple[int, int]] = []  # (user_idx, item_idx)
        hard_negatives: list[tuple[int, int]] = []
        observed_pairs: set[tuple[int, int]] = set()

        for iid in graph.infon_map:
            inf = self.store.get_infon(iid)
            if inf is None or inf.predicate not in interaction_set:
                continue
            if inf.subject not in user_name_to_idx:
                continue
            if inf.object not in item_name_to_idx:
                continue
            u_idx = user_name_to_idx[inf.subject]
            i_idx = item_name_to_idx[inf.object]
            observed_pairs.add((u_idx, i_idx))
            pol = getattr(inf, "polarity", 1)
            # "avoid" / "reject" style anchors flip polarity semantically
            negative_relations = {"avoids", "rejects"}
            is_negative = (pol < 0) or (inf.predicate in negative_relations)
            if is_negative:
                hard_negatives.append((u_idx, i_idx))
            else:
                positives.append((u_idx, i_idx))

        if not positives:
            if verbose:
                print("  no positive interactions found")
            return {"epochs": 0, "n_positive": 0,
                    "n_hard_negative": len(hard_negatives),
                    "losses": [], "final_loss": float("nan")}

        item_indices_tensor = torch.tensor(list(item_name_to_idx.values()),
                                           dtype=torch.long)

        # Freeze everything except the recommender head
        for p in self.parameters():
            p.requires_grad = False
        for p in self.recommender_head.parameters():
            p.requires_grad = True

        optimizer = torch.optim.Adam(self.recommender_head.parameters(), lr=lr)
        rng = random.Random(42)
        losses = []
        self.train()

        for epoch in range(epochs):
            optimizer.zero_grad()
            h = self.forward(graph)

            # Build positive/negative pair batches
            pos_u_idx = torch.tensor([p[0] for p in positives], dtype=torch.long)
            pos_i_idx = torch.tensor([p[1] for p in positives], dtype=torch.long)

            # Sample soft negatives: for each positive, a random
            # non-observed item
            soft_neg_idx = []
            for u, _ in positives:
                while True:
                    candidate = rng.choice(list(item_name_to_idx.values()))
                    if (u, candidate) not in observed_pairs:
                        soft_neg_idx.append(candidate)
                        break
                    if len(observed_pairs) >= len(item_name_to_idx):
                        soft_neg_idx.append(candidate)
                        break
            soft_neg_idx = torch.tensor(soft_neg_idx, dtype=torch.long)

            u_h = h[pos_u_idx]
            pos_i_h = h[pos_i_idx]
            neg_i_h = h[soft_neg_idx]

            pos_score = self.recommender_head(u_h, pos_i_h)
            neg_score = self.recommender_head(u_h, neg_i_h)
            bpr = -F.logsigmoid(pos_score - neg_score).mean()

            # Hard negatives: push their score BELOW a shared positive
            hard_loss = torch.zeros(1).squeeze()
            if hard_negatives:
                hn_u_idx = torch.tensor([h[0] for h in hard_negatives],
                                        dtype=torch.long)
                hn_i_idx = torch.tensor([h[1] for h in hard_negatives],
                                        dtype=torch.long)
                hn_score = self.recommender_head(h[hn_u_idx], h[hn_i_idx])
                # Margin-style: hard negatives should score ≤ 0
                hard_loss = F.relu(hn_score + 0.1).mean()

            # Diversity regularizer on top-5 scored items for a random user
            div_loss = torch.zeros(1).squeeze()
            if diversity_weight > 0:
                rand_u = rng.choice(list(user_name_to_idx.values()))
                all_item_h = h[item_indices_tensor]
                scores_for_u = self.recommender_head(
                    h[rand_u].unsqueeze(0).expand(all_item_h.shape[0], -1),
                    all_item_h,
                )
                topk = min(5, all_item_h.shape[0])
                top_idx = scores_for_u.topk(topk).indices
                div_loss = self.diversity_head(all_item_h[top_idx])

            loss = bpr + 0.5 * hard_loss + diversity_weight * div_loss
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if verbose and (epoch + 1) % 15 == 0:
                print(f"    rec-head epoch {epoch+1}/{epochs}: "
                      f"total={loss.item():.4f}  bpr={bpr.item():.4f}  "
                      f"hard={hard_loss.item():.4f}  div={div_loss.item():.4f}")

        for p in self.parameters():
            p.requires_grad = True
        self.eval()
        self._recommender_trained = True

        return {
            "epochs": epochs,
            "final_loss": losses[-1],
            "first_loss": losses[0],
            "n_positive": len(positives),
            "n_hard_negative": len(hard_negatives),
            "n_users": len(user_anchor_ids),
            "n_items": len(item_anchor_ids),
            "losses": losses,
        }

    def recommend(self, user: str,
                  graph: HyperGraph | None = None,
                  k: int = 5,
                  item_types: tuple = ("feature", "actor"),
                  max_infons: int = 500,
                  verbose: bool = False) -> list[dict]:
        """Return top-k recommended items for a given user anchor."""
        if graph is None:
            graph = self.builder.build(max_infons=max_infons,
                                       feature_dim=self.hidden_dim)
        u_idx = graph.anchor_map.get(user)
        if u_idx is None:
            if verbose:
                print(f"  user {user!r} not in schema")
            return []

        item_names = [n for n in graph.anchor_map
                      if self.schema.types.get(n) in item_types
                      and n != user]
        if not item_names:
            return []

        with torch.no_grad():
            h = self.forward(graph)
            u_h = h[u_idx].unsqueeze(0).expand(len(item_names), -1)
            item_indices = torch.tensor(
                [graph.anchor_map[n] for n in item_names], dtype=torch.long,
            )
            item_h = h[item_indices]
            scores = self.recommender_head(u_h, item_h).cpu().numpy()

        ranked = sorted(
            zip(item_names, scores.tolist()),
            key=lambda x: x[1], reverse=True,
        )[:k]
        out = [{"item": name, "score": float(score),
                "item_type": self.schema.types.get(name, "unknown")}
               for name, score in ranked]

        if verbose:
            print(f"  top {len(out)} items for user={user!r}:")
            for r in out:
                print(f"    {r['item']:20s}  ({r['item_type']})  "
                      f"score={r['score']:+.4f}")

        return out

    def explain_recommendation(self, user: str, item: str,
                               graph: HyperGraph | None = None,
                               top_k: int = 5,
                               steps: int = 16,
                               max_infons: int = 500,
                               verbose: bool = False) -> list[dict]:
        """Which infons most explain why `item` was recommended to `user`?

        Integrated gradients on the recommender score w.r.t. input node
        features; returns the infons whose presence most increases or
        decreases the user→item score.
        """
        if graph is None:
            graph = self.builder.build(max_infons=max_infons,
                                       feature_dim=self.hidden_dim)
        u_idx = graph.anchor_map.get(user)
        i_idx = graph.anchor_map.get(item)
        if u_idx is None or i_idx is None:
            return []

        baseline = torch.zeros_like(graph.node_features)
        actual = graph.node_features
        total_grad = torch.zeros_like(actual)

        self.train()
        for step in range(1, steps + 1):
            alpha = step / steps
            interp = (baseline + alpha * (actual - baseline)).clone().detach()
            interp.requires_grad_(True)
            g_interp = HyperGraph(
                node_ids=graph.node_ids,
                node_types=graph.node_types,
                node_features=interp,
                edge_index=graph.edge_index,
                edge_types=graph.edge_types,
                edge_weights=graph.edge_weights,
                anchor_type_groups=graph.anchor_type_groups,
                infon_indices=graph.infon_indices,
                infon_map=graph.infon_map,
                anchor_map=graph.anchor_map,
                situation_features=graph.situation_features,
            )
            h = self.forward(g_interp)
            score = self.recommender_head(
                h[u_idx].unsqueeze(0),
                h[i_idx].unsqueeze(0),
            ).squeeze()
            grad = torch.autograd.grad(score, interp, retain_graph=False)[0]
            total_grad += grad

        self.eval()
        attr_per_node = ((actual - baseline) * (total_grad / steps)).sum(dim=-1)
        attr_np = attr_per_node.detach().cpu().numpy()

        # Rank INFON nodes (not anchors) by attribution magnitude
        results = []
        for iid, idx in graph.infon_map.items():
            inf = self.store.get_infon(iid)
            results.append({
                "infon_id": iid,
                "score": float(attr_np[idx]),
                "subject": inf.subject if inf else None,
                "predicate": inf.predicate if inf else None,
                "object": inf.object if inf else None,
                "sentence": inf.sentence if inf else None,
            })
        results.sort(key=lambda x: abs(x["score"]), reverse=True)
        top = results[:top_k]

        if verbose:
            print(f"  explaining recommend({user!r} → {item!r}):")
            for r in top:
                label = f"{r['subject']}/{r['predicate']}/{r['object']}"
                print(f"    {label:40s}  IG={r['score']:+.4f}")

        return top

    # ── SELF-DISCOVERY: UNIFIED PIPELINE ──────────────────────────────

    def self_discover(self,
                      corpus: list[dict],
                      interaction_pair: tuple = ("adopts", "avoids"),
                      schema_rounds: int = 3,
                      verbose: bool = False) -> dict:
        """Run all five self-discovery stages in sequence.

        1. Schema auto-expansion (anchors grow from seed).
        2. Role-typing head (predict missing role type).
        3. Learned DS source weights (trust per-source).
        4. Interaction family discovery (positive/negative relations).
        5. Edge-type discovery (residual-coupling clusters).

        Each stage runs the existing specialized routine. Stages 2-5
        reuse the post-expansion graph so they benefit from any
        anchors added in stage 1.

        Returns a dict summarizing what every stage produced.
        """
        out: dict = {}

        if verbose:
            print("  [1/5] schema auto-expansion")
        schema_result = self.self_discover_schema(
            corpus=corpus, max_rounds=schema_rounds,
            min_cluster_size=2, min_npmi=-1.0,
            per_round_fit_epochs=15, n_propose=6,
            verbose=verbose,
        )
        out["schema_expansion"] = schema_result

        # Rebuild graph after expansion so downstream stages see the
        # extended schema's anchors.
        graph = self.builder.build(feature_dim=self.hidden_dim)
        # Rebuild the role head to reflect any new types.
        role_types = sorted(set(self.schema.types.values()))
        if role_types != self._role_types:
            self._role_types = role_types
            self.role_head = RoleTypeHead(self.hidden_dim, len(role_types))
            self._role_head_trained = False

        if verbose:
            print("  [2/5] role-type head")
        role_result = self.train_role_type_head(
            graph=graph, epochs=40, lr=1e-2, verbose=False,
        )
        out["role_typing"] = role_result

        if verbose:
            print("  [3/5] learned DS source weights")
        weights_result = self.train_source_weights(
            graph=graph, epochs=40, lr=5e-2, verbose=False,
        )
        out["source_weights"] = weights_result

        if verbose:
            print("  [4/5] interaction family discovery")
        pos_seed, neg_seed = interaction_pair
        if (pos_seed in self.schema.names
                and neg_seed in self.schema.names):
            family_result = self.discover_interaction_family(
                positive_seed=pos_seed, negative_seed=neg_seed,
                graph=graph, k_positive=5, k_negative=5, verbose=False,
            )
        else:
            family_result = {"positive_family": [], "negative_family": [],
                             "skipped_reason":
                                 f"seeds {interaction_pair} not in schema"}
        out["interaction_family"] = family_result

        if verbose:
            print("  [5/5] edge-type discovery")
        edge_result = self.discover_edge_types(
            graph=graph, k=3, top_pair_fraction=0.1, verbose=False,
        )
        out["edge_types"] = edge_result

        if verbose:
            print("\n  self_discover complete.")
            print(f"    final schema size: "
                  f"{schema_result['final_schema_size']}")
            print(f"    role-typing accuracy: "
                  f"{role_result.get('train_accuracy', float('nan')):.3f}")
            print(f"    top source weight: "
                  f"{max(weights_result.get('weights', {}).values(), default=0.0):.3f}")
            print(f"    edge-type clusters: "
                  f"{len(edge_result.get('clusters', []))}")

        return out

    # ── SELF-DISCOVERY: EDGE-TYPE DISCOVERY ───────────────────────────

    def discover_edge_types(self,
                            graph: "HyperGraph | None" = None,
                            k: int = 3,
                            top_pair_fraction: float = 0.05,
                            max_infons: int = 500,
                            verbose: bool = False) -> dict:
        """Propose candidate new edge types from residual couplings.

        Identify infon pairs whose post-message-passing cosine
        similarity is high but not explained by any existing edge in
        the graph connecting them. Spectral-cluster those residual
        pairs into k candidate new edge types. Each cluster is
        characterized by a few representative pairs.
        """
        if graph is None:
            graph = self.builder.build(max_infons=max_infons,
                                       feature_dim=self.hidden_dim)
        if not self._fitted:
            self.fit(graph=graph, verbose=verbose)

        infon_ids = list(graph.infon_map.keys())
        if len(infon_ids) < 4:
            return {"clusters": [], "n_candidate_pairs": 0}

        infon_indices = [graph.infon_map[i] for i in infon_ids]
        idx_tensor = torch.tensor(infon_indices, dtype=torch.long)

        with torch.no_grad():
            h = self.forward(graph)
            infon_h = F.normalize(h[idx_tensor], dim=-1)
            sim = (infon_h @ infon_h.T).cpu().numpy()

        # Build an "explained" mask: pairs already connected by ANY
        # existing infon-to-infon edge are explained.
        n = len(infon_ids)
        explained = np.zeros((n, n), dtype=bool)
        id_to_local = {iid: i for i, iid in enumerate(infon_ids)}
        for edge in self.store.get_edges(limit=max_infons * 10):
            if edge.source in id_to_local and edge.target in id_to_local:
                i, j = id_to_local[edge.source], id_to_local[edge.target]
                explained[i, j] = True
                explained[j, i] = True

        # Mask diagonal + explained pairs; rank the rest
        np.fill_diagonal(sim, -np.inf)
        sim[explained] = -np.inf

        # Top fraction of remaining pairs
        flat = sim[np.triu_indices(n, k=1)]
        finite_mask = np.isfinite(flat)
        if not finite_mask.any():
            return {"clusters": [], "n_candidate_pairs": 0}
        threshold = np.quantile(flat[finite_mask], 1 - top_pair_fraction)

        candidate_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if sim[i, j] >= threshold and np.isfinite(sim[i, j]):
                    candidate_pairs.append((i, j, float(sim[i, j])))

        if len(candidate_pairs) < k:
            # Not enough pairs to cluster; return singletons
            clusters = [
                {"pairs": [(infon_ids[i], infon_ids[j])],
                 "mean_similarity": s,
                 "size": 1}
                for i, j, s in candidate_pairs
            ]
            return {"clusters": clusters,
                    "n_candidate_pairs": len(candidate_pairs)}

        # Feature vector per candidate pair: concat(h_i - h_j, h_i + h_j)
        feats = []
        for i, j, _ in candidate_pairs:
            vi = infon_h[i].cpu().numpy()
            vj = infon_h[j].cpu().numpy()
            feats.append(np.concatenate([vi - vj, vi + vj]))
        feat_arr = np.stack(feats).astype(np.float32)

        # Cluster with our existing spectral / k-means utility
        labels = self._kmeans(feat_arr, k=k)

        clusters = []
        for cid in range(k):
            member_indices = [i for i, lab in enumerate(labels) if int(lab) == cid]
            if not member_indices:
                continue
            members = [(infon_ids[candidate_pairs[mi][0]],
                        infon_ids[candidate_pairs[mi][1]],
                        candidate_pairs[mi][2])
                       for mi in member_indices]
            # Representative examples: top-3 by similarity
            members.sort(key=lambda x: x[2], reverse=True)
            reps = []
            for src_id, tgt_id, s in members[:3]:
                src_inf = self.store.get_infon(src_id)
                tgt_inf = self.store.get_infon(tgt_id)
                reps.append({
                    "source": (src_inf.subject, src_inf.predicate,
                               src_inf.object) if src_inf else None,
                    "target": (tgt_inf.subject, tgt_inf.predicate,
                               tgt_inf.object) if tgt_inf else None,
                    "similarity": s,
                })
            clusters.append({
                "cluster_id": cid,
                "size": len(members),
                "mean_similarity": float(np.mean([m[2] for m in members])),
                "representative_pairs": reps,
            })

        if verbose:
            print(f"  candidate pairs: {len(candidate_pairs)}")
            print(f"  clustered into {len(clusters)} proposed edge types")
            for c in clusters:
                print(f"    cluster {c['cluster_id']}: size={c['size']}, "
                      f"mean_sim={c['mean_similarity']:.3f}")
                for rep in c["representative_pairs"]:
                    src = rep["source"] or ("?",) * 3
                    tgt = rep["target"] or ("?",) * 3
                    print(f"      {src[0]}/{src[1]}/{src[2]}  ~~>  "
                          f"{tgt[0]}/{tgt[1]}/{tgt[2]}  "
                          f"(sim={rep['similarity']:.3f})")

        return {"clusters": clusters,
                "n_candidate_pairs": len(candidate_pairs)}

    # ── SELF-DISCOVERY: INTERACTION FAMILY ────────────────────────────

    def discover_interaction_family(self,
                                    positive_seed: str,
                                    negative_seed: str,
                                    graph: "HyperGraph | None" = None,
                                    k_positive: int = 5,
                                    k_negative: int = 5,
                                    max_infons: int = 500,
                                    verbose: bool = False) -> dict:
        """Given one positive and one negative seed relation, find
        nearby relation anchors in the trained GNN's embedding space.

        Returns two ranked lists (excluding the seeds themselves): the
        top-k relations closest to the positive seed and the top-k
        closest to the negative seed. Also returns their cosine
        similarities to the seeds. Relations that score higher against
        the other seed are excluded from each list.
        """
        if graph is None:
            graph = self.builder.build(max_infons=max_infons,
                                       feature_dim=self.hidden_dim)
        if not self._fitted:
            self.fit(graph=graph, verbose=verbose)

        pos_idx = graph.anchor_map.get(positive_seed)
        neg_idx = graph.anchor_map.get(negative_seed)
        if pos_idx is None or neg_idx is None:
            return {"positive_family": [], "negative_family": [],
                    "error": "seed not in schema"}

        # Candidate relation anchors
        relation_names = [n for n in graph.anchor_map
                          if self.schema.types.get(n) == "relation"
                          and n not in (positive_seed, negative_seed)]
        if not relation_names:
            return {"positive_family": [], "negative_family": []}

        with torch.no_grad():
            h = self.forward(graph)
            pos_h = F.normalize(h[pos_idx].unsqueeze(0), dim=-1)
            neg_h = F.normalize(h[neg_idx].unsqueeze(0), dim=-1)
            cand_idx = torch.tensor(
                [graph.anchor_map[n] for n in relation_names],
                dtype=torch.long,
            )
            cand_h = F.normalize(h[cand_idx], dim=-1)
            sim_pos = (cand_h @ pos_h.T).squeeze(-1).cpu().numpy()
            sim_neg = (cand_h @ neg_h.T).squeeze(-1).cpu().numpy()

        pos_list, neg_list = [], []
        for name, sp, sn in zip(relation_names, sim_pos, sim_neg):
            if sp > sn:
                pos_list.append((name, float(sp), float(sn)))
            else:
                neg_list.append((name, float(sp), float(sn)))

        pos_list.sort(key=lambda x: x[1], reverse=True)
        neg_list.sort(key=lambda x: x[2], reverse=True)

        pos_top = [{"anchor": n, "sim_to_positive": sp,
                    "sim_to_negative": sn}
                   for n, sp, sn in pos_list[:k_positive]]
        neg_top = [{"anchor": n, "sim_to_positive": sp,
                    "sim_to_negative": sn}
                   for n, sp, sn in neg_list[:k_negative]]

        if verbose:
            print(f"  positive family (closer to {positive_seed!r}):")
            for r in pos_top:
                print(f"    {r['anchor']:20s}  +sim={r['sim_to_positive']:+.3f}  "
                      f"-sim={r['sim_to_negative']:+.3f}")
            print(f"  negative family (closer to {negative_seed!r}):")
            for r in neg_top:
                print(f"    {r['anchor']:20s}  +sim={r['sim_to_positive']:+.3f}  "
                      f"-sim={r['sim_to_negative']:+.3f}")

        return {
            "positive_family": pos_top,
            "negative_family": neg_top,
            "positive_seed": positive_seed,
            "negative_seed": negative_seed,
            "n_candidates": len(relation_names),
        }

    # ── SELF-DISCOVERY: LEARNED DS SOURCE WEIGHTS ─────────────────────

    def train_source_weights(self,
                             graph: "HyperGraph | None" = None,
                             epochs: int = 40,
                             lr: float = 5e-2,
                             max_infons: int = 500,
                             verbose: bool = False) -> dict:
        """Learn per-source trust weights against the GNN's self-readout.

        Each of the four DS sources produces a per-infon mass vector
        (4-d: S, R, U, θ). The GNN's post-message-passing readout
        produces another 4-d mass per infon. We treat the GNN's
        readout as a pseudo-ground-truth and fit the source weights to
        a convex combination of the four source masses that minimizes
        KL to the GNN's readout.

        This is self-supervised in the strict sense: no human labels;
        the GNN teaches the weights, the weights will later inform
        the GNN's teacher during the next round.
        """
        from .dempster_shafer import (
            mass_from_polarity, mass_from_triple_alignment,
            mass_from_anchor_distance, mass_from_confidence,
            mass_from_evidentiality, mass_from_modality,
        )

        if graph is None:
            graph = self.builder.build(max_infons=max_infons,
                                       feature_dim=self.hidden_dim)
        if not self._fitted:
            self.fit(graph=graph, verbose=verbose)

        infon_ids = list(graph.infon_map.keys())
        targets = []
        source_tensors = []
        for iid in infon_ids:
            infon = self.store.get_infon(iid)
            if infon is None:
                continue
            claim_anchors = {}
            for role in [infon.subject, infon.predicate, infon.object]:
                claim_anchors[role] = infon.confidence
            sources = [
                mass_from_polarity(infon),
                mass_from_triple_alignment(claim_anchors, infon,
                                           self.schema.types),
                mass_from_anchor_distance(claim_anchors, infon,
                                          self.schema.types),
                mass_from_confidence(infon),
                mass_from_evidentiality(infon),
                mass_from_modality(infon),
            ]
            # Stack sources into a (6, 4) tensor: rows = sources, cols = {S,R,U,θ}
            src_mat = torch.tensor([
                [s.supports, s.refutes, s.uncertain, s.theta]
                for s in sources
            ], dtype=torch.float32)
            source_tensors.append(src_mat)
            targets.append(graph.infon_map[iid])

        if not source_tensors:
            return {"epochs": 0, "final_loss": float("nan"),
                    "n_examples": 0, "weights": None, "losses": []}

        S = torch.stack(source_tensors)  # (n_infons, 6, 4)
        target_idx = torch.tensor(targets, dtype=torch.long)

        # Freeze all but the learned-weights module
        for p in self.parameters():
            p.requires_grad = False
        for p in self.ds_source_weights.parameters():
            p.requires_grad = True

        optimizer = torch.optim.Adam(self.ds_source_weights.parameters(),
                                     lr=lr)
        losses = []
        self.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            with torch.no_grad():
                h = self.forward(graph)
                gnn_mass = self.mass_readout(h[target_idx])  # (n, 4)
            # Weighted combination of the four sources
            w = self.ds_source_weights()  # (4,)
            # (n_infons, 4, 4) × (4,) -> (n_infons, 4)
            combined = (S * w.view(1, -1, 1)).sum(dim=1)
            # Floor to avoid log(0); then renormalize so rows sum to 1.
            combined = combined.clamp(min=1e-8)
            combined = combined / combined.sum(dim=-1, keepdim=True)
            # KL(combined || gnn_mass). Target must also be floored so
            # 0 * log(0) doesn't propagate NaN through reverse KL entries.
            gnn_safe = gnn_mass.clamp(min=1e-8)
            gnn_safe = gnn_safe / gnn_safe.sum(dim=-1, keepdim=True)
            loss = F.kl_div(
                combined.log(),
                gnn_safe,
                reduction="batchmean",
                log_target=False,
            )
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if verbose and (epoch + 1) % 10 == 0:
                w_vals = self.ds_source_weights().detach().cpu().numpy()
                print(f"    ds-weights epoch {epoch+1}/{epochs}: "
                      f"loss={loss.item():.4f}, w={[round(x, 3) for x in w_vals]}")

        for p in self.parameters():
            p.requires_grad = True
        self.eval()
        self._ds_weights_trained = True

        final_weights = self.ds_source_weights().detach().cpu().numpy().tolist()
        source_names = ["polarity", "triple_alignment",
                        "anchor_distance", "confidence",
                        "evidentiality", "modality"]

        return {
            "epochs": epochs,
            "final_loss": losses[-1],
            "first_loss": losses[0],
            "n_examples": len(source_tensors),
            "weights": dict(zip(source_names, final_weights)),
            "losses": losses,
        }

    # ── SELF-DISCOVERY: ROLE TYPING HEAD ──────────────────────────────

    def train_role_type_head(self,
                             graph: "HyperGraph | None" = None,
                             epochs: int = 50,
                             lr: float = 1e-2,
                             max_infons: int = 500,
                             verbose: bool = False) -> dict:
        """Self-supervised: mask one role in a triple, predict its type.

        For every extracted infon we have (subject_anchor, predicate_anchor,
        object_anchor). Three training examples per infon, one per
        masked role. The head sees the two unmasked anchors' embeddings
        plus a one-hot indicating which role is missing, and must
        predict the masked role's anchor-type.
        """
        if self.role_head is None or not self._role_types:
            if verbose:
                print("  no role types in schema; cannot train role head")
            return {"epochs": 0, "final_loss": float("nan"),
                    "n_examples": 0, "losses": []}

        if graph is None:
            graph = self.builder.build(max_infons=max_infons,
                                       feature_dim=self.hidden_dim)
        if not self._fitted:
            self.fit(graph=graph, verbose=verbose)

        type_to_idx = {t: i for i, t in enumerate(self._role_types)}
        roles = ("subject", "predicate", "object")

        examples = []  # (h_a_idx, h_b_idx, mask_onehot, target_type_idx)
        for iid in graph.infon_map:
            inf = self.store.get_infon(iid)
            if inf is None:
                continue
            anchors = {"subject": inf.subject, "predicate": inf.predicate,
                       "object": inf.object}
            types = {r: self.schema.types.get(a) for r, a in anchors.items()}
            anchor_idx = {r: graph.anchor_map.get(a)
                          for r, a in anchors.items()}
            if not all(anchor_idx.values()):
                continue
            if not all(types.values()):
                continue

            for masked_i, masked in enumerate(roles):
                other_roles = [r for r in roles if r != masked]
                a_idx = anchor_idx[other_roles[0]]
                b_idx = anchor_idx[other_roles[1]]
                target_t = types[masked]
                if target_t not in type_to_idx:
                    continue
                mask_oh = [0.0, 0.0, 0.0]
                mask_oh[masked_i] = 1.0
                examples.append((a_idx, b_idx, mask_oh,
                                 type_to_idx[target_t]))

        if not examples:
            if verbose:
                print("  no role-typing examples extracted")
            return {"epochs": 0, "final_loss": float("nan"),
                    "n_examples": 0, "losses": []}

        a_idx_t = torch.tensor([e[0] for e in examples], dtype=torch.long)
        b_idx_t = torch.tensor([e[1] for e in examples], dtype=torch.long)
        mask_t = torch.tensor([e[2] for e in examples], dtype=torch.float32)
        targets = torch.tensor([e[3] for e in examples], dtype=torch.long)

        # Freeze everything except the role head
        for p in self.parameters():
            p.requires_grad = False
        for p in self.role_head.parameters():
            p.requires_grad = True

        optimizer = torch.optim.Adam(self.role_head.parameters(), lr=lr)
        losses = []
        self.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            h = self.forward(graph)
            h_a = h[a_idx_t]
            h_b = h[b_idx_t]
            logits = self.role_head(h_a, h_b, mask_t)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if verbose and (epoch + 1) % 10 == 0:
                print(f"    role-head epoch {epoch+1}/{epochs}: "
                      f"loss={loss.item():.4f}")

        for p in self.parameters():
            p.requires_grad = True
        self.eval()
        self._role_head_trained = True

        # Training accuracy
        with torch.no_grad():
            h = self.forward(graph)
            logits = self.role_head(h[a_idx_t], h[b_idx_t], mask_t)
            preds = logits.argmax(dim=-1)
            train_acc = (preds == targets).float().mean().item()

        return {
            "epochs": epochs,
            "final_loss": losses[-1],
            "first_loss": losses[0],
            "n_examples": len(examples),
            "train_accuracy": float(train_acc),
            "n_types": len(self._role_types),
            "role_types": list(self._role_types),
            "losses": losses,
        }

    def predict_role_type(self, anchor_a: str, anchor_b: str,
                          masked_role: str,
                          graph: "HyperGraph | None" = None,
                          max_infons: int = 500) -> dict:
        """Predict the type of a missing role given the two known anchors.

        Args:
            anchor_a, anchor_b: schema anchor names for the two known roles
            masked_role: 'subject' | 'predicate' | 'object'
        """
        if self.role_head is None or not self._role_head_trained:
            return {"type": None, "probs": {}}

        if graph is None:
            graph = self.builder.build(max_infons=max_infons,
                                       feature_dim=self.hidden_dim)

        a_idx = graph.anchor_map.get(anchor_a)
        b_idx = graph.anchor_map.get(anchor_b)
        if a_idx is None or b_idx is None:
            return {"type": None, "probs": {}}

        role_idx = {"subject": 0, "predicate": 1, "object": 2}[masked_role]
        mask_oh = torch.zeros(3)
        mask_oh[role_idx] = 1.0

        with torch.no_grad():
            h = self.forward(graph)
            logits = self.role_head(
                h[a_idx].unsqueeze(0),
                h[b_idx].unsqueeze(0),
                mask_oh.unsqueeze(0),
            )
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        top_idx = int(probs.argmax())
        return {
            "type": self._role_types[top_idx],
            "probs": {t: float(probs[i])
                      for i, t in enumerate(self._role_types)},
        }

    # ── SELF-DISCOVERY: SCHEMA AUTO-EXPANSION ─────────────────────────

    def self_discover_schema(self,
                             corpus: list[dict],
                             max_rounds: int = 5,
                             min_cluster_size: int = 2,
                             min_npmi: float = 0.0,
                             stop_silhouette_delta: float = 0.02,
                             per_round_fit_epochs: int = 20,
                             n_propose: int = 8,
                             verbose: bool = False) -> dict:
        """Iterative schema expansion from a seed.

        Each round:
          1. Ingest `corpus` against the current schema (via self.store).
          2. Rebuild the graph and re-fit the GNN.
          3. Call discover_anchors to propose candidate new anchors.
          4. Accept a candidate if its size >= min_cluster_size and its
             internal coherence >= min_npmi.
          5. Call encoder._projector.refresh() so the next round's
             extraction can use the new anchors.

        Stops when the silhouette score changes by less than
        stop_silhouette_delta OR when no new anchors are accepted in a
        round OR after max_rounds.

        Returns a dict summarizing what was discovered.
        """
        # We only ingest once — the corpus is fixed. Subsequent rounds
        # re-run extraction against the *updated* schema by re-ingesting.
        history = []
        prev_silhouette = -2.0  # impossible value

        for round_idx in range(max_rounds):
            # (1) Re-ingest against the latest schema. We clear the store
            # so extracted infons reflect the current anchor set rather
            # than accumulating across rounds.
            if corpus:
                try:
                    self.store.clear()
                except AttributeError:
                    pass
                if hasattr(self.encoder, "refresh_schema"):
                    self.encoder.refresh_schema(self.schema)
                elif hasattr(self.encoder, "_projector") and \
                        self.encoder._projector is not None:
                    self.encoder._projector.refresh(self.schema)
                from .extract import extract_infons
                from .config import CognitionConfig
                cfg = CognitionConfig()
                infons, edges = extract_infons(
                    corpus, self.encoder, self.schema, cfg,
                )
                if infons:
                    self.store.put_infons(infons)
                if edges:
                    self.store.put_edges(edges)

            # (2) Rebuild graph + refit
            graph = self.builder.build(max_infons=1000,
                                       feature_dim=self.hidden_dim)
            self._fitted = False  # force refit with fresh schema
            self.fit(graph=graph, epochs=per_round_fit_epochs, verbose=False)

            # (3) Propose candidate anchors via the existing Kan step
            schema_ext, discovered, disc_stats = self.discover_anchors(
                graph=graph, n_anchors=n_propose, verbose=False,
            )

            accepted = []
            for da in discovered:
                if da.name in self.schema.names:
                    continue
                if da.size < min_cluster_size:
                    continue
                if da.coherence < min_npmi:
                    continue
                # Derive tokens from the cluster's surface tokens
                tokens = [t for t in da.tokens if len(t) >= 2][:5] or [da.name]
                added = self.schema.add_anchor(
                    da.name, da.inferred_type, tokens=tokens,
                )
                if added:
                    accepted.append({
                        "name": da.name,
                        "type": da.inferred_type,
                        "size": da.size,
                        "coherence": da.coherence,
                        "tokens": tokens,
                    })

            history.append({
                "round": round_idx + 1,
                "schema_size": len(self.schema.names),
                "discovered": len(discovered),
                "accepted": len(accepted),
                "silhouette": disc_stats["silhouette"],
                "accepted_anchors": accepted,
            })

            if verbose:
                print(f"  [round {round_idx+1}] schema size="
                      f"{len(self.schema.names)}, discovered="
                      f"{len(discovered)}, accepted={len(accepted)}, "
                      f"silhouette={disc_stats['silhouette']:.3f}")
                for a in accepted:
                    print(f"    + {a['name']:25s} ({a['type']}, "
                          f"size={a['size']}, coh={a['coherence']:.3f})")

            # (5) Stopping rules
            if not accepted:
                if verbose:
                    print("    stopping: no new anchors accepted this round")
                break
            if abs(disc_stats["silhouette"] - prev_silhouette) < \
                    stop_silhouette_delta and round_idx > 0:
                if verbose:
                    print(f"    stopping: silhouette change "
                          f"{abs(disc_stats['silhouette'] - prev_silhouette):.4f}"
                          f" < {stop_silhouette_delta}")
                break
            prev_silhouette = disc_stats["silhouette"]

        return {
            "final_schema_size": len(self.schema.names),
            "rounds_run": len(history),
            "history": history,
            "final_schema_names": list(self.schema.names),
        }

    # ── TEMPORAL SUCCESSOR HEAD (self-supervised from document order) ─

    def train_temporal_successor_head(self,
                                      graph: HyperGraph | None = None,
                                      epochs: int = 60,
                                      lr: float = 1e-2,
                                      max_infons: int = 500,
                                      use_timestamps: bool = True,
                                      verbose: bool = False) -> dict:
        """Self-supervised training on document-order precedence.

        Supervision comes from doc_id + sent_id: if infon A and infon B
        were extracted from the same document and A's sent_id precedes
        B's, then (A, B) is a positive precedence pair and (B, A) is a
        negative. No tense parsing. No hand-coded rules.

        Optionally, timestamped pairs override document order when
        available (timestamps are authoritative when present).

        GNN frozen; only head parameters move.
        """
        from datetime import datetime

        if graph is None:
            graph = self.builder.build(max_infons=max_infons,
                                       feature_dim=self.hidden_dim)
        if not self._fitted:
            self.fit(graph=graph, verbose=verbose)

        # Group infons by doc_id, preserving sent_id ordering.
        doc_groups: dict[str, list[tuple[str, str, Infon]]] = defaultdict(list)
        for iid in graph.infon_map:
            inf = self.store.get_infon(iid)
            if inf is None or not inf.doc_id:
                continue
            doc_groups[inf.doc_id].append((iid, inf.sent_id or iid, inf))

        # Deterministic sort within each doc
        for doc_id in doc_groups:
            doc_groups[doc_id].sort(key=lambda x: x[1])

        # Collect positive pairs (earlier, later) and their reverses
        pairs_pos: list[tuple[int, int]] = []
        pairs_neg: list[tuple[int, int]] = []
        for doc_id, items in doc_groups.items():
            for a in range(len(items)):
                for b in range(a + 1, len(items)):
                    id_a, _, _ = items[a]
                    id_b, _, _ = items[b]
                    idx_a = graph.infon_map.get(id_a)
                    idx_b = graph.infon_map.get(id_b)
                    if idx_a is None or idx_b is None:
                        continue
                    pairs_pos.append((idx_a, idx_b))
                    pairs_neg.append((idx_b, idx_a))

        # Augment with timestamp-derived pairs when present
        if use_timestamps:
            def parse_ts(s):
                if not s:
                    return None
                try:
                    return datetime.strptime(s.split("T")[0], "%Y-%m-%d")
                except Exception:
                    return None

            timestamped = []
            for iid in graph.infon_map:
                inf = self.store.get_infon(iid)
                if inf is None:
                    continue
                ts = parse_ts(getattr(inf, "timestamp", None))
                if ts is not None:
                    idx = graph.infon_map.get(iid)
                    if idx is not None:
                        timestamped.append((ts, idx, iid))
            timestamped.sort(key=lambda x: x[0])
            for i in range(len(timestamped)):
                for j in range(i + 1, len(timestamped)):
                    t_i, idx_i, _ = timestamped[i]
                    t_j, idx_j, _ = timestamped[j]
                    if t_i >= t_j:
                        continue
                    pairs_pos.append((idx_i, idx_j))
                    pairs_neg.append((idx_j, idx_i))

        if not pairs_pos:
            if verbose:
                print("  no document-order or timestamp pairs available")
            return {"epochs": 0, "final_loss": float("nan"),
                    "n_positive": 0, "losses": []}

        pos_i = torch.tensor([p[0] for p in pairs_pos], dtype=torch.long)
        pos_j = torch.tensor([p[1] for p in pairs_pos], dtype=torch.long)
        neg_i = torch.tensor([p[0] for p in pairs_neg], dtype=torch.long)
        neg_j = torch.tensor([p[1] for p in pairs_neg], dtype=torch.long)

        # Freeze all but the temporal head
        for p in self.parameters():
            p.requires_grad = False
        for p in self.temporal_head.parameters():
            p.requires_grad = True

        optimizer = torch.optim.Adam(self.temporal_head.parameters(), lr=lr)
        losses = []
        self.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            h = self.forward(graph)
            pos_scores = self.temporal_head.logits(h[pos_i], h[pos_j])
            neg_scores = self.temporal_head.logits(h[neg_i], h[neg_j])
            # BCE: positives -> 1, negatives -> 0
            loss = (F.binary_cross_entropy_with_logits(
                        pos_scores, torch.ones_like(pos_scores))
                    + F.binary_cross_entropy_with_logits(
                        neg_scores, torch.zeros_like(neg_scores))) / 2
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if verbose and (epoch + 1) % 15 == 0:
                print(f"    temporal-head epoch {epoch+1}/{epochs}: "
                      f"loss={loss.item():.4f}")

        for p in self.parameters():
            p.requires_grad = True
        self.eval()
        self._temporal_head_trained = True

        # Evaluate train accuracy on the positive / negative pairs
        with torch.no_grad():
            h = self.forward(graph)
            pos_pred = self.temporal_head(h[pos_i], h[pos_j]) > 0.5
            neg_pred = self.temporal_head(h[neg_i], h[neg_j]) > 0.5
            train_acc = ((pos_pred.float().sum()
                          + (~neg_pred).float().sum())
                         / (2 * len(pos_i))).item()

        return {
            "epochs": epochs,
            "final_loss": losses[-1],
            "first_loss": losses[0],
            "n_positive": len(pairs_pos),
            "train_accuracy": float(train_acc),
            "losses": losses,
        }

    def predict_temporal_successors(self, infon_id: str,
                                    graph: HyperGraph | None = None,
                                    k: int = 5,
                                    max_infons: int = 500,
                                    verbose: bool = False) -> list[dict]:
        """Rank candidate successor infons for a given source infon
        using the trained temporal head.
        """
        if graph is None:
            graph = self.builder.build(max_infons=max_infons,
                                       feature_dim=self.hidden_dim)
        if not self._fitted:
            self.fit(graph=graph, verbose=verbose)
        if not self._temporal_head_trained:
            if verbose:
                print("  temporal head not trained; "
                      "call train_temporal_successor_head() first")
            return []

        src_idx = graph.infon_map.get(infon_id)
        if src_idx is None:
            return []

        other_ids = [i for i in graph.infon_map if i != infon_id]
        other_indices = torch.tensor(
            [graph.infon_map[i] for i in other_ids], dtype=torch.long,
        )

        with torch.no_grad():
            h = self.forward(graph)
            src_h = h[src_idx].unsqueeze(0).expand(len(other_ids), -1)
            scores = self.temporal_head(src_h, h[other_indices]).cpu().numpy()

        ranked = sorted(
            zip(other_ids, scores.tolist()),
            key=lambda x: x[1], reverse=True,
        )[:k]
        results = []
        for iid, score in ranked:
            inf = self.store.get_infon(iid)
            results.append({
                "infon_id": iid,
                "score": float(score),
                "subject": inf.subject if inf else None,
                "predicate": inf.predicate if inf else None,
                "object": inf.object if inf else None,
            })

        if verbose:
            src_inf = self.store.get_infon(infon_id)
            if src_inf:
                print(f"  after: {src_inf.subject}/{src_inf.predicate}"
                      f"/{src_inf.object}")
            print(f"  top {len(results)} learned successors:")
            for r in results:
                label = f"{r['subject']}/{r['predicate']}/{r['object']}"
                print(f"    {label:40s}  score={r['score']:.4f}")

        return results

    def refine_temporal_learned(self, graph: HyperGraph | None = None,
                                threshold: float = 0.7,
                                require_shared_anchor: bool = True,
                                max_infons: int = 500,
                                verbose: bool = False) -> list[Edge]:
        """Populate NEXT edges from the trained temporal head.

        For every ordered pair (i, j) sharing at least one anchor (of
        any role, not just subject), evaluate the head. If the score
        exceeds `threshold`, emit a NEXT edge from i to j. Symmetric
        pairs are resolved by keeping only the direction with higher
        score.

        Returns the list of new edges (also written to the store).
        Much less restrictive than the tense-based heuristic: it
        picks up NEXT edges through shared predicate or object, not
        only shared subject.
        """
        if graph is None:
            graph = self.builder.build(max_infons=max_infons,
                                       feature_dim=self.hidden_dim)
        if not self._temporal_head_trained:
            if verbose:
                print("  temporal head not trained; training now "
                      "with default settings")
            self.train_temporal_successor_head(graph=graph, verbose=verbose)

        # Pre-compute each infon's anchor set (subject, predicate, object)
        infon_anchors: dict[str, set[str]] = {}
        infon_cache: dict[str, Infon] = {}
        for iid in graph.infon_map:
            inf = self.store.get_infon(iid)
            if inf is None:
                continue
            infon_cache[iid] = inf
            infon_anchors[iid] = {inf.subject, inf.predicate, inf.object}

        # Candidate pairs: i != j, and either share an anchor or
        # require_shared_anchor is False
        candidates = []
        ids = list(graph.infon_map.keys())
        for a, i_id in enumerate(ids):
            if i_id not in infon_anchors:
                continue
            for j_id in ids[a + 1:]:
                if j_id not in infon_anchors:
                    continue
                if require_shared_anchor and not (
                        infon_anchors[i_id] & infon_anchors[j_id]):
                    continue
                candidates.append((i_id, j_id))

        if not candidates:
            return []

        # Batch-score both directions, take whichever is stronger
        i_idx = torch.tensor(
            [graph.infon_map[p[0]] for p in candidates], dtype=torch.long,
        )
        j_idx = torch.tensor(
            [graph.infon_map[p[1]] for p in candidates], dtype=torch.long,
        )

        with torch.no_grad():
            h = self.forward(graph)
            s_ij = self.temporal_head(h[i_idx], h[j_idx]).cpu().numpy()
            s_ji = self.temporal_head(h[j_idx], h[i_idx]).cpu().numpy()

        new_edges: list[Edge] = []
        for (i_id, j_id), sij, sji in zip(candidates, s_ij, s_ji):
            if sij > sji and sij >= threshold:
                new_edges.append(Edge(
                    source=i_id, target=j_id,
                    edge_type="NEXT", weight=float(sij),
                    metadata={"source": "temporal_head",
                              "score_fwd": float(sij),
                              "score_rev": float(sji)},
                ))
            elif sji > sij and sji >= threshold:
                new_edges.append(Edge(
                    source=j_id, target=i_id,
                    edge_type="NEXT", weight=float(sji),
                    metadata={"source": "temporal_head",
                              "score_fwd": float(sji),
                              "score_rev": float(sij)},
                ))

        if new_edges:
            self.store.put_edges(new_edges)

        if verbose:
            print(f"  temporal-head refinement:")
            print(f"    candidate pairs: {len(candidates)}")
            print(f"    edges added (score > {threshold}): {len(new_edges)}")

        return new_edges

    def evaluate_expression(self, expr: dict,
                            max_infons: int = 500) -> MassFunction:
        """Evaluate an IKL expression and return its DS mass.

        Builds the graph, runs message passing, evaluates the expression
        tree, and reads out the mass.
        """
        graph = self.builder.build(max_infons=max_infons,
                                   feature_dim=self.hidden_dim)
        with torch.no_grad():
            h = self.forward(graph)
            result_h = self.compound_query(expr, graph, h)
            masses = self.mass_readout.to_mass_functions(result_h.unsqueeze(0))
        return masses[0]


# Attach sklearn-style predict() / score() / transform() wrappers
# around the existing reason() / query() methods. The rich return
# types stay available; the new methods just expose a uniform
# contract for use in model-selection helpers (sweep, cross-val).
from .estimator import patch_estimator_api as _patch_estimator_api
_patch_estimator_api(HypergraphReasoner)


# ═══════════════════════════════════════════════════════════════════════
# RESULT TYPE
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ReasoningResult:
    """Output of HypergraphReasoner.reason()."""
    query: str = ""
    verdict: str = "NOT ENOUGH INFO"
    mass: MassFunction = field(default_factory=lambda: MassFunction(theta=1.0))
    per_infon_masses: list[MassFunction] = field(default_factory=list)
    n_nodes: int = 0
    n_edges: int = 0
    n_relevant: int = 0


@dataclass
class RefinementResult:
    """Output of HypergraphReasoner.refine()."""
    infons_updated: int = 0
    temporal_added: int = 0
    causal_added: int = 0
    contradictions_found: int = 0
    pairs_checked: int = 0
    temporal_edges: list[Edge] = field(default_factory=list)
    causal_edges: list[Edge] = field(default_factory=list)
    contradiction_edges: list[Edge] = field(default_factory=list)


@dataclass
class CausalView:
    """A DAG view over infon-to-infon causal / temporal edges.

    Built from CAUSES + NEXT edges (by default) after breaking cycles
    via a weight-greedy construction. Exposes ancestor / descendant /
    path queries in infon-id space.
    """
    adj: dict[str, list[tuple[str, float, str]]]
    reverse_adj: dict[str, list[tuple[str, float, str]]]
    edges: list[tuple[str, str, float, str]]
    infon_ids: list[str]

    def ancestors(self, infon_id: str) -> set[str]:
        """All causal ancestors of an infon (transitive closure on reverse_adj)."""
        seen: set[str] = set()
        stack = [infon_id]
        while stack:
            node = stack.pop()
            for parent, _, _ in self.reverse_adj.get(node, []):
                if parent not in seen:
                    seen.add(parent)
                    stack.append(parent)
        return seen

    def descendants(self, infon_id: str) -> set[str]:
        """All descendants reachable via forward edges."""
        seen: set[str] = set()
        stack = [infon_id]
        while stack:
            node = stack.pop()
            for child, _, _ in self.adj.get(node, []):
                if child not in seen:
                    seen.add(child)
                    stack.append(child)
        return seen

    def ancestors_with_distance(self, infon_id: str) -> dict[str, int]:
        """BFS on reverse edges; returns ancestor → shortest-path length."""
        distances: dict[str, int] = {}
        frontier = [(infon_id, 0)]
        visited = {infon_id}
        while frontier:
            node, dist = frontier.pop(0)
            for parent, _, _ in self.reverse_adj.get(node, []):
                if parent not in visited:
                    visited.add(parent)
                    distances[parent] = dist + 1
                    frontier.append((parent, dist + 1))
        return distances

    def paths(self, source: str, target: str,
              max_depth: int = 6) -> list[list[str]]:
        """Enumerate all simple paths source → target up to max_depth."""
        paths = []
        def dfs(node, visited, path):
            if node == target:
                paths.append(list(path))
                return
            if len(path) >= max_depth:
                return
            for nb, _, _ in self.adj.get(node, []):
                if nb not in visited:
                    visited.add(nb)
                    path.append(nb)
                    dfs(nb, visited, path)
                    path.pop()
                    visited.remove(nb)
        dfs(source, {source}, [source])
        return paths

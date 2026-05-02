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
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from collections import defaultdict, Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .infon import Infon, Edge
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
    """Build a HyperGraph from the store + encoder."""

    def __init__(self, store, encoder, schema):
        self.store = store
        self.encoder = encoder
        self.schema = schema

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

        # Infon features: encode sentences, reduce
        infon_sentences = []
        infon_node_indices = []
        for inf in infons:
            if inf.sentence and inf.infon_id in infon_map:
                infon_sentences.append(inf.sentence)
                infon_node_indices.append(infon_map[inf.infon_id])
        if infon_sentences:
            sparse = self.encoder.encode_sparse(infon_sentences)
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
                 situation_dim: int = 16):
        super().__init__()
        self.store = store
        self.encoder = encoder
        self.schema = schema
        self.hidden_dim = hidden_dim
        self.builder = HypergraphBuilder(store, encoder, schema)
        self._fitted = False

        # Message passing layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            in_d = hidden_dim
            self.layers.append(
                TypedMessagePassingLayer(in_d, hidden_dim,
                                        situation_dim=situation_dim)
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

        # Find relevant infon nodes via query anchor overlap
        relevant_indices = []
        relevant_weights = []
        for inf_id, node_idx in graph.infon_map.items():
            score = 0.0
            infon = self.store.get_infon(inf_id)
            if infon is None:
                continue
            for role in [infon.subject, infon.predicate, infon.object]:
                score = max(score, query_activations.get(role, 0.0))
            if score > 0.05:
                relevant_indices.append(node_idx)
                relevant_weights.append(score)

        if not relevant_indices:
            return ReasoningResult(
                query=query, verdict="NOT ENOUGH INFO",
                mass=MassFunction(theta=1.0),
                n_nodes=graph.n_nodes, n_edges=graph.n_edges,
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

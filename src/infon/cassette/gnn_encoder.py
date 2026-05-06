"""Sheaf-structured hypergraph encoder.

Trained offline on synthgen.generate() output. Shipped frozen alongside
the cassette store (or per-store as _model/gnn.pt) and called at query
time as a prior into the symbolic reasoner.

Design (see chat log for derivation):

  • Role-typed stalks. Each node carries a vector per (role, polarity).
    A node-as-subject and the same node-as-object have different stalks —
    parameters are shared but projections preserve role information.
    This is the sheaf inductive bias: "the meaning of toyota depends on
    which role it occupies in this proposition."

  • Per-relation-kind restriction maps. Three kinds: connective,
    terminal, reportive. Messages through connective edges propagate;
    messages through reportive edges dampen. The map IS the learned
    edge semantics.

  • H¹ discrepancy as a feature. After each message pass we compute the
    disagreement between a node's source-stalk and the restriction of
    its neighbor's target-stalk. High discrepancy = structural
    contradiction. Feeds the verdict head as an anomaly signal.

  • Chain-verdict head consumes the path sequence. Not just endpoints —
    the order and polarity of each edge matters for retraction
    detection. Concretely: we pool over (src_stalk, tgt_stalk,
    edge_kind, polarity) tuples along the chain, plus the global
    discrepancy, and map to 3 verdict classes.

Not scope:
  • Text encoding. We're schema-indexed, not sentence-indexed. The GNN
    reads triples, not prose. SPLADE handles text elsewhere.
  • PyG dependency. We write the message passing ourselves because the
    per-relation-kind routing is core to the design and shouldn't hide
    behind a generic GNNConv. Also keeps deploys small.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════
# VOCABULARY  — small fixed feature set, schema-independent
# ═══════════════════════════════════════════════════════════════════════
#
# Inputs to the GNN are structural, not semantic. We don't embed the
# string "toyota" — we embed the node's (role, polarity, confidence,
# recency) tuple. This is what makes the trained model transferable
# across corpora with different anchor vocabularies: it learns
# graph-structural rules, not lexical ones.

RELATION_KINDS = ["connective", "terminal", "reportive"]
KIND_TO_IDX = {k: i for i, k in enumerate(RELATION_KINDS)}

ROLES = ["subject", "object"]
ROLE_TO_IDX = {r: i for i, r in enumerate(ROLES)}


@dataclass
class HypergraphBatch:
    """A batch of chains ready for the GNN.

    A chain is a sequence of edges, each edge is a typed relation
    between two nodes. For training we encode the chain explicitly
    (not the whole graph) because the task is chain-verdict — we know
    which edges matter.

    Shapes:
      edges:        (B, max_len, feature_dim)  — zero-padded, right-aligned
      edge_mask:    (B, max_len)               — 1 where an edge exists, 0 pad
      chain_verdict (int):  0=SUPPORTS, 1=REFUTES, 2=NOT_ENOUGH_INFO
    """
    edges: torch.Tensor         # (B, L, F)
    edge_mask: torch.Tensor     # (B, L)
    verdicts: torch.Tensor      # (B,) int


# Edge feature dimensions, fixed. The anchor-ID features are CRITICAL
# for detecting disconnected chains — without them the GNN can't tell
# "path leads to target" from "path leads somewhere else".
#   [kind_one_hot(3), polarity(1), confidence(1), log_gap(1), is_last(1),
#    touches_source(1), touches_target(1), connects_prev(1)] = 10
EDGE_FEATURE_DIM = 10


def encode_edge(*, kind: str, polarity: int, confidence: float,
                gap_days: int = 0, is_last: bool = False,
                touches_source: bool = False,
                touches_target: bool = False,
                connects_prev: bool = True) -> list[float]:
    """One edge → a feature vector.

    touches_source / touches_target: flags that this edge has source or
    target as one of its endpoints. Disconnected chains will never set
    touches_target on any edge — that's the signal.

    connects_prev: True when this edge's subject == previous edge's
    object. False marks a break in the chain (the graph has the edges
    but they don't form a continuous path)."""
    vec = [0.0] * len(RELATION_KINDS)
    vec[KIND_TO_IDX.get(kind, 0)] = 1.0
    vec.append(float(polarity))
    vec.append(float(confidence))
    vec.append(math.log1p(max(0, gap_days)))
    vec.append(1.0 if is_last else 0.0)
    vec.append(1.0 if touches_source else 0.0)
    vec.append(1.0 if touches_target else 0.0)
    vec.append(1.0 if connects_prev else 0.0)
    return vec


def encode_chain_from_synth(synth_graph, relation_kinds: dict[str, str]) -> list[list[float]]:
    """Build an edge feature list for a SynthGraph.

    Walk the path from source to target in timestamp order, encoding each
    edge. Returns list[list[float]] — an outer list of edges, each a
    feature vector. The caller pads to max_len.
    """
    from .synthgen import SynthGraph  # lazy import to avoid circular
    assert isinstance(synth_graph, SynthGraph)

    # Sort the synth graph's edges by time, keep only those on a path
    # reachable from source. For training we use a permissive definition:
    # "any polarity-1 chain + the retraction twins on matching triples."
    # The GNN should learn to weight them; we don't prune.
    source = synth_graph.chain_source
    target = synth_graph.chain_target
    sorted_edges = sorted(synth_graph.edges, key=lambda x: x.t)

    rows = []
    prev_obj = None
    for e in sorted_edges:
        kind = relation_kinds.get(e.predicate, "connective")
        touches_source = (e.subject == source or e.object == source)
        touches_target = (e.subject == target or e.object == target)
        connects_prev = (prev_obj is None) or (e.subject == prev_obj)
        rows.append({
            "kind": kind, "polarity": e.polarity,
            "confidence": e.confidence, "gap_days": 0,
            "is_last": False,
            "touches_source": touches_source,
            "touches_target": touches_target,
            "connects_prev": connects_prev,
        })
        prev_obj = e.object
    if rows:
        rows[-1]["is_last"] = True
    return [encode_edge(**r) for r in rows]


VERDICT_TO_IDX = {"SUPPORTS": 0, "REFUTES": 1, "NOT_ENOUGH_INFO": 2}
IDX_TO_VERDICT = {v: k for k, v in VERDICT_TO_IDX.items()}


def batch_from_synth(graphs, relation_kinds: dict[str, str],
                     max_len: int = 12) -> HypergraphBatch:
    """Pad + stack SynthGraphs into a tensor batch."""
    batch_edges = []
    batch_mask = []
    batch_verdicts = []
    for g in graphs:
        rows = encode_chain_from_synth(g, relation_kinds)
        # Truncate or pad to max_len.
        rows = rows[-max_len:]  # keep the most recent max_len edges
        mask = [1.0] * len(rows)
        while len(rows) < max_len:
            rows.insert(0, [0.0] * EDGE_FEATURE_DIM)
            mask.insert(0, 0.0)
        batch_edges.append(rows)
        batch_mask.append(mask)
        batch_verdicts.append(VERDICT_TO_IDX[g.chain_verdict])

    return HypergraphBatch(
        edges=torch.tensor(batch_edges, dtype=torch.float32),
        edge_mask=torch.tensor(batch_mask, dtype=torch.float32),
        verdicts=torch.tensor(batch_verdicts, dtype=torch.long),
    )


# ═══════════════════════════════════════════════════════════════════════
# SHEAF GNN
# ═══════════════════════════════════════════════════════════════════════

class StalkEncoder(nn.Module):
    """Edge feature → initial stalk vector.

    The "stalk" here is the edge-level representation; we treat each edge
    as a node in a line-graph. This is simpler than dual per-endpoint
    stalks and matches our chain structure (we care about edges, not
    isolated nodes). The role distinction is implicit in the edge's
    (kind, polarity) tuple.
    """
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, edges: torch.Tensor) -> torch.Tensor:
        return self.proj(edges)


class SheafRestrictionMap(nn.Module):
    """Per-relation-kind bilinear map between adjacent stalks.

    Three kinds → three separate linear maps. This is the "sheaf"
    property: the map is a function of the edge type, not just the
    stalks. Messages through a connective edge use W_connective; through
    a reportive edge, W_reportive (which can learn to dampen).

    We parameterize the maps as gated linears: m_k = sigmoid(gate) * W_k(x).
    The gate gives the model an easy way to learn "reportive messages
    should barely propagate" without needing a separate polarity head.
    """
    def __init__(self, hidden_dim: int, n_kinds: int = len(RELATION_KINDS)):
        super().__init__()
        self.n_kinds = n_kinds
        # One Linear per kind (weight and bias).
        self.W = nn.Parameter(torch.empty(n_kinds, hidden_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(n_kinds, hidden_dim))
        self.gate = nn.Parameter(torch.zeros(n_kinds, hidden_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, stalk: torch.Tensor, kind_onehot: torch.Tensor) -> torch.Tensor:
        """stalk: (B, L, d); kind_onehot: (B, L, n_kinds) → (B, L, d)."""
        # Select the right W, b, gate per edge via one-hot mixing. For
        # each position, w_i = sum_k kind_onehot[i,k] * W[k].
        W_mix = torch.einsum("blk,kij->blij", kind_onehot, self.W)
        b_mix = torch.einsum("blk,kj->blj", kind_onehot, self.b)
        gate_mix = torch.einsum("blk,kj->blj", kind_onehot, self.gate)
        y = torch.einsum("blij,bli->blj", W_mix, stalk) + b_mix
        return torch.sigmoid(gate_mix) * y


class SheafMessageLayer(nn.Module):
    """One round of sheaf-aware message passing along a chain.

    Since our chain is a linear sequence (line graph), "neighbors" are
    the previous and next edges. We pass messages in both directions
    and aggregate. The restriction map is applied before the message.

    H¹ discrepancy is computed per-step as the residual between a node's
    current stalk and the restriction of its neighbor's. High residual
    means structural disagreement (e.g. a retraction next to an
    affirmation of the same triple).
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.restrict_fwd = SheafRestrictionMap(hidden_dim)
        self.restrict_bwd = SheafRestrictionMap(hidden_dim)
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, stalk: torch.Tensor, kind_onehot: torch.Tensor,
                mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """stalk: (B, L, d). Returns (updated_stalk, discrepancy)
        where discrepancy: (B, L) is the per-position H¹ magnitude."""
        # Restriction under forward and backward edges. Shift the sequence
        # to get prev/next stalks at each position.
        fwd_msg = self.restrict_fwd(stalk, kind_onehot)    # (B, L, d)
        bwd_msg = self.restrict_bwd(stalk, kind_onehot)

        # Shift: at position i, prev is fwd_msg[i-1], next is bwd_msg[i+1].
        prev = torch.roll(fwd_msg, shifts=1, dims=1)
        prev[:, 0, :] = 0.0
        nxt = torch.roll(bwd_msg, shifts=-1, dims=1)
        nxt[:, -1, :] = 0.0

        # Aggregated neighborhood message = mean of prev + next (no
        # attention: keeps layer parameter-light).
        agg = 0.5 * (prev + nxt)

        # H¹ discrepancy = L2 distance between self-stalk and agg, per
        # position. Masked-out positions contribute 0.
        discrepancy = torch.norm(stalk - agg, dim=-1) * mask

        # Update: MLP over (stalk, agg, stalk-agg residual).
        inp = torch.cat([stalk, agg, stalk - agg], dim=-1)
        new = self.update(inp)
        # Residual + mask.
        new = (stalk + new) * mask.unsqueeze(-1)
        return new, discrepancy


class ChainVerdictHead(nn.Module):
    """Reads the full chain of stalks + discrepancies → 3-class verdict.

    We pool three things:
      • the last (most-recent) stalk (endpoint signal)
      • mean of all stalks (chain summary)
      • sum of H¹ discrepancy (structural anomaly signal)
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, stalk: torch.Tensor, discrepancy: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """→ logits (B, 3)."""
        # Last real edge per batch (mask's rightmost 1).
        lengths = mask.sum(dim=1).long().clamp(min=1)
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, stalk.shape[-1])
        last = stalk.gather(1, idx).squeeze(1)           # (B, d)

        # Mean over valid positions.
        summed = (stalk * mask.unsqueeze(-1)).sum(dim=1)
        mean = summed / mask.sum(dim=1, keepdim=True).clamp(min=1)

        # Total structural discrepancy.
        total_discrep = discrepancy.sum(dim=1, keepdim=True)  # (B, 1)

        feat = torch.cat([last, mean, total_discrep], dim=-1)
        return self.head(feat)


class SheafHypergraphEncoder(nn.Module):
    """Full stack: StalkEncoder → N × SheafMessageLayer → ChainVerdictHead.

    Default 3 layers handles chain depths up to 6-7 hops with margin.
    The hidden_dim is tiny by design — the task is structural, not
    knowledge-dense, and smaller models train faster + transfer better.
    """
    def __init__(self, hidden_dim: int = 64, n_layers: int = 3):
        super().__init__()
        self.stalk_encoder = StalkEncoder(EDGE_FEATURE_DIM, hidden_dim)
        self.layers = nn.ModuleList(
            [SheafMessageLayer(hidden_dim) for _ in range(n_layers)]
        )
        self.verdict_head = ChainVerdictHead(hidden_dim)

    def forward(self, batch: HypergraphBatch) -> dict[str, torch.Tensor]:
        """Return {"logits": (B,3), "discrepancy": (B,L)}."""
        # Initial stalks.
        stalk = self.stalk_encoder(batch.edges) * batch.edge_mask.unsqueeze(-1)

        # Kind one-hot for restriction maps (slice of edge features).
        kind_onehot = batch.edges[..., :len(RELATION_KINDS)]

        total_discrep = torch.zeros_like(batch.edge_mask)
        for layer in self.layers:
            stalk, disc = layer(stalk, kind_onehot, batch.edge_mask)
            total_discrep = total_discrep + disc

        logits = self.verdict_head(stalk, total_discrep, batch.edge_mask)
        return {"logits": logits, "discrepancy": total_discrep}

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ═══════════════════════════════════════════════════════════════════════
# TRAIN / EVAL
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TrainResult:
    model: SheafHypergraphEncoder
    train_loss: list[float]
    val_acc: list[float]
    best_val_acc: float
    best_epoch: int


def train(model: SheafHypergraphEncoder,
          train_batch: HypergraphBatch,
          val_batch: HypergraphBatch,
          *,
          epochs: int = 40,
          batch_size: int = 64,
          lr: float = 2e-3,
          device: str = "cpu",
          verbose: bool = True) -> TrainResult:
    """Full-batch training loop. Returns the result + the model in-place.

    No gradient accumulation, no scheduler — the model is small and the
    dataset is small. Keeps the training script readable as a probe.
    """
    model.to(device)
    train_edges = train_batch.edges.to(device)
    train_mask = train_batch.edge_mask.to(device)
    train_y = train_batch.verdicts.to(device)
    val_edges = val_batch.edges.to(device)
    val_mask = val_batch.edge_mask.to(device)
    val_y = val_batch.verdicts.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    train_loss_hist: list[float] = []
    val_acc_hist: list[float] = []
    best_val_acc = 0.0
    best_epoch = 0
    best_state = None

    n = train_edges.shape[0]
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            mini = HypergraphBatch(
                edges=train_edges[idx],
                edge_mask=train_mask[idx],
                verdicts=train_y[idx],
            )
            out = model(mini)
            loss = F.cross_entropy(out["logits"], mini.verdicts)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * idx.shape[0]
        epoch_loss /= n
        train_loss_hist.append(epoch_loss)

        # Val.
        model.eval()
        with torch.no_grad():
            out = model(HypergraphBatch(edges=val_edges, edge_mask=val_mask,
                                          verdicts=val_y))
            preds = out["logits"].argmax(dim=-1)
            acc = (preds == val_y).float().mean().item()
        val_acc_hist.append(acc)
        if acc > best_val_acc:
            best_val_acc = acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:>3}  train_loss={epoch_loss:.4f}  "
                  f"val_acc={acc:.3f}  best={best_val_acc:.3f} @ {best_epoch}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainResult(
        model=model,
        train_loss=train_loss_hist,
        val_acc=val_acc_hist,
        best_val_acc=best_val_acc,
        best_epoch=best_epoch,
    )


def evaluate(model: SheafHypergraphEncoder,
             batch: HypergraphBatch,
             device: str = "cpu") -> dict:
    """Per-kind accuracy + confusion matrix.

    Relies on the caller tracking synthgraph.kind alongside the batch —
    we pass it back as a list so the probe can slice."""
    model.eval()
    model.to(device)
    with torch.no_grad():
        out = model(HypergraphBatch(
            edges=batch.edges.to(device),
            edge_mask=batch.edge_mask.to(device),
            verdicts=batch.verdicts.to(device),
        ))
        preds = out["logits"].argmax(dim=-1).cpu().numpy()
        gold = batch.verdicts.numpy()
    return {"preds": preds, "gold": gold, "logits": out["logits"].cpu()}

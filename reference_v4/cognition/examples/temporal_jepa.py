"""Temporal-JEPA on the infon hypergraph.

Idea (from V-JEPA / V-JEPA 2): treat the hypergraph as a latent
dynamical system. Given an infon's embedding h_s and an edge relation
r ∈ {NEXT, CAUSES, ENTAILS}, a small predictor P_r learns to map h_s
to the embedding of the target infon h_t *in the latent space of an
EMA-smoothed teacher*.

Architecture
------------
  Student:  GNN layers (trainable)       → h_student(i)
  Teacher:  EMA copy of same GNN         → h_teacher(i)   (stop-grad)
  Predictor: per-relation 2-layer MLP    → P_r(h_student(s))

Loss:
  L = Σ over edges (s, r, t)  ||  P_r(h_student(s)) − stop_grad(h_teacher(t))  ||²

VICReg-style variance + covariance terms on teacher targets prevent
collapse without needing contrast.

Why it's the right shape for this graph
---------------------------------------
1. The asymmetry (student predicts, teacher targets) is what BT/VICReg
   lacked — the feat_noise collapse we saw was because BT forced
   equality between two views that carried different information.
   JEPA lets the views play different roles.

2. Relation-conditioning makes it a *world model*: once P_NEXT is
   trained, we can ask "what infon follows this one?" — rollout in
   latent space.

3. The DS mass readout is unchanged. JEPA shapes the embedding; DS
   reads out calibrated verdicts from it. The calibration invariants
   we care about should ride along.

Metrics reported
----------------
- Prediction MSE per relation (smaller = better world model)
- Target variance (VICReg-style; high = no collapse)
- Off-diagonal covariance RMS (low = decorrelated axes)
- Effective rank of the embedding space
- Downstream reasoning accuracy + θ calibration vs the pre-JEPA baseline
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F

from cognition import Cognition, CognitionConfig
from cognition.logic import HypergraphReasoner, REL_TO_IDX


# ═════════════════════════════════════════════════════════════════════
# Corpus — denser than the BT run so we get usable NEXT / CAUSES edges
# ═════════════════════════════════════════════════════════════════════

SCHEMA = {
    "toyota":   {"type": "actor",    "tokens": ["toyota"]},
    "honda":    {"type": "actor",    "tokens": ["honda"]},
    "tesla":    {"type": "actor",    "tokens": ["tesla"]},
    "panasonic":{"type": "actor",    "tokens": ["panasonic"]},
    "catl":     {"type": "actor",    "tokens": ["catl"]},
    "invests":  {"type": "relation", "tokens": ["invest", "invests", "investment"]},
    "partners": {"type": "relation", "tokens": ["partner", "partners", "partnership"]},
    "produces": {"type": "relation", "tokens": ["produce", "produces", "produced"]},
    "expands":  {"type": "relation", "tokens": ["expand", "expands", "expansion"]},
    "delays":   {"type": "relation", "tokens": ["delay", "delays", "delayed"]},
    "acquires": {"type": "relation", "tokens": ["acquire", "acquires", "acquired"]},
    "battery":  {"type": "feature",  "tokens": ["battery", "batteries"]},
    "factory":  {"type": "feature",  "tokens": ["factory", "plant"]},
    "ev":       {"type": "feature",  "tokens": ["ev", "electric vehicle"]},
    "japan":    {"type": "market",   "tokens": ["japan", "japanese"]},
    "china":    {"type": "market",   "tokens": ["china", "chinese"]},
    "na":       {"type": "market",   "tokens": ["north america", "united states"]},
}

DOCS = [
    # Docs with timestamps so NEXT edges actually form during consolidate().
    {"id": "d01", "timestamp": "2024-01-10",
     "text": "Toyota invests in battery technology in Japan."},
    {"id": "d02", "timestamp": "2024-02-15",
     "text": "Toyota partners with Panasonic on battery development."},
    {"id": "d03", "timestamp": "2024-03-22",
     "text": "Toyota produces prototype batteries."},
    {"id": "d04", "timestamp": "2024-01-05",
     "text": "Tesla expands its battery factory in North America."},
    {"id": "d05", "timestamp": "2024-02-28",
     "text": "Tesla produces batteries at its Gigafactory."},
    {"id": "d06", "timestamp": "2024-04-01",
     "text": "Tesla acquires battery supply chain assets."},
    {"id": "d07", "timestamp": "2024-01-20",
     "text": "Honda partners with CATL on battery supply in China."},
    {"id": "d08", "timestamp": "2024-02-10",
     "text": "Honda delays its EV production timeline."},
    {"id": "d09", "timestamp": "2024-03-15",
     "text": "Honda invests in battery research."},
    {"id": "d10", "timestamp": "2024-01-25",
     "text": "CATL expands battery production in China."},
    {"id": "d11", "timestamp": "2024-02-20",
     "text": "CATL produces batteries for Japanese automakers."},
    {"id": "d12", "timestamp": "2024-01-30",
     "text": "Panasonic invests in battery factory in Japan."},
]


# ═════════════════════════════════════════════════════════════════════
# Predictor and EMA helpers
# ═════════════════════════════════════════════════════════════════════

class RelationPredictor(nn.Module):
    """Per-relation 2-layer MLP. Given h_s in latent space, predict h_t."""

    def __init__(self, hidden_dim: int, n_relations: int,
                 proj_dim: int = 128):
        super().__init__()
        # One predictor per relation, but share the trunk — trunk sees
        # hidden_dim + a learned per-relation embedding.
        self.rel_embed = nn.Embedding(n_relations, hidden_dim)
        self.trunk = nn.Sequential(
            nn.Linear(2 * hidden_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, hidden_dim),
        )

    def forward(self, h_src: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
        """h_src: (N, D), rel: (N,) long → (N, D) predicted targets."""
        rel_e = self.rel_embed(rel)
        x = torch.cat([h_src, rel_e], dim=-1)
        return self.trunk(x)


@torch.no_grad()
def update_ema(student: nn.Module, teacher: nn.Module, tau: float):
    """Polyak update: teacher ← tau * teacher + (1-tau) * student."""
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(tau).add_(ps.data, alpha=1.0 - tau)


def forward_gnn(layers: nn.ModuleList, graph) -> torch.Tensor:
    h = graph.node_features
    for layer in layers:
        h = layer(h, graph.edge_index, graph.edge_types,
                  graph.edge_weights, graph.situation_features)
    return h


# ═════════════════════════════════════════════════════════════════════
# VICReg-style variance + covariance on teacher targets
# ═════════════════════════════════════════════════════════════════════

def variance_term(z: torch.Tensor, gamma: float = 1.0,
                  eps: float = 1e-4) -> torch.Tensor:
    """Hinge: per-dim std should be ≥ gamma."""
    std = torch.sqrt(z.var(dim=0) + eps)
    return F.relu(gamma - std).mean()


def covariance_term(z: torch.Tensor) -> torch.Tensor:
    """Off-diagonal covariance magnitude, normalized by dim."""
    n, d = z.shape
    z = z - z.mean(dim=0, keepdim=True)
    cov = (z.T @ z) / max(n - 1, 1)
    off = cov - torch.diag(torch.diagonal(cov))
    return off.pow(2).sum() / d


def effective_rank(z: torch.Tensor) -> float:
    with torch.no_grad():
        s = torch.linalg.svdvals(z - z.mean(dim=0, keepdim=True))
        s2 = s.pow(2)
        s2 = s2 / (s2.sum() + 1e-12)
        return math.exp(-(s2 * (s2 + 1e-12).log()).sum().item())


# ═════════════════════════════════════════════════════════════════════
# Experiment driver
# ═════════════════════════════════════════════════════════════════════

def build_cog(tmpdir: str):
    schema_path = os.path.join(tmpdir, "schema.json")
    with open(schema_path, "w") as f:
        json.dump(SCHEMA, f)
    cog = Cognition(CognitionConfig(
        schema_path=schema_path,
        db_path=os.path.join(tmpdir, "cog.db"),
        activation_threshold=0.2,
        min_confidence=0.02,
        top_k_per_role=3,
        quality_threshold=0.04,
        max_triples_per_sentence=2,
    ))
    for d in DOCS:
        cog.ingest([d])
    cog.consolidate()
    return cog


QUERIES = [
    ("Did Toyota invest in batteries?",       "SUPPORTS"),
    ("Did Tesla produce batteries?",          "SUPPORTS"),
    ("Did Honda partner with CATL?",          "SUPPORTS"),
    ("Did Panasonic invest in batteries?",    "SUPPORTS"),
    ("Did Tesla acquire battery assets?",     "SUPPORTS"),
    ("Did Tesla acquire CATL?",               "NOT_ENOUGH_INFO"),
    ("Did Honda produce batteries in Japan?", "NOT_ENOUGH_INFO"),
    ("Did Toyota merge with Honda?",          "NOT_ENOUGH_INFO"),
]


def evaluate(reasoner) -> dict:
    reasoner.eval()
    correct = 0
    theta_s, theta_n = [], []
    for q, gold in QUERIES:
        r = reasoner.reason(q)
        if r.verdict == gold:
            correct += 1
        if gold == "SUPPORTS":
            theta_s.append(r.mass.theta)
        else:
            theta_n.append(r.mass.theta)
    return {
        "accuracy": correct / len(QUERIES),
        "theta_supports": sum(theta_s) / max(len(theta_s), 1),
        "theta_nei":      sum(theta_n) / max(len(theta_n), 1),
    }


def run_jepa(use_sheaf: bool = True,
             hidden_dim: int = 32,
             epochs: int = 60,
             lr: float = 3e-3,
             tau: float = 0.996,
             lambda_var: float = 1.0,
             lambda_cov: float = 0.05,
             use_all_edges: bool = True,
             seed: int = 0,
             verbose: bool = True):
    """Main driver. Returns a dict of summary metrics."""
    torch.manual_seed(seed)
    random.seed(seed)

    with tempfile.TemporaryDirectory() as tmpdir:
        cog = build_cog(tmpdir)
        reasoner = HypergraphReasoner(
            cog.store, cog.encoder, cog.schema,
            hidden_dim=hidden_dim, n_layers=2, use_sheaf=use_sheaf,
        )
        graph = reasoner.builder.build(feature_dim=hidden_dim)

        # Run GNN-level refinement so CAUSES / NEXT edges populate.
        # refine() is the existing method that discovers causal/contrary
        # edges via the IKL operators; it writes back to the store.
        try:
            reasoner.fit(graph=graph, epochs=10,
                         laplacian_weight=0.1 if use_sheaf else 0.0,
                         verbose=False)
            reasoner.refine(verbose=False)
            # Rebuild the graph so new CAUSES edges enter edge_index.
            graph = reasoner.builder.build(feature_dim=hidden_dim)
        except Exception as e:
            if verbose:
                print(f"  (refine skipped: {e.__class__.__name__})")

        # Count temporal / causal edges available to JEPA
        edge_types = graph.edge_types
        next_idx = REL_TO_IDX["NEXT"]
        causes_idx = REL_TO_IDX["CAUSES"]
        entails_idx = REL_TO_IDX["ENTAILS"]

        jepa_rels = {next_idx, causes_idx, entails_idx}
        if use_all_edges:
            jepa_rels |= {REL_TO_IDX["INITIATES"], REL_TO_IDX["ASSERTS"],
                          REL_TO_IDX["TARGETS"]}

        mask = torch.tensor(
            [int(t) in jepa_rels for t in edge_types],
            dtype=torch.bool,
        )
        if mask.sum() == 0:
            print("  no eligible edges for JEPA — aborting")
            return None

        src_idx = graph.edge_index[0][mask]
        tgt_idx = graph.edge_index[1][mask]
        rel_idx = graph.edge_types[mask]

        n_edges = int(mask.sum())
        if verbose:
            n_next = int((rel_idx == next_idx).sum())
            n_caus = int((rel_idx == causes_idx).sum())
            n_ent = int((rel_idx == entails_idx).sum())
            n_spoke = n_edges - n_next - n_caus - n_ent
            print(f"  JEPA edges: {n_edges} total  "
                  f"(NEXT={n_next}, CAUSES={n_caus}, "
                  f"ENTAILS={n_ent}, spoke={n_spoke})")

        # Baseline accuracy BEFORE JEPA finetune
        pre = evaluate(reasoner)
        if verbose:
            print(f"  BASELINE (after fit + refine):")
            print(f"    accuracy = {pre['accuracy']:.0%}   "
                  f"θ_S = {pre['theta_supports']:.2f}   "
                  f"θ_NEI = {pre['theta_nei']:.2f}")

        # ── JEPA setup ───────────────────────────────────────────
        # Teacher = frozen deep copy of current student layers
        teacher = copy.deepcopy(reasoner.layers)
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.eval()

        predictor = RelationPredictor(
            hidden_dim=hidden_dim,
            n_relations=len(REL_TO_IDX),
        )

        # Only train the GNN layers + predictor (DS readout untouched)
        params = list(reasoner.layers.parameters()) + list(predictor.parameters())
        opt = torch.optim.Adam(params, lr=lr)

        history = []
        for epoch in range(epochs):
            reasoner.train()
            opt.zero_grad()

            h_student = forward_gnn(reasoner.layers, graph)
            with torch.no_grad():
                h_teacher = forward_gnn(teacher, graph)

            # Predict target embeddings from source + relation
            h_src = h_student[src_idx]
            preds = predictor(h_src, rel_idx)
            targets = h_teacher[tgt_idx]

            # Per-relation prediction MSE
            pred_loss = F.mse_loss(preds, targets)

            # VICReg variance + covariance on the TEACHER embedding
            # of infon nodes (prevents collapse across infon subspace)
            infon_indices = torch.tensor(graph.infon_indices, dtype=torch.long)
            h_inf = h_student[infon_indices]
            var_t = variance_term(h_inf)
            cov_t = covariance_term(h_inf)

            loss = pred_loss + lambda_var * var_t + lambda_cov * cov_t
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            update_ema(reasoner.layers, teacher, tau)

            with torch.no_grad():
                er = effective_rank(h_inf.detach())

            history.append({
                "epoch": epoch + 1,
                "pred_loss": pred_loss.item(),
                "var": var_t.item(),
                "cov": cov_t.item(),
                "eff_rank": er,
                "loss": loss.item(),
            })

        # Per-relation prediction quality after training
        reasoner.eval()
        with torch.no_grad():
            h_student = forward_gnn(reasoner.layers, graph)
            h_teacher = forward_gnn(teacher, graph)
            preds = predictor(h_student[src_idx], rel_idx)
            targets = h_teacher[tgt_idx]
            err = (preds - targets).pow(2).sum(-1)
            # Nearest-neighbour accuracy: is the predicted vector closest
            # (among all infon embeddings) to its true target?
            infon_indices = torch.tensor(graph.infon_indices, dtype=torch.long)
            bank = h_teacher[infon_indices]
            # cosine sim
            preds_n = F.normalize(preds, dim=-1)
            bank_n = F.normalize(bank, dim=-1)
            sim = preds_n @ bank_n.T
            # For each edge, the correct rank of tgt_idx among infons
            tgt_pos = {int(ii): k for k, ii in enumerate(graph.infon_indices)}
            ks = []
            for i in range(preds.shape[0]):
                t = int(tgt_idx[i])
                if t not in tgt_pos:
                    continue
                rank = (sim[i] > sim[i, tgt_pos[t]]).sum().item() + 1
                ks.append(rank)
            top1 = sum(1 for r in ks if r == 1) / max(len(ks), 1)
            top3 = sum(1 for r in ks if r <= 3) / max(len(ks), 1)
            mrr = sum(1.0 / r for r in ks) / max(len(ks), 1)

        post = evaluate(reasoner)

        cog.close()

        return {
            "pre":   pre,
            "post":  post,
            "history": history,
            "pred_top1": top1,
            "pred_top3": top3,
            "mrr":       mrr,
            "n_edges":   n_edges,
        }


def main():
    print("Temporal-JEPA (NEXT/CAUSES/ENTAILS + spoke) on the hypergraph")
    print("=" * 64)

    r = run_jepa(use_sheaf=True, epochs=60, tau=0.996,
                 lambda_var=1.0, lambda_cov=0.05, use_all_edges=True)
    if r is None:
        return

    first = r["history"][0]; last = r["history"][-1]
    print()
    print(f"{'metric':<25s} {'before':>10s} {'after':>10s}")
    print("-" * 48)
    print(f"{'pred MSE':<25s} {first['pred_loss']:>10.3f} {last['pred_loss']:>10.3f}")
    print(f"{'variance hinge':<25s} {first['var']:>10.3f} {last['var']:>10.3f}")
    print(f"{'covariance':<25s} {first['cov']:>10.3f} {last['cov']:>10.3f}")
    print(f"{'eff-rank(infons)':<25s} {first['eff_rank']:>10.2f} {last['eff_rank']:>10.2f}")
    print()
    print(f"prediction top-1:   {r['pred_top1']:.0%}")
    print(f"prediction top-3:   {r['pred_top3']:.0%}")
    print(f"prediction MRR:     {r['mrr']:.3f}")
    print()
    print(f"reasoning accuracy: {r['pre']['accuracy']:.0%} → {r['post']['accuracy']:.0%}")
    print(f"θ on SUPPORTS:      {r['pre']['theta_supports']:.2f} → {r['post']['theta_supports']:.2f}")
    print(f"θ on NEI:           {r['pre']['theta_nei']:.2f} → {r['post']['theta_nei']:.2f}")

    print("\n— Ablation: NEXT+CAUSES only (no spoke edges) —")
    r2 = run_jepa(use_sheaf=True, epochs=60, tau=0.996,
                  lambda_var=1.0, lambda_cov=0.05, use_all_edges=False,
                  seed=1, verbose=True)
    if r2 is not None:
        print(f"  pred top-1: {r2['pred_top1']:.0%}   "
              f"MRR: {r2['mrr']:.3f}   "
              f"acc: {r2['pre']['accuracy']:.0%} → {r2['post']['accuracy']:.0%}")


if __name__ == "__main__":
    main()

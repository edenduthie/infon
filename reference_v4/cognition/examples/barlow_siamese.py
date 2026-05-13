"""Barlow-Twins Siamese experiment on infon embeddings.

Premise: the same event, observed two ways, should embed to the same
point — *and* different events should occupy different directions.
Barlow Twins (Zbontar et al. 2021) enforces both at once via the
cross-correlation matrix of two views:

    L_BT = Σ_i (1 - C_ii)²    +    λ · Σ_{i≠j} C_ij²
             │── invariance ──│   │── redundancy reduction ──│

where C is the batch cross-correlation between projected views A and B.
No negatives, no asymmetric predictor, no temperature.

Setup here
----------
- View A: the GNN's per-infon embedding from the EV corpus.
- View B: three corruption schemes applied to the *input* features
  before the same GNN processes them:
    1. role-mask    — zero out one of the spoke edges (subject, pred, obj)
    2. edge-drop    — randomly drop 30% of edges incident to the infon
    3. polarity-strip — swap sign on the infon's polarity metadata
- Both views go through the same SheafMessagePassingLayer reasoner,
  then a shared `BarlowHead` projects to a 128-d space for the BT loss.
- We measure, over the training run:
    * on-diagonal C_ii mean (should climb toward 1 → invariance)
    * off-diagonal C_ij RMS (should drop toward 0 → no collapse)
    * effective rank of the projected embeddings (should stay high)
    * the DS reasoning accuracy doesn't regress (BT isn't fighting the
      teacher)

The Sheaf layer plus BT is the natural combination: the sheaf-Laplacian
already asks edges to be coherent at the stalk level; Barlow Twins asks
*nodes* to be invariant under corruption. Different targets, compatible
gradients.
"""
from __future__ import annotations

import json
import math
import os
import random
import tempfile
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cognition import Cognition, CognitionConfig
from cognition.logic import HypergraphReasoner


# ═════════════════════════════════════════════════════════════════════
# Corpus (re-used from tests/test_sheaf.py, expanded a bit)
# ═════════════════════════════════════════════════════════════════════

SCHEMA = {
    "toyota":   {"type": "actor",    "tokens": ["toyota"]},
    "honda":    {"type": "actor",    "tokens": ["honda"]},
    "tesla":    {"type": "actor",    "tokens": ["tesla"]},
    "panasonic":{"type": "actor",    "tokens": ["panasonic"]},
    "catl":     {"type": "actor",    "tokens": ["catl"]},
    "invests":  {"type": "relation", "tokens":
                 ["invest", "invests", "investment"]},
    "partners": {"type": "relation", "tokens":
                 ["partner", "partners", "partnership"]},
    "produces": {"type": "relation", "tokens":
                 ["produce", "produces", "produced"]},
    "acquires": {"type": "relation", "tokens":
                 ["acquire", "acquires", "acquired"]},
    "battery":  {"type": "feature",  "tokens": ["battery", "batteries"]},
    "factory":  {"type": "feature",  "tokens": ["factory", "plant"]},
    "japan":    {"type": "market",   "tokens": ["japan", "japanese"]},
    "china":    {"type": "market",   "tokens": ["china", "chinese"]},
    "na":       {"type": "market",   "tokens":
                 ["north america", "united states"]},
}


DOCS = [
    {"id": "d01", "text": "Toyota invests in battery technology in Japan."},
    {"id": "d02", "text": "Tesla produces batteries at its factory."},
    {"id": "d03", "text": "Honda partners with CATL on battery supply in China."},
    {"id": "d04", "text": "Panasonic invests in battery factory in Japan."},
    {"id": "d05", "text": "CATL produces batteries for multiple Japanese automakers."},
    {"id": "d06", "text": "Toyota partners with Panasonic on battery development in Japan."},
    {"id": "d07", "text": "Tesla acquires battery supply chain assets."},
    {"id": "d08", "text": "Honda invests in battery research."},
    {"id": "d09", "text": "Toyota produces prototype batteries."},
    {"id": "d10", "text": "Tesla invests in North America factory."},
]


# ═════════════════════════════════════════════════════════════════════
# Barlow Head + loss
# ═════════════════════════════════════════════════════════════════════

class BarlowHead(nn.Module):
    """Shared 2-layer projection head. Output is BN'd along the feature
    axis so the cross-correlation matrix is well-defined without extra
    normalization inside the loss."""

    def __init__(self, in_dim: int, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.bn = nn.BatchNorm1d(proj_dim, affine=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.bn(self.net(h))


def barlow_twins_loss(z_a: torch.Tensor,
                      z_b: torch.Tensor,
                      lambda_off: float = 5e-3) -> tuple[torch.Tensor, dict]:
    """Standard BT loss.

    z_a, z_b: (N, D) already batch-normalized along D.
    """
    n, d = z_a.shape
    # Cross-correlation over the *batch* axis (Zbontar et al.)
    c = (z_a.T @ z_b) / n

    on_diag = torch.diagonal(c)
    off_diag = c - torch.diag(on_diag)

    invariance = ((1.0 - on_diag) ** 2).sum()
    redundancy = (off_diag ** 2).sum()
    loss = invariance + lambda_off * redundancy

    metrics = {
        "on_diag_mean":  on_diag.mean().item(),
        "off_diag_rms":  math.sqrt((off_diag ** 2).mean().item()),
        "invariance":    invariance.item(),
        "redundancy":    redundancy.item(),
        "loss":          loss.item(),
    }
    return loss, metrics


def effective_rank(z: torch.Tensor) -> float:
    """Participation ratio of squared singular values —
    1.0 if all mass in one direction (collapse), D if uniform."""
    with torch.no_grad():
        s = torch.linalg.svdvals(z - z.mean(dim=0, keepdim=True))
        s2 = (s ** 2)
        s2 = s2 / (s2.sum() + 1e-12)
        entropy = -(s2 * (s2 + 1e-12).log()).sum()
        return math.exp(entropy.item())


# ═════════════════════════════════════════════════════════════════════
# View corruptions
# ═════════════════════════════════════════════════════════════════════

def corrupt_role_mask(graph, reasoner, rng):
    """Zero out one role's spoke edge per infon. The GNN will miss one
    of the INITIATES/ASSERTS/TARGETS signals for each infon, forcing
    the other two to carry the content."""
    new_edges_keep = []
    infon_node_set = set(graph.infon_map.values())
    dropped_pairs: set[tuple[int, int]] = set()

    # For each infon, pick one spoke-role to drop
    from cognition.logic import REL_TO_IDX
    spoke_rels = {REL_TO_IDX["INITIATES"], REL_TO_IDX["ASSERTS"], REL_TO_IDX["TARGETS"]}

    # First pass: per-infon, choose one spoke rel to drop
    drop_rel_for_infon: dict[int, int] = {}
    for inf_idx in infon_node_set:
        drop_rel_for_infon[inf_idx] = rng.choice(list(spoke_rels))

    # Second pass: keep edges unless they're the dropped spoke
    n_edges = graph.edge_index.shape[1]
    keep = torch.ones(n_edges, dtype=torch.bool)
    for e in range(n_edges):
        r = int(graph.edge_types[e])
        if r not in spoke_rels:
            continue
        s = int(graph.edge_index[0, e])
        t = int(graph.edge_index[1, e])
        # Spoke edges: either source or target is the infon
        inf = t if t in infon_node_set else (s if s in infon_node_set else None)
        if inf is None:
            continue
        if drop_rel_for_infon.get(inf) == r:
            keep[e] = False
    new_ei = graph.edge_index[:, keep]
    new_et = graph.edge_types[keep]
    new_ew = graph.edge_weights[keep]
    return new_ei, new_et, new_ew


def corrupt_edge_drop(graph, frac: float, rng):
    """Randomly drop `frac` of all edges."""
    n_edges = graph.edge_index.shape[1]
    keep = torch.tensor(
        [rng.random() >= frac for _ in range(n_edges)],
        dtype=torch.bool,
    )
    return (graph.edge_index[:, keep],
            graph.edge_types[keep],
            graph.edge_weights[keep])


def corrupt_feature_noise(features: torch.Tensor, sigma: float,
                          rng: torch.Generator) -> torch.Tensor:
    """Add Gaussian noise to node features."""
    noise = torch.randn(features.shape, generator=rng) * sigma
    return features + noise


# ═════════════════════════════════════════════════════════════════════
# Experiment
# ═════════════════════════════════════════════════════════════════════

@dataclass
class CorruptionConfig:
    name: str
    kind: str        # "edge_drop" | "role_mask" | "feat_noise"
    param: float = 0.0


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


def run_experiment(corruption: CorruptionConfig,
                   use_sheaf: bool = True,
                   freeze_trunk: bool = False,
                   hidden_dim: int = 32,
                   proj_dim: int = 128,
                   epochs: int = 40,
                   lr: float = 3e-3,
                   lambda_off: float = 5e-3,
                   seed: int = 0):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    feat_rng = torch.Generator().manual_seed(seed + 7)

    with tempfile.TemporaryDirectory() as tmpdir:
        cog = build_cog(tmpdir)

        reasoner = HypergraphReasoner(
            cog.store, cog.encoder, cog.schema,
            hidden_dim=hidden_dim, n_layers=2, use_sheaf=use_sheaf,
        )
        graph = reasoner.builder.build(feature_dim=hidden_dim)

        # Warm-start the reasoner so the GNN isn't totally random.
        reasoner.fit(graph=graph, epochs=10,
                     laplacian_weight=0.1 if use_sheaf else 0.0,
                     verbose=False)
        reasoner.train()

        head = BarlowHead(hidden_dim, proj_dim=proj_dim)
        if freeze_trunk:
            for p in reasoner.layers.parameters():
                p.requires_grad_(False)
            opt = torch.optim.Adam(head.parameters(), lr=lr)
        else:
            opt = torch.optim.Adam(
                list(reasoner.layers.parameters())
                + list(head.parameters()),
                lr=lr,
            )

        infon_idx = torch.tensor(
            [i for i, t in enumerate(graph.node_types) if t == "infon"],
            dtype=torch.long,
        )
        if len(infon_idx) < 8:
            print(f"  too few infons ({len(infon_idx)}) for batch-norm BT; "
                  f"skipping {corruption.name}")
            return None

        history = []

        for epoch in range(epochs):
            opt.zero_grad()

            # View A — clean
            h_a = graph.node_features
            for layer in reasoner.layers:
                h_a = layer(h_a, graph.edge_index, graph.edge_types,
                            graph.edge_weights, graph.situation_features)

            # View B — corrupted
            if corruption.kind == "edge_drop":
                ei, et, ew = corrupt_edge_drop(graph, corruption.param, rng)
                feats = graph.node_features
            elif corruption.kind == "role_mask":
                ei, et, ew = corrupt_role_mask(graph, reasoner, rng)
                feats = graph.node_features
            elif corruption.kind == "feat_noise":
                ei, et, ew = graph.edge_index, graph.edge_types, graph.edge_weights
                feats = corrupt_feature_noise(
                    graph.node_features, corruption.param, feat_rng,
                )
            else:
                raise ValueError(corruption.kind)

            h_b = feats
            for layer in reasoner.layers:
                h_b = layer(h_b, ei, et, ew, graph.situation_features)

            # Project through shared BN'd head
            z_a = head(h_a[infon_idx])
            z_b = head(h_b[infon_idx])

            loss, m = barlow_twins_loss(z_a, z_b, lambda_off=lambda_off)

            loss.backward()
            trainable = list(head.parameters())
            if not freeze_trunk:
                trainable = list(reasoner.layers.parameters()) + trainable
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()

            er = effective_rank(z_a.detach())
            m["eff_rank"] = er
            m["epoch"] = epoch + 1
            history.append(m)

        reasoner.eval()

        # After training — evaluate downstream reasoning accuracy
        queries = [
            ("Did Toyota invest in batteries?",       "SUPPORTS"),
            ("Did Tesla produce batteries?",          "SUPPORTS"),
            ("Did Honda partner with CATL?",          "SUPPORTS"),
            ("Did Panasonic invest in batteries?",    "SUPPORTS"),
            ("Did Tesla acquire CATL?",               "NOT_ENOUGH_INFO"),
            ("Did Honda produce batteries in Japan?", "NOT_ENOUGH_INFO"),
            ("Did Toyota merge with Honda?",          "NOT_ENOUGH_INFO"),
        ]
        correct = 0
        theta_s, theta_n = [], []
        for q, gold in queries:
            r = reasoner.reason(q)
            if r.verdict == gold:
                correct += 1
            if gold == "SUPPORTS":
                theta_s.append(r.mass.theta)
            else:
                theta_n.append(r.mass.theta)

        cog.close()

        return {
            "name":         corruption.name,
            "history":      history,
            "accuracy":     correct / len(queries),
            "theta_supports": sum(theta_s) / max(len(theta_s), 1),
            "theta_nei":      sum(theta_n) / max(len(theta_n), 1),
            "n_infons":     int(len(infon_idx)),
        }


def main():
    corruptions = [
        CorruptionConfig("edge_drop_30",  "edge_drop",  0.30),
        CorruptionConfig("edge_drop_50",  "edge_drop",  0.50),
        CorruptionConfig("role_mask_1",   "role_mask",  0.0),
        CorruptionConfig("feat_noise_0.3","feat_noise", 0.30),
    ]

    # Baseline (no BT training at all) for comparison
    print("Baseline (no Barlow training, just the DS-fit sheaf reasoner)")
    print("=" * 64)
    with tempfile.TemporaryDirectory() as tmp:
        cog = build_cog(tmp)
        reasoner = HypergraphReasoner(
            cog.store, cog.encoder, cog.schema,
            hidden_dim=32, n_layers=2, use_sheaf=True,
        )
        graph = reasoner.builder.build(feature_dim=32)
        reasoner.fit(graph=graph, epochs=10,
                     laplacian_weight=0.1, verbose=False)
        queries = [
            ("Did Toyota invest in batteries?",       "SUPPORTS"),
            ("Did Tesla produce batteries?",          "SUPPORTS"),
            ("Did Honda partner with CATL?",          "SUPPORTS"),
            ("Did Panasonic invest in batteries?",    "SUPPORTS"),
            ("Did Tesla acquire CATL?",               "NOT_ENOUGH_INFO"),
            ("Did Honda produce batteries in Japan?", "NOT_ENOUGH_INFO"),
            ("Did Toyota merge with Honda?",          "NOT_ENOUGH_INFO"),
        ]
        correct = 0; ts = []; tn = []
        for q, gold in queries:
            r = reasoner.reason(q)
            if r.verdict == gold: correct += 1
            (ts if gold == "SUPPORTS" else tn).append(r.mass.theta)
        baseline_acc = correct / len(queries)
        baseline_ts = sum(ts) / len(ts); baseline_tn = sum(tn) / len(tn)
        print(f"  baseline acc = {baseline_acc:.0%}   "
              f"θ_S = {baseline_ts:.2f}   θ_NEI = {baseline_tn:.2f}")
        cog.close()

    # Two training modes: joint (trunk + head) and probe-only (frozen trunk)
    for freeze in (False, True):
        mode = "probe (frozen trunk)" if freeze else "joint (trunk + head)"
        print(f"\n{'='*64}\nMode: {mode}\n{'='*64}")
        results = []
        for c in corruptions:
            print(f"\n── corruption: {c.name} "
                  f"(kind={c.kind}, param={c.param}) ──")
            r = run_experiment(c, use_sheaf=True,
                               freeze_trunk=freeze, epochs=40, seed=0)
            if r is None: continue
            first = r["history"][0]; last = r["history"][-1]
            print(f"  C_ii        : {first['on_diag_mean']:+.3f} → "
                  f"{last['on_diag_mean']:+.3f}")
            print(f"  C_ij (rms)  : {first['off_diag_rms']:.3f}  → "
                  f"{last['off_diag_rms']:.3f}")
            print(f"  eff-rank    : {first['eff_rank']:.2f}  → "
                  f"{last['eff_rank']:.2f}  /128")
            print(f"  acc         : {r['accuracy']:.0%}  "
                  f"(baseline {baseline_acc:.0%})")
            print(f"  θ_SUPPORTS  : {r['theta_supports']:.2f}  "
                  f"(baseline {baseline_ts:.2f})")
            print(f"  θ_NEI       : {r['theta_nei']:.2f}  "
                  f"(baseline {baseline_tn:.2f})")
            results.append(r)

        print("\n  Summary:")
        print(f"  {'corruption':<18s} {'C_ii':>7s} {'C_ij':>7s} "
              f"{'rank':>6s} {'Δacc':>7s} {'Δθ_NEI':>8s}")
        print("  " + "-" * 58)
        for r in results:
            last = r["history"][-1]
            dacc = r["accuracy"] - baseline_acc
            dthn = r["theta_nei"] - baseline_tn
            print(f"  {r['name']:<18s} {last['on_diag_mean']:>+7.3f} "
                  f"{last['off_diag_rms']:>7.3f} "
                  f"{last['eff_rank']:>6.2f} "
                  f"{dacc:>+7.0%} {dthn:>+8.2f}")


if __name__ == "__main__":
    main()

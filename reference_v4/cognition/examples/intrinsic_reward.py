"""Intrinsic-reward MDP — the reward signal is the JEPA predictor's
own surprise, not a hand-coded DS source.

Premise
-------
In the current pipeline, Dempster–Shafer masses come from six hand-
crafted heuristics (polarity, triple alignment, anchor distance,
confidence, evidentiality, modality). That's the last human-authored
signal in the learning loop. A LeCun-style self-supervised MDP would
derive the reward from the *world model's own behavior*:

    An infon whose incoming edges are well-predicted by the JEPA
    predictor → coherent, dynamics-consistent → high SUPPORTS.

    An infon whose incoming edges fail prediction → surprising,
    off-manifold → high θ (ignorance).

    An infon whose *mirror* (role-swapped) has lower prediction error
    than itself → likely refuted.

Mapping to DS mass (S, R, U, θ)
-------------------------------
For each infon node i, collect the set of incoming edges E_i =
{(s, r, i)}. Define per-edge prediction error
    e_sri = || P_r(student(s)) - teacher(i) ||²

Per-infon surprise
    surprise_i = mean_{e ∈ E_i} e

Per-infon coherence
    coh_i = exp(-surprise_i / σ)    ∈ (0, 1]

Intrinsic mass
    supports_i = coh_i · polarity_i
    refutes_i  = coh_i · (1 - polarity_i)
    theta_i    = 1 - coh_i
    uncertain_i = 0

This reuses polarity — still a structured metadata field — but it's
derived from the extractor, not a hand-tuned DS source. The key
*learned* quantity, coherence, is entirely from the JEPA predictor.

Run two variants
----------------
1. DS heuristic teacher (baseline, 6 hand-coded sources)
2. Intrinsic reward from JEPA prediction error — zero hand-coded
   mass sources; the reward signal is the predictor's surprise.

Both use the same GNN + sheaf layer + same self-generated query set.
Measure accuracy + θ calibration on identical queries.
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
from cognition.dempster_shafer import MassFunction


CORPUS = [
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


# ═════════════════════════════════════════════════════════════════════
# Relation predictor (same as Temporal-JEPA)
# ═════════════════════════════════════════════════════════════════════

class RelationPredictor(nn.Module):
    def __init__(self, hidden_dim: int, n_relations: int,
                 proj_dim: int = 128):
        super().__init__()
        self.rel_embed = nn.Embedding(n_relations, hidden_dim)
        self.trunk = nn.Sequential(
            nn.Linear(2 * hidden_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, hidden_dim),
        )

    def forward(self, h_src: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
        rel_e = self.rel_embed(rel)
        return self.trunk(torch.cat([h_src, rel_e], dim=-1))


@torch.no_grad()
def ema(student, teacher, tau: float):
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(tau).add_(ps.data, alpha=1.0 - tau)


def forward_gnn(layers, graph):
    h = graph.node_features
    for layer in layers:
        h = layer(h, graph.edge_index, graph.edge_types,
                  graph.edge_weights, graph.situation_features)
    return h


# ═════════════════════════════════════════════════════════════════════
# Pretrain the JEPA world model (shared by both variants)
# ═════════════════════════════════════════════════════════════════════

def pretrain_jepa(reasoner, graph, epochs: int = 50, lr: float = 3e-3,
                  tau: float = 0.996, verbose: bool = False):
    teacher = copy.deepcopy(reasoner.layers)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()

    predictor = RelationPredictor(
        hidden_dim=reasoner.hidden_dim,
        n_relations=len(REL_TO_IDX),
    )

    src = graph.edge_index[0]
    tgt = graph.edge_index[1]
    rel = graph.edge_types

    params = list(reasoner.layers.parameters()) + list(predictor.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    for ep in range(epochs):
        reasoner.train()
        opt.zero_grad()
        h_s = forward_gnn(reasoner.layers, graph)
        with torch.no_grad():
            h_t = forward_gnn(teacher, graph)
        preds = predictor(h_s[src], rel)
        targets = h_t[tgt]
        loss = F.mse_loss(preds, targets)
        # VICReg variance on infon subspace (anti-collapse)
        infon_idx = torch.tensor(graph.infon_indices, dtype=torch.long)
        z = h_s[infon_idx]
        std = torch.sqrt(z.var(dim=0) + 1e-4)
        var_term = F.relu(1.0 - std).mean()
        loss = loss + 1.0 * var_term
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        ema(reasoner.layers, teacher, tau)

    reasoner.eval()
    return predictor, teacher


# ═════════════════════════════════════════════════════════════════════
# The key new function: derive per-infon mass from JEPA prediction error
# ═════════════════════════════════════════════════════════════════════

def intrinsic_masses(reasoner, predictor, teacher, graph,
                     sigma: float = None,
                     verbose: bool = False) -> torch.Tensor:
    """For each infon node, compute an intrinsic mass function derived
    purely from the world model's predictive coherence."""
    reasoner.eval()
    with torch.no_grad():
        h_s = forward_gnn(reasoner.layers, graph)
        h_t = forward_gnn(teacher, graph)
        src = graph.edge_index[0]
        tgt = graph.edge_index[1]
        rel = graph.edge_types
        if src.numel() == 0:
            n = len(graph.infon_map)
            return torch.tensor([[0.0, 0.0, 0.0, 1.0]] * n)

        preds = predictor(h_s[src], rel)
        targets = h_t[tgt]
        per_edge_err = (preds - targets).pow(2).sum(-1)   # (E,)

        # Group by target infon node
        infon_node_set = set(graph.infon_map.values())
        per_infon_err: dict[int, list[torch.Tensor]] = {}
        for e in range(src.shape[0]):
            t = int(tgt[e])
            if t in infon_node_set:
                per_infon_err.setdefault(t, []).append(per_edge_err[e])

        errs = []
        for iid in graph.infon_map:
            idx = graph.infon_map[iid]
            if idx in per_infon_err:
                errs.append(float(torch.stack(per_infon_err[idx]).mean()))
            else:
                errs.append(float("nan"))

        err_tensor = torch.tensor(errs, dtype=torch.float32)
        # Calibrate σ from the distribution of observed errors
        valid = err_tensor[~torch.isnan(err_tensor)]
        if sigma is None:
            sigma = float(valid.median()) + 1e-4
        if verbose:
            print(f"    median err={valid.median():.4f}  "
                  f"q25={valid.quantile(0.25):.4f}  "
                  f"q75={valid.quantile(0.75):.4f}  σ={sigma:.4f}")

        # Coherence: high when prediction error is low
        coh = torch.exp(-err_tensor / sigma)
        coh = torch.where(torch.isnan(coh), torch.tensor(0.0), coh)

        # Polarity per infon
        polarities = []
        for iid in graph.infon_map:
            inf = reasoner.store.get_infon(iid)
            polarities.append(1.0 if (inf is None or inf.polarity) else 0.0)
        pol = torch.tensor(polarities, dtype=torch.float32)

        supports = coh * pol
        refutes  = coh * (1.0 - pol)
        theta    = 1.0 - coh
        uncert   = torch.zeros_like(theta)

        mass = torch.stack([supports, refutes, uncert, theta], dim=-1)
        # Re-normalize (defensive)
        mass = mass / mass.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return mass


# ═════════════════════════════════════════════════════════════════════
# DS heuristic masses (for the baseline variant)
# ═════════════════════════════════════════════════════════════════════

def ds_heuristic_masses(reasoner, graph) -> torch.Tensor:
    from cognition.dempster_shafer import (
        mass_from_polarity, mass_from_triple_alignment,
        mass_from_anchor_distance, mass_from_confidence,
        mass_from_evidentiality, mass_from_modality,
        combine_multiple,
    )
    out = []
    for iid in graph.infon_map:
        inf = reasoner.store.get_infon(iid)
        if inf is None:
            out.append([0.25, 0.25, 0.25, 0.25])
            continue
        claim = {inf.subject: inf.confidence,
                 inf.predicate: inf.confidence,
                 inf.object: inf.confidence}
        sources = [
            mass_from_polarity(inf),
            mass_from_triple_alignment(claim, inf, reasoner.schema.types),
            mass_from_anchor_distance(claim, inf, reasoner.schema.types),
            mass_from_confidence(inf),
            mass_from_evidentiality(inf),
            mass_from_modality(inf),
        ]
        m = combine_multiple(sources)
        out.append([m.supports, m.refutes, m.uncertain, m.theta])
    return torch.tensor(out, dtype=torch.float32)


# ═════════════════════════════════════════════════════════════════════
# Train the mass readout against a given target
# ═════════════════════════════════════════════════════════════════════

def train_readout(reasoner, graph, target: torch.Tensor,
                  epochs: int = 30, lr: float = 1e-3,
                  verbose: bool = False):
    infon_idx = torch.tensor(
        [graph.infon_map[iid] for iid in graph.infon_map],
        dtype=torch.long,
    )
    opt = torch.optim.Adam(reasoner.mass_readout.parameters(), lr=lr)
    for ep in range(epochs):
        reasoner.train()
        opt.zero_grad()
        h = forward_gnn(reasoner.layers, graph)
        pred = reasoner.mass_readout(h[infon_idx])
        loss = F.kl_div(pred.log().clamp(min=-20),
                        target, reduction="batchmean")
        loss.backward()
        opt.step()
        if verbose and (ep + 1) % 10 == 0:
            print(f"      readout epoch {ep+1}  KL={loss.item():.4f}")
    reasoner.eval()


# ═════════════════════════════════════════════════════════════════════
# Self-generated query set (same as self_supervised_full.py)
# ═════════════════════════════════════════════════════════════════════

def generate_queries(cog, max_supports: int = 8, min_conf: float = 0.05,
                     seed: int = 0):
    import re
    rng = random.Random(seed)
    infons = cog.store.query_infons(limit=200)
    good = [i for i in infons if i.confidence >= min_conf]
    good.sort(key=lambda x: -x.confidence)

    actors = [n for n in cog.schema.names
              if cog.schema.types.get(n) == "actor"]
    feats = [n for n in cog.schema.names
             if cog.schema.types.get(n) in ("feature", "market")]

    queries, seen = [], set()
    for inf in good:
        if len(queries) >= max_supports: break
        key = (inf.subject, inf.predicate, inf.object)
        if key in seen: continue
        seen.add(key)
        queries.append((f"Did {inf.subject} {inf.predicate} {inf.object}?",
                        "SUPPORTS"))

    nei = []
    for q, _ in list(queries):
        m = re.match(r"Did (\S+) (\S+) (.+)\?", q)
        if not m: continue
        s, p, o = m.groups()
        which = rng.choice(["subj", "obj"])
        if which == "subj" and actors:
            cands = [a for a in actors if a != s]
            if not cands: continue
            s2 = rng.choice(cands); cand = (s2, p, o)
        elif which == "obj" and feats:
            cands = [a for a in feats if a != o]
            if not cands: continue
            o2 = rng.choice(cands); cand = (s, p, o2)
        else: continue
        if cand in seen: continue
        nei.append((f"Did {cand[0]} {cand[1]} {cand[2]}?",
                    "NOT_ENOUGH_INFO"))
    queries.extend(nei[:max_supports])
    return queries


def evaluate_on(reasoner, queries):
    reasoner.eval()
    if not queries:
        return {"accuracy": float("nan"),
                "theta_supports": float("nan"),
                "theta_nei": float("nan")}
    correct = 0; tS, tN = [], []
    for q, gold in queries:
        r = reasoner.reason(q)
        if r.verdict == gold: correct += 1
        (tS if gold == "SUPPORTS" else tN).append(r.mass.theta)
    return {
        "accuracy": correct / len(queries),
        "theta_supports": sum(tS) / max(len(tS), 1),
        "theta_nei":      sum(tN) / max(len(tN), 1),
    }


# ═════════════════════════════════════════════════════════════════════
# Driver
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
    for d in CORPUS:
        cog.ingest([d])
    cog.consolidate()
    return cog


def run_variant(label: str, target_source: str, seed: int = 0):
    print(f"\n== Variant: {label} ==")
    torch.manual_seed(seed)
    with tempfile.TemporaryDirectory() as tmp:
        cog = build_cog(tmp)
        reasoner = HypergraphReasoner(
            cog.store, cog.encoder, cog.schema,
            hidden_dim=32, n_layers=2, use_sheaf=True,
        )
        graph = reasoner.builder.build(feature_dim=32)
        # Warm-start the GNN (no heuristic) — just L_F regularizer
        reasoner.fit(graph=graph, epochs=10,
                     sheaf_weight=0.0, laplacian_weight=0.1,
                     verbose=False)

        queries = generate_queries(cog, seed=seed)
        print(f"  infons={cog.stats()['infon_count']}  "
              f"queries: {sum(1 for _,g in queries if g=='SUPPORTS')}S "
              f"+ {sum(1 for _,g in queries if g=='NOT_ENOUGH_INFO')}N")

        # Pretrain the JEPA world model (shared, no labels)
        print("  pretraining JEPA world model...")
        predictor, teacher = pretrain_jepa(reasoner, graph,
                                           epochs=50, verbose=False)

        # Build the training target per variant
        if target_source == "ds":
            print("  target = DS heuristic masses (six hand-coded sources)")
            target = ds_heuristic_masses(reasoner, graph)
        elif target_source == "jepa_intrinsic":
            print("  target = intrinsic mass from JEPA prediction error")
            target = intrinsic_masses(
                reasoner, predictor, teacher, graph, verbose=True,
            )
        else:
            raise ValueError(target_source)

        # Fit the mass readout to that target
        train_readout(reasoner, graph, target, epochs=30)

        # Evaluate on the self-generated query set
        ev = evaluate_on(reasoner, queries)
        print(f"  accuracy={ev['accuracy']:.0%}  "
              f"θ_S={ev['theta_supports']:.2f}  "
              f"θ_N={ev['theta_nei']:.2f}")

        # Diagnostic: show the target distribution
        valid = target
        print(f"  target stats:  "
              f"S̄={valid[:, 0].mean():.2f}  R̄={valid[:, 1].mean():.2f}  "
              f"Ū={valid[:, 2].mean():.2f}  θ̄={valid[:, 3].mean():.2f}")

        cog.close()
        return {"label": label, **ev,
                "target_theta_mean": float(valid[:, 3].mean()),
                "target_supports_mean": float(valid[:, 0].mean())}


def main():
    print("Intrinsic-Reward MDP — reward = JEPA predictor surprise")
    print("=" * 72)

    results = []
    results.append(run_variant(
        "A. baseline — DS heuristic teacher",
        target_source="ds",
        seed=0,
    ))
    results.append(run_variant(
        "B. intrinsic — mass from JEPA prediction error",
        target_source="jepa_intrinsic",
        seed=0,
    ))

    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"  {'variant':<52s} {'acc':>6s} {'θ_S':>6s} {'θ_N':>6s}  {'tgt_θ̄':>7s}")
    print("  " + "-" * 78)
    for r in results:
        print(f"  {r['label'][:52]:<52s} "
              f"{r['accuracy']:>6.0%} "
              f"{r['theta_supports']:>6.2f} "
              f"{r['theta_nei']:>6.2f}  "
              f"{r['target_theta_mean']:>7.2f}")


if __name__ == "__main__":
    main()

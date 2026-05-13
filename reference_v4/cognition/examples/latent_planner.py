"""Latent-space planner for reason().

We now have a world model `P_r(h_s) → ĥ_t` trained with Temporal-JEPA.
The question: does planning through that model (multi-step rollout)
verify claims better than one-shot retrieval?

Setup
-----
Given a claim Q = (subject, predicate, object):
  1. Encode Q with SPLADE → activated anchors; take subject/object anchor
     embeddings as start/goal in latent space.
  2. From the subject's latent, run *k-step beam search* through the
     predictor: at each step, consider each relation r and expand the
     top-B successor states.
  3. For each trajectory of length ≤ K, compute:
        chain_coherence  = exp(−Σ per-step prediction error)
        endpoint_match   = cosine(final_state, goal_embedding)
        trajectory_score = chain_coherence · (1 + endpoint_match) / 2
  4. The best trajectory's score becomes a SUPPORTS-mass; the supremum
     over alternatives gives us a contrastive refutes-mass; the residual
     is θ.

The key insight: if the world model is a faithful MDP of the hypergraph,
claims whose subject *can reach* their object via high-coherence edges
should score high. Claims where no coherent path exists should get high
θ — which is exactly the calibration we want.

Compared to the one-shot reasoner
---------------------------------
`reason()` filters infons by role-wise overlap then combines DS masses.
It fundamentally *retrieves* — it can't answer "has this chain of events
been built up" without an explicit NEXT walk.

The latent planner *plans*. Given (Toyota, acquires, CATL), even if no
single infon has that triple, the planner may still find a coherent
latent trajectory {Toyota → (invests) → battery → (acquired_by_? ) → CATL}
that lands close to the goal. That's a world model, not a look-up.

Metrics
-------
- accuracy on the same self-generated query set from intrinsic_reward.py
- θ calibration (should still say "I don't know" on impossible trajectories)
- comparison against one-shot `reason()` on the same queries
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import re
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F

from cognition import Cognition, CognitionConfig
from cognition.logic import HypergraphReasoner, REL_TO_IDX, NUM_RELATIONS


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
# World-model predictor (from Temporal-JEPA)
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

    def forward(self, h_src, rel):
        rel_e = self.rel_embed(rel)
        return self.trunk(torch.cat([h_src, rel_e], dim=-1))


@torch.no_grad()
def ema(s, t, tau):
    for ps, pt in zip(s.parameters(), t.parameters()):
        pt.data.mul_(tau).add_(ps.data, alpha=1.0 - tau)


def forward_gnn(layers, graph):
    h = graph.node_features
    for layer in layers:
        h = layer(h, graph.edge_index, graph.edge_types,
                  graph.edge_weights, graph.situation_features)
    return h


def pretrain_jepa(reasoner, graph, epochs=60, lr=3e-3, tau=0.996):
    teacher = copy.deepcopy(reasoner.layers)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    predictor = RelationPredictor(reasoner.hidden_dim, NUM_RELATIONS)
    params = list(reasoner.layers.parameters()) + list(predictor.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    src = graph.edge_index[0]
    tgt = graph.edge_index[1]
    rel = graph.edge_types

    for ep in range(epochs):
        reasoner.train()
        opt.zero_grad()
        h_s = forward_gnn(reasoner.layers, graph)
        with torch.no_grad():
            h_t = forward_gnn(teacher, graph)
        preds = predictor(h_s[src], rel)
        targets = h_t[tgt]
        loss = F.mse_loss(preds, targets)
        # VICReg variance
        infon_idx = torch.tensor(graph.infon_indices, dtype=torch.long)
        z = h_s[infon_idx]
        loss = loss + 1.0 * F.relu(1.0 - torch.sqrt(z.var(dim=0) + 1e-4)).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        ema(reasoner.layers, teacher, tau)
    reasoner.eval()
    return predictor, teacher


# ═════════════════════════════════════════════════════════════════════
# Latent-space beam planner
# ═════════════════════════════════════════════════════════════════════

class LatentPlanner:
    """Plans trajectories in latent embedding space using the trained
    relation predictor as the transition model."""

    def __init__(self, reasoner, predictor, teacher, graph,
                 max_depth: int = 3, beam_size: int = 8,
                 transition_relations: list[int] | None = None):
        self.reasoner = reasoner
        self.predictor = predictor
        self.teacher = teacher
        self.graph = graph
        self.max_depth = max_depth
        self.beam_size = beam_size
        # Only follow transition-type edges during planning (not spokes).
        if transition_relations is None:
            transition_relations = [
                REL_TO_IDX["NEXT"],
                REL_TO_IDX["CAUSES"],
                REL_TO_IDX["ENTAILS"],
                REL_TO_IDX["INITIATES"],
                REL_TO_IDX["ASSERTS"],
                REL_TO_IDX["TARGETS"],
            ]
        self.relations = transition_relations

        # Precompute the teacher-embedding bank (immutable during planning)
        with torch.no_grad():
            self.h_teacher = forward_gnn(teacher, graph)
            self.h_student = forward_gnn(reasoner.layers, graph)

    def anchor_embedding(self, anchor_name: str) -> torch.Tensor | None:
        """Return the student embedding for an anchor node."""
        idx = self.graph.anchor_map.get(anchor_name)
        if idx is None:
            return None
        return self.h_student[idx]

    def closest_node(self, h: torch.Tensor,
                     restrict_to: set[int] | None = None) -> tuple[int, float]:
        """Find the teacher-bank node nearest `h` in cosine similarity.
        Optionally restrict to a subset of node indices."""
        bank = self.h_teacher
        if restrict_to is not None:
            idxs = torch.tensor(sorted(restrict_to), dtype=torch.long)
            bank = bank[idxs]
        else:
            idxs = torch.arange(bank.shape[0])
        sim = F.cosine_similarity(
            F.normalize(h.unsqueeze(0), dim=-1),
            F.normalize(bank, dim=-1),
            dim=-1,
        )
        k = int(sim.argmax())
        return int(idxs[k]), float(sim[k])

    def plan(self, start_anchor: str, goal_anchor: str
             ) -> tuple[float, list[dict]]:
        """Return (best_score, [trajectory_entries]) for the top trajectory
        from start anchor to goal anchor in latent space."""
        start = self.anchor_embedding(start_anchor)
        goal = self.anchor_embedding(goal_anchor)
        if start is None or goal is None:
            return 0.0, []

        # Beam entries: (score_log, cumulative_error, current_latent, path)
        # path is a list of (rel_idx, predicted_latent) tuples.
        beam = [(0.0, start, [])]
        best_global = (0.0, [])

        for step in range(self.max_depth):
            expansions = []
            for (cum_err, h_cur, path) in beam:
                # Every relation defines one candidate successor
                for r in self.relations:
                    r_t = torch.tensor([r], dtype=torch.long)
                    h_next = self.predictor(h_cur.unsqueeze(0), r_t).squeeze(0)
                    # Prediction error vs nearest teacher node
                    nn_idx, sim_nn = self.closest_node(h_next)
                    # local error = 1 - cosine(h_next, nearest teacher)
                    err_local = (1.0 - sim_nn)
                    cum_err2 = cum_err + err_local
                    expansions.append((cum_err2, h_next, path + [(r, nn_idx)]))

            # Keep top-B by cumulative error (ascending)
            expansions.sort(key=lambda x: x[0])
            beam = expansions[:self.beam_size]

            # Score every current beam trajectory against the goal
            for (cum_err, h_cur, path) in beam:
                coherence = math.exp(-cum_err / max(len(path), 1))
                endpoint_sim = F.cosine_similarity(
                    h_cur.unsqueeze(0), goal.unsqueeze(0), dim=-1,
                ).item()
                endpoint_match = max(0.0, endpoint_sim)
                score = coherence * endpoint_match
                if score > best_global[0]:
                    best_global = (score, path.copy())

        return best_global


# ═════════════════════════════════════════════════════════════════════
# Planner-based verdict
# ═════════════════════════════════════════════════════════════════════

def planner_verdict(planner: LatentPlanner,
                    subject: str, predicate: str, obj: str,
                    supports_threshold: float = 0.25,
                    nei_threshold: float = 0.08,
                    anchor_names: list[str] | None = None,
                    ) -> tuple[str, dict]:
    """Score (subj, pred, obj) via latent planning, return a verdict + mass.

    Strategy:
      - Forward plan: score(subj → obj)
      - Contrastive plan: best score(subj → alt_obj) for a few random
        other objects. Used to compute refute-mass.
    """
    score, path = planner.plan(subject, obj)

    # Contrastive — how well can the planner reach an arbitrary other object
    # from the same subject? This gives us a baseline.
    contrastive_scores = []
    if anchor_names:
        candidates = [a for a in anchor_names if a != obj][:6]
        for alt in candidates:
            s_alt, _ = planner.plan(subject, alt)
            contrastive_scores.append(s_alt)
    if contrastive_scores:
        contrastive_max = max(contrastive_scores)
    else:
        contrastive_max = 0.0

    # Calibrate DS mass
    # supports: how well the true path works, above the null
    supports = max(0.0, score - contrastive_max * 0.5)
    # refutes: how well an alternative outperformed the true path
    refutes = max(0.0, contrastive_max - score) * 0.5
    # theta: residual ignorance if neither stands out
    certainty = min(1.0, supports + refutes)
    theta = 1.0 - certainty
    uncertain = 0.0

    total = supports + refutes + uncertain + theta
    supports /= total; refutes /= total; theta /= total

    if theta > 1.0 - nei_threshold:
        verdict = "NOT_ENOUGH_INFO"
    elif supports > supports_threshold and supports > refutes:
        verdict = "SUPPORTS"
    elif refutes > supports:
        verdict = "REFUTES"
    else:
        verdict = "NOT_ENOUGH_INFO"

    return verdict, {
        "supports": supports,
        "refutes": refutes,
        "theta": theta,
        "raw_score": score,
        "contrastive": contrastive_max,
    }


# ═════════════════════════════════════════════════════════════════════
# Self-generated query set (same as before)
# ═════════════════════════════════════════════════════════════════════

def generate_queries(cog, max_supports=8, min_conf=0.05, seed=0):
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
        if len(queries) >= max_supports:
            break
        key = (inf.subject, inf.predicate, inf.object)
        if key in seen: continue
        seen.add(key)
        queries.append({"q": f"Did {inf.subject} {inf.predicate} {inf.object}?",
                        "triple": (inf.subject, inf.predicate, inf.object),
                        "gold": "SUPPORTS"})
    nei = []
    for item in list(queries):
        s, p, o = item["triple"]
        which = rng.choice(["subj", "obj"])
        if which == "subj" and actors:
            c = [a for a in actors if a != s]
            if not c: continue
            s2 = rng.choice(c)
            triple = (s2, p, o)
        elif which == "obj" and feats:
            c = [a for a in feats if a != o]
            if not c: continue
            o2 = rng.choice(c)
            triple = (s, p, o2)
        else: continue
        if triple in seen: continue
        seen.add(triple)
        nei.append({"q": f"Did {triple[0]} {triple[1]} {triple[2]}?",
                    "triple": triple,
                    "gold": "NOT_ENOUGH_INFO"})
    queries.extend(nei[:max_supports])
    return queries


# ═════════════════════════════════════════════════════════════════════
# Driver
# ═════════════════════════════════════════════════════════════════════

def build_cog(tmp):
    schema_path = os.path.join(tmp, "schema.json")
    with open(schema_path, "w") as f:
        json.dump(SCHEMA, f)
    cog = Cognition(CognitionConfig(
        schema_path=schema_path,
        db_path=os.path.join(tmp, "cog.db"),
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


def main():
    print("Latent-Space Planner — verify claims via k-step rollout")
    print("=" * 72)
    torch.manual_seed(0)

    with tempfile.TemporaryDirectory() as tmp:
        cog = build_cog(tmp)
        reasoner = HypergraphReasoner(
            cog.store, cog.encoder, cog.schema,
            hidden_dim=32, n_layers=2, use_sheaf=True,
        )
        graph = reasoner.builder.build(feature_dim=32)
        reasoner.fit(graph=graph, epochs=10,
                     sheaf_weight=0.0, laplacian_weight=0.1,
                     verbose=False)

        queries = generate_queries(cog, seed=0)
        print(f"  infons={cog.stats()['infon_count']}  "
              f"queries: {sum(1 for q in queries if q['gold']=='SUPPORTS')}S "
              f"+ {sum(1 for q in queries if q['gold']=='NOT_ENOUGH_INFO')}N")

        print("  pretraining JEPA world model...")
        predictor, teacher = pretrain_jepa(reasoner, graph, epochs=60)

        # ── Variant A: one-shot retrieval reasoner ──
        print("\n== Variant A: one-shot reason() (retrieval + Dempster combine) ==")
        correct_a = 0; tS_a, tN_a = [], []
        for item in queries:
            r = reasoner.reason(item["q"])
            if r.verdict == item["gold"]:
                correct_a += 1
            (tS_a if item["gold"] == "SUPPORTS" else tN_a).append(r.mass.theta)
        print(f"  accuracy={correct_a}/{len(queries)} = {correct_a/len(queries):.0%}")
        print(f"  θ_S={sum(tS_a)/len(tS_a):.2f}  θ_N={sum(tN_a)/len(tN_a):.2f}")

        # ── Variant B: latent-space planner ──
        print("\n== Variant B: latent planner (beam search through predictor) ==")
        for (max_depth, beam) in [(2, 6), (3, 8), (4, 10)]:
            planner = LatentPlanner(
                reasoner, predictor, teacher, graph,
                max_depth=max_depth, beam_size=beam,
            )
            anchor_names = list(cog.schema.names)
            correct_b = 0; tS_b, tN_b = [], []
            per_query = []
            for item in queries:
                s, p, o = item["triple"]
                verdict, mass = planner_verdict(
                    planner, s, p, o, anchor_names=anchor_names,
                )
                per_query.append((item, verdict, mass))
                if verdict == item["gold"]:
                    correct_b += 1
                (tS_b if item["gold"] == "SUPPORTS" else tN_b).append(mass["theta"])
            print(f"  depth={max_depth} beam={beam}:  "
                  f"acc={correct_b}/{len(queries)} = {correct_b/len(queries):.0%}  "
                  f"θ_S={sum(tS_b)/len(tS_b):.2f}  "
                  f"θ_N={sum(tN_b)/len(tN_b):.2f}")

        # Detailed view for the best planner setting (last iteration)
        print("\n  Per-query detail (depth=4, beam=10):")
        for item, verdict, mass in per_query:
            s, p, o = item["triple"]
            ok = "✓" if verdict == item["gold"] else "✗"
            print(f"    {ok} ({s}, {p}, {o})  "
                  f"gold={item['gold']:<16s}  pred={verdict:<16s}  "
                  f"S={mass['supports']:.2f}  θ={mass['theta']:.2f}  "
                  f"raw={mass['raw_score']:.3f}  contrast={mass['contrastive']:.3f}")

        cog.close()


if __name__ == "__main__":
    main()

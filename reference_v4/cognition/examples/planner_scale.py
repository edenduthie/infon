"""Planner vs retrieval — bigger corpus test.

The hand-written 12-doc run told us planning couldn't beat retrieval
because hub anchors (battery) made every path score similar. This
script generates a larger, more heterogeneous corpus via synth.py
and repeats the head-to-head with one addition: the contrastive
score is *per-object* (measured against random objects of the same
type) rather than over all anchors. That's the density-normalization
fix the small run called for.
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
from cognition.logic import HypergraphReasoner, REL_TO_IDX, NUM_RELATIONS
from cognition.synth import (
    Schema as SynthSchema, generate_corpus, DEFAULT_TEMPLATES,
)


# ═════════════════════════════════════════════════════════════════════
# Schema (richer than 12-doc version — more actors, more markets)
# ═════════════════════════════════════════════════════════════════════

SYNTH_SCHEMA = SynthSchema(
    actors=["Toyota", "Honda", "Tesla", "Panasonic", "CATL",
            "Ford", "BMW", "LG", "Samsung", "Hyundai"],
    relations=["invests", "partners", "produces", "expands",
               "delays", "acquires"],
    features=["batteries", "EVs", "factories", "supply",
              "chips", "software"],
    markets=["Japan", "China", "America", "Korea", "Europe"],
)

def schema_to_anchor_dict(s: SynthSchema) -> dict:
    out = {}
    for a in s.actors:
        out[a.lower()] = {"type": "actor", "tokens": [a.lower()]}
    for r in s.relations:
        # give each verb the usual inflections
        out[r] = {"type": "relation",
                  "tokens": [r, r.rstrip("s"), r.rstrip("s") + "ed",
                             r.rstrip("s") + "ing"]}
    for f in s.features:
        out[f] = {"type": "feature",
                  "tokens": [f, f.rstrip("s")]}
    for m in s.markets:
        out[m.lower()] = {"type": "market", "tokens": [m.lower()]}
    return out


# ═════════════════════════════════════════════════════════════════════
# Predictor + JEPA (same as before)
# ═════════════════════════════════════════════════════════════════════

class RelationPredictor(nn.Module):
    def __init__(self, hidden_dim, n_relations, proj_dim=128):
        super().__init__()
        self.rel_embed = nn.Embedding(n_relations, hidden_dim)
        self.trunk = nn.Sequential(
            nn.Linear(2 * hidden_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, hidden_dim),
        )
    def forward(self, h_src, rel):
        return self.trunk(torch.cat([h_src, self.rel_embed(rel)], dim=-1))


@torch.no_grad()
def ema(s, t, tau):
    for ps, pt in zip(s.parameters(), t.parameters()):
        pt.data.mul_(tau).add_(ps.data, alpha=1.0 - tau)


def forward_gnn(layers, g):
    h = g.node_features
    for l in layers:
        h = l(h, g.edge_index, g.edge_types, g.edge_weights,
              g.situation_features)
    return h


def pretrain_jepa(reasoner, graph, epochs=50, lr=3e-3, tau=0.996):
    teacher = copy.deepcopy(reasoner.layers)
    for p in teacher.parameters(): p.requires_grad_(False)
    teacher.eval()
    pred = RelationPredictor(reasoner.hidden_dim, NUM_RELATIONS)
    params = list(reasoner.layers.parameters()) + list(pred.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    src = graph.edge_index[0]
    tgt = graph.edge_index[1]
    rel = graph.edge_types
    infon_idx = torch.tensor(graph.infon_indices, dtype=torch.long)

    for ep in range(epochs):
        reasoner.train()
        opt.zero_grad()
        h_s = forward_gnn(reasoner.layers, graph)
        with torch.no_grad():
            h_t = forward_gnn(teacher, graph)
        preds = pred(h_s[src], rel)
        loss = F.mse_loss(preds, h_t[tgt])
        z = h_s[infon_idx]
        loss = loss + 1.0 * F.relu(1.0 - torch.sqrt(z.var(dim=0) + 1e-4)).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        ema(reasoner.layers, teacher, tau)
    reasoner.eval()
    return pred, teacher


# ═════════════════════════════════════════════════════════════════════
# Planner — same beam search, but contrastive uses type-matched sample
# ═════════════════════════════════════════════════════════════════════

class LatentPlanner:
    def __init__(self, reasoner, predictor, teacher, graph,
                 max_depth=3, beam_size=8):
        self.reasoner = reasoner
        self.predictor = predictor
        self.teacher = teacher
        self.graph = graph
        self.max_depth = max_depth
        self.beam_size = beam_size
        self.relations = [
            REL_TO_IDX["NEXT"], REL_TO_IDX["CAUSES"], REL_TO_IDX["ENTAILS"],
            REL_TO_IDX["INITIATES"], REL_TO_IDX["ASSERTS"],
            REL_TO_IDX["TARGETS"],
        ]
        with torch.no_grad():
            self.h_teacher = forward_gnn(teacher, graph)
            self.h_student = forward_gnn(reasoner.layers, graph)

    def anchor_emb(self, name):
        idx = self.graph.anchor_map.get(name)
        return None if idx is None else self.h_student[idx]

    def closest_sim(self, h):
        sim = F.cosine_similarity(
            F.normalize(h.unsqueeze(0), dim=-1),
            F.normalize(self.h_teacher, dim=-1),
            dim=-1,
        )
        return float(sim.max())

    def plan(self, start_anchor, goal_anchor):
        start = self.anchor_emb(start_anchor)
        goal = self.anchor_emb(goal_anchor)
        if start is None or goal is None:
            return 0.0
        beam = [(0.0, start)]
        best = 0.0
        for step in range(self.max_depth):
            expand = []
            for cum_err, h_cur in beam:
                for r in self.relations:
                    r_t = torch.tensor([r], dtype=torch.long)
                    h_next = self.predictor(h_cur.unsqueeze(0), r_t).squeeze(0)
                    sim_nn = self.closest_sim(h_next)
                    expand.append((cum_err + (1.0 - sim_nn), h_next))
            expand.sort(key=lambda x: x[0])
            beam = expand[:self.beam_size]
            for cum_err, h_cur in beam:
                coh = math.exp(-cum_err / max(step + 1, 1))
                endpoint = F.cosine_similarity(
                    h_cur.unsqueeze(0), goal.unsqueeze(0), dim=-1,
                ).item()
                score = coh * max(0.0, endpoint)
                if score > best:
                    best = score
        return best


def planner_verdict(planner, schema, subject, predicate, obj,
                    types: dict, seed=0, n_contrast=8):
    """Type-matched contrastive scoring: the contrast is computed over
    other objects of the *same type* as `obj`, not over all anchors.
    This normalizes for hub density."""
    rng = random.Random(hash((subject, predicate, obj, seed)) & 0xffff)
    obj_type = types.get(obj, "feature")
    same_type = [n for n in schema.names
                 if types.get(n) == obj_type and n != obj]
    contrast_sample = rng.sample(same_type, min(n_contrast, len(same_type)))

    raw = planner.plan(subject, obj)
    contrasts = [planner.plan(subject, c) for c in contrast_sample]
    contrast_mean = sum(contrasts) / max(len(contrasts), 1)
    contrast_max = max(contrasts) if contrasts else 0.0

    # Supports: raw must exceed the *mean* of same-type alternatives
    supports = max(0.0, raw - contrast_mean)
    refutes = max(0.0, contrast_max - raw) * 0.5
    certainty = min(1.0, supports + refutes)
    theta = 1.0 - certainty

    total = supports + refutes + theta + 1e-9
    s, r, t = supports / total, refutes / total, theta / total

    if t > 0.85:
        verdict = "NOT_ENOUGH_INFO"
    elif s > 0.15 and s > r:
        verdict = "SUPPORTS"
    elif r > s:
        verdict = "REFUTES"
    else:
        verdict = "NOT_ENOUGH_INFO"
    return verdict, {"supports": s, "refutes": r, "theta": t,
                     "raw": raw, "contrast_mean": contrast_mean,
                     "contrast_max": contrast_max}


# ═════════════════════════════════════════════════════════════════════
# Queries generated from the store
# ═════════════════════════════════════════════════════════════════════

def generate_queries(cog, n_each=15, min_conf=0.05, seed=0):
    rng = random.Random(seed)
    infons = cog.store.query_infons(limit=500)
    good = [i for i in infons if i.confidence >= min_conf]
    good.sort(key=lambda x: -x.confidence)
    types = cog.schema.types
    actors = [n for n in cog.schema.names if types.get(n) == "actor"]
    feats = [n for n in cog.schema.names
             if types.get(n) in ("feature", "market")]

    queries, seen = [], set()
    for inf in good:
        if len(queries) >= n_each:
            break
        k = (inf.subject, inf.predicate, inf.object)
        if k in seen: continue
        seen.add(k)
        queries.append({"q": f"Did {inf.subject} {inf.predicate} {inf.object}?",
                        "triple": k, "gold": "SUPPORTS"})

    nei = []
    for item in list(queries):
        s, p, o = item["triple"]
        if rng.random() < 0.5 and actors:
            c = [a for a in actors if a != s]
            if not c: continue
            triple = (rng.choice(c), p, o)
        else:
            c = [a for a in feats if a != o]
            if not c: continue
            triple = (s, p, rng.choice(c))
        if triple in seen: continue
        seen.add(triple)
        nei.append({"q": f"Did {triple[0]} {triple[1]} {triple[2]}?",
                    "triple": triple, "gold": "NOT_ENOUGH_INFO"})
    queries.extend(nei[:n_each])
    return queries


# ═════════════════════════════════════════════════════════════════════
# Build the corpus
# ═════════════════════════════════════════════════════════════════════

def build_corpus_docs(n_sentences: int, seed: int = 0) -> list[dict]:
    """Synthesize n sentences and pack them into documents (~3 per doc)."""
    examples = generate_corpus(SYNTH_SCHEMA, n=n_sentences, seed=seed,
                               templates=DEFAULT_TEMPLATES)
    docs = []
    rng = random.Random(seed)
    # 3 sentences per doc on average
    group_size = 3
    for i in range(0, len(examples), group_size):
        bunch = examples[i:i + group_size]
        text = " ".join(ex.sentence for ex in bunch)
        docs.append({
            "id": f"synth-{i:04d}",
            "timestamp": f"2024-{1 + (i // 30) % 12:02d}-{1 + (i % 28):02d}",
            "text": text,
        })
    return docs


def build_cog(tmp: str, n_sentences: int, seed: int):
    schema_path = os.path.join(tmp, "schema.json")
    schema_dict = schema_to_anchor_dict(SYNTH_SCHEMA)
    with open(schema_path, "w") as f:
        json.dump(schema_dict, f)
    cog = Cognition(CognitionConfig(
        schema_path=schema_path,
        db_path=os.path.join(tmp, "cog.db"),
        activation_threshold=0.2,
        min_confidence=0.02,
        top_k_per_role=3,
        quality_threshold=0.04,
        max_triples_per_sentence=2,
    ))
    docs = build_corpus_docs(n_sentences, seed=seed)
    # Ingest in batches of 20 to avoid one-big-transaction latency
    for i in range(0, len(docs), 20):
        cog.ingest(docs[i:i + 20])
    cog.consolidate()
    return cog, docs


def main():
    print("Scale experiment — planner vs retrieval on a bigger corpus")
    print("=" * 72)

    for n_sent in [60, 200]:
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as tmp:
            print(f"\n── n_sentences = {n_sent} ──")
            cog, docs = build_cog(tmp, n_sent, seed=0)
            stats = cog.stats()
            print(f"  docs={len(docs)}  infons={stats['infon_count']}  "
                  f"anchors={stats['anchors']}")

            reasoner = HypergraphReasoner(
                cog.store, cog.encoder, cog.schema,
                hidden_dim=32, n_layers=2, use_sheaf=True,
            )
            graph = reasoner.builder.build(feature_dim=32, max_infons=500)
            reasoner.fit(graph=graph, epochs=10,
                         sheaf_weight=0.0, laplacian_weight=0.1,
                         verbose=False)

            queries = generate_queries(cog, n_each=15, seed=0)
            s_cnt = sum(1 for q in queries if q["gold"] == "SUPPORTS")
            n_cnt = sum(1 for q in queries if q["gold"] == "NOT_ENOUGH_INFO")
            print(f"  queries: {s_cnt}S + {n_cnt}NEI")

            # Variant A: retrieval
            cA = 0; tSa, tNa = [], []
            for item in queries:
                r = reasoner.reason(item["q"])
                if r.verdict == item["gold"]: cA += 1
                (tSa if item["gold"] == "SUPPORTS"
                 else tNa).append(r.mass.theta)
            aA = cA / len(queries)

            # Variant B: planner (depth 3, beam 8, type-matched contrast)
            print("  pretraining JEPA world model...")
            pred, teacher = pretrain_jepa(reasoner, graph, epochs=50)
            planner = LatentPlanner(reasoner, pred, teacher, graph,
                                    max_depth=3, beam_size=8)

            cB = 0; tSb, tNb = [], []
            for item in queries:
                s, p, o = item["triple"]
                v, m = planner_verdict(planner, cog.schema, s, p, o,
                                       cog.schema.types, seed=0,
                                       n_contrast=8)
                if v == item["gold"]: cB += 1
                (tSb if item["gold"] == "SUPPORTS"
                 else tNb).append(m["theta"])
            aB = cB / len(queries)

            # Variant C: ensemble — use planner only when retrieval returns
            # NEI with moderate-to-high θ (≥ 0.6)
            cC = 0; tSc, tNc = [], []
            for item in queries:
                r = reasoner.reason(item["q"])
                verdict = r.verdict
                theta = r.mass.theta
                if verdict == "NOT_ENOUGH_INFO" and theta >= 0.6:
                    s, p, o = item["triple"]
                    vp, mp = planner_verdict(planner, cog.schema, s, p, o,
                                             cog.schema.types, seed=0,
                                             n_contrast=8)
                    if vp != "NOT_ENOUGH_INFO":
                        # Only flip away from NEI when planner is confident
                        if mp["supports"] > 0.25:
                            verdict = vp
                            theta = mp["theta"]
                if verdict == item["gold"]: cC += 1
                (tSc if item["gold"] == "SUPPORTS"
                 else tNc).append(theta)
            aC = cC / len(queries)

            print(f"\n  Variant           acc     θ_S    θ_NEI")
            print(f"  ---------------- ------  ------  ------")
            print(f"  A. retrieval     {aA:>5.0%}  "
                  f"{sum(tSa)/max(len(tSa),1):>5.2f}  "
                  f"{sum(tNa)/max(len(tNa),1):>5.2f}")
            print(f"  B. planner       {aB:>5.0%}  "
                  f"{sum(tSb)/max(len(tSb),1):>5.2f}  "
                  f"{sum(tNb)/max(len(tNb),1):>5.2f}")
            print(f"  C. ensemble      {aC:>5.0%}  "
                  f"{sum(tSc)/max(len(tSc),1):>5.2f}  "
                  f"{sum(tNc)/max(len(tNc),1):>5.2f}")

            cog.close()


if __name__ == "__main__":
    main()

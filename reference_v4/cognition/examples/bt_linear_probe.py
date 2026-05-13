"""Barlow Twins + linear probe — isolated representation quality test.

Protocol
--------
Four variants, same corpus, same queries, same seeds:

  A. No BT, trunk+readout trained jointly against DS teacher (original)
  B. No BT, trunk first fit (only sheaf-L_F), then frozen, readout
     fit against DS teacher       → "random trunk + linear probe"
  C. BT pretrains trunk on two corrupted views (edge_drop + role_mask
     mixed), then trunk is frozen and readout fit against DS teacher
                                    → "BT trunk + linear probe"
  D. BT pretrains trunk (same as C), then trunk + readout jointly
     fine-tuned against DS teacher → "BT pretrain + joint finetune"

If C > B on accuracy, BT genuinely improved the representation.
If D > A, BT pretraining is worth adopting as a default.

Uses the same 60-sentence synthetic corpus as planner_scale.py, because
that was where retrieval hit 54% with room to move.
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
from cognition.logic import (
    HypergraphReasoner, REL_TO_IDX, NUM_RELATIONS,
    SheafMessagePassingLayer,
)
from cognition.synth import Schema as SynthSchema, generate_corpus, DEFAULT_TEMPLATES


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
        out[r] = {"type": "relation",
                  "tokens": [r, r.rstrip("s"), r.rstrip("s") + "ed",
                             r.rstrip("s") + "ing"]}
    for f in s.features:
        out[f] = {"type": "feature", "tokens": [f, f.rstrip("s")]}
    for m in s.markets:
        out[m.lower()] = {"type": "market", "tokens": [m.lower()]}
    return out


def build_corpus_docs(n_sentences, seed=0):
    exs = generate_corpus(SYNTH_SCHEMA, n=n_sentences, seed=seed,
                          templates=DEFAULT_TEMPLATES)
    docs = []
    for i in range(0, len(exs), 3):
        bunch = exs[i:i + 3]
        docs.append({
            "id": f"synth-{i:04d}",
            "timestamp": f"2024-{1 + (i // 30) % 12:02d}-{1 + (i % 28):02d}",
            "text": " ".join(e.sentence for e in bunch),
        })
    return docs


def build_cog(tmp, n_sentences=60, seed=0):
    schema_path = os.path.join(tmp, "schema.json")
    with open(schema_path, "w") as f:
        json.dump(schema_to_anchor_dict(SYNTH_SCHEMA), f)
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
    for i in range(0, len(docs), 20):
        cog.ingest(docs[i:i + 20])
    cog.consolidate()
    return cog


def forward_gnn(layers, graph):
    h = graph.node_features
    for l in layers:
        h = l(h, graph.edge_index, graph.edge_types, graph.edge_weights,
              graph.situation_features)
    return h


# ═════════════════════════════════════════════════════════════════════
# BT head + loss
# ═════════════════════════════════════════════════════════════════════

class BarlowHead(nn.Module):
    def __init__(self, in_dim, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.bn = nn.BatchNorm1d(proj_dim, affine=False)

    def forward(self, h):
        return self.bn(self.net(h))


def barlow_loss(za, zb, lam=5e-3):
    n = za.shape[0]
    c = (za.T @ zb) / n
    on = torch.diagonal(c)
    off = c - torch.diag(on)
    return ((1.0 - on) ** 2).sum() + lam * (off ** 2).sum()


def effective_rank(z):
    with torch.no_grad():
        s = torch.linalg.svdvals(z - z.mean(dim=0, keepdim=True))
        s2 = s.pow(2); s2 = s2 / (s2.sum() + 1e-12)
        return math.exp(-(s2 * (s2 + 1e-12).log()).sum().item())


# ═════════════════════════════════════════════════════════════════════
# View corruptions (from the original BT run)
# ═════════════════════════════════════════════════════════════════════

def corrupt_edge_drop(graph, frac, rng):
    n = graph.edge_index.shape[1]
    keep = torch.tensor(
        [rng.random() >= frac for _ in range(n)], dtype=torch.bool,
    )
    return (graph.edge_index[:, keep], graph.edge_types[keep],
            graph.edge_weights[keep])


def corrupt_role_mask(graph, rng):
    spoke = {REL_TO_IDX["INITIATES"], REL_TO_IDX["ASSERTS"],
             REL_TO_IDX["TARGETS"]}
    infon_set = set(graph.infon_map.values())
    drop_for = {i: rng.choice(list(spoke)) for i in infon_set}
    n = graph.edge_index.shape[1]
    keep = torch.ones(n, dtype=torch.bool)
    for e in range(n):
        r = int(graph.edge_types[e])
        if r not in spoke: continue
        s = int(graph.edge_index[0, e]); t = int(graph.edge_index[1, e])
        inf = t if t in infon_set else (s if s in infon_set else None)
        if inf is None: continue
        if drop_for.get(inf) == r: keep[e] = False
    return (graph.edge_index[:, keep], graph.edge_types[keep],
            graph.edge_weights[keep])


# ═════════════════════════════════════════════════════════════════════
# BT pretraining of the trunk
# ═════════════════════════════════════════════════════════════════════

def bt_pretrain(reasoner, graph, epochs=60, lr=3e-3, lam=5e-3,
                proj_dim=128, seed=0, verbose=False):
    rng = random.Random(seed)
    head = BarlowHead(reasoner.hidden_dim, proj_dim=proj_dim)
    params = list(reasoner.layers.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    infon_idx = torch.tensor(
        [i for i, t in enumerate(graph.node_types) if t == "infon"],
        dtype=torch.long,
    )
    if len(infon_idx) < 8:
        return None

    for ep in range(epochs):
        reasoner.train()
        opt.zero_grad()

        h_a = forward_gnn(reasoner.layers, graph)

        # Alternate corruption: edge-drop on even epochs, role-mask on odd
        if ep % 2 == 0:
            ei, et, ew = corrupt_edge_drop(graph, 0.30, rng)
        else:
            ei, et, ew = corrupt_role_mask(graph, rng)

        h_b = graph.node_features
        for l in reasoner.layers:
            h_b = l(h_b, ei, et, ew, graph.situation_features)

        za = head(h_a[infon_idx]); zb = head(h_b[infon_idx])
        loss = barlow_loss(za, zb, lam=lam)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

    reasoner.eval()
    with torch.no_grad():
        h = forward_gnn(reasoner.layers, graph)
        er = effective_rank(h[infon_idx])
    return {"eff_rank": er, "final_loss": loss.item()}


# ═════════════════════════════════════════════════════════════════════
# Readout training (with / without frozen trunk)
# ═════════════════════════════════════════════════════════════════════

def build_ds_target(reasoner, graph):
    from cognition.dempster_shafer import (
        mass_from_polarity, mass_from_triple_alignment,
        mass_from_anchor_distance, mass_from_confidence,
        mass_from_evidentiality, mass_from_modality, combine_multiple,
    )
    out = []
    for iid in graph.infon_map:
        inf = reasoner.store.get_infon(iid)
        if inf is None:
            out.append([0.25, 0.25, 0.25, 0.25]); continue
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


def train_readout(reasoner, graph, target, epochs=40, lr=1e-3,
                  freeze_trunk=True):
    infon_idx = torch.tensor(
        [graph.infon_map[iid] for iid in graph.infon_map],
        dtype=torch.long,
    )
    if freeze_trunk:
        for p in reasoner.layers.parameters():
            p.requires_grad_(False)
        params = list(reasoner.mass_readout.parameters())
    else:
        for p in reasoner.layers.parameters():
            p.requires_grad_(True)
        params = (list(reasoner.layers.parameters())
                  + list(reasoner.mass_readout.parameters()))
    opt = torch.optim.Adam(params, lr=lr)

    for ep in range(epochs):
        reasoner.train()
        opt.zero_grad()
        h = forward_gnn(reasoner.layers, graph)
        pred = reasoner.mass_readout(h[infon_idx])
        loss = F.kl_div(pred.log().clamp(min=-20),
                        target, reduction="batchmean")
        loss.backward()
        opt.step()

    # Unfreeze (defensive — in case downstream code re-trains)
    for p in reasoner.layers.parameters():
        p.requires_grad_(True)
    reasoner.eval()


# ═════════════════════════════════════════════════════════════════════
# Queries + evaluation
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
        if len(queries) >= n_each: break
        k = (inf.subject, inf.predicate, inf.object)
        if k in seen: continue
        seen.add(k)
        queries.append((f"Did {inf.subject} {inf.predicate} {inf.object}?",
                        "SUPPORTS"))
    nei = []
    for q, _ in list(queries):
        import re
        m = re.match(r"Did (\S+) (\S+) (.+)\?", q)
        if not m: continue
        s, p, o = m.groups()
        which = rng.choice(["subj", "obj"])
        if which == "subj" and actors:
            c = [a for a in actors if a != s]
            if not c: continue
            triple = (rng.choice(c), p, o)
        elif which == "obj" and feats:
            c = [a for a in feats if a != o]
            if not c: continue
            triple = (s, p, rng.choice(c))
        else: continue
        if triple in seen: continue
        seen.add(triple)
        nei.append((f"Did {triple[0]} {triple[1]} {triple[2]}?",
                    "NOT_ENOUGH_INFO"))
    queries.extend(nei[:n_each])
    return queries


def evaluate(reasoner, queries):
    reasoner.eval()
    c = 0; tS, tN = [], []
    for q, gold in queries:
        r = reasoner.reason(q)
        if r.verdict == gold: c += 1
        (tS if gold == "SUPPORTS" else tN).append(r.mass.theta)
    return {"acc": c / len(queries),
            "tS": sum(tS) / max(len(tS), 1),
            "tN": sum(tN) / max(len(tN), 1)}


# ═════════════════════════════════════════════════════════════════════
# Driver
# ═════════════════════════════════════════════════════════════════════

def fresh_reasoner(cog, hidden_dim=32):
    r = HypergraphReasoner(
        cog.store, cog.encoder, cog.schema,
        hidden_dim=hidden_dim, n_layers=2, use_sheaf=True,
    )
    graph = r.builder.build(feature_dim=hidden_dim, max_infons=500)
    return r, graph


def infon_eff_rank(reasoner, graph):
    reasoner.eval()
    infon_idx = torch.tensor(
        [i for i, t in enumerate(graph.node_types) if t == "infon"],
        dtype=torch.long,
    )
    with torch.no_grad():
        h = forward_gnn(reasoner.layers, graph)
    return effective_rank(h[infon_idx])


def main():
    print("Barlow Twins + linear probe — representation quality test")
    print("=" * 72)
    torch.manual_seed(0)

    with tempfile.TemporaryDirectory() as tmp:
        cog = build_cog(tmp, n_sentences=60, seed=0)
        print(f"  corpus: {cog.stats()['infon_count']} infons")

        # Single shared query set for fair comparison
        queries = generate_queries(cog, n_each=15, seed=0)
        print(f"  queries: "
              f"{sum(1 for _,g in queries if g=='SUPPORTS')}S + "
              f"{sum(1 for _,g in queries if g=='NOT_ENOUGH_INFO')}NEI")

        results = {}

        # ── Variant A: joint training (original recipe)
        print("\n[A] joint trunk+readout training (original) ...")
        r, g = fresh_reasoner(cog)
        r.fit(graph=g, epochs=20, laplacian_weight=0.1, verbose=False)
        results["A"] = {**evaluate(r, queries),
                        "rank": infon_eff_rank(r, g)}

        # ── Variant B: random-ish trunk (L_F only) + frozen linear probe
        print("[B] sheaf-only trunk (no task gradient) + linear probe ...")
        r, g = fresh_reasoner(cog)
        # Train trunk with ONLY sheaf-Laplacian (no KL teacher)
        params = list(r.layers.parameters())
        opt = torch.optim.Adam(params, lr=3e-3)
        for ep in range(40):
            r.train(); opt.zero_grad()
            h = forward_gnn(r.layers, g)
            last = r.layers[-1]
            lap = last.sheaf_discrepancy(
                h, g.edge_index, g.edge_types, g.edge_weights,
            )
            lap.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        target = build_ds_target(r, g)
        train_readout(r, g, target, epochs=40, freeze_trunk=True)
        results["B"] = {**evaluate(r, queries),
                        "rank": infon_eff_rank(r, g)}

        # ── Variant C: BT pretrain + frozen linear probe
        print("[C] BT-pretrained trunk + frozen linear probe ...")
        r, g = fresh_reasoner(cog)
        stats_c = bt_pretrain(r, g, epochs=80, lr=3e-3)
        if stats_c: print(f"    BT pretrain: eff-rank={stats_c['eff_rank']:.2f}")
        target = build_ds_target(r, g)
        train_readout(r, g, target, epochs=40, freeze_trunk=True)
        results["C"] = {**evaluate(r, queries),
                        "rank": infon_eff_rank(r, g)}

        # ── Variant D: BT pretrain + joint finetune
        print("[D] BT-pretrained trunk + joint finetune ...")
        r, g = fresh_reasoner(cog)
        bt_pretrain(r, g, epochs=80, lr=3e-3)
        target = build_ds_target(r, g)
        train_readout(r, g, target, epochs=40, freeze_trunk=False)
        results["D"] = {**evaluate(r, queries),
                        "rank": infon_eff_rank(r, g)}

        cog.close()

        print("\n" + "=" * 72)
        print("Summary")
        print("=" * 72)
        print(f"  {'variant':<48s}  {'acc':>6s} {'θ_S':>6s} {'θ_N':>6s} {'rank':>6s}")
        print("  " + "-" * 76)
        labels = {
            "A": "A. joint training (original)",
            "B": "B. L_F-only trunk (frozen) + linear probe",
            "C": "C. BT trunk (frozen) + linear probe",
            "D": "D. BT pretrain + joint finetune",
        }
        for k in "ABCD":
            v = results[k]
            print(f"  {labels[k]:<48s}  "
                  f"{v['acc']:>5.0%}  "
                  f"{v['tS']:>5.2f}  "
                  f"{v['tN']:>5.2f}  "
                  f"{v['rank']:>5.2f}")


if __name__ == "__main__":
    main()

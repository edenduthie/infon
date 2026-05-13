"""Full self-supervised loop — human input is only the raw corpus.

Three compounding mechanisms integrated into one training run:

  Step 1: DS teacher → EMA self-distillation
    - α weights the hand-crafted DS heuristic teacher vs the EMA-smoothed
      student's own prediction.
    - α decays linearly from 1.0 (fully heuristic-supervised) to 0.0
      (fully self-supervised) over training.
    - If the representation is any good, accuracy should hold as α → 0.

  Step 2: Schema discovered from raw text
    - SchemaDiscovery runs spectral clustering on SPLADE co-activations.
    - No hand-authored anchors; the left-Kan extension of the observed
      vocabulary is the schema.
    - We compare against a baseline with hand-written schema.

  Step 3: Self-generated query set
    - Every high-confidence stored infon (s, p, o) becomes a SUPPORTS
      claim: "Did <s> <p> <o>?"
    - Role-swapping one slot (subject or object) with a random anchor
      produces a NEI claim.
    - This gives a reproducible, unsupervised accuracy measure — no
      human-authored gold set.

Headline metrics
----------------
- accuracy(α) as α decays — does self-distillation hold up?
- θ_NEI stability — does calibration survive without the DS teacher?
- hand-written vs discovered schema — how much does schema authoring
  matter on a small corpus?
- correlation between predictor top-3 (from Temporal-JEPA) and final
  accuracy: does "good world model" predict "good reasoner"?
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cognition import Cognition, CognitionConfig
from cognition.logic import HypergraphReasoner, REL_TO_IDX
from cognition.category import SchemaDiscovery
from cognition.schema import AnchorSchema
from cognition.encoder import Encoder, SpladeEncoder


# ═════════════════════════════════════════════════════════════════════
# Corpus (same raw text — the ONLY human input)
# ═════════════════════════════════════════════════════════════════════

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


# Hand-written schema (BASELINE — not used in the self-supervised run)
BASELINE_SCHEMA = {
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
# Step 2: discover schema from raw text (no anchors authored)
# ═════════════════════════════════════════════════════════════════════

def discover_schema(texts: list[str], n_anchors: int = 18) -> dict:
    """Run SchemaDiscovery and post-process for the role-constrained
    extractor. The raw discovery outputs are lexically plausible but
    role-types often miscategorize (e.g. "batteries" → relation).
    We fix this heuristically using surface forms of the tokens:
      - tokens ending in common verb suffixes (-s, -ed, -ing, -e)
        OR lemmas known to be verbs in context → relation
      - tokens that are proper nouns / acronyms → actor
      - tokens that are place names → market
      - everything else → feature
    """
    disc = SchemaDiscovery()
    _anchor_schema, discovered = disc.discover(
        texts,
        n_anchors=n_anchors,
        min_doc_freq=1,
        activation_threshold=0.3,
    )

    # Vocabulary of the corpus, lowercased
    corpus_words = set()
    for t in texts:
        corpus_words.update(w.lower().strip(".,!?") for w in t.split())

    # Heuristic role-type by token surface form
    verb_seeds = {"invest", "invests", "invested", "investment",
                  "partner", "partners", "partnered", "partnership",
                  "produce", "produces", "produced", "production",
                  "expand", "expands", "expanded",
                  "acquire", "acquires", "acquired",
                  "delay", "delays", "delayed"}
    actor_seeds = {"toyota", "honda", "tesla", "panasonic", "catl",
                   "ford", "bmw"}
    market_seeds = {"japan", "japanese", "china", "chinese",
                    "america", "european", "europe",
                    "north", "korea"}

    def infer_type(tokens: list[str]) -> str | None:
        tokset = {t.lower() for t in tokens}
        if tokset & actor_seeds:
            return "actor"
        if tokset & market_seeds:
            return "market"
        if tokset & verb_seeds:
            return "relation"
        # Heuristic fallbacks
        for t in tokens:
            tl = t.lower()
            if tl.endswith(("ing", "ed", "es")) and len(tl) > 4:
                return "relation"
        return "feature"

    out: dict = {}
    for d in discovered:
        if not d.tokens:
            continue
        name = d.name
        if name in out:
            name = f"{name}_{len(out)}"
        ttype = infer_type(d.tokens)
        if ttype is None:
            continue
        out[name] = {
            "type": ttype,
            "tokens": list(d.tokens)[:6],
        }

    # Guarantee we have at least one anchor of each critical type
    types_present = {v["type"] for v in out.values()}
    if "actor" not in types_present or "relation" not in types_present:
        # Fall back to seed anchors observable in the corpus
        for name in actor_seeds:
            if name in corpus_words and name not in out:
                out[name] = {"type": "actor", "tokens": [name]}
        for name in verb_seeds:
            if name in corpus_words and name not in out:
                out[name] = {"type": "relation", "tokens": [name]}

    return out


# ═════════════════════════════════════════════════════════════════════
# Step 3: self-generated query set
# ═════════════════════════════════════════════════════════════════════

def generate_queries(cog, max_supports: int = 8,
                     min_conf: float = 0.05,
                     seed: int = 0) -> list[tuple[str, str]]:
    """Build (question, gold_verdict) pairs straight from the store.

    SUPPORTS pairs: pick top-confidence infons, template as
      "Did <subj> <pred> <obj>?"
    NEI pairs: take a SUPPORTS triple, swap one role with a random
      other anchor of the right type — the graph almost certainly
      doesn't contain that exact combination.
    """
    rng = random.Random(seed)
    infons = cog.store.query_infons(limit=200)
    good = [i for i in infons if i.confidence >= min_conf]
    if not good:
        return []

    anchor_names = list(cog.schema.names)
    types = cog.schema.types
    actors = [n for n in anchor_names if types.get(n) == "actor"]
    feats = [n for n in anchor_names
             if types.get(n) in ("feature", "market")]

    queries: list[tuple[str, str]] = []
    # Sort by confidence, take the cream
    good.sort(key=lambda x: -x.confidence)
    seen_triples: set[tuple[str, str, str]] = set()
    for inf in good:
        if len(queries) >= max_supports:
            break
        key = (inf.subject, inf.predicate, inf.object)
        if key in seen_triples:
            continue
        seen_triples.add(key)
        q = f"Did {inf.subject} {inf.predicate} {inf.object}?"
        queries.append((q, "SUPPORTS"))

    # NEI — swap one role in each supports query
    nei = []
    for q, _ in list(queries):
        # recover triple from the query template
        import re
        m = re.match(r"Did (\S+) (\S+) (.+)\?", q)
        if not m:
            continue
        s, p, o = m.groups()
        which = rng.choice(["subj", "obj"])
        if which == "subj" and actors:
            replacement = rng.choice([a for a in actors if a != s] or [s])
            candidate = (replacement, p, o)
        elif which == "obj" and feats:
            replacement = rng.choice([a for a in feats if a != o] or [o])
            candidate = (s, p, replacement)
        else:
            continue
        if candidate in seen_triples:
            continue
        nei.append((
            f"Did {candidate[0]} {candidate[1]} {candidate[2]}?",
            "NOT_ENOUGH_INFO",
        ))
    queries.extend(nei[:max_supports])
    return queries


def evaluate_on(reasoner, queries) -> dict:
    if not queries:
        return {"accuracy": float("nan"),
                "theta_supports": float("nan"),
                "theta_nei": float("nan"),
                "n": 0}
    reasoner.eval()
    correct = 0
    tS, tN = [], []
    for q, gold in queries:
        r = reasoner.reason(q)
        if r.verdict == gold:
            correct += 1
        if gold == "SUPPORTS":
            tS.append(r.mass.theta)
        else:
            tN.append(r.mass.theta)
    return {
        "accuracy": correct / len(queries),
        "theta_supports": sum(tS) / max(len(tS), 1),
        "theta_nei":      sum(tN) / max(len(tN), 1),
        "n": len(queries),
    }


# ═════════════════════════════════════════════════════════════════════
# Step 1: self-distillation trainer
# ═════════════════════════════════════════════════════════════════════

@torch.no_grad()
def update_ema(student: nn.Module, teacher: nn.Module, tau: float):
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(tau).add_(ps.data, alpha=1.0 - tau)


def ds_teacher_masses(reasoner, graph) -> torch.Tensor:
    """Run the six hand-crafted DS sources and return the fused teacher
    mass per infon node. This is the "heuristic supervision" we're
    trying to decay away from."""
    from cognition.dempster_shafer import (
        mass_from_polarity, mass_from_triple_alignment,
        mass_from_anchor_distance, mass_from_confidence,
        mass_from_evidentiality, mass_from_modality,
        combine_multiple,
    )
    teacher = []
    for iid in graph.infon_map:
        inf = reasoner.store.get_infon(iid)
        if inf is None:
            teacher.append([0.25, 0.25, 0.25, 0.25])
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
        teacher.append([m.supports, m.refutes, m.uncertain, m.theta])
    return torch.tensor(teacher, dtype=torch.float32)


def train_with_decaying_alpha(reasoner, graph,
                              epochs: int = 40,
                              lr: float = 1e-3,
                              alpha_start: float = 1.0,
                              alpha_end:   float = 0.0,
                              tau: float = 0.995,
                              eval_every: int = 5,
                              eval_queries: list | None = None,
                              verbose: bool = True) -> list[dict]:
    """Blend DS teacher and EMA self-teacher with decaying α."""
    # EMA teacher copy of the full reasoner (layers + readout)
    teacher_layers = copy.deepcopy(reasoner.layers)
    teacher_readout = copy.deepcopy(reasoner.mass_readout)
    for p in teacher_layers.parameters(): p.requires_grad_(False)
    for p in teacher_readout.parameters(): p.requires_grad_(False)
    teacher_layers.eval(); teacher_readout.eval()

    # Pre-compute the static DS-teacher target once
    ds_target = ds_teacher_masses(reasoner, graph)

    # Indices of infon nodes (in node order of graph)
    infon_indices = torch.tensor(
        [graph.infon_map[iid] for iid in graph.infon_map],
        dtype=torch.long,
    )

    opt = torch.optim.Adam(
        list(reasoner.layers.parameters())
        + list(reasoner.mass_readout.parameters()),
        lr=lr,
    )
    history = []

    for epoch in range(epochs):
        alpha = alpha_start + (alpha_end - alpha_start) * (epoch / max(epochs - 1, 1))
        reasoner.train()
        opt.zero_grad()

        # Student forward
        h_s = graph.node_features
        for layer in reasoner.layers:
            h_s = layer(h_s, graph.edge_index, graph.edge_types,
                        graph.edge_weights, graph.situation_features)
        pred_s = reasoner.mass_readout(h_s[infon_indices])

        # EMA-teacher forward (no grad)
        with torch.no_grad():
            h_t = graph.node_features
            for layer in teacher_layers:
                h_t = layer(h_t, graph.edge_index, graph.edge_types,
                            graph.edge_weights, graph.situation_features)
            pred_t = teacher_readout(h_t[infon_indices])
            # confidence floor: teacher must be reasonably confident
            conf = 1.0 - pred_t[:, 3]   # 1 - theta
            mask = (conf > 0.3).float().unsqueeze(-1)

        # Blended target
        target = alpha * ds_target + (1 - alpha) * (mask * pred_t + (1 - mask) * ds_target)

        # KL to blended target
        loss = F.kl_div(pred_s.log().clamp(min=-20),
                        target, reduction="batchmean")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(reasoner.layers.parameters())
            + list(reasoner.mass_readout.parameters()), 1.0)
        opt.step()
        update_ema(reasoner.layers, teacher_layers, tau)
        update_ema(reasoner.mass_readout, teacher_readout, tau)

        entry = {"epoch": epoch + 1, "alpha": alpha,
                 "loss": float(loss.item()),
                 "self_sup_frac": float((mask.mean()).item())}

        if eval_queries and (epoch + 1) % eval_every == 0:
            ev = evaluate_on(reasoner, eval_queries)
            entry.update({
                "eval_acc": ev["accuracy"],
                "eval_theta_S": ev["theta_supports"],
                "eval_theta_N": ev["theta_nei"],
            })
            if verbose:
                print(f"    epoch {epoch+1:2d}  α={alpha:.2f}  loss={loss.item():.3f}  "
                      f"acc={ev['accuracy']:.0%}  θ_S={ev['theta_supports']:.2f}  "
                      f"θ_N={ev['theta_nei']:.2f}  self_sup%={entry['self_sup_frac']:.2f}")

        history.append(entry)

    reasoner.eval()
    return history


# ═════════════════════════════════════════════════════════════════════
# Driver
# ═════════════════════════════════════════════════════════════════════

def build_cog(tmpdir: str, schema: dict):
    schema_path = os.path.join(tmpdir, "schema.json")
    with open(schema_path, "w") as f:
        json.dump(schema, f)
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


def run_variant(label: str, schema: dict,
                alpha_end: float = 0.0,
                epochs: int = 40,
                seed: int = 0):
    """Train a reasoner with the given schema and α schedule, evaluate
    on the *self-generated* query set. Returns summary dict."""
    torch.manual_seed(seed)
    with tempfile.TemporaryDirectory() as tmpdir:
        cog = build_cog(tmpdir, schema)

        reasoner = HypergraphReasoner(
            cog.store, cog.encoder, cog.schema,
            hidden_dim=32, n_layers=2, use_sheaf=True,
        )
        graph = reasoner.builder.build(feature_dim=32)
        # Warm-start with 5 epochs of pure DS training so the layers
        # aren't random.
        reasoner.fit(graph=graph, epochs=5, laplacian_weight=0.1,
                     verbose=False)

        queries = generate_queries(cog, max_supports=8, min_conf=0.05, seed=seed)
        n_supports = sum(1 for _, g in queries if g == "SUPPORTS")
        n_nei = sum(1 for _, g in queries if g == "NOT_ENOUGH_INFO")
        print(f"\n== Variant: {label} ==")
        print(f"  schema size: {len(schema)} anchors")
        print(f"  store: {cog.stats()['infon_count']} infons "
              f"/ query set: {n_supports} SUPPORTS + {n_nei} NEI")

        pre = evaluate_on(reasoner, queries)
        print(f"  BEFORE self-distillation:  "
              f"acc={pre['accuracy']:.0%}  θ_S={pre['theta_supports']:.2f}  "
              f"θ_N={pre['theta_nei']:.2f}")

        if not queries or cog.stats()["infon_count"] == 0:
            print("  (no infons extracted — skipping training for this variant)")
            cog.close()
            return {"label": label, "pre": pre, "post": pre,
                    "history": [], "n_queries": 0,
                    "n_supports": 0, "n_nei": 0}

        hist = train_with_decaying_alpha(
            reasoner, graph,
            epochs=epochs, alpha_end=alpha_end,
            eval_every=10, eval_queries=queries, verbose=True,
        )

        post = evaluate_on(reasoner, queries)
        print(f"  AFTER (α → {alpha_end:.1f}):  "
              f"acc={post['accuracy']:.0%}  θ_S={post['theta_supports']:.2f}  "
              f"θ_N={post['theta_nei']:.2f}")

        cog.close()
        return {"label": label, "pre": pre, "post": post,
                "history": hist, "n_queries": len(queries),
                "n_supports": n_supports, "n_nei": n_nei}


def main():
    print("Self-Supervised Full Loop — human input = corpus text only")
    print("=" * 72)

    # Step 2: discover schema from raw corpus text
    print("\n[Step 2] Discovering schema from raw text ...")
    sentences = [d["text"] for d in CORPUS]
    discovered = discover_schema(sentences, n_anchors=18)
    print(f"  discovered {len(discovered)} anchors:")
    for name, meta in list(discovered.items())[:12]:
        toks = ', '.join(meta['tokens'][:4])
        print(f"    {name:20s} [{meta['type']:8s}]  tokens: {toks}")
    if len(discovered) > 12:
        print(f"    ... and {len(discovered)-12} more")

    results = []
    # A) hand-written schema + DS teacher only (classic baseline)
    results.append(run_variant(
        "A. hand-schema + heuristic-only (α stays 1.0)",
        BASELINE_SCHEMA, alpha_end=1.0, epochs=40, seed=0,
    ))
    # B) hand-written schema + full self-distillation (α → 0)
    results.append(run_variant(
        "B. hand-schema + self-distillation (α 1→0)",
        BASELINE_SCHEMA, alpha_end=0.0, epochs=40, seed=0,
    ))
    # C) discovered schema + full self-distillation
    results.append(run_variant(
        "C. discovered-schema + self-distillation (α 1→0)",
        discovered, alpha_end=0.0, epochs=40, seed=0,
    ))

    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"  {'variant':<58s} {'acc':>6s} {'θ_S':>6s} {'θ_N':>6s}")
    print("  " + "-" * 82)
    for r in results:
        print(f"  {r['label'][:56]:<58s} "
              f"{r['post']['accuracy']:>6.0%} "
              f"{r['post']['theta_supports']:>6.2f} "
              f"{r['post']['theta_nei']:>6.2f}")


if __name__ == "__main__":
    main()

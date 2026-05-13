"""Real-data eval: can the GNN generalize from synthgen to real cassettes?

The model trained on synthgen's default config hit 99% on synthetic test.
This probe asks the fairer question: on a REAL corpus with real extraction
noise, does it still beat the 88.5% symbolic baseline?

Test protocol:
  1. Ingest the 15-doc AI-chip corpus under a pre-tuned schema.
  2. For each hand-picked claim with known gold verdict:
     a. Run reason_connectivity() — get the MCTS-discovered chain.
     b. Encode that chain as a GNN input (same 10-dim features).
     c. Get GNN verdict.
     d. Compare both to gold.
  3. Report agreement with gold for both methods.

Key caveat baked into this design: the GNN was trained on synthgen.
Synthgen's default_config uses 15 actors (a000..a014) and relation kinds
(connective/terminal/reportive). Real corpora use different anchor names
— the GNN can only work because our features are schema-INDEPENDENT
(kind + polarity + position flags, no anchor-id embedding). That's the
real transfer test.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "cognition", "src"))

import torch

from cognition.cassette import InfonStore
from cognition.cassette.reason_path import (
    reason_connectivity, _expand, _entity_set, _Node,
    chain_mass, _label_from_mass,
)
from cognition.cassette.dsl import Query
from cognition.cassette.reader import hydrate_locs
from cognition.cassette.gnn_encoder import (
    SheafHypergraphEncoder, HypergraphBatch, encode_edge,
    IDX_TO_VERDICT, EDGE_FEATURE_DIM,
)
from cognition.cassette.synthgen import _default_config


# ─── corpus + schema (pre-tuned to 93% coverage — from Commit 4 probe) ──
SCHEMA = {
    "nvidia":    {"type": "actor", "tokens": ["nvidia"]},
    "tsmc":      {"type": "actor", "tokens": ["tsmc"]},
    "intel":     {"type": "actor", "tokens": ["intel"]},
    "amd":       {"type": "actor", "tokens": ["amd"]},
    "openai":    {"type": "actor", "tokens": ["openai"]},
    "anthropic": {"type": "actor", "tokens": ["anthropic"]},
    "microsoft": {"type": "actor", "tokens": ["microsoft", "azure"]},
    "google":    {"type": "actor", "tokens": ["google", "alphabet"]},
    "samsung":   {"type": "actor", "tokens": ["samsung"]},
    "sk_hynix":  {"type": "actor", "tokens": ["hynix"]},
    "aws":       {"type": "actor", "tokens": ["aws"]},

    "partner":   {"type": "relation",
                   "tokens": ["partner", "partnered", "partnership", "venture"]},
    "supply":    {"type": "relation",
                   "tokens": ["supply", "supplies", "supplier", "sources"]},
    "invest":    {"type": "relation",
                   "tokens": ["invest", "invested", "investment"]},
    "acquire":   {"type": "relation", "tokens": ["acquire", "acquired"]},
    "compete":   {"type": "relation", "tokens": ["compete", "competing"]},
    "mention":   {"type": "relation", "tokens": ["mention", "mentioned"]},

    "hbm":       {"type": "feature", "tokens": ["hbm", "high-bandwidth memory"]},
    "b200":      {"type": "feature", "tokens": ["b200"]},
    "tpu":       {"type": "feature", "tokens": ["tpu"]},
    "3nm":       {"type": "feature", "tokens": ["3nm"]},
    "foundry":   {"type": "feature", "tokens": ["foundry"]},
    "compute":   {"type": "feature", "tokens": ["compute", "datacenter", "data center"]},
}

DOCS = [
    {"id": "d1",  "text": "Nvidia partnered with TSMC to produce the new B200 chip on the 3nm process.", "timestamp": "2026-01-05"},
    {"id": "d2",  "text": "SK Hynix supplies HBM memory to Nvidia for its datacenter GPU line.", "timestamp": "2026-01-08"},
    {"id": "d3",  "text": "Samsung supplies HBM to Nvidia, competing with SK Hynix.", "timestamp": "2026-01-12"},
    {"id": "d4",  "text": "Microsoft invested heavily in datacenter capacity for its OpenAI partnership.", "timestamp": "2026-01-15"},
    {"id": "d5",  "text": "OpenAI partnered with Microsoft for Azure compute.", "timestamp": "2026-01-20"},
    {"id": "d6",  "text": "Google invested in custom TPU development, competing with Nvidia's datacenter GPUs.", "timestamp": "2026-02-01"},
    {"id": "d7",  "text": "Anthropic partnered with Google for TPU compute.", "timestamp": "2026-02-05"},
    {"id": "d8",  "text": "Intel announced a foundry push, aiming to compete with TSMC.", "timestamp": "2026-02-10"},
    {"id": "d9",  "text": "AMD invested in HBM supply contracts with SK Hynix.", "timestamp": "2026-02-15"},
    {"id": "d10", "text": "Nvidia acquired a small startup specializing in datacenter networking.", "timestamp": "2026-02-20"},
    {"id": "d11", "text": "Samsung's HBM supply to Nvidia fell through after quality issues.", "timestamp": "2026-03-01"},
    {"id": "d12", "text": "TSMC's 3nm production bottleneck is delaying Nvidia's B200 ramp.", "timestamp": "2026-03-05"},
    {"id": "d13", "text": "OpenAI and Microsoft extended their partnership with a multi-year commitment.", "timestamp": "2026-03-10"},
    {"id": "d14", "text": "Anthropic also partnered with AWS for additional compute redundancy.", "timestamp": "2026-03-15"},
    {"id": "d15", "text": "Intel scrapped its foundry plans, citing insufficient customer commitments.", "timestamp": "2026-04-01"},
]

# Hand-picked eval claims.  gold comes from reading the docs.
# KIND column tells us which failure mode (if any) we're testing.
CLAIMS = [
    # (source, target, gold_verdict, notes)
    ("nvidia",    "tsmc",        "SUPPORTS",        "direct, d1"),
    ("openai",    "microsoft",   "SUPPORTS",        "direct, d4+d5+d13"),
    ("anthropic", "google",      "SUPPORTS",        "direct, d7"),
    ("nvidia",    "catl",        "NOT_ENOUGH_INFO", "catl absent — no anchor"),
    ("intel",     "tsmc",        "NOT_ENOUGH_INFO", "d8 says 'aiming to compete', reportive"),
    ("samsung",   "nvidia",      "SUPPORTS",        "d3 supply"),
    ("anthropic", "nvidia",      "NOT_ENOUGH_INFO", "no direct link"),
    ("nvidia",    "sk_hynix",    "SUPPORTS",        "d2 reverse direction"),
    ("google",    "anthropic",   "SUPPORTS",        "d7 reverse"),
    ("ford",      "tsmc",        "NOT_ENOUGH_INFO", "ford not in corpus"),
]


# ═══════════════════════════════════════════════════════════════════════
# CHAIN-EDGE CONVERTER — real cassette chain → GNN features
# ═══════════════════════════════════════════════════════════════════════

def infer_relation_kinds(schema_dict: dict) -> dict[str, str]:
    """Map each relation anchor → kind for the GNN encoding.

    We use the same heuristic as synthgen's default: partner/supply/
    acquire/license/invest = connective; mention/report/discuss =
    reportive; everything else = terminal.

    A real deployment would let the agent supply this mapping
    (it's part of synthgen config). For this probe we hardcode it."""
    connective_names = {"partner", "supply", "acquire", "license", "invest"}
    reportive_names  = {"mention", "describe", "discuss", "review",
                         "announce", "report"}
    out = {}
    for name, info in schema_dict.items():
        if info.get("type") != "relation":
            continue
        if name in connective_names:
            out[name] = "connective"
        elif name in reportive_names:
            out[name] = "reportive"
        else:
            out[name] = "terminal"
    return out


def encode_mcts_path(path_edges: list,
                     source: str, target: str,
                     relation_kinds: dict,
                     max_len: int = 12) -> list[list[float]]:
    """Convert a MCTS-discovered `path_edges` (list[list[Infon]]) into the
    GNN's 10-dim per-edge feature rows. Mirrors encode_chain_from_synth.

    path_edges is already ordered source → ... → target. Each element is
    a list of Infons for that hop's triple (may include retractions)."""
    # Flatten retractions onto their triple's row — sort by timestamp so
    # the retraction appears AFTER the affirmation if there is one.
    rows = []
    prev_obj = None
    for edge_infons in path_edges:
        # Pick a representative infon for the feature row. If there are
        # multiple (affirm + refute), encode them both as separate rows
        # in time order. The GNN saw this pattern in synthgen's
        # retracted examples.
        sorted_infons = sorted(edge_infons, key=lambda i: i.timestamp or "")
        for inf in sorted_infons:
            kind = relation_kinds.get(inf.predicate, "terminal")
            touches_source = (inf.subject == source or inf.object == source)
            touches_target = (inf.subject == target or inf.object == target)
            connects_prev = (prev_obj is None) or (inf.subject == prev_obj)
            rows.append({
                "kind": kind,
                "polarity": inf.polarity,
                "confidence": inf.confidence,
                "gap_days": 0,
                "is_last": False,
                "touches_source": touches_source,
                "touches_target": touches_target,
                "connects_prev": connects_prev,
            })
            prev_obj = inf.object
    if rows:
        rows[-1]["is_last"] = True

    # Pad / truncate to max_len.
    rows = rows[-max_len:]
    vecs = [encode_edge(**r) for r in rows]
    mask = [1.0] * len(vecs)
    while len(vecs) < max_len:
        vecs.insert(0, [0.0] * EDGE_FEATURE_DIM)
        mask.insert(0, 0.0)
    return vecs, mask


def gnn_verdict(model: SheafHypergraphEncoder,
                path_edges: list, source: str, target: str,
                relation_kinds: dict) -> tuple[str, list[float]]:
    """Run the GNN on a single real chain. Returns (label, class_probs)."""
    if not path_edges:
        # No chain discovered → NEI, trivially.
        return "NOT_ENOUGH_INFO", [0.0, 0.0, 1.0]
    vecs, mask = encode_mcts_path(path_edges, source, target, relation_kinds)
    batch = HypergraphBatch(
        edges=torch.tensor([vecs], dtype=torch.float32),
        edge_mask=torch.tensor([mask], dtype=torch.float32),
        verdicts=torch.tensor([0], dtype=torch.long),  # dummy
    )
    model.eval()
    with torch.no_grad():
        out = model(batch)
        probs = torch.softmax(out["logits"], dim=-1)[0].tolist()
        idx = int(probs.index(max(probs)))
    return IDX_TO_VERDICT[idx], probs


# ═══════════════════════════════════════════════════════════════════════
# SHARED MCTS WRAPPER — returns both symbolic verdict and its path_edges
# ═══════════════════════════════════════════════════════════════════════

def run_with_path(store: InfonStore, source: str, target: str):
    """Call reason_connectivity but also get back the path_edges it walked.

    reason_connectivity returns a Verdict with `sources` (flat infon list)
    which we need to regroup by triple to get the edge-wise structure the
    GNN expects. That's fine for our purposes — the GNN's connects_prev
    flag handles minor ordering differences robustly."""
    v = store.connect(source, target, max_hops=3)

    # Regroup sources into edges by their (subject, predicate, object).
    from collections import OrderedDict
    grouped: "OrderedDict[tuple, list]" = OrderedDict()
    for inf in v.sources:
        k = (inf.subject, inf.predicate, inf.object)
        grouped.setdefault(k, []).append(inf)
    path_edges = list(grouped.values())
    return v, path_edges


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    tmp = tempfile.mkdtemp(prefix="gnn_real_")
    try:
        # Build the store.
        schema_path = os.path.join(tmp, "schema.json")
        with open(schema_path, "w") as f:
            json.dump(SCHEMA, f)
        store = InfonStore(os.path.join(tmp, "store"), schema_path=schema_path)
        r = store.ingest(DOCS)
        covered = r["report"].n_docs - len(r["report"].docs_with_zero_infons)
        print(f"store: {r['n_infons']} infons, {covered}/{r['report'].n_docs} docs\n")

        # Train the GNN fresh so the probe is self-contained. Same
        # hyperparameters as probe_gnn_train.py.
        print("── training GNN on synthgen (default config) ──")
        from cognition.cassette.synthgen import generate
        from cognition.cassette.gnn_encoder import batch_from_synth, train

        cfg = _default_config()
        cfg.n_samples = 2000
        cfg.seed = 7
        train_graphs = generate(cfg)

        cfg_val = _default_config()
        cfg_val.n_samples = 300
        cfg_val.seed = 11
        val_graphs = generate(cfg_val)

        rel_kinds = {r.name: r.kind for r in cfg.relations}
        train_batch = batch_from_synth(train_graphs, rel_kinds)
        val_batch = batch_from_synth(val_graphs, rel_kinds)

        model = SheafHypergraphEncoder(hidden_dim=64, n_layers=3)
        train(model, train_batch, val_batch, epochs=25,
              batch_size=64, lr=2e-3, verbose=False)
        print(f"  trained, {model.n_params():,} params\n")

        # Evaluate on real claims.
        print("─" * 78)
        print(f"  {'claim':<38}  {'gold':<16}  {'symbolic':<16}  {'gnn':<16}")
        print("─" * 78)

        real_kinds = infer_relation_kinds(SCHEMA)
        sym_correct = 0
        gnn_correct = 0

        for source, target, gold, notes in CLAIMS:
            verdict, path_edges = run_with_path(store, source, target)
            sym_label = verdict.label
            gnn_label, probs = gnn_verdict(model, path_edges, source, target,
                                             real_kinds)

            sym_ok = "✓" if sym_label == gold else "✗"
            gnn_ok = "✓" if gnn_label == gold else "✗"
            sym_correct += (sym_label == gold)
            gnn_correct += (gnn_label == gold)

            claim_label = f"{source}→{target}"
            print(f"  {claim_label:<38}  "
                  f"{gold:<16}  "
                  f"{sym_ok}{sym_label:<15}  "
                  f"{gnn_ok}{gnn_label:<15}")

        print("─" * 78)
        n = len(CLAIMS)
        print(f"  symbolic: {sym_correct}/{n} = {sym_correct/n:.0%}")
        print(f"  gnn:      {gnn_correct}/{n} = {gnn_correct/n:.0%}")

        # Detail where the two disagree.
        disagree = []
        for source, target, gold, notes in CLAIMS:
            verdict, path_edges = run_with_path(store, source, target)
            sym_label = verdict.label
            gnn_label, probs = gnn_verdict(model, path_edges, source, target,
                                            real_kinds)
            if sym_label != gnn_label:
                disagree.append((source, target, gold, sym_label,
                                  gnn_label, probs, path_edges))

        if disagree:
            print("\n── disagreements (gnn vs symbolic) ──")
            for source, target, gold, sym, gnn, probs, pe in disagree:
                print(f"\n  {source} → {target}  (gold={gold})")
                print(f"    symbolic: {sym}")
                print(f"    gnn:      {gnn}  probs=[S:{probs[0]:.2f} "
                      f"R:{probs[1]:.2f} N:{probs[2]:.2f}]")
                print(f"    path_edges: {len(pe)} hops")
                for i, edge in enumerate(pe):
                    for inf in edge:
                        mark = "¬" if inf.polarity == 0 else " "
                        print(f"      hop{i}: {mark}{inf.subject}/"
                              f"{inf.predicate}/{inf.object}  "
                              f"conf={inf.confidence}")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()

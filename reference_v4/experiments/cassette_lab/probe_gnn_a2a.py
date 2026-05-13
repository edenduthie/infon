"""A/B: GNN-augmented vs symbolic reasoner on hand-curated actor-to-actor corpus.

The curated corpus bypasses SPLADE extraction so the reasoner has the
actor-to-actor edges it needs. This is the fair test the probe_gnn_real
run couldn't be: the bottleneck earlier was extraction, not reasoning.

Protocol:
  1. Write curated cassettes (30 infons, 10 cassettes).
  2. Train the sheaf GNN on synthgen (default config) and save under
     <root>/_model/gnn.pt.
  3. Run each claim twice:
     a. reason_connectivity(use_gnn=False) — symbolic only.
     b. reason_connectivity(use_gnn=True)  — GNN blended at weight=0.5.
  4. Report per-claim verdicts, overall accuracy, per-kind breakdown.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "cognition", "src"))

import torch

from cognition.cassette import InfonStore
from cognition.cassette.reason_path import reason_connectivity, _GNN_CACHE
from cognition.cassette.synthgen import _default_config, generate
from cognition.cassette.gnn_encoder import (
    SheafHypergraphEncoder, batch_from_synth, train,
)

from curated_corpus import (
    write_curated_store, CLAIMS, RELATION_KINDS,
)


def train_and_save(root: str, hidden_dim: int = 64, n_layers: int = 3,
                   epochs: int = 25, verbose: bool = False):
    """Train the GNN and save to <root>/_model/gnn.pt alongside cassettes."""
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

    model = SheafHypergraphEncoder(hidden_dim=hidden_dim, n_layers=n_layers)
    r = train(model, train_batch, val_batch, epochs=epochs,
              batch_size=64, lr=2e-3, verbose=verbose)

    model_dir = os.path.join(root, "_model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "gnn.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "best_val_acc": r.best_val_acc,
    }, model_path)
    return r.best_val_acc, model_path


def label_cell(ok: bool, label: str) -> str:
    mark = "✓" if ok else "✗"
    return f"{mark}{label:<16}"


def main():
    tmp = tempfile.mkdtemp(prefix="gnn_a2a_")
    try:
        root = os.path.join(tmp, "store")

        # ── 1. Build curated store.
        print("── curated corpus ──")
        m = write_curated_store(root)
        n_infons = sum(c["n_records"] for c in m.cassettes)
        print(f"  {len(m.cassettes)} cassettes, {n_infons} infons")

        # ── 2. Train GNN, save under <root>/_model/gnn.pt.
        print("\n── training GNN on synthgen ──")
        val_acc, model_path = train_and_save(root)
        print(f"  best val acc on synthgen: {val_acc:.3f}")
        print(f"  saved: {model_path}")

        # Clear cache so the test actually loads from disk.
        _GNN_CACHE.clear()

        # ── 3. A/B on real claims. Note the GNN path uses its own
        # predicate kinds since our predicates (partner/supply/acquire)
        # happen to match the synthgen vocabulary by design.
        manifest = m   # head manifest saved at write time

        # For reason_connectivity we need InfonStore-ish setup. Cheat:
        # create a minimal InfonStore pointing at root, set a stub schema.
        schema_path = os.path.join(tmp, "schema.json")
        stub = {}
        # Make a flat actor/relation schema from the curated infons so
        # InfonStore doesn't complain.
        for cid in [c["cassette_id"] for c in manifest.cassettes]:
            pass
        for c in manifest.cassettes:
            for s in c.get("subjects", []):
                stub[s] = {"type": "actor", "tokens": [s]}
            for p in c.get("predicates", []):
                stub[p] = {"type": "relation", "tokens": [p]}
            for o in c.get("objects", []):
                stub.setdefault(o, {"type": "actor", "tokens": [o]})
        with open(schema_path, "w") as f:
            json.dump(stub, f)

        store = InfonStore(root, schema_path=schema_path)

        print("\n── A/B: symbolic vs GNN-augmented ──")
        print(f"{'claim':<38}  {'gold':<16}  {'symbolic':<17}  {'gnn blend':<17}  notes")
        print("─" * 110)

        sym_correct = 0
        gnn_correct = 0
        per_kind = {"SUPPORTS": [0, 0], "REFUTES": [0, 0],
                     "NOT_ENOUGH_INFO": [0, 0]}  # [sym_correct, gnn_correct]
        per_kind_total = {"SUPPORTS": 0, "REFUTES": 0, "NOT_ENOUGH_INFO": 0}

        for (source, target, gold, notes) in CLAIMS:
            v_sym = reason_connectivity(
                store.manifest, source, target,
                connective_predicates={"partner", "supply", "acquire",
                                        "license", "invest"},
                use_gnn=False,
            )
            v_gnn = reason_connectivity(
                store.manifest, source, target,
                connective_predicates={"partner", "supply", "acquire",
                                        "license", "invest"},
                use_gnn=True,
                gnn_weight=0.5,
                relation_kinds=RELATION_KINDS,
            )
            sym_ok = v_sym.label == gold
            gnn_ok = v_gnn.label == gold
            sym_correct += sym_ok
            gnn_correct += gnn_ok
            per_kind[gold][0] += sym_ok
            per_kind[gold][1] += gnn_ok
            per_kind_total[gold] += 1

            print(f"  {source + '→' + target:<36}  "
                  f"{gold:<16}  "
                  f"{label_cell(sym_ok, v_sym.label)}  "
                  f"{label_cell(gnn_ok, v_gnn.label)}  "
                  f"{notes}")

        n = len(CLAIMS)
        print("─" * 110)
        print(f"  symbolic: {sym_correct}/{n} = {sym_correct/n:.0%}")
        print(f"  gnn:      {gnn_correct}/{n} = {gnn_correct/n:.0%}")

        print("\n  by gold label:")
        for label in ("SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"):
            total = per_kind_total[label]
            if total == 0:
                continue
            s, g = per_kind[label]
            print(f"    {label:<16} sym={s}/{total}  gnn={g}/{total}")

    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()

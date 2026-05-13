"""Train the sheaf GNN on synthgen and compare to the 88.5% symbolic baseline.

Design:
  • 2000 training samples, 400 held-out test samples.
  • Identical synthgen config for both (different seeds for the splits).
  • Report: overall accuracy, per-kind accuracy, confusion matrix.
  • Target: >= 95% overall. Non-negotiable floor: must beat 88.5%.
"""

from __future__ import annotations

import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "cognition", "src"))

from cognition.cassette.synthgen import (
    SynthGenConfig, RelationSpec, generate, summarize, _default_config,
)
from cognition.cassette.gnn_encoder import (
    SheafHypergraphEncoder, batch_from_synth, train, evaluate,
    IDX_TO_VERDICT,
)


def make_split(base_config: SynthGenConfig, n: int, seed: int) -> list:
    cfg = SynthGenConfig(
        n_actors=base_config.n_actors,
        n_features=base_config.n_features,
        relations=base_config.relations,
        n_samples=n,
        chain_length_min=base_config.chain_length_min,
        chain_length_max=base_config.chain_length_max,
        chain_length_bias=base_config.chain_length_bias,
        polarity_flip_rate=base_config.polarity_flip_rate,
        anomaly_rate=base_config.anomaly_rate,
        cycle_rate=base_config.cycle_rate,
        noise_edge_rate=base_config.noise_edge_rate,
        seed=seed,
        description=base_config.description,
    )
    return generate(cfg)


def main():
    print("═" * 72)
    print("  Sheaf GNN training probe — locked against 88.5% symbolic baseline")
    print("═" * 72)

    base = _default_config()
    print(f"\nconfig: {len(base.relations)} relations, "
          f"{base.n_actors} actors, {base.n_features} features")

    # Splits.
    train_graphs = make_split(base, n=2000, seed=7)
    val_graphs   = make_split(base, n=300,  seed=11)
    test_graphs  = make_split(base, n=400,  seed=17)
    print(f"splits: train={len(train_graphs)} val={len(val_graphs)} "
          f"test={len(test_graphs)}")

    # Relation kinds lookup (agent would provide this; here we reuse config).
    relation_kinds = {r.name: r.kind for r in base.relations}

    print("\n── train-set distribution ──")
    for k, v in summarize(train_graphs).items():
        print(f"  {k}: {v}")

    # Build batches.
    train_batch = batch_from_synth(train_graphs, relation_kinds)
    val_batch   = batch_from_synth(val_graphs, relation_kinds)
    test_batch  = batch_from_synth(test_graphs, relation_kinds)

    print(f"\ntrain_edges shape: {tuple(train_batch.edges.shape)}")

    # Model.
    model = SheafHypergraphEncoder(hidden_dim=64, n_layers=3)
    print(f"model params: {model.n_params():,}")

    # Train.
    print("\n── training ──")
    t0 = time.perf_counter()
    result = train(model, train_batch, val_batch,
                    epochs=40, batch_size=64, lr=2e-3, verbose=True)
    wall = time.perf_counter() - t0
    print(f"training wall: {wall:.0f}s")
    print(f"best val acc: {result.best_val_acc:.3f} @ epoch {result.best_epoch}")

    # Test.
    print("\n── test ──")
    out = evaluate(model, test_batch)
    preds, gold = out["preds"], out["gold"]
    correct = (preds == gold).sum()
    total = len(gold)
    acc = correct / total
    print(f"  overall: {correct}/{total} = {acc:.1%}  "
          f"(baseline 88.5%, delta {acc - 0.885:+.1%})")

    # Per-kind accuracy.
    from collections import defaultdict
    kind_stats = defaultdict(lambda: [0, 0])
    for g, p, y in zip(test_graphs, preds, gold):
        kind_stats[g.kind][1] += 1
        if p == y:
            kind_stats[g.kind][0] += 1

    print("\n  by kind:")
    # Prior symbolic baseline per-kind for comparison (from probe_synthgen.py):
    symbolic_per_kind = {
        "clean": 0.93, "disconnected": 0.96, "cyclic": 0.92,
        "retracted": 0.86, "anomaly": 0.06,
    }
    for kind in ("clean", "disconnected", "cyclic", "retracted", "anomaly"):
        a, t = kind_stats[kind]
        gnn_acc = a / max(1, t)
        sym_acc = symbolic_per_kind.get(kind, 0.0)
        delta = gnn_acc - sym_acc
        print(f"    {kind:<14} {a}/{t} = {gnn_acc:.0%}  "
              f"(sym {sym_acc:.0%}, Δ{delta:+.0%})")

    # Confusion matrix.
    print("\n  confusion (gold → predicted):")
    verdicts = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
    labels = ["SUP", "REF", "NOT"]
    confusion = Counter()
    for p, y in zip(preds, gold):
        confusion[(IDX_TO_VERDICT[y], IDX_TO_VERDICT[p])] += 1
    for g_label in verdicts:
        row = [confusion.get((g_label, p_label), 0) for p_label in verdicts]
        print(f"    {g_label:<16} → " + "  ".join(
            f"{labels[i]}:{n:>3}" for i, n in enumerate(row)
        ))

    # Headline.
    print("\n" + "─" * 72)
    if acc >= 0.95:
        print(f"  ✓ GNN hits {acc:.0%} — exceeds 95% target, clearly beats baseline")
    elif acc > 0.885:
        print(f"  ~ GNN at {acc:.0%} beats baseline ({acc - 0.885:+.0%}) but "
              f"under 95% target — acceptable, room to improve")
    else:
        print(f"  ✗ GNN at {acc:.0%} does NOT beat 88.5% symbolic baseline — "
              f"something is wrong")


if __name__ == "__main__":
    main()

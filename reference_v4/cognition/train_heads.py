"""Train multi-task classification heads on the frozen SPLADE backbone.

One script, one dataset, all 4 heads trained simultaneously.

Usage:
    python train_heads.py --n-samples 5000 --epochs 10 --batch-size 64
    python train_heads.py --from-cache data/training_samples.json  # skip regeneration

Output:
    cognition/src/cognition/model/heads.pt  (130KB)
"""

from __future__ import annotations

import sys
import json
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from cognition.encoder import SpladeEncoder
from cognition.heads import (
    CognitionHeads, TrainingSample, generate_training_data,
)


# ═══════════════════════════════════════════════════════════════════════
# DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════

def load_fever_for_training(limit: int = 3000) -> list[dict]:
    """Load FEVER gold evidence for training data generation."""
    from datasets import load_dataset

    print(f"  Loading FEVER gold evidence (limit={limit})...")
    ds = load_dataset("copenlu/fever_gold_evidence", split="train", streaming=True)
    claims = []
    for row in ds:
        claims.append({
            "claim": row["claim"],
            "label": row["label"],
            "evidence": row["evidence"],
        })
        if len(claims) >= limit:
            break
    print(f"  Loaded {len(claims)} claims from FEVER train split.")
    return claims


def samples_to_json(samples: list[TrainingSample]) -> list[dict]:
    """Serialize training samples to JSON-compatible dicts."""
    return [
        {
            "premise": s.premise,
            "hypothesis": s.hypothesis,
            "nli_label": s.nli_label,
            "relevance_label": s.relevance_label,
            "polarity_label": s.polarity_label,
            "relation_type_label": s.relation_type_label,
        }
        for s in samples
    ]


def samples_from_json(data: list[dict]) -> list[TrainingSample]:
    """Deserialize training samples from JSON."""
    return [
        TrainingSample(
            premise=d["premise"],
            hypothesis=d["hypothesis"],
            nli_label=d.get("nli_label"),
            relevance_label=d.get("relevance_label"),
            polarity_label=d.get("polarity_label"),
            relation_type_label=d.get("relation_type_label"),
        )
        for d in data
    ]


# ═══════════════════════════════════════════════════════════════════════
# CLS ENCODING
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def encode_all_cls(encoder: SpladeEncoder, texts: list[str],
                   batch_size: int = 64) -> torch.Tensor:
    """Extract [CLS] embeddings from the frozen backbone for all texts."""
    all_cls = []
    device = encoder.device

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = encoder.tokenizer(
            batch, max_length=encoder.max_length, padding=True,
            truncation=True, return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        outputs = encoder.model.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_emb = outputs.last_hidden_state[:, 0, :]  # (batch, 128)
        all_cls.append(cls_emb.cpu())

    return torch.cat(all_cls, dim=0)


# ═══════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════

def train(
    samples: list[TrainingSample],
    encoder: SpladeEncoder,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_split: float = 0.1,
    save_path: Path | None = None,
) -> CognitionHeads:
    """Train all 4 heads simultaneously on multi-task data.

    Strategy:
      1. Pre-compute ALL [CLS] embeddings (backbone is frozen, so we only
         need one forward pass through the encoder).
      2. Train heads on the cached embeddings — pure linear layers, fast.
    """
    print("\n═══ Training Multi-Task Heads ═══")
    print(f"  Samples: {len(samples)}")
    print(f"  Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")

    # Step 1: Collect all unique texts and encode once
    print("\n  [1/3] Encoding texts with frozen backbone...")
    t0 = time.time()

    # Gather all unique texts (premises and hypotheses)
    premise_texts = [s.premise for s in samples]
    hypothesis_texts = [s.hypothesis for s in samples if s.hypothesis]

    all_texts = list(set(premise_texts + hypothesis_texts))
    text_to_idx = {t: i for i, t in enumerate(all_texts)}
    print(f"    {len(all_texts)} unique texts to encode...")

    all_cls = encode_all_cls(encoder, all_texts, batch_size=batch_size)
    print(f"    Encoded in {time.time() - t0:.1f}s → shape {all_cls.shape}")

    # Step 2: Build per-sample tensors
    print("\n  [2/3] Building training tensors...")

    # For each sample, look up its premise/hypothesis CLS embedding
    premise_cls = torch.stack([all_cls[text_to_idx[s.premise]] for s in samples])
    hypothesis_cls = torch.stack([
        all_cls[text_to_idx[s.hypothesis]] if s.hypothesis else torch.zeros(128)
        for s in samples
    ])

    # Labels (use -1 for N/A so we can mask in loss)
    nli_labels = torch.tensor([s.nli_label if s.nli_label is not None else -1 for s in samples])
    rel_labels = torch.tensor([s.relevance_label if s.relevance_label is not None else -1 for s in samples])
    pol_labels = torch.tensor([s.polarity_label if s.polarity_label is not None else -1 for s in samples])
    rtype_labels = torch.tensor([s.relation_type_label if s.relation_type_label is not None else -1 for s in samples])

    # Train/val split
    n = len(samples)
    n_val = int(n * val_split)
    n_train = n - n_val
    indices = torch.randperm(n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    print(f"    Train: {n_train}, Val: {n_val}")

    # Step 3: Train
    print("\n  [3/3] Training heads...")
    hidden_dim = all_cls.shape[1]  # should be 128
    heads = CognitionHeads(hidden_dim=hidden_dim)

    optimizer = torch.optim.Adam(heads.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss(ignore_index=-1)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        heads.train()
        # Shuffle training indices
        perm = train_idx[torch.randperm(n_train)]

        epoch_losses = {"nli": 0.0, "rel": 0.0, "pol": 0.0, "rtype": 0.0}
        n_batches = 0

        for i in range(0, n_train, batch_size):
            batch_idx = perm[i:i + batch_size]

            # Get batch data
            b_premise = premise_cls[batch_idx]
            b_hyp = hypothesis_cls[batch_idx]
            b_nli = nli_labels[batch_idx]
            b_rel = rel_labels[batch_idx]
            b_pol = pol_labels[batch_idx]
            b_rtype = rtype_labels[batch_idx]

            optimizer.zero_grad()
            loss = torch.tensor(0.0)

            # NLI head: needs premise + hypothesis (pair task)
            nli_mask = b_nli >= 0
            if nli_mask.any():
                nli_logits = heads.nli(b_premise[nli_mask], b_hyp[nli_mask])
                nli_loss = ce_loss(nli_logits, b_nli[nli_mask])
                loss = loss + nli_loss
                epoch_losses["nli"] += nli_loss.item()

            # Relevance head: premise + hypothesis (pair task)
            rel_mask = b_rel >= 0
            if rel_mask.any():
                rel_logits = heads.relevance(b_hyp[rel_mask], b_premise[rel_mask])
                rel_loss = ce_loss(rel_logits, b_rel[rel_mask])
                loss = loss + rel_loss
                epoch_losses["rel"] += rel_loss.item()

            # Polarity head: premise only (single sentence)
            pol_mask = b_pol >= 0
            if pol_mask.any():
                pol_logits = heads.polarity(b_premise[pol_mask])
                pol_loss = ce_loss(pol_logits, b_pol[pol_mask])
                loss = loss + pol_loss
                epoch_losses["pol"] += pol_loss.item()

            # Relation type head: premise only (single sentence)
            rtype_mask = b_rtype >= 0
            if rtype_mask.any():
                rtype_logits = heads.relation_type(b_premise[rtype_mask])
                rtype_loss = ce_loss(rtype_logits, b_rtype[rtype_mask])
                loss = loss + rtype_loss
                epoch_losses["rtype"] += rtype_loss.item()

            if loss.requires_grad:
                loss.backward()
                optimizer.step()
            n_batches += 1

        # Validation
        heads.eval()
        with torch.no_grad():
            val_loss = torch.tensor(0.0)
            b_premise = premise_cls[val_idx]
            b_hyp = hypothesis_cls[val_idx]

            nli_mask = nli_labels[val_idx] >= 0
            if nli_mask.any():
                nli_logits = heads.nli(b_premise[nli_mask], b_hyp[nli_mask])
                val_loss = val_loss + ce_loss(nli_logits, nli_labels[val_idx][nli_mask])

            rel_mask = rel_labels[val_idx] >= 0
            if rel_mask.any():
                rel_logits = heads.relevance(b_hyp[rel_mask], b_premise[rel_mask])
                val_loss = val_loss + ce_loss(rel_logits, rel_labels[val_idx][rel_mask])

            pol_mask = pol_labels[val_idx] >= 0
            if pol_mask.any():
                pol_logits = heads.polarity(b_premise[pol_mask])
                val_loss = val_loss + ce_loss(pol_logits, pol_labels[val_idx][pol_mask])

            rtype_mask = rtype_labels[val_idx] >= 0
            if rtype_mask.any():
                rtype_logits = heads.relation_type(b_premise[rtype_mask])
                val_loss = val_loss + ce_loss(rtype_logits, rtype_labels[val_idx][rtype_mask])

        val_loss_val = val_loss.item()

        # Per-head train loss
        avg_losses = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}
        print(f"    Epoch {epoch+1:2d}/{epochs} | "
              f"NLI={avg_losses['nli']:.3f} Rel={avg_losses['rel']:.3f} "
              f"Pol={avg_losses['pol']:.3f} RT={avg_losses['rtype']:.3f} | "
              f"Val={val_loss_val:.3f}")

        if val_loss_val < best_val_loss:
            best_val_loss = val_loss_val
            best_state = {k: v.clone() for k, v in heads.state_dict().items()}

    # Restore best
    if best_state:
        heads.load_state_dict(best_state)
    heads.eval()

    # Save
    if save_path:
        heads.save(save_path)
        print(f"\n  Saved heads to {save_path}/heads.pt")

    # Report param counts
    counts = heads.param_count()
    print(f"\n  Parameters: {counts}")

    return heads


# ═══════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def evaluate_heads(heads: CognitionHeads, encoder: SpladeEncoder,
                   samples: list[TrainingSample], batch_size: int = 64):
    """Quick accuracy evaluation on held-out samples."""
    print("\n═══ Evaluation ═══")

    # Encode
    premise_texts = [s.premise for s in samples]
    hypothesis_texts = [s.hypothesis for s in samples if s.hypothesis]
    all_texts = list(set(premise_texts + hypothesis_texts))
    text_to_idx = {t: i for i, t in enumerate(all_texts)}
    all_cls = encode_all_cls(encoder, all_texts, batch_size=batch_size)

    premise_cls = torch.stack([all_cls[text_to_idx[s.premise]] for s in samples])
    hypothesis_cls = torch.stack([
        all_cls[text_to_idx[s.hypothesis]] if s.hypothesis else torch.zeros(128)
        for s in samples
    ])

    heads.eval()
    with torch.no_grad():
        # NLI accuracy
        nli_samples = [(i, s) for i, s in enumerate(samples) if s.nli_label is not None]
        if nli_samples:
            idx = [i for i, _ in nli_samples]
            labels = [s.nli_label for _, s in nli_samples]
            logits = heads.nli(premise_cls[idx], hypothesis_cls[idx])
            preds = logits.argmax(dim=-1).tolist()
            nli_acc = sum(1 for p, l in zip(preds, labels) if p == l) / len(labels)
            print(f"  NLI accuracy:      {nli_acc:.1%} ({len(labels)} samples)")

        # Relevance accuracy
        rel_samples = [(i, s) for i, s in enumerate(samples) if s.relevance_label is not None]
        if rel_samples:
            idx = [i for i, _ in rel_samples]
            labels = [s.relevance_label for _, s in rel_samples]
            logits = heads.relevance(hypothesis_cls[idx], premise_cls[idx])
            preds = logits.argmax(dim=-1).tolist()
            rel_acc = sum(1 for p, l in zip(preds, labels) if p == l) / len(labels)
            print(f"  Relevance accuracy: {rel_acc:.1%} ({len(labels)} samples)")

        # Polarity accuracy
        pol_samples = [(i, s) for i, s in enumerate(samples) if s.polarity_label is not None]
        if pol_samples:
            idx = [i for i, _ in pol_samples]
            labels = [s.polarity_label for _, s in pol_samples]
            logits = heads.polarity(premise_cls[idx])
            preds = logits.argmax(dim=-1).tolist()
            pol_acc = sum(1 for p, l in zip(preds, labels) if p == l) / len(labels)
            print(f"  Polarity accuracy:  {pol_acc:.1%} ({len(labels)} samples)")

        # Relation type accuracy
        rt_samples = [(i, s) for i, s in enumerate(samples) if s.relation_type_label is not None]
        if rt_samples:
            idx = [i for i, _ in rt_samples]
            labels = [s.relation_type_label for _, s in rt_samples]
            logits = heads.relation_type(premise_cls[idx])
            preds = logits.argmax(dim=-1).tolist()
            rt_acc = sum(1 for p, l in zip(preds, labels) if p == l) / len(labels)
            print(f"  RelType accuracy:   {rt_acc:.1%} ({len(labels)} samples)")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train multi-task heads on SPLADE backbone")
    parser.add_argument("--n-samples", type=int, default=5000,
                        help="Number of training samples to generate")
    parser.add_argument("--n-fever-claims", type=int, default=3000,
                        help="FEVER claims to load for data generation")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--from-cache", type=str, default=None,
                        help="Load cached training samples from JSON")
    parser.add_argument("--save-cache", type=str, default=None,
                        help="Save generated training samples to JSON")
    parser.add_argument("--output", type=str,
                        default=str(Path(__file__).parent / "src" / "cognition" / "model"),
                        help="Directory to save heads.pt")
    args = parser.parse_args()

    # Load or generate training data
    if args.from_cache:
        print(f"Loading cached samples from {args.from_cache}...")
        with open(args.from_cache) as f:
            data = json.load(f)
        samples = samples_from_json(data)
    else:
        fever_claims = load_fever_for_training(limit=args.n_fever_claims)
        print(f"  Generating {args.n_samples} multi-task training samples...")
        samples = generate_training_data(fever_claims, n_samples=args.n_samples)
        print(f"  Generated {len(samples)} samples.")

        if args.save_cache:
            cache_path = Path(args.save_cache)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(samples_to_json(samples), f)
            print(f"  Cached to {cache_path}")

    # Split: 90% train, 10% eval
    n_eval = max(100, len(samples) // 10)
    train_samples = samples[:-n_eval]
    eval_samples = samples[-n_eval:]

    # Load encoder (frozen backbone)
    print("\nLoading SPLADE encoder (frozen)...")
    encoder = SpladeEncoder()
    print(f"  Device: {encoder.device}")
    print(f"  Hidden dim: {encoder.model.config.hidden_size}")

    # Train
    save_path = Path(args.output)
    heads = train(
        train_samples, encoder,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=save_path,
    )

    # Evaluate on held-out set
    evaluate_heads(heads, encoder, eval_samples, batch_size=args.batch_size)

    # Quick sanity: test NLI on a few manual examples
    print("\n═══ Sanity Check (manual NLI) ═══")
    test_pairs = [
        ("Albert Einstein was born in Ulm, Germany.", "Einstein was born in Germany.", "entail"),
        ("Albert Einstein was born in Ulm, Germany.", "Einstein was born in France.", "contradict"),
        ("The cat sat on the mat.", "Einstein was born in Germany.", "neutral"),
    ]
    test_premises = [p for p, h, _ in test_pairs]
    test_hyps = [h for p, h, _ in test_pairs]
    all_test = list(set(test_premises + test_hyps))
    test_cls = encode_all_cls(encoder, all_test, batch_size=8)
    test_map = {t: i for i, t in enumerate(all_test)}

    heads.eval()
    with torch.no_grad():
        for premise, hyp, expected in test_pairs:
            p_cls = test_cls[test_map[premise]].unsqueeze(0)
            h_cls = test_cls[test_map[hyp]].unsqueeze(0)
            logits = heads.nli(p_cls, h_cls)
            probs = torch.softmax(logits, dim=-1)[0]
            pred_idx = probs.argmax().item()
            labels = ["entail", "neutral", "contradict"]
            print(f"  P: {premise[:50]}...")
            print(f"  H: {hyp[:50]}...")
            print(f"  → {labels[pred_idx]} (probs: E={probs[0]:.2f} N={probs[1]:.2f} C={probs[2]:.2f}) expected={expected}")
            print()


if __name__ == "__main__":
    main()

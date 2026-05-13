"""Train multilingual SPLADE on xlm-roberta-base for EN + KO + JA.

Single-stage joint training using SpladeLoss + SparseMultipleNegativesRankingLoss
on ~400K+ triples per language from high-quality parquet datasets:

  EN: sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1 (502K)
  JA: bclavie/mmarco-japanese-hard-negatives (391K)
  KO: nlpai-lab/ko-triplet-v1.0 (744K)

Usage:
    python train_splade_multilingual.py                    # full run
    python train_splade_multilingual.py --max-samples 5000 --epochs 1  # test
    python train_splade_multilingual.py --checkpoint models/splade-xlmr-multilingual/checkpoint-1000

Output:
    models/splade-xlmr-multilingual/final/
"""
from __future__ import annotations
import argparse
from pathlib import Path
from urllib.parse import urlsplit

from datasets import Dataset, load_dataset, concatenate_datasets

from sentence_transformers import (
    SparseEncoder,
    SparseEncoderTrainer,
    SparseEncoderTrainingArguments,
    SparseEncoderModelCardData,
)
try:
    from sentence_transformers.sparse_encoder.modules import MLMTransformer, SpladePooling
except ImportError:
    from sentence_transformers.sparse_encoder.models import MLMTransformer, SpladePooling
from sentence_transformers.sparse_encoder.losses import (
    SpladeLoss,
    SparseMultipleNegativesRankingLoss,
)

BACKBONE = "xlm-roberta-base"
OUTPUT_DIR = "models/splade-xlmr-multilingual"
EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM = 8
LR = 5e-5
WARMUP = 0.05
Q_REG = 5e-1
D_REG = 3e-1
MAX_SAMPLES = 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--backbone", type=str, default=BACKBONE)
    p.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    p.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--grad-accum", type=int, default=GRAD_ACCUM)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--bf16", action="store_true", default=False)
    return p.parse_args()


def load_en(max_samples: int) -> Dataset:
    split = f"train[:{max_samples}]" if max_samples > 0 else "train"
    ds = load_dataset(
        "sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1",
        "triplet", split=split,
    )
    print(f"EN: {len(ds)} triples")
    return ds


def load_ja(max_samples: int) -> Dataset:
    split = f"train[:{max_samples}]" if max_samples > 0 else "train"
    ds = load_dataset("bclavie/mmarco-japanese-hard-negatives", split=split)
    rows = []
    for row in ds:
        q = row["query"]
        positives = row.get("positives") or []
        negatives = row.get("negatives") or row.get("bm25_negatives") or []
        if positives and negatives:
            rows.append({"query": q, "positive": positives[0], "negative": negatives[0]})
    out = Dataset.from_list(rows)
    print(f"JA: {len(out)} triples (from {len(ds)} rows)")
    return out


def load_ko(max_samples: int) -> Dataset:
    split = f"train[:{max_samples}]" if max_samples > 0 else "train"
    ds = load_dataset("nlpai-lab/ko-triplet-v1.0", split=split)
    ds = ds.rename_columns({"document": "positive", "hard_negative": "negative"})
    print(f"KO: {len(ds)} triples")
    return ds


def build_model(backbone: str) -> SparseEncoder:
    model = SparseEncoder(
        modules=[
            MLMTransformer(backbone),
            SpladePooling(pooling_strategy="max", chunk_size=64),
        ],
        model_card_data=SparseEncoderModelCardData(
            language=["en", "ko", "ja"],
            license="apache-2.0",
            model_name="SPLADE-XLM-R-Multilingual",
        ),
    )
    print(f"Model: {sum(t.numel() for t in model.parameters()) / 1e6:.1f}M params")
    return model


def train(args):
    if args.checkpoint:
        model = SparseEncoder(args.checkpoint)
    else:
        model = build_model(args.backbone)

    en_ds = load_en(args.max_samples)
    ja_ds = load_ja(args.max_samples)
    ko_ds = load_ko(args.max_samples)

    combined = concatenate_datasets([en_ds, ja_ds, ko_ds]).shuffle(seed=42)

    def _safe_text(example):
        for col in ("query", "positive", "negative"):
            try:
                urlsplit(example[col])
            except ValueError:
                return False
        return True

    before = len(combined)
    combined = combined.filter(_safe_text, num_proc=4)
    if len(combined) < before:
        print(f"Filtered {before - len(combined)} bad rows")
    print(f"Combined: {len(combined)} triples")

    loss = SpladeLoss(
        model=model,
        loss=SparseMultipleNegativesRankingLoss(model=model),
        query_regularizer_weight=Q_REG,
        document_regularizer_weight=D_REG,
    )

    training_args = SparseEncoderTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=WARMUP,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        fp16=args.fp16 and not args.bf16,
        bf16=args.bf16,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=2000,
        logging_strategy="steps",
        logging_steps=100,
        save_total_limit=2,
        run_name="splade-xlmr-multilingual",
        dataloader_num_workers=4,
    )

    trainer = SparseEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=combined.select_columns(["query", "positive", "negative"]),
        loss=loss,
    )

    trainer.train(resume_from_checkpoint=args.checkpoint)

    final_path = Path(args.output_dir) / "final"
    model.save_pretrained(str(final_path))
    print(f"Saved: {final_path}")
    return str(final_path)


def export_for_cognition(model_path: str, output_path: str | None = None):
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    output_path = output_path or str(Path(model_path).parent / "cognition-export")
    mlm_path = Path(model_path)
    for subdir in ["1_MLMTransformer", "0_MLMTransformer", ""]:
        candidate = mlm_path / subdir if subdir else mlm_path
        if (candidate / "config.json").exists():
            mlm_path = candidate
            break

    tokenizer = AutoTokenizer.from_pretrained(str(mlm_path))
    model_mlm = AutoModelForMaskedLM.from_pretrained(str(mlm_path))
    Path(output_path).mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_path)
    model_mlm.save_pretrained(output_path)

    for label, text in [("EN", "Toyota invested in battery technology"),
                        ("KO", "도요타가 배터리 기술에 투자했다"),
                        ("JA", "トヨタがバッテリー技術に投資した")]:
        tokens = tokenizer.tokenize(text)
        print(f"  {label}: {len(tokens)} tokens")
    print(f"Exported: {output_path}")


def main():
    args = parse_args()
    final_path = train(args)
    export_for_cognition(final_path, str(Path(args.output_dir) / "cognition-export"))


if __name__ == "__main__":
    main()

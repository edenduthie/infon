"""Run benchmarks on real FEVER and HoVer data.

Uses:
  - FEVER: copenlu/fever_gold_evidence (claims + gold evidence text inline)
  - HoVer: Dzeniks/hover (claims + evidence text + labels)

Both include evidence text directly, so no Wikipedia dump needed.
"""

from __future__ import annotations

import json
import sys
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from collections import defaultdict

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from infon import Cognition, CognitionConfig, Infon, Edge, QueryResult
from infon.category import SchemaDiscovery
from infon.dempster_shafer import (
    verify_claim, VerificationVerdict,
    MassFunction, combine_multiple,
)
from infon.heads import CognitionHeads

SCHEMA_PATH = Path(__file__).parent / "schemas" / "wikipedia_general.json"
RESULTS_DIR = Path(__file__).parent / "results"


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_fever_hf(limit: int = 200) -> list[dict]:
    """Load FEVER claims with gold evidence from HuggingFace."""
    from datasets import load_dataset

    ds = load_dataset("copenlu/fever_gold_evidence", split="validation", streaming=True)
    claims = []
    for row in ds:
        claims.append({
            "id": row["original_id"],
            "claim": row["claim"],
            "label": row["label"],
            "evidence": row["evidence"],  # list of [doc_title, sent_id, text]
            "verifiable": row.get("verifiable", "VERIFIABLE"),
        })
        if len(claims) >= limit:
            break
    return claims


def load_hover_hf(limit: int = 200, split: str = "train") -> list[dict]:
    """Load HoVer claims with evidence from HuggingFace."""
    from datasets import load_dataset

    ds = load_dataset("Dzeniks/hover", split=split, streaming=True)
    claims = []
    for row in ds:
        claims.append({
            "uid": row["id"],
            "claim": row["claim"],
            "label": "SUPPORTED" if row["label"] == 1 else "NOT_SUPPORTED",
            "evidence_text": row["evidence"],  # full evidence paragraph
        })
        if len(claims) >= limit:
            break
    return claims


# ═══════════════════════════════════════════════════════════════════════
# FEVER EVALUATION
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FEVERClaimResult:
    claim_id: int
    claim: str
    gold_label: str
    gold_evidence_texts: list[str] = field(default_factory=list)

    # Cognition results
    infons: list = field(default_factory=list)
    n_infons: int = 0

    # Metrics
    predicted_label: str = ""
    label_correct: bool = False

    # Evidence overlap (token-level since we have text, not sent IDs)
    evidence_precision: float = 0.0
    evidence_recall: float = 0.0
    evidence_f1: float = 0.0


def evaluate_fever_infon(claims: list[dict], schema_path: Path,
                              condition: str, top_k: int = 20) -> dict:
    """Evaluate infon on FEVER claims.

    Since copenlu/fever_gold_evidence gives us evidence TEXT directly,
    we ingest the evidence, query with the claim, and measure:
    1. Whether the infons capture the gold evidence content (token overlap)
    2. Whether polarity detection gets the label right
    """
    print(f"\n  [{condition}] Evaluating {len(claims)} FEVER claims...")
    t0 = time.time()

    results = []
    for i, claim in enumerate(claims):
        if (i + 1) % 50 == 0:
            print(f"    [{i+1}/{len(claims)}]")

        claim_text = claim["claim"]
        gold_label = claim["label"]
        gold_evidence = claim["evidence"]

        # Parse evidence: each item is [doc_title, sent_idx, evidence_text]
        evidence_texts = []
        documents = []
        for ev in gold_evidence:
            if len(ev) >= 3:
                doc_title, sent_idx, ev_text = ev[0], ev[1], ev[2]
                evidence_texts.append(ev_text)
                documents.append({
                    "text": ev_text,
                    "id": f"{doc_title}__sent_{sent_idx}",
                })

        if not documents:
            results.append(FEVERClaimResult(
                claim_id=claim["id"], claim=claim_text,
                gold_label=gold_label, predicted_label="NOT ENOUGH INFO",
            ))
            continue

        # Fresh infon instance per claim
        config = CognitionConfig(
            schema_path=str(schema_path),
            db_path=":memory:",
            activation_threshold=0.2,
            min_confidence=0.03,
            top_k_per_role=5,
            default_top_k=top_k,
        )
        cog = Cognition(config)
        cog.ingest(documents)

        # Query with claim
        result = cog.query(claim_text, top_k=top_k, include_chains=False)
        cog.close()

        # Predict label via Dempster-Shafer belief combination
        verdict = verify_claim(
            result.infons,
            claim_anchors=result.anchors_activated,
            schema_types=cog.schema.types,
        )
        predicted_label = verdict.label

        # Evidence overlap: compare infon sentences against gold evidence
        p, r, f1 = _compute_token_overlap(result.infons, evidence_texts)

        results.append(FEVERClaimResult(
            claim_id=claim["id"],
            claim=claim_text,
            gold_label=gold_label,
            gold_evidence_texts=evidence_texts,
            n_infons=len(result.infons),
            predicted_label=predicted_label,
            label_correct=(predicted_label == gold_label),
            evidence_precision=p,
            evidence_recall=r,
            evidence_f1=f1,
        ))

    elapsed = time.time() - t0

    # Aggregate
    n = len(results)
    label_acc = sum(1 for r in results if r.label_correct) / n
    avg_p = sum(r.evidence_precision for r in results) / n
    avg_r = sum(r.evidence_recall for r in results) / n
    avg_f1 = sum(r.evidence_f1 for r in results) / n
    avg_infons = sum(r.n_infons for r in results) / n

    # Per-label breakdown
    by_label = {}
    for label in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]:
        subset = [r for r in results if r.gold_label == label]
        if subset:
            acc = sum(1 for r in subset if r.label_correct) / len(subset)
            by_label[label] = {"n": len(subset), "accuracy": acc}

    metrics = {
        "condition": condition,
        "n_claims": n,
        "time_s": elapsed,
        "label_accuracy": label_acc,
        "evidence_precision": avg_p,
        "evidence_recall": avg_r,
        "evidence_f1": avg_f1,
        "avg_infons": avg_infons,
        "by_label": by_label,
    }

    print(f"    Done in {elapsed:.1f}s")
    print(f"    Label acc={label_acc:.3f} "
          f"Evidence P={avg_p:.3f} R={avg_r:.3f} F1={avg_f1:.3f}")
    print(f"    By label: {by_label}")

    return metrics


def evaluate_fever_rag(claims: list[dict], top_k: int = 5) -> dict:
    """RAG baseline on FEVER.

    Dense encode claim + evidence sentences, retrieve by cosine similarity.
    """
    from baselines.rag_baseline import DenseEncoder

    print(f"\n  [rag] Evaluating {len(claims)} FEVER claims...")
    t0 = time.time()

    encoder = DenseEncoder()

    results = []
    for i, claim in enumerate(claims):
        if (i + 1) % 50 == 0:
            print(f"    [{i+1}/{len(claims)}]")

        claim_text = claim["claim"]
        gold_label = claim["label"]
        gold_evidence = claim["evidence"]

        evidence_texts = [ev[2] for ev in gold_evidence if len(ev) >= 3]
        if not evidence_texts:
            results.append(FEVERClaimResult(
                claim_id=claim["id"], claim=claim_text,
                gold_label=gold_label, predicted_label="NOT ENOUGH INFO",
            ))
            continue

        # Encode claim and evidence
        claim_emb = encoder.encode([claim_text])
        ev_embs = encoder.encode(evidence_texts)

        # Cosine similarity (already normalized)
        scores = (claim_emb @ ev_embs.T)[0]
        top_indices = scores.argsort()[::-1][:top_k]

        retrieved_texts = [evidence_texts[j] for j in top_indices if j < len(evidence_texts)]

        # Label: RAG uses similarity threshold
        max_score = float(scores.max()) if len(scores) > 0 else 0.0
        if max_score > 0.7:
            predicted_label = "SUPPORTS"
        elif max_score > 0.4:
            predicted_label = "REFUTES"  # uncertain
        else:
            predicted_label = "NOT ENOUGH INFO"

        # Token overlap of retrieved vs gold
        p, r, f1 = _compute_token_overlap_texts(retrieved_texts, evidence_texts)

        results.append(FEVERClaimResult(
            claim_id=claim["id"],
            claim=claim_text,
            gold_label=gold_label,
            gold_evidence_texts=evidence_texts,
            predicted_label=predicted_label,
            label_correct=(predicted_label == gold_label),
            evidence_precision=p,
            evidence_recall=r,
            evidence_f1=f1,
        ))

    elapsed = time.time() - t0
    n = len(results)
    label_acc = sum(1 for r in results if r.label_correct) / n
    avg_p = sum(r.evidence_precision for r in results) / n
    avg_r = sum(r.evidence_recall for r in results) / n
    avg_f1 = sum(r.evidence_f1 for r in results) / n

    by_label = {}
    for label in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]:
        subset = [r for r in results if r.gold_label == label]
        if subset:
            acc = sum(1 for r in subset if r.label_correct) / len(subset)
            by_label[label] = {"n": len(subset), "accuracy": acc}

    metrics = {
        "condition": "rag",
        "n_claims": n,
        "time_s": elapsed,
        "label_accuracy": label_acc,
        "evidence_precision": avg_p,
        "evidence_recall": avg_r,
        "evidence_f1": avg_f1,
        "by_label": by_label,
    }

    print(f"    Done in {elapsed:.1f}s")
    print(f"    Label acc={label_acc:.3f} "
          f"Evidence P={avg_p:.3f} R={avg_r:.3f} F1={avg_f1:.3f}")
    print(f"    By label: {by_label}")

    return metrics


# ═══════════════════════════════════════════════════════════════════════
# FEVER EVALUATION WITH TRAINED HEADS
# ═══════════════════════════════════════════════════════════════════════

def evaluate_fever_heads(claims: list[dict], schema_path: Path, top_k: int = 20) -> dict:
    """Evaluate FEVER using trained NLI + Relevance heads on SPLADE backbone.

    Instead of heuristic DS sources, we:
      1. Encode claim and each evidence sentence with the backbone
      2. Use Relevance head to filter irrelevant evidence
      3. Use NLI head on (evidence, claim) pairs to get mass functions
      4. Combine masses via Dempster's rule for the verdict
    """
    from infon.encoder import SpladeEncoder
    import torch

    print(f"\n  [heads] Evaluating {len(claims)} FEVER claims with trained heads...")
    t0 = time.time()

    # Load encoder and heads
    encoder = SpladeEncoder()
    heads_path = Path(__file__).parent.parent / "src" / "infon" / "model"
    heads = CognitionHeads.load(heads_path)

    results = []
    for i, claim in enumerate(claims):
        if (i + 1) % 50 == 0:
            print(f"    [{i+1}/{len(claims)}]")

        claim_text = claim["claim"]
        gold_label = claim["label"]
        gold_evidence = claim["evidence"]

        evidence_texts = [ev[2] for ev in gold_evidence if len(ev) >= 3]
        if not evidence_texts:
            results.append(FEVERClaimResult(
                claim_id=claim["id"], claim=claim_text,
                gold_label=gold_label, predicted_label="NOT ENOUGH INFO",
            ))
            continue

        # Encode claim + evidence with backbone
        all_texts = [claim_text] + evidence_texts
        cls_embs = heads.encode_cls(encoder, all_texts)
        claim_cls = cls_embs[0:1]  # (1, 128)
        evidence_cls = cls_embs[1:]  # (n_ev, 128)

        # Step 1: Relevance filtering
        claim_expanded = claim_cls.expand(evidence_cls.shape[0], -1)
        relevant_mask = heads.relevance.predict_relevant(
            claim_expanded, evidence_cls, threshold=0.4
        )
        relevant_indices = [j for j, r in enumerate(relevant_mask) if r]

        if not relevant_indices:
            # No relevant evidence → NEI
            results.append(FEVERClaimResult(
                claim_id=claim["id"], claim=claim_text,
                gold_label=gold_label,
                gold_evidence_texts=evidence_texts,
                n_infons=len(evidence_texts),
                predicted_label="NOT ENOUGH INFO",
                label_correct=(gold_label == "NOT ENOUGH INFO"),
                evidence_precision=0.0, evidence_recall=0.0, evidence_f1=0.0,
            ))
            continue

        # Step 2: NLI on relevant pairs → mass functions
        rel_evidence_cls = evidence_cls[relevant_indices]
        claim_for_nli = claim_cls.expand(rel_evidence_cls.shape[0], -1)
        masses = heads.nli.predict_mass(rel_evidence_cls, claim_for_nli)

        # Step 3: Combine masses via Dempster's rule
        # Use top-5 most decisive (lowest theta)
        decisive = sorted(masses, key=lambda m: m.theta)[:5]
        combined = combine_multiple(decisive) if decisive else MassFunction(theta=1.0)

        # Pignistic decision
        total_focal = combined.supports + combined.refutes + combined.uncertain
        if total_focal > 0:
            pig_s = combined.supports + combined.theta * (combined.supports / total_focal)
            pig_r = combined.refutes + combined.theta * (combined.refutes / total_focal)
        else:
            pig_s = combined.theta / 3
            pig_r = combined.theta / 3

        if pig_r > 0.15 and pig_r > pig_s:
            predicted_label = "REFUTES"
        elif pig_s > 0.25 and pig_s > pig_r:
            predicted_label = "SUPPORTS"
        else:
            predicted_label = "NOT ENOUGH INFO"

        # Evidence overlap
        rel_texts = [evidence_texts[j] for j in relevant_indices]
        p, r, f1 = _compute_token_overlap_texts(rel_texts, evidence_texts)

        results.append(FEVERClaimResult(
            claim_id=claim["id"],
            claim=claim_text,
            gold_label=gold_label,
            gold_evidence_texts=evidence_texts,
            n_infons=len(relevant_indices),
            predicted_label=predicted_label,
            label_correct=(predicted_label == gold_label),
            evidence_precision=p,
            evidence_recall=r,
            evidence_f1=f1,
        ))

    elapsed = time.time() - t0
    n = len(results)
    label_acc = sum(1 for r in results if r.label_correct) / n
    avg_p = sum(r.evidence_precision for r in results) / n
    avg_r = sum(r.evidence_recall for r in results) / n
    avg_f1 = sum(r.evidence_f1 for r in results) / n
    avg_infons = sum(r.n_infons for r in results) / n

    by_label = {}
    for label in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]:
        subset = [r for r in results if r.gold_label == label]
        if subset:
            acc = sum(1 for r in subset if r.label_correct) / len(subset)
            by_label[label] = {"n": len(subset), "accuracy": acc}

    metrics = {
        "condition": "heads",
        "n_claims": n,
        "time_s": elapsed,
        "label_accuracy": label_acc,
        "evidence_precision": avg_p,
        "evidence_recall": avg_r,
        "evidence_f1": avg_f1,
        "avg_infons": avg_infons,
        "by_label": by_label,
    }

    print(f"    Done in {elapsed:.1f}s")
    print(f"    Label acc={label_acc:.3f} "
          f"Evidence P={avg_p:.3f} R={avg_r:.3f} F1={avg_f1:.3f}")
    print(f"    By label: {by_label}")

    return metrics


# ═══════════════════════════════════════════════════════════════════════
# HoVer EVALUATION
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class HoVerClaimResult:
    claim_id: str
    claim: str
    gold_label: str
    gold_evidence_text: str = ""

    # Cognition results
    n_infons: int = 0
    n_edges: int = 0
    n_paths: int = 0

    # Metrics
    predicted_label: str = ""
    label_correct: bool = False
    evidence_precision: float = 0.0
    evidence_recall: float = 0.0
    evidence_f1: float = 0.0
    path_coverage: float = 0.0


def evaluate_hover_infon(claims: list[dict], schema_path: Path,
                              condition: str, chain_depth: int = 15) -> dict:
    """Evaluate infon on HoVer claims with hyperedge traversal.

    HoVer evidence is a paragraph spanning multiple documents. We split it
    into sentences, ingest with consolidation (to build NEXT edges),
    then query with chain traversal.
    """
    print(f"\n  [{condition}] Evaluating {len(claims)} HoVer claims...")
    t0 = time.time()

    results = []
    for i, claim in enumerate(claims):
        if (i + 1) % 50 == 0:
            print(f"    [{i+1}/{len(claims)}]")

        claim_text = claim["claim"]
        gold_label = claim["label"]
        evidence_text = claim.get("evidence_text", "")

        if not evidence_text.strip():
            results.append(HoVerClaimResult(
                claim_id=claim["uid"], claim=claim_text,
                gold_label=gold_label, predicted_label="NOT_SUPPORTED",
            ))
            continue

        # Split evidence into sentences for multi-hop ingestion
        from infon.extract import split_sentences
        ev_sentences = split_sentences(evidence_text)

        # Each sentence gets a synthetic doc_id to simulate multi-doc
        documents = []
        for j, sent in enumerate(ev_sentences):
            if sent.strip():
                documents.append({
                    "text": sent,
                    "id": f"doc_{j}",
                    "timestamp": f"2025-01-{j+1:02d}",
                })

        if not documents:
            results.append(HoVerClaimResult(
                claim_id=claim["uid"], claim=claim_text,
                gold_label=gold_label, predicted_label="NOT_SUPPORTED",
            ))
            continue

        # Fresh infon with consolidation for NEXT edges
        config = CognitionConfig(
            schema_path=str(schema_path),
            db_path=":memory:",
            activation_threshold=0.2,
            min_confidence=0.03,
            top_k_per_role=5,
            default_top_k=100,
            consolidation_interval=1,
        )
        cog = Cognition(config)
        cog.ingest(documents, consolidate_now=True)

        # Query with chain traversal
        result = cog.query(claim_text, top_k=50, include_chains=True)
        cog.close()

        # Count NEXT edges (multi-hop paths)
        next_edges = [e for e in result.edges if e.edge_type == "NEXT"]

        # Path coverage: how many evidence sentences are covered by infons?
        covered_docs = set()
        for infon in result.infons:
            covered_docs.add(infon.doc_id)
        path_coverage = len(covered_docs) / len(documents) if documents else 0.0

        # Evidence token overlap
        infon_sentences = list(set(inf.sentence for inf in result.infons))
        p, r, f1 = _compute_token_overlap_texts(infon_sentences, ev_sentences)

        # Label prediction via Dempster-Shafer
        verdict = verify_claim(
            result.infons,
            claim_anchors=result.anchors_activated,
            schema_types=cog.schema.types,
        )
        # Map FEVER labels to HoVer labels
        if verdict.label == "SUPPORTS":
            predicted_label = "SUPPORTED"
        else:
            predicted_label = "NOT_SUPPORTED"

        results.append(HoVerClaimResult(
            claim_id=claim["uid"],
            claim=claim_text,
            gold_label=gold_label,
            gold_evidence_text=evidence_text,
            n_infons=len(result.infons),
            n_edges=len(next_edges),
            n_paths=_count_paths(next_edges),
            predicted_label=predicted_label,
            label_correct=(predicted_label == gold_label),
            evidence_precision=p,
            evidence_recall=r,
            evidence_f1=f1,
            path_coverage=path_coverage,
        ))

    elapsed = time.time() - t0
    n = len(results)
    label_acc = sum(1 for r in results if r.label_correct) / n
    avg_p = sum(r.evidence_precision for r in results) / n
    avg_r = sum(r.evidence_recall for r in results) / n
    avg_f1 = sum(r.evidence_f1 for r in results) / n
    avg_coverage = sum(r.path_coverage for r in results) / n
    avg_edges = sum(r.n_edges for r in results) / n
    avg_paths = sum(r.n_paths for r in results) / n

    metrics = {
        "condition": condition,
        "n_claims": n,
        "time_s": elapsed,
        "label_accuracy": label_acc,
        "fact_precision": avg_p,
        "fact_recall": avg_r,
        "fact_f1": avg_f1,
        "path_coverage": avg_coverage,
        "avg_infons": sum(r.n_infons for r in results) / n,
        "avg_next_edges": avg_edges,
        "avg_paths": avg_paths,
    }

    print(f"    Done in {elapsed:.1f}s")
    print(f"    Label acc={label_acc:.3f} "
          f"Fact P={avg_p:.3f} R={avg_r:.3f} F1={avg_f1:.3f}")
    print(f"    Path coverage={avg_coverage:.3f} "
          f"Avg edges={avg_edges:.1f} Avg paths={avg_paths:.1f}")

    return metrics


def evaluate_hover_rag(claims: list[dict], top_k: int = 5) -> dict:
    """RAG baseline on HoVer — flat retrieval, no multi-hop structure."""
    from baselines.rag_baseline import DenseEncoder

    print(f"\n  [rag] Evaluating {len(claims)} HoVer claims...")
    t0 = time.time()

    encoder = DenseEncoder()

    results = []
    for i, claim in enumerate(claims):
        if (i + 1) % 50 == 0:
            print(f"    [{i+1}/{len(claims)}]")

        claim_text = claim["claim"]
        gold_label = claim["label"]
        evidence_text = claim.get("evidence_text", "")

        if not evidence_text.strip():
            results.append(HoVerClaimResult(
                claim_id=claim["uid"], claim=claim_text,
                gold_label=gold_label, predicted_label="NOT_SUPPORTED",
            ))
            continue

        # Split evidence into sentences
        from infon.extract import split_sentences
        ev_sentences = split_sentences(evidence_text)

        if not ev_sentences:
            results.append(HoVerClaimResult(
                claim_id=claim["uid"], claim=claim_text,
                gold_label=gold_label, predicted_label="NOT_SUPPORTED",
            ))
            continue

        # Dense retrieve: rank evidence sentences by similarity to claim
        claim_emb = encoder.encode([claim_text])
        ev_embs = encoder.encode(ev_sentences)

        scores = (claim_emb @ ev_embs.T)[0]
        top_indices = scores.argsort()[::-1][:top_k]
        retrieved = [ev_sentences[j] for j in top_indices if j < len(ev_sentences)]

        # Coverage: fraction of evidence sentences retrieved
        path_coverage = len(top_indices) / len(ev_sentences) if ev_sentences else 0.0

        # Token overlap
        p, r, f1 = _compute_token_overlap_texts(retrieved, ev_sentences)

        # Label
        max_score = float(scores.max()) if len(scores) > 0 else 0.0
        predicted_label = "SUPPORTED" if max_score > 0.5 else "NOT_SUPPORTED"

        results.append(HoVerClaimResult(
            claim_id=claim["uid"],
            claim=claim_text,
            gold_label=gold_label,
            gold_evidence_text=evidence_text,
            predicted_label=predicted_label,
            label_correct=(predicted_label == gold_label),
            evidence_precision=p,
            evidence_recall=r,
            evidence_f1=f1,
            path_coverage=path_coverage,
        ))

    elapsed = time.time() - t0
    n = len(results)
    label_acc = sum(1 for r in results if r.label_correct) / n
    avg_p = sum(r.evidence_precision for r in results) / n
    avg_r = sum(r.evidence_recall for r in results) / n
    avg_f1 = sum(r.evidence_f1 for r in results) / n
    avg_coverage = sum(r.path_coverage for r in results) / n

    metrics = {
        "condition": "rag",
        "n_claims": n,
        "time_s": elapsed,
        "label_accuracy": label_acc,
        "fact_precision": avg_p,
        "fact_recall": avg_r,
        "fact_f1": avg_f1,
        "path_coverage": avg_coverage,
    }

    print(f"    Done in {elapsed:.1f}s")
    print(f"    Label acc={label_acc:.3f} "
          f"Fact P={avg_p:.3f} R={avg_r:.3f} F1={avg_f1:.3f}")
    print(f"    Path coverage={avg_coverage:.3f}")

    return metrics


# ═══════════════════════════════════════════════════════════════════════
# HoVer EVALUATION WITH TRAINED HEADS
# ═══════════════════════════════════════════════════════════════════════

def evaluate_hover_heads(claims: list[dict], schema_path: Path) -> dict:
    """Evaluate HoVer using trained NLI + Relevance heads.

    Same approach as FEVER heads: encode, filter, NLI → DS → verdict.
    """
    from infon.encoder import SpladeEncoder
    from infon.extract import split_sentences
    import torch

    print(f"\n  [heads] Evaluating {len(claims)} HoVer claims with trained heads...")
    t0 = time.time()

    encoder = SpladeEncoder()
    heads_path = Path(__file__).parent.parent / "src" / "infon" / "model"
    heads = CognitionHeads.load(heads_path)

    results = []
    for i, claim in enumerate(claims):
        if (i + 1) % 50 == 0:
            print(f"    [{i+1}/{len(claims)}]")

        claim_text = claim["claim"]
        gold_label = claim["label"]
        evidence_text = claim.get("evidence_text", "")

        if not evidence_text.strip():
            results.append(HoVerClaimResult(
                claim_id=claim["uid"], claim=claim_text,
                gold_label=gold_label, predicted_label="NOT_SUPPORTED",
            ))
            continue

        ev_sentences = split_sentences(evidence_text)
        if not ev_sentences:
            results.append(HoVerClaimResult(
                claim_id=claim["uid"], claim=claim_text,
                gold_label=gold_label, predicted_label="NOT_SUPPORTED",
            ))
            continue

        # Encode all
        all_texts = [claim_text] + ev_sentences
        cls_embs = heads.encode_cls(encoder, all_texts)
        claim_cls = cls_embs[0:1]
        evidence_cls = cls_embs[1:]

        # Relevance filtering
        claim_expanded = claim_cls.expand(evidence_cls.shape[0], -1)
        relevant_mask = heads.relevance.predict_relevant(
            claim_expanded, evidence_cls, threshold=0.4
        )
        relevant_indices = [j for j, r in enumerate(relevant_mask) if r]

        if not relevant_indices:
            results.append(HoVerClaimResult(
                claim_id=claim["uid"], claim=claim_text,
                gold_label=gold_label, predicted_label="NOT_SUPPORTED",
                label_correct=(gold_label == "NOT_SUPPORTED"),
            ))
            continue

        # NLI on relevant pairs
        rel_evidence_cls = evidence_cls[relevant_indices]
        claim_for_nli = claim_cls.expand(rel_evidence_cls.shape[0], -1)
        masses = heads.nli.predict_mass(rel_evidence_cls, claim_for_nli)

        # Combine via Dempster
        decisive = sorted(masses, key=lambda m: m.theta)[:5]
        combined = combine_multiple(decisive) if decisive else MassFunction(theta=1.0)

        # Pignistic decision (binary for HoVer)
        total_focal = combined.supports + combined.refutes + combined.uncertain
        if total_focal > 0:
            pig_s = combined.supports + combined.theta * (combined.supports / total_focal)
        else:
            pig_s = combined.theta / 3

        predicted_label = "SUPPORTED" if pig_s > 0.3 else "NOT_SUPPORTED"

        # Token overlap
        rel_texts = [ev_sentences[j] for j in relevant_indices]
        p, r, f1 = _compute_token_overlap_texts(rel_texts, ev_sentences)

        results.append(HoVerClaimResult(
            claim_id=claim["uid"],
            claim=claim_text,
            gold_label=gold_label,
            gold_evidence_text=evidence_text,
            n_infons=len(relevant_indices),
            predicted_label=predicted_label,
            label_correct=(predicted_label == gold_label),
            evidence_precision=p,
            evidence_recall=r,
            evidence_f1=f1,
            path_coverage=len(relevant_indices) / len(ev_sentences),
        ))

    elapsed = time.time() - t0
    n = len(results)
    label_acc = sum(1 for r in results if r.label_correct) / n
    avg_p = sum(r.evidence_precision for r in results) / n
    avg_r = sum(r.evidence_recall for r in results) / n
    avg_f1 = sum(r.evidence_f1 for r in results) / n
    avg_coverage = sum(r.path_coverage for r in results) / n

    metrics = {
        "condition": "heads",
        "n_claims": n,
        "time_s": elapsed,
        "label_accuracy": label_acc,
        "fact_precision": avg_p,
        "fact_recall": avg_r,
        "fact_f1": avg_f1,
        "path_coverage": avg_coverage,
    }

    print(f"    Done in {elapsed:.1f}s")
    print(f"    Label acc={label_acc:.3f} "
          f"Fact P={avg_p:.3f} R={avg_r:.3f} F1={avg_f1:.3f}")
    print(f"    Path coverage={avg_coverage:.3f}")

    return metrics


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _predict_fever_label(infons: list, gold_label: str = "") -> str:
    """Predict FEVER label from infon polarity + confidence."""
    if not infons:
        return "NOT ENOUGH INFO"

    top_k = min(5, len(infons))
    top_infons = sorted(infons, key=lambda x: -x.confidence)[:top_k]

    affirmed_weight = sum(inf.confidence for inf in top_infons if inf.polarity == 1)
    negated_weight = sum(inf.confidence for inf in top_infons if inf.polarity == 0)
    total = affirmed_weight + negated_weight

    if total < 0.05:
        return "NOT ENOUGH INFO"

    negation_ratio = negated_weight / total if total > 0 else 0.0

    if negation_ratio > 0.4:
        return "REFUTES"
    elif affirmed_weight > 0.1:
        return "SUPPORTS"
    else:
        return "NOT ENOUGH INFO"


def _predict_hover_label(infons: list) -> str:
    """Predict HoVer label from infons."""
    if not infons:
        return "NOT_SUPPORTED"
    high_conf = [inf for inf in infons if inf.confidence > 0.15]
    affirmed = [inf for inf in high_conf if inf.polarity == 1]
    if len(affirmed) >= 2:
        return "SUPPORTED"
    return "NOT_SUPPORTED"


def _tokenize(text: str) -> set[str]:
    """Simple whitespace + lowercase tokenization."""
    import re
    return set(re.findall(r'\b\w+\b', text.lower()))


def _compute_token_overlap(infons: list, evidence_texts: list[str]
                           ) -> tuple[float, float, float]:
    """Token-level overlap between infon sentences and gold evidence."""
    if not infons or not evidence_texts:
        return 0.0, 0.0, 0.0

    infon_tokens = set()
    for inf in infons:
        infon_tokens.update(_tokenize(inf.sentence))

    gold_tokens = set()
    for text in evidence_texts:
        gold_tokens.update(_tokenize(text))

    if not infon_tokens or not gold_tokens:
        return 0.0, 0.0, 0.0

    # Remove stopwords for meaningful overlap
    stopwords = {"the", "a", "an", "is", "was", "are", "were", "be", "been",
                 "being", "have", "has", "had", "do", "does", "did", "will",
                 "would", "could", "should", "may", "might", "shall", "can",
                 "of", "in", "to", "for", "with", "on", "at", "from", "by",
                 "and", "or", "but", "not", "no", "that", "this", "it", "its"}
    infon_tokens -= stopwords
    gold_tokens -= stopwords

    if not infon_tokens or not gold_tokens:
        return 0.0, 0.0, 0.0

    overlap = infon_tokens & gold_tokens
    precision = len(overlap) / len(infon_tokens)
    recall = len(overlap) / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _compute_token_overlap_texts(retrieved: list[str], gold: list[str]
                                  ) -> tuple[float, float, float]:
    """Token overlap between two lists of text strings."""
    if not retrieved or not gold:
        return 0.0, 0.0, 0.0

    stopwords = {"the", "a", "an", "is", "was", "are", "were", "be", "been",
                 "being", "have", "has", "had", "do", "does", "did", "will",
                 "would", "could", "should", "may", "might", "shall", "can",
                 "of", "in", "to", "for", "with", "on", "at", "from", "by",
                 "and", "or", "but", "not", "no", "that", "this", "it", "its"}

    ret_tokens = set()
    for t in retrieved:
        ret_tokens.update(_tokenize(t))
    ret_tokens -= stopwords

    gold_tokens = set()
    for t in gold:
        gold_tokens.update(_tokenize(t))
    gold_tokens -= stopwords

    if not ret_tokens or not gold_tokens:
        return 0.0, 0.0, 0.0

    overlap = ret_tokens & gold_tokens
    precision = len(overlap) / len(ret_tokens)
    recall = len(overlap) / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _count_paths(edges: list) -> int:
    """Count distinct NEXT chains (connected components of NEXT edges)."""
    if not edges:
        return 0
    # Simple: count source nodes that aren't targets of any other NEXT edge
    targets = {e.target for e in edges}
    starts = {e.source for e in edges if e.source not in targets}
    return max(len(starts), 1)


# ═══════════════════════════════════════════════════════════════════════
# SCHEMA DISCOVERY
# ═══════════════════════════════════════════════════════════════════════

def discover_schema_from_evidence(claims: list[dict], n_anchors: int = 50) -> Path:
    """Run Kan extension on the evidence corpus."""
    from infon.encoder import SpladeEncoder

    print("\n  [discover] Running schema discovery (Kan extension)...")

    # Collect evidence sentences
    sentences = []
    for claim in claims:
        ev_text = claim.get("evidence_text", "")
        if ev_text:
            from infon.extract import split_sentences
            sentences.extend(split_sentences(ev_text))
        for ev in claim.get("evidence", []):
            if len(ev) >= 3:
                sentences.append(ev[2])

    sentences = [s for s in sentences if s.strip()][:5000]
    print(f"    Corpus: {len(sentences)} sentences")

    t0 = time.time()
    discoverer = SchemaDiscovery()
    schema, discovered = discoverer.discover(
        sentences, n_anchors=n_anchors,
        min_doc_freq=3, activation_threshold=0.2,
    )
    elapsed = time.time() - t0

    out_path = Path(__file__).parent / "schemas" / "wikipedia_discovered.json"
    schema.save(out_path)

    type_counts = defaultdict(int)
    for da in discovered:
        type_counts[da.inferred_type] += 1

    print(f"    Discovered {len(discovered)} anchors in {elapsed:.1f}s")
    print(f"    Types: {dict(type_counts)}")
    print(f"    Top: {[da.name for da in discovered[:8]]}")
    print(f"    Saved: {out_path}")

    return out_path


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fever", action="store_true")
    parser.add_argument("--hover", action="store_true")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--conditions", default="rag,fixed,discovered,heads")
    parser.add_argument("--discover-anchors", type=int, default=50)
    args = parser.parse_args()

    if not args.fever and not args.hover:
        args.fever = True
        args.hover = True

    conditions = set(args.conditions.split(","))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # ── FEVER ──────────────────────────────────────────────────────────
    if args.fever:
        print("=" * 70)
        print(f"FEVER BENCHMARK (limit={args.limit})")
        print("=" * 70)

        print("  Loading FEVER claims from HuggingFace...")
        fever_claims = load_fever_hf(limit=args.limit)
        print(f"  Loaded {len(fever_claims)} claims")

        fever_results = {}

        if "rag" in conditions:
            fever_results["rag"] = evaluate_fever_rag(fever_claims)

        if "fixed" in conditions:
            fever_results["fixed"] = evaluate_fever_infon(
                fever_claims, SCHEMA_PATH, "fixed"
            )

        if "discovered" in conditions:
            disc_path = discover_schema_from_evidence(
                fever_claims, n_anchors=args.discover_anchors
            )
            fever_results["discovered"] = evaluate_fever_infon(
                fever_claims, disc_path, "discovered"
            )

        if "heads" in conditions:
            fever_results["heads"] = evaluate_fever_heads(
                fever_claims, SCHEMA_PATH
            )

        all_results["fever"] = fever_results

    # ── HoVer ──────────────────────────────────────────────────────────
    if args.hover:
        print("\n" + "=" * 70)
        print(f"HoVer BENCHMARK (limit={args.limit})")
        print("=" * 70)

        print("  Loading HoVer claims from HuggingFace...")
        hover_claims = load_hover_hf(limit=args.limit)
        print(f"  Loaded {len(hover_claims)} claims")

        hover_results = {}

        if "rag" in conditions:
            hover_results["rag"] = evaluate_hover_rag(hover_claims)

        if "fixed" in conditions:
            hover_results["fixed"] = evaluate_hover_infon(
                hover_claims, SCHEMA_PATH, "fixed"
            )

        if "discovered" in conditions:
            disc_path = Path(__file__).parent / "schemas" / "wikipedia_discovered.json"
            if not disc_path.exists():
                disc_path = discover_schema_from_evidence(
                    hover_claims, n_anchors=args.discover_anchors
                )
            hover_results["discovered"] = evaluate_hover_infon(
                hover_claims, disc_path, "discovered"
            )

        if "heads" in conditions:
            hover_results["heads"] = evaluate_hover_heads(
                hover_claims, SCHEMA_PATH
            )

        all_results["hover"] = hover_results

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    _print_table(all_results)

    # Save
    results_file = RESULTS_DIR / "benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {results_file}")


def _print_table(all_results):
    """Print final comparison table."""
    if "fever" in all_results:
        print("\n┌──────────────────────────────────────────────────────────────────┐")
        print("│ FEVER: Infons (Atomic Units) vs Text Snippets                    │")
        print("├──────────────┬────────┬─────────┬────────┬──────┬────────────────┤")
        print("│ Condition    │ Label  │ Ev.Prec │ Ev.Rec │Ev.F1 │ Evidence Unit  │")
        print("├──────────────┼────────┼─────────┼────────┼──────┼────────────────┤")
        for name, m in all_results["fever"].items():
            la = m.get("label_accuracy", 0)
            p = m.get("evidence_precision", 0)
            r = m.get("evidence_recall", 0)
            f1 = m.get("evidence_f1", 0)
            unit = "Text snippet" if name == "rag" else "Infon (atomic)"
            print(f"│ {name:<12} │ {la:.3f}  │  {p:.3f}  │ {r:.3f}  │{f1:.3f}│ {unit:<14} │")
        print("└──────────────┴────────┴─────────┴────────┴──────┴────────────────┘")

    if "hover" in all_results:
        print("\n┌──────────────────────────────────────────────────────────────────┐")
        print("│ HoVer: Hyperedges (Relational Paths) vs Linked Documents         │")
        print("├──────────────┬────────┬──────┬──────┬──────┬──────┬──────────────┤")
        print("│ Condition    │ Label  │ Prec │ Rec  │  F1  │ Cov. │ Reason. Unit │")
        print("├──────────────┼────────┼──────┼──────┼──────┼──────┼──────────────┤")
        for name, m in all_results["hover"].items():
            la = m.get("label_accuracy", 0)
            p = m.get("fact_precision", 0)
            r = m.get("fact_recall", 0)
            f1 = m.get("fact_f1", 0)
            cov = m.get("path_coverage", 0)
            unit = "Document" if name == "rag" else "Hyperedge"
            print(f"│ {name:<12} │ {la:.3f}  │{p:.3f}│{r:.3f}│{f1:.3f}│{cov:.3f}│ {unit:<12} │")
        print("└──────────────┴────────┴──────┴──────┴──────┴──────┴──────────────┘")


if __name__ == "__main__":
    main()

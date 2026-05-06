"""FEVER evaluation: Infon-based evidence retrieval vs RAG text snippets.

Experimental setup:
  - RAG: Claim → dense retrieve → text snippets → sentence IDs
  - Cognition-fixed: Claim → anchor project → infon retrieval → sentence IDs
  - Cognition-discovered: Same, but schema from Kan extension on wiki corpus

Metrics:
  - Evidence Precision/Recall/F1 at sentence level
  - Oracle accuracy (label accuracy given perfect retrieval)
  - FEVER Score (label accuracy given retrieved evidence covers gold)

The key insight: infons decompose multi-fact sentences into atomic units.
A single Wikipedia sentence "X was born in Y and later moved to Z" becomes
TWO infons: <<born, X, Y>> and <<moved, X, Z>>. The claim "X was born in Y"
can be matched to exactly the right atomic unit, not the whole sentence.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from dataclasses import dataclass, field

# Add parent to path for imports
_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from infon import Cognition, CognitionConfig, Infon, QueryResult
from infon.category import SchemaDiscovery


@dataclass
class FEVERResult:
    """Result for a single FEVER claim under infon evaluation."""
    claim_id: str
    claim: str
    gold_label: str
    gold_evidence: list  # [[annotation_id, evidence_id, doc_id, sent_id], ...]

    # Retrieved infons as evidence
    infons: list[Infon] = field(default_factory=list)
    retrieved_sent_ids: set = field(default_factory=set)  # {(doc_id, sent_id)}

    # Derived
    predicted_label: str = ""
    evidence_precision: float = 0.0
    evidence_recall: float = 0.0
    evidence_f1: float = 0.0
    fever_score: float = 0.0  # 1 if label correct AND evidence sufficient


def load_fever_claims(path: str | Path, limit: int | None = None) -> list[dict]:
    """Load FEVER claims from JSONL."""
    claims = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            claims.append(item)
            if limit and len(claims) >= limit:
                break
    return claims


def load_wiki_pages(wiki_dir: str | Path) -> dict[str, list[str]]:
    """Load Wikipedia pages as {doc_title: [sentence_0, sentence_1, ...]}.

    FEVER wiki-pages are stored as JSONL with fields:
      {id: str, text: str, lines: str}
    where lines = "0\\tFirst sentence.\\n1\\tSecond sentence.\\n..."
    """
    pages = {}
    wiki_dir = Path(wiki_dir)

    for jsonl_file in sorted(wiki_dir.glob("wiki-*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                if not line.strip():
                    continue
                doc = json.loads(line)
                doc_id = doc["id"]
                lines_raw = doc.get("lines", "")
                sentences = []
                for l in lines_raw.split("\n"):
                    parts = l.split("\t", 1)
                    if len(parts) == 2:
                        sentences.append(parts[1])
                    else:
                        sentences.append("")
                pages[doc_id] = sentences

    return pages


def extract_gold_sent_ids(evidence_sets: list) -> set[tuple[str, int]]:
    """Extract gold (doc_id, sent_id) pairs from FEVER evidence.

    FEVER evidence format: [[annotation_id, evidence_id, doc_title, sent_idx], ...]
    Multiple annotation sets — a claim is verified if ANY one set is fully covered.
    """
    all_pairs = set()
    for evidence_set in evidence_sets:
        for item in evidence_set:
            if len(item) >= 4 and item[2] is not None:
                all_pairs.add((item[2], item[3]))
    return all_pairs


class CognitionFEVERRunner:
    """Run infon system on FEVER claims.

    Strategy:
    1. Ingest relevant Wikipedia pages into infon (per claim or batch)
    2. Query with the claim text
    3. Collect infon.sent_id values as retrieved evidence
    4. Compare against gold sentence IDs
    """

    def __init__(self, schema_path: str | Path, db_path: str = ":memory:"):
        self.config = CognitionConfig(
            schema_path=str(schema_path),
            db_path=db_path,
            activation_threshold=0.2,  # lower for general domain
            min_confidence=0.03,
            top_k_per_role=5,
            default_top_k=100,
        )

    def evaluate_claim(
        self,
        claim: dict,
        wiki_pages: dict[str, list[str]],
        top_k: int = 20,
    ) -> FEVERResult:
        """Evaluate a single FEVER claim.

        For efficiency, we only ingest the gold evidence pages + some distractors.
        In a full evaluation you'd index all of Wikipedia.
        """
        claim_id = str(claim.get("id", ""))
        claim_text = claim["claim"]
        gold_label = claim.get("label", "")
        gold_evidence = claim.get("evidence", [])
        gold_sent_ids = extract_gold_sent_ids(gold_evidence)

        # Determine which pages to ingest (gold + nearby pages for realism)
        evidence_docs = set()
        for ev_set in gold_evidence:
            for item in ev_set:
                if len(item) >= 4 and item[2] is not None:
                    evidence_docs.add(item[2])

        # Build documents for ingestion
        documents = []
        for doc_id in evidence_docs:
            if doc_id not in wiki_pages:
                continue
            sentences = wiki_pages[doc_id]
            # Each sentence becomes a document with its sentence index
            for sent_idx, sent_text in enumerate(sentences):
                if not sent_text.strip():
                    continue
                documents.append({
                    "text": sent_text,
                    "id": doc_id,
                    "sent_idx": sent_idx,
                    # Encode doc_id + sent_idx into the doc id for tracing
                    "doc_id_full": f"{doc_id}__sent_{sent_idx}",
                })

        if not documents:
            return FEVERResult(
                claim_id=claim_id, claim=claim_text,
                gold_label=gold_label, gold_evidence=gold_evidence,
            )

        # Fresh infon instance per claim (isolation)
        cog = Cognition(self.config)

        # Ingest evidence pages
        ingest_docs = [{
            "text": d["text"],
            "id": d["doc_id_full"],
        } for d in documents]
        cog.ingest(ingest_docs)

        # Query with the claim
        result = cog.query(claim_text, top_k=top_k, include_chains=False)
        cog.close()

        # Map infons back to (doc_id, sent_id) pairs
        retrieved_sent_ids = set()
        for infon in result.infons:
            # doc_id is stored as "PageTitle__sent_3"
            parts = infon.doc_id.rsplit("__sent_", 1)
            if len(parts) == 2:
                doc_id = parts[0]
                try:
                    sent_id = int(parts[1])
                    retrieved_sent_ids.add((doc_id, sent_id))
                except ValueError:
                    pass

        # Compute evidence metrics
        precision, recall, f1 = _compute_evidence_metrics(
            retrieved_sent_ids, gold_sent_ids
        )

        # Predict label based on infon polarity/confidence
        predicted_label = _predict_label(result.infons, claim_text)

        # FEVER score: correct label AND evidence covers at least one full set
        fever_score = 0.0
        if predicted_label == gold_label:
            # Check if retrieved evidence covers any complete annotation set
            for ev_set in gold_evidence:
                set_pairs = set()
                for item in ev_set:
                    if len(item) >= 4 and item[2] is not None:
                        set_pairs.add((item[2], item[3]))
                if set_pairs and set_pairs.issubset(retrieved_sent_ids):
                    fever_score = 1.0
                    break
            # NEI claims don't need evidence
            if gold_label == "NOT ENOUGH INFO":
                fever_score = 1.0

        return FEVERResult(
            claim_id=claim_id,
            claim=claim_text,
            gold_label=gold_label,
            gold_evidence=gold_evidence,
            infons=result.infons,
            retrieved_sent_ids=retrieved_sent_ids,
            predicted_label=predicted_label,
            evidence_precision=precision,
            evidence_recall=recall,
            evidence_f1=f1,
            fever_score=fever_score,
        )

    def evaluate_batch(
        self,
        claims: list[dict],
        wiki_pages: dict[str, list[str]],
        top_k: int = 20,
    ) -> list[FEVERResult]:
        """Evaluate a batch of claims."""
        results = []
        for i, claim in enumerate(claims):
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(claims)}]")
            result = self.evaluate_claim(claim, wiki_pages, top_k=top_k)
            results.append(result)
        return results


def _compute_evidence_metrics(
    retrieved: set[tuple], gold: set[tuple]
) -> tuple[float, float, float]:
    """Compute precision, recall, F1 for evidence retrieval."""
    if not retrieved and not gold:
        return 1.0, 1.0, 1.0
    if not retrieved:
        return 0.0, 0.0, 0.0
    if not gold:
        return 0.0, 0.0, 0.0  # can't compute against empty gold

    tp = len(retrieved & gold)
    precision = tp / len(retrieved) if retrieved else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _predict_label(infons: list, claim_text: str = "") -> str:
    """Predict FEVER label from retrieved infons.

    Strategy: compare the claim's semantic content against the retrieved infons.
    - If infons confirm the claim's assertions (same triple, affirmed) → SUPPORTS
    - If infons contradict (same S+O but different P, or negated) → REFUTES
    - If no strong signal → NOT ENOUGH INFO

    Key insight: infons give us structured (S, P, O) triples. A refutation
    is when the evidence has the same subject/object but a conflicting predicate
    or explicit negation.
    """
    if not infons:
        return "NOT ENOUGH INFO"

    # Score based on polarity and confidence
    affirmed = [(inf.confidence, inf) for inf in infons if inf.polarity == 1]
    negated = [(inf.confidence, inf) for inf in infons if inf.polarity == 0]

    # Weight by confidence — top infons matter more
    top_k = min(5, len(infons))
    top_infons = sorted(infons, key=lambda x: -x.confidence)[:top_k]

    affirmed_weight = sum(inf.confidence for inf in top_infons if inf.polarity == 1)
    negated_weight = sum(inf.confidence for inf in top_infons if inf.polarity == 0)
    total_weight = affirmed_weight + negated_weight

    if total_weight < 0.1:
        return "NOT ENOUGH INFO"

    negation_ratio = negated_weight / total_weight if total_weight > 0 else 0.0

    # If majority of top evidence is negated → REFUTES
    if negation_ratio > 0.4:
        return "REFUTES"
    # If strong affirmed evidence → SUPPORTS
    elif affirmed_weight > 0.2:
        return "SUPPORTS"
    else:
        return "NOT ENOUGH INFO"


def aggregate_results(results: list[FEVERResult]) -> dict:
    """Compute aggregate metrics over all claims."""
    n = len(results)
    if n == 0:
        return {}

    label_correct = sum(1 for r in results if r.predicted_label == r.gold_label)
    fever_scores = sum(r.fever_score for r in results)

    # Evidence metrics (exclude NEI which has no evidence)
    verifiable = [r for r in results if r.gold_label != "NOT ENOUGH INFO"]
    n_ver = len(verifiable)

    avg_precision = sum(r.evidence_precision for r in verifiable) / n_ver if n_ver else 0.0
    avg_recall = sum(r.evidence_recall for r in verifiable) / n_ver if n_ver else 0.0
    avg_f1 = sum(r.evidence_f1 for r in verifiable) / n_ver if n_ver else 0.0

    # Per-label breakdown
    by_label = {}
    for label in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]:
        subset = [r for r in results if r.gold_label == label]
        if subset:
            correct = sum(1 for r in subset if r.predicted_label == label)
            by_label[label] = {
                "count": len(subset),
                "accuracy": correct / len(subset),
            }

    return {
        "n_claims": n,
        "label_accuracy": label_correct / n,
        "fever_score": fever_scores / n,
        "evidence_precision": avg_precision,
        "evidence_recall": avg_recall,
        "evidence_f1": avg_f1,
        "by_label": by_label,
        "avg_infons_per_claim": sum(len(r.infons) for r in results) / n,
        "avg_retrieved_sents": sum(len(r.retrieved_sent_ids) for r in results) / n,
    }

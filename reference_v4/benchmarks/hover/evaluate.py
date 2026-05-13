"""HoVer evaluation: Hyperedge traversal vs RAG document linking.

Experimental setup:
  - RAG: Claim → dense retrieve → linked documents → reasoning path (flat)
  - Cognition-fixed: Claim → anchor → NEXT chain traversal → relational paths
  - Cognition-discovered: Same, schema from Kan extension

HoVer requires MULTI-HOP reasoning: verifying a claim like "The director of
Film X was born in Country Y which has population Z" requires linking 3 facts
across 3 Wikipedia pages.

Key insight: cognition's NEXT edges (hyperedges linking infons through shared
anchors) naturally form the multi-hop reasoning chains that HoVer requires.
When we ingest the Wikipedia pages, anchor-based temporal sequencing creates
paths like:

  <<directed, PersonA, FilmX>> --NEXT(PersonA)--> <<born, PersonA, CountryY>>
  <<located, CountryY, Region>> --NEXT(CountryY)--> <<population, CountryY, Z>>

The RAG baseline must rediscover these paths via iterative retrieval or
explicit document linking — cognition has them as first-class graph structure.

Metrics:
  - Supporting Fact Precision/Recall/F1 (sentence level)
  - Path Coverage: fraction of gold multi-hop chains recovered
  - Hop Accuracy: per-hop retrieval accuracy (1-hop, 2-hop, 3-hop, 4-hop)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "cognition" / "src"))

from cognition import Cognition, CognitionConfig, Infon, Edge, QueryResult
from cognition.category import SchemaDiscovery


@dataclass
class HoVerResult:
    """Result for a single HoVer claim under cognition evaluation."""
    claim_id: str
    claim: str
    gold_label: str
    gold_supporting_facts: list  # [[doc_id, sent_id], ...]
    num_hops: int  # number of documents in the gold reasoning chain

    # Retrieved evidence via hyperedge traversal
    infons: list[Infon] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)  # NEXT edges traversed
    retrieved_facts: set = field(default_factory=set)  # {(doc_id, sent_id)}

    # Reasoning path analysis
    paths_found: list[list[str]] = field(default_factory=list)  # anchor chains
    path_coverage: float = 0.0  # fraction of gold chain covered

    # Metrics
    predicted_label: str = ""
    fact_precision: float = 0.0
    fact_recall: float = 0.0
    fact_f1: float = 0.0
    per_hop_recall: list[float] = field(default_factory=list)


def load_hover_claims(path: str | Path, limit: int | None = None) -> list[dict]:
    """Load HoVer claims from JSON."""
    with open(path) as f:
        data = json.load(f)
    if limit:
        data = data[:limit]
    return data


def extract_gold_facts(supporting_facts: list) -> set[tuple[str, int]]:
    """Extract gold (doc_id, sent_id) pairs from HoVer supporting facts."""
    return {(sf[0], sf[1]) for sf in supporting_facts if len(sf) >= 2}


def group_facts_by_doc(supporting_facts: list) -> dict[str, list[int]]:
    """Group supporting facts by document for hop analysis."""
    by_doc = defaultdict(list)
    for sf in supporting_facts:
        if len(sf) >= 2:
            by_doc[sf[0]].append(sf[1])
    return dict(by_doc)


class CognitionHoVerRunner:
    """Run cognition system on HoVer claims with hyperedge traversal.

    Strategy:
    1. Ingest all relevant Wikipedia pages (gold + distractors)
    2. Run consolidation to build NEXT edges across shared anchors
    3. Query with claim — retrieve infons AND walk NEXT chains
    4. The NEXT chains are the multi-hop reasoning paths
    5. Map traversed infons back to (doc_id, sent_id) pairs
    """

    def __init__(self, schema_path: str | Path, db_path: str = ":memory:"):
        self.config = CognitionConfig(
            schema_path=str(schema_path),
            db_path=db_path,
            activation_threshold=0.2,
            min_confidence=0.03,
            top_k_per_role=5,
            default_top_k=200,  # larger for multi-hop
            consolidation_interval=1,  # consolidate after every ingest for NEXT edges
        )

    def evaluate_claim(
        self,
        claim: dict,
        wiki_pages: dict[str, list[str]],
        chain_depth: int = 15,
    ) -> HoVerResult:
        """Evaluate a single HoVer claim via hyperedge traversal."""
        claim_id = str(claim.get("uid", ""))
        claim_text = claim["claim"]
        gold_label = claim.get("label", "")
        gold_facts = claim.get("supporting_facts", [])
        gold_fact_set = extract_gold_facts(gold_facts)
        facts_by_doc = group_facts_by_doc(gold_facts)
        num_hops = len(facts_by_doc)  # distinct documents = distinct hops

        # Determine which pages to ingest
        evidence_docs = set(sf[0] for sf in gold_facts if len(sf) >= 2)

        # Build documents for ingestion — each sentence separately
        documents = []
        for doc_id in evidence_docs:
            if doc_id not in wiki_pages:
                continue
            sentences = wiki_pages[doc_id]
            for sent_idx, sent_text in enumerate(sentences):
                if not sent_text.strip():
                    continue
                documents.append({
                    "text": sent_text,
                    "id": f"{doc_id}__sent_{sent_idx}",
                    # Use a synthetic timestamp to enable NEXT edge ordering
                    # Documents processed in order get sequential timestamps
                    "timestamp": f"2025-01-{len(documents)+1:02d}",
                })

        if not documents:
            return HoVerResult(
                claim_id=claim_id, claim=claim_text,
                gold_label=gold_label,
                gold_supporting_facts=gold_facts,
                num_hops=num_hops,
            )

        # Fresh cognition instance with consolidation
        cog = Cognition(self.config)
        cog.ingest(documents, consolidate_now=True)

        # Query with chain traversal enabled — this walks NEXT edges
        result = cog.query(
            claim_text,
            top_k=100,
            include_chains=True,
        )
        cog.close()

        # Map infons back to (doc_id, sent_id)
        retrieved_facts = set()
        for infon in result.infons:
            parts = infon.doc_id.rsplit("__sent_", 1)
            if len(parts) == 2:
                doc_id = parts[0]
                try:
                    sent_id = int(parts[1])
                    retrieved_facts.add((doc_id, sent_id))
                except ValueError:
                    pass

        # Analyze reasoning paths from NEXT edges
        paths_found = _extract_reasoning_paths(result.edges, result.infons)

        # Path coverage: what fraction of gold documents are linked in our paths?
        docs_in_paths = set()
        for infon in result.infons:
            parts = infon.doc_id.rsplit("__sent_", 1)
            if len(parts) == 2:
                docs_in_paths.add(parts[0])
        path_coverage = (
            len(docs_in_paths & evidence_docs) / len(evidence_docs)
            if evidence_docs else 0.0
        )

        # Per-hop recall: for each gold document, did we retrieve any of its facts?
        per_hop_recall = []
        for doc_id in sorted(facts_by_doc.keys()):
            doc_gold = {(doc_id, sid) for sid in facts_by_doc[doc_id]}
            doc_retrieved = retrieved_facts & doc_gold
            hop_recall = len(doc_retrieved) / len(doc_gold) if doc_gold else 0.0
            per_hop_recall.append(hop_recall)

        # Evidence metrics
        precision, recall, f1 = _compute_fact_metrics(retrieved_facts, gold_fact_set)

        # Label prediction from traversed infons
        predicted_label = _predict_hover_label(result.infons)

        return HoVerResult(
            claim_id=claim_id,
            claim=claim_text,
            gold_label=gold_label,
            gold_supporting_facts=gold_facts,
            num_hops=num_hops,
            infons=result.infons,
            edges=result.edges,
            retrieved_facts=retrieved_facts,
            paths_found=paths_found,
            path_coverage=path_coverage,
            predicted_label=predicted_label,
            fact_precision=precision,
            fact_recall=recall,
            fact_f1=f1,
            per_hop_recall=per_hop_recall,
        )

    def evaluate_batch(
        self,
        claims: list[dict],
        wiki_pages: dict[str, list[str]],
        chain_depth: int = 15,
    ) -> list[HoVerResult]:
        """Evaluate a batch of HoVer claims."""
        results = []
        for i, claim in enumerate(claims):
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(claims)}]")
            result = self.evaluate_claim(claim, wiki_pages, chain_depth)
            results.append(result)
        return results


def _extract_reasoning_paths(edges: list[Edge], infons: list[Infon]) -> list[list[str]]:
    """Extract reasoning paths from NEXT edges.

    A path = sequence of anchors linked through NEXT edges.
    E.g., [PersonA, FilmX, CountryY] means we traversed:
      infon_about_PersonA --NEXT--> infon_about_FilmX --NEXT--> infon_about_CountryY
    """
    paths = []
    if not edges:
        return paths

    # Build adjacency from NEXT edges
    adj = defaultdict(list)
    for edge in edges:
        if edge.edge_type == "NEXT":
            anchor = edge.metadata.get("anchor", "")
            adj[edge.source].append((edge.target, anchor))

    # Walk from each starting infon
    infon_ids = {inf.infon_id for inf in infons}
    visited_starts = set()

    for infon in infons:
        if infon.infon_id in visited_starts:
            continue
        # Check if this infon starts a NEXT chain
        if infon.infon_id in adj:
            path = [infon.subject]  # start with subject anchor
            current = infon.infon_id
            seen = {current}
            while current in adj:
                nexts = adj[current]
                if not nexts:
                    break
                target, anchor = nexts[0]
                if target in seen:
                    break
                seen.add(target)
                path.append(anchor)
                current = target
            if len(path) > 1:
                paths.append(path)
                visited_starts.update(seen)

    return paths


def _compute_fact_metrics(
    retrieved: set[tuple], gold: set[tuple]
) -> tuple[float, float, float]:
    """Compute precision, recall, F1 for supporting fact retrieval."""
    if not retrieved and not gold:
        return 1.0, 1.0, 1.0
    if not retrieved:
        return 0.0, 0.0, 0.0
    if not gold:
        return 0.0, 0.0, 0.0

    tp = len(retrieved & gold)
    precision = tp / len(retrieved) if retrieved else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _predict_hover_label(infons: list) -> str:
    """Predict HoVer label (SUPPORTED/NOT_SUPPORTED) from infons."""
    if not infons:
        return "NOT_SUPPORTED"

    # HoVer is binary: SUPPORTED if evidence confirms, NOT_SUPPORTED otherwise
    high_conf = [inf for inf in infons if inf.confidence > 0.2]
    affirmed = [inf for inf in high_conf if inf.polarity == 1]

    if len(affirmed) >= 2:  # multi-hop needs multiple confirming infons
        return "SUPPORTED"
    return "NOT_SUPPORTED"


def aggregate_results(results: list[HoVerResult]) -> dict:
    """Compute aggregate metrics over all HoVer claims."""
    n = len(results)
    if n == 0:
        return {}

    label_correct = sum(1 for r in results if r.predicted_label == r.gold_label)

    avg_precision = sum(r.fact_precision for r in results) / n
    avg_recall = sum(r.fact_recall for r in results) / n
    avg_f1 = sum(r.fact_f1 for r in results) / n
    avg_path_coverage = sum(r.path_coverage for r in results) / n

    # Per-hop analysis (group by number of hops)
    by_hops = defaultdict(list)
    for r in results:
        by_hops[r.num_hops].append(r)

    hop_breakdown = {}
    for num_hops, subset in sorted(by_hops.items()):
        hop_recalls = []
        for r in subset:
            for i, hr in enumerate(r.per_hop_recall):
                while len(hop_recalls) <= i:
                    hop_recalls.append([])
                hop_recalls[i].append(hr)

        hop_breakdown[f"{num_hops}_hop"] = {
            "count": len(subset),
            "fact_f1": sum(r.fact_f1 for r in subset) / len(subset),
            "path_coverage": sum(r.path_coverage for r in subset) / len(subset),
            "per_hop_recall": [
                sum(hrs) / len(hrs) if hrs else 0.0
                for hrs in hop_recalls
            ],
        }

    return {
        "n_claims": n,
        "label_accuracy": label_correct / n,
        "fact_precision": avg_precision,
        "fact_recall": avg_recall,
        "fact_f1": avg_f1,
        "path_coverage": avg_path_coverage,
        "by_hops": hop_breakdown,
        "avg_infons_per_claim": sum(len(r.infons) for r in results) / n,
        "avg_edges_per_claim": sum(len(r.edges) for r in results) / n,
        "avg_paths_per_claim": sum(len(r.paths_found) for r in results) / n,
    }

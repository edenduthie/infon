"""RAG baseline: dense retrieval over Wikipedia sentences.

Standard approach: encode claims + evidence sentences with a dense encoder,
retrieve by cosine similarity, and classify SUPPORTS/REFUTES/NEI.

This gives the comparison point:
  - FEVER: text snippets as evidence units, matched by sentence IDs
  - HoVer: linked documents as reasoning units, matched by document chains
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class RAGResult:
    """A single RAG retrieval result."""
    claim_id: str
    claim: str
    retrieved_sentences: list[dict] = field(default_factory=list)
    # Each: {doc_id, sent_id, text, score}
    predicted_label: str = ""
    gold_label: str = ""
    gold_evidence: list = field(default_factory=list)


class DenseEncoder:
    """Thin wrapper around sentence-transformers for dense encoding."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return self.model.encode(texts, batch_size=batch_size,
                                 show_progress_bar=False, normalize_embeddings=True)


class RAGIndex:
    """In-memory FAISS index over Wikipedia sentences."""

    def __init__(self, encoder: DenseEncoder):
        self.encoder = encoder
        self.sentences: list[dict] = []  # {doc_id, sent_id, text}
        self._index = None

    def build(self, wiki_sentences: list[dict], batch_size: int = 256):
        """Build index from Wikipedia sentence records.

        Each record: {doc_id: str, sent_id: int, text: str}
        """
        import faiss

        self.sentences = wiki_sentences
        texts = [s["text"] for s in wiki_sentences]

        # Encode in batches
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embs = self.encoder.encode(batch, batch_size=batch_size)
            all_embeddings.append(embs)

        embeddings = np.vstack(all_embeddings).astype(np.float32)

        # Build FAISS index (inner product on normalized vectors = cosine)
        self._index = faiss.IndexFlatIP(self.encoder.dim)
        self._index.add(embeddings)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve top-k sentences for a claim."""
        q_emb = self.encoder.encode([query]).astype(np.float32)
        scores, indices = self._index.search(q_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            sent = self.sentences[idx].copy()
            sent["score"] = float(score)
            results.append(sent)
        return results


class RAGBaseline:
    """Full RAG baseline for FEVER/HoVer evaluation.

    Approach:
    1. Index Wikipedia evidence sentences with dense embeddings
    2. For each claim, retrieve top-k sentences by cosine similarity
    3. Evidence = retrieved sentence IDs
    4. Label = majority vote on NLI (or threshold-based)
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.encoder = DenseEncoder(model_name)
        self.index = RAGIndex(self.encoder)

    def build_index(self, wiki_sentences: list[dict], batch_size: int = 256):
        """Build the retrieval index."""
        self.index.build(wiki_sentences, batch_size=batch_size)

    def evaluate_fever(self, claims: list[dict], top_k: int = 5) -> list[RAGResult]:
        """Run FEVER evaluation.

        Each claim: {id, claim, label, evidence: [[_, _, doc_id, sent_id], ...]}
        """
        results = []
        for item in claims:
            claim_id = str(item.get("id", ""))
            claim_text = item["claim"]
            gold_label = item.get("label", "")
            gold_evidence = item.get("evidence", [])

            retrieved = self.index.retrieve(claim_text, top_k=top_k)

            result = RAGResult(
                claim_id=claim_id,
                claim=claim_text,
                retrieved_sentences=retrieved,
                gold_label=gold_label,
                gold_evidence=gold_evidence,
            )
            results.append(result)

        return results

    def evaluate_hover(self, claims: list[dict], top_k: int = 10) -> list[RAGResult]:
        """Run HoVer evaluation.

        Each claim: {uid, claim, label, supporting_facts: [[doc_id, sent_id], ...]}
        """
        results = []
        for item in claims:
            claim_id = str(item.get("uid", ""))
            claim_text = item["claim"]
            gold_label = item.get("label", "")
            gold_evidence = item.get("supporting_facts", [])

            retrieved = self.index.retrieve(claim_text, top_k=top_k)

            result = RAGResult(
                claim_id=claim_id,
                claim=claim_text,
                retrieved_sentences=retrieved,
                gold_label=gold_label,
                gold_evidence=gold_evidence,
            )
            results.append(result)

        return results

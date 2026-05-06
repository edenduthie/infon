"""CognitionSystem: fact-checking via the InfonStore pipeline.

Ingests evidence documents into a temporary InfonStore, builds a dynamic
schema from the claim text, then uses store.ask() to retrieve a Verdict.

IMPORTANT: This class never calls any LLM. No Analyst, no boto3. Purely
the InfonStore/SPLADE extraction + DS reasoning pipeline.

Variants:
  "symbolic"  — standard InfonStore with SPLADE + Dempster-Shafer reasoning
  "gnn"       — same as symbolic for now (GNN scoring is applied inside
                the store pipeline automatically when weights are available)
"""

from __future__ import annotations

import re
import tempfile

from cognition.cassette.store import InfonStore
from cognition.cassette.dsl import Query

from reference_v4.benchmarks.types import EvalClaim, MassFunction

# Minimum word length for schema anchor creation
_MIN_ANCHOR_LEN = 4

# Common English stop words to exclude from schema anchors
_STOP_WORDS = frozenset({
    "the", "this", "that", "with", "from", "have", "been", "will", "were",
    "they", "them", "their", "also", "more", "some", "such", "when",
    "into", "than", "then", "does", "what", "which", "about", "would",
    "could", "should", "other", "these", "those", "after", "before",
    "being", "each", "both", "only", "over", "very", "even", "most",
    "there", "here", "just", "while", "during", "between", "through",
    "because", "without", "however", "therefore", "although",
})

# Typical English relation verbs — these get 'relation' type in the schema
_RELATION_HINTS = frozenset({
    "reduces", "reduce", "increases", "increase", "causes", "cause",
    "prevents", "prevent", "shows", "show", "suggests", "suggest",
    "supports", "support", "refutes", "refute", "indicates", "indicate",
    "demonstrates", "demonstrate", "proves", "prove", "disproves", "disprove",
    "treats", "treat", "improves", "improve", "decreases", "decrease",
    "inhibits", "inhibit", "promotes", "promote", "affects", "affect",
    "produces", "produce", "contains", "contain", "requires", "require",
    "links", "link", "associates", "associate", "correlates", "correlate",
    "leads", "lead", "results", "result", "enables", "enable",
})


def _build_schema_from_claim(claim_text: str) -> dict:
    """Build a minimal schema dict from the words in the claim text.

    Each content word becomes an anchor. Relation-hint verbs get type
    'relation'; everything else gets type 'actor'. Short words and stop
    words are excluded.

    This is an approximation — the InfonStore's SPLADE encoder will find
    semantically similar words in the evidence docs even if the exact
    token doesn't appear.
    """
    tokens = re.findall(r"[a-z]+", claim_text.lower())
    schema: dict = {}
    for token in tokens:
        if len(token) < _MIN_ANCHOR_LEN:
            continue
        if token in _STOP_WORDS:
            continue
        if token in schema:
            continue
        anchor_type = "relation" if token in _RELATION_HINTS else "actor"
        schema[token] = {"type": anchor_type, "tokens": [token]}
    return schema


def _extract_claim_subject(claim_text: str) -> str | None:
    """Heuristically extract the first content word as the query subject.

    The InfonStore reasoner uses the query subject to retrieve candidate
    evidence. We pick the first anchor-eligible word from the claim as a
    best guess.
    """
    tokens = re.findall(r"[a-z]+", claim_text.lower())
    for token in tokens:
        if len(token) >= _MIN_ANCHOR_LEN and token not in _STOP_WORDS:
            return token
    return None


def _verdict_to_mass(verdict) -> MassFunction:
    """Convert a cognition Verdict into the benchmark MassFunction.

    The cognition Verdict.mass has fields: supports, refutes, uncertain, theta.
    We map these to the benchmark MassFunction: m_s, m_r, m_u, m_theta.
    """
    m = verdict.mass
    return MassFunction(
        m_s=m.supports,
        m_r=m.refutes,
        m_u=m.uncertain,
        m_theta=m.theta,
    )


class CognitionSystem:
    """InfonStore-based fact-checking system.

    Creates a temporary store per evaluate() call, ingests evidence docs,
    and queries with the claim to produce a Dempster-Shafer MassFunction.

    Args:
        variant: "symbolic" (default) or "gnn". Both use the same
                 InfonStore pipeline; the GNN scorer fires automatically
                 when model weights are loaded.
    """

    def __init__(self, variant: str = "symbolic") -> None:
        self.variant = variant
        self.name = f"cognition_{variant}"

    def evaluate(self, claim: EvalClaim) -> MassFunction:
        """Evaluate the claim against its evidence docs using InfonStore.

        Returns a MassFunction. Never calls any LLM.
        """
        if not claim.evidence_docs:
            return MassFunction(m_s=0.0, m_r=0.0, m_u=0.0, m_theta=1.0)

        schema = _build_schema_from_claim(claim.claim_text)
        if not schema:
            return MassFunction(m_s=0.0, m_r=0.0, m_u=0.0, m_theta=1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            store = InfonStore(tmpdir)
            store.set_schema_from_dict(schema)

            docs = [
                {"text": doc, "source": f"doc_{i}"}
                for i, doc in enumerate(claim.evidence_docs)
            ]
            result = store.ingest(docs)

            if result["n_infons"] == 0:
                return MassFunction(m_s=0.0, m_r=0.0, m_u=0.0, m_theta=1.0)

            verdict = self._ask_claim(store, claim)
            return _verdict_to_mass(verdict)

    def _ask_claim(self, store: InfonStore, claim: EvalClaim):
        """Build a Query from the claim text and ask the store.

        Strategy: extract the first content word as subject. If no useful
        anchors are found, fall back to a broad subject-less query.
        """
        subject = _extract_claim_subject(claim.claim_text)
        if subject:
            query = Query().where(subject=subject)
        else:
            query = Query()
        return store.ask(query)

"""SymbolicFloor: a simple keyword-overlap baseline for fact-checking.

This is the floor baseline — it uses only surface-level text features:
  1. Keyword overlap between claim and evidence docs (proxy for relevance)
  2. Presence of negation words in the evidence (proxy for refutation)

No neural models are used. This establishes a trivially achievable lower
bound that any serious system should surpass.

Algorithm:
  - Compute word overlap between claim tokens and each evidence doc.
  - If max overlap < threshold → abstain (m_theta = 1.0, full ignorance).
  - If negation words detected in the most-relevant evidence doc:
      → m_r = 0.7, m_theta = 0.3 (lean REFUTES, uncertain)
  - Otherwise:
      → m_s = 0.7, m_theta = 0.3 (lean SUPPORTS, uncertain)
"""

from __future__ import annotations

import re

from benchmarks.types import EvalClaim, MassFunction

# Words whose presence in evidence suggests negation/refutation
_NEGATION_WORDS = frozenset({
    "not", "no", "never", "doesn't", "don't", "didn't", "cannot",
    "can't", "won't", "neither", "nor", "without", "lack", "lacks",
    "lacking", "absent", "false", "incorrect", "wrong", "contrary",
    "contradict", "contradicts", "contradicted", "refute", "refutes",
    "disprove", "disproves", "denied", "denies", "deny",
})

# Mass assigned to the leading focal element (supports or refutes)
_DECISION_MASS = 0.7
_THETA_RESIDUAL = 1.0 - _DECISION_MASS


def _tokenize(text: str) -> set[str]:
    """Lower-case alpha tokens only — simple, fast, reproducible."""
    return set(re.findall(r"[a-z]+", text.lower()))


def _overlap_score(claim_tokens: set[str], doc: str) -> float:
    """Jaccard-like overlap: |claim ∩ doc| / |claim| (precision of claim coverage).

    Returns 0.0 if claim_tokens is empty.
    """
    if not claim_tokens:
        return 0.0
    doc_tokens = _tokenize(doc)
    common = claim_tokens & doc_tokens
    return len(common) / len(claim_tokens)


def _has_negation(doc: str) -> bool:
    """Return True if the doc contains at least one negation cue word."""
    doc_tokens = _tokenize(doc)
    return bool(doc_tokens & _NEGATION_WORDS)


class SymbolicFloor:
    """Keyword-overlap baseline — the simplest possible fact-checking system.

    Args:
        threshold: minimum overlap score to consider evidence relevant.
                   Below this, the system abstains (returns full ignorance).
    """

    name = "symbolic_floor"

    def __init__(self, threshold: float = 0.3) -> None:
        self.threshold = threshold

    def evaluate(self, claim: EvalClaim) -> MassFunction:
        """Return a MassFunction based on keyword overlap with evidence docs.

        Decision logic:
          - No evidence → full ignorance (m_theta = 1.0)
          - Max overlap < threshold → abstain (m_theta = 1.0)
          - Negation found in best-matching evidence → lean REFUTES
          - Otherwise → lean SUPPORTS
        """
        if not claim.evidence_docs:
            return MassFunction(m_s=0.0, m_r=0.0, m_u=0.0, m_theta=1.0)

        claim_tokens = _tokenize(claim.claim_text)

        # Score each evidence doc; find the most relevant one
        scores = [_overlap_score(claim_tokens, doc) for doc in claim.evidence_docs]
        max_score = max(scores)

        if max_score < self.threshold:
            # Evidence is not relevant enough to commit to a verdict
            return MassFunction(m_s=0.0, m_r=0.0, m_u=0.0, m_theta=1.0)

        # Use the highest-scoring doc for the negation check
        best_doc = claim.evidence_docs[scores.index(max_score)]

        if _has_negation(best_doc):
            return MassFunction(
                m_s=0.0,
                m_r=_DECISION_MASS,
                m_u=0.0,
                m_theta=_THETA_RESIDUAL,
            )
        return MassFunction(
            m_s=_DECISION_MASS,
            m_r=0.0,
            m_u=0.0,
            m_theta=_THETA_RESIDUAL,
        )

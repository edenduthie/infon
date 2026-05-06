"""CognitionSystem: fact-checking via the InfonStore pipeline.

Ingests evidence documents into a temporary InfonStore, builds a dynamic
schema from the evidence corpus via SPLADE vocabulary extraction, then
uses store.ask() to retrieve a Verdict.

IMPORTANT: This class never calls any LLM. No Analyst, no boto3. Purely
the InfonStore/SPLADE extraction + DS reasoning pipeline.

Schema strategy: run the SpladeEncoder directly on the evidence documents
(and the claim text) to find the BERT vocabulary tokens that actually
activate in the corpus. These tokens become the schema anchors, classified
as 'actor' or 'relation' via pattern matching. This is the algorithmic
two-pass approach described in the design doc — no spectral clustering,
but it achieves the same goal: a schema grounded in the evidence
vocabulary rather than the raw claim words.

Why not claim-word schema: SPLADE activates sub-word pieces (e.g. "macro"
for "Macron", "critic" for "criticism"). A schema built from raw claim
words like "macron" and "criticism" maps to the correct BERT token IDs
via the AnchorProjector, but critically: the claim words include NO
relation-type anchors (verbs like "cancelled", "deported" are NOT in the
original _RELATION_HINTS set). Without at least one relation anchor per
sentence, extract_infons() cannot form any (subject, predicate, object)
triple, so n_infons=0 every time.

Variants:
  "symbolic"  — standard InfonStore with SPLADE + Dempster-Shafer
  "gnn"       — same; GNN scoring fires automatically when weights loaded
"""

from __future__ import annotations

import re
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

from cognition.cassette.store import InfonStore
from cognition.cassette.dsl import Query
from cognition.encoder import SpladeEncoder

from reference_v4.benchmarks.types import EvalClaim, MassFunction

# Minimum clean-token length to include in schema
_MIN_ANCHOR_LEN = 3

# Common English stop-words to exclude
_STOP_WORDS = frozenset({
    "the", "this", "that", "with", "from", "have", "been", "will", "were",
    "they", "them", "their", "also", "more", "some", "such", "when",
    "into", "than", "then", "does", "what", "which", "about", "would",
    "could", "should", "other", "these", "those", "after", "before",
    "being", "each", "both", "only", "over", "very", "even", "most",
    "there", "here", "just", "while", "during", "between", "through",
    "because", "without", "however", "therefore", "although", "answer",
    "found", "question", "claim", "claims",
})

# Domain-specific relation verbs for fact-checking contexts.
# Scientific verbs (from the original design) + common news/political verbs.
_RELATION_TOKENS = frozenset({
    # Scientific
    "reduces", "reduce", "increases", "increase", "causes", "cause",
    "prevents", "prevent", "shows", "show", "suggests", "suggest",
    "supports", "support", "refutes", "refute", "indicates", "indicate",
    "demonstrates", "demonstrate", "proves", "prove", "disproves", "disprove",
    "treats", "treat", "improves", "improve", "decreases", "decrease",
    "inhibits", "inhibit", "promotes", "promote", "affects", "affect",
    "produces", "produce", "contains", "contain", "requires", "require",
    "links", "link", "associates", "associate", "correlates", "correlate",
    "leads", "lead", "results", "result", "enables", "enable",
    # News / political / factual
    "stated", "state", "claimed", "claim", "reported", "report",
    "denied", "deny", "confirmed", "confirm", "announced", "announce",
    "described", "describe", "criticised", "criticize", "criticized",
    "critic", "attacked", "attack", "condemned", "condemn",
    "cancelled", "cancel", "deported", "deport", "arrested", "arrest",
    "imposed", "impose", "banned", "said", "told", "added",
    "accused", "accuse", "charged", "charge", "voted", "vote",
    "signed", "sign", "passed", "pass", "approved", "approve",
    "rejected", "reject", "blocked", "block", "violated", "violate",
    "issued", "issue", "ordered", "order", "declared", "declare",
    "killed", "killed", "injured", "injure", "died", "die",
    "won", "lost", "defeated", "defeat", "elected", "elect",
    "founded", "found", "created", "create", "launched", "launch",
    "published", "publish", "released", "release", "removed", "remove",
    "resigned", "resign", "replaced", "replace", "fired", "fire",
    "hired", "hire", "appointed", "appoint", "named", "name",
    "received", "receive", "awarded", "award", "earned", "earn",
    "spent", "spend", "funded", "fund", "paid", "pay",
    "raised", "raise", "increased", "decreased",
    "changed", "change", "expanded", "expand", "reduced",
    "connected", "connect", "linked", "disputed", "dispute",
    "supported", "opposed", "oppose",
})

# Minimum cumulative SPLADE activation across the evidence corpus
_SPLADE_ACTIVATION_FLOOR = 0.3

# Maximum number of schema anchors
_MAX_SCHEMA_ANCHORS = 60

# Process-level SPLADE encoder singleton — loaded once per process
_SPLADE: SpladeEncoder | None = None


def _get_splade() -> SpladeEncoder:
    global _SPLADE
    if _SPLADE is None:
        _SPLADE = SpladeEncoder()
    return _SPLADE


def _classify_anchor(clean_token: str) -> str:
    """Classify a cleaned SPLADE token as 'relation' or 'actor'.

    Checks the explicit relation set first. Falls back to verb-suffix
    heuristics: past-tense -ed and -tion/-ise/-ize patterns are strong
    indicators that a BERT token represents an action rather than an entity.
    """
    if clean_token in _RELATION_TOKENS:
        return "relation"
    # Verb-suffix heuristics — past tense and nominalisations of verbs
    if clean_token.endswith("ed") and len(clean_token) > 4:
        return "relation"
    return "actor"


def _build_schema_from_evidence(
    evidence_docs: list[str],
    claim_text: str,
) -> dict:
    """Build a schema by running SPLADE on the evidence corpus + claim.

    Algorithm:
      1. Run SpladeEncoder.encode_sparse() on all evidence texts plus the
         claim (so query-subject tokens are always in scope).
      2. Sum activations across texts to find the most-active vocabulary
         tokens for this specific claim.
      3. Filter noise: WordPiece continuations, single-char pieces,
         common stop-words, special tokens.
      4. Classify surviving tokens as 'actor' or 'relation'.
      5. Return a schema dict compatible with InfonStore.set_schema_from_dict.

    This grounds the schema in what SPLADE actually sees in the evidence —
    no claim-word/evidence-word mismatch, and the schema will naturally
    include the relation (verb) tokens that activate in the corpus.
    """
    splade = _get_splade()
    vocab = splade.tokenizer.get_vocab()
    inv_vocab = {v: k for k, v in vocab.items()}

    all_texts = list(evidence_docs) + [claim_text]

    # Sum SPLADE activations across all texts
    combined = np.zeros(len(vocab), dtype=np.float32)
    for text in all_texts:
        with torch.no_grad():
            inputs = splade.tokenizer(
                [text],
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True,
            )
            outputs = splade.model(**inputs)
            sparse = (
                torch.log1p(F.relu(outputs.logits))
                * inputs["attention_mask"].unsqueeze(-1)
            )
            combined += sparse.max(dim=1).values[0].cpu().numpy()

    # Top tokens sorted by cumulative activation
    top_ids = np.argsort(combined)[::-1]

    schema: dict = {}
    for tid in top_ids.tolist():
        if len(schema) >= _MAX_SCHEMA_ANCHORS:
            break
        val = float(combined[tid])
        if val < _SPLADE_ACTIVATION_FLOOR:
            break

        tok = inv_vocab.get(tid, "")
        if not tok:
            continue
        # Drop WordPiece continuations (##) and special tokens ([CLS] etc.)
        if tok.startswith("##") or tok.startswith("["):
            continue
        # Strip SentencePiece boundary marker
        clean = tok.lstrip("▁")
        if len(clean) < _MIN_ANCHOR_LEN:
            continue
        if clean in _STOP_WORDS:
            continue
        if clean in schema:
            continue

        schema[clean] = {
            "type": _classify_anchor(clean),
            "tokens": [clean],
        }

    return schema


def _extract_claim_subject(claim_text: str, schema: dict) -> str | None:
    """Extract the first content word from the claim that appears in the schema.

    Prefers an exact schema anchor name match so the query subject is
    guaranteed to have a corresponding anchor. Falls back to the first
    claim word that passes basic filters.
    """
    tokens = re.findall(r"[a-z]+", claim_text.lower())
    # First pass: prefer tokens that are schema actors
    for token in tokens:
        if len(token) >= _MIN_ANCHOR_LEN and token not in _STOP_WORDS:
            if token in schema and schema[token]["type"] == "actor":
                return token
    # Second pass: any anchor present in schema
    for token in tokens:
        if len(token) >= _MIN_ANCHOR_LEN and token not in _STOP_WORDS:
            if token in schema:
                return token
    # Fallback: first eligible claim word even if not in schema
    for token in tokens:
        if len(token) >= _MIN_ANCHOR_LEN and token not in _STOP_WORDS:
            return token
    return None


def _verdict_to_mass(verdict) -> MassFunction:
    """Convert a cognition Verdict into the benchmark MassFunction.

    Verdict.mass fields: supports, refutes, uncertain, theta.
    MassFunction fields: m_s, m_r, m_u, m_theta.
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

        Schema is built from the SPLADE activations of the evidence corpus
        so that the schema tokens are guaranteed to activate in the evidence,
        enabling extract_infons() to form valid triples.
        """
        if not claim.evidence_docs:
            return MassFunction(m_s=0.0, m_r=0.0, m_u=0.0, m_theta=1.0)

        schema = _build_schema_from_evidence(claim.evidence_docs, claim.claim_text)
        if not schema:
            return MassFunction(m_s=0.0, m_r=0.0, m_u=0.0, m_theta=1.0)

        # Ensure at least one relation anchor — without one, extract_infons
        # can never form a (subject, predicate, object) triple.
        if not any(v["type"] == "relation" for v in schema.values()):
            return MassFunction(m_s=0.0, m_r=0.0, m_u=0.0, m_theta=1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            store = InfonStore(tmpdir)
            store.set_schema_from_dict(schema)

            docs = [
                {"id": f"doc_{i}", "text": doc, "source": f"doc_{i}"}
                for i, doc in enumerate(claim.evidence_docs)
            ]
            result = store.ingest(docs)

            if result["n_infons"] == 0:
                return MassFunction(m_s=0.0, m_r=0.0, m_u=0.0, m_theta=1.0)

            verdict = self._ask_claim(store, schema, claim)
            return _verdict_to_mass(verdict)

    def _ask_claim(self, store: InfonStore, schema: dict, claim: EvalClaim):
        """Build a Query from the claim text and ask the store.

        Strategy: try all claim content words that appear as actor anchors
        in the schema. Among those that produce a non-vacuous verdict
        (m_theta < 1), return the most informative one (lowest m_theta).
        Fall back to the first schema actor if no claim-matched subject
        produces evidence, then to a per-predicate query.

        An empty query immediately returns NOT_ENOUGH_INFO in reason(),
        so we always provide at least a subject or predicate.
        """
        # Collect candidate actor subjects from the claim text
        claim_tokens = re.findall(r"[a-z]+", claim.claim_text.lower())
        claim_actors = [
            t for t in claim_tokens
            if len(t) >= _MIN_ANCHOR_LEN
            and t not in _STOP_WORDS
            and t in schema
            and schema[t]["type"] == "actor"
        ]

        # Also add all schema actors as fallback candidates
        all_actors = [
            k for k, v in schema.items() if v["type"] == "actor"
        ]

        # Try each candidate; keep the verdict with the most evidence
        best_verdict = None
        best_theta = 1.0

        for subject in (claim_actors or all_actors[:10]):
            v = store.ask(Query().where(subject=subject))
            if v.mass.theta < best_theta:
                best_theta = v.mass.theta
                best_verdict = v
            if best_theta < 0.5:
                break  # good enough evidence found

        if best_verdict is not None and best_theta < 1.0:
            return best_verdict

        # Last resort: try each relation predicate
        relation_anchors = [k for k, v in schema.items() if v["type"] == "relation"]
        for predicate in relation_anchors[:5]:
            v = store.ask(Query().where(predicate=predicate))
            if v.mass.theta < best_theta:
                best_theta = v.mass.theta
                best_verdict = v
            if best_theta < 0.5:
                break

        if best_verdict is not None:
            return best_verdict

        # No evidence found — return the vacuous verdict explicitly
        from cognition.cassette.reason import Verdict as _Verdict
        return _Verdict(label="NOT_ENOUGH_INFO", claim=Query())

"""Surface-level metadata extractors for sentence-grained epistemic cues.

Three extractors:
    extract_evidentiality(sentence)  → {assertion, hedge, attribution}
    extract_modality(sentence)       → {certain, possible, obligation}
    resolve_coreference(doc)         → pronouns → actor names, per document

All rule-based on surface cues. Cheap, deterministic, and abundant on real
corpora. The downstream GNN decides how much to trust each.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


# ── Surface cue lexicons ────────────────────────────────────────────────

# Attribution: "X said", "according to Y", "Y reported that"
_ATTRIBUTION_PATTERNS = [
    r"\b(?:said|stated|reported|announced|claimed|wrote|told|noted|commented|added)\b",
    r"\baccording to\b",
    r"\bas reported by\b",
]
# Hedge: uncertainty markers in the main clause
_HEDGE_CUES = {
    "may", "might", "could", "possibly", "perhaps", "likely",
    "suggests", "seems", "appears", "may have", "reportedly",
    "allegedly", "expected to", "forecast", "projected",
    "is thought", "believed to", "potentially", "probably",
}
# Assertion: strong declarative markers
_ASSERTION_CUES = {
    "confirmed", "announced", "officially", "publicly", "declared",
    "verified", "is", "are", "was", "were", "has", "have", "produces",
    "produced", "invested", "invests", "acquires", "acquired",
}

# Modality: three-way split
_CERTAIN_CUES = {
    "will", "did", "does", "is", "are", "must", "definitely",
    "certainly", "clearly", "demonstrably", "proven",
}
_POSSIBLE_CUES = {
    "may", "might", "could", "possibly", "perhaps", "potentially",
    "conceivably",
}
_OBLIGATION_CUES = {
    "must", "should", "ought", "required", "needs to", "has to",
    "is required to", "mandated", "obligated",
}

# Pronouns to resolve
_COREF_PRONOUNS = {
    "it", "its", "they", "their", "them",
    "the company", "the firm", "the automaker", "the manufacturer",
    "this company", "that company",
}

# A subset of pronouns that strongly denote the subject (so they take
# precedence over any actor mention elsewhere in the same sentence).
_SUBJECT_PRONOUNS = {
    "the company", "the firm", "the automaker", "the manufacturer",
    "this company", "that company",
}


@dataclass
class EvidentialityScore:
    label: str                  # 'assertion' | 'hedge' | 'attribution'
    scores: dict[str, float]    # non-negative, sum to 1


@dataclass
class ModalityScore:
    label: str                  # 'certain' | 'possible' | 'obligation'
    scores: dict[str, float]


def _token_set(sentence: str) -> set[str]:
    """Lowercase token set + simple bigrams for multi-word cue matches."""
    s = sentence.lower()
    unigrams = set(re.findall(r"\b[\w']+\b", s))
    # Multi-word cues
    bigrams = set()
    words = re.findall(r"\b[\w']+\b", s)
    for i in range(len(words) - 1):
        bigrams.add(f"{words[i]} {words[i+1]}")
    trigrams = set()
    for i in range(len(words) - 2):
        trigrams.add(f"{words[i]} {words[i+1]} {words[i+2]}")
    return unigrams | bigrams | trigrams


def extract_evidentiality(sentence: str) -> EvidentialityScore:
    """Classify a sentence's epistemic stance by surface cues.

    Three classes:
        assertion   — declarative, unhedged statement
        hedge       — explicit uncertainty markers in main clause
        attribution — claim is quoted / sourced to another party
    """
    if not sentence or not sentence.strip():
        return EvidentialityScore(
            label="assertion",
            scores={"assertion": 1 / 3, "hedge": 1 / 3, "attribution": 1 / 3},
        )

    tokens = _token_set(sentence)

    # Count cue hits per class
    n_attribution = 0
    for pat in _ATTRIBUTION_PATTERNS:
        if re.search(pat, sentence, re.IGNORECASE):
            n_attribution += 1

    n_hedge = sum(1 for cue in _HEDGE_CUES if cue in tokens)
    n_assertion = sum(1 for cue in _ASSERTION_CUES if cue in tokens)

    # Small prior so no class is zero
    s = {
        "assertion": 1.0 + n_assertion,
        "hedge": 0.5 + 2.0 * n_hedge,
        "attribution": 0.5 + 3.0 * n_attribution,
    }
    total = sum(s.values())
    scores = {k: v / total for k, v in s.items()}
    label = max(scores, key=scores.get)
    return EvidentialityScore(label=label, scores=scores)


def extract_modality(sentence: str) -> ModalityScore:
    """Classify a sentence's modality: certain, possible, or obligation."""
    if not sentence or not sentence.strip():
        return ModalityScore(
            label="certain",
            scores={"certain": 1 / 3, "possible": 1 / 3, "obligation": 1 / 3},
        )

    tokens = _token_set(sentence)
    n_certain = sum(1 for cue in _CERTAIN_CUES if cue in tokens)
    n_possible = sum(1 for cue in _POSSIBLE_CUES if cue in tokens)
    n_obligation = sum(1 for cue in _OBLIGATION_CUES if cue in tokens)

    s = {
        "certain": 0.5 + 1.0 * n_certain,
        "possible": 0.5 + 2.0 * n_possible,
        "obligation": 0.5 + 2.0 * n_obligation,
    }
    total = sum(s.values())
    scores = {k: v / total for k, v in s.items()}
    label = max(scores, key=scores.get)
    return ModalityScore(label=label, scores=scores)


# ── Coreference resolution (lightweight, per-document) ─────────────────

def resolve_coreference(doc_text: str,
                        actor_names: list[str],
                        lookback: int = 3) -> list[str]:
    """Split a document into sentences and rewrite pronouns to actor names.

    Simple algorithm: track the most recently mentioned actor in a sliding
    window of `lookback` sentences. If a later sentence contains a pronoun
    (from _COREF_PRONOUNS), and its own actor-anchor tokens are absent,
    prepend the resolved actor to the sentence.

    Returns a list of (possibly rewritten) sentences in original order.
    """
    if not doc_text:
        return []

    sents = re.split(r"(?<=[.!?])\s+", doc_text.strip())
    sents = [s for s in sents if s]
    if not sents:
        return []

    # Lowercase actor list + surface variants
    actor_tokens = {a.lower() for a in actor_names}

    def sentence_has_actor(sent: str) -> str | None:
        low = sent.lower()
        for a in actor_tokens:
            # Word-boundary match to avoid "catl" matching "catalyst"
            if re.search(rf"\b{re.escape(a)}\b", low):
                return a
        return None

    def sentence_has_pronoun(sent: str) -> bool:
        low = sent.lower()
        for p in _COREF_PRONOUNS:
            if re.search(rf"\b{re.escape(p)}\b", low):
                return True
        return False

    def sentence_has_subject_pronoun(sent: str) -> bool:
        low = sent.lower()
        for p in _SUBJECT_PRONOUNS:
            if re.search(rf"\b{re.escape(p)}\b", low):
                return True
        return False

    # Walk through, tracking the most recent actor
    window: list[tuple[int, str]] = []   # (sentence_index, actor)
    resolved = list(sents)

    for i, sent in enumerate(sents):
        actor_here = sentence_has_actor(sent)
        has_subject_pronoun = sentence_has_subject_pronoun(sent)
        # Subject-pronoun phrases ("the company") take precedence over a
        # co-occurring actor mention — they typically refer back to the
        # previous subject, not to the actor named in the same clause.
        if has_subject_pronoun:
            fresh = [(j, a) for j, a in window if i - j <= lookback]
            if fresh:
                most_recent = fresh[-1][1]
                resolved[i] = f"{most_recent.title()} {sent}"
            # Still update the window if an actor appears
            if actor_here:
                window.append((i, actor_here))
        elif actor_here:
            window.append((i, actor_here))
        elif sentence_has_pronoun(sent):
            fresh = [(j, a) for j, a in window if i - j <= lookback]
            if fresh:
                most_recent = fresh[-1][1]
                resolved[i] = f"{most_recent.title()} {sent}"
        # Always trim so memory stays bounded
        window = [(j, a) for j, a in window if i - j <= lookback]

    return resolved

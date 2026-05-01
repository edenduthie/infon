"""Text extraction pipeline for converting natural language to Infons.

The pipeline processes text in the following stages:
1. Sentence splitting — split document text into sentences
2. SPLADE encoding — encode each sentence to the anchor space
3. Triple formation — form candidate triples from activated anchors
4. Span finding — find character spans in sentence that activated anchors
5. Negation detection — detect lexical negation and set polarity
6. Tense/aspect classification — classify verb tense and aspect
7. Importance scoring — compute multi-dimensional importance score
8. Infon construction — assemble Infon instances with TextGrounding

No mocks — real encoder, real schema, real Infon models.
"""

import re
import uuid
from datetime import UTC, datetime
from itertools import product

from infon.encoder import encode, encode_batch
from infon.grounding import Grounding, TextGrounding
from infon.infon import ImportanceScore, Infon
from infon.schema import AnchorSchema

# Negation words that flip polarity
NEGATION_WORDS = {
    "not",
    "never",
    "no",
    "without",
    "lack",
    "lacks",
    "lacking",
    "n't",  # contractions like "doesn't", "isn't"
    "none",
    "neither",
    "nor",
    "nothing",
}

# Default activation threshold for anchor filtering
DEFAULT_ACTIVATION_THRESHOLD = 0.1

# Default top-K for triple formation
DEFAULT_TOP_K = 5


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using simple rule-based approach.
    
    Args:
        text: Input text to split
        
    Returns:
        List of sentence strings
    """
    if not text or not text.strip():
        return []
    
    # Simple sentence splitting on period, exclamation, question mark
    # followed by whitespace and capital letter
    # This is a minimal implementation - production would use spaCy or nltk
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    
    # Filter out empty sentences
    return [s.strip() for s in sentences if s.strip()]


def _form_triples(
    anchor_activations: dict[str, float],
    schema: AnchorSchema,
    threshold: float = DEFAULT_ACTIVATION_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
) -> list[tuple[str, str, str, float]]:
    """Form candidate triples from activated anchors.
    
    Forms cartesian product of top-K activated actors × relations × features/actors.
    Returns triples where all components are above the activation threshold.
    
    Args:
        anchor_activations: Dict mapping anchor_key -> activation
        schema: AnchorSchema for type filtering
        threshold: Minimum activation threshold (default 0.1)
        top_k: Maximum number of anchors per type to consider (default 5)
        
    Returns:
        List of (subject, predicate, object, min_activation) tuples
    """
    # Filter by threshold
    activated = {k: v for k, v in anchor_activations.items() if v >= threshold}
    
    if not activated:
        return []
    
    # Separate by type
    actors = []
    relations = []
    features = []
    
    for anchor_key, activation in activated.items():
        anchor = schema.anchors.get(anchor_key)
        if not anchor:
            continue
            
        if anchor.type == "actor":
            actors.append((anchor_key, activation))
        elif anchor.type == "relation":
            relations.append((anchor_key, activation))
        elif anchor.type == "feature":
            features.append((anchor_key, activation))
    
    # Sort by activation (descending) and take top-K
    actors = sorted(actors, key=lambda x: x[1], reverse=True)[:top_k]
    relations = sorted(relations, key=lambda x: x[1], reverse=True)[:top_k]
    features = sorted(features, key=lambda x: x[1], reverse=True)[:top_k]
    
    # Objects can be actors or features
    objects = actors + features
    
    # Form triples: subject (actor) × predicate (relation) × object (actor or feature)
    triples = []
    for (subj, subj_act), (pred, pred_act), (obj, obj_act) in product(actors, relations, objects):
        # Skip self-loops (subject == object)
        if subj == obj:
            continue
        
        # Minimum activation across the triple components
        min_activation = min(subj_act, pred_act, obj_act)
        
        triples.append((subj, pred, obj, min_activation))
    
    return triples


def _find_spans(sentence: str, anchor_key: str, schema: AnchorSchema) -> tuple[int, int]:
    """Find character span in sentence that activated the anchor.
    
    Uses exact token match (case-insensitive). Falls back to (0, len(sentence))
    if no exact match is found.
    
    Args:
        sentence: Source sentence text
        anchor_key: Anchor key to find
        schema: AnchorSchema for token lookup
        
    Returns:
        Tuple of (char_start, char_end)
    """
    anchor = schema.anchors.get(anchor_key)
    if not anchor:
        return (0, len(sentence))
    
    sentence_lower = sentence.lower()
    
    # Try to find exact match for any of the anchor's tokens
    for token in anchor.tokens:
        token_lower = token.lower()
        # Use word boundary matching to avoid partial matches
        pattern = r'\b' + re.escape(token_lower) + r'\b'
        match = re.search(pattern, sentence_lower)
        if match:
            return (match.start(), match.end())
    
    # Fallback: entire sentence
    return (0, len(sentence))


def _detect_negation(sentence: str, predicate_span: tuple[int, int]) -> bool:
    """Detect lexical negation in scope of the predicate.
    
    Checks for negation words before or near the predicate span.
    
    Args:
        sentence: Source sentence text
        predicate_span: (char_start, char_end) of predicate
        
    Returns:
        True if negation detected (polarity should be False), False otherwise
    """
    # Get the context around the predicate (20 chars before to predicate end)
    pred_start, pred_end = predicate_span
    context_start = max(0, pred_start - 20)
    context = sentence[context_start:pred_end].lower()
    
    # Tokenize context and check for negation words
    tokens = re.findall(r'\b\w+(?:\'[tn])?\b', context)
    
    for token in tokens:
        if token in NEGATION_WORDS:
            return True
    
    return False


def _classify_tense(sentence: str) -> dict[str, str]:
    """Classify verb tense and aspect from the sentence.
    
    This is a minimal implementation. Production would use spaCy or similar.
    
    Args:
        sentence: Source sentence text
        
    Returns:
        Dict with 'tense' and 'aspect' keys (currently returns placeholder values)
    """
    # Minimal implementation - just return present/simple as default
    # A real implementation would use POS tagging and dependency parsing
    return {
        "tense": "present",  # present | past | future
        "aspect": "simple",  # simple | progressive | perfect
    }


def _score_importance(
    activation: float,
    sentence_position: int,
    total_sentences: int,
) -> ImportanceScore:
    """Compute multi-dimensional importance score.
    
    Args:
        activation: Minimum activation across triple components
        sentence_position: Index of sentence in document (0-indexed)
        total_sentences: Total number of sentences in document
        
    Returns:
        ImportanceScore instance
    """
    # Activation component comes from encoder
    # SPLADE can produce values > 1, so we normalize to [0, 1] using tanh-like scaling
    # tanh(x/2) keeps most values in reasonable range while mapping (0, inf) -> (0, 1)
    import math
    activation_score = min(1.0, max(0.0, math.tanh(activation / 2.0)))
    
    # Coherence: placeholder (would measure semantic coherence with context)
    coherence_score = 0.7
    
    # Specificity: placeholder (would measure how specific vs generic the triple is)
    specificity_score = 0.5
    
    # Novelty: placeholder (would compare against existing constraints in store)
    # For extracted infons, we assume high novelty since we don't have store access here
    novelty_score = 0.8
    
    # Reinforcement: starts at baseline for first extraction
    reinforcement_score = 0.5
    
    return ImportanceScore(
        activation=activation_score,
        coherence=coherence_score,
        specificity=specificity_score,
        novelty=novelty_score,
        reinforcement=reinforcement_score,
    )


def extract_text(
    text: str,
    doc_id: str,
    schema: AnchorSchema,
    *,
    threshold: float = DEFAULT_ACTIVATION_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
) -> list[Infon]:
    """Extract Infons from a single text document.

    For ingest pipelines that extract from many documents at once
    (docstrings, markdown files), prefer :func:`extract_text_batch` —
    batching the SPLADE encoder across sentences from all documents is
    significantly faster on CPU.

    Args:
        text: Source text.
        doc_id: Stable identifier for this document (used in grounding).
        schema: Anchor schema.
        threshold: Minimum SPLADE activation for an anchor to be considered
            (default 0.1, the spec value). Pass a higher value (e.g. 0.3)
            for code-mode ingest where the cartesian-product triple
            formation otherwise produces too much noise.
        top_k: Max anchors per type kept for triple formation (default 5).
            Smaller values cap the explosion of triples per sentence.
    """
    return extract_text_batch(
        [(text, doc_id)], schema, threshold=threshold, top_k=top_k
    )


def extract_text_batch(
    items: list[tuple[str, str]],
    schema: AnchorSchema,
    *,
    threshold: float = DEFAULT_ACTIVATION_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
) -> list[Infon]:
    """Extract Infons from many ``(text, doc_id)`` pairs in one batched pass.

    All sentences across all input documents are collected up-front and
    encoded through SPLADE in batches. This collapses the per-sentence
    Python + tokenizer overhead that dominated the per-document
    ``extract_text`` path and is the single largest CPU win for the deep
    init pipeline (docstring + markdown extraction).

    Args:
        items: List of ``(text, doc_id)`` pairs.
        schema: Anchor schema.
        threshold: Minimum SPLADE activation for an anchor to be retained.
        top_k: Max anchors per type kept for triple formation.

    Returns:
        Flat list of Infon instances, in document-then-sentence order.
    """
    if not items:
        return []

    # 1) Split every document into sentences while remembering which doc
    #    and which sentence-position each came from. We carry the total
    #    sentence count per doc so importance scoring sees the right
    #    "position in document" denominator.
    flat_sentences: list[str] = []
    flat_meta: list[tuple[str, int, int]] = []  # (doc_id, sent_id, total_sentences_in_doc)
    for text, doc_id in items:
        sentences = _split_sentences(text)
        if not sentences:
            continue
        total = len(sentences)
        for sent_id, sentence in enumerate(sentences):
            flat_sentences.append(sentence)
            flat_meta.append((doc_id, sent_id, total))

    if not flat_sentences:
        return []

    # 2) Batched SPLADE → anchor-space activations for every sentence.
    activations_list = encode_batch(flat_sentences, schema)

    # 3) Per-sentence: form triples, build infons.
    import math

    infons: list[Infon] = []
    for sentence, (doc_id, sent_id, total_sents), anchor_activations in zip(
        flat_sentences, flat_meta, activations_list
    ):
        if not anchor_activations:
            continue

        triples = _form_triples(
            anchor_activations, schema, threshold=threshold, top_k=top_k
        )
        if not triples:
            continue

        for subject, predicate, obj, min_activation in triples:
            pred_span = _find_spans(sentence, predicate, schema)
            is_negated = _detect_negation(sentence, pred_span)
            polarity = not is_negated

            importance = _score_importance(
                activation=min_activation,
                sentence_position=sent_id,
                total_sentences=total_sents,
            )

            text_grounding = TextGrounding(
                grounding_type="text",
                doc_id=doc_id,
                sent_id=sent_id,
                char_start=0,
                char_end=len(sentence),
                sentence_text=sentence,
            )
            grounding = Grounding(root=text_grounding)

            confidence = min(1.0, max(0.0, math.tanh(min_activation / 2.0)))

            infons.append(
                Infon(
                    id=str(uuid.uuid4()),
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    polarity=polarity,
                    grounding=grounding,
                    confidence=confidence,
                    timestamp=datetime.now(UTC),
                    importance=importance,
                    kind="extracted",
                    reinforcement_count=1,
                )
            )

    return infons

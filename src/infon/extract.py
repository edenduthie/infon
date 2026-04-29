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
from datetime import datetime, timezone
from itertools import product
from typing import Any

from infon.encoder import encode
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


def extract_text(text: str, doc_id: str, schema: AnchorSchema) -> list[Infon]:
    """Extract Infons from natural language text.
    
    Processes text through the full extraction pipeline:
    1. Split into sentences
    2. Encode each sentence to anchor space
    3. Form triples from activated anchors
    4. Find spans, detect negation, classify tense
    5. Score importance
    6. Construct Infon instances with TextGrounding
    
    Args:
        text: Natural language text to extract from
        doc_id: Document identifier for grounding
        schema: AnchorSchema defining the anchor vocabulary
        
    Returns:
        List of extracted Infon instances (empty if no valid triples found)
    """
    # Stage 1: Sentence splitting
    sentences = _split_sentences(text)
    
    if not sentences:
        return []
    
    infons = []
    
    # Process each sentence
    for sent_id, sentence in enumerate(sentences):
        # Stage 2: SPLADE encoding to anchor space
        anchor_activations = encode(sentence, schema)
        
        if not anchor_activations:
            continue
        
        # Stage 3: Triple formation
        triples = _form_triples(anchor_activations, schema)
        
        if not triples:
            continue
        
        # Stage 4-8: Process each triple
        for subject, predicate, obj, min_activation in triples:
            # Stage 4: Span finding
            subj_span = _find_spans(sentence, subject, schema)
            pred_span = _find_spans(sentence, predicate, schema)
            obj_span = _find_spans(sentence, obj, schema)
            
            # Stage 5: Negation detection
            is_negated = _detect_negation(sentence, pred_span)
            polarity = not is_negated  # True = affirmative, False = negated
            
            # Stage 6: Tense/aspect classification
            tense_info = _classify_tense(sentence)
            
            # Stage 7: Importance scoring
            importance = _score_importance(
                activation=min_activation,
                sentence_position=sent_id,
                total_sentences=len(sentences),
            )
            
            # Stage 8: Infon construction
            text_grounding = TextGrounding(
                grounding_type="text",
                doc_id=doc_id,
                sent_id=sent_id,
                char_start=0,  # Use sentence start for now
                char_end=len(sentence),  # Use sentence end for now
                sentence_text=sentence,
            )
            grounding = Grounding(root=text_grounding)
            
            # Confidence is based on minimum activation, normalized to [0, 1]
            import math
            confidence = min(1.0, max(0.0, math.tanh(min_activation / 2.0)))
            
            infon = Infon(
                id=str(uuid.uuid4()),
                subject=subject,
                predicate=predicate,
                object=obj,
                polarity=polarity,
                grounding=grounding,
                confidence=confidence,
                timestamp=datetime.now(timezone.utc),
                importance=importance,
                kind="extracted",
                reinforcement_count=1,
            )
            
            infons.append(infon)
    
    return infons

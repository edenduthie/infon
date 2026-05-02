"""Extraction pipeline: documents -> sentences -> anchor activations -> infons.

The core of the cognition system. Takes raw text documents and produces
grounded infon triples with spans, support types, hierarchy metadata,
spatial context, and importance scores.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from .encoder import Encoder
from .infon import Edge, Infon
from .schema import AnchorSchema

# English sentence splitter — split after .!? followed by whitespace + capital.
# BlackMagic is English-only; use cognition if you need CJK / Korean handling.
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

# Negation cues
_NEGATION = re.compile(
    r'\b(not|no|never|neither|nor|cannot|can\'t|won\'t|doesn\'t|didn\'t|hasn\'t|haven\'t|isn\'t|aren\'t|wasn\'t|weren\'t)\b',
    re.IGNORECASE,
)

# Temporal expressions
_TEMPORAL_PATTERNS = [
    (re.compile(r'\b(Q[1-4])\s+(\d{4})\b'), "quarter"),
    (re.compile(r'\b(H[12])\s+(\d{4})\b'), "half"),
    (re.compile(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b', re.IGNORECASE), "month"),
    (re.compile(r'\b(\d{4})\b'), "year"),
    (re.compile(r'\b(today|yesterday|last week|this week|next week|last month|this month|next month|last year|this year|next year)\b', re.IGNORECASE), "relative"),
]

# Tense detection (simplified)
_TENSE_PATTERNS = [
    (re.compile(r'\b(will|shall|going to)\b', re.IGNORECASE), "future"),
    (re.compile(r'\b(would|could|might|may)\b', re.IGNORECASE), "conditional"),
    (re.compile(r'\b(is|are|am)\s+\w+ing\b', re.IGNORECASE), "present_continuous"),
    (re.compile(r'\b(was|were|did|had|\w+ed)\b', re.IGNORECASE), "past"),
]


def split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    parts = _SENT_SPLIT.split(text.strip())
    return [s.strip() for s in parts if s.strip()]


def _make_infon_id(doc_id: str, sent_idx: int, subj: str, pred: str, obj: str) -> str:
    """Deterministic infon ID from its components."""
    raw = f"{doc_id}:{sent_idx}:{subj}:{pred}:{obj}"
    return "inf_" + hashlib.sha256(raw.encode()).hexdigest()[:12]


def _detect_polarity(sentence: str) -> int:
    """Detect negation in sentence. Returns 0 if negated, 1 if affirmed."""
    return 0 if _NEGATION.search(sentence) else 1


def _detect_tense(sentence: str) -> str:
    """Simple tense detection from sentence text."""
    for pattern, tense in _TENSE_PATTERNS:
        if pattern.search(sentence):
            return tense
    return "present"


def _extract_temporal_refs(sentence: str) -> list[dict]:
    """Extract temporal references from sentence text."""
    refs = []
    for pattern, precision in _TEMPORAL_PATTERNS:
        for m in pattern.finditer(sentence):
            refs.append({
                "text": m.group(),
                "precision": precision,
                "confidence": 0.8,
            })
    return refs


def _classify_support(anchor_name: str, schema: AnchorSchema,
                      sentence_lower: str) -> str:
    """Classify how an anchor was grounded: direct, semantic, or hierarchical."""
    info = schema.anchors.get(anchor_name, {})
    tokens = info.get("tokens", [])

    # Direct: token appears verbatim in sentence
    for tok in tokens:
        if tok.lower() in sentence_lower:
            return "direct"

    # Hierarchical: check if any child anchor's tokens appear
    for child in schema.get_descendants(anchor_name):
        child_info = schema.anchors.get(child, {})
        for tok in child_info.get("tokens", []):
            if tok.lower() in sentence_lower:
                return "hierarchical"

    # Semantic: model activated but no lexical match
    return "semantic"


def _find_span(anchor_name: str, schema: AnchorSchema,
               sentence: str) -> dict | None:
    """Find the character span of an anchor's tokens in a sentence."""
    info = schema.anchors.get(anchor_name, {})
    tokens = info.get("tokens", [])
    if not tokens:
        return None

    sorted_tokens = sorted(tokens, key=len, reverse=True)
    # English only — use \b word boundaries unconditionally.
    pattern = re.compile(
        r'\b(' + '|'.join(re.escape(t) for t in sorted_tokens) + r')\b',
        re.IGNORECASE,
    )
    m = pattern.search(sentence)
    if m:
        return {"text": m.group(), "start": m.start(), "end": m.end()}
    return None


def _infer_locations(infon: Infon, schema: AnchorSchema) -> list[dict]:
    """Infer spatial locations from anchor metadata."""
    locations = []
    seen = set()

    for role, meta_field in [("subject", "subject_meta"),
                              ("predicate", "predicate_meta"),
                              ("object", "object_meta")]:
        meta = getattr(infon, meta_field)
        cc = meta.get("country_code")
        if cc and cc not in seen:
            seen.add(cc)
            locations.append({
                "name": cc,
                "level": "country",
                "country_code": cc,
                "macro_region": meta.get("macro_region", ""),
                "confidence": 0.5,
                "source": "inferred",
            })

    return locations


def _compute_importance(activation: float, coherence: float,
                        specificity: float, novelty: float,
                        config) -> float:
    """Compute composite importance score."""
    return (config.w_activation * activation
            + config.w_coherence * coherence
            + config.w_specificity * specificity
            + config.w_novelty * novelty)


def extract_infons(
    documents: list[dict],
    encoder: Encoder,
    schema: AnchorSchema,
    config,
) -> tuple[list[Infon], list[Edge]]:
    """Extract infons from documents.

    Each document: {"text": str, "id": str, "timestamp": str (optional)}

    Returns:
        (infons, edges) — infons with grounding, hierarchy, spatial, temporal,
        importance; plus spoke edges linking infons to anchors.
    """
    # 1. Split documents into sentences
    records = []  # (sentence, doc_id, sent_idx, timestamp)
    for doc in documents:
        text = doc["text"]
        doc_id = doc.get("id", doc.get("doc_id", ""))
        timestamp = doc.get("timestamp", doc.get("ts"))
        for i, sent in enumerate(split_sentences(text)):
            records.append((sent, doc_id, i, timestamp))

    if not records:
        return [], []

    # 2. Encode all sentences (SPLADE → project → anchor scores)
    sentences = [r[0] for r in records]
    activations = encoder.encode(sentences, batch_size=config.batch_size)
    # (n_sents, n_anchors) — SPLADE scores in ~[0, 5]

    # Normalize to [0, 1] per-sentence for confidence scoring
    row_maxes = activations.max(axis=1, keepdims=True)
    row_maxes = np.where(row_maxes > 0, row_maxes, 1.0)
    activations_norm = activations / row_maxes

    # 3. Compute corpus-level IDF for specificity
    anchor_doc_freq = np.zeros(len(encoder.anchor_names), dtype=np.float32)
    for row in activations:
        anchor_doc_freq += (row > config.activation_threshold).astype(np.float32)
    n_docs = max(len(documents), 1)
    idf = np.log1p(n_docs / (1 + anchor_doc_freq))
    idf_norm = idf / (idf.max() + 1e-8)

    # Build name→index lookup
    name_to_idx = {name: i for i, name in enumerate(encoder.anchor_names)}

    infons = []
    edges = []

    for sent_idx, (sentence, doc_id, local_sent_idx, timestamp) in enumerate(records):
        raw_scores = activations[sent_idx]     # raw SPLADE-projected scores
        probs = activations_norm[sent_idx]      # normalized to [0,1] for confidence
        sentence_lower = sentence.lower()

        # 4. Partition activations by role
        subjects = []
        predicates = []
        objects = []

        for i, name in enumerate(encoder.anchor_names):
            if raw_scores[i] < config.activation_threshold:
                continue
            atype = schema.types.get(name, "")
            role = schema.role_for_type(atype)
            entry = (name, float(probs[i]))  # normalized score for confidence
            if role == "subject":
                subjects.append(entry)
            elif role == "predicate":
                predicates.append(entry)
            else:
                objects.append(entry)

        # Sort by score, take top-k
        subjects.sort(key=lambda x: -x[1])
        predicates.sort(key=lambda x: -x[1])
        objects.sort(key=lambda x: -x[1])

        subjects = subjects[:config.top_k_per_role]
        predicates = predicates[:config.top_k_per_role]
        objects = objects[:config.top_k_per_role]

        if not subjects or not predicates or not objects:
            continue

        # 5. Detect sentence-level features
        polarity = _detect_polarity(sentence)
        tense = _detect_tense(sentence)
        temporal_refs = _extract_temporal_refs(sentence)
        sent_id = f"{doc_id}_{local_sent_idx:04d}"

        # 6. Form cartesian triples
        for subj_name, subj_prob in subjects:
            for pred_name, pred_prob in predicates:
                for obj_name, obj_prob in objects:
                    # Geometric mean confidence
                    confidence = float((subj_prob * pred_prob * obj_prob) ** (1/3))
                    if confidence < config.min_confidence:
                        continue

                    infon_id = _make_infon_id(doc_id, local_sent_idx,
                                               subj_name, pred_name, obj_name)

                    # Spans
                    spans = {}
                    for role, anchor_name in [("subject", subj_name),
                                               ("predicate", pred_name),
                                               ("object", obj_name)]:
                        span = _find_span(anchor_name, schema, sentence)
                        if span:
                            spans[role] = span

                    # Support types
                    support = {
                        "subject": _classify_support(subj_name, schema, sentence_lower),
                        "predicate": _classify_support(pred_name, schema, sentence_lower),
                        "object": _classify_support(obj_name, schema, sentence_lower),
                    }

                    # Hierarchy metadata
                    subject_meta = schema.get_hierarchy(subj_name)
                    subject_meta["type"] = schema.types.get(subj_name, "")
                    predicate_meta = schema.get_hierarchy(pred_name)
                    predicate_meta["type"] = schema.types.get(pred_name, "")
                    object_meta = schema.get_hierarchy(obj_name)
                    object_meta["type"] = schema.types.get(obj_name, "")

                    # Importance
                    activation = confidence
                    subj_idx = name_to_idx.get(subj_name, 0)
                    pred_idx = name_to_idx.get(pred_name, 0)
                    obj_idx = name_to_idx.get(obj_name, 0)
                    specificity = float(np.mean([
                        idf_norm[subj_idx], idf_norm[pred_idx], idf_norm[obj_idx],
                    ]))

                    infon = Infon(
                        infon_id=infon_id,
                        subject=subj_name,
                        predicate=pred_name,
                        object=obj_name,
                        polarity=polarity,
                        confidence=confidence,
                        sentence=sentence,
                        doc_id=doc_id,
                        sent_id=sent_id,
                        spans=spans,
                        support=support,
                        subject_meta=subject_meta,
                        predicate_meta=predicate_meta,
                        object_meta=object_meta,
                        timestamp=timestamp,
                        precision=temporal_refs[0]["precision"] if temporal_refs else "unknown",
                        temporal_refs=temporal_refs,
                        tense=tense,
                        activation=activation,
                        specificity=specificity,
                        novelty=1.0,
                        importance=_compute_importance(
                            activation, 0.0, specificity, 1.0, config,
                        ),
                    )

                    # Infer locations from metadata
                    infon.locations = _infer_locations(infon, schema)

                    infons.append(infon)

                    # 7. Create spoke edges
                    edges.append(Edge(
                        source=subj_name, target=infon_id,
                        edge_type="INITIATES",
                        weight=subj_prob,
                    ))
                    edges.append(Edge(
                        source=infon_id, target=pred_name,
                        edge_type="ASSERTS",
                        weight=pred_prob,
                    ))
                    edges.append(Edge(
                        source=infon_id, target=obj_name,
                        edge_type="TARGETS",
                        weight=obj_prob,
                    ))

                    # Location edges
                    for loc in infon.locations:
                        edges.append(Edge(
                            source=infon_id, target=loc["name"],
                            edge_type="LOCATED_AT",
                            weight=loc["confidence"],
                            metadata={"level": loc.get("level", "")},
                        ))

    return infons, edges

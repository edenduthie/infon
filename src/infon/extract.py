"""Extraction pipeline: documents -> sentences -> anchor activations -> infons.

The core of the cognition system. Takes raw text documents and produces
grounded infon triples with spans, support types, hierarchy metadata,
spatial context, and importance scores.
"""

from __future__ import annotations

import re
import hashlib
import numpy as np
from collections import defaultdict
from dataclasses import asdict

from .atom import Infon, Span, Edge
from .schema import AnchorSchema
from .encoder import Encoder

# Multilingual sentence splitter:
#   Latin EN:    "foo. Bar"           — split after .!? + whitespace + capital
#   Korean:      "문장이다. 다음은…"  — split after .!? + whitespace + Hangul
#   CJK (JA/ZH): "文。次の…"          — split after full-width 。！？
# Hangul syllable range U+AC00–U+D7A3.
_SENT_SPLIT = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z가-힯])|(?<=[。！？])'
)

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
    # \b word-boundary only for ASCII; CJK has no word boundaries so non-ASCII
    # tokens match as bare substrings. When an English anchor activates
    # semantically on a JA/KO sentence the span lookup will simply return None —
    # the support type already reflects that via 'semantic' classification.
    alts = []
    for t in sorted_tokens:
        esc = re.escape(t)
        alts.append(rf"\b{esc}\b" if t.isascii() else esc)
    pattern = re.compile("|".join(alts), re.IGNORECASE)
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
    # 1. Split documents into sentences. Optionally apply coreference
    # resolution so pronoun-only sentences ("The company announced ...")
    # become extractable against the actor vocabulary.
    from .metadata import (
        extract_evidentiality, extract_modality, resolve_coreference,
    )
    use_coref = getattr(config, "coreference", True)
    actor_names = [n for n in schema.names
                   if schema.types.get(n) == "actor"]

    records = []  # (sentence, doc_id, sent_idx, timestamp, orig_sentence)
    for doc in documents:
        text = doc["text"]
        doc_id = doc.get("id", doc.get("doc_id", ""))
        timestamp = doc.get("timestamp", doc.get("ts"))
        orig_sents = split_sentences(text)
        if use_coref and actor_names:
            resolved_sents = resolve_coreference(
                " ".join(orig_sents), actor_names=actor_names, lookback=3,
            )
            # resolve_coreference may return a different number of
            # sentences if the input had unusual punctuation; fall back
            # to orig ordering if lengths mismatch.
            if len(resolved_sents) != len(orig_sents):
                resolved_sents = orig_sents
        else:
            resolved_sents = orig_sents
        for i, (sent_resolved, sent_orig) in enumerate(
                zip(resolved_sents, orig_sents)):
            records.append(
                (sent_resolved, doc_id, i, timestamp, sent_orig)
            )

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

    for sent_idx, (sentence, doc_id, local_sent_idx, timestamp,
                    orig_sentence) in enumerate(records):
        raw_scores = activations[sent_idx]     # raw SPLADE-projected scores
        probs = activations_norm[sent_idx]      # normalized to [0,1] for confidence
        sentence_lower = sentence.lower()
        # Evidentiality / modality are computed on the ORIGINAL sentence
        # (coreference-expanded text can introduce spurious cues like
        # "Toyota it may..." that throw off the hedge classifier).
        evid = extract_evidentiality(orig_sentence)
        modl = extract_modality(orig_sentence)

        # 4. Partition activations by role
        subjects = []
        predicates = []
        objects = []

        for i, name in enumerate(encoder.anchor_names):
            if raw_scores[i] < config.activation_threshold:
                continue
            atype = schema.types.get(name, "")
            entry = (name, float(probs[i]))  # normalized score for confidence
            # Role eligibility: actors can be BOTH subject and object.
            # The old code partitioned exclusively (actor→subject only,
            # feature→object only), which silently dropped actor-to-actor
            # triples like "Nvidia partnered with TSMC" because TSMC is
            # typed as actor and therefore never eligible for the object
            # slot. Dual-partitioning lets the joint-score filter decide
            # the right role per sentence — features still win the object
            # slot when they activate higher, which is the common case.
            if atype == "relation":
                predicates.append(entry)
                continue
            if atype == "actor":
                subjects.append(entry)
                objects.append(entry)  # ALSO eligible as object
            else:
                # feature / market / location / variant / etc.
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

        # 6. Form triples with joint-score filtering + role-type
        # hard constraints + per-sentence cap.
        #
        # Gather all candidate (subject, predicate, object) triples
        # with their joint score, apply hard role-type constraints,
        # then keep only the top `max_triples_per_sentence` above
        # `quality_threshold`. This replaces the naive Cartesian
        # product + min_confidence threshold that produced too many
        # rubbish triples on ambiguous sentences.
        candidates = []
        quality_floor = getattr(config, "quality_threshold",
                                config.min_confidence)
        # Pre-compute start position per anchor (first mention in sentence).
        # Used to break direction ties between two actors: in English active
        # voice, the subject usually appears before the object. We penalize
        # (not drop) reversed candidates so the correct direction dominates
        # when both are plausible. Sentences without a direct span (semantic
        # match only) get a None and skip the penalty.
        anchor_pos: dict[str, int | None] = {}
        for nm, _ in subjects + objects:
            if nm in anchor_pos:
                continue
            sp = _find_span(nm, schema, sentence)
            anchor_pos[nm] = sp["start"] if sp else None

        for subj_name, subj_prob in subjects:
            if schema.types.get(subj_name) != "actor":
                continue  # hard role-type constraint
            for pred_name, pred_prob in predicates:
                if schema.types.get(pred_name) != "relation":
                    continue  # hard role-type constraint
                for obj_name, obj_prob in objects:
                    obj_type = schema.types.get(obj_name, "")
                    if obj_type in ("", "relation"):
                        continue
                    # Reject trivially-degenerate triples where the
                    # subject and object are the same anchor
                    if subj_name == obj_name:
                        continue
                    # Joint geometric-mean score over all three roles
                    joint = float(
                        (subj_prob * pred_prob * obj_prob) ** (1 / 3)
                    )
                    # Word-order penalty: for actor-to-actor triples where
                    # both endpoints have a direct span, prefer the order
                    # that matches sentence order. A 0.7× multiplier on the
                    # reversed direction is enough to break ties without
                    # killing the candidate entirely (leaves room for
                    # passive-voice recovery on re-read).
                    if schema.types.get(obj_name) == "actor":
                        sp_s = anchor_pos.get(subj_name)
                        sp_o = anchor_pos.get(obj_name)
                        if sp_s is not None and sp_o is not None:
                            if sp_s > sp_o:
                                joint *= 0.7
                    if joint < quality_floor:
                        continue
                    candidates.append(
                        (joint, subj_name, subj_prob, pred_name, pred_prob,
                         obj_name, obj_prob)
                    )

        # Rank and keep only the top-K per sentence
        candidates.sort(key=lambda c: -c[0])
        max_triples = getattr(config, "max_triples_per_sentence", 3)
        candidates = candidates[:max_triples]

        for (joint, subj_name, subj_prob, pred_name, pred_prob,
             obj_name, obj_prob) in candidates:
                    confidence = joint
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
                        evidentiality=evid.label,
                        modality=modl.label,
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

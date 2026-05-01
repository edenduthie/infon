"""Query engine and retrieval pipeline.

The retrieve() function implements a multi-stage retrieval pipeline:
1. Query encoding: Encode query text to anchor space
2. Anchor expansion: Expand activated anchors to descendants
3. Candidate fetch: Retrieve all infons matching candidate anchors (subject,
   object, OR predicate)
4. Keyword fallback: If no SPLADE candidates, match query tokens against
   subject/predicate/object columns directly
5. Valence scoring: Apply persona-specific valence weights
6. Relevance scoring: Compute final score with overlap, confidence, importance, valence
7. NEXT-edge context: Fetch temporal context for top candidates
8. Ranking and deduplication: Sort by score, deduplicate, return top-K

All stages use real dependencies (no mocks): real encoder, real store, real schema.
"""

import re
from dataclasses import dataclass

from infon.encoder import encode
from infon.infon import Infon
from infon.personas import PersonaType, get_valence
from infon.schema import AnchorSchema
from infon.store import InfonStore

# Words removed before keyword matching — they create noise without signal.
_STOP_WORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for",
        "from", "how", "i", "in", "is", "it", "its", "of", "on", "or", "that",
        "the", "this", "to", "was", "were", "what", "when", "where", "which",
        "who", "why", "with", "you", "your",
    }
)


def _query_tokens(query: str) -> list[str]:
    """Extract significant tokens from a query: lower-cased, stop-words removed,
    minimum length 2. Used by the keyword-fallback path."""
    return [
        t
        for t in re.split(r"\W+", query.lower())
        if t and len(t) > 1 and t not in _STOP_WORDS
    ]


@dataclass
class ScoredInfon:
    """An Infon with retrieval score and temporal context.
    
    Attributes:
        infon: The retrieved Infon
        score: Relevance score (higher is better)
        context: List of neighboring infons via NEXT edges (preceding and following)
    """
    infon: Infon
    score: float
    context: list[Infon]


def retrieve(
    query: str,
    store: InfonStore,
    schema: AnchorSchema,
    *,
    limit: int = 10,
    persona: PersonaType | None = None,
) -> list[ScoredInfon]:
    """Retrieve and rank infons for a query.
    
    Pipeline stages:
    1. Query encoding: Encode query to anchor space via encode(query, schema)
    2. Anchor expansion: Expand to descendants via schema.descendants()
    3. Candidate fetch: Retrieve all infons with subject/object in candidate set
    4. Valence scoring: Apply persona valence weights to predicates
    5. Relevance scoring: score = anchor_overlap × confidence × reinforcement × (1 + valence)
    6. NEXT-edge context: Fetch preceding/following infons via NEXT edges
    7. Ranking & dedup: Sort by score desc, deduplicate by triple, return top-K
    
    Args:
        query: Query string
        store: InfonStore to search
        schema: AnchorSchema for encoding and expansion
        limit: Maximum number of results (default 10)
        persona: Optional persona for valence weighting (investor, engineer, executive, regulator, analyst)
        
    Returns:
        List of ScoredInfon, sorted by score descending, up to limit results
    """
    # Stage 1: Query encoding
    # Encode query to anchor space (returns dict[anchor_key -> activation])
    query_anchors = encode(query, schema)

    # Stage 2: Anchor expansion
    # Expand each activated anchor to include all descendants
    candidate_anchors: set[str] = set(query_anchors.keys())
    for anchor_key in list(query_anchors.keys()):
        descendants = schema.descendants(anchor_key)
        candidate_anchors.update(descendants)

    # If SPLADE produced no anchors (typical when the schema is relation-only
    # and the query's content words don't match any relation tokens), skip the
    # candidate fetch and go straight to the keyword fallback. We must NOT
    # early-return here — that would defeat the fallback.
    if not candidate_anchors:
        return _keyword_fallback(query, store, limit=limit, persona=persona)
    
    # Stage 3: Candidate fetch
    # Retrieve all infons where subject, object, OR predicate is in the
    # candidate anchor set. The predicate fan-out matters for AST infons
    # whose predicate is a relation anchor (e.g. ``calls``) — without it,
    # a query like "what calls foo" would never surface AST call sites.
    candidate_infons: list[Infon] = []
    seen_ids = set()

    for anchor_key in candidate_anchors:
        for matches in (
            store.query(subject=anchor_key, limit=1000),
            store.query(object=anchor_key, limit=1000),
            store.query(predicate=anchor_key, limit=1000),
        ):
            for infon in matches:
                if infon.id not in seen_ids:
                    candidate_infons.append(infon)
                    seen_ids.add(infon.id)

    # Stage 4: Keyword fallback
    # If SPLADE + anchor expansion produced no candidates (typical when the
    # schema has no actor anchors covering the user's free-text symbols),
    # fall back to keyword matching against the existing infons. This keeps
    # search useful with the default code schema, which only ships the eight
    # built-in relation anchors.
    if not candidate_infons:
        return _keyword_fallback(query, store, limit=limit, persona=persona)
    
    # Stage 5 & 6: Valence scoring and Relevance scoring
    # Pre-compute keyword tokens once; we use them to bias ranking toward
    # infons whose subject/predicate/object literally mention the user's
    # query terms. Without this, queries like "what calls InfonStore" pull
    # in every "calls" infon at uniform score regardless of whether
    # InfonStore actually appears.
    keyword_tokens = _query_tokens(query)
    scored_infons: list[tuple[Infon, float]] = []

    for infon in candidate_infons:
        # Calculate anchor overlap (how many query anchors match this infon)
        overlap_score = 0.0

        # Count overlaps with expanded candidate set across all three positions.
        for position in (infon.subject, infon.object, infon.predicate):
            if position in candidate_anchors:
                if position in query_anchors:
                    overlap_score += query_anchors[position]
                else:
                    # Descendant of an activated anchor — lower weight.
                    overlap_score += 0.5

        # Normalize overlap (3 positions × max activation 1.0 = 3.0)
        overlap_score = overlap_score / 3.0

        # Keyword bonus: fraction of query tokens that appear in the infon's
        # subject/predicate/object. Multiplies the score by (1 + bonus), so an
        # infon matching every query token can get a 2× boost over one that
        # matches only the predicate.
        keyword_bonus = 0.0
        if keyword_tokens:
            haystack = (
                f"{infon.subject} {infon.predicate} {infon.object}".lower()
            )
            matches = sum(1 for t in keyword_tokens if t in haystack)
            keyword_bonus = matches / len(keyword_tokens)
        
        # Get valence weight for predicate
        valence_weight = get_valence(persona, infon.predicate)
        
        # Compute final relevance score
        # score = overlap × confidence × reinforcement × (1 + valence) × (1 + keyword_bonus)
        score = (
            overlap_score
            * infon.confidence
            * infon.importance.reinforcement
            * (1.0 + valence_weight)
            * (1.0 + keyword_bonus)
        )

        scored_infons.append((infon, score))
    
    # Stage 7a: Sort by score descending
    scored_infons.sort(key=lambda x: x[1], reverse=True)
    
    # Stage 7b: Deduplicate by (subject, predicate, object), keeping highest score
    seen_triples: set[tuple[str, str, str]] = set()
    deduplicated: list[tuple[Infon, float]] = []
    
    for infon, score in scored_infons:
        triple = (infon.subject, infon.predicate, infon.object)
        if triple not in seen_triples:
            seen_triples.add(triple)
            deduplicated.append((infon, score))
    
    # Take top-K
    top_k = deduplicated[:limit]
    
    # Stage 6: NEXT-edge context
    # For each top-K result, fetch NEXT-edge neighbors
    results: list[ScoredInfon] = []
    
    for infon, score in top_k:
        # Fetch NEXT edges (both outgoing and incoming)
        context_infons: list[Infon] = []
        
        # Get outgoing NEXT edges (following infons)
        outgoing_edges = store.get_edges(infon.id, edge_type="NEXT")
        for edge in outgoing_edges:
            # Fetch the target infon
            target_infon = store.get(edge["to_infon_id"])
            if target_infon:
                context_infons.append(target_infon)
        
        # Get incoming NEXT edges (preceding infons)
        # We need to query edges where to_infon_id = infon.id
        # The store doesn't have a direct method for this, so we'll skip incoming for now
        # (this is a simplification - full implementation would add get_incoming_edges)
        
        results.append(
            ScoredInfon(
                infon=infon,
                score=score,
                context=context_infons,
            )
        )

    return results


def _keyword_fallback(
    query: str,
    store: InfonStore,
    *,
    limit: int,
    persona: PersonaType | None,
) -> list[ScoredInfon]:
    """Substring-match query tokens against subject/predicate/object columns.

    Used when SPLADE retrieval finds no candidates — typical when the schema
    has no actor anchors covering the user's free-text symbols (the default
    code-mode schema is relation-only). Returns the top ``limit`` matches
    ranked by token overlap × confidence × reinforcement × persona valence.
    """
    tokens = _query_tokens(query)
    if not tokens:
        return []

    # Pull infons in moderate batches; for typical kbs (<100k infons) one
    # generous read is fine.
    all_infons = store.query(limit=10000)
    if not all_infons:
        return []

    scored: list[tuple[Infon, float]] = []
    for infon in all_infons:
        haystack = f"{infon.subject} {infon.predicate} {infon.object}".lower()
        match_count = sum(1 for token in tokens if token in haystack)
        if match_count == 0:
            continue
        valence_weight = get_valence(persona, infon.predicate)
        score = (
            match_count
            * infon.confidence
            * infon.importance.reinforcement
            * (1.0 + valence_weight)
        )
        scored.append((infon, score))

    if not scored:
        return []

    scored.sort(key=lambda x: x[1], reverse=True)

    seen: set[tuple[str, str, str]] = set()
    deduped: list[ScoredInfon] = []
    for infon, score in scored:
        triple = (infon.subject, infon.predicate, infon.object)
        if triple in seen:
            continue
        seen.add(triple)
        deduped.append(ScoredInfon(infon=infon, score=score, context=[]))
        if len(deduped) >= limit:
            break

    return deduped

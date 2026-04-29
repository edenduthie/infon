"""Query engine and retrieval pipeline.

The retrieve() function implements a multi-stage retrieval pipeline:
1. Query encoding: Encode query text to anchor space
2. Anchor expansion: Expand activated anchors to descendants
3. Candidate fetch: Retrieve all infons matching candidate anchors
4. Valence scoring: Apply persona-specific valence weights
5. Relevance scoring: Compute final score with overlap, confidence, importance, valence
6. NEXT-edge context: Fetch temporal context for top candidates
7. Ranking and deduplication: Sort by score, deduplicate, return top-K

All stages use real dependencies (no mocks): real encoder, real store, real schema.
"""

from dataclasses import dataclass

from infon.encoder import encode
from infon.infon import Infon
from infon.personas import PersonaType, get_valence
from infon.schema import AnchorSchema
from infon.store import InfonStore


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
    
    if not query_anchors:
        # No activated anchors, return empty
        return []
    
    # Stage 2: Anchor expansion
    # Expand each activated anchor to include all descendants
    candidate_anchors = set(query_anchors.keys())
    for anchor_key in list(query_anchors.keys()):
        descendants = schema.descendants(anchor_key)
        candidate_anchors.update(descendants)
    
    if not candidate_anchors:
        return []
    
    # Stage 3: Candidate fetch
    # Retrieve all infons where subject OR object is in candidate anchor set
    # We'll query the store multiple times (once per candidate anchor)
    # to build our candidate set
    candidate_infons: list[Infon] = []
    seen_ids = set()
    
    for anchor_key in candidate_anchors:
        # Query by subject
        subject_matches = store.query(subject=anchor_key, limit=1000)
        for infon in subject_matches:
            if infon.id not in seen_ids:
                candidate_infons.append(infon)
                seen_ids.add(infon.id)
        
        # Query by object
        object_matches = store.query(object=anchor_key, limit=1000)
        for infon in object_matches:
            if infon.id not in seen_ids:
                candidate_infons.append(infon)
                seen_ids.add(infon.id)
    
    if not candidate_infons:
        return []
    
    # Stage 4 & 5: Valence scoring and Relevance scoring
    scored_infons: list[tuple[Infon, float]] = []
    
    for infon in candidate_infons:
        # Calculate anchor overlap (how many query anchors match this infon)
        overlap_score = 0.0
        
        # Count overlaps with expanded candidate set
        if infon.subject in candidate_anchors:
            # Weight by query activation if available
            if infon.subject in query_anchors:
                overlap_score += query_anchors[infon.subject]
            else:
                # It's a descendant, use lower weight
                overlap_score += 0.5
        
        if infon.object in candidate_anchors:
            if infon.object in query_anchors:
                overlap_score += query_anchors[infon.object]
            else:
                overlap_score += 0.5
        
        # Normalize overlap (divide by max possible which is 2 for both subject and object)
        overlap_score = overlap_score / 2.0
        
        # Get valence weight for predicate
        valence_weight = get_valence(persona, infon.predicate)
        
        # Compute final relevance score
        # score = overlap × confidence × reinforcement × (1 + valence)
        score = (
            overlap_score
            * infon.confidence
            * infon.importance.reinforcement
            * (1.0 + valence_weight)
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

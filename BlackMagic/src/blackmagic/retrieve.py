"""Query engine: anchor resolution, subgraph retrieval, persona valence.

Implements the Atomic Architecture's recognition → prediction → evaluation
loop at query time:
1. Encode query → anchor activations (recognition)
2. Retrieve matching infons + walk NEXT edges (prediction/experience)
3. Score with persona-relative valence (evaluation)
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime

from .encoder import Encoder
from .infon import Infon, QueryResult
from .schema import AnchorSchema


def _expand_descendants(anchor_name: str, schema: AnchorSchema) -> list[str]:
    """Expand a parent anchor to itself + all descendants for retrieval.

    Only expands if the anchor has children (is a parent node).
    Leaf anchors return just themselves — no upward expansion.
    """
    names = [anchor_name]
    if schema.get_children(anchor_name):
        names.extend(schema.get_descendants(anchor_name))
    return names


# ── Persona definitions ─────────────────────────────────────────────────

PERSONAS = {
    "investor": {
        "goals": ["maximize returns", "minimize risk", "identify opportunity"],
        "positive_predicates": ["invest", "grow", "launch", "expand", "partner"],
        "negative_predicates": ["decline", "divest", "delay", "cancel", "lose"],
        "focus_types": ["actor", "market", "feature"],
    },
    "engineer": {
        "goals": ["improve technology", "solve problems", "ship products"],
        "positive_predicates": ["launch", "develop", "patent", "innovate", "improve"],
        "negative_predicates": ["delay", "cancel", "fail", "recall"],
        "focus_types": ["feature", "actor"],
    },
    "executive": {
        "goals": ["grow business", "beat competition", "enter markets"],
        "positive_predicates": ["grow", "expand", "launch", "dominate", "partner"],
        "negative_predicates": ["decline", "lose", "exit", "shrink", "cancel"],
        "focus_types": ["actor", "market", "relation"],
    },
    "regulator": {
        "goals": ["ensure compliance", "protect consumers", "maintain stability"],
        "positive_predicates": ["regulate", "comply", "standardize", "certify"],
        "negative_predicates": ["violate", "evade", "monopolize", "lobby"],
        "focus_types": ["actor", "relation"],
    },
    "analyst": {
        "goals": ["understand trends", "identify patterns", "forecast"],
        "positive_predicates": ["grow", "shift", "emerge", "trend"],
        "negative_predicates": ["stagnate", "plateau"],
        "focus_types": ["feature", "market", "relation"],
    },
}


def detect_persona(query: str, anchor_activations: dict[str, float],
                   schema: AnchorSchema) -> str:
    """Detect the most likely persona from query text and activated anchors.

    Uses simple heuristic: which persona's focus types and predicates
    best match the query's activated anchors.
    """
    query_lower = query.lower()

    # Direct keyword detection
    if any(w in query_lower for w in ("invest", "return", "portfolio", "risk")):
        return "investor"
    if any(w in query_lower for w in ("technology", "engineering", "develop", "patent")):
        return "engineer"
    if any(w in query_lower for w in ("market share", "strategy", "compete", "revenue")):
        return "executive"
    if any(w in query_lower for w in ("regulat", "compliance", "policy", "standard")):
        return "regulator"

    # Score by activated anchor types
    scores = defaultdict(float)
    for anchor, prob in anchor_activations.items():
        atype = schema.types.get(anchor, "")
        for persona_name, persona in PERSONAS.items():
            if atype in persona["focus_types"]:
                scores[persona_name] += prob

    if scores:
        return max(scores, key=scores.get)
    return "analyst"  # default


def compute_valence(infon: Infon, persona: str, goal: str = "") -> dict:
    """Compute query-time valence for an infon relative to a persona.

    Returns {valence, salience, urgency, temporal_relevance} — never stored.
    """
    persona_def = PERSONAS.get(persona, PERSONAS["analyst"])

    # Valence: positive/negative relative to persona's preferences
    valence = 0.0
    pred = infon.predicate
    if pred in persona_def["positive_predicates"]:
        valence = 0.5 + 0.5 * infon.confidence
    elif pred in persona_def["negative_predicates"]:
        valence = -(0.5 + 0.5 * infon.confidence)
    else:
        valence = 0.1 * infon.confidence  # neutral slight positive

    # Flip for negation
    if infon.polarity == 0:
        valence = -valence

    # Salience: how relevant to the persona's focus types
    salience = 0.0
    for role_meta in [infon.subject_meta, infon.predicate_meta, infon.object_meta]:
        if role_meta.get("type") in persona_def["focus_types"]:
            salience += 0.33

    # Urgency: future-tense + high confidence = urgent
    urgency = 0.0
    if infon.tense in ("future", "conditional"):
        urgency = 0.5 + 0.5 * infon.confidence
    elif infon.tense == "present_continuous":
        urgency = 0.3 + 0.3 * infon.confidence

    # Temporal relevance: recency
    temporal_relevance = _recency_score(infon.timestamp)

    return {
        "valence": round(max(-1.0, min(1.0, valence)), 3),
        "salience": round(min(1.0, salience), 3),
        "urgency": round(min(1.0, urgency), 3),
        "temporal_relevance": round(temporal_relevance, 3),
    }


def _recency_score(timestamp: str | None) -> float:
    """Score recency: 1.0 for today, decaying to 0.1 over ~2 years."""
    if not timestamp:
        return 0.5  # unknown = middle
    try:
        ts = timestamp[:10]
        dt = datetime.strptime(ts, "%Y-%m-%d")
        days = max(0, (datetime.now(tz=None) - dt).days)
        return max(0.1, math.exp(-days / 365))
    except (ValueError, TypeError):
        return 0.5


def query(
    text: str,
    encoder: Encoder,
    schema: AnchorSchema,
    store,
    config,
    persona: str | None = None,
    goal: str = "",
    top_k: int | None = None,
    min_importance: float = 0.0,
    include_chains: bool = True,
    chain_depth: int = 10,
    contrary: bool = False,
) -> QueryResult:
    """Execute a cognition query.

    1. Encode query → anchor activations
    2. Detect persona (or use provided)
    3. Retrieve infons for top activated anchors
    4. Optionally walk NEXT chains for prediction
    5. Score with persona-relative valence
    6. Return ranked QueryResult

    When ``contrary=True`` the ranking lens is inverted: negated infons
    and refuting evidence float to the top, and valence is flipped.  This
    lets the caller ask "show me evidence *against* the query thesis"
    without changing what's stored.

    Args:
        text: natural language query
        encoder: trained Encoder instance
        schema: AnchorSchema instance
        store: StoreBackend instance
        config: CognitionConfig
        persona: override persona detection
        goal: optional goal string for valence tuning
        top_k: max results (default: config.default_top_k)
        min_importance: filter low-importance infons
        include_chains: whether to walk NEXT edges
        chain_depth: how far to walk NEXT chains
        contrary: invert ranking — prefer refuting / negated evidence
    """
    top_k = top_k or config.default_top_k

    # 1. Encode query
    activations = encoder.encode_single(text)

    # 2. Detect persona
    if not persona:
        persona = detect_persona(text, activations, schema)

    # 3. Retrieve infons for activated anchors (with descendant expansion)
    seen_ids = set()
    infons = []

    # Sort anchors by activation strength
    sorted_anchors = sorted(activations.items(), key=lambda x: -x[1])

    for anchor_name, prob in sorted_anchors[:15]:  # top 15 anchors
        role = schema.role_for_type(schema.types.get(anchor_name, ""))
        query_names = _expand_descendants(anchor_name, schema)
        for name in query_names:
            anchor_infons = store.get_infons_for_anchor(
                name, role=role, limit=top_k,
            )
            for inf in anchor_infons:
                if inf.infon_id not in seen_ids and inf.importance >= min_importance:
                    seen_ids.add(inf.infon_id)
                    infons.append(inf)

    # 4. Walk NEXT chains for temporal prediction
    edges = []
    if include_chains:
        chain_anchors = [a for a, _ in sorted_anchors[:5]]
        for anchor in chain_anchors:
            for inf in infons[:10]:
                chain = store.get_next_chain(
                    inf.infon_id, anchor, limit=chain_depth,
                )
                for edge in chain:
                    edges.append(edge)
                    # Fetch chain infons too
                    if edge.target not in seen_ids:
                        chain_infon = store.get_infon(edge.target)
                        if chain_infon:
                            seen_ids.add(edge.target)
                            infons.append(chain_infon)

    # 5. Score with valence
    valence_map = {}
    for inf in infons:
        v = compute_valence(inf, persona, goal)
        if contrary:
            v["valence"] = -v["valence"]
        valence_map[inf.infon_id] = v

    # 6. Rank: composite of importance + valence salience + temporal relevance
    #    In contrary mode, prefer negated-polarity infons (the counter-evidence).
    def rank_score(inf: Infon) -> float:
        v = valence_map.get(inf.infon_id, {})
        base = (
            0.4 * inf.importance
            + 0.3 * v.get("salience", 0)
            + 0.2 * v.get("temporal_relevance", 0.5)
            + 0.1 * abs(v.get("valence", 0))
        )
        if contrary and inf.polarity == 0:
            base += 0.25
        return base

    infons.sort(key=rank_score, reverse=True)
    infons = infons[:top_k]

    # Build timeline (chronologically ordered)
    timeline = sorted(
        [inf for inf in infons if inf.timestamp],
        key=lambda inf: inf.timestamp,
    )

    # Get constraints for the query anchors (with descendant expansion)
    constraints = []
    for anchor_name, _ in sorted_anchors[:5]:
        role = schema.role_for_type(schema.types.get(anchor_name, ""))
        query_names = _expand_descendants(anchor_name, schema)
        for name in query_names:
            if role == "subject":
                constraints.extend(store.get_constraints(subject=name, limit=10))
            elif role == "predicate":
                constraints.extend(store.get_constraints(predicate=name, limit=10))
            else:
                constraints.extend(store.get_constraints(object=name, limit=10))

    # Deduplicate constraints
    seen_triples = set()
    unique_constraints = []
    for c in constraints:
        key = c.triple_key()
        if key not in seen_triples:
            seen_triples.add(key)
            unique_constraints.append(c)
    unique_constraints.sort(key=lambda c: -c.score)

    return QueryResult(
        query=text,
        persona=persona,
        infons=infons,
        constraints=unique_constraints,
        edges=edges,
        valence={iid: v.get("valence", 0) for iid, v in valence_map.items()},
        timeline=timeline,
        anchors_activated=activations,
    )
